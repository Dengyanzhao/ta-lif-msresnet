"""Single-run trainer for the factorial TA-LIF x MS-ResNet experiment."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    RunConfig,
    artifact_paths_for_protocol,
    load_protocol,
    load_run_config,
)
from .data import build_loaders
from .freeze import FreezeGateError, verify_formal_freeze
from .models import build_model
from .preflight import check_protocol
from .pathing import artifact_path_reference
from .pilot_v3 import (
    PilotV3Error,
    expected_pilot_plan_payload,
    require_v3_author_freeze,
    resolve_pilot_block,
    validate_health_report as validate_v3_health_report,
)
from .pilot_v4 import (
    PilotV4Error,
    expected_pilot_plan_payload as expected_v4_pilot_plan_payload,
    require_v4_author_freeze,
    resolve_pilot_block as resolve_v4_pilot_block,
    validate_health_report as validate_v4_health_report,
)
from .pilot_v5 import (
    PilotV5Error,
    expected_pilot_plan_payload as expected_v5_pilot_plan_payload,
    require_v5_author_freeze,
    resolve_pilot_block as resolve_v5_pilot_block,
    validate_health_report as validate_v5_health_report,
)
from .utils import (
    AverageMeter,
    JSONLLogger,
    append_csv_row,
    atomic_write_json,
    capture_rng_state,
    environment_manifest,
    json_safe,
    load_checkpoint,
    resolve_device,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
    sha256_file,
    stable_hash,
    upsert_csv_row,
    utc_now,
)


SEED_METRIC_FIELDS: Tuple[str, ...] = (
    "run_id", "experiment", "dataset", "depth", "time_steps", "condition", "topology", "neuron",
    "seed", "config_hash", "protocol_hash", "split_manifest_sha256", "shared_weight_sha256",
    "training_environment_identity", "training_environment_sha256",
    "status", "failed", "best_epoch",
    "best_val_loss", "best_val_accuracy", "test_loss", "test_accuracy", "test_samples",
    "train_loss_auc", "converged", "convergence_epoch",
    "training_gradient_cv", "training_gradient_cv_method", "gradient_cv", "gradient_cv_method",
    "jacobian_phi", "jacobian_varphi",
    "spike_rate", "peak_training_memory_bytes", "train_epoch_seconds", "started_at", "finished_at", "duration_s",
    "test_checkpoint_sha256", "test_evaluated_at",
    "diagnostic_id", "diagnostic_batch_sha256",
    "diagnostic_protocol_hash", "jacobian_method", "jacobian_probes", "jacobian_probe_seed",
    "jacobian_time_index",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_PLAN_PATH = PROJECT_ROOT / "environment" / "unfrozen_pilot_plan.json"
OPTIMIZER_GROUP_MANIFEST_SCHEMA_VERSION = 1
_OPTIMIZER_GROUP_IDENTITY_KEYS = (
    "index",
    "role",
    "parameter_names",
    "parameter_count",
    "parameter_numel",
    "initial_lr",
    "weight_decay",
)


def _is_ta_parameter(name: str) -> bool:
    lowered = name.lower().replace("-", "_")
    tokens = (
        "ta_lif", "talif", "threshold_window", "adaptive_threshold", "theta_low", "theta_high",
        "theta_l", "theta_h", "delta_min", "window_low", "window_high", "activity_count",
        ".center", ".raw_width",
    )
    return any(token in lowered for token in tokens)


def split_parameter_groups(model: torch.nn.Module, is_ta_model: bool) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter], List[str]]:
    """Partition trainable parameters and return TA names for audit logs."""

    ta_ids: set[int] = set()
    if is_ta_model and hasattr(model, "ta_parameters"):
        try:
            ta_ids = {id(parameter) for parameter in model.ta_parameters()}  # type: ignore[attr-defined]
        except TypeError:
            ta_ids = set()
    base: List[torch.nn.Parameter] = []
    ta: List[torch.nn.Parameter] = []
    ta_names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_ta = is_ta_model and (id(parameter) in ta_ids or _is_ta_parameter(name))
        if is_ta:
            ta.append(parameter)
            ta_names.append(name)
        else:
            base.append(parameter)
    if is_ta_model and not ta:
        raise RuntimeError(
            "TA-LIF condition exposes no identifiable TA parameters. Name them with a TA/threshold-window "
            "prefix or implement model.ta_parameters()."
        )
    return base, ta, ta_names


_NORMALIZATION_MODULES = (
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
    torch.nn.GroupNorm,
    torch.nn.LayerNorm,
)


def _split_base_weight_decay_groups(
    model: torch.nn.Module,
    ta_parameters: Sequence[torch.nn.Parameter],
) -> Tuple[
    List[torch.nn.Parameter],
    List[str],
    List[torch.nn.Parameter],
    List[str],
]:
    """Separate base weights from normalization parameters and biases."""

    ta_ids = {id(parameter) for parameter in ta_parameters}
    normalization_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, _NORMALIZATION_MODULES)
        for parameter in module.parameters(recurse=False)
    }
    decay: List[torch.nn.Parameter] = []
    decay_names: List[str] = []
    no_decay: List[torch.nn.Parameter] = []
    no_decay_names: List[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in ta_ids:
            continue
        if (
            id(parameter) in normalization_ids
            or name == "bias"
            or name.endswith(".bias")
        ):
            no_decay.append(parameter)
            no_decay_names.append(name)
        else:
            decay.append(parameter)
            decay_names.append(name)
    return decay, decay_names, no_decay, no_decay_names


def build_optimizer_and_scheduler(
    model: torch.nn.Module, config: RunConfig,
) -> Tuple[
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    List[torch.nn.Parameter],
    List[str],
    int,
]:
    opt = config.optimizer
    if config.protocol_version < 4 and (
        opt.warmup_epochs != 0
        or opt.exclude_norm_and_bias_from_weight_decay
    ):
        raise RuntimeError("v4 optimizer repairs require protocol_version >= 4")
    is_ta_model = config.model.neuron == "ta_lif"
    base, ta, ta_names = split_parameter_groups(model, is_ta_model)
    groups: List[Dict[str, Any]] = []
    group_specs: List[Dict[str, Any]] = []

    def add_group(
        parameters: Sequence[torch.nn.Parameter],
        names: Sequence[str],
        *,
        role: str,
        lr: float,
        weight_decay: float,
    ) -> None:
        groups.append(
            {
                "params": list(parameters),
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
        group_specs.append(
            {
                "role": role,
                "parameter_names": list(names),
                "initial_lr": float(lr),
                "weight_decay": float(weight_decay),
            }
        )

    parameter_name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    if opt.exclude_norm_and_bias_from_weight_decay:
        decay, decay_names, no_decay, no_decay_names = _split_base_weight_decay_groups(
            model, ta
        )
        if len(decay) + len(no_decay) != len(base):
            raise RuntimeError("Base optimizer parameter partition is incomplete")
        if decay:
            add_group(
                decay, decay_names, role="base_decay", lr=opt.lr,
                weight_decay=opt.weight_decay,
            )
        if no_decay:
            add_group(
                no_decay, no_decay_names, role="base_no_decay",
                lr=opt.lr, weight_decay=0.0,
            )
    else:
        add_group(
            base, [parameter_name_by_id[id(parameter)] for parameter in base],
            role="base", lr=opt.lr, weight_decay=opt.weight_decay,
        )
    if ta:
        # The scheduler sees the intended TA learning rate from the beginning;
        # gradients are disabled until the prespecified 5% warm-up ends.
        for parameter in ta:
            parameter.requires_grad_(False)
        add_group(
            ta, ta_names, role="ta", lr=opt.lr * opt.ta_lr_scale,
            weight_decay=opt.ta_weight_decay,
        )
    optimizer = torch.optim.SGD(groups, lr=opt.lr, momentum=opt.momentum)
    setattr(optimizer, "_talif_group_specs", tuple(group_specs))
    setattr(
        optimizer,
        "_talif_parameter_ids",
        frozenset(id(parameter) for group in groups for parameter in group["params"]),
    )
    if opt.warmup_epochs:
        def schedule_factor(epoch: int) -> float:
            warmup = min(1.0, float(epoch + 1) / float(opt.warmup_epochs))
            decays = sum(epoch >= milestone for milestone in opt.milestones)
            return warmup * (opt.gamma ** decays)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=schedule_factor
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(opt.milestones), gamma=opt.gamma,
        )
    activation_epoch = int(math.ceil(opt.epochs * opt.ta_start_fraction))
    return optimizer, scheduler, ta, ta_names, activation_epoch


def optimizer_group_manifest(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Return auditable group identities without polluting optimizer state."""

    specs = getattr(optimizer, "_talif_group_specs", None)
    if not isinstance(specs, Sequence) or len(specs) != len(optimizer.param_groups):
        raise RuntimeError("Optimizer group specifications are missing or malformed")
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    seen: set[int] = set()
    groups: List[Dict[str, Any]] = []
    for index, (group, raw_spec) in enumerate(zip(optimizer.param_groups, specs)):
        if not isinstance(raw_spec, Mapping):
            raise RuntimeError("Optimizer group specification must be a mapping")
        parameters = list(group["params"])
        identifiers = [id(parameter) for parameter in parameters]
        if len(set(identifiers)) != len(identifiers) or seen.intersection(identifiers):
            raise RuntimeError("Optimizer parameter groups contain duplicate parameters")
        seen.update(identifiers)
        try:
            names = [name_by_id[identifier] for identifier in identifiers]
        except KeyError as exc:
            raise RuntimeError("Optimizer contains a parameter not owned by the model") from exc
        expected_names = list(raw_spec.get("parameter_names", ()))
        if names != expected_names:
            raise RuntimeError(
                f"Optimizer group {index} parameter order differs from its audit specification"
            )
        initial_lr = float(raw_spec["initial_lr"])
        if "initial_lr" in group and float(group["initial_lr"]) != initial_lr:
            raise RuntimeError(f"Optimizer group {index} initial_lr drifted")
        weight_decay = float(group.get("weight_decay", 0.0))
        if weight_decay != float(raw_spec["weight_decay"]):
            raise RuntimeError(f"Optimizer group {index} weight_decay drifted")
        groups.append(
            {
                "index": index,
                "role": str(raw_spec["role"]),
                "parameter_names": names,
                "parameter_count": len(parameters),
                "parameter_numel": sum(parameter.numel() for parameter in parameters),
                "initial_lr": initial_lr,
                "current_lr": float(group["lr"]),
                "weight_decay": weight_decay,
            }
        )
    expected_ids = getattr(optimizer, "_talif_parameter_ids", None)
    model_ids = {id(parameter) for parameter in model.parameters()}
    if not isinstance(expected_ids, frozenset) or seen != expected_ids:
        raise RuntimeError(
            "Optimizer parameter groups do not cover every model parameter exactly once"
        )
    if not seen.issubset(model_ids):
        raise RuntimeError("Optimizer parameter ownership changed after construction")
    return {
        "schema_version": OPTIMIZER_GROUP_MANIFEST_SCHEMA_VERSION,
        "groups": groups,
    }


def _optimizer_learning_rates(optimizer: torch.optim.Optimizer) -> Dict[str, float]:
    specs = getattr(optimizer, "_talif_group_specs", ())
    return {
        str(spec.get("role", index)): float(group["lr"])
        for index, (group, spec) in enumerate(zip(optimizer.param_groups, specs))
    }


def _set_ta_enabled(parameters: Iterable[torch.nn.Parameter], enabled: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad_(enabled)


def _ta_enabled_for_epoch(
    parameters: Sequence[torch.nn.Parameter],
    epoch: int,
    activation_epoch: int,
) -> bool:
    """Report TA as enabled only when the model actually owns TA parameters."""

    return bool(parameters) and epoch >= activation_epoch


def _reset_state(model: torch.nn.Module) -> None:
    if hasattr(model, "reset_state"):
        model.reset_state()  # type: ignore[attr-defined]


def _reduce_logits(logits: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Reduce temporal logits to [B, C] while preserving ordinary logits."""

    if logits.ndim == 2:
        return logits
    if logits.ndim != 3:
        raise RuntimeError(f"Model logits must be [B,C], [T,B,C], or [B,T,C], got {tuple(logits.shape)}")
    if logits.shape[0] == batch_size:
        return logits.mean(dim=1)
    if logits.shape[1] == batch_size:
        return logits.mean(dim=0)
    raise RuntimeError(f"Cannot identify batch dimension in logits {tuple(logits.shape)} for batch {batch_size}")


def _forward(model: torch.nn.Module, inputs: torch.Tensor, collect_activity: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
    _reset_state(model)
    result = model(inputs, collect_activity=True) if collect_activity else model(inputs)
    diagnostics: Dict[str, Any] = {}
    if isinstance(result, tuple):
        if len(result) != 2:
            raise RuntimeError("Model tuple output must be (logits, diagnostics)")
        logits, raw_diagnostics = result
        if isinstance(raw_diagnostics, Mapping):
            diagnostics = dict(raw_diagnostics)
        else:
            diagnostics = {"value": raw_diagnostics}
    else:
        logits = result
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("Model output logits must be a torch.Tensor")
    return _reduce_logits(logits, inputs.shape[0]), diagnostics


def _accumulate_surrogate_layer_counts(
    accumulator: Dict[str, Dict[str, float]],
    diagnostics: Mapping[str, Any],
) -> None:
    """Accumulate exact surrogate support counts for each neuron layer."""

    layers = diagnostics.get("layers")
    if not isinstance(layers, Mapping):
        return
    for raw_name, raw_values in layers.items():
        if not isinstance(raw_values, Mapping):
            continue
        active = raw_values.get("surrogate_active")
        elements = raw_values.get("surrogate_elements")
        if (
            not isinstance(active, (int, float))
            or isinstance(active, bool)
            or not isinstance(elements, (int, float))
            or isinstance(elements, bool)
        ):
            continue
        active_value = float(active)
        element_value = float(elements)
        if (
            not math.isfinite(active_value)
            or not math.isfinite(element_value)
            or active_value < 0.0
            or element_value < 0.0
            or active_value > element_value
        ):
            raise RuntimeError(f"Invalid surrogate support counts for layer {raw_name!r}")
        if element_value == 0.0:
            continue
        row = accumulator.setdefault(
            str(raw_name), {"surrogate_active": 0.0, "surrogate_elements": 0.0}
        )
        row["surrogate_active"] += active_value
        row["surrogate_elements"] += element_value


def _surrogate_layer_summary(
    accumulator: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Return per-layer coverage after aggregating counts across batches."""

    result: Dict[str, Dict[str, float]] = {}
    for name in sorted(accumulator):
        active = float(accumulator[name]["surrogate_active"])
        elements = float(accumulator[name]["surrogate_elements"])
        if elements <= 0.0:
            continue
        result[name] = {
            "surrogate_active": active,
            "surrogate_elements": elements,
            "surrogate_coverage": active / elements,
        }
    return result


def _gradient_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2).item()
            squared += value * value
    return math.sqrt(squared)


def _block_gradient_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Return one parameter-gradient L2 norm per residual block."""

    values: Dict[str, float] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ not in {"SpikingBasicBlock", "MSBasicBlock"}:
            continue
        squared = 0.0
        for parameter in module.parameters(recurse=True):
            if parameter.grad is not None:
                norm = parameter.grad.detach().float().norm(2).item()
                squared += norm * norm
        values[name] = math.sqrt(squared)
    return values


def _parameter_role_norms(model: torch.nn.Module) -> Dict[str, float]:
    """Summarize parameter norms by owner for collapse diagnostics."""

    modules = dict(model.named_modules())
    squared = {
        "convolution": 0.0,
        "normalization": 0.0,
        "classifier": 0.0,
        "ta": 0.0,
        "other": 0.0,
    }
    for name, parameter in model.named_parameters():
        owner_name = name.rsplit(".", 1)[0] if "." in name else ""
        owner = modules.get(owner_name)
        if _is_ta_parameter(name):
            role = "ta"
        elif isinstance(owner, torch.nn.Conv2d):
            role = "convolution"
        elif isinstance(owner, _NORMALIZATION_MODULES):
            role = "normalization"
        elif isinstance(owner, torch.nn.Linear):
            role = "classifier"
        else:
            role = "other"
        norm = parameter.detach().float().norm(2).item()
        squared[role] += norm * norm
    return {role: math.sqrt(value) for role, value in squared.items()}


def _coefficient_of_variation(values: Sequence[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size < 2 or float(finite.mean()) == 0.0:
        return float("nan")
    return float(finite.std(ddof=1) / finite.mean())


def _apply_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    probability: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply one deterministic-under-seed spatial CutMix draw."""

    if alpha <= 0.0 or probability <= 0.0 or inputs.shape[0] < 2:
        return inputs, targets, targets, 1.0
    if float(torch.rand((), device=inputs.device).item()) >= probability:
        return inputs, targets, targets, 1.0
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    height, width = int(inputs.shape[-2]), int(inputs.shape[-1])
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_h, cut_w = int(height * cut_ratio), int(width * cut_ratio)
    center_y = int(torch.randint(height, (), device=inputs.device).item())
    center_x = int(torch.randint(width, (), device=inputs.device).item())
    y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
    x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
    permutation = torch.randperm(inputs.shape[0], device=inputs.device)
    mixed = inputs.clone()
    mixed[..., y1:y2, x1:x2] = inputs[permutation, ..., y1:y2, x1:x2]
    adjusted = 1.0 - ((y2 - y1) * (x2 - x1) / float(height * width))
    return mixed, targets, targets[permutation], adjusted


def _batch_limit_reached(batch_index: int, limit_batches: int | None) -> bool:
    return limit_batches is not None and batch_index >= limit_batches


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: RunConfig,
    logger: JSONLLogger,
    scaler: Any,
) -> Dict[str, Any]:
    model.train()
    losses, accuracies = AverageMeter(), AverageMeter()
    global_gradient_norms: List[float] = []
    block_gradient_cvs: List[float] = []
    block_nonzero_fractions: List[float] = []
    all_zero_block_gradient_batches = 0
    collect_activity = bool(config.analysis.get("collect_activity", False))
    diagnostics_accumulator: Dict[str, List[float]] = {}
    surrogate_layer_counts: Dict[str, Dict[str, float]] = {}
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader):
        if _batch_limit_reached(batch_index, config.runtime.limit_batches):
            break
        inputs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        inputs, targets_a, targets_b, cutmix_weight = _apply_cutmix(
            inputs,
            targets,
            alpha=config.optimizer.cutmix_alpha,
            probability=config.optimizer.cutmix_probability,
        )
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(config.runtime.amp and device.type == "cuda")
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits, diagnostics = _forward(model, inputs, collect_activity)
            loss_a = F.cross_entropy(
                logits, targets_a, label_smoothing=config.optimizer.label_smoothing
            )
            loss_b = F.cross_entropy(
                logits, targets_b, label_smoothing=config.optimizer.label_smoothing
            )
            loss = cutmix_weight * loss_a + (1.0 - cutmix_weight) * loss_b
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite training loss at epoch={epoch}, batch={batch_index}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = _gradient_norm(model)
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient norm at epoch={epoch}, batch={batch_index}")
        global_gradient_norms.append(grad_norm)
        block_norms = _block_gradient_norms(model)
        nonzero_block_fraction = (
            sum(value > 0.0 for value in block_norms.values()) / len(block_norms)
            if block_norms
            else float("nan")
        )
        if math.isfinite(nonzero_block_fraction):
            block_nonzero_fractions.append(nonzero_block_fraction)
            if nonzero_block_fraction == 0.0:
                all_zero_block_gradient_batches += 1
        block_cv = _coefficient_of_variation(list(block_norms.values()))
        if math.isfinite(block_cv):
            block_gradient_cvs.append(block_cv)
        if config.optimizer.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(targets.shape[0])
        predictions = logits.argmax(dim=1)
        correct = (
            cutmix_weight * float((predictions == targets_a).sum().item())
            + (1.0 - cutmix_weight) * float((predictions == targets_b).sum().item())
        )
        losses.update(float(loss.detach().item()), batch_size)
        accuracies.update(correct / batch_size, batch_size)
        for key, value in diagnostics.items():
            safe = json_safe(value)
            if isinstance(safe, (int, float)) and math.isfinite(float(safe)):
                diagnostics_accumulator.setdefault(str(key), []).append(float(safe))
        if collect_activity:
            _accumulate_surrogate_layer_counts(surrogate_layer_counts, diagnostics)
        if (batch_index + 1) % config.runtime.log_every == 0:
            logger.log(
                "train_batch", epoch=epoch, batch=batch_index + 1,
                loss=losses.average, accuracy=accuracies.average,
                global_gradient_norm=grad_norm, block_gradient_cv=block_cv,
                nonzero_block_gradient_fraction=nonzero_block_fraction,
                block_gradient_norms=block_norms, cutmix_weight=cutmix_weight,
            )
    if losses.count == 0:
        raise RuntimeError("Training loader produced no batches")
    elapsed = time.perf_counter() - started
    grad_mean = float(np.mean(global_gradient_norms)) if global_gradient_norms else float("nan")
    grad_cv = float(np.mean(block_gradient_cvs)) if block_gradient_cvs else float("nan")
    result: Dict[str, Any] = {
        "loss": losses.average,
        "accuracy": accuracies.average,
        "samples": losses.count,
        "gradient_mean": grad_mean,
        "gradient_cv": grad_cv,
        "gradient_cv_method": "mean_batch_cv_of_residual_block_parameter_gradient_l2",
        "nonzero_block_gradient_fraction_mean": (
            float(np.mean(block_nonzero_fractions))
            if block_nonzero_fractions
            else float("nan")
        ),
        "nonzero_block_gradient_fraction_min": (
            float(np.min(block_nonzero_fractions))
            if block_nonzero_fractions
            else float("nan")
        ),
        "all_zero_block_gradient_batches": all_zero_block_gradient_batches,
        "all_zero_block_gradient_batch_fraction": (
            all_zero_block_gradient_batches / len(block_nonzero_fractions)
            if block_nonzero_fractions
            else float("nan")
        ),
        "parameter_norms": _parameter_role_norms(model),
        "seconds": elapsed,
    }
    diagnostic_summary: Dict[str, Any] = {
        key: float(np.mean(values)) for key, values in diagnostics_accumulator.items()
    }
    layer_summary = _surrogate_layer_summary(surrogate_layer_counts)
    if layer_summary:
        coverages = [row["surrogate_coverage"] for row in layer_summary.values()]
        total_active = sum(row["surrogate_active"] for row in layer_summary.values())
        total_elements = sum(row["surrogate_elements"] for row in layer_summary.values())
        diagnostic_summary.update(
            {
                "surrogate_coverage": total_active / total_elements,
                "surrogate_coverage_min": min(coverages),
                "surrogate_coverage_by_layer": layer_summary,
                "surrogate_aggregation_definition": (
                    "sum active and observed elements within each neuron layer across "
                    "train batches, then take the minimum layer coverage"
                ),
            }
        )
    result["diagnostics"] = diagnostic_summary
    return result


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[Any],
    device: torch.device,
    config: RunConfig,
    split: str,
) -> Dict[str, Any]:
    model.eval()
    losses, accuracies = AverageMeter(), AverageMeter()
    collect_activity = bool(config.analysis.get("collect_activity", False))
    diagnostics_accumulator: Dict[str, List[float]] = {}
    for batch_index, batch in enumerate(loader):
        if _batch_limit_reached(batch_index, config.runtime.limit_batches):
            break
        inputs, targets = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        logits, diagnostics = _forward(model, inputs, collect_activity)
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"Non-finite {split} loss at batch={batch_index}")
        batch_size = int(targets.shape[0])
        correct = int((logits.argmax(dim=1) == targets).sum().item())
        losses.update(float(loss.item()), batch_size)
        accuracies.update(correct / batch_size, batch_size)
        for key, value in diagnostics.items():
            safe = json_safe(value)
            if isinstance(safe, (int, float)) and math.isfinite(float(safe)):
                diagnostics_accumulator.setdefault(str(key), []).append(float(safe))
    if losses.count == 0:
        raise RuntimeError(f"{split} loader produced no batches")
    return {
        "loss": losses.average,
        "accuracy": accuracies.average,
        "samples": losses.count,
        "diagnostics": {key: float(np.mean(values)) for key, values in diagnostics_accumulator.items()},
    }


def _loss_auc(losses: Sequence[float]) -> float:
    if not losses:
        return float("nan")
    if len(losses) == 1:
        return float(losses[0])
    return float(sum((losses[i - 1] + losses[i]) * 0.5 for i in range(1, len(losses))))


def _diagnostic_value(history: Sequence[Mapping[str, Any]], names: Sequence[str]) -> float:
    values: List[float] = []
    normalized = {name.lower().replace("-", "_") for name in names}
    for row in history:
        diagnostics = row.get("diagnostics", {})
        if not isinstance(diagnostics, Mapping):
            continue
        for key, value in diagnostics.items():
            if str(key).lower().replace("-", "_") in normalized and isinstance(value, (int, float)):
                values.append(float(value))
    return float(np.mean(values)) if values else float("nan")


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val: Mapping[str, Any],
    config: RunConfig,
    train_history: Sequence[Mapping[str, Any]],
    training_environment_sha256: str | None = None,
) -> Dict[str, Any]:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "optimizer_group_manifest": optimizer_group_manifest(model, optimizer),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_val": dict(best_val),
        "config": config.as_dict(),
        "config_hash": config.config_hash,
        "execution_hash": config.execution_hash,
        "train_history": list(train_history),
        "rng_state": capture_rng_state(),
    }
    if training_environment_sha256 is not None:
        payload["training_environment_sha256"] = training_environment_sha256
    return payload


def _validate_optimizer_group_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint: Mapping[str, Any],
    *,
    require_manifest: bool,
) -> None:
    """Validate group identity before PyTorch can restore state by position."""

    saved_manifest = checkpoint.get("optimizer_group_manifest")
    if saved_manifest is None:
        if require_manifest:
            raise RuntimeError(
                "Resume checkpoint is missing the required optimizer_group_manifest"
            )
        return
    if not isinstance(saved_manifest, Mapping):
        raise RuntimeError("Resume optimizer_group_manifest must be a mapping")
    if saved_manifest.get("schema_version") != OPTIMIZER_GROUP_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("Resume optimizer_group_manifest schema is unsupported")
    saved_groups = saved_manifest.get("groups")
    if not isinstance(saved_groups, list):
        raise RuntimeError("Resume optimizer_group_manifest groups must be a list")
    current_groups = optimizer_group_manifest(model, optimizer)["groups"]
    current_state_groups = optimizer.state_dict()["param_groups"]
    optimizer_state = checkpoint.get("optimizer_state")
    state_groups = (
        optimizer_state.get("param_groups")
        if isinstance(optimizer_state, Mapping)
        else None
    )
    if not isinstance(state_groups, list):
        raise RuntimeError("Resume checkpoint optimizer_state param_groups are malformed")
    if not (
        len(saved_groups)
        == len(current_groups)
        == len(current_state_groups)
        == len(state_groups)
    ):
        raise RuntimeError("Resume optimizer group counts do not match")
    for index, (saved, current, current_state, state) in enumerate(
        zip(saved_groups, current_groups, current_state_groups, state_groups)
    ):
        if not isinstance(saved, Mapping) or not isinstance(state, Mapping):
            raise RuntimeError(f"Resume optimizer group {index} is malformed")
        for key in _OPTIMIZER_GROUP_IDENTITY_KEYS:
            if saved.get(key) != current.get(key):
                raise RuntimeError(
                    f"Resume optimizer group {index} differs for {key}"
                )
        if any(key in state for key in ("parameter_names", "role")):
            raise RuntimeError(
                f"Resume optimizer group {index} embeds audit metadata in optimizer_state"
            )
        if len(state.get("params", ())) != saved["parameter_count"]:
            raise RuntimeError(
                f"Resume optimizer group {index} state parameter count differs"
            )
        if list(state["params"]) != list(current_state["params"]):
            raise RuntimeError(
                f"Resume optimizer group {index} state parameter order differs"
            )
        for manifest_key, state_key in (
            ("current_lr", "lr"),
            ("initial_lr", "initial_lr"),
            ("weight_decay", "weight_decay"),
        ):
            if state_key not in state or float(saved[manifest_key]) != float(state[state_key]):
                raise RuntimeError(
                    f"Resume optimizer group {index} {manifest_key} differs from optimizer_state"
                )


def _validate_resume_optimizer_manifest_identity(
    previous_run_manifest: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
) -> None:
    """Bind a pre-existing run manifest to its resume checkpoint."""

    previous_manifest = previous_run_manifest.get("optimizer_group_manifest")
    checkpoint_manifest = checkpoint.get("optimizer_group_manifest")
    if previous_manifest is None and checkpoint_manifest is None:
        return
    if previous_manifest is None or checkpoint_manifest is None:
        raise RuntimeError(
            "Previous run manifest and resume checkpoint disagree on "
            "optimizer_group_manifest presence"
        )
    manifests = {
        "previous run": previous_manifest,
        "resume checkpoint": checkpoint_manifest,
    }
    groups_by_source: Dict[str, list[Any]] = {}
    for source, manifest in manifests.items():
        if not isinstance(manifest, Mapping):
            raise RuntimeError(f"{source} optimizer_group_manifest must be a mapping")
        if manifest.get("schema_version") != OPTIMIZER_GROUP_MANIFEST_SCHEMA_VERSION:
            raise RuntimeError(
                f"{source} optimizer_group_manifest schema is unsupported"
            )
        groups = manifest.get("groups")
        if not isinstance(groups, list):
            raise RuntimeError(
                f"{source} optimizer_group_manifest groups must be a list"
            )
        groups_by_source[source] = groups

    previous_groups = groups_by_source["previous run"]
    checkpoint_groups = groups_by_source["resume checkpoint"]
    if len(previous_groups) != len(checkpoint_groups):
        raise RuntimeError(
            "Previous run manifest and resume checkpoint optimizer group counts differ"
        )
    for index, (previous, saved) in enumerate(
        zip(previous_groups, checkpoint_groups)
    ):
        if not isinstance(previous, Mapping) or not isinstance(saved, Mapping):
            raise RuntimeError(f"Resume optimizer group identity {index} is malformed")
        for key in _OPTIMIZER_GROUP_IDENTITY_KEYS:
            if previous.get(key) != saved.get(key):
                raise RuntimeError(
                    "Previous run manifest and resume checkpoint optimizer group "
                    f"{index} differ for {key}"
                )


def validate_resume_checkpoint_path(path: str | Path) -> Path:
    """Permit deterministic continuation only from the post-epoch checkpoint."""

    checkpoint = Path(path)
    if checkpoint.name.lower() != "last.pt":
        raise ValueError(
            "Resume is only supported from last.pt. best.pt is saved before the epoch scheduler "
            "step, and failed.pt may contain a partially completed epoch."
        )
    return checkpoint


def _load_resume(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    config: RunConfig,
    expected_training_environment_sha256: str | None = None,
    expected_run_manifest: Mapping[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any], List[Dict[str, Any]]]:
    path = validate_resume_checkpoint_path(path)
    checkpoint = load_checkpoint(path, map_location="cpu")
    if checkpoint.get("config_hash") != config.config_hash:
        raise RuntimeError("Resume checkpoint config_hash does not match the current run")
    if expected_training_environment_sha256 is not None and checkpoint.get(
        "training_environment_sha256"
    ) != expected_training_environment_sha256:
        raise RuntimeError(
            "Resume checkpoint training environment differs from the current process"
        )
    checkpoint_config = checkpoint.get("config", {})
    checkpoint_runtime = (
        checkpoint_config.get("runtime", {}) if isinstance(checkpoint_config, Mapping) else {}
    )
    if isinstance(checkpoint_runtime, Mapping) and checkpoint_runtime.get("dry_run"):
        raise RuntimeError("A smoke/dry-run checkpoint cannot be resumed as a full experiment")
    if expected_run_manifest is not None:
        _validate_resume_optimizer_manifest_identity(
            expected_run_manifest, checkpoint
        )
    _validate_optimizer_group_checkpoint(
        model, optimizer, checkpoint, require_manifest=config.protocol_version >= 4
    )
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    for group in optimizer.param_groups:
        group.pop("parameter_names", None)
        group.pop("role", None)
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    saved_manifest = checkpoint.get("optimizer_group_manifest")
    if saved_manifest is not None and optimizer_group_manifest(model, optimizer) != dict(
        saved_manifest
    ):
        raise RuntimeError(
            "Loaded optimizer groups differ from optimizer_group_manifest"
        )
    if "rng_state" in checkpoint:
        restore_rng_state(checkpoint["rng_state"])
    history = checkpoint.get("train_history", [])
    if not isinstance(history, list):
        raise RuntimeError("Resume checkpoint train_history is malformed")
    return (
        int(checkpoint["epoch"]) + 1,
        dict(checkpoint.get("best_val", {})),
        [dict(item) for item in history],
    )


def _validate_resume_run_environment(
    run_dir: Path,
    resume_path: str | Path,
    current_environment_sha256: str,
    *,
    expected_run_id: str,
    expected_config_hash: str,
) -> dict[str, Any]:
    """Validate the pre-existing run identity before its manifest is replaced."""

    checkpoint = Path(resume_path).resolve()
    if not checkpoint.is_file():
        raise RuntimeError("Resume checkpoint last.pt is missing")
    if checkpoint.parent != run_dir.resolve():
        raise RuntimeError("Resume last.pt must belong to the current run directory")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Resume requires the previous run_manifest.json")
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read previous run manifest: {exc}") from exc
    if not isinstance(previous, dict):
        raise RuntimeError("Previous run manifest must be a JSON object")
    if previous.get("run_id") != expected_run_id:
        raise RuntimeError(
            "Previous run manifest run_id does not match the current run"
        )
    if previous.get("config_hash") != expected_config_hash:
        raise RuntimeError(
            "Previous run manifest config_hash does not match the current run"
        )
    environment = previous.get("environment")
    previous_hash = (
        environment.get("training_environment_sha256")
        if isinstance(environment, Mapping)
        else None
    )
    if previous_hash != current_environment_sha256:
        raise RuntimeError(
            "Cross-environment resume is forbidden: previous and current identities differ"
        )
    return previous


def _manifest_sha256(loaders: Mapping[str, Any]) -> str:
    manifest = loaders.get("_manifest", {})
    if isinstance(manifest, Mapping):
        value = manifest.get("manifest_sha256")
        if value:
            return str(value)
        return stable_hash(dict(manifest))
    return ""


def _shared_weight_sha256(model: torch.nn.Module) -> str:
    """Hash convolution/BN/classifier initialization for C1-C4 pairing audits."""

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not (
            "conv" in name or ".bn" in name or name.startswith("stem_bn")
            or name.startswith("fc.") or ".shortcut." in name
        ):
            continue
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _seed_metric_base(config: RunConfig, started_at: str) -> Dict[str, Any]:
    return {
        "run_id": config.runtime.run_id,
        "experiment": config.experiment,
        "dataset": config.data.dataset,
        "depth": config.model.depth,
        "time_steps": config.model.time_steps,
        "condition": config.model.condition,
        "topology": config.model.topology,
        "neuron": config.model.neuron,
        "seed": config.runtime.seed,
        "config_hash": config.config_hash,
        "protocol_hash": config.analysis.get("protocol_hash", ""),
        "split_manifest_sha256": "",
        "shared_weight_sha256": "",
        "training_environment_identity": "",
        "training_environment_sha256": "",
        "status": "running",
        "failed": 0,
        "best_epoch": "",
        "best_val_loss": "",
        "best_val_accuracy": "",
        "test_loss": "",
        "test_accuracy": "",
        "test_samples": "",
        "train_loss_auc": "",
        "converged": "",
        "convergence_epoch": "",
        "training_gradient_cv": "",
        "training_gradient_cv_method": "mean_batch_cv_of_residual_block_parameter_gradient_l2",
        "gradient_cv": "",
        "gradient_cv_method": "",
        "jacobian_phi": "",
        "jacobian_varphi": "",
        "spike_rate": "",
        "peak_training_memory_bytes": "",
        "train_epoch_seconds": "",
        "started_at": started_at,
        "finished_at": "",
        "duration_s": "",
        "test_checkpoint_sha256": "",
        "test_evaluated_at": "",
        "diagnostic_id": "",
        "diagnostic_batch_sha256": "",
        "diagnostic_protocol_hash": "",
        "jacobian_method": "",
        "jacobian_probes": "",
        "jacobian_probe_seed": "",
        "jacobian_time_index": "",
    }


def _training_environment_identity(
    device: torch.device, *, amp: bool, deterministic: bool
) -> tuple[str, str]:
    environment = environment_manifest()
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        resolved_device = f"cuda:{index}"
        hardware: dict[str, Any] = {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": int(properties.total_memory),
            "multiprocessor_count": int(properties.multi_processor_count),
        }
    else:
        resolved_device = str(device)
        hardware = {"name": "cpu"}
    identity = {
        "device": resolved_device,
        "hardware": hardware,
        "software": {
            "platform": environment["platform"],
            "python": environment["python"],
            "pytorch": environment["pytorch"],
            "numpy": environment["numpy"],
            "cuda_version": environment["cuda_version"],
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "precision": "amp_float16" if amp and device.type == "cuda" else "float32",
        "determinism": {
            "requested": bool(deterministic),
            "torch_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "torch_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run(
    config: RunConfig,
    *,
    orchestrator_evidence: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Execute a run, preserving both successful and failed run artifacts."""

    expected_frozen_environment: str | None = None
    if config.protocol_version in (4, 5):
        if not isinstance(orchestrator_evidence, Mapping):
            raise ValueError(
                f"Protocol v{config.protocol_version} requires validated "
                "pilot/formal execution evidence"
            )
        if orchestrator_evidence.get("protocol_version") != config.protocol_version:
            raise ValueError(
                f"Protocol v{config.protocol_version} execution evidence has the wrong version"
            )
        execution_stage = orchestrator_evidence.get("execution_stage")
        if execution_stage not in {"pilot", "formal"}:
            raise ValueError(
                f"Protocol v{config.protocol_version} execution evidence has no valid stage"
            )
        if execution_stage == "formal":
            expected_frozen_environment = str(
                orchestrator_evidence.get("training_environment_sha256", "")
            )
            if len(expected_frozen_environment) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_frozen_environment
            ):
                raise ValueError(
                    f"Protocol v{config.protocol_version} formal evidence has no valid "
                    "training environment hash"
                )
        else:
            runtime_context = orchestrator_evidence.get("runtime_context")
            if not isinstance(runtime_context, Mapping) or runtime_context.get("pass") is not True:
                raise ValueError(
                    f"Protocol v{config.protocol_version} pilot evidence has no validated "
                    "health runtime context"
                )
    if config.protocol_version < 4 and (
        config.model.terminal_neuron_mode != "always"
        or config.optimizer.warmup_epochs != 0
        or config.optimizer.exclude_norm_and_bias_from_weight_decay
    ):
        raise ValueError("Post-v3 repair controls cannot execute under protocols v1-v3")
    if config.final_test:
        raise ValueError(
            "Training configs must keep final_test=false; use scripts/evaluate_checkpoints.py "
            "after all best checkpoints and the model-selection rule are frozen"
        )
    started_at = utc_now()
    started_clock = time.perf_counter()
    seed_everything(config.runtime.seed, config.runtime.deterministic)
    device = resolve_device(config.runtime.device)
    training_environment, training_environment_sha256 = _training_environment_identity(
        device, amp=config.runtime.amp, deterministic=config.runtime.deterministic
    )
    if (
        expected_frozen_environment is not None
        and training_environment_sha256 != expected_frozen_environment
    ):
        raise RuntimeError(
            f"Protocol v{config.protocol_version} current training environment differs "
            "from the pilot/formal freeze"
        )
    run_dir = Path(config.runtime.output_dir) / config.runtime.run_id
    existing_metrics = run_dir / "seed_metrics.json"
    if existing_metrics.exists() and not config.runtime.resume and not config.runtime.dry_run:
        raise FileExistsError(
            f"Run {config.runtime.run_id} already has terminal metrics; refusing to overwrite it"
        )
    if config.runtime.resume:
        if not run_dir.is_dir():
            raise RuntimeError("Resume run directory is missing")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    logger = JSONLLogger(run_dir / "events.jsonl")
    metrics_row = _seed_metric_base(config, started_at)
    metrics_row.update(
        {
            "training_environment_identity": training_environment,
            "training_environment_sha256": training_environment_sha256,
        }
    )
    previous_manifest: dict[str, Any] | None = None
    if config.runtime.resume:
        previous_manifest = _validate_resume_run_environment(
            run_dir,
            config.runtime.resume,
            training_environment_sha256,
            expected_run_id=config.runtime.run_id,
            expected_config_hash=config.config_hash,
        )
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any = None
    best_val: Dict[str, Any] = {"accuracy": -math.inf, "loss": math.inf, "epoch": -1}
    current_epoch = -1
    resume_writes_authorized = not bool(config.runtime.resume)
    if resume_writes_authorized:
        atomic_write_json(run_dir / "resolved_config.json", config.as_dict())
    try:
        # Keep the independent test data outside the entire model-selection
        # loop.  It is constructed only after best.pt has been frozen.
        loaders = build_loaders(config, final_test=False, seed=config.runtime.seed)
        metrics_row["split_manifest_sha256"] = _manifest_sha256(loaders)
        model_cfg = dataclasses.asdict(config.model)
        # The model uses a private CPU generator for initialization.  The run
        # seed therefore yields identical shared weights across C1-C4 while
        # still varying initialization across the five complete blocks.
        model_cfg["init_seed"] = config.runtime.seed
        model = build_model(model_cfg).to(device)
        metrics_row["shared_weight_sha256"] = _shared_weight_sha256(model)
        report = model.parameter_report() if hasattr(model, "parameter_report") else {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        }
        optimizer, scheduler, ta_parameters, ta_names, ta_activation_epoch = build_optimizer_and_scheduler(model, config)
        manifest = {
            "run_id": config.runtime.run_id,
            "status": "running",
            "created_at": started_at,
            "config_hash": config.config_hash,
            "execution_hash": config.execution_hash,
            "config": config.as_dict(),
            "split_manifest_sha256": metrics_row["split_manifest_sha256"],
            "parameter_report": json_safe(report),
            "shared_weight_sha256": metrics_row["shared_weight_sha256"],
            "ta_parameter_names": ta_names,
            "ta_activation_epoch_zero_based": ta_activation_epoch,
            "optimizer_group_manifest": optimizer_group_manifest(model, optimizer),
            "environment": {
                **environment_manifest(),
                "training_environment_identity": json.loads(training_environment),
                "training_environment_sha256": training_environment_sha256,
            },
        }
        if orchestrator_evidence is not None:
            manifest["orchestrator_evidence"] = dict(orchestrator_evidence)
        if previous_manifest is not None:
            prior_history = previous_manifest.get("resume_history", [])
            if not isinstance(prior_history, list):
                raise RuntimeError("Previous run manifest resume_history is malformed")
            manifest["resume_history"] = [
                *prior_history,
                {
                    "resumed_at": started_at,
                    "checkpoint": "last.pt",
                    "training_environment_sha256": training_environment_sha256,
                },
            ]
        start_epoch = 0
        train_history: List[Dict[str, Any]] = []
        if config.runtime.resume:
            start_epoch, best_val, train_history = _load_resume(
                config.runtime.resume,
                model,
                optimizer,
                scheduler,
                config,
                expected_training_environment_sha256=training_environment_sha256,
                expected_run_manifest=previous_manifest,
            )
            manifest["optimizer_group_manifest"] = optimizer_group_manifest(
                model, optimizer
            )
            resume_writes_authorized = True
            atomic_write_json(run_dir / "resolved_config.json", config.as_dict())
        # Commit the new audit record only after every resume check and state
        # load has succeeded.  A rejected resume leaves the previous manifest
        # byte-for-byte intact.
        atomic_write_json(run_dir / "run_manifest.json", manifest)
        logger.log(
            "run_started", device=str(device), dry_run=config.runtime.dry_run,
            config_hash=config.config_hash, parameter_report=report,
            optimizer_group_manifest=manifest["optimizer_group_manifest"],
            ta_activation_epoch_zero_based=ta_activation_epoch,
            orchestrator_evidence=(
                dict(orchestrator_evidence) if orchestrator_evidence is not None else None
            ),
        )
        if config.runtime.resume:
            logger.log("resumed", checkpoint=config.runtime.resume, start_epoch=start_epoch)
        # Resume must restore whether TA parameters are still frozen.
        _set_ta_enabled(
            ta_parameters,
            _ta_enabled_for_epoch(ta_parameters, start_epoch, ta_activation_epoch),
        )
        amp_enabled = bool(config.runtime.amp and device.type == "cuda")
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        epochs = 1 if config.runtime.dry_run else config.optimizer.epochs
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for current_epoch in range(start_epoch, epochs):
            sampler = getattr(loaders["train"], "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(current_epoch)
            ta_enabled = _ta_enabled_for_epoch(
                ta_parameters, current_epoch, ta_activation_epoch
            )
            _set_ta_enabled(ta_parameters, ta_enabled)
            train_metrics = train_one_epoch(model, loaders["train"], optimizer, device, current_epoch, config, logger, scaler)
            val_metrics = evaluate(model, loaders["val"], device, config, "val")
            history_row = dict(train_metrics)
            history_row.update(
                {"epoch": current_epoch, "val_accuracy": val_metrics["accuracy"], "val_loss": val_metrics["loss"]}
            )
            train_history.append(history_row)
            improved = (
                val_metrics["accuracy"] > best_val.get("accuracy", -math.inf)
                or (
                    val_metrics["accuracy"] == best_val.get("accuracy", -math.inf)
                    and val_metrics["loss"] < best_val.get("loss", math.inf)
                )
            )
            if improved:
                best_val = {"accuracy": val_metrics["accuracy"], "loss": val_metrics["loss"], "epoch": current_epoch}
                save_checkpoint(
                    run_dir / "best.pt",
                    _checkpoint_payload(
                        model, optimizer, scheduler, current_epoch, best_val, config, train_history
                        , training_environment_sha256
                    ),
                )
            logger.log(
                "epoch_completed", epoch=current_epoch, train=train_metrics, val=val_metrics,
                learning_rates=_optimizer_learning_rates(optimizer),
                ta_enabled=ta_enabled, best_val=best_val,
            )
            scheduler.step()
            if (current_epoch + 1) % config.runtime.checkpoint_every == 0 or current_epoch + 1 == epochs:
                save_checkpoint(
                    run_dir / "last.pt",
                    _checkpoint_payload(
                        model, optimizer, scheduler, current_epoch, best_val, config, train_history
                        , training_environment_sha256
                    ),
                )
        if not train_history:
            raise RuntimeError("No epoch was run; check resume epoch and configured total epochs")

        gradient_cvs = [float(item["gradient_cv"]) for item in train_history if math.isfinite(float(item["gradient_cv"]))]
        epoch_times = [float(item["seconds"]) for item in train_history]
        thresholds = config.analysis.get("validation_accuracy_thresholds", {})
        threshold = thresholds.get(config.data.dataset) if isinstance(thresholds, Mapping) else None
        convergence_epochs = [
            int(item["epoch"])
            for item in train_history
            if threshold is not None and float(item.get("val_accuracy", -math.inf)) >= float(threshold)
        ]
        metrics_row.update({
            "status": "dry_run" if config.runtime.dry_run else "complete",
            "best_epoch": int(best_val["epoch"]) + 1,
            "best_val_loss": best_val["loss"],
            "best_val_accuracy": best_val["accuracy"],
            "train_loss_auc": _loss_auc([float(item["loss"]) for item in train_history]),
            "converged": int(bool(convergence_epochs)) if threshold is not None else "",
            "convergence_epoch": min(convergence_epochs) + 1 if convergence_epochs else "",
            "training_gradient_cv": float(np.mean(gradient_cvs)) if gradient_cvs else "",
            "training_gradient_cv_method": "mean_batch_cv_of_residual_block_parameter_gradient_l2",
            "jacobian_phi": _diagnostic_value(train_history, ("jacobian_phi", "phi")),
            "jacobian_varphi": _diagnostic_value(train_history, ("jacobian_varphi", "varphi")),
            "spike_rate": _diagnostic_value(train_history, ("spike_rate",)),
            "peak_training_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "train_epoch_seconds": float(np.mean(epoch_times)),
        })
        metrics_row["finished_at"] = utc_now()
        metrics_row["duration_s"] = time.perf_counter() - started_clock
        atomic_write_json(run_dir / "seed_metrics.json", metrics_row)
        manifest.update(
            {
                "status": metrics_row["status"],
                "finished_at": metrics_row["finished_at"],
                "optimizer_group_manifest": optimizer_group_manifest(
                    model, optimizer
                ),
            }
        )
        atomic_write_json(run_dir / "run_manifest.json", manifest)
        metrics_csv = Path(config.runtime.output_dir) / "seed_metrics.csv"
        if config.runtime.resume or config.runtime.dry_run:
            upsert_csv_row(metrics_csv, metrics_row, SEED_METRIC_FIELDS, key_field="run_id")
        else:
            append_csv_row(metrics_csv, metrics_row, SEED_METRIC_FIELDS)
        logger.log("run_completed", metrics=metrics_row)
        return metrics_row
    except BaseException as exc:
        if config.runtime.resume and not resume_writes_authorized:
            raise
        metrics_row.update({
            "status": "failed",
            "failed": 1,
            "finished_at": utc_now(),
            "duration_s": time.perf_counter() - started_clock,
        })
        failure = {
            "run_id": config.runtime.run_id,
            "timestamp": metrics_row["finished_at"],
            "config_hash": config.config_hash,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "epoch": current_epoch,
        }
        atomic_write_json(run_dir / "failure.json", failure)
        atomic_write_json(run_dir / "seed_metrics.json", metrics_row)
        try:
            metrics_csv = Path(config.runtime.output_dir) / "seed_metrics.csv"
            if config.runtime.resume or config.runtime.dry_run:
                upsert_csv_row(metrics_csv, metrics_row, SEED_METRIC_FIELDS, key_field="run_id")
            else:
                append_csv_row(metrics_csv, metrics_row, SEED_METRIC_FIELDS)
        except Exception as csv_exc:  # retain both failures; never hide the training exception
            failure["seed_metrics_append_error"] = repr(csv_exc)
            atomic_write_json(run_dir / "failure.json", failure)
        if model is not None and optimizer is not None and scheduler is not None:
            try:
                save_checkpoint(
                    run_dir / "failed.pt",
                    _checkpoint_payload(
                        model, optimizer, scheduler, current_epoch, best_val, config,
                        train_history if "train_history" in locals() else [],
                        training_environment_sha256,
                    ),
                )
            except Exception as checkpoint_exc:
                failure["failure_checkpoint_error"] = repr(checkpoint_exc)
                atomic_write_json(run_dir / "failure.json", failure)
        logger.log("run_failed", failure=failure)
        raise


def _override_config(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    runtime = config.runtime
    if args.dry_run:
        smoke_root = Path(args.output_dir or runtime.output_dir) / "smoke"
        runtime = dataclasses.replace(
            runtime,
            dry_run=True,
            device=args.device or "cpu",
            output_dir=str(smoke_root),
            limit_batches=args.limit_batches or runtime.limit_batches or 1,
        )
    if args.device is not None:
        runtime = dataclasses.replace(runtime, device=args.device)
    if args.resume is not None:
        runtime = dataclasses.replace(runtime, resume=args.resume)
    if args.output_dir is not None and not args.dry_run:
        runtime = dataclasses.replace(runtime, output_dir=args.output_dir)
    return dataclasses.replace(config, runtime=runtime, final_test=False)


def _formal_execution_evidence(
    protocol: Mapping[str, Any],
    freeze_manifest: Mapping[str, Any],
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Extract the one pilot-bound environment authorized for v4/v5 formal work."""

    version = int(protocol.get("protocol_version", 0))
    if version not in (4, 5):
        raise ValueError("Frozen TA-LIF formal evidence requires protocol_version 4 or 5")
    pilot = freeze_manifest.get("pilot_validation")
    if not isinstance(pilot, Mapping):
        raise ValueError(f"V{version} freeze manifest has no pilot_validation binding")
    expected_top = {
        "status": "PASS",
        "pass": True,
        "protocol_hash": stable_hash(protocol),
    }
    mismatches = [
        key for key, expected in expected_top.items() if pilot.get(key) != expected
    ]
    datasets = pilot.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != {
        "cifar100",
        "cifar10dvs",
    }:
        mismatches.append("datasets")
        datasets = {}
    environment_hashes = {
        str(evidence.get("environment_sha256", ""))
        for evidence in datasets.values()
        if isinstance(evidence, Mapping)
    }
    if (
        len(environment_hashes) != 1
        or len(next(iter(environment_hashes), "")) != 64
        or any(
            character not in "0123456789abcdef"
            for character in next(iter(environment_hashes), "")
        )
    ):
        mismatches.append("single shared pilot training environment")
    pilot_environment_sha256 = pilot.get("training_environment_sha256")
    if version == 5 and (
        not isinstance(pilot_environment_sha256, str)
        or environment_hashes != {pilot_environment_sha256}
    ):
        mismatches.append("aggregate pilot training_environment_sha256")
    if mismatches:
        raise ValueError(
            f"V{version} formal execution evidence is incomplete or inconsistent: "
            + ", ".join(mismatches)
        )

    validation_sha256 = str(pilot.get("sha256", ""))
    if len(validation_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in validation_sha256
    ):
        raise ValueError(f"V{version} pilot validation artifact hash is missing")
    manifest_path = PROJECT_ROOT / artifact_paths_for_protocol(protocol)[
        "freeze_manifest"
    ]
    if not manifest_path.is_file():
        raise ValueError(f"V{version} freeze manifest disappeared after verification")
    return {
        "protocol_version": version,
        "execution_stage": "formal",
        "protocol": artifact_path_reference(protocol_path, PROJECT_ROOT),
        "protocol_hash": stable_hash(protocol),
        "pilot_validation": str(pilot.get("path", "")),
        "pilot_validation_sha256": validation_sha256,
        "freeze_manifest": artifact_path_reference(manifest_path, PROJECT_ROOT),
        "freeze_manifest_sha256": sha256_file(manifest_path),
        "training_environment_sha256": next(iter(environment_hashes)),
    }


def _v4_formal_execution_evidence(
    protocol: Mapping[str, Any],
    freeze_manifest: Mapping[str, Any],
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Backward-compatible v4 wrapper used by existing release tooling."""

    if int(protocol.get("protocol_version", 0)) != 4:
        raise ValueError("V4 formal evidence requires protocol_version 4")
    return _formal_execution_evidence(
        protocol,
        freeze_manifest,
        protocol_path=protocol_path,
    )


def _load_pilot_runtime_validator(version: int):
    tool_path = PROJECT_ROOT / "scripts" / f"pilot_health_gate_v{version}.py"
    scripts_root = str(tool_path.parent)
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    spec = importlib.util.spec_from_file_location(
        f"talif_v{version}_runtime_health_gate", tool_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load the v{version} runtime health validator")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise ValueError(
            f"Cannot import the v{version} runtime health validator: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    validator = getattr(module, "validate_current_runtime_against_health_report", None)
    if not callable(validator):
        raise TypeError(f"The v{version} runtime health validator has no entry point")
    return validator


def _validate_orchestrator_pilot_plan(
    plan_path: Path,
    *,
    config: RunConfig,
    config_path: str | Path,
    protocol_path: str | Path,
    protocol_hash: str,
    output_dir_was_explicit: bool,
) -> dict[str, Any]:
    """Revalidate the runner-created global pilot plan inside the trainer."""

    resolved_plan = plan_path.resolve()
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Pilot plan must be a mapping")
    protocol = load_protocol(protocol_path)
    version = int(protocol.get("protocol_version", 1))
    is_talif_pilot = (
        version in (3, 4, 5)
        and protocol.get("study_stage") == "pilot_and_formal"
    )
    talif_block: Any = None
    runtime_context: Mapping[str, Any] | None = None
    if is_talif_pilot:
        contracts = {
            3: (
                resolve_pilot_block,
                PilotV3Error,
                require_v3_author_freeze,
                validate_v3_health_report,
                expected_pilot_plan_payload,
                "NON_REPORTABLE_V3_PILOT_PLAN",
            ),
            4: (
                resolve_v4_pilot_block,
                PilotV4Error,
                require_v4_author_freeze,
                validate_v4_health_report,
                expected_v4_pilot_plan_payload,
                "NON_REPORTABLE_V4_PILOT_PLAN",
            ),
            5: (
                resolve_v5_pilot_block,
                PilotV5Error,
                require_v5_author_freeze,
                validate_v5_health_report,
                expected_v5_pilot_plan_payload,
                "NON_REPORTABLE_V5_PILOT_PLAN",
            ),
        }
        (
            resolve_block,
            pilot_error,
            require_author_freeze,
            validate_health_report,
            build_expected_plan,
            artifact_class,
        ) = contracts[version]
        try:
            talif_block = resolve_block(
                protocol,
                str(payload.get("dataset", "")),
                repository_root=PROJECT_ROOT,
            )
        except pilot_error as exc:
            raise ValueError(
                f"Pilot plan does not identify a valid v{version} block: {exc}"
            ) from exc
        if resolved_plan != talif_block.pilot_plan:
            raise ValueError(
                f"Protocol v{version} pilot plan path differs from the dataset binding"
            )
        if (
            payload.get("version") != version
            or payload.get("artifact_class") != artifact_class
            or payload.get("non_reportable") is not True
        ):
            raise ValueError(
                f"Pilot plan is not a recognized v{version} non-reportable plan"
            )
        try:
            require_author_freeze(protocol, repository_root=PROJECT_ROOT)
        except pilot_error as exc:
            raise ValueError(
                f"Protocol v{version} pilot requires author freeze: {exc}"
            ) from exc
    else:
        default_plan = PILOT_PLAN_PATH.resolve()
        if resolved_plan != default_plan and (
            resolved_plan.parent != default_plan.parent
            or not resolved_plan.name.startswith("unfrozen_pilot_plan_")
            or resolved_plan.suffix.lower() != ".json"
        ):
            raise ValueError(
                "Pilot plan must use the default audited path or a versioned "
                "unfrozen_pilot_plan_*.json sibling"
            )
        if payload.get("version") not in {1, 2} or payload.get("non_reportable") is not True:
            raise ValueError("Pilot plan is not a recognized non-reportable plan")
    if payload.get("protocol_hash") != protocol_hash:
        raise ValueError("Pilot plan protocol hash differs from the current protocol")
    if payload.get("protocol_path") != artifact_path_reference(protocol_path, PROJECT_ROOT):
        raise ValueError("Pilot plan protocol path differs from the requested protocol")
    if not is_talif_pilot and protocol.get("protocol_status", {}).get("frozen") is True:
        raise ValueError("A frozen protocol cannot use the legacy unfrozen pilot plan")
    is_v2_pilot = (
        protocol.get("protocol_version") == 2
        and protocol.get("study_stage") == "pilot"
    )
    if is_v2_pilot:
        if payload.get("version") != 2:
            raise ValueError("The v2 pilot requires a health-bound version-2 plan")
        acceptance = protocol.get("pilot_acceptance")
        if not isinstance(acceptance, Mapping):
            raise ValueError("The v2 pilot protocol has no pilot_acceptance mapping")
        if payload.get("acceptance_hash") != stable_hash(dict(acceptance)):
            raise ValueError("Pilot plan acceptance hash differs from the protocol")
        health_path = Path(str(acceptance.get("health_output", "")))
        if not health_path.is_absolute():
            health_path = PROJECT_ROOT / health_path
        health_path = health_path.resolve()
        if payload.get("health_report") != artifact_path_reference(
            health_path, PROJECT_ROOT
        ):
            raise ValueError("Pilot plan health-report path differs from the protocol")
        if not health_path.is_file() or payload.get("health_report_sha256") != sha256_file(
            health_path
        ):
            raise ValueError("Pilot plan health-report SHA-256 is missing or stale")
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        current_commit = completed.stdout.strip()
        if completed.returncode != 0 or payload.get("git_commit") != current_commit:
            raise ValueError("Pilot plan Git commit differs from the current checkout")
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if status.returncode != 0 or status.stdout.strip():
            raise ValueError("The tracked worktree changed after the pilot health gate")
    elif is_talif_pilot:
        assert talif_block is not None
        try:
            health_report = validate_health_report(
                talif_block.health_output,
                protocol_path,
                talif_block.dataset,
                repository_root=PROJECT_ROOT,
                require_current_tracked_clean=True,
            )
        except pilot_error as exc:
            raise ValueError(
                f"V{version} pilot health report is invalid: {exc}"
            ) from exc
        manifest_path = Path(config_path).resolve().parent / "run_manifest.csv"
        try:
            with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                manifest_rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ValueError(
                f"Cannot read the v{version} pilot run manifest: {exc}"
            ) from exc
        expected_plan = build_expected_plan(
            block=talif_block,
            protocol_path=protocol_path,
            config_dir=Path(config_path).resolve().parent,
            manifest_rows=manifest_rows,
            health_report=health_report,
            repository_root=PROJECT_ROOT,
        )
        if dict(payload) != expected_plan:
            raise ValueError(
                f"V{version} pilot plan differs from the protocol-bound dataset block"
            )
        if (
            config.data.dataset != talif_block.dataset
            or config.runtime.seed != talif_block.seed
        ):
            raise ValueError(
                f"Current run dataset/seed differs from the v{version} pilot plan"
            )
        runtime_validator = _load_pilot_runtime_validator(version)
        reference_config = load_run_config(
            Path(config_path).resolve().parent
            / next(
                row["config_file"]
                for row in manifest_rows
                if row.get("dataset") == talif_block.dataset
                and row.get("condition") == "C1"
            ),
            protocol_path,
        )
        try:
            runtime_context = runtime_validator(
                health_report,
                protocol=protocol,
                reference_config=reference_config,
                block=talif_block,
                device=config.runtime.device,
            )
        except pilot_error as exc:
            raise ValueError(
                f"V{version} pilot runtime/data context is invalid: {exc}"
            ) from exc
    if not output_dir_was_explicit:
        raise ValueError("Pilot execution requires an explicit orchestrator output directory")
    protocol_output_root = protocol.get("output_root", "results")
    formal_root = Path(str(protocol_output_root))
    if not formal_root.is_absolute():
        formal_root = PROJECT_ROOT / formal_root
    actual_root = Path(config.runtime.output_dir).resolve()
    if payload.get("pilot_output_root") != artifact_path_reference(actual_root, PROJECT_ROOT):
        raise ValueError("Pilot output directory differs from the global plan")
    if payload.get("formal_output_root") != artifact_path_reference(formal_root, PROJECT_ROOT):
        raise ValueError("Formal output directory differs from the global plan")
    pilot_root = actual_root
    formal_root = formal_root.resolve()
    if pilot_root == formal_root or formal_root in pilot_root.parents or pilot_root in formal_root.parents:
        raise ValueError("Pilot and formal output roots are not isolated")
    runs = payload.get("runs")
    expected_conditions = (
        {"C1", "C2"} if is_talif_pilot else {"C1", "C2", "C3", "C4"}
    )
    expected_run_count = len(expected_conditions)
    if not isinstance(runs, list) or len(runs) != expected_run_count:
        raise ValueError(
            f"Pilot plan must contain exactly {expected_run_count} authorized runs"
        )
    conditions = {str(run.get("condition")) for run in runs if isinstance(run, Mapping)}
    if conditions != expected_conditions:
        raise ValueError(
            "Pilot plan conditions must be exactly " + ",".join(sorted(expected_conditions))
        )
    matches = [
        run
        for run in runs
        if isinstance(run, Mapping) and run.get("run_id") == config.runtime.run_id
    ]
    if len(matches) != 1:
        raise ValueError("Current run is not uniquely authorized by the pilot plan")
    planned = matches[0]
    if planned.get("config_hash") != config.config_hash:
        raise ValueError("Current run config hash differs from the pilot plan")
    if planned.get("config_file") != artifact_path_reference(config_path, PROJECT_ROOT):
        raise ValueError("Current run config path differs from the pilot plan")
    if planned.get("config_file_sha256") != sha256_file(config_path):
        raise ValueError("Current run config file SHA-256 differs from the pilot plan")
    evidence = {
        "pilot_plan": artifact_path_reference(resolved_plan, PROJECT_ROOT),
        "pilot_plan_sha256": sha256_file(resolved_plan),
        "plan_version": payload.get("version"),
        "protocol_hash": protocol_hash,
    }
    if is_v2_pilot:
        evidence.update(
            {
                "acceptance_hash": payload.get("acceptance_hash"),
                "git_commit": payload.get("git_commit"),
                "health_report": payload.get("health_report"),
                "health_report_sha256": payload.get("health_report_sha256"),
            }
        )
    elif is_talif_pilot:
        assert runtime_context is not None
        evidence.update(
            {
                "protocol_version": version,
                "execution_stage": "pilot",
                "acceptance_hash": payload.get("acceptance_hash"),
                "block_hash": payload.get("block_hash"),
                "dataset": payload.get("dataset"),
                "git_commit": payload.get("git_commit"),
                "health_report": payload.get("health_report"),
                "health_report_sha256": payload.get("health_report_sha256"),
                "attempt_receipt": payload.get("attempt_receipt"),
                "attempt_receipt_sha256": payload.get("attempt_receipt_sha256"),
                "runtime_context": dict(runtime_context),
            }
        )
        if version == 5:
            evidence.update(
                {
                    "development_probe_hash": payload.get("development_probe_hash"),
                    "health_seed": payload.get("health_seed"),
                    "pilot_seed": payload.get("pilot_seed"),
                }
            )
    return evidence


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to one generated run YAML")
    parser.add_argument(
        "--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"),
        help="Current protocol YAML; required for hash/freeze checks",
    )
    parser.add_argument("--orchestrator-pilot-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--resume", help="Checkpoint to resume; config hashes must match")
    parser.add_argument("--output-dir", help="Override the result root in the frozen run config")
    parser.add_argument("--dry-run", action="store_true", help="Run one epoch with a limited number of batches")
    parser.add_argument("--limit-batches", type=int, help="Limit train/val batches for a smoke run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.limit_batches is not None and args.limit_batches < 1:
        raise SystemExit("--limit-batches must be >= 1")
    if args.limit_batches is not None and not args.dry_run:
        raise SystemExit("--limit-batches is only permitted together with --dry-run")
    if args.resume is not None:
        try:
            validate_resume_checkpoint_path(args.resume)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    pilot_active = args.orchestrator_pilot_plan is not None
    if pilot_active and args.dry_run:
        raise SystemExit("The orchestrator pilot plan cannot be used for a dry run")
    protocol = load_protocol(args.protocol)
    protocol_version = int(protocol.get("protocol_version", 1))
    if protocol_version in (3, 4, 5) and args.dry_run:
        raise SystemExit(
            f"Protocol v{protocol_version} training dry-runs are disabled; use "
            "generate_run_configs.py --dry-run and reserve training commands for "
            "the author-frozen health/pilot gates"
        )
    report = check_protocol(
        args.protocol,
        mode="smoke" if args.dry_run else ("pilot" if pilot_active else "full"),
        project_root=PROJECT_ROOT,
        check_dependencies=False,
    )
    if not report.ok:
        raise SystemExit("Protocol preflight blocked training:\n" + "\n".join(f"- {e}" for e in report.errors))
    formal_freeze: Mapping[str, Any] | None = None
    if not args.dry_run and not pilot_active:
        try:
            formal_freeze = verify_formal_freeze(
                project_root=PROJECT_ROOT,
                protocol_path=args.protocol,
                matrix_dir=Path(args.config).resolve().parent,
            )
        except FreezeGateError as exc:
            raise SystemExit(f"Formal freeze-manifest gate blocked training: {exc}") from exc
    config = _override_config(load_run_config(args.config, args.protocol), args)
    if config.analysis.get("protocol_hash") != report.protocol_hash:
        raise SystemExit("Run config protocol hash is stale; regenerate configs from the current protocol")
    if (
        protocol_version in (3, 4, 5)
        and not args.dry_run
        and not pilot_active
    ):
        expected_output = Path(str(protocol["output_root"]))
        if not expected_output.is_absolute():
            expected_output = PROJECT_ROOT / expected_output
        if Path(config.runtime.output_dir).resolve() != expected_output.resolve():
            raise SystemExit(
                f"Protocol v{protocol_version} formal training must use its isolated output_root"
            )
    orchestrator_evidence: dict[str, Any] | None = None
    if args.orchestrator_pilot_plan is not None:
        try:
            orchestrator_evidence = _validate_orchestrator_pilot_plan(
                args.orchestrator_pilot_plan,
                config=config,
                config_path=args.config,
                protocol_path=args.protocol,
                protocol_hash=report.protocol_hash,
                output_dir_was_explicit=args.output_dir is not None,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Pilot plan validation failed: {exc}") from exc
    elif protocol_version in (4, 5) and not args.dry_run:
        if formal_freeze is None:
            raise SystemExit(
                f"Protocol v{protocol_version} formal execution has no verified freeze manifest"
            )
        try:
            orchestrator_evidence = _formal_execution_evidence(
                protocol,
                formal_freeze,
                protocol_path=args.protocol,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"V{protocol_version} formal environment gate failed: {exc}"
            ) from exc
    try:
        result = run(config, orchestrator_evidence=orchestrator_evidence)
    except BaseException as exc:
        print(f"Run {config.runtime.run_id} failed: {exc}", file=sys.stderr)
        return 1
    print(f"Run {result['run_id']} finished with status={result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
