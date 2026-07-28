#!/usr/bin/env python3
"""Diagnose MS-ResNet train/eval divergence without producing reportable results.

The diagnostic trains one fixed CIFAR-100 batch once per C1--C4 condition, then
evaluates the same final model state three ways: ordinary eval-mode BatchNorm,
full train mode, and eval mode with only BatchNorm layers using batch statistics.
It also captures each BatchNorm input distribution at every simulation step.

The JSON artifact is engineering evidence only.  Its seed must never be reused
for a pilot/formal run, and the artifact must never enter manuscript results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import pilot_health_gate as gate  # noqa: E402
from talif_msresnet.config import RunConfig, load_protocol, load_run_config  # noqa: E402
from talif_msresnet.train import _forward  # noqa: E402
from talif_msresnet.utils import (  # noqa: E402
    atomic_write_json,
    environment_manifest,
    sha256_file,
    stable_hash,
    utc_now,
)


ARTIFACT_CLASS = "NON_REPORTING_MSRESNET_BATCHNORM_ROOT_CAUSE_DIAGNOSTIC"
CONDITIONS_IN_ORDER = ("C1", "C2", "C3", "C4")
FORBIDDEN_EXPERIMENT_SEEDS = frozenset({11, 22, 33, 44, 55, 77, 88})
FAILED_HEALTH_SEED = 88
FAILED_HEALTH_COMMIT = "14e5d2421709654a8e6c5fc8250a786ad5513f10"
FAILED_HEALTH_SHA256 = "6aaa1d0ae2c046187e6c348449dd998668a522f87769ddaac88f5df1f511412f"
FAILED_PROTOCOL_HASH = "77109228d1f6479d47a97a011bf52a60987e9083ff8bab5d7ed64c7176c6af8a"
FAILED_ACCEPTANCE_HASH = "5ef063ed53d00b70f4d8b16aec55bf8353f00afb529db517b8bd540cb5bdee29"
ENGINEERING_SEED = 314159
ENGINEERING_OUTPUT_ROOT = REPOSITORY_ROOT / "results" / "pilot" / "engineering_diagnostics"
ENGINEERING_OUTPUT = (
    ENGINEERING_OUTPUT_ROOT / "ms_bn_root_cause_from_14e5d24_seed314159.json"
)
CANONICAL_PROTOCOL = REPOSITORY_ROOT / "configs" / "protocol_v2r2_seed88_of80_e120.yaml"
CANONICAL_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "v2r2_seed88_of80_e120_generated"
    / "E1_cifar100_d20_t6_C1_s88.yaml"
)
OVERFIT_STEPS = 80
BATCH_SIZE = 8
MINIMUM_ACCURACY = 0.50
MAXIMUM_LOSS_FRACTION = 0.90


class DiagnosticError(RuntimeError):
    """A fail-closed engineering-diagnostic violation."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CANONICAL_CONFIG,
        help="v2r2 C1 reference config used only for model/data construction.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=CANONICAL_PROTOCOL,
        help="v2r2 protocol used only to validate the reference config and environment.",
    )
    parser.add_argument(
        "--failed-health-report",
        required=True,
        type=Path,
        help="Archived seed-88 v2r2 FAIL JSON bound to its known SHA-256.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-gpu-substring", default="RTX 5090")
    parser.add_argument(
        "--engineering-seed",
        required=True,
        type=int,
        help="Fresh engineering-only seed; known pilot/formal seeds are rejected.",
    )
    parser.add_argument(
        "--confirm-nonreporting",
        required=True,
        choices=("NONREPORTING_SINGLE_ATTEMPT",),
        help="Acknowledge that this engineering seed is consumed once and cannot be reported.",
    )
    return parser


def start_receipt_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.attempt.json")


def validate_arguments(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.engineering_seed < 0:
        raise DiagnosticError("--engineering-seed must be non-negative")
    if args.engineering_seed in FORBIDDEN_EXPERIMENT_SEEDS:
        raise DiagnosticError(
            f"Seed {args.engineering_seed} is reserved or already consumed; "
            "choose a fresh engineering seed"
        )
    if args.engineering_seed != ENGINEERING_SEED:
        raise DiagnosticError(
            f"This one-attempt diagnostic is bound to engineering seed {ENGINEERING_SEED}"
        )
    if not args.device or args.device == "auto":
        raise DiagnosticError("An explicit CUDA device is required")
    if args.protocol.resolve() != CANONICAL_PROTOCOL.resolve():
        raise DiagnosticError(f"Diagnostic protocol is pinned to {CANONICAL_PROTOCOL.resolve()}")
    if args.config.resolve() != CANONICAL_CONFIG.resolve():
        raise DiagnosticError(f"Diagnostic config is pinned to {CANONICAL_CONFIG.resolve()}")
    output = args.output.resolve()
    receipt = start_receipt_path(output)
    engineering_root = ENGINEERING_OUTPUT_ROOT.resolve()
    if output != ENGINEERING_OUTPUT.resolve():
        raise DiagnosticError(
            f"Diagnostic output is pinned to {ENGINEERING_OUTPUT.resolve()}"
        )
    if output.parent != engineering_root:
        raise DiagnosticError(f"Diagnostic output must be under {engineering_root}")
    formal_roots = (
        REPOSITORY_ROOT / "results" / "runs",
        REPOSITORY_ROOT / "results" / "pilot" / "v2_seed77_e120",
        REPOSITORY_ROOT / "results" / "pilot" / "v2r2_seed88_of80_e120",
    )
    gate.validate_output_path(
        output,
        formal_roots=formal_roots,
        repository_root=REPOSITORY_ROOT,
    )
    if receipt.exists():
        raise FileExistsError(
            f"Diagnostic seed was already claimed by start receipt: {receipt}"
        )
    return output, receipt


def claim_attempt_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably claim the one allowed attempt without a check-then-write race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_failed_health_report(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Archived seed-88 health report is missing: {resolved}")
    observed_hash = sha256_file(resolved)
    if observed_hash != FAILED_HEALTH_SHA256:
        raise DiagnosticError(
            "Archived seed-88 health report SHA-256 mismatch: "
            f"expected {FAILED_HEALTH_SHA256}, got {observed_hash}"
        )
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"Cannot parse archived seed-88 health report: {exc}") from exc
    conditions = report.get("functional_health_by_condition")
    timing = report.get("cuda_train_step_timing")
    thresholds = report.get("thresholds")
    valid_pattern = bool(
        report.get("status") == "FAIL"
        and report.get("pass") is False
        and report.get("fatal_error") is None
        and report.get("seed") == FAILED_HEALTH_SEED
        and report.get("git_commit") == FAILED_HEALTH_COMMIT
        and report.get("protocol_hash") == FAILED_PROTOCOL_HASH
        and report.get("acceptance_hash") == FAILED_ACCEPTANCE_HASH
        and isinstance(conditions, Mapping)
        and set(conditions) == set(CONDITIONS_IN_ORDER)
        and conditions["C1"].get("pass") is True
        and conditions["C2"].get("pass") is True
        and conditions["C3"].get("pass") is False
        and conditions["C4"].get("pass") is False
        and isinstance(timing, Mapping)
        and timing.get("pass") is True
        and isinstance(thresholds, Mapping)
        and thresholds.get("overfit_steps") == OVERFIT_STEPS
        and thresholds.get("minimum_overfit_accuracy") == MINIMUM_ACCURACY
        and thresholds.get("maximum_overfit_loss_fraction") == MAXIMUM_LOSS_FRACTION
        and thresholds.get("minimum_ta_routed_gradient_coverage") == 0.90
        and thresholds.get("maximum_ta_enabled_over_frozen_step_ratio") == 3.0
    )
    if not valid_pattern:
        raise DiagnosticError(
            "Archived health report does not encode the bound C1/C2 PASS, C3/C4 FAIL pattern"
        )
    return report


def validate_reference_identity(
    protocol: Mapping[str, Any],
    config: RunConfig,
) -> dict[str, Any]:
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise DiagnosticError("Canonical v2r2 protocol has no pilot_acceptance mapping")
    protocol_hash = stable_hash(protocol)
    acceptance_hash = stable_hash(dict(acceptance))
    checks = {
        "protocol_hash": protocol_hash == FAILED_PROTOCOL_HASH,
        "acceptance_hash": acceptance_hash == FAILED_ACCEPTANCE_HASH,
        "acceptance_identity": acceptance.get("identity") == "v2r2_seed88_of80_e120",
        "acceptance_seed": acceptance.get("seed") == FAILED_HEALTH_SEED,
        "acceptance_dataset": acceptance.get("dataset") == "cifar100",
        "acceptance_conditions": acceptance.get("conditions") == list(CONDITIONS_IN_ORDER),
        "config_seed": config.runtime.seed == FAILED_HEALTH_SEED,
        "config_condition": config.model.condition == "C1",
        "config_dataset": config.data.dataset == "cifar100",
        "config_topology": config.model.topology == "spiking_resnet",
        "config_neuron": config.model.neuron == "lif",
        "config_time_steps": config.model.time_steps == 6,
        "config_protocol_hash": config.analysis.get("protocol_hash") == FAILED_PROTOCOL_HASH,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DiagnosticError("Canonical v2r2 reference identity mismatch at: " + ", ".join(failed))
    return {
        "identity": acceptance["identity"],
        "protocol_hash": protocol_hash,
        "acceptance_hash": acceptance_hash,
        "config_run_id": config.runtime.run_id,
        "checks": checks,
    }


def validate_failed_source_binding(
    failed_report: Mapping[str, Any],
    *,
    protocol_path: Path,
    config_path: Path,
    split_manifest: Mapping[str, Any],
) -> dict[str, str]:
    source = failed_report.get("source")
    if not isinstance(source, Mapping):
        raise DiagnosticError("Archived seed-88 health report has no source binding")
    expected = {
        "protocol_sha256": sha256_file(protocol_path),
        "config_sha256": sha256_file(config_path),
        "split_manifest_sha256": str(split_manifest.get("manifest_sha256", "")),
    }
    mismatches = [key for key, value in expected.items() if source.get(key) != value]
    if mismatches:
        raise DiagnosticError(
            "Current protocol/config/data split differs from archived seed-88 health source at: "
            + ", ".join(mismatches)
        )
    if source.get("dataset") != "cifar100":
        raise DiagnosticError("Archived seed-88 health source is not CIFAR-100")
    return expected


def _git_blob_sha256(commit: str, repository_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{repository_path}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise DiagnosticError(f"Cannot read {repository_path} from failed commit: {error}")
    return hashlib.sha256(result.stdout).hexdigest()


def validate_diagnostic_script_identity() -> dict[str, str]:
    repository_path = "scripts/diagnose_msresnet_batchnorm.py"
    script_path = REPOSITORY_ROOT / repository_path
    commands = {
        "tracked_path": ["git", "ls-files", "--error-unmatch", repository_path],
        "head_blob": ["git", "rev-parse", f"HEAD:{repository_path}"],
        "worktree_blob": [
            "git",
            "hash-object",
            f"--path={repository_path}",
            str(script_path),
        ],
    }
    values: dict[str, str] = {}
    for name, command in commands.items():
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise DiagnosticError(
                "Diagnostic script must be committed and identical to current HEAD: "
                + result.stderr.strip()
            )
        values[name] = result.stdout.strip()
    if values["head_blob"] != values["worktree_blob"]:
        raise DiagnosticError("Diagnostic script worktree bytes differ from current HEAD")
    return {
        "path": repository_path,
        "git_blob": values["head_blob"],
        "sha256": sha256_file(script_path),
    }


def validate_failed_runtime_files_unchanged(
    *,
    protocol_path: Path,
    config_path: Path,
) -> dict[str, str]:
    paths = [
        "src/talif_msresnet/config.py",
        "src/talif_msresnet/data.py",
        "src/talif_msresnet/models.py",
        "src/talif_msresnet/neurons.py",
        "src/talif_msresnet/train.py",
        "src/talif_msresnet/utils.py",
        "scripts/pilot_health_gate.py",
    ]
    for supplied in (protocol_path.resolve(), config_path.resolve()):
        try:
            paths.append(supplied.relative_to(REPOSITORY_ROOT.resolve()).as_posix())
        except ValueError as exc:
            raise DiagnosticError(f"Bound input is outside the repository: {supplied}") from exc
    result: dict[str, str] = {}
    for repository_path in sorted(set(paths)):
        current_path = REPOSITORY_ROOT / repository_path
        if not current_path.is_file():
            raise FileNotFoundError(f"Bound runtime file is missing: {current_path}")
        failed_hash = _git_blob_sha256(FAILED_HEALTH_COMMIT, repository_path)
        current_hash = sha256_file(current_path)
        if current_hash != failed_hash:
            raise DiagnosticError(
                f"Runtime file differs from failed seed-88 commit: {repository_path}"
            )
        result[repository_path] = current_hash
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", FAILED_HEALTH_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise DiagnosticError("Current HEAD is not descended from the failed seed-88 commit")
    return result


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _batchnorm_modules(model: nn.Module) -> dict[str, nn.BatchNorm2d]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
    }


@contextmanager
def preserved_evaluation_mode(model: nn.Module, mode: str) -> Iterator[None]:
    """Temporarily select an evaluation mode and restore all BN state exactly."""

    if mode not in {"eval_running_stats", "train_batch_stats", "bn_only_batch_stats"}:
        raise ValueError(f"Unsupported evaluation mode: {mode}")
    modules = list(model.modules())
    training_flags = [module.training for module in modules]
    batchnorm = _batchnorm_modules(model)
    bn_state = {
        name: {
            "running_mean": module.running_mean.detach().clone()
            if module.running_mean is not None
            else None,
            "running_var": module.running_var.detach().clone()
            if module.running_var is not None
            else None,
            "num_batches_tracked": module.num_batches_tracked.detach().clone()
            if module.num_batches_tracked is not None
            else None,
        }
        for name, module in batchnorm.items()
    }
    try:
        if mode == "train_batch_stats":
            model.train()
        else:
            model.eval()
            if mode == "bn_only_batch_stats":
                for module in batchnorm.values():
                    module.train()
        yield
    finally:
        with torch.no_grad():
            for name, module in batchnorm.items():
                saved = bn_state[name]
                for field in ("running_mean", "running_var", "num_batches_tracked"):
                    target = getattr(module, field)
                    source = saved[field]
                    if target is not None and source is not None:
                        target.copy_(source)
        for module, training in zip(modules, training_flags):
            module.training = training


def _bn_input_observer(
    model: nn.Module,
) -> tuple[dict[str, list[dict[str, Any]]], list[Any]]:
    rows = {name: [] for name in _batchnorm_modules(model)}
    handles: list[Any] = []
    for module_name, module in _batchnorm_modules(model).items():

        def hook(
            current: nn.BatchNorm2d,
            inputs: tuple[Any, ...],
            *,
            name: str = module_name,
        ) -> None:
            value = inputs[0]
            if not isinstance(value, torch.Tensor) or value.ndim != 4:
                raise DiagnosticError(f"{name}: BatchNorm input is not a 4-D tensor")
            observed = value.detach().float()
            reduce_dims = (0, 2, 3)
            batch_mean = observed.mean(dim=reduce_dims)
            batch_variance = observed.var(dim=reduce_dims, unbiased=False)
            running_mean = current.running_mean
            running_variance = current.running_var
            if running_mean is None or running_variance is None:
                raise DiagnosticError(f"{name}: BatchNorm running statistics are unavailable")
            running_mean_f = running_mean.detach().float()
            running_variance_f = running_variance.detach().float()
            epsilon = float(current.eps)
            mean_rmse = torch.sqrt(torch.mean((batch_mean - running_mean_f) ** 2))
            variance_log_ratio = torch.log(
                (batch_variance + epsilon) / (running_variance_f + epsilon)
            )
            rows[name].append(
                {
                    "time_index_zero_based": len(rows[name]),
                    "batch_mean": batch_mean.cpu().tolist(),
                    "batch_variance": batch_variance.cpu().tolist(),
                    "running_mean": running_mean_f.cpu().tolist(),
                    "running_variance": running_variance_f.cpu().tolist(),
                    "mean_rmse": float(mean_rmse.item()),
                    "mean_absolute_log_variance_ratio": float(
                        variance_log_ratio.abs().mean().item()
                    ),
                }
            )

        handles.append(module.register_forward_pre_hook(hook))
    return rows, handles


@torch.no_grad()
def evaluate_fixed_batch(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
    *,
    mode: str,
    capture_batchnorm: bool = False,
) -> tuple[dict[str, Any], torch.Tensor]:
    inputs, targets = batch
    rows: dict[str, list[dict[str, Any]]] = {}
    handles: list[Any] = []
    with preserved_evaluation_mode(model, mode):
        if capture_batchnorm:
            rows, handles = _bn_input_observer(model)
        try:
            logits, _diagnostics = _forward(model, inputs, collect_activity=False)
        finally:
            for handle in handles:
                handle.remove()
    loss = float(F.cross_entropy(logits, targets).item())
    accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise DiagnosticError(f"{mode}: fixed-batch evaluation produced non-finite metrics")
    result: dict[str, Any] = {
        "mode": mode,
        "loss": loss,
        "accuracy": accuracy,
        "logits_sha256": _tensor_sha256(logits),
    }
    if capture_batchnorm:
        result["batchnorm_inputs_by_layer_and_timestep"] = rows
    return result, logits.detach().cpu().clone()


def _with_threshold_decision(metrics: Mapping[str, Any], initial_loss: float) -> dict[str, Any]:
    loss = float(metrics["loss"])
    accuracy = float(metrics["accuracy"])
    loss_fraction = loss / initial_loss if initial_loss > 0 else math.inf
    return {
        **dict(metrics),
        "loss_fraction_from_initial_eval": loss_fraction,
        "passes_existing_overfit_thresholds": bool(
            math.isfinite(loss_fraction)
            and loss_fraction <= MAXIMUM_LOSS_FRACTION
            and accuracy >= MINIMUM_ACCURACY
        ),
    }


def root_cause_decision(
    condition_reports: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if set(condition_reports) != set(CONDITIONS_IN_ORDER):
        raise DiagnosticError("Root-cause decision requires exactly C1--C4")
    del batch_size
    checks: dict[str, bool] = {}
    for condition in ("C1", "C2"):
        standard = condition_reports[condition]["ordinary_eval"]
        batch_stats = condition_reports[condition]["bn_only_batch_stats"]
        checks[f"{condition}_ordinary_eval_reproduces_pass"] = bool(
            standard["passes_existing_overfit_thresholds"]
        )
        checks[f"{condition}_batch_stats_remains_pass"] = bool(
            batch_stats["passes_existing_overfit_thresholds"]
        )
    for condition in ("C3", "C4"):
        standard = condition_reports[condition]["ordinary_eval"]
        batch_stats = condition_reports[condition]["bn_only_batch_stats"]
        checks[f"{condition}_ordinary_eval_reproduces_failure"] = not bool(
            standard["passes_existing_overfit_thresholds"]
        )
        checks[f"{condition}_batch_stats_restores_pass"] = bool(
            batch_stats["passes_existing_overfit_thresholds"]
        )
    for condition in CONDITIONS_IN_ORDER:
        checks[f"{condition}_train_and_bn_only_logits_bitwise_equal"] = bool(
            condition_reports[condition]["train_and_bn_only_logits_bitwise_equal"]
        )
        checks[f"{condition}_counterfactual_state_unchanged"] = bool(
            condition_reports[condition]["counterfactual_state_unchanged"]
        )
        checks[f"{condition}_bn_called_once_per_configured_timestep"] = bool(
            condition_reports[condition]["bn_called_once_per_configured_timestep"]
        )
    reproduction_names = [
        "C1_ordinary_eval_reproduces_pass",
        "C2_ordinary_eval_reproduces_pass",
        "C3_ordinary_eval_reproduces_failure",
        "C4_ordinary_eval_reproduces_failure",
    ]
    recovery_names = [
        "C1_batch_stats_remains_pass",
        "C2_batch_stats_remains_pass",
        "C3_batch_stats_restores_pass",
        "C4_batch_stats_restores_pass",
    ]
    integrity_names = [
        name
        for name in checks
        if name.endswith("_logits_bitwise_equal")
        or name.endswith("_state_unchanged")
        or name.endswith("_configured_timestep")
    ]
    reproduced = all(checks[name] for name in reproduction_names)
    recovered = all(checks[name] for name in recovery_names)
    integrity_pass = all(checks[name] for name in integrity_names)
    supported = bool(reproduced and recovered and integrity_pass)
    if supported:
        status = "BN_BATCH_STATS_SUFFICIENT_FOR_FIXED_BATCH_DIVERGENCE"
        next_action = (
            "The counterfactual supports evaluating one fair temporal-BatchNorm candidate "
            "across C1--C4 under a new identity"
        )
    elif reproduced and integrity_pass:
        status = "BN_BATCH_STATS_NOT_SUFFICIENT"
        next_action = (
            "Do not implement temporal BatchNorm on this evidence; stop this diagnostic without "
            "retry, while recognizing that the distinct temporal-BN intervention was not tested"
        )
    else:
        status = "INCONCLUSIVE"
        next_action = "Stop without retry because the bound failure pattern was not reproduced cleanly"
    return {
        "status": status,
        "pass": supported,
        "hypothesis": (
            "Using BatchNorm running statistics is sufficient to produce the observed fixed-batch "
            "MS-ResNet train/eval divergence"
        ),
        "inference_scope": "fixed_batch_nonreporting_engineering_health_only",
        "interpretation_limit": (
            "Support means BatchNorm statistics are sufficient to explain this fixed-batch "
            "health failure; it does not establish that temporal BatchNorm improves validation "
            "or test generalization"
        ),
        "temporal_batchnorm_status": (
            "CANDIDATE_NOT_TESTED; per-timestep BatchNorm input statistics are descriptive only"
        ),
        "failure_pattern_reproduced": reproduced,
        "counterfactual_integrity_pass": integrity_pass,
        "batchnorm_counterfactual_recovers_all_conditions": recovered,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "next_action": next_action,
    }


def run_condition_diagnostic(
    base: RunConfig,
    *,
    condition: str,
    seed: int,
    device: torch.device,
    output_parent: Path,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    gate.seed_everything(seed, deterministic=True)
    config = gate.diagnostic_config(
        base,
        condition=condition,
        seed=seed,
        device=device,
        output_parent=output_parent,
    )
    model, optimizer, _scheduler, _ta_parameters = gate._build_training_objects(
        config,
        device,
        ta_enabled=config.model.neuron == "ta_lif",
    )
    initial, _initial_logits = evaluate_fixed_batch(
        model,
        batch,
        mode="eval_running_stats",
    )
    train_losses: list[float] = []
    for step in range(OVERFIT_STEPS):
        metrics = gate._one_step(model, optimizer, config, batch, device, step)
        train_losses.append(float(metrics["loss"]))

    state_before_evaluations = model_state_sha256(model)
    ordinary, _ordinary_logits = evaluate_fixed_batch(
        model,
        batch,
        mode="eval_running_stats",
        capture_batchnorm=True,
    )
    train_mode, train_logits = evaluate_fixed_batch(
        model,
        batch,
        mode="train_batch_stats",
    )
    bn_only, bn_only_logits = evaluate_fixed_batch(
        model,
        batch,
        mode="bn_only_batch_stats",
    )
    state_after_evaluations = model_state_sha256(model)
    initial_loss = float(initial["loss"])
    observations = ordinary["batchnorm_inputs_by_layer_and_timestep"]
    expected_timesteps = int(config.model.time_steps)
    bn_call_counts = {name: len(rows) for name, rows in observations.items()}
    return {
        "condition": condition,
        "topology": config.model.topology,
        "neuron": config.model.neuron,
        "initial_eval": initial,
        "train_loss_history": train_losses,
        "ordinary_eval": _with_threshold_decision(ordinary, initial_loss),
        "train_batch_stats": _with_threshold_decision(train_mode, initial_loss),
        "bn_only_batch_stats": _with_threshold_decision(bn_only, initial_loss),
        "train_and_bn_only_logits_bitwise_equal": bool(
            torch.equal(train_logits, bn_only_logits)
        ),
        "state_dict_sha256_before_counterfactuals": state_before_evaluations,
        "state_dict_sha256_after_counterfactuals": state_after_evaluations,
        "counterfactual_state_unchanged": bool(
            state_before_evaluations == state_after_evaluations
        ),
        "batchnorm_call_counts": bn_call_counts,
        "expected_calls_per_batchnorm": expected_timesteps,
        "bn_called_once_per_configured_timestep": bool(
            bn_call_counts
            and all(count == expected_timesteps for count in bn_call_counts.values())
        ),
    }


def run_diagnostic(
    args: argparse.Namespace,
    base: RunConfig,
    protocol: Mapping[str, Any],
    output: Path,
    failed_health: Mapping[str, Any],
    failed_health_report: Path,
    runtime_file_hashes: Mapping[str, str],
    reference_identity: Mapping[str, Any],
    diagnostic_script_identity: Mapping[str, str],
) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.perf_counter()
    git_identity = gate.repository_git_identity(REPOSITORY_ROOT)
    if not git_identity["tracked_clean"]:
        raise DiagnosticError("Diagnostic evidence requires a clean tracked worktree")
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise DiagnosticError("Reference protocol has no pilot_acceptance mapping")
    environment_binding = acceptance.get("environment")
    if not isinstance(environment_binding, Mapping):
        raise DiagnosticError("Reference protocol has no environment binding")
    device = torch.device(args.device)
    hardware = gate.require_target_cuda(device, args.expected_gpu_substring)
    gate.validate_runtime_environment(environment_binding, hardware)
    gate.seed_everything(args.engineering_seed, deterministic=True)
    resolved, inputs, targets, split_manifest = gate.load_fixed_real_batch(
        base,
        seed=args.engineering_seed,
        required_batch_size=BATCH_SIZE,
    )
    failed_source_binding = validate_failed_source_binding(
        failed_health,
        protocol_path=args.protocol,
        config_path=args.config,
        split_manifest=split_manifest,
    )
    batch = (
        inputs[:BATCH_SIZE].to(device, non_blocking=False),
        targets[:BATCH_SIZE].to(device, non_blocking=False),
    )
    gpu_idle = gate.gpu_idle_precheck(device)
    condition_reports: dict[str, Any] = {}
    for condition in CONDITIONS_IN_ORDER:
        condition_reports[condition] = run_condition_diagnostic(
            resolved,
            condition=condition,
            seed=args.engineering_seed,
            device=device,
            output_parent=output.parent,
            batch=batch,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    decision = root_cause_decision(condition_reports, batch_size=BATCH_SIZE)
    report = {
        "schema": "ta-lif-msresnet-ms-bn-root-cause-diagnostic-v1",
        "artifact_class": ARTIFACT_CLASS,
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "interpretation_scope": "FIXED_BATCH_ENGINEERING_DIAGNOSTIC_ONLY",
        "status": decision["status"],
        "pass": decision["pass"],
        "started_at": started_at,
        "completed_at": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "seed": args.engineering_seed,
        "seed_disposition": "CONSUMED_ENGINEERING_ONLY_NEVER_REUSE",
        "failed_seed": FAILED_HEALTH_SEED,
        "failed_seed_training_steps": 0,
        "failed_health_report": str(failed_health_report.resolve()),
        "failed_health_report_sha256": FAILED_HEALTH_SHA256,
        "failed_health_commit": FAILED_HEALTH_COMMIT,
        "failed_health_source_binding": failed_source_binding,
        "runtime_file_hashes_matching_failed_commit": dict(runtime_file_hashes),
        "reference_identity": dict(reference_identity),
        "diagnostic_script_identity": dict(diagnostic_script_identity),
        "git_commit": git_identity["git_commit"],
        "tracked_clean": git_identity["tracked_clean"],
        "protocol": str(args.protocol.resolve()),
        "protocol_sha256": sha256_file(args.protocol),
        "protocol_hash": stable_hash(protocol),
        "reference_config": str(args.config.resolve()),
        "reference_config_sha256": sha256_file(args.config),
        "source": {
            "dataset": resolved.data.dataset,
            "batch_size": BATCH_SIZE,
            "overfit_steps": OVERFIT_STEPS,
            "fixed_batch_sha256": gate.tensor_batch_sha256(inputs[:BATCH_SIZE], targets[:BATCH_SIZE]),
            "fixed_batch_shape": list(inputs[:BATCH_SIZE].shape),
            "split_manifest_sha256": split_manifest.get("manifest_sha256"),
        },
        "thresholds_reused_for_engineering_decision": {
            "minimum_accuracy": MINIMUM_ACCURACY,
            "maximum_loss_fraction": MAXIMUM_LOSS_FRACTION,
        },
        "environment": {
            **environment_manifest(),
            "hardware": hardware,
            "precision": "float32",
            "amp": False,
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu_idle_precheck": gpu_idle,
        },
        "conditions": condition_reports,
        "root_cause_decision": decision,
    }
    report["report_content_sha256"] = stable_hash(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output: Path | None = None
    receipt: Path | None = None
    try:
        output, receipt = validate_arguments(args)
        protocol = load_protocol(args.protocol)
        config = load_run_config(args.config, args.protocol)
        reference_identity = validate_reference_identity(protocol, config)
        failed_health = validate_failed_health_report(args.failed_health_report)
        runtime_file_hashes = validate_failed_runtime_files_unchanged(
            protocol_path=args.protocol,
            config_path=args.config,
        )
        diagnostic_script_identity = validate_diagnostic_script_identity()
        output.parent.mkdir(parents=True, exist_ok=True)
        claim_attempt_receipt(
            receipt,
            {
                "schema": "ta-lif-msresnet-ms-bn-root-cause-diagnostic-attempt-v1",
                "artifact_class": ARTIFACT_CLASS,
                "status": "ATTEMPT_CLAIMED_SEED_CONSUMED_NO_RETRY",
                "started_at": utc_now(),
                "seed": args.engineering_seed,
                "output": str(output),
                "failed_health_report_sha256": FAILED_HEALTH_SHA256,
                "git_commit": gate.repository_git_identity(REPOSITORY_ROOT)["git_commit"],
                "diagnostic_script_identity": diagnostic_script_identity,
            },
        )
        report = run_diagnostic(
            args,
            config,
            protocol,
            output,
            failed_health,
            args.failed_health_report,
            runtime_file_hashes,
            reference_identity,
            diagnostic_script_identity,
        )
        atomic_write_json(output, report)
        print(report["status"])
        print(f"NON_REPORTING_OUTPUT={output}")
        print(f"ENGINEERING_SEED_{args.engineering_seed}_CONSUMED_DO_NOT_RETRY")
        return 0 if report["pass"] else 1
    except Exception as exc:
        if output is not None and receipt is not None and receipt.exists() and not output.exists():
            atomic_write_json(
                output,
                {
                    "schema": "ta-lif-msresnet-ms-bn-root-cause-diagnostic-v1",
                    "artifact_class": ARTIFACT_CLASS,
                    "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
                    "confirmatory_analysis_eligibility": False,
                    "status": "ERROR",
                    "pass": False,
                    "seed": args.engineering_seed,
                    "seed_disposition": "CONSUMED_ENGINEERING_ONLY_NEVER_REUSE",
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "completed_at": utc_now(),
                },
            )
        print(f"DIAGNOSTIC_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
