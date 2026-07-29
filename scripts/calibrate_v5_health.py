#!/usr/bin/env python3
"""Collect development-only GPU evidence for a prospective v5 health gate."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)
SCRIPTS_PATH = str(SCRIPTS_ROOT)
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(1, SCRIPTS_PATH)

import pilot_health_gate as legacy_gate
import pilot_health_gate_v3 as v3_gate

from talif_msresnet.config import RunConfig, load_protocol, load_run_config
from talif_msresnet.data import build_loaders
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.pilot_v4 import (
    exclusive_create_json,
    repository_git_identity,
)
from talif_msresnet.train import (
    _ta_enabled_for_epoch,
    evaluate,
    train_one_epoch,
)
from talif_msresnet.utils import (
    environment_manifest,
    sha256_file,
    stable_hash,
    utc_now,
)

PLAN_ARTIFACT_CLASS = "DEVELOPMENT_ONLY_V5_HEALTH_DESIGN_PROBE"
REPORT_ARTIFACT_CLASS = "DEVELOPMENT_ONLY_V5_HEALTH_DESIGN_PROBE_RESULT"
REPORTING_ELIGIBILITY = "FORBIDDEN_FROM_MANUSCRIPT_RESULTS"
REQUIRED_PRE_V5_SEED_EXCLUSIONS = frozenset(
    {
        11,
        22,
        33,
        44,
        55,
        77,
        88,
        314159,
        474123945,
        799312121,
        1882214332,
        836017246,
        126468528,
        2075015328,
        419442269,
        1033863572,
        1367073951,
        1975342236,
        1983855948,
        375760402,
        595643067,
        671880744,
        1045910319,
        615268440,
        1230259817,
        1487387499,
        856172396,
        1846577336,
        1248366457,
        874085245,
        2019339204,
    }
)
TOP_LEVEL_KEYS = {
    "schema_version",
    "plan_id",
    "prospective_protocol_version",
    "plan_status",
    "artifact_class",
    "reporting_eligibility",
    "confirmatory_analysis_eligibility",
    "decision_scope",
    "design_basis",
    "reference",
    "environment",
    "conditions",
    "output_root",
    "seed_derivation",
    "fixed_batch_probe",
    "formal_schedule_probe",
    "datasets",
}


class CalibrationError(RuntimeError):
    """Raised before a development probe can consume GPU time."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "v5_health_calibration.yaml",
    )
    parser.add_argument("--dataset", required=True, choices=("cifar100", "cifar10dvs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        help="Fresh JSON path under the plan's development output root.",
    )
    return parser


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationError(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        unknown = sorted(observed - expected)
        raise CalibrationError(f"{name} keys differ; missing={missing}, unknown={unknown}")


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationError(f"{name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise CalibrationError(f"{name} must be >= {minimum}")
    return int(value)


def _repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _require_within(path: Path, parent: Path, name: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise CalibrationError(f"{name} must remain under {parent}") from exc


def _seed_digest(namespace: str, label: str, counter: int) -> tuple[str, int]:
    payload = f"{namespace}|{label}|{counter}".encode()
    digest = hashlib.sha256(payload).digest()
    return digest.hex(), int.from_bytes(digest[:8], "big") & 0x7FFFFFFF


def require_head_bound_sources(paths: Sequence[str | Path]) -> list[str]:
    """Require every new runtime input to be tracked and byte-current at HEAD."""

    relative_paths: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        _require_within(path, REPOSITORY_ROOT, "runtime source")
        relative_paths.append(path.relative_to(REPOSITORY_ROOT).as_posix())

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative_paths],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        raise CalibrationError(
            "development runtime inputs must be committed before GPU execution: "
            f"{tracked.stderr.strip() or tracked.stdout.strip()}"
        )
    current = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *relative_paths],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if current.returncode == 1:
        raise CalibrationError("development runtime inputs differ from the current HEAD")
    if current.returncode != 0:
        raise CalibrationError(
            "cannot compare development runtime inputs with HEAD: "
            f"{current.stderr.strip() or current.stdout.strip()}"
        )
    return relative_paths


def validate_plan(raw: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(raw)
    _exact_keys(plan, TOP_LEVEL_KEYS, "plan")
    if plan["schema_version"] != 1:
        raise CalibrationError("schema_version must be 1")
    if plan["prospective_protocol_version"] != 5:
        raise CalibrationError("prospective_protocol_version must be 5")
    if plan["plan_status"] != "development_only_not_a_frozen_protocol":
        raise CalibrationError("the calibration plan must remain development-only")
    if plan["artifact_class"] != PLAN_ARTIFACT_CLASS:
        raise CalibrationError("artifact_class is not the v5 development-probe class")
    if plan["reporting_eligibility"] != REPORTING_ELIGIBILITY:
        raise CalibrationError("development evidence must be forbidden from manuscript results")
    if plan["confirmatory_analysis_eligibility"] is not False:
        raise CalibrationError("confirmatory_analysis_eligibility must be false")
    design_basis = _mapping(plan["design_basis"], "design_basis")
    _exact_keys(
        design_basis,
        {
            "threshold_source",
            "calibration_role",
            "threshold_tuning_from_calibration",
            "pilot_or_formal_seed_consumption",
            "same_output_overwrite",
        },
        "design_basis",
    )
    if (
        design_basis["threshold_tuning_from_calibration"] != "forbidden"
        or design_basis["pilot_or_formal_seed_consumption"] != "none"
        or design_basis["same_output_overwrite"] != "forbidden"
    ):
        raise CalibrationError("design_basis must preserve development-only isolation")
    if list(plan["conditions"]) != ["C1", "C2"]:
        raise CalibrationError("conditions must be the paired ordered C1/C2 block")

    environment = _mapping(plan["environment"], "environment")
    _exact_keys(
        environment,
        {
            "expected_gpu_substring",
            "pytorch_version",
            "cuda_runtime",
            "precision",
            "deterministic",
        },
        "environment",
    )
    if environment["precision"] != "float32" or environment["deterministic"] is not True:
        raise CalibrationError("the probe requires deterministic float32 execution")

    output_root = _repository_path(str(plan["output_root"]))
    _require_within(
        output_root,
        (REPOSITORY_ROOT / "results" / "development").resolve(),
        "output_root",
    )

    fixed = _mapping(plan["fixed_batch_probe"], "fixed_batch_probe")
    _exact_keys(
        fixed,
        {
            "batch_size",
            "observation_steps",
            "optimizer_regime",
            "c2_ta_state",
            "learning_metrics_role",
            "minimum_ta_routed_gradient_coverage_observation",
        },
        "fixed_batch_probe",
    )
    _positive_int(fixed["batch_size"], "fixed_batch_probe.batch_size")
    steps = [
        _positive_int(value, "observation step", allow_zero=True)
        for value in fixed["observation_steps"]
    ]
    if steps != sorted(set(steps)) or not steps or steps[0] != 0 or steps[-1] == 0:
        raise CalibrationError("observation_steps must be unique, increasing, and start at zero")
    if fixed["optimizer_regime"] != "base_lr_without_warmup_scheduler_steps":
        raise CalibrationError("fixed-batch probe must use base LR without scheduler steps")
    if fixed["learning_metrics_role"] != "descriptive_only_no_pass_fail_threshold":
        raise CalibrationError("fixed-batch learning metrics must remain descriptive")
    minimum_coverage = fixed["minimum_ta_routed_gradient_coverage_observation"]
    if minimum_coverage != 0.0:
        raise CalibrationError("the TA coverage observation must not impose a tuned cutoff")

    formal = _mapping(plan["formal_schedule_probe"], "formal_schedule_probe")
    _exact_keys(
        formal,
        {
            "epochs",
            "train_batches_per_epoch",
            "validation_batches_per_epoch",
            "use_frozen_v4_optimizer_schedule",
            "required_activation_epoch_zero_based",
            "required_pre_activation_epochs",
            "required_post_activation_epochs",
            "learning_metrics_role",
        },
        "formal_schedule_probe",
    )
    epochs = _positive_int(formal["epochs"], "formal_schedule_probe.epochs")
    activation = _positive_int(
        formal["required_activation_epoch_zero_based"],
        "formal_schedule_probe.required_activation_epoch_zero_based",
        allow_zero=True,
    )
    if list(formal["required_pre_activation_epochs"]) != list(range(activation)):
        raise CalibrationError("required_pre_activation_epochs must cover every epoch before TA")
    if list(formal["required_post_activation_epochs"]) != list(range(activation, epochs)):
        raise CalibrationError("required_post_activation_epochs must cover every remaining epoch")
    if formal["use_frozen_v4_optimizer_schedule"] is not True:
        raise CalibrationError("formal-schedule probe must retain the reviewed v4 optimizer")
    if formal["learning_metrics_role"] != "descriptive_only_no_pass_fail_threshold":
        raise CalibrationError("formal-schedule learning metrics must remain descriptive")

    seed_derivation = _mapping(plan["seed_derivation"], "seed_derivation")
    _exact_keys(
        seed_derivation,
        {"namespace", "algorithm", "collision_policy", "excluded_values"},
        "seed_derivation",
    )
    namespace = str(seed_derivation["namespace"])
    if seed_derivation["algorithm"] != "sha256_first_eight_bytes_big_endian_mask_positive_31_bit":
        raise CalibrationError("seed derivation algorithm differs from the recorded rule")
    if (
        seed_derivation["collision_policy"]
        != "reject_zero_excluded_or_previous_then_increment_decimal_counter"
    ):
        raise CalibrationError("seed collision policy differs from the recorded rule")
    excluded_values = [
        _positive_int(value, "seed_derivation.excluded_values")
        for value in seed_derivation["excluded_values"]
    ]
    if len(excluded_values) != len(set(excluded_values)):
        raise CalibrationError("seed_derivation.excluded_values contains duplicates")
    excluded = set(excluded_values)
    missing_exclusions = sorted(REQUIRED_PRE_V5_SEED_EXCLUSIONS - excluded)
    if missing_exclusions:
        raise CalibrationError(
            f"seed_derivation.excluded_values omits pre-v5 values: {missing_exclusions}"
        )
    used: set[int] = set()

    datasets = _mapping(plan["datasets"], "datasets")
    if set(datasets) != {"cifar100", "cifar10dvs"}:
        raise CalibrationError("datasets must define exactly cifar100 and cifar10dvs")
    for dataset in ("cifar100", "cifar10dvs"):
        item = _mapping(datasets[dataset], f"datasets.{dataset}")
        _exact_keys(
            item,
            {
                "reference_config",
                "reference_config_file_sha256",
                "default_output",
                "fixed_batch_seed",
                "formal_schedule_seed",
            },
            f"datasets.{dataset}",
        )
        reference_config = _repository_path(str(item["reference_config"]))
        _require_within(
            reference_config, (REPOSITORY_ROOT / "configs").resolve(), "reference_config"
        )
        default_output = _repository_path(str(item["default_output"]))
        _require_within(default_output, output_root, f"datasets.{dataset}.default_output")
        if default_output.suffix.lower() != ".json":
            raise CalibrationError("default_output must be a JSON file")
        for role in ("fixed_batch_seed", "formal_schedule_seed"):
            record = _mapping(item[role], f"datasets.{dataset}.{role}")
            _exact_keys(record, {"label", "counter", "input_sha256", "value"}, role)
            counter = _positive_int(record["counter"], f"{role}.counter", allow_zero=True)
            digest, derived = _seed_digest(namespace, str(record["label"]), counter)
            value = _positive_int(record["value"], f"{role}.value")
            if record["input_sha256"] != digest or value != derived:
                raise CalibrationError(f"{dataset} {role} does not match deterministic derivation")
            if value in excluded or value in used:
                raise CalibrationError(f"{dataset} {role} overlaps an excluded or prior seed")
            used.add(value)

    reference = _mapping(plan["reference"], "reference")
    _exact_keys(
        reference,
        {
            "protocol",
            "protocol_file_sha256",
            "protocol_hash",
            "source_git_commit",
            "v4_failure_evidence",
            "protected_absent_v4_paths",
        },
        "reference",
    )
    failure = _mapping(reference["v4_failure_evidence"], "v4_failure_evidence")
    _exact_keys(
        failure,
        {
            "dataset",
            "seed",
            "receipt",
            "receipt_sha256",
            "report",
            "report_sha256",
            "expected_receipt_status",
            "expected_report_status",
        },
        "v4_failure_evidence",
    )
    if failure["dataset"] != "cifar100" or failure["seed"] != 1975342236:
        raise CalibrationError("the development plan must remain bound to the observed v4 failure")
    return plan


def load_plan(path: str | Path) -> dict[str, Any]:
    plan_path = Path(path).resolve()
    try:
        raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationError(f"cannot read calibration plan {plan_path}: {exc}") from exc
    return validate_plan(_mapping(raw, "plan"))


def _read_json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read {name} {path}: {exc}") from exc
    return _mapping(raw, name)


def verify_bound_inputs(
    plan: Mapping[str, Any], dataset: str
) -> tuple[Path, Mapping[str, Any], Path, RunConfig, dict[str, Any]]:
    reference = _mapping(plan["reference"], "reference")
    protocol_path = _repository_path(str(reference["protocol"]))
    if (
        not protocol_path.is_file()
        or sha256_file(protocol_path) != reference["protocol_file_sha256"]
    ):
        raise CalibrationError("the reviewed v4 protocol file is missing or changed")
    protocol = load_protocol(protocol_path)
    if protocol.get("protocol_version") != 4 or stable_hash(protocol) != reference["protocol_hash"]:
        raise CalibrationError("the reviewed v4 protocol mapping is missing or changed")

    failure = _mapping(reference["v4_failure_evidence"], "v4_failure_evidence")
    receipt_path = _repository_path(str(failure["receipt"]))
    report_path = _repository_path(str(failure["report"]))
    if not receipt_path.is_file() or sha256_file(receipt_path) != failure["receipt_sha256"]:
        raise CalibrationError("the exact v4 attempt receipt is required and must remain unchanged")
    if not report_path.is_file() or sha256_file(report_path) != failure["report_sha256"]:
        raise CalibrationError("the exact v4 failure report is required and must remain unchanged")
    receipt = _read_json(receipt_path, "v4 attempt receipt")
    report = _read_json(report_path, "v4 failure report")
    if (
        receipt.get("status") != failure["expected_receipt_status"]
        or receipt.get("seed") != failure["seed"]
        or receipt.get("git_commit") != reference["source_git_commit"]
        or receipt.get("protocol_hash") != reference["protocol_hash"]
    ):
        raise CalibrationError("v4 attempt receipt identity differs from the bound failure")
    if (
        report.get("status") != failure["expected_report_status"]
        or report.get("pass") is not False
        or report.get("seed") != failure["seed"]
        or report.get("git_commit") != reference["source_git_commit"]
        or report.get("protocol_hash") != reference["protocol_hash"]
        or report.get("fatal_error") is not None
        or report.get("attempt_receipt_sha256") != failure["receipt_sha256"]
    ):
        raise CalibrationError("v4 report is not the bound nonfatal FAIL artifact")
    for raw_path in reference["protected_absent_v4_paths"]:
        if _repository_path(str(raw_path)).exists():
            raise CalibrationError(
                f"protected v4 CIFAR10-DVS artifact unexpectedly exists: {raw_path}"
            )

    dataset_plan = _mapping(_mapping(plan["datasets"], "datasets")[dataset], dataset)
    config_path = _repository_path(str(dataset_plan["reference_config"]))
    if (
        not config_path.is_file()
        or sha256_file(config_path) != dataset_plan["reference_config_file_sha256"]
    ):
        raise CalibrationError(f"the reviewed {dataset} reference config is missing or changed")
    config = load_run_config(config_path, protocol_path)
    expected_seed = int(protocol["pilot_acceptance"]["datasets"][dataset]["seed"])
    if (
        config.data.dataset != dataset
        or config.model.condition != "C1"
        or config.runtime.seed != expected_seed
        or config.runtime.run_id != config_path.stem
    ):
        raise CalibrationError(
            f"the {dataset} reference config is not the canonical v4 C1 pilot config"
        )
    evidence = {
        "receipt": artifact_path_reference(receipt_path, REPOSITORY_ROOT),
        "receipt_sha256": sha256_file(receipt_path),
        "report": artifact_path_reference(report_path, REPOSITORY_ROOT),
        "report_sha256": sha256_file(report_path),
        "report_status": report["status"],
        "report_fatal_error": report.get("fatal_error"),
    }
    return protocol_path, protocol, config_path, config, evidence


def resolve_output(plan: Mapping[str, Any], dataset: str, requested: Path | None) -> Path:
    dataset_plan = _mapping(_mapping(plan["datasets"], "datasets")[dataset], dataset)
    output = _repository_path(
        requested if requested is not None else str(dataset_plan["default_output"])
    )
    output_root = _repository_path(str(plan["output_root"]))
    _require_within(output, output_root, "output")
    if output.suffix.lower() != ".json":
        raise CalibrationError("output must be a JSON file")
    if output.exists():
        raise CalibrationError("development output already exists; use a fresh attempt path")
    return output


def _require_probe_device(
    seed: int, device_name: str, environment: Mapping[str, Any]
) -> tuple[torch.device, Mapping[str, Any]]:
    """Enable deterministic runtime flags before the CUDA contract checks them."""

    legacy_gate.seed_everything(seed, deterministic=True)
    device = torch.device(device_name)
    hardware = legacy_gate.require_target_cuda(device, str(environment["expected_gpu_substring"]))
    legacy_gate.validate_runtime_environment(environment, hardware)
    return device, hardware


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().double().clone()
        for name, parameter in model.named_parameters()
    }


def _parameter_delta(model: torch.nn.Module, before: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    squared = 0.0
    updated = 0
    total = 0
    for name, parameter in model.named_parameters():
        difference = parameter.detach().cpu().double() - before[name]
        norm = float(difference.norm(2).item())
        squared += norm * norm
        updated += int(norm > 0.0)
        total += 1
    return {
        "l2": math.sqrt(squared),
        "updated_parameter_tensors": updated,
        "parameter_tensor_count": total,
    }


def _named_parameter_snapshot(model: torch.nn.Module, names: set[str]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().double().clone()
        for name, parameter in model.named_parameters()
        if name in names
    }


def _named_parameter_delta(
    model: torch.nn.Module,
    before: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    current = dict(model.named_parameters())
    squared = 0.0
    updated = 0
    for name, original in before.items():
        norm = float((current[name].detach().cpu().double() - original).norm(2).item())
        squared += norm * norm
        updated += int(norm > 0.0)
    return {
        "l2": math.sqrt(squared),
        "updated_parameter_tensors": updated,
        "parameter_tensor_count": len(before),
    }


def _optimizer_groups(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
    raw_specs = getattr(optimizer, "_talif_group_specs", ())
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(optimizer.param_groups):
        spec = raw_specs[index] if index < len(raw_specs) else {}
        rows.append(
            {
                "index": index,
                "role": spec.get("role", f"group_{index}"),
                "lr": float(group["lr"]),
                "weight_decay": float(group["weight_decay"]),
                "parameter_count": len(group["params"]),
            }
        )
    return rows


def _selected_training_metrics(metrics: Mapping[str, Any], step: int) -> dict[str, Any]:
    return {
        "step": step,
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "gradient_mean": float(metrics["gradient_mean"]),
        "residual_gradient_batch_coverage": float(metrics["nonzero_block_gradient_fraction_mean"]),
    }


def run_fixed_batch_condition(
    base: RunConfig,
    *,
    condition: str,
    seed: int,
    device: torch.device,
    output_parent: Path,
    batch: tuple[torch.Tensor, torch.Tensor],
    settings: Mapping[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    legacy_gate.seed_everything(seed, deterministic=True)
    formal_config = legacy_gate.diagnostic_config(
        base,
        condition=condition,
        seed=seed,
        device=device,
        output_parent=output_parent,
    )
    fixed_config = dataclasses.replace(
        formal_config,
        optimizer=dataclasses.replace(formal_config.optimizer, warmup_epochs=0),
    )
    model, optimizer, scheduler, _ta_parameters = legacy_gate._build_training_objects(
        fixed_config,
        device,
        ta_enabled=condition == "C2",
    )
    before = _parameter_snapshot(model)
    observation_steps = [int(value) for value in settings["observation_steps"]]
    initial = legacy_gate._fixed_batch_metrics(model, batch)
    observations: list[dict[str, Any]] = [
        {
            "step": 0,
            **initial,
            "parameter_delta": _parameter_delta(model, before),
            "optimizer_groups": _optimizer_groups(optimizer),
        }
    ]
    print(
        f"V5_FIXED_PROGRESS dataset={base.data.dataset} condition={condition} "
        f"step=0/{observation_steps[-1]}",
        flush=True,
    )
    history: list[dict[str, Any]] = []
    route_union: dict[str, set[int]] = {}
    ta_accumulator = legacy_gate._new_ta_gradient_accumulator(model)
    shared_gradient_report: Mapping[str, Any] | None = None
    for step in range(1, observation_steps[-1] + 1):
        metrics = legacy_gate._step_with_ta_tracking(
            model,
            optimizer,
            fixed_config,
            batch,
            device,
            step - 1,
            route_union,
            ta_accumulator,
        )
        history.append(_selected_training_metrics(metrics, step))
        if step == 1:
            shared_gradient_report = legacy_gate.inspect_shared_gradients(model)
        if step in observation_steps:
            observations.append(
                {
                    "step": step,
                    **legacy_gate._fixed_batch_metrics(model, batch),
                    "parameter_delta": _parameter_delta(model, before),
                    "optimizer_groups": _optimizer_groups(optimizer),
                }
            )
            print(
                f"V5_FIXED_PROGRESS dataset={base.data.dataset} condition={condition} "
                f"step={step}/{observation_steps[-1]}",
                flush=True,
            )
    if shared_gradient_report is None:
        raise CalibrationError("fixed-batch probe did not execute a training step")
    if condition == "C2":
        ta_report = legacy_gate.summarize_ta_gradients(
            route_union,
            ta_accumulator,
            minimum_coverage=float(settings["minimum_ta_routed_gradient_coverage_observation"]),
        )
    else:
        ta_report = {"pass": True, "status": "not_applicable_lif_condition"}
    checkpoint_report = legacy_gate.checkpoint_next_step_check(
        config=formal_config,
        batch=batch,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    anomalies: list[str] = []
    if not shared_gradient_report["pass"]:
        anomalies.append("shared convolution/classifier gradient invariant failed")
    if condition == "C2" and not ta_report["pass"]:
        anomalies.append("TA routed-gradient invariant failed")
    if not checkpoint_report["pass"]:
        anomalies.append("checkpoint exact-next-step invariant failed")
    if float(observations[-1]["parameter_delta"]["l2"]) <= 0.0:
        anomalies.append("no parameter update was observed")
    return {
        "condition": condition,
        "seed": seed,
        "optimizer_regime": settings["optimizer_regime"],
        "warmup_epochs_for_fixed_batch": fixed_config.optimizer.warmup_epochs,
        "scheduler_class_not_stepped": type(scheduler).__name__,
        "learning_metrics_role": settings["learning_metrics_role"],
        "observations": observations,
        "training_history": history,
        "shared_conv_and_fc_gradients": dict(shared_gradient_report),
        "ta_gradients": ta_report,
        "checkpoint_boundary": checkpoint_report,
        "integrity_anomalies": anomalies,
    }


class _CountedIterable:
    def __init__(self, source: Iterable[Any], limit: int) -> None:
        self.source = source
        self.limit = int(limit)
        self.count = 0

    def __iter__(self) -> Iterator[Any]:
        for item in itertools.islice(self.source, self.limit):
            self.count += 1
            yield item


class _CaptureLogger:
    def __init__(self) -> None:
        self.batch_events: list[dict[str, Any]] = []

    def log(self, event: str, **values: Any) -> None:
        if event == "train_batch":
            self.batch_events.append(dict(values))


def _ta_gradient_observation(model: torch.nn.Module, names: set[str]) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    present = 0
    finite = 0
    nonzero = 0
    for name in names:
        gradient = parameters[name].grad
        if gradient is None:
            continue
        present += 1
        is_finite = bool(torch.isfinite(gradient).all().item())
        finite += int(is_finite)
        nonzero += int(is_finite and bool((gradient != 0).any().item()))
    return {
        "parameter_tensor_count": len(names),
        "gradient_present_count": present,
        "finite_gradient_count": finite,
        "nonzero_gradient_count": nonzero,
    }


def _ta_optimizer_state_count(
    optimizer: torch.optim.Optimizer, ta_parameters: Sequence[torch.nn.Parameter]
) -> int:
    return sum(parameter in optimizer.state for parameter in ta_parameters)


def _ta_lr_ratio(groups: Sequence[Mapping[str, Any]]) -> float | None:
    base_lrs = [float(row["lr"]) for row in groups if str(row["role"]).startswith("base")]
    ta_lrs = [float(row["lr"]) for row in groups if row["role"] == "ta"]
    if not base_lrs or len(ta_lrs) != 1 or base_lrs[0] <= 0.0:
        return None
    if any(not math.isclose(value, base_lrs[0], rel_tol=0.0, abs_tol=1e-15) for value in base_lrs):
        return None
    return ta_lrs[0] / base_lrs[0]


def _expected_epoch_lrs(config: RunConfig, epoch: int) -> tuple[float, float]:
    optimizer = config.optimizer
    warmup = (
        min(1.0, float(epoch + 1) / float(optimizer.warmup_epochs))
        if optimizer.warmup_epochs
        else 1.0
    )
    decays = sum(epoch >= milestone for milestone in optimizer.milestones)
    base_lr = float(optimizer.lr * warmup * (optimizer.gamma**decays))
    return base_lr, float(base_lr * optimizer.ta_lr_scale)


def _absolute_lr_schedule_matches(
    groups: Sequence[Mapping[str, Any]], expected_base_lr: float, expected_ta_lr: float
) -> bool:
    for row in groups:
        role = str(row["role"])
        expected = expected_ta_lr if role == "ta" else expected_base_lr
        if not math.isclose(float(row["lr"]), expected, rel_tol=0.0, abs_tol=1e-15):
            return False
    return bool(groups)


def run_formal_schedule_condition(
    base: RunConfig,
    *,
    condition: str,
    seed: int,
    device: torch.device,
    output_parent: Path,
    settings: Mapping[str, Any],
    expected_split_manifest_sha256: str,
) -> dict[str, Any]:
    legacy_gate.seed_everything(seed, deterministic=True)
    config = legacy_gate.diagnostic_config(
        base,
        condition=condition,
        seed=seed,
        device=device,
        output_parent=output_parent,
    )
    config = v3_gate._resolved_data_config(config)
    config = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, limit_batches=None, log_every=1),
        analysis={**config.analysis, "collect_activity": True},
    )
    loaders = build_loaders(config, final_test=False, seed=seed)
    manifest = loaders.get("_manifest")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("manifest_sha256") != expected_split_manifest_sha256
    ):
        raise CalibrationError("formal-schedule probe split differs from the fixed-batch split")

    model, optimizer, scheduler, ta_parameters = legacy_gate._build_training_objects(
        config, device, ta_enabled=False
    )
    activation_epoch = math.ceil(config.optimizer.epochs * config.optimizer.ta_start_fraction)
    if activation_epoch != int(settings["required_activation_epoch_zero_based"]):
        raise CalibrationError("formal TA activation epoch differs from the probe plan")
    epochs = int(settings["epochs"])
    train_limit = int(settings["train_batches_per_epoch"])
    val_limit = int(settings["validation_batches_per_epoch"])
    fixed_validation_batches = list(itertools.islice(loaders["val"], val_limit))
    if len(fixed_validation_batches) != val_limit:
        raise CalibrationError(f"{condition} validation loader has fewer than {val_limit} batches")
    initial_validation = evaluate(model, fixed_validation_batches, device, config, "val")
    ta_names = {
        name
        for name, _parameter in model.named_parameters()
        if name.endswith((".center", ".raw_width"))
    }
    ta_before = _named_parameter_snapshot(model, ta_names)
    logger = _CaptureLogger()
    scaler = legacy_gate._grad_scaler()
    epoch_rows: list[dict[str, Any]] = []

    for epoch in range(epochs):
        sampler = getattr(loaders["train"], "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ta_enabled = _ta_enabled_for_epoch(ta_parameters, epoch, activation_epoch)
        legacy_gate._set_ta_enabled(ta_parameters, ta_enabled)
        groups_before = _optimizer_groups(optimizer)
        expected_base_lr, expected_ta_lr = _expected_epoch_lrs(config, epoch)
        absolute_lr_match = _absolute_lr_schedule_matches(
            groups_before, expected_base_lr, expected_ta_lr
        )
        state_before = _ta_optimizer_state_count(optimizer, ta_parameters)
        logger.batch_events.clear()
        train_batches = _CountedIterable(loaders["train"], train_limit)
        train_metrics = train_one_epoch(
            model,
            train_batches,
            optimizer,
            device,
            epoch,
            config,
            logger,  # type: ignore[arg-type]
            scaler,
        )
        validation_batches = _CountedIterable(fixed_validation_batches, val_limit)
        validation_metrics = evaluate(model, validation_batches, device, config, "val")
        if train_batches.count != train_limit or len(logger.batch_events) != train_limit:
            raise CalibrationError(f"{condition} epoch {epoch} did not execute all train batches")
        if validation_batches.count != val_limit:
            raise CalibrationError(
                f"{condition} epoch {epoch} did not execute all validation batches"
            )
        epoch_rows.append(
            {
                "epoch": epoch,
                "ta_enabled": ta_enabled,
                "ta_requires_grad_count": sum(
                    parameter.requires_grad for parameter in ta_parameters
                ),
                "ta_optimizer_state_count_before": state_before,
                "ta_optimizer_state_count_after": _ta_optimizer_state_count(
                    optimizer, ta_parameters
                ),
                "ta_gradient_observation": _ta_gradient_observation(model, ta_names),
                "ta_parameter_delta": _named_parameter_delta(model, ta_before),
                "optimizer_groups_before_epoch": groups_before,
                "expected_base_lr": expected_base_lr,
                "expected_ta_lr": expected_ta_lr if ta_names else None,
                "absolute_lr_schedule_match": absolute_lr_match,
                "ta_lr_over_base_lr": _ta_lr_ratio(groups_before),
                "train_batches": train_batches.count,
                "validation_batches": validation_batches.count,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "validation_loss": float(validation_metrics["loss"]),
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "gradient_mean": float(train_metrics["gradient_mean"]),
                "residual_gradient_batch_coverage": float(
                    train_metrics["nonzero_block_gradient_fraction_mean"]
                ),
                "surrogate_support_coverage": float(
                    train_metrics.get("diagnostics", {}).get("surrogate_coverage_min", float("nan"))
                ),
            }
        )
        scheduler.step()
        epoch_rows[-1]["optimizer_groups_after_scheduler"] = _optimizer_groups(optimizer)
        print(
            f"V5_SCHEDULE_PROGRESS dataset={base.data.dataset} condition={condition} "
            f"epoch={epoch + 1}/{epochs} ta_enabled={str(ta_enabled).lower()}",
            flush=True,
        )

    anomalies: list[str] = []
    required_states = [condition == "C2" and epoch >= activation_epoch for epoch in range(epochs)]
    observed_states = [bool(row["ta_enabled"]) for row in epoch_rows]
    if observed_states != required_states:
        anomalies.append("TA activation state sequence differs from the formal helper")
    numeric_keys = (
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
        "gradient_mean",
        "residual_gradient_batch_coverage",
        "surrogate_support_coverage",
    )
    if any(not math.isfinite(float(row[key])) for row in epoch_rows for key in numeric_keys):
        anomalies.append("nonfinite formal-schedule observation")
    if any(not bool(row["absolute_lr_schedule_match"]) for row in epoch_rows):
        anomalies.append("absolute optimizer LR differs from the frozen warmup schedule")
    if condition == "C2":
        pre_activation = epoch_rows[:activation_epoch]
        post_activation = epoch_rows[activation_epoch:]
        if any(float(row["ta_parameter_delta"]["l2"]) != 0.0 for row in pre_activation):
            anomalies.append("TA parameters changed before the activation boundary")
        if any(int(row["ta_optimizer_state_count_after"]) != 0 for row in pre_activation):
            anomalies.append("TA optimizer state was created before activation")
        if not post_activation or float(post_activation[-1]["ta_parameter_delta"]["l2"]) <= 0.0:
            anomalies.append("TA parameters did not update after activation")
        first_enabled = post_activation[0]
        gradient = first_enabled["ta_gradient_observation"]
        if (
            int(gradient["gradient_present_count"]) != len(ta_names)
            or int(gradient["finite_gradient_count"]) != len(ta_names)
            or int(gradient["nonzero_gradient_count"]) == 0
        ):
            anomalies.append("TA gradients were missing, nonfinite, or all zero at activation")
        ratio = first_enabled["ta_lr_over_base_lr"]
        if ratio is None or not math.isclose(
            float(ratio), config.optimizer.ta_lr_scale, rel_tol=0.0, abs_tol=1e-12
        ):
            anomalies.append("TA/base learning-rate ratio differs at activation")
    final_validation_loss = float(epoch_rows[-1]["validation_loss"])
    initial_validation_loss = float(initial_validation["loss"])
    return {
        "condition": condition,
        "seed": seed,
        "epochs": epochs,
        "train_batches_per_epoch": train_limit,
        "validation_batches_per_epoch": val_limit,
        "ta_activation_epoch_zero_based": activation_epoch,
        "initial_validation_before_training": {
            "loss": initial_validation_loss,
            "accuracy": float(initial_validation["accuracy"]),
        },
        "final_validation_loss": final_validation_loss,
        "descriptive_validation_loss_change_fraction": (
            (initial_validation_loss - final_validation_loss) / initial_validation_loss
            if initial_validation_loss > 0.0
            else None
        ),
        "learning_metrics_role": settings["learning_metrics_role"],
        "epoch_metrics": epoch_rows,
        "integrity_anomalies": anomalies,
    }


def run_probe(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    dataset: str,
    protocol_path: Path,
    config_path: Path,
    config: RunConfig,
    failure_evidence: Mapping[str, Any],
    output: Path,
    device_name: str,
    git_identity: Mapping[str, Any],
) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.perf_counter()
    dataset_plan = _mapping(_mapping(plan["datasets"], "datasets")[dataset], dataset)
    fixed_seed = int(_mapping(dataset_plan["fixed_batch_seed"], "fixed_batch_seed")["value"])
    formal_seed = int(
        _mapping(dataset_plan["formal_schedule_seed"], "formal_schedule_seed")["value"]
    )
    environment = _mapping(plan["environment"], "environment")
    device, hardware = _require_probe_device(fixed_seed, device_name, environment)
    fixed_settings = _mapping(plan["fixed_batch_probe"], "fixed_batch_probe")
    formal_settings = _mapping(plan["formal_schedule_probe"], "formal_schedule_probe")
    required_batch_size = int(fixed_settings["batch_size"])
    resolved, inputs, targets, split_manifest, source_identity = v3_gate.load_fixed_real_batch(
        config,
        seed=fixed_seed,
        required_batch_size=required_batch_size,
    )
    fixed_batch = (
        inputs[:required_batch_size].to(device, non_blocking=False),
        targets[:required_batch_size].to(device, non_blocking=False),
    )
    fixed_reports: dict[str, Any] = {}
    formal_reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix=f"v5-design-probe-{dataset}-", dir=str(output.parent)
    ) as temporary:
        temporary_root = Path(temporary)
        for condition in plan["conditions"]:
            fixed_reports[str(condition)] = run_fixed_batch_condition(
                resolved,
                condition=str(condition),
                seed=fixed_seed,
                device=device,
                output_parent=output.parent,
                batch=fixed_batch,
                settings=fixed_settings,
                checkpoint_path=temporary_root / str(condition) / "last.pt",
            )
            gc.collect()
            torch.cuda.empty_cache()
            formal_reports[str(condition)] = run_formal_schedule_condition(
                config,
                condition=str(condition),
                seed=formal_seed,
                device=device,
                output_parent=output.parent,
                settings=formal_settings,
                expected_split_manifest_sha256=str(split_manifest["manifest_sha256"]),
            )
            gc.collect()
            torch.cuda.empty_cache()
    anomalies = [
        f"fixed_batch/{condition}: {message}"
        for condition, report in fixed_reports.items()
        for message in report["integrity_anomalies"]
    ]
    anomalies.extend(
        f"formal_schedule/{condition}: {message}"
        for condition, report in formal_reports.items()
        for message in report["integrity_anomalies"]
    )
    status = "COMPLETED_WITH_INTEGRITY_ANOMALIES" if anomalies else "COMPLETED"
    return {
        "schema_version": 1,
        "artifact_class": REPORT_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "prospective_protocol_version": 5,
        "purpose": "development-only GPU integration and activation-boundary evidence",
        "status": status,
        "pass": None,
        "thresholds_evaluated": [],
        "decision": "REVIEW_DESCRIPTIVE_TRAJECTORIES_BEFORE_FREEZING_V5",
        "dataset": dataset,
        "conditions": list(plan["conditions"]),
        "development_seeds": {
            "fixed_batch": fixed_seed,
            "formal_schedule": formal_seed,
            "disposition": "DEVELOPMENT_ONLY_EXCLUDE_FROM_V5_HEALTH_PILOT_AND_FORMAL",
        },
        "plan": artifact_path_reference(plan_path, REPOSITORY_ROOT),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_hash": stable_hash(plan),
        "git_commit": git_identity["git_commit"],
        "tracked_clean": git_identity["tracked_clean"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "v4_failure_evidence": dict(failure_evidence),
        "source": {
            "protocol": artifact_path_reference(protocol_path, REPOSITORY_ROOT),
            "protocol_file_sha256": sha256_file(protocol_path),
            "reference_config": artifact_path_reference(config_path, REPOSITORY_ROOT),
            "reference_config_file_sha256": sha256_file(config_path),
            "reference_config_hash": config.config_hash,
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "fixed_batch_sha256": legacy_gate.tensor_batch_sha256(inputs, targets),
            "fixed_batch_shape": list(inputs.shape),
            "development_entrypoint": artifact_path_reference(Path(__file__), REPOSITORY_ROOT),
            "development_entrypoint_sha256": sha256_file(Path(__file__)),
            "training_source": "src/talif_msresnet/train.py",
            "training_source_sha256": sha256_file(
                REPOSITORY_ROOT / "src" / "talif_msresnet" / "train.py"
            ),
            "v4_termination_record": "V4_TALIF_ONLY_TERMINATION.md",
            "v4_termination_record_sha256": sha256_file(
                REPOSITORY_ROOT / "V4_TALIF_ONLY_TERMINATION.md"
            ),
            **source_identity,
        },
        "environment": {
            **environment_manifest(),
            "hardware": hardware,
            "python_implementation": platform.python_implementation(),
            "precision": "float32",
            "amp": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
        },
        "fixed_batch_probe_by_condition": fixed_reports,
        "formal_schedule_probe_by_condition": formal_reports,
        "integrity_anomalies": anomalies,
    }


def _fatal_report(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    dataset: str,
    git_identity: Mapping[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_class": REPORT_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "prospective_protocol_version": 5,
        "purpose": "development-only GPU integration and activation-boundary evidence",
        "status": "ERROR",
        "pass": None,
        "thresholds_evaluated": [],
        "dataset": dataset,
        "plan": artifact_path_reference(plan_path, REPOSITORY_ROOT),
        "plan_file_sha256": sha256_file(plan_path),
        "plan_hash": stable_hash(plan),
        "git_commit": git_identity.get("git_commit"),
        "tracked_clean": git_identity.get("tracked_clean"),
        "finished_at": utc_now(),
        "fatal_error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "seed_disposition": "DEVELOPMENT_ONLY_RETRY_REQUIRES_A_FRESH_OUTPUT_PATH",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan_path = args.plan.resolve()
        plan = load_plan(plan_path)
        require_head_bound_sources(
            (
                plan_path,
                Path(__file__),
                REPOSITORY_ROOT / "src" / "talif_msresnet" / "train.py",
                REPOSITORY_ROOT / "V4_TALIF_ONLY_TERMINATION.md",
            )
        )
        output = resolve_output(plan, args.dataset, args.output)
        protocol_path, _protocol, config_path, config, failure_evidence = verify_bound_inputs(
            plan, args.dataset
        )
        git_identity = repository_git_identity(REPOSITORY_ROOT)
        if git_identity["tracked_clean"] is not True:
            raise CalibrationError("development evidence requires a clean tracked worktree")
        if not config.runtime.deterministic or config.runtime.amp:
            raise CalibrationError("reference config must use deterministic float32 execution")
        output.parent.mkdir(parents=True, exist_ok=True)
        print(
            "V5_DESIGN_PROBE_PRECHECK_PASS "
            f"dataset={args.dataset} output={artifact_path_reference(output, REPOSITORY_ROOT)}"
        )
        print("DEVELOPMENT_SEED_EXECUTION_BEGINS")
    except Exception as exc:  # noqa: BLE001 - normalize all preflight failures for the CLI.
        print(f"V5_DESIGN_PROBE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        report = run_probe(
            plan=plan,
            plan_path=plan_path,
            dataset=args.dataset,
            protocol_path=protocol_path,
            config_path=config_path,
            config=config,
            failure_evidence=failure_evidence,
            output=output,
            device_name=args.device,
            git_identity=git_identity,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve interrupted probe evidence.
        report = _fatal_report(
            plan=plan,
            plan_path=plan_path,
            dataset=args.dataset,
            git_identity=git_identity,
            exc=exc,
        )
    try:
        exclusive_create_json(output, report)
    except FileExistsError:
        print("V5_DESIGN_PROBE_BLOCKED: output appeared during execution", file=sys.stderr)
        return 2
    print(f"V5_DESIGN_PROBE_{report['status']} dataset={args.dataset}")
    print(f"DEVELOPMENT_ONLY_OUTPUT={output}")
    print("NO_V5_HEALTH_PILOT_OR_FORMAL_SEED_WAS_USED")
    return 0 if report["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
