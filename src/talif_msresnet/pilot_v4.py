"""Fail-closed contracts for the isolated TA-LIF-only v4 pilot.

The v4 repair study deliberately has its own evidence schema.  Reusing the v3
schema would make a v4 PASS ambiguous and would also retain v3's three-author
sign-off assumption.  This module contains no training loop; it validates the
author freeze, one-shot seed receipt, health report, and dataset pilot plan.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    CONDITION_SPECS,
    RunConfig,
    active_conditions_for_protocol,
    artifact_paths_for_protocol,
    confirmation_fields_for_protocol,
    generate_v4_pilot_matrix,
    validate_run_mapping,
)
from .pathing import artifact_path_reference
from .pilot_v3 import repository_git_identity
from .utils import sha256_file, stable_hash, utc_now


V4_PILOT_CONDITIONS = ("C1", "C2")
V4_PILOT_DATASETS = ("cifar100", "cifar10dvs")
HEALTH_SCHEMA_VERSION = 4
HEALTH_ARTIFACT_CLASS = "NON_REPORTING_V4_IMPLEMENTATION_HEALTH_GATE"
ATTEMPT_SCHEMA_VERSION = 1
ATTEMPT_ARTIFACT_CLASS = "NON_REPORTING_V4_HEALTH_ATTEMPT"
ATTEMPT_STATUS = "ATTEMPT_CLAIMED_SEED_CONSUMED_NO_RETRY"
REPORTING_ELIGIBILITY = "FORBIDDEN_FROM_MANUSCRIPT_RESULTS"
SIGNED_STATUS = (
    "Status: **AUTHORIZED - ACCOUNTABLE AUTHOR/USER APPROVAL RECORDED; PROTOCOL FROZEN**"
)
PROHIBITED_UNRECORDED_APPROVERS = ("Peng Yan", "Song Wang")
LONGITUDINAL_RESIDUAL_COVERAGE_DEFINITION = (
    "minimum across epochs of the mean, over train batches, of the fraction "
    "of residual blocks with nonzero parameter-gradient L2 norm"
)
LONGITUDINAL_SURROGATE_COVERAGE_DEFINITION = (
    "minimum across epochs and neuron layers of active rectangular-surrogate "
    "elements divided by observed elements in that layer"
)
LONGITUDINAL_LOSS_REDUCTION_DEFINITION = (
    "relative validation-loss reduction from epoch zero to the final health "
    "epoch on the same fixed first validation batches"
)


class PilotV4Error(RuntimeError):
    """Raised when v4 pilot evidence cannot satisfy its frozen contract."""


@dataclass(frozen=True)
class PilotBlock:
    dataset: str
    seed: int
    conditions: tuple[str, ...]
    health_output: Path
    attempt_receipt: Path
    pilot_output_root: Path
    pilot_plan: Path
    formal_output_root: Path
    protocol_hash: str
    acceptance_hash: str
    block_hash: str


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PilotV4Error(f"{label} must be a mapping")
    return value


def _inside_repository(value: Any, repository_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PilotV4Error(f"{label} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise PilotV4Error(f"{label} must be repository-relative without '..'")
    resolved = (repository_root / path).resolve()
    try:
        resolved.relative_to(repository_root.resolve())
    except ValueError as exc:
        raise PilotV4Error(f"{label} escapes the repository") from exc
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    resolved_root = root.resolve()
    return resolved == resolved_root or resolved_root in resolved.parents


def _responsible_author(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotV4Error("protocol_status.confirmed_by must identify the authorizer")
    normalized = value.strip()
    prohibited = [
        name for name in PROHIBITED_UNRECORDED_APPROVERS if name.casefold() in normalized.casefold()
    ]
    if prohibited:
        raise PilotV4Error(
            "v4 confirmed_by claims unrecorded co-author approval: " + ", ".join(prohibited)
        )
    responsible = normalized.split("(", 1)[0].strip()
    if not responsible:
        raise PilotV4Error("protocol_status.confirmed_by has no author identity")
    return responsible


def require_v4_author_freeze(
    protocol: Mapping[str, Any],
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Require the exact accountable-author authorization before v4 GPU work."""

    if int(protocol.get("protocol_version", 0)) != 4:
        raise PilotV4Error("Author-freeze validation requires protocol_version 4")
    status = _mapping(protocol.get("protocol_status"), "protocol.protocol_status")
    if status.get("frozen") is not True:
        raise PilotV4Error("v4 GPU work requires protocol_status.frozen=true")
    confirmations = _mapping(status.get("confirmations"), "protocol.protocol_status.confirmations")
    missing = [
        field
        for field in confirmation_fields_for_protocol(protocol)
        if confirmations.get(field) is not True
    ]
    if missing:
        raise PilotV4Error("v4 author confirmations are incomplete: " + ", ".join(missing))
    confirmed_by = str(status.get("confirmed_by") or "").strip()
    responsible = _responsible_author(confirmed_by)
    confirmed_at = str(status.get("confirmed_at") or "").strip()
    candidate = f"{confirmed_at[:-1]}+00:00" if confirmed_at.endswith("Z") else confirmed_at
    try:
        timestamp = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise PilotV4Error("protocol_status.confirmed_at is not valid ISO-8601") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise PilotV4Error("protocol_status.confirmed_at must include a UTC offset")

    root = Path(repository_root).resolve()
    signoff = _inside_repository(
        artifact_paths_for_protocol(protocol)["signoff"], root, "v4 sign-off"
    )
    if not signoff.is_file():
        raise PilotV4Error(f"The v4 authorization record is missing: {signoff}")
    text = signoff.read_text(encoding="utf-8")
    if re.findall(r"^Status:.*$", text, flags=re.MULTILINE) != [SIGNED_STATUS]:
        raise PilotV4Error("The v4 authorization record is not in its exact frozen state")
    if re.search(r"\[[^\]\r\n]*PENDING[^\]\r\n]*\]", text, flags=re.IGNORECASE):
        raise PilotV4Error("The v4 authorization record contains a pending placeholder")
    for field in confirmation_fields_for_protocol(protocol):
        if re.search(rf"^- \[[xX]\] `{re.escape(field)}`:", text, flags=re.MULTILINE) is None:
            raise PilotV4Error(f"The v4 authorization checklist is not checked: {field}")
    authorizer = (
        rf"^- Authorizer: {re.escape(responsible)}, accountable author and "
        rf"Codex task user\s*$"
    )
    if re.search(authorizer, text, flags=re.MULTILINE) is None:
        raise PilotV4Error("The v4 authorization record does not identify the authorizer")
    for name in PROHIBITED_UNRECORDED_APPROVERS:
        disclosure = (
            rf"^- Independent approval from {re.escape(name)}: "
            rf"not asserted in this record\s*$"
        )
        if re.search(disclosure, text, flags=re.MULTILINE) is None:
            raise PilotV4Error(f"The v4 record lacks the non-approval disclosure for {name}")
    return {
        "confirmed_by": confirmed_by,
        "responsible_author": responsible,
        "confirmed_at": confirmed_at,
        "signoff": artifact_path_reference(signoff, root),
        "signoff_sha256": sha256_file(signoff),
    }


def resolve_pilot_block(
    protocol: Mapping[str, Any],
    dataset: str,
    *,
    repository_root: str | Path,
) -> PilotBlock:
    """Resolve one exact dataset block from the canonical v4 acceptance map."""

    root = Path(repository_root).resolve()
    if int(protocol.get("protocol_version", 0)) != 4:
        raise PilotV4Error("The v4 pilot layer requires protocol_version 4")
    conditions = tuple(active_conditions_for_protocol(protocol))
    if conditions != V4_PILOT_CONDITIONS:
        raise PilotV4Error("v4 active conditions must be exactly ordered C1,C2")
    acceptance = _mapping(protocol.get("pilot_acceptance"), "protocol.pilot_acceptance")
    if tuple(acceptance.get("conditions", ())) != conditions:
        raise PilotV4Error("pilot_acceptance.conditions must be exactly ordered C1,C2")
    datasets = _mapping(acceptance.get("datasets"), "protocol.pilot_acceptance.datasets")
    if tuple(datasets) != V4_PILOT_DATASETS:
        raise PilotV4Error("pilot_acceptance.datasets must be exactly ordered cifar100,cifar10dvs")
    normalized = str(dataset).lower().replace("-", "")
    if normalized not in V4_PILOT_DATASETS:
        raise PilotV4Error(f"Unsupported v4 pilot dataset: {dataset!r}")
    profile = _mapping(datasets.get(normalized), f"protocol.pilot_acceptance.datasets.{normalized}")
    seed = profile.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed <= 0:
        raise PilotV4Error(f"Invalid v4 pilot seed for {normalized}")
    health_output = _inside_repository(
        profile.get("health_output"), root, f"{normalized}.health_output"
    )
    attempt_receipt = _inside_repository(
        profile.get("attempt_receipt"), root, f"{normalized}.attempt_receipt"
    )
    pilot_output_root = _inside_repository(
        profile.get("pilot_output_root"), root, f"{normalized}.pilot_output_root"
    )
    pilot_plan = _inside_repository(profile.get("pilot_plan"), root, f"{normalized}.pilot_plan")
    formal_output_root = _inside_repository(
        protocol.get("output_root"), root, "protocol.output_root"
    )
    if len({health_output, attempt_receipt, pilot_output_root, pilot_plan}) != 4:
        raise PilotV4Error(f"{normalized} v4 pilot artifact paths must be distinct")
    if _is_within(pilot_output_root, formal_output_root) or _is_within(
        formal_output_root, pilot_output_root
    ):
        raise PilotV4Error("v4 pilot and formal output roots must be disjoint")
    if _is_within(health_output, pilot_output_root) or _is_within(
        attempt_receipt, pilot_output_root
    ):
        raise PilotV4Error("v4 health evidence must be outside the training output root")

    protocol_hash = stable_hash(protocol)
    acceptance_hash = stable_hash(dict(acceptance))
    block_binding = {
        "dataset": normalized,
        "profile": dict(profile),
        "conditions": list(conditions),
        "environment": acceptance.get("environment"),
        "overfit": acceptance.get("overfit"),
        "longitudinal_health": acceptance.get("longitudinal_health"),
        "timing": acceptance.get("timing"),
        "schedule": acceptance.get("schedule"),
    }
    return PilotBlock(
        dataset=normalized,
        seed=seed,
        conditions=conditions,
        health_output=health_output,
        attempt_receipt=attempt_receipt,
        pilot_output_root=pilot_output_root,
        pilot_plan=pilot_plan,
        formal_output_root=formal_output_root,
        protocol_hash=protocol_hash,
        acceptance_hash=acceptance_hash,
        block_hash=stable_hash(block_binding),
    )


def expected_pilot_configs(
    protocol: Mapping[str, Any], dataset: str
) -> tuple[RunConfig, RunConfig]:
    rows = [
        row
        for row in generate_v4_pilot_matrix(protocol)
        if row.get("data", {}).get("dataset") == dataset
    ]
    configs = tuple(validate_run_mapping(row, protocol) for row in rows)
    if len(configs) != 2 or tuple(item.model.condition for item in configs) != (
        "C1",
        "C2",
    ):
        raise PilotV4Error(f"{dataset} does not generate one ordered C1/C2 pilot block")
    return configs  # type: ignore[return-value]


def attempt_receipt_payload(
    *,
    block: PilotBlock,
    protocol_path: str | Path,
    config_path: str | Path,
    config: RunConfig,
    git_identity: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    config_file = Path(config_path).resolve()
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "artifact_class": ATTEMPT_ARTIFACT_CLASS,
        "status": ATTEMPT_STATUS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "dataset": block.dataset,
        "seed": block.seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity.get("git_commit"),
        "tracked_clean": git_identity.get("tracked_clean"),
        "protocol": artifact_path_reference(protocol_file, root),
        "protocol_file_sha256": sha256_file(protocol_file),
        "reference_config": artifact_path_reference(config_file, root),
        "reference_config_sha256": sha256_file(config_file),
        "reference_config_hash": config.config_hash,
        "claimed_at": utc_now(),
        "seed_disposition": "CONSUMED_NONREPORTING_V4_HEALTH_NEVER_RETRY",
    }


def exclusive_create_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Create a durable one-shot claim and never overwrite prior evidence."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return destination


def validate_fixed_batch_shape(
    dataset: str,
    shape: Sequence[Any],
    *,
    minimum_batch_size: int,
    time_steps: int,
    in_channels: int,
) -> list[int]:
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise PilotV4Error("fixed batch shape must be a sequence")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise PilotV4Error("fixed batch shape must contain positive integers")
    values = [int(value) for value in shape]
    if not values or values[0] < minimum_batch_size:
        raise PilotV4Error("fixed batch does not contain the required number of samples")
    if dataset == "cifar100":
        if len(values) != 4 or values[1] != in_channels:
            raise PilotV4Error("CIFAR-100 fixed input must be [B,C,H,W]")
    elif dataset == "cifar10dvs":
        if len(values) != 5 or values[1] != time_steps or values[2] != in_channels:
            raise PilotV4Error("CIFAR10-DVS fixed input must be [B,T,C,H,W]")
    else:
        raise PilotV4Error(f"Unsupported fixed-batch dataset: {dataset}")
    return values


def _finite(value: Any, *, positive: bool = False) -> bool:
    valid = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    return bool(valid and (not positive or float(value) > 0.0))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotV4Error(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotV4Error(f"{label} must be a JSON object")
    return value


def validate_attempt_receipt(
    path: str | Path,
    *,
    block: PilotBlock,
    protocol_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    receipt_path = Path(path).resolve()
    if receipt_path != block.attempt_receipt or not receipt_path.is_file():
        raise PilotV4Error("The canonical dataset-specific attempt receipt is missing")
    receipt = _read_json(receipt_path, "v4 health attempt receipt")
    expected = {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "artifact_class": ATTEMPT_ARTIFACT_CLASS,
        "status": ATTEMPT_STATUS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "dataset": block.dataset,
        "seed": block.seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "protocol_file_sha256": sha256_file(protocol_path),
        "seed_disposition": "CONSUMED_NONREPORTING_V4_HEALTH_NEVER_RETRY",
    }
    failures = [
        f"attempt receipt {key} mismatch"
        for key, expected_value in expected.items()
        if receipt.get(key) != expected_value
    ]
    if receipt.get("tracked_clean") is not True:
        failures.append("attempt receipt was not claimed from a clean tracked tree")
    commit = receipt.get("git_commit")
    if not (
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit)
    ):
        failures.append("attempt receipt git_commit is invalid")
    root = Path(repository_root).resolve()
    if receipt.get("protocol") != artifact_path_reference(protocol_path, root):
        failures.append("attempt receipt protocol path mismatch")
    if failures:
        raise PilotV4Error("V4 health attempt receipt rejected:\n- " + "\n- ".join(failures))
    return receipt


def _validate_longitudinal_condition(
    item: Mapping[str, Any],
    *,
    condition: str,
    acceptance: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []

    def require(result: bool, message: str) -> None:
        if not result:
            failures.append(message)

    epochs = int(acceptance["epochs"])
    train_batches = int(acceptance["train_batches_per_epoch"])
    val_batches = int(acceptance["validation_batches_per_epoch"])
    require(item.get("pass") is True, f"{condition} longitudinal health did not pass")
    require(item.get("epochs") == epochs, f"{condition} longitudinal epoch count mismatch")
    require(
        item.get("train_batches_per_epoch") == train_batches,
        f"{condition} longitudinal train-batch count mismatch",
    )
    require(
        item.get("validation_batches_per_epoch") == val_batches,
        f"{condition} longitudinal validation-batch count mismatch",
    )
    history = item.get("epoch_metrics")
    require(
        isinstance(history, list)
        and len(history) == epochs
        and [row.get("epoch") for row in history if isinstance(row, Mapping)]
        == list(range(epochs)),
        f"{condition} longitudinal history is incomplete",
    )
    coverage = item.get("minimum_epoch_residual_gradient_batch_coverage")
    require(
        item.get("residual_gradient_batch_coverage_definition")
        == LONGITUDINAL_RESIDUAL_COVERAGE_DEFINITION,
        f"{condition} longitudinal residual-gradient definition mismatch",
    )
    require(
        _finite(coverage)
        and float(coverage) >= float(acceptance["minimum_residual_gradient_batch_coverage"]),
        f"{condition} longitudinal residual-gradient coverage failed",
    )
    surrogate = item.get("minimum_epoch_surrogate_support_coverage")
    require(
        item.get("surrogate_support_coverage_definition")
        == LONGITUDINAL_SURROGATE_COVERAGE_DEFINITION,
        f"{condition} longitudinal surrogate-support definition mismatch",
    )
    require(
        _finite(surrogate)
        and float(surrogate) >= float(acceptance["minimum_surrogate_support_coverage"]),
        f"{condition} longitudinal surrogate-support coverage failed",
    )
    reduction = item.get("loss_reduction_fraction")
    initial_validation_loss = item.get("initial_validation_loss")
    final_validation_loss = item.get("final_validation_loss")
    require(
        item.get("loss_reduction_definition") == LONGITUDINAL_LOSS_REDUCTION_DEFINITION,
        f"{condition} longitudinal loss-reduction definition mismatch",
    )
    require(
        _finite(initial_validation_loss, positive=True)
        and _finite(final_validation_loss),
        f"{condition} longitudinal validation-loss endpoints are invalid",
    )
    if _finite(initial_validation_loss, positive=True) and _finite(final_validation_loss):
        expected_reduction = (
            float(initial_validation_loss) - float(final_validation_loss)
        ) / float(initial_validation_loss)
        require(
            _finite(reduction)
            and math.isclose(
                float(reduction), expected_reduction, rel_tol=1e-12, abs_tol=1e-12
            ),
            f"{condition} longitudinal loss reduction is inconsistent",
        )
    require(
        _finite(reduction)
        and float(reduction) >= float(acceptance["minimum_loss_reduction_fraction"]),
        f"{condition} longitudinal loss reduction failed",
    )
    ratio = item.get("parameter_norm_ratio")
    require(
        _finite(ratio, positive=True)
        and float(acceptance["parameter_norm_ratio_minimum"])
        <= float(ratio)
        <= float(acceptance["parameter_norm_ratio_maximum"]),
        f"{condition} longitudinal parameter-norm ratio failed",
    )
    require(
        item.get("nonfinite_observations") == int(acceptance["maximum_nonfinite_observations"]),
        f"{condition} longitudinal nonfinite count failed",
    )
    return failures


def validate_health_report_payload(
    report: Mapping[str, Any],
    *,
    block: PilotBlock,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
    receipt: Mapping[str, Any],
    receipt_sha256: str,
) -> None:
    """Validate every v4 scientific PASS field without reading external state."""

    acceptance = _mapping(protocol.get("pilot_acceptance"), "protocol.pilot_acceptance")
    overfit = _mapping(acceptance.get("overfit"), "pilot_acceptance.overfit")
    longitudinal = _mapping(
        acceptance.get("longitudinal_health"),
        "pilot_acceptance.longitudinal_health",
    )
    timing = _mapping(acceptance.get("timing"), "pilot_acceptance.timing")
    failures: list[str] = []

    def require(result: bool, message: str) -> None:
        if not result:
            failures.append(message)

    expected_top = {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "dataset": block.dataset,
        "seed": block.seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": receipt.get("git_commit"),
        "attempt_receipt_sha256": receipt_sha256,
    }
    for key, value in expected_top.items():
        require(report.get(key) == value, f"health report {key} mismatch")
    require(report.get("tracked_clean") is True, "health report tracked_clean must be true")
    require(report.get("failures") == [], "health report contains failure records")
    expected_thresholds = {
        "overfit": dict(overfit),
        "longitudinal_health": dict(longitudinal),
        "timing": dict(timing),
    }
    require(report.get("thresholds") == expected_thresholds, "health threshold binding mismatch")

    source = report.get("source")
    require(isinstance(source, Mapping), "health source evidence is missing")
    if isinstance(source, Mapping):
        require(source.get("dataset") == block.dataset, "health source dataset mismatch")
        require(
            source.get("config_hash") == reference_config.config_hash,
            "health reference config hash mismatch",
        )
        require(
            source.get("overfit_batch_size") == overfit.get("batch_size"),
            "health overfit batch-size mismatch",
        )
        require(
            source.get("timing_batch_size") == timing.get("batch_size"),
            "health timing batch-size mismatch",
        )
        try:
            validate_fixed_batch_shape(
                block.dataset,
                source.get("fixed_batch_shape", ()),
                minimum_batch_size=int(timing["batch_size"]),
                time_steps=reference_config.model.time_steps,
                in_channels=reference_config.model.in_channels,
            )
        except PilotV4Error as exc:
            failures.append(str(exc))
        for key in ("split_manifest_sha256", "fixed_batch_sha256"):
            value = source.get(key)
            require(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"health source {key} is invalid",
            )
        if block.dataset == "cifar10dvs":
            value = source.get("dvs_index_sha256")
            require(
                isinstance(value, str)
                and len(value) == 64
                and value == source.get("split_source_fingerprint"),
                "DVS index/source identity is invalid",
            )

    environment_binding = acceptance.get("environment")
    environment = report.get("environment")
    require(isinstance(environment_binding, Mapping), "environment binding is missing")
    require(isinstance(environment, Mapping), "environment evidence is missing")
    if isinstance(environment_binding, Mapping) and isinstance(environment, Mapping):
        hardware = environment.get("hardware")
        require(isinstance(hardware, Mapping), "hardware evidence is missing")
        if isinstance(hardware, Mapping):
            require(
                str(environment_binding["expected_gpu_substring"]).lower()
                in str(hardware.get("name", "")).lower(),
                "GPU binding mismatch",
            )
        require(
            environment.get("pytorch") == environment_binding.get("pytorch_version"),
            "PyTorch version mismatch",
        )
        require(
            str(environment.get("cuda_version")) == str(environment_binding.get("cuda_runtime")),
            "CUDA runtime mismatch",
        )
        require(environment.get("precision") == "float32", "precision must be float32")
        require(environment.get("amp") is False, "amp must be false")
        require(environment.get("torch_deterministic") is True, "determinism is disabled")
        require(
            environment.get("torch_deterministic_warn_only") is False,
            "determinism is warn-only",
        )
        require(environment.get("cudnn_benchmark") is False, "cuDNN benchmark is enabled")
        require(
            environment.get("cudnn_deterministic") is True,
            "cuDNN deterministic mode is disabled",
        )
        idle = environment.get("gpu_idle_precheck")
        require(
            isinstance(idle, Mapping) and idle.get("pass") is True,
            "GPU idle precheck did not pass",
        )

    conditions = report.get("functional_health_by_condition")
    require(isinstance(conditions, Mapping), "per-condition health evidence is missing")
    if isinstance(conditions, Mapping):
        require(set(conditions) == set(block.conditions), "health condition set mismatch")
        for condition in block.conditions:
            item = conditions.get(condition)
            require(
                isinstance(item, Mapping) and item.get("pass") is True,
                f"{condition} functional health did not pass",
            )
            if not isinstance(item, Mapping):
                continue
            overfit_result = item.get("overfit")
            require(isinstance(overfit_result, Mapping), f"{condition} overfit evidence missing")
            if isinstance(overfit_result, Mapping):
                require(overfit_result.get("pass") is True, f"{condition} overfit did not pass")
                require(
                    overfit_result.get("steps") == overfit.get("steps"),
                    f"{condition} overfit step mismatch",
                )
                require(
                    _finite(overfit_result.get("loss_fraction"))
                    and float(overfit_result["loss_fraction"])
                    <= float(overfit["maximum_loss_fraction"]),
                    f"{condition} overfit loss threshold failed",
                )
                require(
                    _finite(overfit_result.get("final_accuracy"))
                    and float(overfit_result["final_accuracy"])
                    >= float(overfit["minimum_accuracy"]),
                    f"{condition} overfit accuracy threshold failed",
                )
                history = overfit_result.get("train_loss_history")
                require(
                    isinstance(history, list)
                    and len(history) == int(overfit["steps"])
                    and all(_finite(value) for value in history),
                    f"{condition} overfit history is incomplete",
                )
            shared = item.get("shared_conv_and_fc_gradients")
            require(
                isinstance(shared, Mapping) and shared.get("pass") is True,
                f"{condition} shared gradients failed",
            )
            ta = item.get("ta_gradients")
            require(
                isinstance(ta, Mapping) and ta.get("pass") is True,
                f"{condition} TA gate failed",
            )
            if isinstance(ta, Mapping):
                if CONDITION_SPECS[condition]["neuron"] == "ta_lif":
                    require(
                        _finite(ta.get("aggregate_routed_gradient_coverage"))
                        and float(ta["aggregate_routed_gradient_coverage"])
                        >= float(overfit["minimum_ta_routed_gradient_coverage"]),
                        f"{condition} routed-gradient coverage failed",
                    )
                    require(
                        ta.get("every_ta_parameter_tensor_has_nonzero_routed_coverage") is True
                        and ta.get("all_ta_gradients_present_and_finite_on_every_probe") is True
                        and ta.get("no_nonzero_gradient_in_unrouted_slots") is True,
                        f"{condition} TA gradient evidence is incomplete",
                    )
                else:
                    require(
                        ta.get("status") == "not_applicable_lif_condition",
                        f"{condition} LIF TA status is invalid",
                    )
            checkpoint = item.get("checkpoint_next_step")
            require(
                isinstance(checkpoint, Mapping) and checkpoint.get("pass") is True,
                f"{condition} checkpoint gate failed",
            )

    longitudinal_results = report.get("longitudinal_health_by_condition")
    require(
        isinstance(longitudinal_results, Mapping),
        "per-condition longitudinal health evidence is missing",
    )
    if isinstance(longitudinal_results, Mapping):
        require(
            set(longitudinal_results) == set(block.conditions),
            "longitudinal condition set mismatch",
        )
        for condition in block.conditions:
            item = longitudinal_results.get(condition)
            if not isinstance(item, Mapping):
                failures.append(f"{condition} longitudinal evidence is missing")
            else:
                failures.extend(
                    _validate_longitudinal_condition(
                        item, condition=condition, acceptance=longitudinal
                    )
                )

    cuda_timing = report.get("cuda_train_step_timing")
    require(
        isinstance(cuda_timing, Mapping) and cuda_timing.get("pass") is True,
        "CUDA timing failed",
    )
    if isinstance(cuda_timing, Mapping):
        timing_conditions = cuda_timing.get("conditions")
        require(
            isinstance(timing_conditions, Mapping)
            and set(timing_conditions) == set(block.conditions),
            "CUDA timing condition set mismatch",
        )
        ratios = cuda_timing.get("ta_enabled_over_frozen_ratios")
        require(
            isinstance(ratios, Mapping) and set(ratios) == {"C2"},
            "TA timing ratio set must be C2 only",
        )
        if isinstance(ratios, Mapping):
            ratio = ratios.get("C2")
            limit = float(timing["maximum_ta_enabled_over_frozen_ratio"])
            round_ratios = ratio.get("round_ratios") if isinstance(ratio, Mapping) else None
            require(
                isinstance(ratio, Mapping)
                and ratio.get("pass") is True
                and _finite(ratio.get("ta_enabled_over_frozen"))
                and float(ratio["ta_enabled_over_frozen"]) <= limit
                and isinstance(round_ratios, list)
                and len(round_ratios) == 2
                and all(_finite(value) and float(value) <= limit for value in round_ratios),
                "C2 TA timing ratio is invalid",
            )

    if failures:
        raise PilotV4Error("V4 pilot health report rejected:\n- " + "\n- ".join(failures))


def validate_health_report(
    report_path: str | Path,
    protocol_path: str | Path,
    dataset: str,
    *,
    repository_root: str | Path,
    expected_git_commit: str | None = None,
    require_current_tracked_clean: bool = True,
) -> dict[str, Any]:
    from .config import load_protocol

    root = Path(repository_root).resolve()
    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    block = resolve_pilot_block(protocol, dataset, repository_root=root)
    path = Path(report_path).resolve()
    if path != block.health_output or not path.is_file():
        raise PilotV4Error("The canonical dataset-specific v4 health report is missing")
    receipt = validate_attempt_receipt(
        block.attempt_receipt,
        block=block,
        protocol_path=protocol_file,
        repository_root=root,
    )
    report = _read_json(path, "v4 health report")
    reference = expected_pilot_configs(protocol, block.dataset)[0]
    validate_health_report_payload(
        report,
        block=block,
        protocol=protocol,
        reference_config=reference,
        receipt=receipt,
        receipt_sha256=sha256_file(block.attempt_receipt),
    )
    if expected_git_commit is None or require_current_tracked_clean:
        identity = repository_git_identity(root)
        if expected_git_commit is None:
            expected_git_commit = str(identity["git_commit"])
        if require_current_tracked_clean and identity["tracked_clean"] is not True:
            raise PilotV4Error("Current tracked worktree is not clean")
    if report.get("git_commit") != expected_git_commit:
        raise PilotV4Error("Health report Git commit differs from the required commit")
    return report


def expected_pilot_plan_payload(
    *,
    block: PilotBlock,
    protocol_path: str | Path,
    config_dir: str | Path,
    manifest_rows: Sequence[Mapping[str, Any]],
    health_report: Mapping[str, Any],
    repository_root: str | Path,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    config_root = Path(config_dir).resolve()
    selected = [row for row in manifest_rows if str(row.get("dataset")) == block.dataset]
    if len(selected) != 2 or [str(row.get("condition")) for row in selected] != [
        "C1",
        "C2",
    ]:
        raise PilotV4Error("A v4 pilot plan requires one ordered C1/C2 dataset block")
    runs: list[dict[str, Any]] = []
    for row in selected:
        config_path = (config_root / str(row.get("config_file", ""))).resolve()
        if not config_path.is_file():
            raise PilotV4Error(f"Pilot config is missing: {config_path}")
        runs.append(
            {
                "condition": str(row["condition"]),
                "run_id": str(row["run_id"]),
                "config_file": artifact_path_reference(config_path, root),
                "config_file_sha256": sha256_file(config_path),
                "config_hash": str(row["config_hash"]),
            }
        )
    return {
        "version": 4,
        "artifact_class": "NON_REPORTABLE_V4_PILOT_PLAN",
        "non_reportable": True,
        "dataset": block.dataset,
        "seed": block.seed,
        "conditions": list(block.conditions),
        "protocol_path": artifact_path_reference(protocol_path, root),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": health_report.get("git_commit"),
        "health_report": artifact_path_reference(block.health_output, root),
        "health_report_sha256": sha256_file(block.health_output),
        "attempt_receipt": artifact_path_reference(block.attempt_receipt, root),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "pilot_output_root": artifact_path_reference(block.pilot_output_root, root),
        "formal_output_root": artifact_path_reference(block.formal_output_root, root),
        "runs": runs,
    }
