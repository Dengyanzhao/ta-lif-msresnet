#!/usr/bin/env python3
"""Validate the frozen, non-reportable V7 six-condition mechanism pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for index, path in enumerate((SRC_ROOT, SCRIPTS_ROOT)):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(index, value)

import validate_v3_pilot as common

from talif_msresnet.config import (
    RunConfig,
    artifact_paths_for_protocol,
    load_protocol,
    load_run_config,
    validate_run_mapping,
)
from talif_msresnet.config_v7 import (
    V7_ACTIVE_CONDITIONS,
    canonicalize_v7_artifact_run_mapping,
    generate_v7_pilot_matrix,
)
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.pilot_v7 import (
    REPORTING_ELIGIBILITY,
    V7_HEALTH_RECOVERY_SEAL_COMMIT,
    V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE,
    V7_PILOT_DEVICE_RECOVERY_FROZEN_DEVICE,
    V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION,
    V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256,
    V7_PILOT_DEVICE_RECOVERY_OUTPUT,
    V7_PILOT_DEVICE_RECOVERY_PROTOCOL_HASH,
    PilotBlock,
    PilotV7Error,
    attempt_receipt_payload,
    exclusive_create_json,
    expected_pilot_configs,
    expected_pilot_plan_payload,
    require_v7_author_freeze,
    resolve_pilot_block,
    validate_attempt_receipt,
    validate_health_report,
    validate_pilot_device_recovery_release,
)
from talif_msresnet.utils import (
    load_checkpoint,
    sha256_file,
    stable_hash,
    utc_now,
)

SCHEMA_VERSION = 1
ARTIFACT_CLASS = "NON_REPORTABLE_V7_MECHANISM_PILOT_ACCEPTANCE"
PASS_DECISION = "ACCEPT_V7_SIX_CONDITION_120_EPOCH_PILOT_RELEASE_FORMAL_FREEZE"
FAIL_DECISION = "BLOCK_V7_AND_REQUIRE_NEW_PROTOCOL_AND_UNUSED_PILOT_SEED"
INVALID_DECISION = "BLOCK_V7_AND_INVESTIGATE_PILOT_EVIDENCE_INTEGRITY"
DEVICE_RECOVERY_ARTIFACT_CLASS = (
    "NON_REPORTABLE_V7_PILOT_VALIDATION_DEVICE_COMPATIBILITY_RECOVERY"
)
DEVICE_RECOVERY_DECISION = (
    "RECOVER_V7_PILOT_PASS_AFTER_DEVICE_EXECUTION_EQUIVALENCE_FIX"
)
ADAPTIVE_CONDITIONS = frozenset({"M2", "M3", "M4", "PLIF"})
TEST_FIELDS = (
    "test_loss",
    "test_accuracy",
    "test_samples",
    "test_checkpoint_sha256",
    "test_evaluated_at",
)


class PilotValidationError(RuntimeError):
    """Raised when the validator cannot establish the V7 pilot contract."""


def _report_without_timestamp(report: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(report)
    normalized.pop("validated_at", None)
    return normalized


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotValidationError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotValidationError(f"{label} must be a JSON object: {path}")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: Any) -> float | None:
    return common._finite_number(value)


def _same_number(observed: Any, expected: float) -> bool:
    return common._same_number(observed, expected)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _load_generated_matrix(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    repository_root: Path,
) -> tuple[
    list[Mapping[str, Any]],
    dict[str, RunConfig],
    list[str],
    Path,
    Path,
]:
    failures: list[str] = []
    manifest_path = config_dir / "matrix_manifest.json"
    csv_path = config_dir / "run_manifest.csv"
    manifest = _read_json(manifest_path, "V7 pilot matrix manifest")
    raw_rows = manifest.get("runs")
    if not isinstance(raw_rows, list):
        raise PilotValidationError("V7 matrix manifest runs must be a list")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        failures.append("matrix manifest contains a non-object run")

    expected_raw = generate_v7_pilot_matrix(protocol)
    expected_configs = [validate_run_mapping(row, protocol) for row in expected_raw]
    expected_by_condition = {config.model.condition: config for config in expected_configs}
    expected_order = [
        (config.model.condition, config.runtime.run_id) for config in expected_configs
    ]
    observed_order = [(str(row.get("condition")), str(row.get("run_id"))) for row in rows]
    if observed_order != expected_order:
        failures.append("matrix manifest is not the ordered six-condition V7 pilot")
    expected_manifest = {
        "protocol": artifact_path_reference(protocol_path, repository_root),
        "protocol_hash": stable_hash(protocol),
        "run_count": len(V7_ACTIVE_CONDITIONS),
        "conditions": list(V7_ACTIVE_CONDITIONS),
        "seeds": list(dict.fromkeys(config.runtime.seed for config in expected_configs)),
        "matrix_hash": stable_hash(expected_raw),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            failures.append(f"matrix manifest {key} mismatch")

    for row, expected in zip(rows, expected_configs):
        expected_file = f"{expected.runtime.run_id}.yaml"
        if row.get("config_file") != expected_file:
            failures.append(f"{expected.runtime.run_id}: config filename is noncanonical")
            continue
        config_path = config_dir / expected_file
        if not config_path.is_file():
            failures.append(f"{expected.runtime.run_id}: generated YAML is missing")
            continue
        if row.get("config_file_sha256") != sha256_file(config_path):
            failures.append(f"{expected.runtime.run_id}: config SHA-256 mismatch")
        try:
            loaded = load_run_config(config_path, protocol_path)
        except ValueError as exc:
            failures.append(f"{expected.runtime.run_id}: generated YAML is invalid: {exc}")
        else:
            if loaded.as_dict() != expected.as_dict():
                failures.append(f"{expected.runtime.run_id}: generated YAML differs from protocol")
        fields = {
            "experiment": expected.experiment,
            "dataset": expected.data.dataset,
            "depth": expected.model.depth,
            "time_steps": expected.model.time_steps,
            "condition": expected.model.condition,
            "topology": expected.model.topology,
            "neuron": expected.model.neuron,
            "seed": expected.runtime.seed,
            "config_hash": expected.config_hash,
            "protocol_hash": expected.analysis["protocol_hash"],
        }
        for key, expected_value in fields.items():
            if str(row.get(key)) != str(expected_value):
                failures.append(f"{expected.runtime.run_id}: matrix row {key} mismatch")

    if not csv_path.is_file():
        failures.append("generated run_manifest.csv is missing")
    else:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise PilotValidationError(
                f"Cannot read generated run manifest {csv_path}: {exc}"
            ) from exc
        normalized = [{key: str(value) for key, value in row.items()} for row in rows]
        if csv_rows != normalized:
            failures.append("run_manifest.csv differs from matrix_manifest.json")
    return rows, expected_by_condition, failures, manifest_path, csv_path


def _bound_evidence(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    repository_root: Path,
    block: PilotBlock,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    for label, path in (
        ("health report", block.health_output),
        ("health attempt receipt", block.attempt_receipt),
        ("pilot plan", block.pilot_plan),
    ):
        if not path.is_file():
            failures.append(f"{label} is missing: {path}")
    if failures:
        return {}, failures

    receipt = _read_json(block.attempt_receipt, "V7 health attempt receipt")
    try:
        validate_attempt_receipt(receipt, block=block)
    except PilotV7Error as exc:
        failures.append(str(exc))
    receipt_source = receipt.get("source")
    runtime_sources = (
        receipt_source.get("runtime_sources_sha256")
        if isinstance(receipt_source, Mapping)
        else None
    )
    if not isinstance(runtime_sources, Mapping):
        failures.append("attempt receipt runtime-source hashes are missing")
    else:
        configs = expected_pilot_configs(protocol)
        config_paths = tuple(config_dir / f"{config.runtime.run_id}.yaml" for config in configs)
        try:
            expected_receipt = attempt_receipt_payload(
                block=block,
                protocol_path=protocol_path,
                config_paths=config_paths,
                configs=configs,
                git_identity={
                    "git_commit": receipt.get("git_commit"),
                    "tracked_clean": receipt.get("tracked_clean"),
                },
                repository_root=repository_root,
                runtime_sources=dict(runtime_sources),
            )
            expected_receipt["claimed_at"] = receipt.get("claimed_at")
            if receipt != expected_receipt:
                failures.append("attempt receipt differs from its exact source-bound payload")
        except (OSError, PilotV7Error) as exc:
            failures.append(f"attempt receipt source binding failed: {exc}")

    try:
        health = validate_health_report(
            block.health_output,
            protocol_path,
            repository_root=repository_root,
            require_current_tracked_clean=True,
        )
    except PilotV7Error as exc:
        health = _read_json(block.health_output, "V7 health report")
        failures.append(f"canonical health validation failed: {exc}")

    plan = _read_json(block.pilot_plan, "V7 pilot plan")
    try:
        expected_plan = expected_pilot_plan_payload(
            block=block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=rows,
            health_report=health,
            repository_root=repository_root,
        )
    except PilotV7Error as exc:
        failures.append(str(exc))
    else:
        if plan != expected_plan:
            failures.append("pilot plan differs from the exact V7 bound payload")
    recovery = health.get("compatibility_recovery")
    if recovery is not None:
        if not isinstance(recovery, Mapping):
            failures.append("health compatibility recovery binding is malformed")
            recovery = None
        elif plan.get("health_compatibility_recovery") != recovery:
            failures.append("pilot plan health compatibility recovery binding mismatch")
    return {
        "health": health,
        "health_path": artifact_path_reference(block.health_output, repository_root),
        "health_sha256": sha256_file(block.health_output),
        "attempt_receipt_path": artifact_path_reference(block.attempt_receipt, repository_root),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "plan": plan,
        "plan_path": artifact_path_reference(block.pilot_plan, repository_root),
        "plan_sha256": sha256_file(block.pilot_plan),
        "health_compatibility_recovery": recovery,
    }, failures


def _optimizer_group_failures(
    manifest: Any,
    *,
    config: RunConfig,
    adaptive_names: Any,
) -> list[str]:
    failures: list[str] = []
    condition = config.model.condition
    expects_adaptive = condition in ADAPTIVE_CONDITIONS
    if not isinstance(manifest, Mapping):
        return ["optimizer-group manifest is missing"]
    if manifest.get("schema_version") != 1:
        failures.append("optimizer-group manifest schema is unsupported")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        return [*failures, "optimizer-group manifest has no groups"]
    if not isinstance(adaptive_names, list):
        failures.append("adaptive parameter names are not a list")
        adaptive_names = []
    expected_roles = ["base_decay", "base_no_decay"]
    if expects_adaptive:
        expected_roles.append("adaptive")
    roles = [group.get("role") for group in groups if isinstance(group, Mapping)]
    if roles != expected_roles:
        failures.append(f"optimizer roles {roles!r} differ from expected {expected_roles!r}")
    all_names: list[str] = []
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping):
            failures.append(f"optimizer group {index} is not a mapping")
            continue
        names = group.get("parameter_names")
        if not isinstance(names, list) or not names:
            failures.append(f"optimizer group {index} has no parameter names")
            names = []
        all_names.extend(str(name) for name in names)
        if group.get("parameter_count") != len(names):
            failures.append(f"optimizer group {index} parameter count mismatch")
        numel = group.get("parameter_numel")
        if not isinstance(numel, int) or numel <= 0:
            failures.append(f"optimizer group {index} parameter numel is invalid")
        initial_lr = _finite_number(group.get("initial_lr"))
        current_lr = _finite_number(group.get("current_lr"))
        weight_decay = _finite_number(group.get("weight_decay"))
        if initial_lr is None or current_lr is None or current_lr < 0:
            failures.append(f"optimizer group {index} learning rate is invalid")
        role = group.get("role")
        if role == "base_decay":
            if initial_lr != config.optimizer.lr:
                failures.append("base-decay initial learning rate mismatch")
            if weight_decay != config.optimizer.weight_decay:
                failures.append("base-decay weight decay mismatch")
        elif role == "base_no_decay":
            if initial_lr != config.optimizer.lr or weight_decay != 0.0:
                failures.append("base-no-decay optimizer contract mismatch")
        elif role == "adaptive":
            expected_lr = config.optimizer.lr * config.optimizer.ta_lr_scale
            if initial_lr != expected_lr:
                failures.append("adaptive initial learning rate mismatch")
            if weight_decay != config.optimizer.ta_weight_decay:
                failures.append("adaptive weight decay mismatch")
            if names != adaptive_names:
                failures.append("adaptive optimizer names differ from run manifest")
    if len(all_names) != len(set(all_names)):
        failures.append("optimizer parameter names are duplicated across groups")
    if expects_adaptive:
        if not adaptive_names:
            failures.append(f"{condition} has no adaptive parameter names")
    elif adaptive_names:
        failures.append(f"{condition} unexpectedly has adaptive parameter names")
    return failures


def _checkpoint_optimizer_failures(
    path: Path,
    *,
    label: str,
    config: RunConfig,
    adaptive_names: Any,
) -> list[str]:
    try:
        checkpoint = load_checkpoint(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - integrity report needs the load error.
        return [f"{label} cannot be loaded for optimizer audit: {type(exc).__name__}: {exc}"]
    failures = _optimizer_group_failures(
        checkpoint.get("optimizer_group_manifest"),
        config=config,
        adaptive_names=adaptive_names,
    )
    manifest = checkpoint.get("optimizer_group_manifest")
    optimizer_state = checkpoint.get("optimizer_state")
    groups = manifest.get("groups") if isinstance(manifest, Mapping) else None
    state_groups = (
        optimizer_state.get("param_groups") if isinstance(optimizer_state, Mapping) else None
    )
    if not isinstance(groups, list) or not isinstance(state_groups, list):
        failures.append(f"{label} optimizer state/group evidence is incomplete")
        return failures
    if len(groups) != len(state_groups):
        failures.append(f"{label} optimizer state group count mismatch")
        return failures
    for index, (group, state) in enumerate(zip(groups, state_groups)):
        if not isinstance(group, Mapping) or not isinstance(state, Mapping):
            failures.append(f"{label} optimizer group {index} is malformed")
            continue
        names = group.get("parameter_names")
        params = state.get("params")
        if not isinstance(names, list) or not isinstance(params, list) or len(names) != len(params):
            failures.append(f"{label} optimizer group {index} state size mismatch")
        for manifest_key, state_key in (
            ("initial_lr", "initial_lr"),
            ("current_lr", "lr"),
            ("weight_decay", "weight_decay"),
        ):
            left = _finite_number(group.get(manifest_key))
            right = _finite_number(state.get(state_key))
            if left is None or right is None or left != right:
                failures.append(f"{label} optimizer group {index} {manifest_key} mismatch")
    return failures


def _orchestrator_failures(
    evidence: Any,
    *,
    block: PilotBlock,
    bound: Mapping[str, Any],
    environment_hash: str,
) -> list[str]:
    if not isinstance(evidence, Mapping):
        return ["run manifest has no orchestrator evidence"]
    expected = {
        "protocol_version": 7,
        "execution_stage": "pilot",
        "plan_version": 7,
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "health_report": bound.get("health_path"),
        "health_report_sha256": bound.get("health_sha256"),
        "attempt_receipt": bound.get("attempt_receipt_path"),
        "attempt_receipt_sha256": bound.get("attempt_receipt_sha256"),
        "pilot_plan": bound.get("plan_path"),
        "pilot_plan_sha256": bound.get("plan_sha256"),
        "git_commit": (
            bound.get("plan", {}).get("git_commit")
            if isinstance(bound.get("plan"), Mapping)
            else None
        ),
    }
    failures = [
        f"orchestrator evidence {key} mismatch"
        for key, value in expected.items()
        if evidence.get(key) != value
    ]
    runtime = evidence.get("runtime_context")
    runtime_expected = {
        "pass": True,
        "protocol_version": 7,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "block_hash": block.block_hash,
        "training_environment_sha256": environment_hash,
    }
    if not isinstance(runtime, Mapping):
        failures.append("orchestrator runtime context is missing")
    else:
        failures.extend(
            f"orchestrator runtime context {key} mismatch"
            for key, value in runtime_expected.items()
            if runtime.get(key) != value
        )
        recovery = bound.get("health_compatibility_recovery")
        if recovery is not None and runtime.get(
            "health_compatibility_recovery"
        ) != recovery:
            failures.append(
                "orchestrator runtime context health compatibility recovery mismatch"
            )
    return failures


def _threshold_summary(
    history: Sequence[Mapping[str, Any]],
    *,
    best_accuracy: float,
    required_best: float,
    required_gradient_coverage: float,
    late_window_epochs: int,
    minimum_late_to_best_ratio: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    integrity: list[str] = []
    thresholds: list[str] = []
    coverages: list[float] = []
    for row in history:
        zero_fraction = _finite_number(row.get("all_zero_block_gradient_batch_fraction"))
        if zero_fraction is None or not 0.0 <= zero_fraction <= 1.0:
            integrity.append(
                f"epoch {row.get('epoch')}: all-zero block-gradient fraction is invalid"
            )
        else:
            coverages.append(1.0 - zero_fraction)
    minimum_coverage = min(coverages) if len(coverages) == len(history) else None
    if minimum_coverage is not None and minimum_coverage < required_gradient_coverage:
        thresholds.append(
            f"minimum gradient coverage {minimum_coverage:.6f} is below "
            f"{required_gradient_coverage:.6f}"
        )
    if best_accuracy < required_best:
        thresholds.append(
            f"best validation accuracy {best_accuracy:.6f} is below {required_best:.6f}"
        )
    late_values = [_finite_number(row.get("val_accuracy")) for row in history[-late_window_epochs:]]
    if len(late_values) != late_window_epochs or any(value is None for value in late_values):
        integrity.append("late validation window is incomplete or non-finite")
        late_mean = None
        late_ratio = None
    else:
        finite_late = [float(value) for value in late_values if value is not None]
        late_mean = sum(finite_late) / len(finite_late)
        late_ratio = late_mean / best_accuracy if best_accuracy > 0.0 else math.inf
        if not math.isfinite(late_ratio):
            integrity.append("late-to-best ratio is non-finite")
        elif late_ratio < minimum_late_to_best_ratio:
            thresholds.append(
                f"late-to-best ratio {late_ratio:.6f} is below {minimum_late_to_best_ratio:.6f}"
            )
    return (
        {
            "best_validation_accuracy": best_accuracy,
            "required_best_validation_accuracy": required_best,
            "minimum_gradient_coverage": minimum_coverage,
            "required_gradient_coverage": required_gradient_coverage,
            "late_window_epochs": late_window_epochs,
            "late_mean_validation_accuracy": late_mean,
            "late_to_best_ratio": late_ratio,
            "minimum_late_to_best_ratio": minimum_late_to_best_ratio,
        },
        integrity,
        thresholds,
    )


def _recovery_config_identity(
    raw: Any,
    *,
    label: str,
    protocol: Mapping[str, Any],
    repository_root: Path,
    expected_config: RunConfig,
) -> tuple[str | None, list[str]]:
    """Prove one raw execution config differs only by the reviewed aliases."""

    failures: list[str] = []
    if not isinstance(raw, Mapping):
        return None, [f"{label} config is not a mapping"]
    runtime = raw.get("runtime")
    if not isinstance(runtime, Mapping):
        return None, [f"{label} config has no runtime mapping"]
    if runtime.get("device") != V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE:
        failures.append(
            f"{label} runtime.device is not the reviewed execution device "
            f"{V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE!r}"
        )
    raw_execution_hash = stable_hash(raw)
    try:
        normalized_raw = canonicalize_v7_artifact_run_mapping(
            raw,
            protocol,
            project_root=repository_root,
            expected_output_dir=expected_config.runtime.output_dir,
            expected_execution_device=V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE,
        )
        normalized = validate_run_mapping(normalized_raw, protocol)
    except Exception as exc:  # noqa: BLE001 - record artifact-integrity failure.
        failures.append(
            f"{label} config cannot be normalized by the reviewed device/path "
            f"equivalence: {type(exc).__name__}: {exc}"
        )
        return raw_execution_hash, failures
    if normalized.as_dict() != expected_config.as_dict():
        failures.append(f"{label} normalized config differs from the frozen run")
    if normalized.execution_hash != expected_config.execution_hash:
        failures.append(
            f"{label} normalized execution hash differs from the frozen run"
        )
    if normalized.config_hash != expected_config.config_hash:
        failures.append(f"{label} normalized scientific hash differs from the frozen run")
    return raw_execution_hash, failures


def _stored_recovery_execution_identity_failures(
    raw: Any,
    stored_execution_hash: Any,
    *,
    label: str,
    protocol: Mapping[str, Any],
    repository_root: Path,
    expected_config: RunConfig,
    reference_raw: Mapping[str, Any],
    reference_execution_hash: str,
) -> list[str]:
    """Bind a manifest/checkpoint config and hash to the raw resolved config."""

    raw_execution_hash, failures = _recovery_config_identity(
        raw,
        label=label,
        protocol=protocol,
        repository_root=repository_root,
        expected_config=expected_config,
    )
    if raw != reference_raw:
        failures.append(f"{label} raw config differs from resolved_config.json")
    if stored_execution_hash != raw_execution_hash:
        failures.append(f"{label} execution_hash does not hash its raw config")
    if stored_execution_hash != reference_execution_hash:
        failures.append(
            f"{label} execution_hash differs from resolved_config.json identity"
        )
    return failures


def _checkpoint_recovery_execution_identity_failures(
    path: Path,
    *,
    label: str,
    protocol: Mapping[str, Any],
    repository_root: Path,
    expected_config: RunConfig,
    reference_raw: Mapping[str, Any],
    reference_execution_hash: str,
) -> list[str]:
    try:
        checkpoint = load_checkpoint(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - record artifact-integrity failure.
        return [
            (
                f"{label} cannot be loaded for execution-identity audit: "
                f"{type(exc).__name__}: {exc}"
            )
        ]
    return _stored_recovery_execution_identity_failures(
        checkpoint.get("config"),
        checkpoint.get("execution_hash"),
        label=label,
        protocol=protocol,
        repository_root=repository_root,
        expected_config=expected_config,
        reference_raw=reference_raw,
        reference_execution_hash=reference_execution_hash,
    )


def _trainer_recovery_launch_failures(
    events: Sequence[Mapping[str, Any]], *, expected_config: RunConfig
) -> list[str]:
    starts = [event for event in events if event.get("event") == "run_started"]
    if len(starts) != 1:
        return [
            (
                "trainer event stream must contain exactly one run_started event for "
                "device recovery"
            )
        ]
    start = starts[0]
    failures: list[str] = []
    if start.get("device") != V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE:
        failures.append(
            "trainer run_started device is not the reviewed execution device "
            f"{V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE!r}"
        )
    if start.get("dry_run") is not False:
        failures.append("trainer run_started does not prove a non-dry-run execution")
    if start.get("config_hash") != expected_config.config_hash:
        failures.append("trainer run_started scientific config hash mismatch")
    return failures


def _matrix_recovery_launch_failures(
    output_root: Path,
    *,
    expected_run_ids: set[str],
    bound: Mapping[str, Any],
) -> list[str]:
    """Prove each matrix launch used the single reviewed CUDA selector."""

    path = output_root / "matrix_events.jsonl"
    if not path.is_file():
        return ["device recovery requires matrix_events.jsonl"]
    try:
        events = common._read_events(path)
    except common.PilotValidationError as exc:
        return [f"cannot read matrix_events.jsonl for device recovery: {exc}"]
    health = bound.get("health")
    health_environment = health.get("environment") if isinstance(health, Mapping) else None
    idle = (
        health_environment.get("gpu_idle_precheck")
        if isinstance(health_environment, Mapping)
        else None
    )
    expected_uuid = idle.get("device_uuid") if isinstance(idle, Mapping) else None
    expected_environment_hash = (
        health_environment.get("training_environment_sha256")
        if isinstance(health_environment, Mapping)
        else None
    )
    failures: list[str] = []
    if not isinstance(expected_uuid, str) or not expected_uuid:
        failures.append("bound health report has no CUDA device UUID")
    for run_id in sorted(expected_run_ids):
        indexed = [
            (index, event)
            for index, event in enumerate(events)
            if event.get("run_id") == run_id
        ]
        starts = [
            (index, event)
            for index, event in indexed
            if event.get("event") == "run_started"
        ]
        if len(starts) != 1:
            failures.append(
                f"{run_id}: matrix event stream must contain exactly one run_started event"
            )
            continue
        start_index, start = starts[0]
        command = start.get("command")
        if not isinstance(command, list) or not all(
            isinstance(value, str) for value in command
        ):
            failures.append(f"{run_id}: matrix run_started command is malformed")
        else:
            device_positions = [
                index for index, value in enumerate(command) if value == "--device"
            ]
            combined_devices = [
                value for value in command if value.startswith("--device=")
            ]
            exact_device_pair = bool(
                len(device_positions) == 1
                and device_positions[0] + 1 < len(command)
                and command[device_positions[0] + 1]
                == V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE
                and not combined_devices
            )
            if not exact_device_pair:
                failures.append(
                    f"{run_id}: matrix command does not contain exactly "
                    f"--device {V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE}"
                )
        contexts = [
            (index, event)
            for index, event in indexed
            if event.get("event") == "run_context_validated"
        ]
        if len(contexts) != 1:
            failures.append(
                f"{run_id}: matrix event stream must contain exactly one "
                "run_context_validated event"
            )
            continue
        context_index, context_event = contexts[0]
        context = context_event.get("context")
        if context_index >= start_index:
            failures.append(f"{run_id}: matrix runtime context was not validated before launch")
        if not isinstance(context, Mapping):
            failures.append(f"{run_id}: matrix runtime context is malformed")
            continue
        if context.get("device_uuid") != expected_uuid:
            failures.append(f"{run_id}: matrix runtime context CUDA UUID mismatch")
        if context.get("training_environment_sha256") != expected_environment_hash:
            failures.append(f"{run_id}: matrix runtime context environment hash mismatch")
        if context.get("device") != V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE:
            failures.append(f"{run_id}: matrix runtime context device mismatch")
    return failures


def _validate_run(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    repository_root: Path,
    block: PilotBlock,
    row: Mapping[str, Any],
    expected_config: RunConfig,
    bound: Mapping[str, Any],
    pilot_contract: Mapping[str, Any],
    allow_device_execution_equivalence: bool = False,
) -> tuple[dict[str, Any], str | None, str | None, str | None]:
    condition = expected_config.model.condition
    run_id = expected_config.runtime.run_id
    run_dir = block.pilot_output_root / run_id
    integrity: list[str] = []
    thresholds: list[str] = []
    required = (
        "seed_metrics.json",
        "run_manifest.json",
        "resolved_config.json",
        "events.jsonl",
        "best.pt",
        "last.pt",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        integrity.append(f"missing required artifacts: {missing}")
    if not list(run_dir.glob("attempt_*.stdout.log")):
        integrity.append("matrix stdout attempt log is missing")
    if not list(run_dir.glob("attempt_*.stderr.log")):
        integrity.append("matrix stderr attempt log is missing")
    if (run_dir / "final_test.json").exists():
        integrity.append("final_test.json exists before formal model selection is frozen")
    config_path = config_dir / str(row.get("config_file", ""))
    if not config_path.is_file():
        integrity.append(f"generated config is missing: {config_path.name}")
    elif sha256_file(config_path) != row.get("config_file_sha256"):
        integrity.append("generated config file SHA-256 mismatch")
    if missing or not config_path.is_file():
        return (
            {
                "run_id": run_id,
                "integrity_failures": integrity,
                "threshold_failures": thresholds,
            },
            None,
            None,
            None,
        )

    try:
        planned = load_run_config(config_path, protocol_path)
        metrics = _read_json(run_dir / "seed_metrics.json", "seed metrics")
        run_manifest = _read_json(run_dir / "run_manifest.json", "run manifest")
        resolved_artifact_raw = _read_json(
            run_dir / "resolved_config.json", "resolved configuration"
        )
        resolved_raw = common._canonicalize_execution_output_dir(
            resolved_artifact_raw,
            expected_output_dir=expected_config.runtime.output_dir,
            actual_output_root=block.pilot_output_root,
        )
        if allow_device_execution_equivalence:
            resolved_raw = canonicalize_v7_artifact_run_mapping(
                resolved_raw,
                protocol,
                project_root=repository_root,
                expected_output_dir=expected_config.runtime.output_dir,
                expected_execution_device=V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE,
            )
        resolved = validate_run_mapping(resolved_raw, protocol)
        events = common._read_events(run_dir / "events.jsonl")
    except (ValueError, PilotValidationError, common.PilotValidationError) as exc:
        integrity.append(str(exc))
        return (
            {
                "run_id": run_id,
                "integrity_failures": integrity,
                "threshold_failures": thresholds,
            },
            None,
            None,
            None,
        )

    expected_hash = expected_config.config_hash
    if planned.as_dict() != expected_config.as_dict():
        integrity.append("generated YAML differs from the in-memory V7 pilot matrix")
    if row.get("config_hash") != expected_hash:
        integrity.append("matrix row config_hash differs from expected")
    if planned.config_hash != expected_hash or resolved.config_hash != expected_hash:
        integrity.append("planned/resolved scientific config hash mismatch")
    if allow_device_execution_equivalence:
        resolved_execution_hash, recovery_identity_failures = _recovery_config_identity(
            resolved_artifact_raw,
            label="resolved_config.json",
            protocol=protocol,
            repository_root=repository_root,
            expected_config=expected_config,
        )
        integrity.extend(recovery_identity_failures)
        if resolved_execution_hash is None:
            integrity.append(
                "resolved_config.json has no recoverable raw execution identity"
            )
        else:
            integrity.extend(
                _stored_recovery_execution_identity_failures(
                    run_manifest.get("config"),
                    run_manifest.get("execution_hash"),
                    label="run_manifest.json",
                    protocol=protocol,
                    repository_root=repository_root,
                    expected_config=expected_config,
                    reference_raw=resolved_artifact_raw,
                    reference_execution_hash=resolved_execution_hash,
                )
            )
        integrity.extend(
            _trainer_recovery_launch_failures(events, expected_config=expected_config)
        )
    checks = {
        "metrics.run_id": (metrics.get("run_id"), run_id),
        "metrics.condition": (metrics.get("condition"), condition),
        "metrics.seed": (metrics.get("seed"), block.pilot_seed),
        "metrics.dataset": (metrics.get("dataset"), block.dataset),
        "metrics.config_hash": (metrics.get("config_hash"), expected_hash),
        "metrics.protocol_hash": (metrics.get("protocol_hash"), block.protocol_hash),
        "manifest.run_id": (run_manifest.get("run_id"), run_id),
        "manifest.config_hash": (run_manifest.get("config_hash"), expected_hash),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            integrity.append(f"{label}={observed!r}, expected {expected!r}")
    if metrics.get("status") != "complete" or metrics.get("failed") not in (
        0,
        "0",
        False,
    ):
        integrity.append("terminal metrics do not record a successful complete run")
    if run_manifest.get("status") != "complete":
        integrity.append("run manifest status is not complete")
    if any(metrics.get(field) not in ("", None) for field in TEST_FIELDS):
        integrity.append("terminal metrics contain forbidden test-set fields")
    if not any(event.get("event") == "run_started" for event in events):
        integrity.append("trainer event stream has no run_started event")
    if not any(event.get("event") == "run_completed" for event in events):
        integrity.append("trainer event stream has no run_completed event")

    environment_text = metrics.get("training_environment_identity")
    environment_hash = str(metrics.get("training_environment_sha256", ""))
    try:
        identity = json.loads(str(environment_text))
    except json.JSONDecodeError:
        identity = None
    if not isinstance(identity, Mapping):
        integrity.append("training environment identity is invalid")
        environment_hash_value: str | None = None
    else:
        environment_hash_value = environment_hash if _is_sha256(environment_hash) else None
        if environment_hash_value is None or common._identity_sha256(identity) != environment_hash:
            integrity.append("training environment identity SHA-256 mismatch")
        integrity.extend(
            common._validate_environment(identity, protocol["pilot_acceptance"]["environment"])
        )
        health = bound.get("health")
        if isinstance(health, Mapping):
            integrity.extend(common._health_environment_failures(identity, health))
            health_environment = health.get("environment")
            health_hash = (
                health_environment.get("training_environment_sha256")
                if isinstance(health_environment, Mapping)
                else None
            )
            if environment_hash != health_hash:
                integrity.append("training environment hash differs from health PASS")
        manifest_environment = run_manifest.get("environment")
        if not isinstance(manifest_environment, Mapping) or (
            manifest_environment.get("training_environment_sha256") != environment_hash
        ):
            integrity.append("run-manifest training environment hash mismatch")
        elif manifest_environment.get("training_environment_identity") != identity:
            integrity.append("run-manifest training environment identity mismatch")

    resume_history = run_manifest.get("resume_history", [])
    if resume_history not in ([], None) and not isinstance(resume_history, list):
        integrity.append("run-manifest resume_history is malformed")
    if isinstance(resume_history, list) and any(
        not isinstance(item, Mapping)
        or item.get("checkpoint") != "last.pt"
        or item.get("training_environment_sha256") != environment_hash
        for item in resume_history
    ):
        integrity.append("resume history is not same-environment last.pt only")

    shared_hash = str(metrics.get("shared_weight_sha256", ""))
    split_hash = str(metrics.get("split_manifest_sha256", ""))
    shared_hash_value = shared_hash if _is_sha256(shared_hash) else None
    split_hash_value = split_hash if _is_sha256(split_hash) else None
    if shared_hash_value is None or run_manifest.get("shared_weight_sha256") != shared_hash:
        integrity.append("shared-weight identity is missing or inconsistent")
    if split_hash_value is None or run_manifest.get("split_manifest_sha256") != split_hash:
        integrity.append("split-manifest identity is missing or inconsistent")
    health = bound.get("health")
    health_source = health.get("source") if isinstance(health, Mapping) else None
    if not isinstance(health_source, Mapping) or (
        split_hash != health_source.get("split_manifest_sha256")
    ):
        integrity.append("training split manifest differs from health PASS")

    epoch_events, resumed_count, resume_failures = common._effective_epoch_events(events)
    integrity.extend(resume_failures)
    expected_resumes = len(resume_history) if isinstance(resume_history, list) else 0
    if resumed_count != expected_resumes:
        integrity.append("resumed-event count differs from run-manifest resume history")
    epochs = int(pilot_contract["epochs"])
    required_best = float(pilot_contract["minimum_best_validation_accuracy"])
    history, best, _convergence, history_failures = common._epoch_evidence(
        epoch_events,
        epochs=epochs,
        convergence_threshold=required_best,
    )
    integrity.extend(history_failures)
    for row_index, event in enumerate(epoch_events):
        train = event.get("train")
        val = event.get("val")
        if not _finite_tree(train) or not _finite_tree(val):
            integrity.append(f"epoch {row_index}: train/validation metrics are non-finite")

    activation = run_manifest.get("ta_activation_epoch_zero_based")
    expected_activation = math.ceil(
        expected_config.optimizer.epochs * expected_config.optimizer.ta_start_fraction
    )
    if activation != expected_activation or expected_activation != 5:
        integrity.append(f"adaptive activation epoch {activation!r} differs from frozen epoch 5")
    expected_states = [
        condition in ADAPTIVE_CONDITIONS and epoch >= expected_activation for epoch in range(epochs)
    ]
    observed_states = [event.get("ta_enabled") for event in epoch_events]
    if observed_states != expected_states:
        integrity.append("epoch ta_enabled sequence differs from the condition contract")

    adaptive_names = run_manifest.get("ta_parameter_names")
    group_manifest = run_manifest.get("optimizer_group_manifest")
    integrity.extend(
        _optimizer_group_failures(
            group_manifest,
            config=expected_config,
            adaptive_names=adaptive_names,
        )
    )
    integrity.extend(
        _orchestrator_failures(
            run_manifest.get("orchestrator_evidence"),
            block=block,
            bound=bound,
            environment_hash=environment_hash,
        )
    )

    summary: dict[str, Any] = {
        "best_validation_accuracy": None,
        "required_best_validation_accuracy": required_best,
        "minimum_gradient_coverage": None,
        "required_gradient_coverage": float(pilot_contract["required_gradient_coverage"]),
        "late_window_epochs": int(pilot_contract["late_window_epochs"]),
        "late_mean_validation_accuracy": None,
        "late_to_best_ratio": None,
        "minimum_late_to_best_ratio": float(pilot_contract["minimum_late_to_best_ratio"]),
    }
    if best is None or len(history) != epochs:
        integrity.append("best validation evidence cannot be reconstructed")
    else:
        best_accuracy = float(best["accuracy"])
        best_epoch = int(best["epoch"])
        best_loss = float(best["loss"])
        metric_checks = {
            "best_epoch": metrics.get("best_epoch") == best_epoch + 1,
            "best_val_accuracy": _same_number(metrics.get("best_val_accuracy"), best_accuracy),
            "best_val_loss": _same_number(metrics.get("best_val_loss"), best_loss),
        }
        for label, passed in metric_checks.items():
            if not passed:
                integrity.append(f"terminal metrics {label} differs from epoch evidence")
        summary, summary_integrity, summary_thresholds = _threshold_summary(
            history,
            best_accuracy=best_accuracy,
            required_best=required_best,
            required_gradient_coverage=float(pilot_contract["required_gradient_coverage"]),
            late_window_epochs=int(pilot_contract["late_window_epochs"]),
            minimum_late_to_best_ratio=float(pilot_contract["minimum_late_to_best_ratio"]),
        )
        integrity.extend(summary_integrity)
        thresholds.extend(summary_thresholds)
        for checkpoint_name, checkpoint_epoch, checkpoint_history in (
            ("best.pt", best_epoch, history[: best_epoch + 1]),
            ("last.pt", epochs - 1, history),
        ):
            checkpoint_path = run_dir / checkpoint_name
            integrity.extend(
                common._checkpoint_failures(
                    checkpoint_path,
                    label=checkpoint_name,
                    expected_epoch=checkpoint_epoch,
                    expected_best=best,
                    expected_history=checkpoint_history,
                    expected_config_hash=expected_hash,
                    expected_environment_hash=environment_hash,
                    protocol=protocol,
                    expected_output_dir=expected_config.runtime.output_dir,
                    actual_output_root=block.pilot_output_root,
                    artifact_config_normalizer=(
                        lambda raw: canonicalize_v7_artifact_run_mapping(
                            raw,
                            protocol,
                            project_root=repository_root,
                            expected_output_dir=expected_config.runtime.output_dir,
                            expected_execution_device=(
                                V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE
                            ),
                        )
                        if allow_device_execution_equivalence
                        else None
                    ),
                )
            )
            integrity.extend(
                _checkpoint_optimizer_failures(
                    checkpoint_path,
                    label=checkpoint_name,
                    config=expected_config,
                    adaptive_names=adaptive_names,
                )
            )
            if (
                allow_device_execution_equivalence
                and resolved_execution_hash is not None
            ):
                integrity.extend(
                    _checkpoint_recovery_execution_identity_failures(
                        checkpoint_path,
                        label=checkpoint_name,
                        protocol=protocol,
                        repository_root=repository_root,
                        expected_config=expected_config,
                        reference_raw=resolved_artifact_raw,
                        reference_execution_hash=resolved_execution_hash,
                    )
                )
    return (
        {
            "run_id": run_id,
            "condition": condition,
            **summary,
            "training_environment_sha256": environment_hash_value,
            "shared_weight_sha256": shared_hash_value,
            "split_manifest_sha256": split_hash_value,
            "adaptive_parameter_names": adaptive_names,
            "optimizer_group_manifest": group_manifest,
            "integrity_failures": integrity,
            "threshold_failures": thresholds,
        },
        environment_hash_value,
        shared_hash_value,
        split_hash_value,
    )


def _aggregate_metrics_csv_failures(output_root: Path, expected_run_ids: set[str]) -> list[str]:
    path = output_root / "seed_metrics.csv"
    if not path.is_file():
        return ["pilot aggregate seed_metrics.csv is missing"]
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error) as exc:
        return [f"cannot read pilot seed_metrics.csv: {exc}"]
    observed = [str(row.get("run_id")) for row in rows]
    if len(observed) != len(expected_run_ids) or set(observed) != expected_run_ids:
        return ["pilot seed_metrics.csv does not contain each of the six runs exactly once"]
    if any(row.get("status") != "complete" for row in rows):
        return ["pilot seed_metrics.csv contains a non-complete run"]
    return []


def validate_pilot(
    *,
    protocol_path: str | Path,
    config_dir: str | Path,
    repository_root: str | Path = PROJECT_ROOT,
    allow_device_execution_equivalence: bool = False,
) -> dict[str, Any]:
    """Validate the complete V7 pilot and return one fail-closed verdict."""

    root = Path(repository_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    config_dir = Path(config_dir).resolve()
    protocol = load_protocol(protocol_path)
    if protocol.get("protocol_version") != 7:
        raise PilotValidationError("This validator accepts only protocol_version 7")
    require_v7_author_freeze(protocol, repository_root=root)
    artifact_paths = artifact_paths_for_protocol(protocol)
    expected_protocol_path = (root / artifact_paths["protocol"]).resolve()
    expected_config_dir = (root / artifact_paths["pilot_matrix"]).resolve()
    if protocol_path != expected_protocol_path:
        raise PilotValidationError(
            f"Protocol path differs from the V7 artifact binding: {protocol_path}"
        )
    if config_dir != expected_config_dir:
        raise PilotValidationError(
            f"Config directory differs from the V7 artifact binding: {config_dir}"
        )
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise PilotValidationError("Protocol has no pilot_acceptance mapping")
    pilot_contract = acceptance.get("pilot")
    if not isinstance(pilot_contract, Mapping):
        raise PilotValidationError("Protocol V7 pilot acceptance mapping is missing")
    if int(pilot_contract.get("epochs", 0)) != 120:
        raise PilotValidationError("The V7 pilot must be exactly 120 epochs")

    rows, expected, matrix_failures, manifest_path, csv_path = _load_generated_matrix(
        protocol=protocol,
        protocol_path=protocol_path,
        config_dir=config_dir,
        repository_root=root,
    )
    block = resolve_pilot_block(protocol, repository_root=root)
    bound, evidence_failures = _bound_evidence(
        protocol=protocol,
        protocol_path=protocol_path,
        config_dir=config_dir,
        repository_root=root,
        block=block,
        rows=rows,
    )
    integrity = [*matrix_failures, *evidence_failures]
    thresholds: list[str] = []
    expected_run_ids = {config.runtime.run_id for config in expected.values()}
    if block.pilot_output_root.is_dir():
        observed_run_ids = {
            path.name
            for path in block.pilot_output_root.iterdir()
            if path.is_dir() and path.name.startswith("E9_")
        }
        if observed_run_ids != expected_run_ids:
            integrity.append(
                "pilot run directories differ from the exact six-run block: "
                f"observed={sorted(observed_run_ids)} "
                f"expected={sorted(expected_run_ids)}"
            )
        integrity.extend(_aggregate_metrics_csv_failures(block.pilot_output_root, expected_run_ids))
    else:
        observed_run_ids = set()
        integrity.append(f"pilot results root is missing: {block.pilot_output_root}")
    if allow_device_execution_equivalence:
        integrity.extend(
            _matrix_recovery_launch_failures(
                block.pilot_output_root,
                expected_run_ids=expected_run_ids,
                bound=bound,
            )
        )
    unexpected_final_tests = list(block.pilot_output_root.glob("**/final_test.json"))
    if unexpected_final_tests:
        integrity.append("pilot output contains forbidden final_test.json artifacts")

    rows_by_condition = {str(row.get("condition")): row for row in rows}
    runs: dict[str, Any] = {}
    environment_hashes: set[str] = set()
    shared_hashes: set[str] = set()
    split_hashes: set[str] = set()
    for condition in V7_ACTIVE_CONDITIONS:
        row = rows_by_condition.get(condition)
        config = expected.get(condition)
        if row is None or config is None:
            integrity.append(f"missing matrix row/config for {condition}")
            continue
        report, environment_hash, shared_hash, split_hash = _validate_run(
            protocol=protocol,
            protocol_path=protocol_path,
            config_dir=config_dir,
            repository_root=root,
            block=block,
            row=row,
            expected_config=config,
            bound=bound,
            pilot_contract=pilot_contract,
            allow_device_execution_equivalence=allow_device_execution_equivalence,
        )
        runs[condition] = report
        integrity.extend(f"{condition}: {failure}" for failure in report["integrity_failures"])
        thresholds.extend(f"{condition}: {failure}" for failure in report["threshold_failures"])
        if environment_hash is not None:
            environment_hashes.add(environment_hash)
        if shared_hash is not None:
            shared_hashes.add(shared_hash)
        if split_hash is not None:
            split_hashes.add(split_hash)
    if len(environment_hashes) != 1:
        integrity.append(
            "training environments are not identical across the six conditions: "
            f"{sorted(environment_hashes)}"
        )
    if len(shared_hashes) != 1:
        integrity.append(
            "shared-weight hashes are not identical across the six conditions: "
            f"{sorted(shared_hashes)}"
        )
    if len(split_hashes) != 1:
        integrity.append(
            "split-manifest hashes are not identical across the six conditions: "
            f"{sorted(split_hashes)}"
        )

    if integrity:
        status, decision, exit_code = "INVALID", INVALID_DECISION, 2
    elif thresholds:
        status, decision, exit_code = "FAIL", FAIL_DECISION, 1
    else:
        status, decision, exit_code = "PASS", PASS_DECISION, 0
    environment_hash = next(iter(environment_hashes)) if len(environment_hashes) == 1 else None
    dataset_report = {
        "status": status,
        "pass": status == "PASS",
        "seed": block.pilot_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "block_hash": block.block_hash,
        "results_root": artifact_path_reference(block.pilot_output_root, root),
        "health_report_path": bound.get("health_path"),
        "health_report_sha256": bound.get("health_sha256"),
        "attempt_receipt_path": bound.get("attempt_receipt_path"),
        "attempt_receipt_sha256": bound.get("attempt_receipt_sha256"),
        "pilot_plan_path": bound.get("plan_path"),
        "pilot_plan_sha256": bound.get("plan_sha256"),
        "environment_sha256": environment_hash,
        "shared_weight_sha256": (next(iter(shared_hashes)) if len(shared_hashes) == 1 else None),
        "split_manifest_sha256": (next(iter(split_hashes)) if len(split_hashes) == 1 else None),
        "runs": runs,
        "integrity_failures": integrity,
        "threshold_failures": thresholds,
    }
    recovery = bound.get("health_compatibility_recovery")
    if recovery is not None:
        dataset_report["health_compatibility_recovery"] = recovery
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": 7,
        "artifact_class": ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "status": status,
        "pass": status == "PASS",
        "decision": decision,
        "exit_code": exit_code,
        "validated_at": utc_now(),
        "protocol_path": artifact_path_reference(protocol_path, root),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_hash": stable_hash(protocol),
        "acceptance_hash": stable_hash(dict(acceptance)),
        "config_dir": artifact_path_reference(config_dir, root),
        "matrix_manifest_sha256": sha256_file(manifest_path),
        "run_manifest_csv_sha256": (sha256_file(csv_path) if csv_path.is_file() else None),
        "required_epochs": int(pilot_contract["epochs"]),
        "thresholds": {
            "minimum_best_validation_accuracy": pilot_contract["minimum_best_validation_accuracy"],
            "required_gradient_coverage": pilot_contract["required_gradient_coverage"],
            "late_window_epochs": pilot_contract["late_window_epochs"],
            "minimum_late_to_best_ratio": pilot_contract["minimum_late_to_best_ratio"],
        },
        "training_environment_sha256": environment_hash,
        "failure_action": protocol["pilot_acceptance"]["run_handling"]["failure_action"],
        "fallback": None,
        "datasets": {block.dataset: dataset_report},
        "integrity_failures": integrity,
        "threshold_failures": thresholds,
    }
    if recovery is not None:
        report["health_compatibility_recovery"] = recovery
    return report


def _device_recovery_release_binding(repository_root: Path) -> dict[str, Any]:
    """Bind the second, record-only seal without weakening the health seal."""
    try:
        return validate_pilot_device_recovery_release(repository_root)
    except PilotV7Error as exc:
        raise PilotValidationError(str(exc)) from exc


def _device_recovery_report(
    *,
    recovered: Mapping[str, Any],
    legacy: Mapping[str, Any],
    original_path: Path,
    original: Mapping[str, Any],
    output: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Build the non-reportable sidecar after exact legacy reproduction."""

    if sha256_file(original_path) != V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256:
        raise PilotValidationError(
            "Canonical INVALID SHA-256 is not the authorized V7 device incident"
        )
    if original.get("status") != "INVALID" or original.get("pass") is not False:
        raise PilotValidationError("Original V7 pilot validation is not the canonical INVALID")
    if original.get("decision") != INVALID_DECISION:
        raise PilotValidationError("Original V7 pilot validation has an unexpected decision")
    if original.get("threshold_failures") != []:
        raise PilotValidationError("Original INVALID contains threshold failures")
    if _report_without_timestamp(original) != _report_without_timestamp(legacy):
        raise PilotValidationError(
            "Legacy validation does not exactly reproduce the canonical INVALID"
        )
    expected_failures = [
        f"{condition}: v7 run runtime.device must be frozen as "
        f"{V7_PILOT_DEVICE_RECOVERY_FROZEN_DEVICE!r}, got "
        f"{V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE!r}"
        for condition in V7_ACTIVE_CONDITIONS
    ]
    if list(original.get("integrity_failures", ()))[: len(expected_failures)] != expected_failures:
        raise PilotValidationError("Original INVALID does not contain the authorized device mismatch")
    if recovered.get("status") != "PASS" or recovered.get("pass") is not True:
        raise PilotValidationError("Device recovery audit did not establish an aggregate PASS")
    if recovered.get("protocol_hash") != V7_PILOT_DEVICE_RECOVERY_PROTOCOL_HASH:
        raise PilotValidationError("Recovered validation has the wrong frozen protocol hash")
    if recovered.get("training_environment_sha256") is None:
        raise PilotValidationError("Recovered validation has no shared training environment")
    release = _device_recovery_release_binding(repository_root)
    health = recovered.get("health_compatibility_recovery")
    if not isinstance(health, Mapping):
        raise PilotValidationError("Recovered validation has no sealed health recovery binding")
    if health.get("recovery_commit") != V7_HEALTH_RECOVERY_SEAL_COMMIT:
        raise PilotValidationError("Recovered validation is not bound to the fixed health seal")
    return {
        "schema_version": 1,
        "artifact_class": DEVICE_RECOVERY_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": DEVICE_RECOVERY_DECISION,
        "exit_code": 0,
        "validated_at": recovered.get("validated_at"),
        "protocol_hash": recovered.get("protocol_hash"),
        "acceptance_hash": recovered.get("acceptance_hash"),
        "pilot_execution_commit": V7_HEALTH_RECOVERY_SEAL_COMMIT,
        "recovery_validator_commit": release["recovery_commit"],
        "release_delta": release,
        "recovery_validator": {
            "path": artifact_path_reference(Path(__file__).resolve(), repository_root),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "original_validation": {
            "path": artifact_path_reference(original_path, repository_root),
            "sha256": sha256_file(original_path),
            "status": "INVALID",
            "decision": original.get("decision"),
            "legacy_reproduction": "EXACT_EXCEPT_VALIDATED_AT",
            "legacy_reproduction_sha256": stable_hash(_report_without_timestamp(legacy)),
        },
        "health_compatibility_recovery": dict(health),
        "recovered_validation": dict(recovered),
        "output": artifact_path_reference(output, repository_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "protocol_v7_mechanism.yaml",
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recover-device-execution-equivalence",
        type=Path,
        help=(
            "Write the one fixed V7 sidecar recovery beside the immutable canonical "
            "INVALID; requires --output"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol_path = args.protocol.resolve()
        protocol = load_protocol(protocol_path)
        acceptance = protocol.get("pilot_acceptance")
        if not isinstance(acceptance, Mapping):
            raise PilotValidationError("Protocol has no pilot_acceptance mapping")
        canonical_output = (PROJECT_ROOT / str(acceptance["validation_output"])).resolve()
        recovery_mode = args.recover_device_execution_equivalence is not None
        if recovery_mode:
            invalid_path = args.recover_device_execution_equivalence.resolve()
            expected_invalid = (PROJECT_ROOT / V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION).resolve()
            if canonical_output != expected_invalid or invalid_path != canonical_output:
                raise PilotValidationError(
                    "--recover-device-execution-equivalence must name the canonical "
                    "V7 validation output"
                )
            if args.output is None:
                raise PilotValidationError("V7 device recovery requires --output")
            output = args.output.resolve()
            expected_output = (PROJECT_ROOT / V7_PILOT_DEVICE_RECOVERY_OUTPUT).resolve()
            if output != expected_output:
                raise PilotValidationError(
                    "V7 device recovery output must be the fixed adjacent "
                    "validation_device_recovery.json artifact"
                )
        else:
            output = canonical_output
            if args.output is not None and args.output.resolve() != output:
                raise PilotValidationError(
                    "--output cannot override pilot_acceptance.validation_output"
                )
        if output.exists():
            raise PilotValidationError(
                "V7 pilot validation output already exists; refusing overwrite"
            )
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else (PROJECT_ROOT / artifact_paths_for_protocol(protocol)["pilot_matrix"]).resolve()
        )
        report = validate_pilot(
            protocol_path=protocol_path,
            config_dir=config_dir,
            repository_root=PROJECT_ROOT,
            allow_device_execution_equivalence=recovery_mode,
        )
        if recovery_mode:
            original = _read_json(invalid_path, "original V7 pilot validation")
            legacy = validate_pilot(
                protocol_path=protocol_path,
                config_dir=config_dir,
                repository_root=PROJECT_ROOT,
                allow_device_execution_equivalence=False,
            )
            stored = _device_recovery_report(
                recovered=report,
                legacy=legacy,
                original_path=invalid_path,
                original=original,
                output=output,
                repository_root=PROJECT_ROOT,
            )
        else:
            stored = report
        exclusive_create_json(output, stored)
    except Exception as exc:  # noqa: BLE001 - one fail-closed CLI marker.
        print(
            f"V7_PILOT_VALIDATION_ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    marker = (
        "V7_PILOT_VALIDATION_DEVICE_RECOVERY_PASS"
        if recovery_mode
        else f"V7_PILOT_VALIDATION_{report['status']}"
    )
    print(marker)
    print(f"NON_REPORTING_OUTPUT={output}")
    return int(stored["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
