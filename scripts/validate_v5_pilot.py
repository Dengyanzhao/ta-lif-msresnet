#!/usr/bin/env python3
"""Validate both independent non-reportable TA-LIF-only v5 pilot blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)
SCRIPTS_PATH = str(SCRIPTS_ROOT)
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(1, SCRIPTS_PATH)

import validate_v3_pilot as common

from talif_msresnet.config import (
    RunConfig,
    artifact_paths_for_protocol,
    generate_v5_pilot_matrix,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.pilot_v5 import (
    ATTEMPT_STATUS,
    HEALTH_ARTIFACT_CLASS,
    HEALTH_DECISION_BASIS,
    HEALTH_SEED_DISPOSITION,
    LEARNING_METRICS_ROLE,
    PILOT_SEED_DISPOSITION,
    REPORTING_ELIGIBILITY,
    V5_PILOT_CONDITIONS,
    V5_PILOT_DATASETS,
    PilotBlock,
    PilotV5Error,
    exclusive_create_json,
    expected_pilot_configs,
    expected_pilot_plan_payload,
    require_v5_author_freeze,
    resolve_pilot_block,
    validate_attempt_receipt,
    validate_health_report,
)
from talif_msresnet.utils import sha256_file, stable_hash, utc_now

SCHEMA_VERSION = 1
ARTIFACT_CLASS = "NON_REPORTABLE_V5_TALIF_ONLY_PILOT_ACCEPTANCE"
PASS_DECISION = "ACCEPT_V5_TALIF_ONLY_120_EPOCH_PILOTS_RELEASE_FORMAL_FREEZE"
FAIL_DECISION = "BLOCK_V5_AND_REQUIRE_NEW_PROTOCOL_AND_UNUSED_PILOT_SEEDS"
INVALID_DECISION = "BLOCK_V5_AND_INVESTIGATE_PILOT_EVIDENCE_INTEGRITY"
RECOVERY_SCHEMA_VERSION = 1
RECOVERY_ARTIFACT_CLASS = "NON_REPORTABLE_V5_PILOT_VALIDATION_RECOVERY"
RECOVERY_DECISION = "RECOVER_V5_PILOT_PASS_AFTER_VALIDATOR_PATH_EQUIVALENCE_FIX"
RECOVERY_OUTPUT_NAME = "validation_path_recovery.json"
RECOVERY_RELEASE_RECORD = "V5_VALIDATION_RECOVERY_RELEASE.json"
RECOVERY_ORIGINAL_VALIDATION_SHA256 = (
    "5cc85680923b6eee1e454dcf4cf7667f4271dc531d5837a529e57bfcaccd7e86"
)
RECOVERY_PILOT_EXECUTION_COMMIT = "6eeadd6389677347fe46ffa8d3bdec8c75455b44"
RECOVERY_PROTOCOL_HASH = "a5b2a664de5142438061f479a37f3b55401499104abb28ee3cf1d674ec1d4527"
RECOVERY_TRAINING_ENVIRONMENT_SHA256 = (
    "f3b837c1554615bb97af91183ec450bf8f4bb43610ad47123aa1f161791d4fb6"
)
RECOVERY_ALLOWED_CHANGED_PATHS = (
    "V5_TALIF_ONLY_RUNBOOK.md",
    "scripts/create_freeze_manifest.py",
    "scripts/validate_v3_pilot.py",
    "scripts/validate_v5_pilot.py",
    "src/talif_msresnet/freeze.py",
    "tests/test_v5_execution_gates.py",
    "tests/test_v5_release_flow.py",
    "tests/test_validate_v5_pilot.py",
)


class PilotValidationError(RuntimeError):
    """Raised when the validator cannot establish the v5 pilot contract."""


def _git_output(repository_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PilotValidationError(
            f"Cannot inspect the v5 recovery release (git {' '.join(arguments)}): "
            f"{detail or 'no details'}"
        )
    return completed.stdout.strip()


def _git_bytes(repository_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise PilotValidationError(
            f"Cannot inspect the v5 recovery release (git {' '.join(arguments)}): "
            f"{detail or 'no details'}"
        )
    return completed.stdout


def _recovery_release_binding(repository_root: Path) -> dict[str, Any]:
    commit = _git_output(repository_root, "rev-parse", "HEAD")
    if (
        len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise PilotValidationError("Current recovery release commit is not a full Git hash")
    record_path = repository_root / RECOVERY_RELEASE_RECORD
    record = _read_json(record_path, "v5 recovery release record")
    implementation_commit = str(record.get("implementation_commit", ""))
    if (
        len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise PilotValidationError("Recovery release record has no implementation commit")
    parents = _git_output(repository_root, "rev-list", "--parents", "-n", "1", commit).split()
    if parents != [commit, implementation_commit]:
        raise PilotValidationError(
            "Recovery release must be the one-parent seal of its implementation commit"
        )
    _git_output(
        repository_root,
        "merge-base",
        "--is-ancestor",
        RECOVERY_PILOT_EXECUTION_COMMIT,
        implementation_commit,
    )
    if _git_output(repository_root, "diff", "--name-only", f"{commit}..HEAD"):
        raise PilotValidationError("Recovery release validation must run at the sealed HEAD")
    if _git_output(repository_root, "status", "--porcelain", "--untracked-files=no"):
        raise PilotValidationError("Recovery requires a clean tracked Git worktree")
    implementation_changes = tuple(
        line.strip()
        for line in _git_output(
            repository_root,
            "diff",
            "--name-status",
            "--find-renames",
            RECOVERY_PILOT_EXECUTION_COMMIT,
            implementation_commit,
        ).splitlines()
        if line.strip()
    )
    expected_changes = tuple(f"M\t{path}" for path in RECOVERY_ALLOWED_CHANGED_PATHS)
    if implementation_changes != expected_changes:
        raise PilotValidationError(
            "Recovery implementation delta is not the exact modification-only allowlist: "
            f"observed={list(implementation_changes)} expected={list(expected_changes)}"
        )
    seal_changes = tuple(
        line.strip()
        for line in _git_output(
            repository_root,
            "diff",
            "--name-status",
            "--find-renames",
            implementation_commit,
            commit,
        ).splitlines()
        if line.strip()
    )
    if seal_changes != (f"A\t{RECOVERY_RELEASE_RECORD}",):
        raise PilotValidationError(
            "Recovery release commit must add only the release record"
        )
    implementation_tree = _git_output(
        repository_root, "rev-parse", f"{implementation_commit}^{{tree}}"
    )
    expected_record_top = {
        "schema": "ta-lif-msresnet-v5-validation-recovery-release-v1",
        "base_pilot_commit": RECOVERY_PILOT_EXECUTION_COMMIT,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "original_validation_sha256": RECOVERY_ORIGINAL_VALIDATION_SHA256,
        "protocol_hash": RECOVERY_PROTOCOL_HASH,
        "training_environment_sha256": RECOVERY_TRAINING_ENVIRONMENT_SHA256,
        "allowed_changed_paths": list(RECOVERY_ALLOWED_CHANGED_PATHS),
    }
    mismatches = [
        key for key, expected in expected_record_top.items() if record.get(key) != expected
    ]
    source_sha256 = record.get("implementation_file_sha256")
    if not isinstance(source_sha256, Mapping) or set(source_sha256) != set(
        RECOVERY_ALLOWED_CHANGED_PATHS
    ):
        mismatches.append("implementation_file_sha256")
        source_sha256 = {}
    for path in RECOVERY_ALLOWED_CHANGED_PATHS:
        expected_sha256 = source_sha256.get(path)
        committed_sha256 = hashlib.sha256(
            _git_bytes(repository_root, "show", f"{implementation_commit}:{path}")
        ).hexdigest()
        if not isinstance(expected_sha256, str) or committed_sha256 != expected_sha256:
            mismatches.append(f"implementation_file_sha256.{path}")
    record_sha256 = record.get("record_sha256")
    record_without_hash = dict(record)
    record_without_hash.pop("record_sha256", None)
    if record_sha256 != stable_hash(record_without_hash):
        mismatches.append("record_sha256")
    if mismatches:
        raise PilotValidationError(
            "Recovery release record is not the sealed reviewed implementation: "
            + ", ".join(mismatches)
        )
    return {
        "base_commit": RECOVERY_PILOT_EXECUTION_COMMIT,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "recovery_commit": commit,
        "allowed_changed_paths": list(RECOVERY_ALLOWED_CHANGED_PATHS),
        "observed_changes": list(implementation_changes),
        "release_record": RECOVERY_RELEASE_RECORD,
        "release_record_sha256": sha256_file(record_path),
        "tracked_clean": True,
    }


def _report_without_timestamp(report: Mapping[str, Any]) -> dict[str, Any]:
    comparable = dict(report)
    comparable.pop("validated_at", None)
    return comparable


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotValidationError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotValidationError(f"{label} must be a JSON object: {path}")
    return value


def _load_generated_matrix(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    repository_root: Path,
) -> tuple[list[Mapping[str, Any]], dict[tuple[str, str], RunConfig], list[str], Path, Path]:
    failures: list[str] = []
    manifest_path = config_dir / "matrix_manifest.json"
    csv_path = config_dir / "run_manifest.csv"
    manifest = _read_json(manifest_path, "v5 pilot matrix manifest")
    raw_rows = manifest.get("runs")
    if not isinstance(raw_rows, list):
        raise PilotValidationError("matrix manifest runs must be a list")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        failures.append("matrix manifest contains a non-object run")
    expected_raw = generate_v5_pilot_matrix(protocol)
    expected_configs = [validate_run_mapping(row, protocol) for row in expected_raw]
    expected_by_key = {
        (config.data.dataset, config.model.condition): config for config in expected_configs
    }
    expected_order = [
        (config.data.dataset, config.model.condition, config.runtime.run_id)
        for config in expected_configs
    ]
    observed_order = [
        (str(row.get("dataset")), str(row.get("condition")), str(row.get("run_id"))) for row in rows
    ]
    if observed_order != expected_order:
        failures.append("matrix manifest is not the ordered two-dataset C1/C2 v5 pilot")
    expected_seeds = list(dict.fromkeys(config.runtime.seed for config in expected_configs))
    checks = {
        "protocol": artifact_path_reference(protocol_path, repository_root),
        "protocol_hash": stable_hash(protocol),
        "run_count": 4,
        "conditions": list(V5_PILOT_CONDITIONS),
        "seeds": expected_seeds,
        "matrix_hash": stable_hash(expected_raw),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            failures.append(f"matrix manifest {key} mismatch")
    if not csv_path.is_file():
        failures.append("generated run_manifest.csv is missing")
    else:
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise PilotValidationError(
                f"Cannot read generated run manifest {csv_path}: {exc}"
            ) from exc
        normalized = [{key: str(value) for key, value in row.items()} for row in rows]
        if csv_rows != normalized:
            failures.append("run_manifest.csv differs from matrix_manifest.json")
    return rows, expected_by_key, failures, manifest_path, csv_path


def _basic_health_failures(
    report: Mapping[str, Any], *, block: PilotBlock, receipt_sha256: str
) -> list[str]:
    expected = {
        "schema_version": 5,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "status": "PASS",
        "pass": True,
        "thresholds_evaluated": [],
        "decision_basis": HEALTH_DECISION_BASIS,
        "learning_metrics_role": LEARNING_METRICS_ROLE,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "dataset": block.dataset,
        "seed": block.health_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "development_probe_hash": block.development_probe_hash,
        "block_hash": block.block_hash,
        "tracked_clean": True,
        "attempt_receipt_sha256": receipt_sha256,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
        "integrity_anomalies": [],
        "failures": [],
    }
    return [
        f"health report {key} mismatch"
        for key, value in expected.items()
        if report.get(key) != value
    ]


def _bound_block_evidence(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    repository_root: Path,
    block: PilotBlock,
    manifest_rows: Sequence[Mapping[str, Any]],
    strict_health_validation: bool,
    evidence_git_commit: str | None,
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
    reference_config = expected_pilot_configs(protocol, block.dataset)[0]
    try:
        receipt = validate_attempt_receipt(
            block.attempt_receipt,
            block=block,
            protocol_path=protocol_path,
            repository_root=repository_root,
            reference_config=reference_config,
        )
    except PilotV5Error as exc:
        receipt = {}
        failures.append(str(exc))
    c1_rows = [
        row
        for row in manifest_rows
        if row.get("dataset") == block.dataset and row.get("condition") == "C1"
    ]
    if len(c1_rows) != 1:
        failures.append(f"{block.dataset}: no unique C1 reference config")
    elif receipt:
        row = c1_rows[0]
        config_path = (config_dir / str(row.get("config_file", ""))).resolve()
        expected_receipt = {
            "reference_config": artifact_path_reference(config_path, repository_root),
            "reference_config_sha256": (
                sha256_file(config_path) if config_path.is_file() else None
            ),
            "reference_config_hash": row.get("config_hash"),
            "reference_config_seed": block.pilot_seed,
            "status": ATTEMPT_STATUS,
        }
        for key, value in expected_receipt.items():
            if receipt.get(key) != value:
                failures.append(f"attempt receipt {key} mismatch")
    report = _read_json(block.health_output, "v5 health report")
    receipt_sha256 = sha256_file(block.attempt_receipt)
    failures.extend(_basic_health_failures(report, block=block, receipt_sha256=receipt_sha256))
    if strict_health_validation:
        try:
            report = validate_health_report(
                block.health_output,
                protocol_path,
                block.dataset,
                repository_root=repository_root,
                expected_git_commit=evidence_git_commit,
                require_current_tracked_clean=evidence_git_commit is None,
            )
        except PilotV5Error as exc:
            failures.append(f"canonical health validation failed: {exc}")
    plan = _read_json(block.pilot_plan, "v5 pilot plan")
    try:
        expected_plan = expected_pilot_plan_payload(
            block=block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=manifest_rows,
            health_report=report,
            repository_root=repository_root,
        )
    except PilotV5Error as exc:
        failures.append(str(exc))
    else:
        if plan != expected_plan:
            failures.append("pilot plan differs from the exact v5 dataset-bound payload")
    return {
        "health": report,
        "health_path": artifact_path_reference(block.health_output, repository_root),
        "health_sha256": sha256_file(block.health_output),
        "attempt_receipt_path": artifact_path_reference(block.attempt_receipt, repository_root),
        "attempt_receipt_sha256": receipt_sha256,
        "plan_path": artifact_path_reference(block.pilot_plan, repository_root),
        "plan_sha256": sha256_file(block.pilot_plan),
    }, failures


def _v5_threshold_summary(
    events: Sequence[Mapping[str, Any]],
    *,
    epochs: int,
    formal_convergence_threshold: float,
    warmup_epochs: int,
    required_best_accuracy: float,
    late_window_epochs: int,
    minimum_late_mean: float,
    minimum_late_to_best_ratio: float,
    minimum_residual_coverage: float,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Reconstruct and apply the frozen v5 pilot-only acceptance statistics."""

    integrity: list[str] = []
    thresholds: list[str] = []
    effective, _resume_count, resume_failures = common._effective_epoch_events(events)
    integrity.extend(resume_failures)
    history, best, _convergence, history_failures = common._epoch_evidence(
        effective,
        epochs=epochs,
        convergence_threshold=formal_convergence_threshold,
    )
    integrity.extend(history_failures)
    if best is None or len(history) != epochs:
        return (
            {
                "best_validation_accuracy": None,
                "late_mean_validation_accuracy": None,
                "late_to_best_ratio": None,
                "minimum_post_warmup_residual_gradient_coverage": None,
            },
            integrity,
            thresholds,
        )
    best_accuracy = float(best["accuracy"])
    late_values = [float(row["val_accuracy"]) for row in history[-late_window_epochs:]]
    if len(late_values) != late_window_epochs:
        integrity.append("late validation window is incomplete")
        late_mean = float("nan")
    else:
        late_mean = sum(late_values) / len(late_values)
    late_ratio = late_mean / best_accuracy if best_accuracy > 0.0 else math.inf

    post_warmup = history[warmup_epochs:]
    residual_coverages: list[float] = []
    for row in post_warmup:
        zero_fraction = common._finite_number(row.get("all_zero_block_gradient_batch_fraction"))
        if zero_fraction is None or not 0.0 <= zero_fraction <= 1.0:
            integrity.append(
                f"epoch {row.get('epoch')}: all-zero residual-gradient fraction is invalid"
            )
            continue
        residual_coverages.append(1.0 - zero_fraction)
    expected_post_warmup = epochs - warmup_epochs
    if len(residual_coverages) != expected_post_warmup:
        integrity.append(
            "post-warmup residual-gradient evidence is incomplete: "
            f"{len(residual_coverages)} of {expected_post_warmup} epochs"
        )
    minimum_observed_coverage = min(residual_coverages) if residual_coverages else float("nan")
    if best_accuracy < required_best_accuracy:
        thresholds.append(
            f"best validation accuracy {best_accuracy:.6f} is below {required_best_accuracy:.6f}"
        )
    if not math.isfinite(late_mean) or late_mean < minimum_late_mean:
        thresholds.append(
            f"final-{late_window_epochs}-epoch mean validation accuracy "
            f"{late_mean:.6f} is below {minimum_late_mean:.6f}"
        )
    if not math.isfinite(late_ratio) or late_ratio < minimum_late_to_best_ratio:
        thresholds.append(
            f"late-to-best accuracy ratio {late_ratio:.6f} is below "
            f"{minimum_late_to_best_ratio:.6f}"
        )
    if (
        not math.isfinite(minimum_observed_coverage)
        or minimum_observed_coverage < minimum_residual_coverage
    ):
        thresholds.append(
            "minimum post-warmup residual-gradient batch coverage "
            f"{minimum_observed_coverage:.6f} is below "
            f"{minimum_residual_coverage:.6f}"
        )
    return (
        {
            "best_validation_accuracy": best_accuracy,
            "required_best_validation_accuracy": required_best_accuracy,
            "late_window_epochs": late_window_epochs,
            "late_mean_validation_accuracy": late_mean,
            "minimum_late_mean_validation_accuracy": minimum_late_mean,
            "late_to_best_ratio": late_ratio,
            "minimum_late_to_best_ratio": minimum_late_to_best_ratio,
            "post_warmup_epoch_count": len(post_warmup),
            "residual_gradient_batch_coverage_definition": (
                "minimum across post-warmup epochs of one minus the fraction of "
                "training batches with all residual-block parameter gradients zero"
            ),
            "minimum_post_warmup_residual_gradient_coverage": minimum_observed_coverage,
            "required_post_warmup_residual_gradient_coverage": minimum_residual_coverage,
        },
        integrity,
        thresholds,
    )


def _validate_run(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    block: PilotBlock,
    row: Mapping[str, Any],
    expected_config: RunConfig,
    health: Mapping[str, Any],
    epochs: int,
    schedule: Mapping[str, Any],
    maximum_ratio: float,
    allow_equivalent_absolute_output_dir: bool,
) -> tuple[dict[str, Any], str | None, str | None, str | None]:
    configured_thresholds = expected_config.analysis.get("validation_accuracy_thresholds")
    if not isinstance(configured_thresholds, Mapping):
        formal_threshold = float("nan")
    else:
        formal_threshold = float(configured_thresholds[block.dataset])
    base_report, environment_hash, shared_hash, split_hash = common._validate_run(
        protocol=protocol,
        protocol_path=protocol_path,
        config_dir=config_dir,
        block=block,  # type: ignore[arg-type]
        row=row,
        expected_config=expected_config,
        health=health,
        epochs=epochs,
        accuracy_threshold=formal_threshold,
        maximum_ratio=maximum_ratio,
        allow_equivalent_absolute_output_dir=allow_equivalent_absolute_output_dir,
    )
    # The legacy helper reconstructs checkpoints and the formal convergence
    # fields.  Its two v3 acceptance failures are replaced by v5's dataset-
    # specific best, late-window, and gradient gates below.
    base_thresholds = [
        failure
        for failure in base_report.get("threshold_failures", [])
        if not str(failure).startswith("best validation accuracy ")
        and failure != "validation accuracy never reached convergence threshold"
    ]
    run_dir = block.pilot_output_root / expected_config.runtime.run_id
    try:
        events = common._read_events(run_dir / "events.jsonl")
    except Exception as exc:  # noqa: BLE001 - malformed evidence becomes INVALID
        base_report.setdefault("integrity_failures", []).append(str(exc))
        return base_report, environment_hash, shared_hash, split_hash
    required_best = float(schedule["required_best_validation_accuracy"][block.dataset])
    minimum_late = float(schedule["minimum_late_mean_accuracy"][block.dataset])
    summary, summary_integrity, summary_thresholds = _v5_threshold_summary(
        events,
        epochs=epochs,
        formal_convergence_threshold=formal_threshold,
        warmup_epochs=expected_config.optimizer.warmup_epochs,
        required_best_accuracy=required_best,
        late_window_epochs=int(schedule["late_window_epochs"]),
        minimum_late_mean=minimum_late,
        minimum_late_to_best_ratio=float(schedule["minimum_late_to_best_ratio"]),
        minimum_residual_coverage=float(schedule["minimum_post_warmup_residual_gradient_coverage"]),
    )
    base_report.update(summary)
    base_report["integrity_failures"] = [
        *base_report.get("integrity_failures", []),
        *summary_integrity,
    ]
    base_report["threshold_failures"] = [
        *base_thresholds,
        *summary_thresholds,
    ]
    return base_report, environment_hash, shared_hash, split_hash


def _cross_dataset_environment_gate(
    datasets_report: Mapping[str, Mapping[str, Any]],
) -> tuple[str | None, list[str]]:
    """Require one valid training environment identity for both dataset blocks."""

    observed = {
        dataset: (
            datasets_report[dataset].get("environment_sha256")
            if isinstance(datasets_report.get(dataset), Mapping)
            else None
        )
        for dataset in V5_PILOT_DATASETS
    }
    valid_hashes = [
        value
        for value in observed.values()
        if isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    ]
    if len(valid_hashes) == len(V5_PILOT_DATASETS) and len(set(valid_hashes)) == 1:
        return valid_hashes[0], []
    detail = ", ".join(f"{dataset}={observed[dataset]!r}" for dataset in V5_PILOT_DATASETS)
    return None, ["training environment hashes are not identical across dataset blocks: " + detail]


def validate_pilot(
    *,
    protocol_path: str | Path,
    config_dir: str | Path,
    repository_root: str | Path = PROJECT_ROOT,
    strict_health_validation: bool = True,
    evidence_git_commit: str | None = None,
    allow_equivalent_absolute_output_dir: bool = False,
) -> dict[str, Any]:
    """Validate both v5 C1/C2 blocks and return one aggregate verdict."""

    root = Path(repository_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    config_dir = Path(config_dir).resolve()
    protocol = load_protocol(protocol_path)
    if protocol.get("protocol_version") != 5:
        raise PilotValidationError("This validator accepts only protocol_version 5")
    require_v5_author_freeze(protocol, repository_root=root)
    artifact_paths = artifact_paths_for_protocol(protocol)
    expected_protocol_path = (root / artifact_paths["protocol"]).resolve()
    expected_config_dir = (root / artifact_paths["pilot_matrix"]).resolve()
    if protocol_path != expected_protocol_path:
        raise PilotValidationError(
            f"Protocol path differs from the v5 artifact binding: {protocol_path}"
        )
    if config_dir != expected_config_dir:
        raise PilotValidationError(
            f"Config directory differs from the v5 artifact binding: {config_dir}"
        )
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise PilotValidationError("Protocol has no pilot_acceptance mapping")
    schedule = acceptance.get("schedule")
    timing = acceptance.get("timing")
    if not isinstance(schedule, Mapping) or not isinstance(timing, Mapping):
        raise PilotValidationError("Protocol pilot schedule/timing mapping is missing")
    epochs = int(schedule.get("epochs", 0))
    if epochs != 120:
        raise PilotValidationError("The v5 pilot schedule must be exactly 120 epochs")
    if timing.get("epoch_time_ratio_conditions") != ["C2"]:
        raise PilotValidationError("The v5 pilot timing condition must be exactly C2")
    if timing.get("ta_state_source") != "events_jsonl_epoch_completed_ta_enabled":
        raise PilotValidationError("Unsupported v5 pilot TA timing state source")

    rows, expected_configs, matrix_failures, manifest_path, csv_path = _load_generated_matrix(
        protocol=protocol,
        protocol_path=protocol_path,
        config_dir=config_dir,
        repository_root=root,
    )
    aggregate_integrity = list(matrix_failures)
    aggregate_thresholds: list[str] = []
    datasets_report: dict[str, Any] = {}
    maximum_ratio = float(timing["maximum_ta_enabled_over_frozen_ratio"])

    for dataset in V5_PILOT_DATASETS:
        block = resolve_pilot_block(protocol, dataset, repository_root=root)
        block_rows = [row for row in rows if row.get("dataset") == dataset]
        block_integrity: list[str] = []
        block_thresholds: list[str] = []
        if [str(row.get("condition")) for row in block_rows] != ["C1", "C2"]:
            block_integrity.append("dataset block is not exactly ordered C1,C2")
        evidence, evidence_failures = _bound_block_evidence(
            protocol=protocol,
            protocol_path=protocol_path,
            config_dir=config_dir,
            repository_root=root,
            block=block,
            manifest_rows=rows,
            strict_health_validation=strict_health_validation,
            evidence_git_commit=evidence_git_commit,
        )
        block_integrity.extend(evidence_failures)
        health = evidence.get("health")
        if not isinstance(health, Mapping):
            health = {}
        expected_run_ids = {
            expected_configs[(dataset, condition)].runtime.run_id
            for condition in V5_PILOT_CONDITIONS
        }
        if block.pilot_output_root.is_dir():
            observed_run_ids = {
                path.name
                for path in block.pilot_output_root.iterdir()
                if path.is_dir() and path.name.startswith("E1_")
            }
        else:
            observed_run_ids = set()
            block_integrity.append(f"pilot results root is missing: {block.pilot_output_root}")
        if observed_run_ids != expected_run_ids:
            block_integrity.append(
                "pilot run directories differ from the exact two-run block: "
                f"observed={sorted(observed_run_ids)} expected={sorted(expected_run_ids)}"
            )

        runs: dict[str, Any] = {}
        environment_hashes: set[str] = set()
        shared_hashes: set[str] = set()
        split_hashes: set[str] = set()
        rows_by_condition = {str(row.get("condition")): row for row in block_rows}
        for condition in V5_PILOT_CONDITIONS:
            row = rows_by_condition.get(condition)
            expected = expected_configs[(dataset, condition)]
            if row is None:
                block_integrity.append(f"missing matrix row for {condition}")
                continue
            run_report, environment_hash, shared_hash, split_hash = _validate_run(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                block=block,
                row=row,
                expected_config=expected,
                health=health,
                epochs=epochs,
                schedule=schedule,
                maximum_ratio=maximum_ratio,
                allow_equivalent_absolute_output_dir=(
                    allow_equivalent_absolute_output_dir
                ),
            )
            runs[condition] = run_report
            block_integrity.extend(
                f"{condition}: {failure}" for failure in run_report["integrity_failures"]
            )
            block_thresholds.extend(
                f"{condition}: {failure}" for failure in run_report["threshold_failures"]
            )
            if environment_hash is not None:
                environment_hashes.add(environment_hash)
            if shared_hash is not None:
                shared_hashes.add(shared_hash)
            if split_hash is not None:
                split_hashes.add(split_hash)
        if len(environment_hashes) != 1:
            block_integrity.append(
                "training environments are not identical within the dataset block: "
                f"{sorted(environment_hashes)}"
            )
        if len(shared_hashes) != 1:
            block_integrity.append(
                "shared-weight hashes are not identical within the dataset block: "
                f"{sorted(shared_hashes)}"
            )
        if len(split_hashes) != 1:
            block_integrity.append(
                "split-manifest hashes are not identical within the dataset block: "
                f"{sorted(split_hashes)}"
            )
        block_status = "INVALID" if block_integrity else "FAIL" if block_thresholds else "PASS"
        datasets_report[dataset] = {
            "status": block_status,
            "pass": block_status == "PASS",
            "seed": block.pilot_seed,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "block_hash": block.block_hash,
            "results_root": artifact_path_reference(block.pilot_output_root, root),
            "health_report_path": evidence.get("health_path"),
            "health_report_sha256": evidence.get("health_sha256"),
            "attempt_receipt_path": evidence.get("attempt_receipt_path"),
            "attempt_receipt_sha256": evidence.get("attempt_receipt_sha256"),
            "pilot_plan_path": evidence.get("plan_path"),
            "pilot_plan_sha256": evidence.get("plan_sha256"),
            "environment_sha256": next(iter(environment_hashes), None)
            if len(environment_hashes) == 1
            else None,
            "shared_weight_sha256": next(iter(shared_hashes), None)
            if len(shared_hashes) == 1
            else None,
            "split_manifest_sha256": next(iter(split_hashes), None)
            if len(split_hashes) == 1
            else None,
            "runs": runs,
            "integrity_failures": block_integrity,
            "threshold_failures": block_thresholds,
        }
        aggregate_integrity.extend(f"{dataset}: {failure}" for failure in block_integrity)
        aggregate_thresholds.extend(f"{dataset}: {failure}" for failure in block_thresholds)

    training_environment_sha256, cross_dataset_environment_failures = (
        _cross_dataset_environment_gate(datasets_report)
    )
    aggregate_integrity.extend(cross_dataset_environment_failures)

    if aggregate_integrity:
        status, decision, exit_code = "INVALID", INVALID_DECISION, 2
    elif aggregate_thresholds:
        status, decision, exit_code = "FAIL", FAIL_DECISION, 1
    else:
        status, decision, exit_code = "PASS", PASS_DECISION, 0
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_version": 5,
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
        "run_manifest_csv_sha256": sha256_file(csv_path) if csv_path.is_file() else None,
        "required_epochs": epochs,
        "dataset_specific_thresholds": {
            "required_best_validation_accuracy": dict(
                schedule["required_best_validation_accuracy"]
            ),
            "minimum_late_mean_accuracy": dict(schedule["minimum_late_mean_accuracy"]),
            "late_window_epochs": schedule["late_window_epochs"],
            "minimum_late_to_best_ratio": schedule["minimum_late_to_best_ratio"],
            "minimum_post_warmup_residual_gradient_coverage": schedule[
                "minimum_post_warmup_residual_gradient_coverage"
            ],
        },
        "dataset_acceptance_thresholds_independent": True,
        "cross_dataset_training_environment_hash_equality_required": True,
        "cross_dataset_shared_weight_or_split_manifest_hash_equality_required": False,
        "training_environment_sha256": training_environment_sha256,
        "failure_action": schedule["failure_action"],
        "fallback": None,
        "datasets": datasets_report,
        "integrity_failures": aggregate_integrity,
        "threshold_failures": aggregate_thresholds,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "protocol_v5_talif_only.yaml",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v5_talif_only_pilot_generated",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--recover-from-invalid",
        type=Path,
        help=(
            "Append one recovery PASS beside an immutable canonical INVALID; "
            "requires --pilot-execution-commit and --output"
        ),
    )
    parser.add_argument(
        "--pilot-execution-commit",
        help="Original reviewed commit bound by the health and pilot evidence",
    )
    return parser


def _recovery_report(
    *,
    report: Mapping[str, Any],
    legacy_report: Mapping[str, Any],
    invalid_path: Path,
    invalid_report: Mapping[str, Any],
    output: Path,
    repository_root: Path,
    pilot_execution_commit: str,
) -> dict[str, Any]:
    invalid_sha256 = sha256_file(invalid_path)
    if invalid_sha256 != RECOVERY_ORIGINAL_VALIDATION_SHA256:
        raise PilotValidationError(
            "Canonical INVALID SHA-256 is not the authorized v5 recovery incident"
        )
    if report.get("status") != "PASS" or report.get("pass") is not True:
        raise PilotValidationError("Recovery audit did not establish an aggregate PASS")
    if invalid_report.get("status") != "INVALID" or invalid_report.get("pass") is not False:
        raise PilotValidationError("--recover-from-invalid must name the original INVALID report")
    if _report_without_timestamp(invalid_report) != _report_without_timestamp(legacy_report):
        raise PilotValidationError(
            "Legacy path behavior does not exactly reproduce the original INVALID"
        )
    if invalid_report.get("protocol_hash") != report.get("protocol_hash"):
        raise PilotValidationError("Original INVALID and recovery audit use different protocols")
    if invalid_report.get("acceptance_hash") != report.get("acceptance_hash"):
        raise PilotValidationError("Original INVALID and recovery audit use different acceptances")
    if report.get("protocol_hash") != RECOVERY_PROTOCOL_HASH:
        raise PilotValidationError("Recovery protocol hash is not the frozen v5 protocol")
    if report.get("training_environment_sha256") != RECOVERY_TRAINING_ENVIRONMENT_SHA256:
        raise PilotValidationError(
            "Recovery training environment is not the audited shared pilot environment"
        )
    recovered_datasets = report.get("datasets")
    if not isinstance(recovered_datasets, Mapping):
        raise PilotValidationError("Original INVALID has no complete dataset evidence")
    for dataset in V5_PILOT_DATASETS:
        recovered_dataset = recovered_datasets.get(dataset)
        if not isinstance(recovered_dataset, Mapping):
            raise PilotValidationError(f"Missing recovery dataset evidence for {dataset}")
    if invalid_report.get("threshold_failures") != []:
        raise PilotValidationError("Original INVALID contains aggregate threshold failures")
    if pilot_execution_commit != RECOVERY_PILOT_EXECUTION_COMMIT:
        raise PilotValidationError(
            "--pilot-execution-commit is not the reviewed v5 pilot release commit"
        )
    observed_commits: set[str] = set()
    for dataset in V5_PILOT_DATASETS:
        health_path = recovered_datasets[dataset].get("health_report_path")
        if not isinstance(health_path, str):
            raise PilotValidationError(f"Recovery has no health report path for {dataset}")
        health = _read_json(repository_root / health_path, f"{dataset} health report")
        observed_commits.add(str(health.get("git_commit", "")))
    if observed_commits != {pilot_execution_commit}:
        raise PilotValidationError(
            "Pilot execution commit differs from the health evidence: "
            f"{sorted(observed_commits)}"
        )
    validator_path = Path(__file__).resolve()
    release = _recovery_release_binding(repository_root)
    return {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "artifact_class": RECOVERY_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": RECOVERY_DECISION,
        "exit_code": 0,
        "validated_at": report.get("validated_at"),
        "protocol_hash": report.get("protocol_hash"),
        "acceptance_hash": report.get("acceptance_hash"),
        "pilot_execution_commit": pilot_execution_commit,
        "recovery_validator_commit": release["recovery_commit"],
        "release_delta": release,
        "recovery_validator": {
            "path": artifact_path_reference(validator_path, repository_root),
            "sha256": sha256_file(validator_path),
        },
        "original_validation": {
            "path": artifact_path_reference(invalid_path, repository_root),
            "sha256": invalid_sha256,
            "status": "INVALID",
            "decision": invalid_report.get("decision"),
            "legacy_reproduction": "EXACT_EXCEPT_VALIDATED_AT",
            "legacy_reproduction_sha256": stable_hash(
                _report_without_timestamp(legacy_report)
            ),
        },
        "recovered_validation": dict(report),
        "output": artifact_path_reference(output, repository_root),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        acceptance = protocol["pilot_acceptance"]
        canonical_output = (PROJECT_ROOT / acceptance["validation_output"]).resolve()
        recovery_mode = args.recover_from_invalid is not None
        if recovery_mode:
            if args.output is None or args.pilot_execution_commit is None:
                raise PilotValidationError(
                    "Recovery requires --output and --pilot-execution-commit"
                )
            invalid_path = args.recover_from_invalid.resolve()
            if invalid_path != canonical_output:
                raise PilotValidationError(
                    "--recover-from-invalid must name the canonical validation output"
                )
            output = args.output.resolve()
            expected_recovery_output = canonical_output.with_name(RECOVERY_OUTPUT_NAME)
            if output != expected_recovery_output:
                raise PilotValidationError(
                    "Recovery output must be the fixed adjacent "
                    f"{RECOVERY_OUTPUT_NAME} artifact"
                )
        else:
            if args.pilot_execution_commit is not None:
                raise PilotValidationError(
                    "--pilot-execution-commit is valid only with --recover-from-invalid"
                )
            output = args.output.resolve() if args.output is not None else canonical_output
            if output != canonical_output:
                raise PilotValidationError(
                    "--output cannot override pilot_acceptance.validation_output"
                )
        if output.exists():
            raise PilotValidationError(
                "V5 pilot validation output already exists; refusing overwrite"
            )
        report = validate_pilot(
            protocol_path=args.protocol,
            config_dir=args.config_dir,
            repository_root=PROJECT_ROOT,
            strict_health_validation=True,
            evidence_git_commit=(
                args.pilot_execution_commit if recovery_mode else None
            ),
            allow_equivalent_absolute_output_dir=recovery_mode,
        )
        if recovery_mode:
            invalid_report = _read_json(invalid_path, "original v5 pilot validation")
            legacy_report = validate_pilot(
                protocol_path=args.protocol,
                config_dir=args.config_dir,
                repository_root=PROJECT_ROOT,
                strict_health_validation=True,
                evidence_git_commit=args.pilot_execution_commit,
                allow_equivalent_absolute_output_dir=False,
            )
            stored_report = _recovery_report(
                report=report,
                legacy_report=legacy_report,
                invalid_path=invalid_path,
                invalid_report=invalid_report,
                output=output,
                repository_root=PROJECT_ROOT,
                pilot_execution_commit=args.pilot_execution_commit,
            )
        else:
            stored_report = report
        exclusive_create_json(output, stored_report)
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed with one marker
        print(f"V5_PILOT_VALIDATION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    marker = "V5_PILOT_VALIDATION_RECOVERY_PASS" if recovery_mode else (
        f"V5_PILOT_VALIDATION_{report['status']}"
    )
    print(marker)
    print(f"NON_REPORTING_OUTPUT={output}")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
