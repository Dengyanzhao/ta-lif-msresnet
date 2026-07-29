#!/usr/bin/env python3
"""Validate both independent non-reportable TA-LIF-only v4 pilot blocks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import validate_v3_pilot as common  # noqa: E402
from talif_msresnet.config import (  # noqa: E402
    RunConfig,
    artifact_paths_for_protocol,
    generate_v4_pilot_matrix,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.pilot_v4 import (  # noqa: E402
    ATTEMPT_STATUS,
    REPORTING_ELIGIBILITY,
    V4_PILOT_CONDITIONS,
    V4_PILOT_DATASETS,
    PilotBlock,
    PilotV4Error,
    expected_pilot_plan_payload,
    require_v4_author_freeze,
    resolve_pilot_block,
    validate_attempt_receipt,
    validate_health_report,
)
from talif_msresnet.utils import (  # noqa: E402
    atomic_write_json,
    sha256_file,
    stable_hash,
    utc_now,
)


SCHEMA_VERSION = 1
ARTIFACT_CLASS = "NON_REPORTABLE_V4_TALIF_ONLY_PILOT_ACCEPTANCE"
PASS_DECISION = "ACCEPT_V4_TALIF_ONLY_120_EPOCH_PILOTS_RELEASE_FORMAL_FREEZE"
FAIL_DECISION = "BLOCK_V4_AND_REQUIRE_NEW_PROTOCOL_AND_UNUSED_PILOT_SEEDS"
INVALID_DECISION = "BLOCK_V4_AND_INVESTIGATE_PILOT_EVIDENCE_INTEGRITY"


class PilotValidationError(RuntimeError):
    """Raised when the validator cannot establish the v4 pilot contract."""


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
    manifest = _read_json(manifest_path, "v4 pilot matrix manifest")
    raw_rows = manifest.get("runs")
    if not isinstance(raw_rows, list):
        raise PilotValidationError("matrix manifest runs must be a list")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        failures.append("matrix manifest contains a non-object run")
    expected_raw = generate_v4_pilot_matrix(protocol)
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
        failures.append("matrix manifest is not the ordered two-dataset C1/C2 v4 pilot")
    expected_seeds = list(dict.fromkeys(config.runtime.seed for config in expected_configs))
    checks = {
        "protocol": artifact_path_reference(protocol_path, repository_root),
        "protocol_hash": stable_hash(protocol),
        "run_count": 4,
        "conditions": list(V4_PILOT_CONDITIONS),
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
        "status": "PASS",
        "pass": True,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "dataset": block.dataset,
        "seed": block.seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "tracked_clean": True,
        "attempt_receipt_sha256": receipt_sha256,
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
    try:
        receipt = validate_attempt_receipt(
            block.attempt_receipt,
            block=block,
            protocol_path=protocol_path,
            repository_root=repository_root,
        )
    except PilotV4Error as exc:
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
            "status": ATTEMPT_STATUS,
        }
        for key, value in expected_receipt.items():
            if receipt.get(key) != value:
                failures.append(f"attempt receipt {key} mismatch")
    report = _read_json(block.health_output, "v4 health report")
    receipt_sha256 = sha256_file(block.attempt_receipt)
    failures.extend(_basic_health_failures(report, block=block, receipt_sha256=receipt_sha256))
    if strict_health_validation:
        try:
            report = validate_health_report(
                block.health_output,
                protocol_path,
                block.dataset,
                repository_root=repository_root,
                require_current_tracked_clean=True,
            )
        except PilotV4Error as exc:
            failures.append(f"canonical health validation failed: {exc}")
    plan = _read_json(block.pilot_plan, "v4 pilot plan")
    try:
        expected_plan = expected_pilot_plan_payload(
            block=block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=manifest_rows,
            health_report=report,
            repository_root=repository_root,
        )
    except PilotV4Error as exc:
        failures.append(str(exc))
    else:
        if plan != expected_plan:
            failures.append("pilot plan differs from the exact v4 dataset-bound payload")
    return {
        "health": report,
        "health_path": artifact_path_reference(block.health_output, repository_root),
        "health_sha256": sha256_file(block.health_output),
        "attempt_receipt_path": artifact_path_reference(block.attempt_receipt, repository_root),
        "attempt_receipt_sha256": receipt_sha256,
        "plan_path": artifact_path_reference(block.pilot_plan, repository_root),
        "plan_sha256": sha256_file(block.pilot_plan),
    }, failures


def _v4_threshold_summary(
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
    """Reconstruct and apply the frozen v4 pilot-only acceptance statistics."""

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
    )
    # The legacy helper reconstructs checkpoints and the formal convergence
    # fields.  Its two v3 acceptance failures are replaced by v4's dataset-
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
    except Exception as exc:
        base_report.setdefault("integrity_failures", []).append(str(exc))
        return base_report, environment_hash, shared_hash, split_hash
    required_best = float(schedule["required_best_validation_accuracy"][block.dataset])
    minimum_late = float(schedule["minimum_late_mean_accuracy"][block.dataset])
    summary, summary_integrity, summary_thresholds = _v4_threshold_summary(
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


def validate_pilot(
    *,
    protocol_path: str | Path,
    config_dir: str | Path,
    repository_root: str | Path = PROJECT_ROOT,
    strict_health_validation: bool = True,
) -> dict[str, Any]:
    """Validate both v4 C1/C2 blocks and return one aggregate verdict."""

    root = Path(repository_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    config_dir = Path(config_dir).resolve()
    protocol = load_protocol(protocol_path)
    if protocol.get("protocol_version") != 4:
        raise PilotValidationError("This validator accepts only protocol_version 4")
    require_v4_author_freeze(protocol, repository_root=root)
    artifact_paths = artifact_paths_for_protocol(protocol)
    expected_protocol_path = (root / artifact_paths["protocol"]).resolve()
    expected_config_dir = (root / artifact_paths["pilot_matrix"]).resolve()
    if protocol_path != expected_protocol_path:
        raise PilotValidationError(
            f"Protocol path differs from the v4 artifact binding: {protocol_path}"
        )
    if config_dir != expected_config_dir:
        raise PilotValidationError(
            f"Config directory differs from the v4 artifact binding: {config_dir}"
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
        raise PilotValidationError("The v4 pilot schedule must be exactly 120 epochs")
    if timing.get("epoch_time_ratio_conditions") != ["C2"]:
        raise PilotValidationError("The v4 pilot timing condition must be exactly C2")
    if timing.get("ta_state_source") != "events_jsonl_epoch_completed_ta_enabled":
        raise PilotValidationError("Unsupported v4 pilot TA timing state source")

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

    for dataset in V4_PILOT_DATASETS:
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
        )
        block_integrity.extend(evidence_failures)
        health = evidence.get("health")
        if not isinstance(health, Mapping):
            health = {}
        expected_run_ids = {
            expected_configs[(dataset, condition)].runtime.run_id
            for condition in V4_PILOT_CONDITIONS
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
        for condition in V4_PILOT_CONDITIONS:
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
            "seed": block.seed,
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

    if aggregate_integrity:
        status, decision, exit_code = "INVALID", INVALID_DECISION, 2
    elif aggregate_thresholds:
        status, decision, exit_code = "FAIL", FAIL_DECISION, 1
    else:
        status, decision, exit_code = "PASS", PASS_DECISION, 0
    return {
        "schema_version": SCHEMA_VERSION,
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
        "dataset_blocks_independent": True,
        "cross_dataset_environment_split_or_shared_hash_equality_required": False,
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
        default=PROJECT_ROOT / "configs" / "protocol_v4_talif_only.yaml",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v4_talif_only_pilot_generated",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        acceptance = protocol["pilot_acceptance"]
        canonical_output = (PROJECT_ROOT / acceptance["validation_output"]).resolve()
        output = args.output.resolve() if args.output is not None else canonical_output
        if output != canonical_output:
            raise PilotValidationError(
                "--output cannot override pilot_acceptance.validation_output"
            )
        report = validate_pilot(
            protocol_path=args.protocol,
            config_dir=args.config_dir,
            repository_root=PROJECT_ROOT,
        )
        atomic_write_json(output, report)
    except Exception as exc:
        print(f"V4_PILOT_VALIDATION_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"V4_PILOT_VALIDATION_{report['status']}")
    print(f"NON_REPORTING_OUTPUT={output}")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
