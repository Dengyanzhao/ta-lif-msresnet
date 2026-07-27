#!/usr/bin/env python3
"""Validate the complete non-reportable v2 120-epoch pilot block."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import (  # noqa: E402
    SUPPORTED_CONDITIONS,
    generate_run_matrix,
    load_protocol,
    load_run_config,
    validate_run_mapping,
)
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.utils import (  # noqa: E402
    atomic_write_json,
    load_checkpoint,
    sha256_file,
    stable_hash,
    utc_now,
)
from pilot_health_gate import (  # noqa: E402
    HealthGateError,
    validate_pilot_health_report,
)


SCHEMA_VERSION = 1
ARTIFACT_CLASS = "NON_REPORTABLE_V2_PILOT_ACCEPTANCE"


class PilotValidationError(RuntimeError):
    """Raised when the validator itself cannot establish the pilot contract."""


def _absolute(path: str | Path, repository_root: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (repository_root / value).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotValidationError(f"JSON artifact must be an object: {path}")
    return value


def _read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PilotValidationError(f"Cannot read event log {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PilotValidationError(f"Malformed JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(item, dict):
            raise PilotValidationError(f"Event at {path}:{line_number} is not an object")
        events.append(item)
    return events


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _same_number(observed: Any, expected: float) -> bool:
    value = _finite_number(observed)
    return value is not None and math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)


def _same_csv_number(observed: Any, expected: float) -> bool:
    try:
        value = float(observed)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(
        value, expected, rel_tol=1e-12, abs_tol=1e-12
    )


def _best_val_equal(observed: Any, expected: Mapping[str, Any]) -> bool:
    return bool(
        isinstance(observed, Mapping)
        and observed.get("epoch") == expected["epoch"]
        and _same_number(observed.get("accuracy"), float(expected["accuracy"]))
        and _same_number(observed.get("loss"), float(expected["loss"]))
    )


def _evidence_equal(observed: Any, expected: Any) -> bool:
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        return set(observed) == set(expected) and all(
            _evidence_equal(observed[key], expected[key]) for key in expected
        )
    if isinstance(observed, list) and isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _evidence_equal(left, right) for left, right in zip(observed, expected)
        )
    if (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        left = float(observed)
        right = float(expected)
        if math.isnan(left) and math.isnan(right):
            return True
        return left == right
    return observed == expected


def _epoch_evidence(
    epoch_events: Sequence[Mapping[str, Any]],
    *,
    candidate_epochs: int,
    convergence_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int | None, list[str]]:
    """Reconstruct trainer history and model-selection outcomes from epoch events."""

    failures: list[str] = []
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    convergence_epoch: int | None = None
    epochs = [event.get("epoch") for event in epoch_events]
    if epochs != list(range(candidate_epochs)):
        failures.append(
            f"epoch_completed sequence is not exactly 0..{candidate_epochs - 1}"
        )
    for index, event in enumerate(epoch_events):
        epoch = event.get("epoch")
        if not isinstance(epoch, int):
            failures.append(f"epoch event {index} has an invalid epoch")
            continue
        train = event.get("train")
        val = event.get("val")
        if not isinstance(train, Mapping) or not isinstance(val, Mapping):
            failures.append(f"epoch {epoch}: train/val evidence is missing")
            continue
        train_loss = _finite_number(train.get("loss"))
        train_accuracy = _finite_number(train.get("accuracy"))
        train_seconds = _finite_number(train.get("seconds"))
        val_accuracy = _finite_number(val.get("accuracy"))
        val_loss = _finite_number(val.get("loss"))
        if train_loss is None or train_accuracy is None:
            failures.append(f"epoch {epoch}: train loss/accuracy is invalid")
        if train_seconds is None or train_seconds <= 0:
            failures.append(f"epoch {epoch}: train seconds is invalid")
        if val_accuracy is None or val_loss is None:
            failures.append(f"epoch {epoch}: validation loss/accuracy is invalid")
            continue
        row = dict(train)
        row.update({"epoch": epoch, "val_accuracy": val_accuracy, "val_loss": val_loss})
        history.append(row)
        if best is None or val_accuracy > float(best["accuracy"]) or (
            val_accuracy == float(best["accuracy"]) and val_loss < float(best["loss"])
        ):
            best = {"accuracy": val_accuracy, "loss": val_loss, "epoch": epoch}
        if convergence_epoch is None and val_accuracy >= convergence_threshold:
            convergence_epoch = epoch
        if best is not None and not _best_val_equal(event.get("best_val"), best):
            failures.append(f"epoch {epoch}: cumulative best_val evidence is inconsistent")
    if len(history) != candidate_epochs:
        failures.append(
            f"reconstructed checkpoint history has {len(history)} rows, expected {candidate_epochs}"
        )
    return history, best, convergence_epoch, failures


def _effective_epoch_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int, list[str]]:
    """Discard abandoned post-checkpoint epoch events at each audited resume boundary."""

    effective: list[Mapping[str, Any]] = []
    resume_count = 0
    failures: list[str] = []
    for event in events:
        event_name = event.get("event")
        if event_name == "resumed":
            resume_count += 1
            start_epoch = event.get("start_epoch")
            checkpoint = Path(str(event.get("checkpoint", "")))
            if not isinstance(start_epoch, int) or start_epoch < 0:
                failures.append("resumed event has an invalid start_epoch")
                continue
            if checkpoint.name != "last.pt":
                failures.append("resumed event is not bound to last.pt")
            effective = [
                item
                for item in effective
                if isinstance(item.get("epoch"), int) and int(item["epoch"]) < start_epoch
            ]
            if [item.get("epoch") for item in effective] != list(range(start_epoch)):
                failures.append(
                    f"resumed start_epoch {start_epoch} has no complete checkpoint prefix"
                )
        elif event_name == "epoch_completed":
            effective.append(event)
    return effective, resume_count, failures


def _checkpoint_failures(
    path: Path,
    *,
    label: str,
    expected_epoch: int,
    expected_best: Mapping[str, Any],
    expected_history: Sequence[Mapping[str, Any]],
    expected_config_hash: str,
    expected_environment_hash: str,
    protocol: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    try:
        checkpoint = load_checkpoint(path, map_location="cpu")
    except Exception as exc:
        return [f"{label} cannot be loaded: {type(exc).__name__}: {exc}"]
    model_state = checkpoint.get("model_state")
    optimizer_state = checkpoint.get("optimizer_state")
    scheduler_state = checkpoint.get("scheduler_state")
    if not isinstance(model_state, Mapping) or not model_state:
        failures.append(f"{label} model_state is missing or empty")
    if not isinstance(optimizer_state, Mapping) or not {
        "state",
        "param_groups",
    }.issubset(optimizer_state):
        failures.append(f"{label} optimizer_state is incomplete")
    if not isinstance(scheduler_state, Mapping) or not scheduler_state:
        failures.append(f"{label} scheduler_state is missing or empty")
    if checkpoint.get("config_hash") != expected_config_hash:
        failures.append(f"{label} config_hash mismatch")
    if checkpoint.get("training_environment_sha256") != expected_environment_hash:
        failures.append(f"{label} training-environment hash mismatch")
    if checkpoint.get("epoch") != expected_epoch:
        failures.append(
            f"{label} epoch={checkpoint.get('epoch')!r}, expected {expected_epoch}"
        )
    if not _best_val_equal(checkpoint.get("best_val"), expected_best):
        failures.append(f"{label} best_val differs from epoch evidence")
    history = checkpoint.get("train_history")
    if not isinstance(history, list) or not _evidence_equal(
        history, list(expected_history)
    ):
        failures.append(f"{label} train_history differs from epoch evidence")
    raw_config = checkpoint.get("config")
    try:
        checkpoint_config = validate_run_mapping(raw_config, protocol)
    except (TypeError, ValueError) as exc:
        failures.append(f"{label} config is invalid: {exc}")
    else:
        if checkpoint_config.config_hash != expected_config_hash:
            failures.append(f"{label} embedded scientific config mismatch")
    return failures


def _validate_environment(identity: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    hardware = identity.get("hardware")
    software = identity.get("software")
    determinism = identity.get("determinism")
    if not isinstance(hardware, Mapping):
        return ["training environment has no hardware mapping"]
    if not isinstance(software, Mapping):
        return ["training environment has no software mapping"]
    if not isinstance(determinism, Mapping):
        return ["training environment has no determinism mapping"]
    gpu_substring = str(expected["expected_gpu_substring"])
    if gpu_substring.lower() not in str(hardware.get("name", "")).lower():
        failures.append(f"GPU name {hardware.get('name')!r} does not contain {gpu_substring!r}")
    if software.get("pytorch") != expected["pytorch_version"]:
        failures.append(f"PyTorch {software.get('pytorch')!r} != {expected['pytorch_version']!r}")
    if str(software.get("cuda_version")) != str(expected["cuda_runtime"]):
        failures.append(
            f"CUDA runtime {software.get('cuda_version')!r} != {expected['cuda_runtime']!r}"
        )
    if identity.get("precision") != expected["precision"]:
        failures.append(f"precision {identity.get('precision')!r} != {expected['precision']!r}")
    deterministic_expected = bool(expected["deterministic"])
    if bool(determinism.get("requested")) is not deterministic_expected:
        failures.append("determinism.requested differs from the protocol")
    if deterministic_expected:
        required = {
            "torch_algorithms_enabled": True,
            "torch_warn_only": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }
        for key, value in required.items():
            if determinism.get(key) is not value:
                failures.append(f"determinism.{key} must be {value}")
        if determinism.get("cublas_workspace_config") not in {":4096:8", ":16:8"}:
            failures.append("determinism.cublas_workspace_config is not an approved value")
    return failures


def _timing_summary(
    condition: str,
    epoch_events: Sequence[Mapping[str, Any]],
    activation_epoch: int,
    maximum_ratio: float,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expects_ta = condition in {"C2", "C4"}
    frozen: list[float] = []
    enabled: list[float] = []
    for event in epoch_events:
        epoch = int(event["epoch"])
        observed_enabled = event.get("ta_enabled")
        expected_enabled = bool(expects_ta and epoch >= activation_epoch)
        if observed_enabled is not expected_enabled:
            failures.append(
                f"{condition} epoch {epoch}: ta_enabled={observed_enabled!r}, "
                f"expected {expected_enabled}"
            )
        train = event.get("train")
        seconds = _finite_number(train.get("seconds") if isinstance(train, Mapping) else None)
        if seconds is None or seconds <= 0:
            failures.append(f"{condition} epoch {epoch}: train.seconds is invalid")
            continue
        (enabled if expected_enabled else frozen).append(seconds)

    summary: dict[str, Any] = {
        "activation_epoch_zero_based": activation_epoch,
        "frozen_epoch_count": len(frozen),
        "enabled_epoch_count": len(enabled),
        "statistic": "median_enabled_train_seconds_over_median_frozen_train_seconds",
        "maximum_allowed": maximum_ratio,
    }
    if not expects_ta:
        summary.update({"status": "NOT_APPLICABLE_LIF", "pass": not failures})
        return summary, failures
    if not frozen or not enabled:
        failures.append(f"{condition}: both frozen and enabled timing epochs are required")
        summary.update({"ratio": None, "pass": False})
        return summary, failures
    frozen_median = float(statistics.median(frozen))
    enabled_median = float(statistics.median(enabled))
    ratio = enabled_median / frozen_median if frozen_median > 0 else math.inf
    ratio_pass = bool(math.isfinite(ratio) and ratio <= maximum_ratio)
    summary.update(
        {
            "frozen_median_seconds": frozen_median,
            "enabled_median_seconds": enabled_median,
            "ratio": ratio,
            "pass": ratio_pass and not failures,
        }
    )
    if not ratio_pass:
        failures.append(
            f"{condition}: enabled/frozen epoch-time ratio {ratio:.6f} exceeds {maximum_ratio:.6f}"
        )
    return summary, failures


def _bound_pilot_evidence(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    protocol_hash: str,
    acceptance: Mapping[str, Any],
    acceptance_hash: str,
    repository_root: Path,
    config_dir: Path,
    results_root: Path,
    manifest_runs: Sequence[Mapping[str, Any]],
    strict_health_validation: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Validate the immutable health-report and orchestrator-plan chain."""

    failures: list[str] = []
    health_path = _absolute(str(acceptance.get("health_output", "")), repository_root)
    plan_path = _absolute(str(acceptance.get("pilot_plan", "")), repository_root)
    if not health_path.is_file():
        return {}, [f"pilot health report is missing: {health_path}"]
    if not plan_path.is_file():
        return {}, [f"pilot plan is missing: {plan_path}"]
    health = _read_json(health_path)
    health_sha256 = sha256_file(health_path)
    health_commit = health.get("git_commit")
    valid_commit = bool(
        isinstance(health_commit, str)
        and len(health_commit) == 40
        and all(character in "0123456789abcdef" for character in health_commit)
    )
    health_checks = {
        "status": (health.get("status"), "PASS"),
        "pass": (health.get("pass"), True),
        "protocol_hash": (health.get("protocol_hash"), protocol_hash),
        "acceptance_hash": (health.get("acceptance_hash"), acceptance_hash),
        "tracked_clean": (health.get("tracked_clean"), True),
        "reporting_eligibility": (
            health.get("reporting_eligibility"),
            "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        ),
        "confirmatory_analysis_eligibility": (
            health.get("confirmatory_analysis_eligibility"),
            False,
        ),
    }
    for label, (observed, expected) in health_checks.items():
        if observed != expected:
            failures.append(
                f"pilot health report {label}={observed!r}, expected {expected!r}"
            )
    if not valid_commit:
        failures.append("pilot health report git_commit is invalid")
    if strict_health_validation:
        try:
            health = validate_pilot_health_report(
                health_path,
                protocol_path,
                repository_root=repository_root,
                require_current_tracked_clean=True,
            )
        except HealthGateError as exc:
            failures.append(f"pilot health report failed canonical validation: {exc}")

    plan = _read_json(plan_path)
    expected_runs = [
        {
            "condition": str(row.get("condition")),
            "run_id": str(row.get("run_id")),
            "config_file": artifact_path_reference(
                config_dir / str(row.get("config_file")),
                repository_root,
            ),
            "config_file_sha256": row.get("config_file_sha256"),
            "config_hash": row.get("config_hash"),
        }
        for row in manifest_runs
    ]
    expected_plan = {
        "version": 2,
        "non_reportable": True,
        "protocol_path": artifact_path_reference(protocol_path, repository_root),
        "protocol_hash": protocol_hash,
        "acceptance_hash": acceptance_hash,
        "git_commit": health_commit,
        "health_report": artifact_path_reference(health_path, repository_root),
        "health_report_sha256": health_sha256,
        "pilot_output_root": artifact_path_reference(results_root, repository_root),
        "formal_output_root": artifact_path_reference(
            _absolute(str(protocol["output_root"]), repository_root), repository_root
        ),
        "runs": expected_runs,
    }
    for key, expected in expected_plan.items():
        if plan.get(key) != expected:
            failures.append(f"pilot plan {key} differs from bound evidence")
    evidence = {
        "pilot_plan": artifact_path_reference(plan_path, repository_root),
        "pilot_plan_sha256": sha256_file(plan_path),
        "plan_version": 2,
        "protocol_hash": protocol_hash,
        "acceptance_hash": acceptance_hash,
        "git_commit": health_commit,
        "health_report": artifact_path_reference(health_path, repository_root),
        "health_report_sha256": health_sha256,
    }
    return {
        "health": health,
        "health_path": str(health_path),
        "health_sha256": health_sha256,
        "plan": plan,
        "plan_path": str(plan_path),
        "plan_sha256": evidence["pilot_plan_sha256"],
        "run_manifest_evidence": evidence,
    }, failures


def validate_pilot(
    *,
    protocol_path: str | Path,
    config_dir: str | Path,
    results_root: str | Path,
    repository_root: str | Path = PROJECT_ROOT,
    strict_health_validation: bool = True,
) -> dict[str, Any]:
    """Return a complete PASS/FAIL/INVALID report without modifying artifacts."""

    repository_root = Path(repository_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    config_dir = Path(config_dir).resolve()
    results_root = Path(results_root).resolve()
    protocol = load_protocol(protocol_path)
    if protocol.get("protocol_version") != 2 or protocol.get("study_stage") != "pilot":
        raise PilotValidationError("This validator accepts only the v2 pilot protocol")
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise PilotValidationError("Protocol has no pilot_acceptance mapping")
    expected_root = _absolute(str(acceptance["pilot_output_root"]), repository_root)
    if results_root != expected_root:
        raise PilotValidationError(
            f"Results root differs from protocol: {results_root} != {expected_root}"
        )

    protocol_hash = stable_hash(protocol)
    acceptance_hash = stable_hash(dict(acceptance))
    integrity_failures: list[str] = []
    threshold_failures: list[str] = []
    runs_report: dict[str, Any] = {}

    manifest_path = config_dir / "matrix_manifest.json"
    manifest = _read_json(manifest_path)
    manifest_runs = manifest.get("runs")
    if manifest.get("protocol_hash") != protocol_hash:
        integrity_failures.append("matrix manifest protocol_hash mismatch")
    if manifest.get("run_count") != 4:
        integrity_failures.append("matrix manifest must contain exactly four runs")
    if manifest.get("conditions") != list(SUPPORTED_CONDITIONS):
        integrity_failures.append("matrix manifest condition order must be C1,C2,C3,C4")
    if manifest.get("seeds") != [acceptance["seed"]]:
        integrity_failures.append("matrix manifest seed differs from pilot_acceptance")
    if not isinstance(manifest_runs, list):
        raise PilotValidationError("matrix manifest runs must be a list")
    if manifest.get("matrix_hash") != stable_hash(generate_run_matrix(protocol)):
        integrity_failures.append("matrix manifest matrix_hash mismatch")

    run_manifest_csv_path = config_dir / "run_manifest.csv"
    csv_manifest_rows: list[dict[str, str]] = []
    if not run_manifest_csv_path.is_file():
        integrity_failures.append("generated run_manifest.csv is missing")
    else:
        try:
            with run_manifest_csv_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                csv_manifest_rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise PilotValidationError(
                f"Cannot read generated run manifest {run_manifest_csv_path}: {exc}"
            ) from exc
        normalized_json_rows = [
            {key: str(value) for key, value in row.items()}
            for row in manifest_runs
            if isinstance(row, Mapping)
        ]
        if csv_manifest_rows != normalized_json_rows:
            integrity_failures.append(
                "run_manifest.csv differs from matrix_manifest.json"
            )

    rows_by_condition: dict[str, Mapping[str, Any]] = {}
    for row in manifest_runs:
        if not isinstance(row, Mapping):
            integrity_failures.append("matrix manifest contains a non-object run")
            continue
        condition = str(row.get("condition", ""))
        if condition in rows_by_condition:
            integrity_failures.append(f"duplicate matrix condition: {condition}")
        rows_by_condition[condition] = row
    if set(rows_by_condition) != set(SUPPORTED_CONDITIONS):
        integrity_failures.append("matrix run conditions are not exactly C1-C4")
    expected_configs = {
        config.model.condition: config
        for config in (
            validate_run_mapping(raw, protocol) for raw in generate_run_matrix(protocol)
        )
    }
    if list(expected_configs) != list(SUPPORTED_CONDITIONS):
        raise PilotValidationError(
            "Protocol does not generate one ordered C1-C4 pilot block"
        )

    bound_evidence, evidence_failures = _bound_pilot_evidence(
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_hash=protocol_hash,
        acceptance=acceptance,
        acceptance_hash=acceptance_hash,
        repository_root=repository_root,
        config_dir=config_dir,
        results_root=results_root,
        manifest_runs=[row for row in manifest_runs if isinstance(row, Mapping)],
        strict_health_validation=strict_health_validation,
    )
    integrity_failures.extend(evidence_failures)

    expected_run_ids = {str(row.get("run_id")) for row in rows_by_condition.values()}
    if results_root.is_dir():
        observed_run_ids = {
            path.name
            for path in results_root.iterdir()
            if path.is_dir() and path.name.startswith("E1_")
        }
    else:
        observed_run_ids = set()
        integrity_failures.append(f"pilot results root is missing: {results_root}")
    if observed_run_ids != expected_run_ids:
        integrity_failures.append(
            "pilot run directories differ from the exact four-run matrix: "
            f"observed={sorted(observed_run_ids)} expected={sorted(expected_run_ids)}"
        )

    matrix_events: list[dict[str, Any]] = []
    historical_failure_run_ids: set[str] = set()
    matrix_events_path = results_root / "matrix_events.jsonl"
    if not matrix_events_path.is_file():
        integrity_failures.append("matrix_events.jsonl is missing")
    else:
        matrix_events = _read_events(matrix_events_path)
        blocked_events = [
            event for event in matrix_events if event.get("event") == "run_blocked"
        ]
        if blocked_events:
            integrity_failures.append("matrix event stream contains blocked runs")
        health = bound_evidence.get("health")
        health_environment = (
            health.get("environment") if isinstance(health, Mapping) else None
        )
        health_source = health.get("source") if isinstance(health, Mapping) else None
        expected_context = None
        if isinstance(health_environment, Mapping) and isinstance(health_source, Mapping):
            idle = health_environment.get("gpu_idle_precheck")
            if isinstance(idle, Mapping):
                expected_context = {
                    "pass": True,
                    "device_uuid": idle.get("device_uuid"),
                    "hardware": health_environment.get("hardware"),
                    "split_manifest_sha256": health_source.get("split_manifest_sha256"),
                    "fixed_batch_sha256": health_source.get("fixed_batch_sha256"),
                    "fixed_batch_shape": health_source.get("fixed_batch_shape"),
                }

        def context_matches(event: Mapping[str, Any]) -> bool:
            context = event.get("context")
            return bool(
                isinstance(context, Mapping)
                and isinstance(expected_context, Mapping)
                and all(context.get(key) == value for key, value in expected_context.items())
            )

        initial_contexts = [
            event
            for event in matrix_events
            if event.get("event") == "pilot_context_validated"
        ]
        if len(initial_contexts) != 1 or not context_matches(initial_contexts[0]):
            integrity_failures.append(
                "matrix event stream lacks the bound initial runtime/data context gate"
            )
        for run_id in sorted(expected_run_ids):
            indexed = [
                (index, event)
                for index, event in enumerate(matrix_events)
                if str(event.get("run_id")) == run_id
            ]
            failed = [
                (index, event)
                for index, event in indexed
                if event.get("event") == "run_failed"
            ]
            succeeded = [
                (index, event)
                for index, event in indexed
                if event.get("event") == "run_finished" and event.get("returncode") == 0
            ]
            if len(succeeded) != 1:
                integrity_failures.append(
                    f"{run_id}: matrix event stream must contain exactly one successful finish"
                )
                continue
            run_contexts = [
                event
                for _index, event in indexed
                if event.get("event") == "run_context_validated"
            ]
            if not run_contexts or not all(context_matches(event) for event in run_contexts):
                integrity_failures.append(
                    f"{run_id}: matrix event stream lacks a bound pre-run context gate"
                )
            if failed:
                historical_failure_run_ids.add(run_id)
                success_index, success_event = succeeded[0]
                if max(index for index, _event in failed) >= success_index:
                    integrity_failures.append(
                        f"{run_id}: unresolved matrix failure is not followed by success"
                    )
                if success_event.get("action") != "resume":
                    integrity_failures.append(
                        f"{run_id}: success after failure was not a last.pt resume"
                    )

    consolidated_path = results_root / "seed_metrics.csv"
    if not consolidated_path.is_file():
        integrity_failures.append("consolidated seed_metrics.csv is missing")
    else:
        try:
            with consolidated_path.open("r", encoding="utf-8-sig", newline="") as handle:
                consolidated = list(csv.DictReader(handle))
        except OSError as exc:
            raise PilotValidationError(
                f"Cannot read consolidated metrics {consolidated_path}: {exc}"
            ) from exc
        consolidated_by_id = {str(row.get("run_id")): row for row in consolidated}
        if len(consolidated) != 4 or set(consolidated_by_id) != expected_run_ids:
            integrity_failures.append(
                "consolidated seed_metrics.csv is not the exact four-run pilot block"
            )

    environment_hashes: set[str] = set()
    shared_weight_hashes: set[str] = set()
    split_hashes: set[str] = set()
    candidate_epochs = int(acceptance["schedule"]["candidate_epochs"])
    accuracy_threshold = float(acceptance["schedule"]["required_best_validation_accuracy"])
    maximum_ratio = float(acceptance["timing"]["maximum_ta_enabled_over_frozen_ratio"])
    timing = acceptance["timing"]
    if timing.get("epoch_time_ratio_statistic") != (
        "median_enabled_train_seconds_over_median_frozen_train_seconds"
    ):
        raise PilotValidationError("Unsupported v2 pilot epoch-time ratio statistic")
    if timing.get("epoch_time_ratio_conditions") != ["C2", "C4"]:
        raise PilotValidationError("Unsupported v2 pilot TA timing condition set")
    if timing.get("ta_state_source") != "events_jsonl_epoch_completed_ta_enabled":
        raise PilotValidationError("Unsupported v2 pilot TA state source")

    for condition in SUPPORTED_CONDITIONS:
        row = rows_by_condition.get(condition)
        if row is None:
            continue
        run_id = str(row.get("run_id"))
        run_dir = results_root / run_id
        run_integrity: list[str] = []
        run_thresholds: list[str] = []
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
            run_integrity.append(f"missing required artifacts: {missing}")
        failure_path = run_dir / "failure.json"
        failed_checkpoint_path = run_dir / "failed.pt"
        has_failure_artifact = failure_path.exists() or failed_checkpoint_path.exists()
        if has_failure_artifact and run_id not in historical_failure_run_ids:
            run_integrity.append(
                "failure artifact exists without a retained matrix run_failed event"
            )
        if run_id in historical_failure_run_ids and not failure_path.is_file():
            run_integrity.append("historical matrix failure has no retained failure.json")
        if not list(run_dir.glob("attempt_*.stdout.log")):
            run_integrity.append("matrix stdout attempt log is missing")
        if not list(run_dir.glob("attempt_*.stderr.log")):
            run_integrity.append("matrix stderr attempt log is missing")
        config_path = config_dir / str(row.get("config_file"))
        if not config_path.is_file():
            run_integrity.append(f"generated config is missing: {config_path.name}")
        elif sha256_file(config_path) != row.get("config_file_sha256"):
            run_integrity.append("generated config file SHA-256 mismatch")

        if missing or not config_path.is_file():
            runs_report[condition] = {
                "run_id": run_id,
                "integrity_failures": run_integrity,
                "threshold_failures": run_thresholds,
            }
            integrity_failures.extend(f"{condition}: {item}" for item in run_integrity)
            continue

        try:
            planned = load_run_config(config_path, protocol_path)
            metrics = _read_json(run_dir / "seed_metrics.json")
            run_manifest = _read_json(run_dir / "run_manifest.json")
            resolved_raw = _read_json(run_dir / "resolved_config.json")
            resolved = validate_run_mapping(resolved_raw, protocol)
            events = _read_events(run_dir / "events.jsonl")
        except (ValueError, PilotValidationError) as exc:
            run_integrity.append(str(exc))
            runs_report[condition] = {
                "run_id": run_id,
                "integrity_failures": run_integrity,
                "threshold_failures": run_thresholds,
            }
            integrity_failures.extend(f"{condition}: {item}" for item in run_integrity)
            continue

        historical_failure: dict[str, Any] | None = None
        if failure_path.is_file():
            try:
                historical_failure = _read_json(failure_path)
            except PilotValidationError as exc:
                run_integrity.append(str(exc))
            else:
                if historical_failure.get("run_id") != run_id:
                    run_integrity.append("retained failure.json belongs to another run")
                if historical_failure.get("config_hash") != row.get("config_hash"):
                    run_integrity.append("retained failure.json config hash mismatch")

        expected_config_hash = str(row.get("config_hash"))
        expected_config = expected_configs[condition]
        if planned.as_dict() != expected_config.as_dict():
            run_integrity.append(
                "generated YAML differs from the in-memory protocol matrix"
            )
        if (
            planned.config_hash != expected_config_hash
            or resolved.config_hash != expected_config_hash
        ):
            run_integrity.append("planned/resolved scientific config hash mismatch")
        checks = {
            "metrics.run_id": (metrics.get("run_id"), run_id),
            "metrics.condition": (metrics.get("condition"), condition),
            "metrics.seed": (metrics.get("seed"), acceptance["seed"]),
            "metrics.dataset": (metrics.get("dataset"), acceptance["dataset"]),
            "metrics.config_hash": (metrics.get("config_hash"), expected_config_hash),
            "metrics.protocol_hash": (metrics.get("protocol_hash"), protocol_hash),
            "manifest.run_id": (run_manifest.get("run_id"), run_id),
            "manifest.config_hash": (run_manifest.get("config_hash"), expected_config_hash),
        }
        for label, (observed, expected) in checks.items():
            if observed != expected:
                run_integrity.append(f"{label}={observed!r}, expected {expected!r}")
        if metrics.get("status") != "complete" or metrics.get("failed") not in (0, "0", False):
            run_integrity.append("terminal metrics do not record a successful complete run")
        if run_manifest.get("status") != "complete":
            run_integrity.append("run manifest status is not complete")
        expected_orchestration = bound_evidence.get("run_manifest_evidence")
        if isinstance(expected_orchestration, Mapping):
            if run_manifest.get("orchestrator_evidence") != expected_orchestration:
                run_integrity.append(
                    "run-manifest orchestrator evidence differs from pilot plan/health report"
                )
            trainer_starts = [event for event in events if event.get("event") == "run_started"]
            if not trainer_starts or any(
                event.get("orchestrator_evidence") != expected_orchestration
                for event in trainer_starts
            ):
                run_integrity.append(
                    "trainer event stream lacks bound orchestrator evidence"
                )
        resume_history = run_manifest.get("resume_history", [])
        if run_id in historical_failure_run_ids:
            if not isinstance(resume_history, list) or not resume_history:
                run_integrity.append(
                    "success after a historical failure has no resume-history record"
                )
        elif resume_history not in ([], None):
            if not isinstance(resume_history, list):
                run_integrity.append("run-manifest resume_history is malformed")
        consolidated_row = consolidated_by_id.get(run_id) if consolidated_path.is_file() else None
        if not isinstance(consolidated_row, Mapping):
            run_integrity.append("run is missing from consolidated seed_metrics.csv")
        else:
            consolidated_checks = {
                "status": "complete",
                "failed": "0",
                "config_hash": expected_config_hash,
                "protocol_hash": protocol_hash,
            }
            for key, expected in consolidated_checks.items():
                if consolidated_row.get(key) != expected:
                    run_integrity.append(
                        f"consolidated {key}={consolidated_row.get(key)!r}, expected {expected!r}"
                    )

        environment_text = metrics.get("training_environment_identity")
        environment_hash = str(metrics.get("training_environment_sha256", ""))
        try:
            identity = json.loads(str(environment_text))
        except json.JSONDecodeError:
            identity = None
        if not isinstance(identity, Mapping):
            run_integrity.append("training environment identity is invalid")
        else:
            if _identity_sha256(identity) != environment_hash:
                run_integrity.append("training environment identity SHA-256 mismatch")
            environment_hashes.add(environment_hash)
            run_integrity.extend(_validate_environment(identity, acceptance["environment"]))
            manifest_environment = run_manifest.get("environment")
            if not isinstance(manifest_environment, Mapping) or (
                manifest_environment.get("training_environment_sha256") != environment_hash
            ):
                run_integrity.append("run-manifest training environment hash mismatch")
            if isinstance(resume_history, list):
                if any(
                    not isinstance(item, Mapping)
                    or item.get("checkpoint") != "last.pt"
                    or item.get("training_environment_sha256") != environment_hash
                    for item in resume_history
                ):
                    run_integrity.append(
                        "resume history is not bound to last.pt in the final environment"
                    )
            health = bound_evidence.get("health")
            health_environment = (
                health.get("environment") if isinstance(health, Mapping) else None
            )
            if isinstance(health_environment, Mapping):
                training_hardware = identity.get("hardware")
                health_hardware = health_environment.get("hardware")
                hardware_keys = (
                    "name",
                    "compute_capability",
                    "total_memory_bytes",
                    "multiprocessor_count",
                )
                if (
                    not isinstance(training_hardware, Mapping)
                    or not isinstance(health_hardware, Mapping)
                    or any(
                        training_hardware.get(key) != health_hardware.get(key)
                        for key in hardware_keys
                    )
                    or identity.get("device") != health_hardware.get("device")
                ):
                    run_integrity.append(
                        "training hardware/device differs from the bound health report"
                    )
                software = identity.get("software")
                if not isinstance(software, Mapping):
                    run_integrity.append("training software identity is missing")
                else:
                    software_checks = {
                        "platform": health_environment.get("platform"),
                        "python": health_environment.get("python"),
                        "pytorch": health_environment.get("pytorch"),
                        "numpy": health_environment.get("numpy"),
                        "cuda_version": health_environment.get("cuda_version"),
                    }
                    for key, expected in software_checks.items():
                        if software.get(key) != expected:
                            run_integrity.append(
                                f"training software {key} differs from the bound health report"
                            )
        shared_hash = str(metrics.get("shared_weight_sha256", ""))
        split_hash = str(metrics.get("split_manifest_sha256", ""))
        if not shared_hash or run_manifest.get("shared_weight_sha256") != shared_hash:
            run_integrity.append("shared-weight identity is missing or inconsistent")
        if not split_hash or run_manifest.get("split_manifest_sha256") != split_hash:
            run_integrity.append("split-manifest identity is missing or inconsistent")
        health = bound_evidence.get("health")
        health_source = health.get("source") if isinstance(health, Mapping) else None
        if isinstance(health_source, Mapping) and (
            split_hash != health_source.get("split_manifest_sha256")
        ):
            run_integrity.append("training split manifest differs from the health report")
        shared_weight_hashes.add(shared_hash)
        split_hashes.add(split_hash)

        epoch_events, resumed_event_count, resume_event_failures = _effective_epoch_events(
            events
        )
        run_integrity.extend(resume_event_failures)
        expected_resume_count = len(resume_history) if isinstance(resume_history, list) else 0
        if resumed_event_count != expected_resume_count:
            run_integrity.append(
                "trainer resumed-event count differs from run-manifest resume_history"
            )
        validation_thresholds = planned.analysis.get("validation_accuracy_thresholds")
        configured_threshold = (
            validation_thresholds.get(planned.data.dataset)
            if isinstance(validation_thresholds, Mapping)
            else None
        )
        if not _same_number(configured_threshold, accuracy_threshold):
            run_integrity.append(
                "configured convergence threshold differs from pilot acceptance"
            )
        reconstructed_history, reconstructed_best, convergence_epoch, epoch_failures = (
            _epoch_evidence(
                epoch_events,
                candidate_epochs=candidate_epochs,
                convergence_threshold=accuracy_threshold,
            )
        )
        run_integrity.extend(epoch_failures)
        activation_epoch = run_manifest.get("ta_activation_epoch_zero_based")
        expected_activation_epoch = int(
            math.ceil(planned.optimizer.epochs * planned.optimizer.ta_start_fraction)
        )
        if activation_epoch != expected_activation_epoch:
            run_integrity.append(
                f"TA activation epoch {activation_epoch!r} != {expected_activation_epoch}"
            )
            timing = {"pass": False, "status": "INVALID_ACTIVATION_EPOCH"}
        else:
            timing, timing_failures = _timing_summary(
                condition, epoch_events, activation_epoch, maximum_ratio
            )
            for item in timing_failures:
                if "ratio" in item and "exceeds" in item:
                    run_thresholds.append(item)
                else:
                    run_integrity.append(item)
        ta_names = run_manifest.get("ta_parameter_names")
        if condition in {"C2", "C4"}:
            if not isinstance(ta_names, list) or not ta_names:
                run_integrity.append(f"{condition} has no recorded TA parameters")
        elif ta_names != []:
            run_integrity.append(f"{condition} unexpectedly records TA parameters")

        best_accuracy: float | None = None
        if reconstructed_best is None:
            run_integrity.append("best validation evidence cannot be reconstructed")
        else:
            best_accuracy = float(reconstructed_best["accuracy"])
            best_loss = float(reconstructed_best["loss"])
            best_epoch = int(reconstructed_best["epoch"])
            metric_checks = {
                "best_epoch": metrics.get("best_epoch") == best_epoch + 1,
                "best_val_accuracy": _same_number(
                    metrics.get("best_val_accuracy"), best_accuracy
                ),
                "best_val_loss": _same_number(metrics.get("best_val_loss"), best_loss),
                "converged": metrics.get("converged")
                in ((1, "1", True) if convergence_epoch is not None else (0, "0", False)),
                "convergence_epoch": (
                    metrics.get("convergence_epoch") == convergence_epoch + 1
                    if convergence_epoch is not None
                    else metrics.get("convergence_epoch") in ("", None)
                ),
            }
            for label, passed in metric_checks.items():
                if not passed:
                    run_integrity.append(f"terminal metrics {label} differs from epoch evidence")
            if isinstance(consolidated_row, Mapping):
                consolidated_evidence = {
                    "best_epoch": str(best_epoch + 1),
                    "best_val_accuracy": best_accuracy,
                    "best_val_loss": best_loss,
                    "converged": "1" if convergence_epoch is not None else "0",
                    "convergence_epoch": (
                        str(convergence_epoch + 1) if convergence_epoch is not None else ""
                    ),
                }
                for label, expected in consolidated_evidence.items():
                    observed = consolidated_row.get(label)
                    matches = (
                        _same_csv_number(observed, expected)
                        if isinstance(expected, float)
                        else observed == expected
                    )
                    if not matches:
                        run_integrity.append(
                            f"consolidated {label} differs from epoch evidence"
                        )
            run_integrity.extend(
                _checkpoint_failures(
                    run_dir / "best.pt",
                    label="best.pt",
                    expected_epoch=best_epoch,
                    expected_best=reconstructed_best,
                    expected_history=reconstructed_history[: best_epoch + 1],
                    expected_config_hash=expected_config_hash,
                    expected_environment_hash=environment_hash,
                    protocol=protocol,
                )
            )
            run_integrity.extend(
                _checkpoint_failures(
                    run_dir / "last.pt",
                    label="last.pt",
                    expected_epoch=candidate_epochs - 1,
                    expected_best=reconstructed_best,
                    expected_history=reconstructed_history,
                    expected_config_hash=expected_config_hash,
                    expected_environment_hash=environment_hash,
                    protocol=protocol,
                )
            )
            if failed_checkpoint_path.is_file():
                try:
                    failed_checkpoint = load_checkpoint(
                        failed_checkpoint_path, map_location="cpu"
                    )
                except Exception as exc:
                    run_integrity.append(
                        "failed.pt cannot be loaded: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    if failed_checkpoint.get("config_hash") != expected_config_hash:
                        run_integrity.append("failed.pt config_hash mismatch")
                    if (
                        failed_checkpoint.get("training_environment_sha256")
                        != environment_hash
                    ):
                        run_integrity.append(
                            "failed.pt training-environment hash mismatch"
                        )
            if best_accuracy < accuracy_threshold:
                run_thresholds.append(
                    f"best validation accuracy {best_accuracy:.6f} is below "
                    f"{accuracy_threshold:.6f}"
                )
            if (
                acceptance["schedule"].get("require_all_conditions_converged") is True
                and convergence_epoch is None
            ):
                run_thresholds.append("validation accuracy never reached convergence threshold")

        runs_report[condition] = {
            "run_id": run_id,
            "best_validation_accuracy": best_accuracy,
            "required_best_validation_accuracy": accuracy_threshold,
            "timing": timing,
            "training_environment_sha256": metrics.get("training_environment_sha256"),
            "shared_weight_sha256": shared_hash,
            "split_manifest_sha256": split_hash,
            "historical_failure": historical_failure,
            "integrity_failures": run_integrity,
            "threshold_failures": run_thresholds,
        }
        integrity_failures.extend(f"{condition}: {item}" for item in run_integrity)
        threshold_failures.extend(f"{condition}: {item}" for item in run_thresholds)

    if len(environment_hashes) != 1:
        integrity_failures.append(
            f"training environments are not identical across four runs: {sorted(environment_hashes)}"
        )
    if len(shared_weight_hashes) != 1:
        integrity_failures.append(
            f"shared-weight hashes are not identical across four runs: {sorted(shared_weight_hashes)}"
        )
    if len(split_hashes) != 1:
        integrity_failures.append(
            f"split-manifest hashes are not identical across four runs: {sorted(split_hashes)}"
        )

    if integrity_failures:
        status = "INVALID"
        decision = "BLOCK_FORMAL_V2_AND_INVESTIGATE_PILOT_INTEGRITY"
        exit_code = 2
    elif threshold_failures:
        status = "FAIL"
        decision = "REQUIRE_FRESH_160_EPOCH_PILOT"
        exit_code = 1
    else:
        status = "PASS"
        decision = "ACCEPT_120_EPOCH_SCHEDULE_FOR_FORMAL_V2_DESIGN"
        exit_code = 0
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_class": ARTIFACT_CLASS,
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": status,
        "pass": status == "PASS",
        "decision": decision,
        "exit_code": exit_code,
        "validated_at": utc_now(),
        "protocol_path": str(protocol_path),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_hash": protocol_hash,
        "acceptance_hash": acceptance_hash,
        "config_dir": str(config_dir),
        "matrix_manifest_sha256": sha256_file(manifest_path),
        "run_manifest_csv_sha256": (
            sha256_file(run_manifest_csv_path) if run_manifest_csv_path.is_file() else None
        ),
        "health_report_path": bound_evidence.get("health_path"),
        "health_report_sha256": bound_evidence.get("health_sha256"),
        "pilot_plan_path": bound_evidence.get("plan_path"),
        "pilot_plan_sha256": bound_evidence.get("plan_sha256"),
        "results_root": str(results_root),
        "candidate_epochs": candidate_epochs,
        "fallback": {
            "epochs": int(acceptance["schedule"]["fallback_epochs"]),
            "action": acceptance["schedule"]["fallback_action"],
            "resume_from_120_epoch_pilot": False,
        },
        "runs": runs_report,
        "integrity_failures": integrity_failures,
        "threshold_failures": threshold_failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "protocol_v2_pilot.yaml",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v2_pilot_generated",
    )
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        acceptance = protocol["pilot_acceptance"]
        results_root = args.results_root or _absolute(acceptance["pilot_output_root"], PROJECT_ROOT)
        expected_output = _absolute(acceptance["validation_output"], PROJECT_ROOT)
        output = (args.output or expected_output).resolve()
        if output != expected_output:
            raise PilotValidationError(
                "--output cannot override pilot_acceptance.validation_output"
            )
        if output.exists():
            raise PilotValidationError(f"Refusing to overwrite validation report: {output}")
        report = validate_pilot(
            protocol_path=args.protocol,
            config_dir=args.config_dir,
            results_root=results_root,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, report)
    except (OSError, ValueError, PilotValidationError) as exc:
        print(f"V2_PILOT_VALIDATION_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"V2_PILOT_{report['status']}")
    print(f"DECISION={report['decision']}")
    print(f"VALIDATION_REPORT={output}")
    if report["status"] == "FAIL":
        print("DO_NOT_RESUME_120_EPOCH_OUTPUTS_IN_THE_160_EPOCH_PILOT")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
