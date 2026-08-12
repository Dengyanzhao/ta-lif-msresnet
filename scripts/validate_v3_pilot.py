#!/usr/bin/env python3
"""Validate both independent non-reportable TA-LIF-only v3 pilot blocks."""

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
    RunConfig,
    artifact_paths_for_protocol,
    generate_v3_pilot_matrix,
    load_protocol,
    load_run_config,
    validate_run_mapping,
)
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.pilot_v3 import (  # noqa: E402
    ATTEMPT_STATUS,
    REPORTING_ELIGIBILITY,
    V3_PILOT_CONDITIONS,
    V3_PILOT_DATASETS,
    PilotBlock,
    PilotV3Error,
    expected_pilot_plan_payload,
    require_v3_author_freeze,
    resolve_pilot_block,
    validate_attempt_receipt,
    validate_health_report,
)
from talif_msresnet.utils import (  # noqa: E402
    atomic_write_json,
    load_checkpoint,
    sha256_file,
    stable_hash,
    utc_now,
)


SCHEMA_VERSION = 1
ARTIFACT_CLASS = "NON_REPORTABLE_V3_TALIF_ONLY_PILOT_ACCEPTANCE"
PASS_DECISION = "ACCEPT_V3_TALIF_ONLY_120_EPOCH_PILOT"
FAIL_DECISION = "BLOCK_V3_AND_REQUIRE_NEW_PROTOCOL_AND_UNUSED_PILOT_SEEDS"
INVALID_DECISION = "BLOCK_V3_AND_INVESTIGATE_PILOT_EVIDENCE_INTEGRITY"


class PilotValidationError(RuntimeError):
    """Raised when the validator cannot establish the v3 pilot contract."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotValidationError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotValidationError(f"{label} must be a JSON object: {path}")
    return value


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PilotValidationError(f"Cannot read event log {path}: {exc}") from exc
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PilotValidationError(
                f"Malformed JSON at {path}:{line_number}: {exc}"
            ) from exc
        if not isinstance(item, dict):
            raise PilotValidationError(
                f"Event at {path}:{line_number} is not a JSON object"
            )
        events.append(item)
    return events


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _identity_sha256(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _same_number(observed: Any, expected: float) -> bool:
    value = _finite_number(observed)
    return value is not None and math.isclose(
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
            _evidence_equal(left, right)
            for left, right in zip(observed, expected)
        )
    if (
        isinstance(observed, (int, float))
        and not isinstance(observed, bool)
        and isinstance(expected, (int, float))
        and not isinstance(expected, bool)
    ):
        left = float(observed)
        right = float(expected)
        return (math.isnan(left) and math.isnan(right)) or left == right
    return observed == expected


def _effective_epoch_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], int, list[str]]:
    """Keep only the final checkpoint-consistent history across allowed resumes."""

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
                if isinstance(item.get("epoch"), int)
                and int(item["epoch"]) < start_epoch
            ]
            if [item.get("epoch") for item in effective] != list(range(start_epoch)):
                failures.append(
                    f"resumed start_epoch {start_epoch} has no complete checkpoint prefix"
                )
        elif event_name == "epoch_completed":
            effective.append(event)
    return effective, resume_count, failures


def _epoch_evidence(
    events: Sequence[Mapping[str, Any]],
    *,
    epochs: int,
    convergence_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, int | None, list[str]]:
    failures: list[str] = []
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    convergence_epoch: int | None = None
    if [event.get("epoch") for event in events] != list(range(epochs)):
        failures.append(f"epoch_completed sequence is not exactly 0..{epochs - 1}")
    for index, event in enumerate(events):
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
        row.update(
            {"epoch": epoch, "val_accuracy": val_accuracy, "val_loss": val_loss}
        )
        history.append(row)
        if best is None or val_accuracy > float(best["accuracy"]) or (
            val_accuracy == float(best["accuracy"])
            and val_loss < float(best["loss"])
        ):
            best = {"accuracy": val_accuracy, "loss": val_loss, "epoch": epoch}
        if convergence_epoch is None and val_accuracy >= convergence_threshold:
            convergence_epoch = epoch
        if best is not None and not _best_val_equal(event.get("best_val"), best):
            failures.append(
                f"epoch {epoch}: cumulative best_val evidence is inconsistent"
            )
    if len(history) != epochs:
        failures.append(
            f"reconstructed checkpoint history has {len(history)} rows, expected {epochs}"
        )
    return history, best, convergence_epoch, failures


def _validate_environment(
    identity: Mapping[str, Any], expected: Mapping[str, Any]
) -> list[str]:
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
    expected_gpu = str(expected["expected_gpu_substring"])
    if expected_gpu.lower() not in str(hardware.get("name", "")).lower():
        failures.append(
            f"GPU name {hardware.get('name')!r} does not contain {expected_gpu!r}"
        )
    if software.get("pytorch") != expected["pytorch_version"]:
        failures.append("training PyTorch version differs from pilot_acceptance")
    if str(software.get("cuda_version")) != str(expected["cuda_runtime"]):
        failures.append("training CUDA runtime differs from pilot_acceptance")
    if identity.get("precision") != expected["precision"]:
        failures.append("training precision differs from pilot_acceptance")
    if bool(determinism.get("requested")) is not bool(expected["deterministic"]):
        failures.append("determinism.requested differs from pilot_acceptance")
    if bool(expected["deterministic"]):
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
            failures.append("determinism.cublas_workspace_config is not approved")
    return failures


def _health_environment_failures(
    identity: Mapping[str, Any], health: Mapping[str, Any]
) -> list[str]:
    failures: list[str] = []
    health_environment = health.get("environment")
    if not isinstance(health_environment, Mapping):
        return ["bound health report has no environment evidence"]
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
        failures.append("training hardware/device differs from the health report")
    software = identity.get("software")
    if not isinstance(software, Mapping):
        failures.append("training software identity is missing")
    else:
        for key in ("platform", "python", "pytorch", "numpy", "cuda_version"):
            if software.get(key) != health_environment.get(key):
                failures.append(
                    f"training software {key} differs from the health report"
                )
    return failures


def _timing_summary(
    condition: str,
    epoch_events: Sequence[Mapping[str, Any]],
    *,
    activation_epoch: int,
    maximum_ratio: float,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    expects_ta = condition == "C2"
    frozen: list[float] = []
    enabled: list[float] = []
    for event in epoch_events:
        epoch = event.get("epoch")
        if not isinstance(epoch, int):
            failures.append("timing event has an invalid epoch")
            continue
        expected_enabled = bool(expects_ta and epoch >= activation_epoch)
        if event.get("ta_enabled") is not expected_enabled:
            failures.append(
                f"{condition} epoch {epoch}: ta_enabled={event.get('ta_enabled')!r}, "
                f"expected {expected_enabled}"
            )
        train = event.get("train")
        seconds = _finite_number(
            train.get("seconds") if isinstance(train, Mapping) else None
        )
        if seconds is None or seconds <= 0:
            failures.append(f"{condition} epoch {epoch}: train.seconds is invalid")
            continue
        (enabled if expected_enabled else frozen).append(seconds)
    summary: dict[str, Any] = {
        "activation_epoch_zero_based": activation_epoch,
        "frozen_epoch_count": len(frozen),
        "enabled_epoch_count": len(enabled),
        "statistic": (
            "median_enabled_train_seconds_over_median_frozen_train_seconds"
        ),
        "maximum_allowed": maximum_ratio,
    }
    if not expects_ta:
        summary.update({"status": "NOT_APPLICABLE_LIF", "pass": not failures})
        return summary, failures
    if not frozen or not enabled:
        failures.append("C2 requires both frozen and enabled timing epochs")
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
            f"C2 enabled/frozen epoch-time ratio {ratio:.6f} exceeds "
            f"{maximum_ratio:.6f}"
        )
    return summary, failures


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
    expected_output_dir: str | None = None,
    actual_output_root: Path | None = None,
    artifact_config_normalizer: Any | None = None,
) -> list[str]:
    try:
        checkpoint = load_checkpoint(path, map_location="cpu")
    except Exception as exc:
        return [f"{label} cannot be loaded: {type(exc).__name__}: {exc}"]
    failures: list[str] = []
    if not isinstance(checkpoint.get("model_state"), Mapping) or not checkpoint.get(
        "model_state"
    ):
        failures.append(f"{label} model_state is missing or empty")
    optimizer_state = checkpoint.get("optimizer_state")
    if not isinstance(optimizer_state, Mapping) or not {
        "state",
        "param_groups",
    }.issubset(optimizer_state):
        failures.append(f"{label} optimizer_state is incomplete")
    if not isinstance(checkpoint.get("scheduler_state"), Mapping) or not checkpoint.get(
        "scheduler_state"
    ):
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
    embedded_raw = checkpoint.get("config")
    if expected_output_dir is not None and actual_output_root is not None:
        embedded_raw = _canonicalize_execution_output_dir(
            embedded_raw,
            expected_output_dir=expected_output_dir,
            actual_output_root=actual_output_root,
        )
    if artifact_config_normalizer is not None:
        try:
            embedded_raw = artifact_config_normalizer(embedded_raw)
        except Exception as exc:  # noqa: BLE001 - report artifact-integrity failures.
            failures.append(
                f"{label} embedded config normalization failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return failures
    try:
        embedded = validate_run_mapping(embedded_raw, protocol)
    except (TypeError, ValueError) as exc:
        failures.append(f"{label} embedded config is invalid: {exc}")
    else:
        if embedded.config_hash != expected_config_hash:
            failures.append(f"{label} embedded scientific config mismatch")
    return failures


def _canonicalize_execution_output_dir(
    raw: Any,
    *,
    expected_output_dir: str,
    actual_output_root: Path,
) -> Any:
    """Normalize only an absolute spelling of the already-bound output root."""

    if not isinstance(raw, Mapping):
        return raw
    runtime = raw.get("runtime")
    if not isinstance(runtime, Mapping):
        return raw
    observed_value = runtime.get("output_dir")
    if observed_value == expected_output_dir or not isinstance(observed_value, str):
        return raw
    observed = Path(observed_value)
    if not observed.is_absolute():
        return raw
    try:
        equivalent = observed.resolve() == actual_output_root.resolve()
    except (OSError, RuntimeError, ValueError):
        equivalent = False
    if not equivalent:
        return raw
    normalized = dict(raw)
    normalized_runtime = dict(runtime)
    normalized_runtime["output_dir"] = expected_output_dir
    normalized["runtime"] = normalized_runtime
    return normalized


def _load_generated_matrix(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    repository_root: Path,
) -> tuple[
    list[Mapping[str, Any]], dict[tuple[str, str], RunConfig], list[str], Path, Path
]:
    failures: list[str] = []
    manifest_path = config_dir / "matrix_manifest.json"
    csv_path = config_dir / "run_manifest.csv"
    manifest = _read_json(manifest_path, "v3 pilot matrix manifest")
    raw_rows = manifest.get("runs")
    if not isinstance(raw_rows, list):
        raise PilotValidationError("matrix manifest runs must be a list")
    rows = [row for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        failures.append("matrix manifest contains a non-object run")
    expected_raw = generate_v3_pilot_matrix(protocol)
    expected_configs = [validate_run_mapping(row, protocol) for row in expected_raw]
    expected_by_key = {
        (config.data.dataset, config.model.condition): config
        for config in expected_configs
    }
    expected_order = [
        (config.data.dataset, config.model.condition, config.runtime.run_id)
        for config in expected_configs
    ]
    observed_order = [
        (str(row.get("dataset")), str(row.get("condition")), str(row.get("run_id")))
        for row in rows
    ]
    if observed_order != expected_order:
        failures.append("matrix manifest is not the ordered two-dataset C1/C2 pilot")
    expected_seeds = sorted(
        int(profile["seed"])
        for profile in protocol["pilot_acceptance"]["datasets"].values()
    )
    checks = {
        "protocol": artifact_path_reference(protocol_path, repository_root),
        "protocol_hash": stable_hash(protocol),
        "run_count": 4,
        "conditions": list(V3_PILOT_CONDITIONS),
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
        normalized = [
            {key: str(value) for key, value in row.items()} for row in rows
        ]
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
    except PilotV3Error as exc:
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
            "reference_config": artifact_path_reference(
                config_path, repository_root
            ),
            "reference_config_sha256": (
                sha256_file(config_path) if config_path.is_file() else None
            ),
            "reference_config_hash": row.get("config_hash"),
            "status": ATTEMPT_STATUS,
        }
        for key, value in expected_receipt.items():
            if receipt.get(key) != value:
                failures.append(f"attempt receipt {key} mismatch")
    report = _read_json(block.health_output, "v3 health report")
    receipt_sha256 = sha256_file(block.attempt_receipt)
    failures.extend(
        _basic_health_failures(report, block=block, receipt_sha256=receipt_sha256)
    )
    if strict_health_validation:
        try:
            report = validate_health_report(
                block.health_output,
                protocol_path,
                block.dataset,
                repository_root=repository_root,
                require_current_tracked_clean=True,
            )
        except PilotV3Error as exc:
            failures.append(f"canonical health validation failed: {exc}")
    plan = _read_json(block.pilot_plan, "v3 pilot plan")
    try:
        expected_plan = expected_pilot_plan_payload(
            block=block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=manifest_rows,
            health_report=report,
            repository_root=repository_root,
        )
    except PilotV3Error as exc:
        failures.append(str(exc))
    else:
        if plan != expected_plan:
            failures.append("pilot plan differs from the exact dataset-bound payload")
    return {
        "health": report,
        "health_path": artifact_path_reference(block.health_output, repository_root),
        "health_sha256": sha256_file(block.health_output),
        "attempt_receipt_path": artifact_path_reference(
            block.attempt_receipt, repository_root
        ),
        "attempt_receipt_sha256": receipt_sha256,
        "plan_path": artifact_path_reference(block.pilot_plan, repository_root),
        "plan_sha256": sha256_file(block.pilot_plan),
    }, failures


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
    accuracy_threshold: float,
    maximum_ratio: float,
    allow_equivalent_absolute_output_dir: bool = False,
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
    config_path = config_dir / str(row.get("config_file", ""))
    if not config_path.is_file():
        integrity.append(f"generated config is missing: {config_path.name}")
    elif sha256_file(config_path) != row.get("config_file_sha256"):
        integrity.append("generated config file SHA-256 mismatch")
    if missing or not config_path.is_file():
        return {
            "run_id": run_id,
            "integrity_failures": integrity,
            "threshold_failures": thresholds,
        }, None, None, None
    try:
        planned = load_run_config(config_path, protocol_path)
        metrics = _read_json(run_dir / "seed_metrics.json", "seed metrics")
        run_manifest = _read_json(run_dir / "run_manifest.json", "run manifest")
        resolved_raw = _read_json(
            run_dir / "resolved_config.json", "resolved configuration"
        )
        if allow_equivalent_absolute_output_dir:
            resolved_raw = _canonicalize_execution_output_dir(
                resolved_raw,
                expected_output_dir=expected_config.runtime.output_dir,
                actual_output_root=block.pilot_output_root,
            )
        resolved = validate_run_mapping(resolved_raw, protocol)
        events = _read_events(run_dir / "events.jsonl")
    except (ValueError, PilotValidationError) as exc:
        integrity.append(str(exc))
        return {
            "run_id": run_id,
            "integrity_failures": integrity,
            "threshold_failures": thresholds,
        }, None, None, None
    expected_config_hash = expected_config.config_hash
    if planned.as_dict() != expected_config.as_dict():
        integrity.append("generated YAML differs from the in-memory v3 pilot matrix")
    if row.get("config_hash") != expected_config_hash:
        integrity.append("matrix row config_hash differs from the expected config")
    if planned.config_hash != expected_config_hash or resolved.config_hash != expected_config_hash:
        integrity.append("planned/resolved scientific config hash mismatch")
    checks = {
        "metrics.run_id": (metrics.get("run_id"), run_id),
        "metrics.condition": (metrics.get("condition"), condition),
        "metrics.seed": (metrics.get("seed"), block.seed),
        "metrics.dataset": (metrics.get("dataset"), block.dataset),
        "metrics.config_hash": (metrics.get("config_hash"), expected_config_hash),
        "metrics.protocol_hash": (metrics.get("protocol_hash"), block.protocol_hash),
        "manifest.run_id": (run_manifest.get("run_id"), run_id),
        "manifest.config_hash": (run_manifest.get("config_hash"), expected_config_hash),
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
        if environment_hash_value is None or _identity_sha256(identity) != environment_hash:
            integrity.append("training environment identity SHA-256 mismatch")
        integrity.extend(
            _validate_environment(identity, protocol["pilot_acceptance"]["environment"])
        )
        integrity.extend(_health_environment_failures(identity, health))
        manifest_environment = run_manifest.get("environment")
        if not isinstance(manifest_environment, Mapping) or (
            manifest_environment.get("training_environment_sha256") != environment_hash
        ):
            integrity.append("run-manifest training environment hash mismatch")

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
    health_source = health.get("source")
    if not isinstance(health_source, Mapping) or (
        split_hash != health_source.get("split_manifest_sha256")
    ):
        integrity.append("training split manifest differs from the health report")

    epoch_events, resumed_count, resume_failures = _effective_epoch_events(events)
    integrity.extend(resume_failures)
    expected_resume_count = len(resume_history) if isinstance(resume_history, list) else 0
    if resumed_count != expected_resume_count:
        integrity.append("resumed-event count differs from run-manifest resume_history")
    configured_thresholds = planned.analysis.get("validation_accuracy_thresholds")
    configured_threshold = (
        configured_thresholds.get(block.dataset)
        if isinstance(configured_thresholds, Mapping)
        else None
    )
    if not _same_number(configured_threshold, accuracy_threshold):
        integrity.append("configured convergence threshold differs from pilot acceptance")
    history, best, convergence_epoch, epoch_failures = _epoch_evidence(
        epoch_events,
        epochs=epochs,
        convergence_threshold=accuracy_threshold,
    )
    integrity.extend(epoch_failures)
    activation_epoch = run_manifest.get("ta_activation_epoch_zero_based")
    expected_activation = int(
        math.ceil(planned.optimizer.epochs * planned.optimizer.ta_start_fraction)
    )
    if activation_epoch != expected_activation:
        integrity.append(
            f"TA activation epoch {activation_epoch!r} != {expected_activation}"
        )
        timing: dict[str, Any] = {
            "pass": False,
            "status": "INVALID_ACTIVATION_EPOCH",
        }
    else:
        timing, timing_failures = _timing_summary(
            condition,
            epoch_events,
            activation_epoch=expected_activation,
            maximum_ratio=maximum_ratio,
        )
        for failure in timing_failures:
            if "ratio" in failure and "exceeds" in failure:
                thresholds.append(failure)
            else:
                integrity.append(failure)
    ta_names = run_manifest.get("ta_parameter_names")
    if condition == "C2":
        if not isinstance(ta_names, list) or not ta_names:
            integrity.append("C2 has no recorded TA parameters")
    elif ta_names != []:
        integrity.append("C1 unexpectedly records TA parameters")

    best_accuracy: float | None = None
    if best is None:
        integrity.append("best validation evidence cannot be reconstructed")
    else:
        best_accuracy = float(best["accuracy"])
        best_epoch = int(best["epoch"])
        best_loss = float(best["loss"])
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
                integrity.append(f"terminal metrics {label} differs from epoch evidence")
        integrity.extend(
            _checkpoint_failures(
                run_dir / "best.pt",
                label="best.pt",
                expected_epoch=best_epoch,
                expected_best=best,
                expected_history=history[: best_epoch + 1],
                expected_config_hash=expected_config_hash,
                expected_environment_hash=environment_hash,
                protocol=protocol,
                expected_output_dir=(
                    expected_config.runtime.output_dir
                    if allow_equivalent_absolute_output_dir
                    else None
                ),
                actual_output_root=(
                    block.pilot_output_root
                    if allow_equivalent_absolute_output_dir
                    else None
                ),
            )
        )
        integrity.extend(
            _checkpoint_failures(
                run_dir / "last.pt",
                label="last.pt",
                expected_epoch=epochs - 1,
                expected_best=best,
                expected_history=history,
                expected_config_hash=expected_config_hash,
                expected_environment_hash=environment_hash,
                protocol=protocol,
                expected_output_dir=(
                    expected_config.runtime.output_dir
                    if allow_equivalent_absolute_output_dir
                    else None
                ),
                actual_output_root=(
                    block.pilot_output_root
                    if allow_equivalent_absolute_output_dir
                    else None
                ),
            )
        )
        if best_accuracy < accuracy_threshold:
            thresholds.append(
                f"best validation accuracy {best_accuracy:.6f} is below "
                f"{accuracy_threshold:.6f}"
            )
        if (
            protocol["pilot_acceptance"]["schedule"].get(
                "require_all_conditions_converged"
            )
            is True
            and convergence_epoch is None
        ):
            thresholds.append("validation accuracy never reached convergence threshold")
    return {
        "run_id": run_id,
        "best_validation_accuracy": best_accuracy,
        "required_best_validation_accuracy": accuracy_threshold,
        "timing": timing,
        "training_environment_sha256": environment_hash_value,
        "shared_weight_sha256": shared_hash_value,
        "split_manifest_sha256": split_hash_value,
        "integrity_failures": integrity,
        "threshold_failures": thresholds,
    }, environment_hash_value, shared_hash_value, split_hash_value


def validate_pilot(
    *,
    protocol_path: str | Path,
    config_dir: str | Path,
    repository_root: str | Path = PROJECT_ROOT,
    strict_health_validation: bool = True,
) -> dict[str, Any]:
    """Validate two independent C1/C2 blocks and return one aggregate verdict."""

    root = Path(repository_root).resolve()
    protocol_path = Path(protocol_path).resolve()
    config_dir = Path(config_dir).resolve()
    protocol = load_protocol(protocol_path)
    if protocol.get("protocol_version") != 3:
        raise PilotValidationError("This validator accepts only protocol_version 3")
    require_v3_author_freeze(protocol, repository_root=root)
    artifact_paths = artifact_paths_for_protocol(protocol)
    expected_protocol_path = (root / artifact_paths["protocol"]).resolve()
    expected_config_dir = (root / artifact_paths["pilot_matrix"]).resolve()
    if protocol_path != expected_protocol_path:
        raise PilotValidationError(
            f"Protocol path differs from the v3 artifact binding: {protocol_path}"
        )
    if config_dir != expected_config_dir:
        raise PilotValidationError(
            f"Config directory differs from the v3 artifact binding: {config_dir}"
        )
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise PilotValidationError("Protocol has no pilot_acceptance mapping")
    schedule = acceptance.get("schedule")
    timing_acceptance = acceptance.get("timing")
    if not isinstance(schedule, Mapping) or not isinstance(timing_acceptance, Mapping):
        raise PilotValidationError("Protocol pilot schedule/timing mapping is missing")
    epochs = int(schedule.get("epochs", 0))
    if epochs != 120:
        raise PilotValidationError("The v3 pilot schedule must be exactly 120 epochs")
    accuracy_threshold = float(schedule["required_best_validation_accuracy"])
    if not math.isclose(accuracy_threshold, 0.60, rel_tol=0.0, abs_tol=0.0):
        raise PilotValidationError("The v3 pilot accuracy threshold must be exactly 0.60")
    if timing_acceptance.get("epoch_time_ratio_conditions") != ["C2"]:
        raise PilotValidationError("The v3 pilot timing condition must be exactly C2")
    if (
        timing_acceptance.get("ta_state_source")
        != "events_jsonl_epoch_completed_ta_enabled"
    ):
        raise PilotValidationError("Unsupported v3 pilot TA timing state source")

    rows, expected_configs, matrix_failures, manifest_path, csv_path = (
        _load_generated_matrix(
            protocol=protocol,
            protocol_path=protocol_path,
            config_dir=config_dir,
            repository_root=root,
        )
    )
    aggregate_integrity = list(matrix_failures)
    aggregate_thresholds: list[str] = []
    datasets_report: dict[str, Any] = {}
    maximum_ratio = float(
        timing_acceptance["maximum_ta_enabled_over_frozen_ratio"]
    )

    for dataset in V3_PILOT_DATASETS:
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
            for condition in V3_PILOT_CONDITIONS
        }
        if block.pilot_output_root.is_dir():
            observed_run_ids = {
                path.name
                for path in block.pilot_output_root.iterdir()
                if path.is_dir() and path.name.startswith("E1_")
            }
        else:
            observed_run_ids = set()
            block_integrity.append(
                f"pilot results root is missing: {block.pilot_output_root}"
            )
        if observed_run_ids != expected_run_ids:
            block_integrity.append(
                "pilot run directories differ from the exact two-run block: "
                f"observed={sorted(observed_run_ids)} "
                f"expected={sorted(expected_run_ids)}"
            )

        runs: dict[str, Any] = {}
        environment_hashes: set[str] = set()
        shared_hashes: set[str] = set()
        split_hashes: set[str] = set()
        rows_by_condition = {
            str(row.get("condition")): row for row in block_rows
        }
        for condition in V3_PILOT_CONDITIONS:
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
                accuracy_threshold=accuracy_threshold,
                maximum_ratio=maximum_ratio,
            )
            runs[condition] = run_report
            block_integrity.extend(
                f"{condition}: {failure}"
                for failure in run_report["integrity_failures"]
            )
            block_thresholds.extend(
                f"{condition}: {failure}"
                for failure in run_report["threshold_failures"]
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
        block_status = (
            "INVALID"
            if block_integrity
            else "FAIL"
            if block_thresholds
            else "PASS"
        )
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
        aggregate_integrity.extend(
            f"{dataset}: {failure}" for failure in block_integrity
        )
        aggregate_thresholds.extend(
            f"{dataset}: {failure}" for failure in block_thresholds
        )

    if aggregate_integrity:
        status = "INVALID"
        decision = INVALID_DECISION
        exit_code = 2
    elif aggregate_thresholds:
        status = "FAIL"
        decision = FAIL_DECISION
        exit_code = 1
    else:
        status = "PASS"
        decision = PASS_DECISION
        exit_code = 0
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
        "run_manifest_csv_sha256": sha256_file(csv_path)
        if csv_path.is_file()
        else None,
        "required_epochs": epochs,
        "required_best_validation_accuracy": accuracy_threshold,
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
        default=PROJECT_ROOT / "configs" / "protocol_v3_talif_only.yaml",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "v3_talif_only_pilot_generated",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol = load_protocol(args.protocol)
        acceptance = protocol["pilot_acceptance"]
        expected_output = (
            PROJECT_ROOT / str(acceptance["validation_output"])
        ).resolve()
        output = (args.output or expected_output).resolve()
        if output != expected_output:
            raise PilotValidationError(
                "--output cannot override pilot_acceptance.validation_output"
            )
        if output.exists():
            raise PilotValidationError(
                f"Refusing to overwrite validation report: {output}"
            )
        report = validate_pilot(
            protocol_path=args.protocol,
            config_dir=args.config_dir,
            repository_root=PROJECT_ROOT,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output, report)
    except (OSError, ValueError, PilotV3Error, PilotValidationError) as exc:
        print(
            f"V3_PILOT_VALIDATION_BLOCKED: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"V3_PILOT_{report['status']}")
    print(f"DECISION={report['decision']}")
    print(f"VALIDATION_REPORT={output}")
    if report["status"] != "PASS":
        print("V3_BLOCKED_NEW_PROTOCOL_AND_UNUSED_PILOT_SEEDS_REQUIRED")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
