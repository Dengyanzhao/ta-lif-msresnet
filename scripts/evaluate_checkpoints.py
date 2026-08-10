#!/usr/bin/env python3
"""Evaluate every frozen best checkpoint exactly once on its independent test set."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)

from talif_msresnet.config import (  # noqa: E402
    EXPECTED_RUN_COUNT,
    artifact_paths_for_protocol,
    expected_run_count_for_protocol,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.config_v6 import (  # noqa: E402
    canonicalize_v6_artifact_run_mapping,
    validate_v6_cifar100_provenance_files,
)
from talif_msresnet.benchmark_v6 import (  # noqa: E402
    V6BenchmarkError,
    validate_receipt as validate_v6_benchmark_receipt,
)
from talif_msresnet.data import build_test_loader  # noqa: E402
from talif_msresnet.freeze import verify_formal_freeze  # noqa: E402
from talif_msresnet.models import build_model  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.train import (  # noqa: E402
    SEED_METRIC_FIELDS,
    _formal_execution_evidence,
    _training_environment_identity,
    evaluate,
)
from talif_msresnet.utils import (  # noqa: E402
    JSONLLogger,
    atomic_write_json,
    load_checkpoint,
    resolve_device,
    seed_everything,
    sha256_file,
    upsert_csv_row,
    utc_now,
)


JOURNAL_NAME = "final_test.in_progress.json"
LOCK_NAME = "final_test.lock"


def _recovery_instruction(run_id: str) -> str:
    return (
        f"Inspect {run_id}/{JOURNAL_NAME}. If stage='results_ready', run this script with "
        f"--recover-run {run_id} to commit without touching test data again. For any earlier stage, "
        "test access is ambiguous: do not delete the journal or rerun automatically; document the "
        "incident and obtain an explicit author decision."
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_expected(
    config_dir: Path,
    expected_count: int = EXPECTED_RUN_COUNT,
) -> list[dict[str, str]]:
    manifest = config_dir / "run_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; generate the frozen matrix first")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "run_id", "experiment", "dataset", "depth", "time_steps", "condition", "topology",
            "neuron", "seed", "config_hash", "protocol_hash", "config_file", "config_file_sha256",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Frozen run_manifest.csv is missing columns: {sorted(missing)}")
        rows = list(reader)
    if (
        len(rows) != expected_count
        or len({row["run_id"] for row in rows}) != expected_count
    ):
        raise RuntimeError(
            f"Final test requires the complete {expected_count}-run matrix"
        )
    for row in rows:
        config_path = config_dir / row["config_file"]
        if not config_path.exists() or sha256_file(config_path) != row["config_file_sha256"]:
            raise RuntimeError(f"Frozen generated config file hash mismatch: {config_path}")
    return rows


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "seed_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Malformed {path}")
    return value


def _audit_consolidated_metrics(
    rows: list[dict[str, str]],
    results_root: Path,
    recovery_run: str | None = None,
    *,
    expected_training_environment_sha256: str | None = None,
) -> dict[str, dict[str, str]]:
    """Bind every result record to the plan and frozen training environment."""

    path = results_root / "seed_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing consolidated seed metrics: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(SEED_METRIC_FIELDS):
            raise RuntimeError(f"Consolidated seed_metrics.csv schema differs from the trainer schema: {path}")
        metrics_rows = list(reader)
    planned = {row["run_id"]: row for row in rows}
    ids = [row.get("run_id", "") for row in metrics_rows]
    expected_count = len(rows)
    if (
        len(metrics_rows) != expected_count
        or len(set(ids)) != expected_count
        or set(ids) != set(planned)
    ):
        missing = sorted(set(planned) - set(ids))
        extra = sorted(set(ids) - set(planned))
        raise RuntimeError(
            f"Consolidated seed_metrics.csv must contain exactly the {expected_count} "
            "unique planned runs; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    by_id = {row["run_id"]: row for row in metrics_rows}
    identity_fields = (
        "experiment", "dataset", "depth", "time_steps", "condition", "topology", "neuron", "seed",
    )
    test_fields = (
        "test_loss", "test_accuracy", "test_samples", "test_checkpoint_sha256", "test_evaluated_at",
    )
    errors: list[str] = []
    for run_id, plan in planned.items():
        metric = by_id[run_id]
        if metric.get("config_hash") != plan.get("config_hash"):
            errors.append(f"{run_id}: consolidated/planned config hashes differ")
        for field in identity_fields:
            if str(metric.get(field, "")) != str(plan.get(field, "")):
                errors.append(f"{run_id}: consolidated {field} differs from the plan")
        per_run = _load_metrics(results_root / run_id)
        manifest_path = results_root / run_id / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{run_id}: cannot read run_manifest.json: {exc}")
            continue
        manifest_config = manifest.get("config", {})
        manifest_analysis = manifest_config.get("analysis", {}) if isinstance(manifest_config, Mapping) else {}
        manifest_environment = manifest.get("environment", {})
        manifest_environment = (
            manifest_environment if isinstance(manifest_environment, Mapping) else {}
        )
        comparisons = {
            "config_hash": (
                metric.get("config_hash", ""), per_run.get("config_hash", ""),
                manifest.get("config_hash", ""), plan.get("config_hash", ""),
            ),
            "protocol_hash": (
                metric.get("protocol_hash", ""), per_run.get("protocol_hash", ""),
                manifest_analysis.get("protocol_hash", "") if isinstance(manifest_analysis, Mapping) else "",
                plan.get("protocol_hash", ""),
            ),
            "split_manifest_sha256": (
                metric.get("split_manifest_sha256", ""), per_run.get("split_manifest_sha256", ""),
                manifest.get("split_manifest_sha256", ""),
            ),
            "shared_weight_sha256": (
                metric.get("shared_weight_sha256", ""), per_run.get("shared_weight_sha256", ""),
                manifest.get("shared_weight_sha256", ""),
            ),
            "status": (metric.get("status", ""), per_run.get("status", "")),
        }
        for field, values in comparisons.items():
            normalized = {str(value) for value in values}
            if len(normalized) != 1 or "" in normalized:
                errors.append(f"{run_id}: {field} differs across plan/consolidated/per-run/manifest")
        if expected_training_environment_sha256 is not None:
            environment_bindings = {
                "consolidated seed_metrics.csv": metric.get(
                    "training_environment_sha256", ""
                ),
                "per-run seed_metrics.json": per_run.get(
                    "training_environment_sha256", ""
                ),
                "run_manifest.json environment.training_environment_sha256": (
                    manifest_environment.get("training_environment_sha256", "")
                ),
            }
            for source, observed in environment_bindings.items():
                if str(observed) != expected_training_environment_sha256:
                    errors.append(
                        f"{run_id}: {source} differs from the freeze-bound "
                        "training_environment_sha256"
                    )

        marker_exists = (results_root / run_id / "final_test.json").exists()
        marker_test_values: dict[str, Any] = {}
        if marker_exists:
            try:
                marker = json.loads((results_root / run_id / "final_test.json").read_text(encoding="utf-8"))
                marker_test_values = {
                    "test_loss": marker.get("test_loss"),
                    "test_accuracy": marker.get("test_accuracy"),
                    "test_samples": marker.get("test_samples"),
                    "test_checkpoint_sha256": marker.get("checkpoint_sha256"),
                    "test_evaluated_at": marker.get("evaluated_at"),
                }
            except Exception as exc:
                errors.append(f"{run_id}: cannot read committed final-test marker: {exc}")
        journal_path = results_root / run_id / JOURNAL_NAME
        recoverable = False
        recovery_expected: dict[str, Any] = {}
        if run_id == recovery_run and journal_path.exists():
            try:
                recovery_journal = json.loads(journal_path.read_text(encoding="utf-8"))
                recoverable = recovery_journal.get("stage") == "results_ready"
                result = recovery_journal.get("test", {})
                recovery_expected = {
                    "test_loss": result.get("loss"),
                    "test_accuracy": result.get("accuracy"),
                    "test_samples": result.get("samples"),
                    "test_checkpoint_sha256": recovery_journal.get("checkpoint_sha256"),
                    "test_evaluated_at": recovery_journal.get("evaluated_at"),
                }
            except Exception:
                recoverable = False
        for field in test_fields:
            csv_value = metric.get(field, "")
            per_value = per_run.get(field, "")
            if recoverable:
                expected = str(recovery_expected.get(field, ""))
                if any(str(value) not in ("", expected) for value in (csv_value, per_value)):
                    errors.append(f"{run_id}: stored {field} differs from the recoverable journal")
            elif str(csv_value) != str(per_value):
                errors.append(f"{run_id}: consolidated/per-run {field} differs")
            if marker_exists and str(csv_value) != str(marker_test_values.get(field, "")):
                errors.append(f"{run_id}: consolidated/marker {field} differs")
        any_test_value = any(
            value not in (None, "")
            for field in test_fields
            for value in (metric.get(field), per_run.get(field))
        )
        if any_test_value and not marker_exists and not recoverable:
            errors.append(f"{run_id}: test fields are populated without a committed marker")
    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:30])
        raise RuntimeError(f"Consolidated seed-metric audit failed:\n{preview}")
    return by_id


def _audit_all_frozen(
    rows: list[dict[str, str]],
    results_root: Path,
    protocol_hash: str,
    protocol: Mapping[str, Any],
    v6_test_source: Mapping[str, str] | None = None,
) -> None:
    is_v6 = int(protocol.get("protocol_version", 1)) == 6
    if is_v6:
        v6_test_source = _require_v6_test_source_binding(v6_test_source)
    errors: list[str] = []
    for row in rows:
        run_id = row["run_id"]
        run_dir = results_root / run_id
        checkpoint = run_dir / "best.pt"
        journal = run_dir / JOURNAL_NAME
        lock = run_dir / LOCK_NAME
        marker = run_dir / "final_test.json"
        if journal.exists() or lock.exists():
            if marker.exists():
                errors.append(
                    f"{run_id}: committed final test has stale transaction files; "
                    f"run --recover-run {run_id} to verify and clean them"
                )
            else:
                errors.append(f"{run_id}: interrupted final-test transaction. {_recovery_instruction(run_id)}")
        try:
            metrics = _load_metrics(run_dir)
        except Exception as exc:
            errors.append(f"{run_id}: {exc}")
            continue
        if metrics.get("status") != "complete":
            errors.append(f"{run_id}: status={metrics.get('status')!r}, expected 'complete'")
        if row.get("config_hash") and str(metrics.get("config_hash")) != row["config_hash"]:
            errors.append(f"{run_id}: seed metric config hash differs from generated manifest")
        if metrics.get("test_accuracy") not in (None, ""):
            if not marker.exists():
                errors.append(f"{run_id}: test metric exists without final_test.json audit marker")
        if not checkpoint.exists():
            errors.append(f"{run_id}: missing best.pt")
            continue
        checkpoint_sha256 = sha256_file(checkpoint)
        if marker.exists():
            try:
                marker_value = json.loads(marker.read_text(encoding="utf-8"))
                if marker_value.get("checkpoint_sha256") != checkpoint_sha256:
                    errors.append(f"{run_id}: final-test marker checkpoint hash differs from best.pt")
                if metrics.get("test_accuracy") in (None, ""):
                    errors.append(f"{run_id}: final-test marker exists but seed metric is empty")
                if is_v6:
                    try:
                        _validate_v6_marker_test_source_record(
                            marker_value,
                            v6_test_source,
                            label=f"{run_id}: final-test marker source",
                        )
                        _validate_v6_manifest_final_test_record(
                            _load_v6_run_manifest(run_dir, label=run_id),
                            v6_test_source,
                            checkpoint_sha256=checkpoint_sha256,
                            evaluated_at=marker_value.get("evaluated_at"),
                            samples=marker_value.get("test_samples"),
                            label=run_id,
                        )
                    except RuntimeError as exc:
                        errors.append(str(exc))
            except Exception as exc:
                errors.append(f"{run_id}: malformed final_test.json: {exc}")
        payload = load_checkpoint(checkpoint, map_location="cpu")
        config = _mapping(payload.get("config"), f"{run_id} checkpoint config")
        if int(protocol.get("protocol_version", 1)) == 6:
            try:
                normalized = canonicalize_v6_artifact_run_mapping(
                    config,
                    protocol,
                    project_root=PROJECT_ROOT,
                )
                resolved = validate_run_mapping(normalized, protocol)
            except Exception as exc:
                errors.append(f"{run_id}: checkpoint run contract is invalid: {exc}")
                continue
            if resolved.runtime.run_id != run_id:
                errors.append(f"{run_id}: checkpoint run_id differs from its directory")
            if resolved.config_hash != row.get("config_hash"):
                errors.append(f"{run_id}: checkpoint config differs from the frozen matrix")
        analysis = config.get("analysis", {})
        analysis = analysis if isinstance(analysis, Mapping) else {}
        if analysis.get("protocol_hash") != protocol_hash:
            errors.append(f"{run_id}: checkpoint protocol hash differs from frozen protocol")
        if str(payload.get("config_hash")) != str(metrics.get("config_hash")):
            errors.append(f"{run_id}: checkpoint and seed metric config hashes differ")
    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:30])
        suffix = f"\n... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise RuntimeError(f"Final-test freeze audit failed:\n{preview}{suffix}")


def _v6_benchmark_binding_rows(
    rows: list[dict[str, str]],
    metrics_by_id: Mapping[str, Mapping[str, Any]],
    results_root: Path,
) -> list[dict[str, Any]]:
    """Bind a V6 benchmark receipt to the actual formal checkpoint evidence."""

    bindings: list[dict[str, Any]] = []
    for row in rows:
        run_id = row["run_id"]
        metrics = metrics_by_id.get(run_id)
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"{run_id}: V6 benchmark binding has no audited metrics")
        checkpoint = results_root / run_id / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{run_id}: V6 benchmark binding is missing best.pt")
        bindings.append(
            {
                "run_id": run_id,
                "checkpoint_sha256": sha256_file(checkpoint),
                "config_hash": str(metrics.get("config_hash", "")),
                "training_environment_sha256": str(
                    metrics.get("training_environment_sha256", "")
                ),
                "split_manifest_sha256": str(metrics.get("split_manifest_sha256", "")),
                "shared_weight_sha256": str(metrics.get("shared_weight_sha256", "")),
            }
        )
    return bindings


def _v6_cifar100_test_source_record(protocol: Mapping[str, Any]) -> dict[str, str]:
    binding = validate_v6_cifar100_provenance_files(
        protocol,
        project_root=PROJECT_ROOT,
    )
    return {
        "test_source": binding["cifar100_test_pickle"],
        "test_source_sha256": binding["cifar100_test_pickle_sha256"],
        **binding,
    }


_V6_FINAL_TEST_SOURCE_KEYS = frozenset(
    {
        "test_source",
        "test_source_sha256",
        "cifar100_source_provenance",
        "cifar100_source_provenance_sha256",
        "cifar100_test_pickle",
        "cifar100_test_pickle_sha256",
        "cifar100_split_manifest",
        "cifar100_split_manifest_sha256",
    }
)


def _require_v6_test_source_binding(value: Mapping[str, str] | None) -> Mapping[str, str]:
    if value is None:
        raise RuntimeError("V6 final-test audit requires a frozen CIFAR-100 source binding")
    observed = set(value)
    if observed != _V6_FINAL_TEST_SOURCE_KEYS:
        raise RuntimeError(
            "V6 final-test source binding has an invalid field set: "
            f"missing={sorted(_V6_FINAL_TEST_SOURCE_KEYS - observed)}, "
            f"extra={sorted(observed - _V6_FINAL_TEST_SOURCE_KEYS)}"
        )
    if any(not isinstance(value[key], str) or not value[key].strip() for key in value):
        raise RuntimeError("V6 final-test source binding contains an empty or non-string value")
    return value


def _test_source_record(
    config: Any,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    if int(protocol.get("protocol_version", 1)) == 6:
        if config.data.dataset != "cifar100":
            raise RuntimeError("V6 final-test source binding requires CIFAR-100")
        return _v6_cifar100_test_source_record(protocol)
    if config.data.dataset != "cifar10dvs":
        return {
            "test_source": f"torchvision official {config.data.dataset} test partition",
            "test_source_sha256": "dataset implementation does not expose one aggregate file",
        }
    source = Path(str(config.data.test_frames_path))
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    index = source / "index.csv" if source.is_dir() else source
    source_hash = sha256_file(index)
    return {
        "test_source": artifact_path_reference(
            source,
            PROJECT_ROOT,
            external_identifier=f"index-sha256:{source_hash}",
        ),
        "test_source_sha256": source_hash,
    }


def _validate_v6_test_source_record(
    observed: Any,
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    _require_v6_test_source_binding(expected)
    if not isinstance(observed, Mapping) or dict(observed) != dict(expected):
        raise RuntimeError(f"{label} differs from the frozen V6 CIFAR-100 test-source binding")


def _validate_v6_marker_test_source_record(
    marker: Any,
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    if not isinstance(marker, Mapping):
        raise RuntimeError(f"{label} is not a JSON object")
    _validate_v6_test_source_record(
        {key: marker.get(key) for key in expected},
        expected,
        label=label,
    )


def _validate_v6_manifest_final_test_record(
    manifest: Any,
    expected_source: Mapping[str, str],
    *,
    checkpoint_sha256: str,
    evaluated_at: Any,
    samples: Any,
    label: str,
) -> None:
    """Require a committed V6 manifest record to match its final-test receipt."""

    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"{label} run manifest is not a JSON object")
    final_test = manifest.get("final_test")
    if not isinstance(final_test, Mapping):
        raise RuntimeError(f"{label} run manifest has no committed final_test record")
    expected_fields = {
        "status": "complete",
        "checkpoint": "best.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "evaluated_at": evaluated_at,
        "samples": samples,
    }
    mismatches = [
        key for key, expected in expected_fields.items() if final_test.get(key) != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"{label} run manifest final_test differs from its receipt at: "
            + ", ".join(mismatches)
        )
    _validate_v6_test_source_record(
        {key: final_test.get(key) for key in expected_source},
        expected_source,
        label=f"{label} run manifest final-test source",
    )


def _load_v6_run_manifest(run_dir: Path, *, label: str) -> Mapping[str, Any]:
    path = run_dir / "run_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} cannot read run manifest: {exc}") from exc
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} run manifest is not a JSON object")
    return value


def _transaction_paths(run_dir: Path) -> tuple[Path, Path]:
    return run_dir / JOURNAL_NAME, run_dir / LOCK_NAME


def _begin_transaction(run_dir: Path, journal: Mapping[str, Any]) -> tuple[Path, Path]:
    journal_path, lock_path = _transaction_paths(run_dir)
    if journal_path.exists() or lock_path.exists():
        raise RuntimeError(f"{run_dir.name}: interrupted/concurrent final test. {_recovery_instruction(run_dir.name)}")
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"{run_dir.name}: final-test lock already exists. {_recovery_instruction(run_dir.name)}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started_at={utc_now()}\n".encode("ascii"))
    finally:
        os.close(descriptor)
    try:
        atomic_write_json(journal_path, journal)
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return journal_path, lock_path


def _cleanup_transaction(run_dir: Path) -> None:
    journal_path, lock_path = _transaction_paths(run_dir)
    journal_path.unlink(missing_ok=True)
    lock_path.unlink(missing_ok=True)


def _commit_final_test(run_dir: Path, results_root: Path, journal: Mapping[str, Any]) -> None:
    if journal.get("stage") != "results_ready":
        raise RuntimeError(f"{run_dir.name}: transaction has no recoverable final-test result")
    result = _mapping(journal.get("test"), "transaction test result")
    checkpoint_hash = str(journal["checkpoint_sha256"])
    evaluated_at = str(journal["evaluated_at"])
    metrics = _load_metrics(run_dir)
    existing_accuracy = metrics.get("test_accuracy")
    if existing_accuracy not in (None, "") and not math.isclose(
        float(existing_accuracy), float(result["accuracy"]), rel_tol=0.0, abs_tol=1e-15,
    ):
        raise RuntimeError(f"{run_dir.name}: existing test metric differs from the transaction journal")
    metrics.update({
        "test_loss": float(result["loss"]),
        "test_accuracy": float(result["accuracy"]),
        "test_samples": int(result["samples"]),
        "test_checkpoint_sha256": checkpoint_hash,
        "test_evaluated_at": evaluated_at,
    })
    upsert_csv_row(
        results_root / "seed_metrics.csv", metrics, SEED_METRIC_FIELDS,
        key_field="run_id", require_existing=True,
    )
    atomic_write_json(run_dir / "seed_metrics.json", metrics)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test_source = dict(_mapping(journal.get("test_source"), "transaction test source"))
    manifest["final_test"] = {
        "status": "complete",
        "checkpoint": "best.pt",
        "checkpoint_sha256": checkpoint_hash,
        "evaluated_at": evaluated_at,
        "samples": int(result["samples"]),
        **test_source,
    }
    atomic_write_json(manifest_path, manifest)

    marker = {
        "format_version": 1,
        "status": "complete",
        "run_id": run_dir.name,
        "config_hash": journal["config_hash"],
        "protocol_hash": journal["protocol_hash"],
        "checkpoint": artifact_path_reference(
            run_dir / "best.pt",
            PROJECT_ROOT,
            external_identifier=f"checkpoint-sha256:{checkpoint_hash}",
        ),
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": int(journal["checkpoint_epoch_zero_based"]) + 1,
        "checkpoint_epoch_zero_based": int(journal["checkpoint_epoch_zero_based"]),
        "model_selection_rule": (
            "highest validation accuracy; ties resolved by lower validation loss; "
            "exact ties resolved by earliest epoch"
        ),
        "evaluated_at": evaluated_at,
        "device": journal["device"],
        "test_loss": float(result["loss"]),
        "test_accuracy": float(result["accuracy"]),
        "test_samples": int(result["samples"]),
        **test_source,
    }
    # This marker is the atomic commit record and is deliberately written last.
    atomic_write_json(run_dir / "final_test.json", marker)
    JSONLLogger(run_dir / "events.jsonl").log("final_test_completed", **marker)


def _recover_one(
    run_dir: Path,
    results_root: Path,
    protocol_hash: str,
    *,
    v6_test_source: Mapping[str, str] | None = None,
) -> str:
    if v6_test_source is not None:
        v6_test_source = _require_v6_test_source_binding(v6_test_source)
    journal_path, lock_path = _transaction_paths(run_dir)
    marker_path = run_dir / "final_test.json"
    if not journal_path.exists():
        if lock_path.exists():
            raise RuntimeError(
                f"{run_dir.name}: lock exists without a journal; test access cannot be established. "
                f"{_recovery_instruction(run_dir.name)}"
            )
        raise FileNotFoundError(f"No interrupted final-test journal for {run_dir.name}")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(run_dir / "best.pt")
    if journal.get("run_id") != run_dir.name or journal.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"{run_dir.name}: journal does not match the current best.pt")
    if journal.get("protocol_hash") != protocol_hash:
        raise RuntimeError(f"{run_dir.name}: journal protocol hash differs from the frozen protocol")
    if v6_test_source is not None:
        _validate_v6_test_source_record(
            journal.get("test_source"),
            v6_test_source,
            label=f"{run_dir.name}: final-test journal source",
        )
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") == "complete" and marker.get("checkpoint_sha256") == checkpoint_hash:
            if v6_test_source is not None:
                if journal.get("stage") != "results_ready":
                    raise RuntimeError(
                        f"{run_dir.name}: committed V6 marker has no results_ready journal"
                    )
                result = _mapping(journal.get("test"), "transaction test result")
                if (
                    marker.get("evaluated_at") != journal.get("evaluated_at")
                    or marker.get("test_samples") != result.get("samples")
                ):
                    raise RuntimeError(
                        f"{run_dir.name}: committed V6 marker differs from its transaction journal"
                    )
                _validate_v6_marker_test_source_record(
                    marker,
                    v6_test_source,
                    label=f"{run_dir.name}: final-test marker source",
                )
                _validate_v6_manifest_final_test_record(
                    _load_v6_run_manifest(run_dir, label=run_dir.name),
                    v6_test_source,
                    checkpoint_sha256=checkpoint_hash,
                    evaluated_at=journal.get("evaluated_at"),
                    samples=result.get("samples"),
                    label=run_dir.name,
                )
            _cleanup_transaction(run_dir)
            return "cleaned_committed_transaction"
        raise RuntimeError(f"{run_dir.name}: marker and transaction journal disagree")
    if journal.get("stage") != "results_ready":
        raise RuntimeError(f"{run_dir.name}: test access is ambiguous. {_recovery_instruction(run_dir.name)}")
    if v6_test_source is not None:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = _load_v6_run_manifest(run_dir, label=run_dir.name)
            if "final_test" in manifest:
                result = _mapping(journal.get("test"), "transaction test result")
                _validate_v6_manifest_final_test_record(
                    manifest,
                    v6_test_source,
                    checkpoint_sha256=checkpoint_hash,
                    evaluated_at=journal.get("evaluated_at"),
                    samples=result.get("samples"),
                    label=run_dir.name,
                )
    _commit_final_test(run_dir, results_root, journal)
    _cleanup_transaction(run_dir)
    return "recovered_without_test_access"


def _evaluate_one(
    run_dir: Path,
    device_name: str,
    results_root: Path,
    protocol: Mapping[str, Any],
    v6_test_source: Mapping[str, str] | None = None,
) -> str:
    if v6_test_source is not None:
        v6_test_source = _require_v6_test_source_binding(v6_test_source)
    checkpoint_path = run_dir / "best.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    marker_path = run_dir / "final_test.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("checkpoint_sha256") == checkpoint_hash and marker.get("status") == "complete":
            if v6_test_source is not None:
                _validate_v6_marker_test_source_record(
                    marker,
                    v6_test_source,
                    label=f"{run_dir.name}: final-test marker source",
                )
                _validate_v6_manifest_final_test_record(
                    _load_v6_run_manifest(run_dir, label=run_dir.name),
                    v6_test_source,
                    checkpoint_sha256=checkpoint_hash,
                    evaluated_at=marker.get("evaluated_at"),
                    samples=marker.get("test_samples"),
                    label=run_dir.name,
                )
            return "already_complete"
        raise FileExistsError(f"{marker_path} belongs to a different checkpoint; refusing to overwrite")
    journal_path, lock_path = _transaction_paths(run_dir)
    if journal_path.exists() or lock_path.exists():
        raise RuntimeError(f"{run_dir.name}: interrupted final test. {_recovery_instruction(run_dir.name)}")

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    raw_config = _mapping(payload.get("config"), "checkpoint config")
    if int(protocol.get("protocol_version", 1)) == 6:
        normalized = canonicalize_v6_artifact_run_mapping(
            raw_config,
            protocol,
            project_root=PROJECT_ROOT,
        )
        config = validate_run_mapping(normalized, protocol)
    else:
        config = validate_run_mapping(raw_config)
    if config.runtime.dry_run:
        raise RuntimeError(f"{run_dir.name}: dry-run checkpoints cannot be final-tested")
    if payload.get("config_hash") != config.config_hash:
        raise RuntimeError(f"{run_dir.name}: checkpoint scientific config hash is invalid")
    metrics = _load_metrics(run_dir)
    if metrics.get("test_accuracy") not in (None, ""):
        raise RuntimeError(f"{run_dir.name}: test_accuracy is already populated without a marker")
    runtime = dataclasses.replace(
        config.runtime, device=device_name, dry_run=False, limit_batches=None, resume=None,
    )
    config = dataclasses.replace(config, runtime=runtime, final_test=False)
    device = resolve_device(device_name)
    model_config = dataclasses.asdict(config.model)
    model_config["init_seed"] = config.runtime.seed
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)

    journal: dict[str, Any] = {
        "format_version": 1,
        "status": "in_progress",
        "stage": "prepared",
        "run_id": run_dir.name,
        "config_hash": config.config_hash,
        "protocol_hash": config.analysis.get("protocol_hash"),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch_zero_based": int(payload["epoch"]),
        "started_at": utc_now(),
        "device": str(device),
        "test_source": _test_source_record(config, protocol),
    }
    if v6_test_source is not None:
        _validate_v6_test_source_record(
            journal["test_source"],
            v6_test_source,
            label=f"{run_dir.name}: final-test source",
        )
    journal_path, _ = _begin_transaction(run_dir, journal)
    test_access_started = False
    try:
        journal["stage"] = "test_access_started"
        atomic_write_json(journal_path, journal)
        test_access_started = True
        loader = build_test_loader(config, seed=config.runtime.seed)
        test = evaluate(model, loader, device, config, "test")
        if not math.isfinite(float(test["accuracy"])) or not math.isfinite(float(test["loss"])):
            raise RuntimeError(f"{run_dir.name}: non-finite final-test metric")
        journal.update({
            "stage": "results_ready",
            "evaluated_at": utc_now(),
            "test": {
                "loss": float(test["loss"]),
                "accuracy": float(test["accuracy"]),
                "samples": int(test["samples"]),
            },
        })
        atomic_write_json(journal_path, journal)
        _commit_final_test(run_dir, results_root, journal)
        _cleanup_transaction(run_dir)
        return "evaluated"
    except BaseException:
        if not test_access_started:
            _cleanup_transaction(run_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root")
    parser.add_argument("--config-dir")
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="Evaluate the first N pending checkpoints after full freeze audit")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--recover-run",
        metavar="RUN_ID",
        help="Commit a results_ready transaction without constructing or reading the test loader again",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.recover_run and args.limit is not None:
        raise SystemExit("--recover-run cannot be combined with --limit")
    report = check_protocol(
        args.protocol,
        mode="final-test",
        project_root=PROJECT_ROOT,
        check_dependencies=True,
    )
    if not report.ok:
        details = "\n".join(f"- {item}" for item in report.errors)
        raise SystemExit(f"Final-test preflight blocked:\n{details}")
    protocol = load_protocol(args.protocol)
    artifacts = artifact_paths_for_protocol(protocol)
    default_results = PROJECT_ROOT / artifacts["formal_results"]
    default_configs = PROJECT_ROOT / artifacts["formal_matrix"]
    results_root = Path(args.results_root or default_results).resolve()
    config_dir = Path(args.config_dir or default_configs).resolve()
    protocol_version = int(protocol.get("protocol_version", 1))
    expected_training_environment_sha256: str | None = None
    v6_test_source: dict[str, str] | None = None
    if protocol_version == 6:
        try:
            v6_test_source = _v6_cifar100_test_source_record(protocol)
        except (OSError, ValueError) as exc:
            raise SystemExit(
                "V6 final-test blocked by CIFAR-100 source provenance: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    if protocol_version in (3, 4, 5, 6):
        if results_root != default_results.resolve():
            raise SystemExit(
                f"Protocol v{protocol_version} final test must use "
                "artifact_paths.formal_results"
            )
        if config_dir != default_configs.resolve():
            raise SystemExit(
                f"Protocol v{protocol_version} final test must use "
                "artifact_paths.formal_matrix"
            )
    if protocol_version in (4, 5, 6):
        freeze_manifest = verify_formal_freeze(
            project_root=PROJECT_ROOT,
            protocol_path=args.protocol,
            matrix_dir=config_dir,
        )
        execution_evidence = _formal_execution_evidence(
            protocol,
            freeze_manifest,
            protocol_path=args.protocol,
        )
        seed_everything(0, deterministic=True)
        selected_device = resolve_device(args.device)
        _identity, environment_sha256 = _training_environment_identity(
            selected_device,
            amp=False,
            deterministic=True,
        )
        if environment_sha256 != execution_evidence["training_environment_sha256"]:
            raise SystemExit(
                f"Protocol v{protocol_version} final-test environment differs from "
                "the pilot/formal freeze"
            )
        expected_training_environment_sha256 = str(
            execution_evidence["training_environment_sha256"]
        )
    rows = _read_expected(
        config_dir,
        expected_count=expected_run_count_for_protocol(protocol),
    )
    metrics_by_id = _audit_consolidated_metrics(
        rows,
        results_root,
        recovery_run=args.recover_run,
        expected_training_environment_sha256=expected_training_environment_sha256,
    )
    if protocol_version == 6:
        protocol_path = Path(args.protocol)
        protocol_path = (
            protocol_path if protocol_path.is_absolute() else PROJECT_ROOT / protocol_path
        ).resolve()
        manifest_path = (PROJECT_ROOT / artifacts["freeze_manifest"]).resolve()
        if not manifest_path.is_file():
            raise SystemExit("V6 final-test blocked: the verified freeze manifest is missing")
        try:
            validate_v6_benchmark_receipt(
                project_root=PROJECT_ROOT,
                protocol=protocol,
                protocol_path=protocol_path,
                protocol_hash=report.protocol_hash,
                expected_freeze_manifest_sha256=sha256_file(manifest_path),
                expected_formal_rows=_v6_benchmark_binding_rows(
                    rows,
                    metrics_by_id,
                    results_root,
                ),
            )
        except (OSError, V6BenchmarkError, RuntimeError) as exc:
            raise SystemExit(
                "V6 final-test blocked by validation-only benchmark receipt: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    if args.recover_run:
        planned_ids = {row["run_id"] for row in rows}
        if args.recover_run not in planned_ids:
            raise SystemExit(f"Unknown planned run_id: {args.recover_run}")
        status = _recover_one(
            results_root / args.recover_run,
            results_root,
            report.protocol_hash,
            v6_test_source=v6_test_source,
        )
        print(f"{args.recover_run}: {status}")
        return 0
    _audit_all_frozen(
        rows,
        results_root,
        report.protocol_hash,
        protocol,
        v6_test_source=v6_test_source,
    )
    pending = [row for row in rows if not (results_root / row["run_id"] / "final_test.json").exists()]
    if args.limit is not None:
        pending = pending[: args.limit]
    failures = 0
    for index, row in enumerate(pending, start=1):
        run_dir = results_root / row["run_id"]
        try:
            status = _evaluate_one(
                run_dir,
                args.device,
                results_root,
                protocol,
                v6_test_source=v6_test_source,
            )
            print(f"[{index}/{len(pending)}] {row['run_id']}: {status}")
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(pending)}] {row['run_id']}: FAILED: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
    print(f"Final-test pass complete: {len(pending) - failures} evaluated, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
