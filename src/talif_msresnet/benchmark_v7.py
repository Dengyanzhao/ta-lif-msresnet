"""Strict validation-only benchmark evidence for the frozen V7 study.

The V7 benchmark is deliberately separate from the historical generic benchmark
scripts.  It records descriptive latency, memory, activity, and operation
proxies for every validation-selected formal checkpoint, never estimates energy,
and produces a receipt that is required before one-time test evaluation.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config_v7 import (
    V7_ACTIVE_CONDITIONS,
    V7_BENCHMARK_CONTRACT,
    V7_BENCHMARK_SEED,
    V7_FORMAL_SEEDS,
    V7_OPERATION_MODE_BY_CONDITION,
    validate_v7_protocol,
)
from .utils import sha256_file, stable_hash

RESULT_SCHEMA = "ta-lif-msresnet-v7-validation-benchmark-v1"
RECEIPT_SCHEMA = "ta-lif-msresnet-v7-validation-benchmark-receipt-v1"
RESULT_ARTIFACT_CLASS = "V7_VALIDATION_ONLY_CHECKPOINT_RESOURCE_AUDIT"
RECEIPT_ARTIFACT_CLASS = "V7_VALIDATION_ONLY_48_CHECKPOINT_BENCHMARK_ACCEPTANCE"
BATCH_FILENAME = "validation_batch.pt"
RECEIPT_FILENAME = "benchmark_completion.json"
_HEX = frozenset("0123456789abcdef")


class V7BenchmarkError(RuntimeError):
    """Raised when V7 benchmark evidence is incomplete or inconsistent."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX)
    )


def _is_git_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and set(value).issubset(_HEX)
    )


def _inside(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise V7BenchmarkError(f"{label} escapes the repository: {resolved}") from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise V7BenchmarkError(f"Artifact path escapes the repository: {path}") from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V7BenchmarkError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V7BenchmarkError(f"{label} must be a JSON object: {path}")
    return value


def expected_run_ids() -> tuple[str, ...]:
    return tuple(
        f"E9_cifar100_d20_t6_{condition}_s{seed}"
        for seed in V7_FORMAL_SEEDS
        for condition in V7_ACTIVE_CONDITIONS
    )


def benchmark_root(protocol: Mapping[str, Any], project_root: str | Path) -> Path:
    validated = validate_v7_protocol(protocol)
    root = Path(project_root).resolve()
    return _inside(root, str(validated["artifact_paths"]["benchmark_results"]), "benchmark root")


def formal_results_root(protocol: Mapping[str, Any], project_root: str | Path) -> Path:
    validated = validate_v7_protocol(protocol)
    root = Path(project_root).resolve()
    return _inside(root, str(validated["artifact_paths"]["formal_results"]), "formal results root")


def expected_batch_path(protocol: Mapping[str, Any], project_root: str | Path) -> Path:
    return benchmark_root(protocol, project_root) / BATCH_FILENAME


def expected_receipt_path(protocol: Mapping[str, Any], project_root: str | Path) -> Path:
    return benchmark_root(protocol, project_root) / RECEIPT_FILENAME


def canonical_bound_paths(
    protocol: Mapping[str, Any], project_root: str | Path
) -> dict[str, Path]:
    """Resolve every repository artifact whose identity is frozen in V7 receipts."""

    validated = validate_v7_protocol(protocol)
    root = Path(project_root).resolve()
    paths = validated["artifact_paths"]
    return {
        "protocol": _inside(root, str(paths["protocol"]), "V7 protocol"),
        "matrix_manifest": _inside(
            root,
            Path(str(paths["formal_matrix"])) / "matrix_manifest.json",
            "V7 formal matrix manifest",
        ),
        "freeze_manifest": _inside(
            root, str(paths["freeze_manifest"]), "V7 freeze manifest"
        ),
    }


def repository_git_identity(project_root: str | Path) -> tuple[str, bool]:
    """Return the checked-out commit and tracked-tree cleanliness.

    Untracked experiment artifacts are intentionally ignored.  Every tracked
    source or protocol modification remains a hard failure.
    """

    root = Path(project_root).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V7BenchmarkError(f"Cannot establish V7 Git identity: {exc}") from exc
    if not _is_git_commit(commit):
        raise V7BenchmarkError(f"V7 Git commit is invalid: {commit!r}")
    return commit, not bool(status.strip())


def _canonical_path_record(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
) -> dict[str, str]:
    paths = canonical_bound_paths(protocol, project_root)
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise V7BenchmarkError(
            "V7 bound artifact is missing: "
            + ", ".join(f"{label}={paths[label]}" for label in missing)
        )
    return {
        "protocol_path": _relative(project_root, paths["protocol"]),
        "protocol_file_sha256": sha256_file(paths["protocol"]),
        "matrix_manifest_path": _relative(project_root, paths["matrix_manifest"]),
        "matrix_manifest_sha256": sha256_file(paths["matrix_manifest"]),
        "freeze_manifest_path": _relative(project_root, paths["freeze_manifest"]),
        "freeze_manifest_sha256": sha256_file(paths["freeze_manifest"]),
    }


def _validate_canonical_path_fields(
    value: Mapping[str, Any],
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    protocol_file_sha256: str,
    matrix_manifest_sha256: str,
    freeze_manifest_sha256: str,
    label: str,
) -> dict[str, str]:
    expected = _canonical_path_record(project_root=project_root, protocol=protocol)
    expected_hashes = {
        "protocol_file_sha256": protocol_file_sha256,
        "matrix_manifest_sha256": matrix_manifest_sha256,
        "freeze_manifest_sha256": freeze_manifest_sha256,
    }
    for field, expected_hash in expected_hashes.items():
        if not _is_sha256(expected_hash):
            raise V7BenchmarkError(f"{label} expected {field} is not a SHA-256")
        if expected[field] != expected_hash:
            raise V7BenchmarkError(
                f"{label} current {field} differs from the frozen expected hash"
            )
    for field in ("protocol_path", "matrix_manifest_path", "freeze_manifest_path"):
        observed = value.get(field)
        if not isinstance(observed, str) or Path(observed).is_absolute():
            raise V7BenchmarkError(f"{label} {field} must be repository-relative")
        if observed != expected[field]:
            raise V7BenchmarkError(f"{label} {field} is not canonical")
        _inside(project_root, observed, f"{label} {field}")
    for field in expected_hashes:
        if value.get(field) != expected[field]:
            raise V7BenchmarkError(f"{label} {field} differs from the bound file")
    return expected


def expected_result_path(
    protocol: Mapping[str, Any], project_root: str | Path, run_id: str
) -> Path:
    if run_id not in expected_run_ids():
        raise V7BenchmarkError(f"Unknown V7 formal run id: {run_id}")
    return benchmark_root(protocol, project_root) / "runs" / f"{run_id}.json"


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V7BenchmarkError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise V7BenchmarkError(f"{label} must be a finite number")
    return number


def _nonnegative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise V7BenchmarkError(f"{label} must be a non-negative integer")
    return int(value)


def _validate_hardware(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V7BenchmarkError("benchmark hardware must be a mapping")
    required = {"device", "device_name", "device_uuid", "driver_version", "torch", "cuda"}
    if set(value) != required:
        raise V7BenchmarkError("benchmark hardware has an unexpected schema")
    copied = dict(value)
    for key in required:
        if not isinstance(copied[key], str) or not copied[key].strip():
            raise V7BenchmarkError(f"benchmark hardware.{key} must be non-empty")
    if not copied["device"].startswith("cuda:"):
        raise V7BenchmarkError("benchmark hardware.device must be an explicit CUDA device")
    return copied


def _validate_input(value: Any, *, batch_file_sha256: str, batch_hash: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V7BenchmarkError("benchmark input must be a mapping")
    required = {
        "source_split",
        "test_data_accessed",
        "file_sha256",
        "batch_sha256",
        "input_seed",
        "batch_size",
    }
    if set(value) != required:
        raise V7BenchmarkError("benchmark input has an unexpected schema")
    copied = dict(value)
    if copied["source_split"] != "validation" or copied["test_data_accessed"] is not False:
        raise V7BenchmarkError("V7 benchmark input must be validation-only")
    if copied["file_sha256"] != batch_file_sha256 or copied["batch_sha256"] != batch_hash:
        raise V7BenchmarkError("V7 benchmark input hash differs from the fixed batch")
    if copied["input_seed"] != V7_BENCHMARK_SEED or copied["batch_size"] != 128:
        raise V7BenchmarkError("V7 benchmark input seed or batch size differs from the protocol")
    return copied


def _validate_measurements(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise V7BenchmarkError("benchmark measurements must be a mapping")
    required = {"parameters", "batch_1", "batch_128"}
    if set(value) != required:
        raise V7BenchmarkError("benchmark measurements have an unexpected schema")
    parameters = value["parameters"]
    if not isinstance(parameters, Mapping) or set(parameters) != {"total", "trainable"}:
        raise V7BenchmarkError("benchmark parameter report has an unexpected schema")
    _nonnegative_integer(parameters["total"], "measurements.parameters.total")
    _nonnegative_integer(parameters["trainable"], "measurements.parameters.trainable")
    copied: dict[str, Any] = {"parameters": dict(parameters)}
    batch_required = {
        "batch_size",
        "latency_mean_ms",
        "latency_sd_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "peak_allocated_bytes",
        "incremental_peak_bytes",
        "spike_rate",
        "synaptic_operation_proxy_per_sample",
        "dense_mac_equivalents_per_sample",
        "shared_threshold_window_accesses_per_sample",
        "threshold_bank_accesses_per_sample",
        "spike_count_updates_per_sample",
        "neuron_operation_mode",
        "operation_count_method",
    }
    for label, expected_size in (("batch_1", 1), ("batch_128", 128)):
        record = value[label]
        if not isinstance(record, Mapping) or set(record) != batch_required:
            raise V7BenchmarkError(f"benchmark measurements.{label} has an unexpected schema")
        if record["batch_size"] != expected_size:
            raise V7BenchmarkError(f"benchmark measurements.{label} has an invalid batch size")
        for field in (
            "latency_mean_ms",
            "latency_sd_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "spike_rate",
            "synaptic_operation_proxy_per_sample",
            "dense_mac_equivalents_per_sample",
            "shared_threshold_window_accesses_per_sample",
            "threshold_bank_accesses_per_sample",
            "spike_count_updates_per_sample",
        ):
            _finite(record[field], f"measurements.{label}.{field}")
        for field in ("peak_allocated_bytes", "incremental_peak_bytes"):
            _nonnegative_integer(record[field], f"measurements.{label}.{field}")
        if record["neuron_operation_mode"] not in set(V7_OPERATION_MODE_BY_CONDITION.values()):
            raise V7BenchmarkError(
                f"measurements.{label}.neuron_operation_mode is invalid"
            )
        if not isinstance(record["operation_count_method"], str) or not record[
            "operation_count_method"
        ].strip():
            raise V7BenchmarkError(f"measurements.{label}.operation_count_method is missing")
        copied[label] = dict(record)
    return copied


def validate_result(
    result: Mapping[str, Any],
    *,
    run_id: str,
    protocol_hash: str,
    git_commit: str,
    checkpoint_sha256: str,
    config_hash: str,
    training_environment_sha256: str,
    split_manifest_sha256: str,
    shared_weight_sha256: str,
    batch_file_sha256: str,
    batch_hash: str,
    protocol_file_sha256: str,
    matrix_manifest_sha256: str,
    freeze_manifest_sha256: str,
    project_root: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable V7 benchmark result against its checkpoint evidence."""

    required = {
        "schema",
        "artifact_class",
        "status",
        "protocol_version",
        "protocol_hash",
        "git_commit",
        "tracked_clean",
        "protocol_path",
        "protocol_file_sha256",
        "matrix_manifest_path",
        "matrix_manifest_sha256",
        "freeze_manifest_path",
        "freeze_manifest_sha256",
        "run_id",
        "checkpoint_sha256",
        "config_hash",
        "training_environment_sha256",
        "split_manifest_sha256",
        "shared_weight_sha256",
        "input",
        "benchmark_contract",
        "hardware",
        "measurements",
        "energy",
        "completed_at",
    }
    if set(result) != required:
        raise V7BenchmarkError(f"{run_id}: benchmark result has an unexpected schema")
    expected = {
        "schema": RESULT_SCHEMA,
        "artifact_class": RESULT_ARTIFACT_CLASS,
        "status": "complete",
        "protocol_version": 7,
        "protocol_hash": protocol_hash,
        "git_commit": git_commit,
        "tracked_clean": True,
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_sha256,
        "config_hash": config_hash,
        "training_environment_sha256": training_environment_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "shared_weight_sha256": shared_weight_sha256,
        "benchmark_contract": V7_BENCHMARK_CONTRACT,
        "energy": {"status": "not_measured", "claim": "forbidden"},
    }
    failures = [key for key, value in expected.items() if result.get(key) != value]
    if failures:
        raise V7BenchmarkError(
            f"{run_id}: benchmark result differs from the frozen contract: {', '.join(failures)}"
        )
    if not _is_git_commit(git_commit):
        raise V7BenchmarkError(f"{run_id}: benchmark Git commit is invalid")
    _validate_canonical_path_fields(
        result,
        project_root=Path(project_root).resolve(),
        protocol=protocol,
        protocol_file_sha256=protocol_file_sha256,
        matrix_manifest_sha256=matrix_manifest_sha256,
        freeze_manifest_sha256=freeze_manifest_sha256,
        label=f"{run_id} benchmark result",
    )
    if not all(
        _is_sha256(value)
        for value in (
            checkpoint_sha256,
            config_hash,
            training_environment_sha256,
            split_manifest_sha256,
            shared_weight_sha256,
        )
    ):
        raise V7BenchmarkError(f"{run_id}: benchmark binding has an invalid SHA-256")
    _validate_input(result["input"], batch_file_sha256=batch_file_sha256, batch_hash=batch_hash)
    _validate_hardware(result["hardware"])
    measurements = _validate_measurements(result["measurements"])
    condition = run_id.split("_t6_", 1)[1].rsplit("_s", 1)[0]
    expected_mode = V7_OPERATION_MODE_BY_CONDITION[condition]
    for label in ("batch_1", "batch_128"):
        record = measurements[label]
        if record["neuron_operation_mode"] != expected_mode:
            raise V7BenchmarkError(
                f"{run_id}: benchmark operation mode differs from frozen condition {condition}"
            )
        shared = float(record["shared_threshold_window_accesses_per_sample"])
        bank = float(record["threshold_bank_accesses_per_sample"])
        count = float(record["spike_count_updates_per_sample"])
        method = str(record["operation_count_method"])
        if expected_mode not in method:
            raise V7BenchmarkError(
                f"{run_id}: benchmark operation method does not identify {expected_mode}"
            )
        if expected_mode == "none" and (shared != 0.0 or bank != 0.0 or count != 0.0):
            raise V7BenchmarkError(f"{run_id}: fixed-neuron benchmark has adaptive control counts")
        if expected_mode == "shared_window" and (shared <= 0.0 or bank != 0.0 or count != 0.0):
            raise V7BenchmarkError(f"{run_id}: shared-window benchmark has bank/history counts")
        if expected_mode == "time_indexed_bank" and (shared != 0.0 or bank <= 0.0 or count != 0.0):
            raise V7BenchmarkError(f"{run_id}: time-indexed benchmark has shared/history counts")
        if expected_mode == "count_indexed_bank" and (shared != 0.0 or bank <= 0.0 or count <= 0.0):
            raise V7BenchmarkError(f"{run_id}: count-indexed benchmark has shared-window counts")
    if not isinstance(result["completed_at"], str) or not result["completed_at"].strip():
        raise V7BenchmarkError(f"{run_id}: benchmark completed_at is missing")
    return dict(result)


def make_benchmark_id(
    *,
    protocol_hash: str,
    git_commit: str,
    run_id: str,
    checkpoint_sha256: str,
    batch_file_sha256: str,
    batch_hash: str,
    hardware: Mapping[str, Any],
    protocol_file_sha256: str,
    matrix_manifest_sha256: str,
    freeze_manifest_sha256: str,
) -> str:
    return stable_hash(
        {
            "protocol_hash": protocol_hash,
            "git_commit": git_commit,
            "run_id": run_id,
            "checkpoint_sha256": checkpoint_sha256,
            "batch_file_sha256": batch_file_sha256,
            "batch_hash": batch_hash,
            "hardware": dict(hardware),
            "protocol_file_sha256": protocol_file_sha256,
            "matrix_manifest_sha256": matrix_manifest_sha256,
            "freeze_manifest_sha256": freeze_manifest_sha256,
            "benchmark_contract": V7_BENCHMARK_CONTRACT,
        }
    )


def build_receipt(
    *,
    project_root: str | Path,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    git_commit: str,
    tracked_clean: bool,
    protocol_file_sha256: str,
    matrix_manifest_sha256: str,
    freeze_manifest_sha256: str,
    batch_file_sha256: str,
    batch_hash: str,
    hardware: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    """Build, but do not write, the all-48 benchmark completion receipt."""

    validated = validate_v7_protocol(protocol)
    root = Path(project_root).resolve()
    current_commit, current_clean = repository_git_identity(root)
    if (
        not _is_git_commit(git_commit)
        or git_commit != current_commit
        or tracked_clean is not True
        or current_clean is not True
    ):
        raise V7BenchmarkError(
            "V7 benchmark receipt requires the current clean tracked release commit"
        )
    path_record = _validate_canonical_path_fields(
        {
            **_canonical_path_record(project_root=root, protocol=validated),
        },
        project_root=root,
        protocol=validated,
        protocol_file_sha256=protocol_file_sha256,
        matrix_manifest_sha256=matrix_manifest_sha256,
        freeze_manifest_sha256=freeze_manifest_sha256,
        label="V7 benchmark receipt",
    )
    expected_ids = expected_run_ids()
    if len(results) != len(expected_ids):
        raise V7BenchmarkError("V7 benchmark receipt requires exactly 48 result rows")
    observed_ids = tuple(str(row.get("run_id", "")) for row in results)
    if observed_ids != expected_ids:
        raise V7BenchmarkError("V7 benchmark result order differs from the frozen matrix")
    hardware_record = _validate_hardware(hardware)
    entries: list[dict[str, Any]] = []
    training_environments: set[str] = set()
    split_hashes: set[str] = set()
    shared_by_seed: dict[int, set[str]] = {}
    for row in results:
        run_id = str(row["run_id"])
        result_path = expected_result_path(validated, root, run_id)
        if not result_path.is_file():
            raise V7BenchmarkError(f"{run_id}: benchmark result is missing")
        result_hash = sha256_file(result_path)
        result = _read_json(result_path, f"{run_id} benchmark result")
        validate_result(
            result,
            run_id=run_id,
            protocol_hash=protocol_hash,
            git_commit=git_commit,
            checkpoint_sha256=str(row["checkpoint_sha256"]),
            config_hash=str(row["config_hash"]),
            training_environment_sha256=str(row["training_environment_sha256"]),
            split_manifest_sha256=str(row["split_manifest_sha256"]),
            shared_weight_sha256=str(row["shared_weight_sha256"]),
            batch_file_sha256=batch_file_sha256,
            batch_hash=batch_hash,
            protocol_file_sha256=protocol_file_sha256,
            matrix_manifest_sha256=matrix_manifest_sha256,
            freeze_manifest_sha256=freeze_manifest_sha256,
            project_root=root,
            protocol=validated,
        )
        if result["hardware"] != hardware_record:
            raise V7BenchmarkError(f"{run_id}: benchmark hardware differs within the matrix")
        seed = row.get("seed")
        condition = row.get("condition")
        if seed not in V7_FORMAL_SEEDS or condition not in V7_ACTIVE_CONDITIONS:
            raise V7BenchmarkError(f"{run_id}: result audit row has an invalid seed or condition")
        training_environments.add(str(row["training_environment_sha256"]))
        split_hashes.add(str(row["split_manifest_sha256"]))
        shared_by_seed.setdefault(int(seed), set()).add(str(row["shared_weight_sha256"]))
        entries.append(
            {
                "run_id": run_id,
                "seed": int(seed),
                "condition": str(condition),
                "benchmark_id": make_benchmark_id(
                    protocol_hash=protocol_hash,
                    git_commit=git_commit,
                    run_id=run_id,
                    checkpoint_sha256=str(row["checkpoint_sha256"]),
                    batch_file_sha256=batch_file_sha256,
                    batch_hash=batch_hash,
                    hardware=hardware_record,
                    protocol_file_sha256=protocol_file_sha256,
                    matrix_manifest_sha256=matrix_manifest_sha256,
                    freeze_manifest_sha256=freeze_manifest_sha256,
                ),
                "checkpoint_sha256": str(row["checkpoint_sha256"]),
                "config_hash": str(row["config_hash"]),
                "shared_weight_sha256": str(row["shared_weight_sha256"]),
                "result_path": _relative(root, result_path),
                "result_sha256": result_hash,
            }
        )
    if len(training_environments) != 1 or not _is_sha256(next(iter(training_environments), None)):
        raise V7BenchmarkError("V7 benchmark runs do not share one valid training environment")
    if len(split_hashes) != 1 or not _is_sha256(next(iter(split_hashes), None)):
        raise V7BenchmarkError("V7 benchmark runs do not share one valid split manifest")
    if set(shared_by_seed) != set(V7_FORMAL_SEEDS) or any(
        len(values) != 1 or not _is_sha256(next(iter(values), None))
        for values in shared_by_seed.values()
    ):
        raise V7BenchmarkError(
            "V7 benchmark conditions do not share one valid initialization within each seed"
        )
    for field, value in (
        ("protocol_hash", protocol_hash),
        ("protocol_file_sha256", protocol_file_sha256),
        ("matrix_manifest_sha256", matrix_manifest_sha256),
        ("freeze_manifest_sha256", freeze_manifest_sha256),
        ("batch_file_sha256", batch_file_sha256),
        ("batch_hash", batch_hash),
    ):
        if not _is_sha256(value):
            raise V7BenchmarkError(f"{field} must be a SHA-256")
    if not isinstance(created_at, str) or not created_at.strip():
        raise V7BenchmarkError("receipt created_at is missing")
    transaction_names = ("final_test.json", "final_test.in_progress.json", "final_test.lock")
    present_transactions = [
        f"{run_id}/{name}"
        for run_id in expected_ids
        for name in transaction_names
        if (formal_results_root(validated, root) / run_id / name).exists()
    ]
    if present_transactions:
        raise V7BenchmarkError(
            "V7 benchmark receipt must precede all final-test markers/transactions: "
            + ", ".join(present_transactions[:5])
        )
    return {
        "schema": RECEIPT_SCHEMA,
        "artifact_class": RECEIPT_ARTIFACT_CLASS,
        "status": "PASS",
        "protocol_version": 7,
        "protocol_hash": protocol_hash,
        "git_commit": git_commit,
        "tracked_clean": True,
        **path_record,
        "formal_results_root": _relative(root, formal_results_root(validated, root)),
        "benchmark_root": _relative(root, benchmark_root(validated, root)),
        "input": {
            "source_split": "validation",
            "test_data_accessed": False,
            "file_sha256": batch_file_sha256,
            "batch_sha256": batch_hash,
            "input_seed": V7_BENCHMARK_SEED,
            "batch_size": 128,
        },
        "hardware": hardware_record,
        "training_environment_sha256": next(iter(training_environments)),
        "split_manifest_sha256": next(iter(split_hashes)),
        "benchmark_contract": dict(V7_BENCHMARK_CONTRACT),
        "final_test_markers_at_completion": 0,
        "run_count": len(entries),
        "runs": entries,
        "created_at": created_at,
    }


def exclusive_write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Write an audit artifact exactly once without an overwrite race."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{target}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    temporary: Path | None = None
    try:
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite immutable V7 artifact: {target}")
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def validate_receipt(
    *,
    project_root: str | Path,
    protocol: Mapping[str, Any],
    protocol_path: str | Path,
    protocol_hash: str,
    expected_git_commit: str,
    expected_matrix_manifest_sha256: str,
    expected_freeze_manifest_sha256: str,
    expected_formal_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Read and fail-closed validate the frozen 48-checkpoint receipt."""

    validated = validate_v7_protocol(protocol)
    root = Path(project_root).resolve()
    canonical_protocol_path = _inside(
        root,
        str(validated["artifact_paths"]["protocol"]),
        "V7 protocol",
    )
    requested_protocol_path = _inside(root, protocol_path, "V7 protocol")
    if requested_protocol_path != canonical_protocol_path:
        raise V7BenchmarkError(
            "V7 benchmark validation requires the canonical protocol path"
        )
    if not canonical_protocol_path.is_file():
        raise V7BenchmarkError(f"V7 canonical protocol is missing: {canonical_protocol_path}")
    current_commit, current_clean = repository_git_identity(root)
    if (
        not _is_git_commit(expected_git_commit)
        or current_commit != expected_git_commit
        or current_clean is not True
    ):
        raise V7BenchmarkError(
            "V7 benchmark receipt validation requires the current clean release commit"
        )
    path_record = _canonical_path_record(project_root=root, protocol=validated)
    protocol_file_sha256 = path_record["protocol_file_sha256"]
    receipt_path = expected_receipt_path(validated, root)
    if not receipt_path.is_file():
        raise V7BenchmarkError(
            "V7 validation-only benchmark receipt is missing; final-test access is blocked"
        )
    receipt = _read_json(receipt_path, "V7 benchmark receipt")
    required = {
        "schema",
        "artifact_class",
        "status",
        "protocol_version",
        "protocol_hash",
        "git_commit",
        "tracked_clean",
        "protocol_path",
        "protocol_file_sha256",
        "matrix_manifest_path",
        "matrix_manifest_sha256",
        "freeze_manifest_path",
        "freeze_manifest_sha256",
        "formal_results_root",
        "benchmark_root",
        "input",
        "hardware",
        "training_environment_sha256",
        "split_manifest_sha256",
        "benchmark_contract",
        "final_test_markers_at_completion",
        "run_count",
        "runs",
        "created_at",
    }
    if set(receipt) != required:
        raise V7BenchmarkError("V7 benchmark receipt has an unexpected schema")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "artifact_class": RECEIPT_ARTIFACT_CLASS,
        "status": "PASS",
        "protocol_version": 7,
        "protocol_hash": protocol_hash,
        "git_commit": expected_git_commit,
        "tracked_clean": True,
        "formal_results_root": _relative(root, formal_results_root(validated, root)),
        "benchmark_root": _relative(root, benchmark_root(validated, root)),
        "benchmark_contract": V7_BENCHMARK_CONTRACT,
        "final_test_markers_at_completion": 0,
        "run_count": len(expected_run_ids()),
    }
    failures = [key for key, value in expected.items() if receipt.get(key) != value]
    for field, value in (
        ("protocol_file_sha256", protocol_file_sha256),
        ("matrix_manifest_sha256", expected_matrix_manifest_sha256),
        ("freeze_manifest_sha256", expected_freeze_manifest_sha256),
    ):
        if receipt.get(field) != value:
            failures.append(field)
    if failures:
        raise V7BenchmarkError(
            "V7 benchmark receipt differs from the frozen contract: " + ", ".join(failures)
        )
    _validate_canonical_path_fields(
        receipt,
        project_root=root,
        protocol=validated,
        protocol_file_sha256=protocol_file_sha256,
        matrix_manifest_sha256=expected_matrix_manifest_sha256,
        freeze_manifest_sha256=expected_freeze_manifest_sha256,
        label="V7 benchmark receipt",
    )
    for field in (
        "protocol_file_sha256",
        "matrix_manifest_sha256",
        "freeze_manifest_sha256",
    ):
        if not _is_sha256(receipt.get(field)):
            raise V7BenchmarkError(f"V7 benchmark receipt {field} is invalid")
    hardware = _validate_hardware(receipt["hardware"])
    batch_path = expected_batch_path(validated, root)
    if not batch_path.is_file():
        raise V7BenchmarkError("V7 validation benchmark batch is missing")
    batch_file_sha256 = sha256_file(batch_path)
    input_record = receipt["input"]
    if not isinstance(input_record, Mapping):
        raise V7BenchmarkError("V7 benchmark receipt input is malformed")
    batch_hash = input_record.get("batch_sha256")
    if not _is_sha256(batch_hash):
        raise V7BenchmarkError("V7 benchmark receipt batch hash is invalid")
    _validate_input(
        input_record, batch_file_sha256=batch_file_sha256, batch_hash=str(batch_hash)
    )
    if not _is_sha256(receipt.get("training_environment_sha256")) or not _is_sha256(
        receipt.get("split_manifest_sha256")
    ):
        raise V7BenchmarkError("V7 benchmark receipt environment/split binding is invalid")
    runs = receipt["runs"]
    if not isinstance(runs, list):
        raise V7BenchmarkError("V7 benchmark receipt runs must be a list")
    if tuple(str(item.get("run_id", "")) for item in runs if isinstance(item, Mapping)) != expected_run_ids():
        raise V7BenchmarkError("V7 benchmark receipt runs differ from the frozen matrix")
    if len(runs) != len(expected_run_ids()):
        raise V7BenchmarkError("V7 benchmark receipt has the wrong run count")
    expected_rows = list(expected_formal_rows)
    if len(expected_rows) != len(expected_run_ids()):
        raise V7BenchmarkError(
            "V7 final-test evidence does not contain the complete 48-run matrix"
        )
    expected_by_id: dict[str, Mapping[str, Any]] = {}
    for row in expected_rows:
        run_id = str(row.get("run_id", ""))
        if run_id in expected_by_id:
            raise V7BenchmarkError(
                f"V7 final-test evidence contains a duplicate run: {run_id}"
            )
        expected_by_id[run_id] = row
    if tuple(expected_by_id) != expected_run_ids():
        raise V7BenchmarkError(
            "V7 final-test evidence run order differs from the frozen matrix"
        )
    for row in expected_rows:
        run_id = str(row["run_id"])
        entry = runs[expected_run_ids().index(run_id)]
        required_bindings = (
            "checkpoint_sha256",
            "config_hash",
            "shared_weight_sha256",
        )
        for field in required_bindings:
            observed = str(row.get(field, ""))
            if not _is_sha256(observed) or entry.get(field) != observed:
                raise V7BenchmarkError(
                    f"{run_id}: benchmark receipt does not match formal {field}"
                )
        for field in ("training_environment_sha256", "split_manifest_sha256"):
            observed = str(row.get(field, ""))
            if not _is_sha256(observed) or observed != receipt.get(field):
                raise V7BenchmarkError(
                    f"{run_id}: formal {field} differs from the benchmark receipt"
                )
    shared_by_seed: dict[int, set[str]] = {}
    for entry in runs:
        if not isinstance(entry, Mapping):
            raise V7BenchmarkError("V7 benchmark receipt has a malformed run entry")
        expected_entry_keys = {
            "run_id",
            "seed",
            "condition",
            "benchmark_id",
            "checkpoint_sha256",
            "config_hash",
            "shared_weight_sha256",
            "result_path",
            "result_sha256",
        }
        if set(entry) != expected_entry_keys:
            raise V7BenchmarkError("V7 benchmark receipt run entry has an unexpected schema")
        run_id = str(entry["run_id"])
        if not _is_sha256(entry.get("checkpoint_sha256")) or not _is_sha256(
            entry.get("config_hash")
        ) or not _is_sha256(entry.get("shared_weight_sha256")) or not _is_sha256(
            entry.get("result_sha256")
        ) or not _is_sha256(entry.get("benchmark_id")):
            raise V7BenchmarkError(f"{run_id}: receipt run hash is invalid")
        expected_seed = int(run_id.rsplit("_s", 1)[1])
        expected_condition = run_id.split("_t6_", 1)[1].rsplit("_s", 1)[0]
        if entry.get("seed") != expected_seed or entry.get("condition") != expected_condition:
            raise V7BenchmarkError(f"{run_id}: receipt seed/condition differs from run identity")
        result_path = _inside(root, str(entry["result_path"]), f"{run_id} result path")
        if result_path != expected_result_path(validated, root, run_id):
            raise V7BenchmarkError(f"{run_id}: receipt result path is not canonical")
        if not result_path.is_file() or sha256_file(result_path) != entry["result_sha256"]:
            raise V7BenchmarkError(f"{run_id}: benchmark result file differs from its receipt")
        result = _read_json(result_path, f"{run_id} benchmark result")
        validate_result(
            result,
            run_id=run_id,
            protocol_hash=protocol_hash,
            git_commit=expected_git_commit,
            checkpoint_sha256=str(entry["checkpoint_sha256"]),
            config_hash=str(entry["config_hash"]),
            training_environment_sha256=str(receipt["training_environment_sha256"]),
            split_manifest_sha256=str(receipt["split_manifest_sha256"]),
            shared_weight_sha256=str(entry["shared_weight_sha256"]),
            batch_file_sha256=batch_file_sha256,
            batch_hash=str(batch_hash),
            protocol_file_sha256=protocol_file_sha256,
            matrix_manifest_sha256=expected_matrix_manifest_sha256,
            freeze_manifest_sha256=expected_freeze_manifest_sha256,
            project_root=root,
            protocol=validated,
        )
        expected_id = make_benchmark_id(
            protocol_hash=protocol_hash,
            git_commit=expected_git_commit,
            run_id=run_id,
            checkpoint_sha256=str(entry["checkpoint_sha256"]),
            batch_file_sha256=batch_file_sha256,
            batch_hash=str(batch_hash),
            hardware=hardware,
            protocol_file_sha256=protocol_file_sha256,
            matrix_manifest_sha256=expected_matrix_manifest_sha256,
            freeze_manifest_sha256=expected_freeze_manifest_sha256,
        )
        if entry["benchmark_id"] != expected_id or result["hardware"] != hardware:
            raise V7BenchmarkError(f"{run_id}: benchmark identity differs from the receipt")
        shared_by_seed.setdefault(expected_seed, set()).add(str(entry["shared_weight_sha256"]))
    if set(shared_by_seed) != set(V7_FORMAL_SEEDS) or any(
        len(values) != 1 or not _is_sha256(next(iter(values), None))
        for values in shared_by_seed.values()
    ):
        raise V7BenchmarkError(
            "V7 benchmark receipt does not preserve shared initialization within each seed"
        )
    return receipt
