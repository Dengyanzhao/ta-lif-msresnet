#!/usr/bin/env python3
"""Run the frozen V7 validation-only resource audit for all 48 checkpoints."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.benchmark import benchmark_model  # noqa: E402
from talif_msresnet.benchmark_v7 import (  # noqa: E402
    RESULT_ARTIFACT_CLASS,
    RESULT_SCHEMA,
    V7BenchmarkError,
    build_receipt,
    canonical_bound_paths,
    exclusive_write_json,
    expected_batch_path,
    expected_receipt_path,
    expected_result_path,
    expected_run_ids,
    formal_results_root,
    repository_git_identity,
    validate_receipt,
    validate_result,
)
from talif_msresnet.config import load_protocol, validate_run_mapping  # noqa: E402
from talif_msresnet.config_v7 import (  # noqa: E402
    V7_ARTIFACT_PATHS,
    V7_BENCHMARK_CONTRACT,
    V7_OPERATION_MODE_BY_CONDITION,
    canonicalize_v7_artifact_run_mapping,
    generate_v7_formal_matrix,
    validate_v7_cifar100_provenance_files,
    validate_v7_protocol,
)
from talif_msresnet.data import (  # noqa: E402
    load_representative_batch_artifact,
    validate_representative_batch_artifact,
)
from talif_msresnet.freeze import verify_formal_freeze  # noqa: E402
from talif_msresnet.models import build_model  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.utils import load_checkpoint, sha256_file, utc_now  # noqa: E402


class V7BenchmarkRunError(RuntimeError):
    """Raised when the resource audit would weaken the frozen V7 boundary."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / V7_ARTIFACT_PATHS["protocol"],
    )
    parser.add_argument("--device", default="cuda:0")
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V7BenchmarkRunError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V7BenchmarkRunError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise V7BenchmarkRunError(f"{label} must be a SHA-256")
    return value


def _exact_path(path: Path, expected: Path, label: str) -> None:
    if path.resolve() != expected.resolve():
        raise V7BenchmarkRunError(f"{label} is not the canonical V7 path: {path}")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise V7BenchmarkRunError(f"V7 bound path escapes the repository: {path}") from exc


def _hardware_identity(device: torch.device) -> dict[str, str]:
    if device.type != "cuda" or device.index is None:
        raise V7BenchmarkRunError("V7 benchmark requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise V7BenchmarkRunError("V7 benchmark requires CUDA")
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=uuid,driver_version,name",
                "--format=csv,noheader,nounits",
                "-i",
                str(device.index),
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise V7BenchmarkRunError(f"Cannot identify CUDA benchmark hardware: {exc}") from exc
    fields = [item.strip() for item in output.split(",")]
    if len(fields) != 3 or any(not item for item in fields):
        raise V7BenchmarkRunError("nvidia-smi returned an invalid hardware identity")
    device_uuid, driver_version, device_name = fields
    torch_name = torch.cuda.get_device_name(device)
    if device_name not in torch_name and torch_name not in device_name:
        raise V7BenchmarkRunError("nvidia-smi and PyTorch disagree about the selected GPU")
    return {
        "device": str(device),
        "device_name": torch_name,
        "device_uuid": device_uuid,
        "driver_version": driver_version,
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V7BenchmarkRunError(f"{label} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise V7BenchmarkRunError(f"{label} is not finite")
    return number


def _measurement(
    result: Any,
    batch_size: int,
    *,
    neuron_operation_mode: str,
) -> dict[str, Any]:
    batch = result.batches[batch_size]
    operations = result.operations[batch_size]
    if operations.get("neuron_operation_mode") != neuron_operation_mode:
        raise V7BenchmarkRunError(
            "Benchmark operation accounting mode differs from the frozen V7 condition"
        )
    return {
        "batch_size": batch_size,
        "latency_mean_ms": _finite(batch.latency_mean_ms, "latency_mean_ms"),
        "latency_sd_ms": _finite(batch.latency_sd_ms, "latency_sd_ms"),
        "latency_p50_ms": _finite(batch.latency_p50_ms, "latency_p50_ms"),
        "latency_p95_ms": _finite(batch.latency_p95_ms, "latency_p95_ms"),
        "peak_allocated_bytes": int(batch.peak_allocated_bytes),
        "incremental_peak_bytes": int(batch.incremental_peak_bytes),
        "spike_rate": _finite(operations.get("firing_rate"), "firing_rate"),
        "synaptic_operation_proxy_per_sample": _finite(
            operations.get("syops_per_sample"), "syops_per_sample"
        ),
        "dense_mac_equivalents_per_sample": _finite(
            operations.get("dense_mac_equivalents_per_sample"),
            "dense_mac_equivalents_per_sample",
        ),
        "shared_threshold_window_accesses_per_sample": _finite(
            operations.get("shared_threshold_window_accesses_per_sample"),
            "shared_threshold_window_accesses_per_sample",
        ),
        "threshold_bank_accesses_per_sample": _finite(
            operations.get("threshold_bank_accesses_per_sample"),
            "threshold_bank_accesses_per_sample",
        ),
        "spike_count_updates_per_sample": _finite(
            operations.get("spike_count_updates_per_sample"),
            "spike_count_updates_per_sample",
        ),
        "neuron_operation_mode": neuron_operation_mode,
        "operation_count_method": str(operations.get("method", "")),
    }


def _load_formal_audit(
    *,
    protocol: Mapping[str, Any],
    results_root: Path,
) -> list[dict[str, Any]]:
    expected = generate_v7_formal_matrix(protocol)
    expected_ids = expected_run_ids()
    if tuple(str(raw.get("run_id")) for raw in expected) != expected_ids:
        raise V7BenchmarkRunError("V7 generated matrix differs from the frozen run order")
    observed = {
        path.name
        for path in results_root.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "logs"
    }
    if observed != set(expected_ids):
        raise V7BenchmarkRunError(
            "Formal result directory set differs from the V7 matrix: "
            f"missing={sorted(set(expected_ids) - observed)[:5]}, "
            f"extra={sorted(observed - set(expected_ids))[:5]}"
        )
    rows: list[dict[str, Any]] = []
    environments: set[str] = set()
    splits: set[str] = set()
    shared_by_seed: dict[int, set[str]] = {}
    for raw in expected:
        resolved = validate_run_mapping(raw, protocol)
        run_id = resolved.runtime.run_id
        run_dir = results_root / run_id
        for required in ("best.pt", "run_manifest.json", "seed_metrics.json"):
            if not (run_dir / required).is_file():
                raise V7BenchmarkRunError(f"{run_id}: missing formal artifact {required}")
        for forbidden in ("final_test.json", "final_test.in_progress.json", "final_test.lock"):
            if (run_dir / forbidden).exists():
                raise V7BenchmarkRunError(
                    f"{run_id}: benchmark must complete before any V7 test transaction"
                )
        metrics = _read_json(run_dir / "seed_metrics.json", f"{run_id} metrics")
        manifest = _read_json(run_dir / "run_manifest.json", f"{run_id} manifest")
        if metrics.get("status") != "complete" or manifest.get("status") != "complete":
            raise V7BenchmarkRunError(f"{run_id}: formal training is not complete")
        if metrics.get("test_accuracy") not in (None, ""):
            raise V7BenchmarkRunError(f"{run_id}: test metric exists before benchmark completion")
        checkpoint_path = run_dir / "best.pt"
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        raw_checkpoint_config = checkpoint.get("config")
        if not isinstance(raw_checkpoint_config, Mapping):
            raise V7BenchmarkRunError(f"{run_id}: checkpoint configuration is missing")
        normalized = canonicalize_v7_artifact_run_mapping(
            raw_checkpoint_config, protocol, project_root=PROJECT_ROOT
        )
        checkpoint_config = validate_run_mapping(normalized, protocol)
        if (
            checkpoint_config.runtime.run_id != run_id
            or checkpoint_config.config_hash != resolved.config_hash
        ):
            raise V7BenchmarkRunError(f"{run_id}: checkpoint differs from the frozen matrix")
        config_hash = _sha256(str(metrics.get("config_hash", "")), f"{run_id} config hash")
        if (
            config_hash != resolved.config_hash
            or str(manifest.get("config_hash", "")) != config_hash
            or checkpoint.get("config_hash") != config_hash
        ):
            raise V7BenchmarkRunError(f"{run_id}: config hash differs across formal artifacts")
        environment = _sha256(
            str(metrics.get("training_environment_sha256", "")),
            f"{run_id} training environment",
        )
        split = _sha256(
            str(metrics.get("split_manifest_sha256", "")), f"{run_id} split manifest"
        )
        shared = _sha256(
            str(metrics.get("shared_weight_sha256", "")), f"{run_id} shared init"
        )
        manifest_environment = manifest.get("environment")
        if not isinstance(manifest_environment, Mapping) or manifest_environment.get(
            "training_environment_sha256"
        ) != environment:
            raise V7BenchmarkRunError(f"{run_id}: manifest training environment differs")
        if (
            manifest.get("split_manifest_sha256") != split
            or manifest.get("shared_weight_sha256") != shared
        ):
            raise V7BenchmarkRunError(f"{run_id}: manifest split or initialization differs")
        environments.add(environment)
        splits.add(split)
        shared_by_seed.setdefault(resolved.runtime.seed, set()).add(shared)
        rows.append(
            {
                "run_id": run_id,
                "seed": resolved.runtime.seed,
                "condition": resolved.model.condition,
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "checkpoint": checkpoint,
                "config": checkpoint_config,
                "config_hash": config_hash,
                "training_environment_sha256": environment,
                "split_manifest_sha256": split,
                "shared_weight_sha256": shared,
            }
        )
    if len(environments) != 1 or len(splits) != 1:
        raise V7BenchmarkRunError("V7 formal matrix has mixed training environments or data splits")
    if any(len(values) != 1 for values in shared_by_seed.values()):
        raise V7BenchmarkRunError("V7 conditions do not share initialization within each seed")
    return rows


def _load_fixed_batch(
    *,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    git_commit: str,
    bound_hashes: Mapping[str, str],
    audit_rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, str, str]:
    path = expected_batch_path(protocol, PROJECT_ROOT)
    if not path.is_file():
        raise V7BenchmarkRunError(
            "Fixed V7 validation batch is missing; run scripts/export_v7_validation_batch.py"
        )
    inputs, targets, metadata = load_representative_batch_artifact(path)
    split = audit_rows[0]["split_manifest_sha256"]
    config = audit_rows[0]["config"]
    content_hash = validate_representative_batch_artifact(
        inputs,
        targets,
        metadata,
        expected_dataset=config.data.dataset,
        expected_protocol_hash=protocol_hash,
        expected_split_manifest_sha256=split,
        expected_in_channels=config.model.in_channels,
        expected_time_steps=config.model.time_steps,
        expected_num_classes=config.model.num_classes,
    )
    expected_metadata = {
        "git_commit": git_commit,
        "tracked_clean": True,
        "protocol": V7_ARTIFACT_PATHS["protocol"],
        "protocol_file_sha256": bound_hashes["protocol_file_sha256"],
        "matrix_manifest": f"{V7_ARTIFACT_PATHS['formal_matrix']}/matrix_manifest.json",
        "matrix_manifest_sha256": bound_hashes["matrix_manifest_sha256"],
        "freeze_manifest": V7_ARTIFACT_PATHS["freeze_manifest"],
        "freeze_manifest_sha256": bound_hashes["freeze_manifest_sha256"],
        "input_seed": V7_BENCHMARK_CONTRACT["input_seed"],
        "batch_size": 128,
        "source_split": "validation",
        "test_data_accessed": False,
    }
    mismatches = [
        key for key, expected in expected_metadata.items() if metadata.get(key) != expected
    ]
    if mismatches:
        raise V7BenchmarkRunError(
            "Fixed V7 validation batch differs from frozen release evidence: "
            + ", ".join(mismatches)
        )
    if int(inputs.shape[0]) != 128 or int(targets.shape[0]) != 128:
        raise V7BenchmarkRunError("Fixed V7 validation batch must contain exactly 128 samples")
    if any(row["split_manifest_sha256"] != split for row in audit_rows):
        raise V7BenchmarkRunError("V7 benchmark cannot use a mixed split matrix")
    return inputs, sha256_file(path), content_hash


def _result_payload(
    *,
    row: Mapping[str, Any],
    protocol_hash: str,
    git_commit: str,
    bound_paths: Mapping[str, Path],
    bound_hashes: Mapping[str, str],
    batch_file_sha256: str,
    batch_hash: str,
    hardware: Mapping[str, Any],
    measurements: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "artifact_class": RESULT_ARTIFACT_CLASS,
        "status": "complete",
        "protocol_version": 7,
        "protocol_hash": protocol_hash,
        "git_commit": git_commit,
        "tracked_clean": True,
        "protocol_path": _relative(bound_paths["protocol"]),
        "protocol_file_sha256": bound_hashes["protocol_file_sha256"],
        "matrix_manifest_path": _relative(bound_paths["matrix_manifest"]),
        "matrix_manifest_sha256": bound_hashes["matrix_manifest_sha256"],
        "freeze_manifest_path": _relative(bound_paths["freeze_manifest"]),
        "freeze_manifest_sha256": bound_hashes["freeze_manifest_sha256"],
        "run_id": row["run_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "config_hash": row["config_hash"],
        "training_environment_sha256": row["training_environment_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "shared_weight_sha256": row["shared_weight_sha256"],
        "input": {
            "source_split": "validation",
            "test_data_accessed": False,
            "file_sha256": batch_file_sha256,
            "batch_sha256": batch_hash,
            "input_seed": V7_BENCHMARK_CONTRACT["input_seed"],
            "batch_size": 128,
        },
        "benchmark_contract": dict(V7_BENCHMARK_CONTRACT),
        "hardware": dict(hardware),
        "measurements": dict(measurements),
        "energy": {"status": "not_measured", "claim": "forbidden"},
        "completed_at": utc_now(),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.device != "cuda:0":
            raise V7BenchmarkRunError("V7 benchmark device is frozen as cuda:0")
        protocol_path = args.protocol.resolve()
        expected_protocol = (PROJECT_ROOT / V7_ARTIFACT_PATHS["protocol"]).resolve()
        _exact_path(protocol_path, expected_protocol, "protocol")
        protocol = validate_v7_protocol(load_protocol(protocol_path))
        preflight = check_protocol(
            protocol_path,
            mode="final-test",
            project_root=PROJECT_ROOT,
            check_dependencies=False,
        )
        if not preflight.ok:
            raise V7BenchmarkRunError(
                "V7 benchmark preflight failed: " + "; ".join(preflight.errors)
            )
        verify_formal_freeze(
            project_root=PROJECT_ROOT,
            protocol_path=protocol_path,
            matrix_dir=PROJECT_ROOT / V7_ARTIFACT_PATHS["formal_matrix"],
        )
        validate_v7_cifar100_provenance_files(protocol, project_root=PROJECT_ROOT)
        git_commit, tracked_clean = repository_git_identity(PROJECT_ROOT)
        if not tracked_clean:
            raise V7BenchmarkRunError("V7 benchmark requires a clean tracked release commit")
        bound_paths = canonical_bound_paths(protocol, PROJECT_ROOT)
        if any(not path.is_file() for path in bound_paths.values()):
            raise V7BenchmarkRunError("V7 benchmark bound release artifact is missing")
        bound_hashes = {
            "protocol_file_sha256": sha256_file(bound_paths["protocol"]),
            "matrix_manifest_sha256": sha256_file(bound_paths["matrix_manifest"]),
            "freeze_manifest_sha256": sha256_file(bound_paths["freeze_manifest"]),
        }
        results_root = formal_results_root(protocol, PROJECT_ROOT)
        if not results_root.is_dir():
            raise V7BenchmarkRunError("V7 formal results root is missing")
        audit_rows = _load_formal_audit(protocol=protocol, results_root=results_root)
        receipt_path = expected_receipt_path(protocol, PROJECT_ROOT)
        if receipt_path.exists():
            validate_receipt(
                project_root=PROJECT_ROOT,
                protocol=protocol,
                protocol_path=protocol_path,
                protocol_hash=preflight.protocol_hash,
                expected_git_commit=git_commit,
                expected_matrix_manifest_sha256=bound_hashes["matrix_manifest_sha256"],
                expected_freeze_manifest_sha256=bound_hashes["freeze_manifest_sha256"],
                expected_formal_rows=audit_rows,
            )
            print(f"V7_BENCHMARK_RECEIPT_ALREADY_PASS={receipt_path}")
            return 0
        inputs, batch_file_sha256, batch_hash = _load_fixed_batch(
            protocol=protocol,
            protocol_hash=preflight.protocol_hash,
            git_commit=git_commit,
            bound_hashes=bound_hashes,
            audit_rows=audit_rows,
        )
        device = torch.device(args.device)
        hardware = _hardware_identity(device)
        inputs = inputs.to(device=device, dtype=torch.float32, non_blocking=False)
        for index, row in enumerate(audit_rows, start=1):
            result_path = expected_result_path(protocol, PROJECT_ROOT, str(row["run_id"]))
            if result_path.exists():
                existing = _read_json(result_path, f"{row['run_id']} benchmark result")
                validate_result(
                    existing,
                    run_id=str(row["run_id"]),
                    protocol_hash=preflight.protocol_hash,
                    git_commit=git_commit,
                    checkpoint_sha256=str(row["checkpoint_sha256"]),
                    config_hash=str(row["config_hash"]),
                    training_environment_sha256=str(row["training_environment_sha256"]),
                    split_manifest_sha256=str(row["split_manifest_sha256"]),
                    shared_weight_sha256=str(row["shared_weight_sha256"]),
                    batch_file_sha256=batch_file_sha256,
                    batch_hash=batch_hash,
                    protocol_file_sha256=bound_hashes["protocol_file_sha256"],
                    matrix_manifest_sha256=bound_hashes["matrix_manifest_sha256"],
                    freeze_manifest_sha256=bound_hashes["freeze_manifest_sha256"],
                    project_root=PROJECT_ROOT,
                    protocol=protocol,
                )
                if existing["hardware"] != hardware:
                    raise V7BenchmarkRunError(
                        f"{row['run_id']}: existing benchmark used a different GPU identity"
                    )
                print(f"[{index}/48] {row['run_id']}: already_complete")
                continue
            config = row["config"]
            model_config = dataclasses.asdict(config.model)
            model_config["init_seed"] = config.runtime.seed
            model = build_model(model_config).to(device=device, dtype=torch.float32)
            model.load_state_dict(row["checkpoint"]["model_state"], strict=True)

            def input_factory(batch_size: int) -> torch.Tensor:
                return inputs[:batch_size]

            operation_mode = V7_OPERATION_MODE_BY_CONDITION[config.model.condition]
            result = benchmark_model(
                model,
                input_factory,
                batch_sizes=tuple(V7_BENCHMARK_CONTRACT["batch_sizes"]),
                warmup_iterations=int(V7_BENCHMARK_CONTRACT["warmup_iterations"]),
                iterations=int(V7_BENCHMARK_CONTRACT["timed_iterations"]),
                device=device,
                neuron_operation_mode=operation_mode,
                energy_constants=None,
            )
            parameter_report = model.parameter_report()
            measurements = {
                "parameters": {
                    "total": int(parameter_report["total"]),
                    "trainable": int(parameter_report["trainable"]),
                },
                "batch_1": _measurement(result, 1, neuron_operation_mode=operation_mode),
                "batch_128": _measurement(result, 128, neuron_operation_mode=operation_mode),
            }
            payload = _result_payload(
                row=row,
                protocol_hash=preflight.protocol_hash,
                git_commit=git_commit,
                bound_paths=bound_paths,
                bound_hashes=bound_hashes,
                batch_file_sha256=batch_file_sha256,
                batch_hash=batch_hash,
                hardware=hardware,
                measurements=measurements,
            )
            validate_result(
                payload,
                run_id=str(row["run_id"]),
                protocol_hash=preflight.protocol_hash,
                git_commit=git_commit,
                checkpoint_sha256=str(row["checkpoint_sha256"]),
                config_hash=str(row["config_hash"]),
                training_environment_sha256=str(row["training_environment_sha256"]),
                split_manifest_sha256=str(row["split_manifest_sha256"]),
                shared_weight_sha256=str(row["shared_weight_sha256"]),
                batch_file_sha256=batch_file_sha256,
                batch_hash=batch_hash,
                protocol_file_sha256=bound_hashes["protocol_file_sha256"],
                matrix_manifest_sha256=bound_hashes["matrix_manifest_sha256"],
                freeze_manifest_sha256=bound_hashes["freeze_manifest_sha256"],
                project_root=PROJECT_ROOT,
                protocol=protocol,
            )
            exclusive_write_json(result_path, payload)
            print(f"[{index}/48] {row['run_id']}: completed")
            del model
            torch.cuda.empty_cache()
        receipt = build_receipt(
            project_root=PROJECT_ROOT,
            protocol=protocol,
            protocol_hash=preflight.protocol_hash,
            git_commit=git_commit,
            tracked_clean=tracked_clean,
            protocol_file_sha256=bound_hashes["protocol_file_sha256"],
            matrix_manifest_sha256=bound_hashes["matrix_manifest_sha256"],
            freeze_manifest_sha256=bound_hashes["freeze_manifest_sha256"],
            batch_file_sha256=batch_file_sha256,
            batch_hash=batch_hash,
            hardware=hardware,
            results=audit_rows,
            created_at=utc_now(),
        )
        exclusive_write_json(receipt_path, receipt)
        print(f"V7_BENCHMARK_RECEIPT_PASS={receipt_path}")
        return 0
    except (V7BenchmarkError, V7BenchmarkRunError, OSError, ValueError, RuntimeError) as exc:
        print(f"V7_BENCHMARK_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
