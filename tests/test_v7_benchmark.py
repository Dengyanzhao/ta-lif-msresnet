from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts import export_v7_validation_batch
from scripts.run_v7_benchmarks import _parameter_measurement
from talif_msresnet import benchmark_v7
from talif_msresnet.benchmark_v7 import (
    RESULT_ARTIFACT_CLASS,
    RESULT_SCHEMA,
    V7BenchmarkError,
    build_receipt,
    expected_batch_path,
    expected_receipt_path,
    expected_result_path,
    expected_run_ids,
    validate_receipt,
)
from talif_msresnet.config import load_protocol
from talif_msresnet.config_v7 import (
    V7_BENCHMARK_CONTRACT,
    V7_OPERATION_MODE_BY_CONDITION,
)
from talif_msresnet.diagnostics import representative_batch_sha256
from talif_msresnet.utils import sha256_file, stable_hash

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SOURCE = ROOT / "configs" / "protocol_v7_mechanism.yaml"
COMMIT = "a" * 40


def _hash(token: str) -> str:
    return stable_hash({"fixture": token})


def test_v7_parameter_measurement_adapts_canonical_model_report() -> None:
    assert _parameter_measurement(
        {"total_parameters": 123, "trainable_parameters": 117}
    ) == {"total": 123, "trainable": 117}


def test_v7_batch_compatibility_rebind_preserves_tensor_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    targets = torch.tensor([1, 2], dtype=torch.long)
    content_hash = representative_batch_sha256(inputs, targets)
    output = tmp_path / "validation_batch.pt"
    old_metadata = {
        "git_commit": "a" * 40,
        "freeze_manifest_sha256": "b" * 64,
        "schema": "fixture",
    }
    torch.save({"inputs": inputs, "targets": targets, "metadata": old_metadata}, output)
    old_file_hash = sha256_file(output)
    monkeypatch.setattr(
        export_v7_validation_batch, "V7_BATCH_RECOVERY_BASE_COMMIT", "a" * 40
    )
    monkeypatch.setattr(
        export_v7_validation_batch,
        "V7_BATCH_RECOVERY_BASE_FREEZE_SHA256",
        "b" * 64,
    )
    monkeypatch.setattr(
        export_v7_validation_batch, "V7_BATCH_RECOVERY_BASE_FILE_SHA256", old_file_hash
    )
    monkeypatch.setattr(
        export_v7_validation_batch,
        "V7_BATCH_RECOVERY_CONTENT_SHA256",
        content_hash,
    )
    expected = dict(old_metadata)
    expected["git_commit"] = "c" * 40
    expected["freeze_manifest_sha256"] = "d" * 64
    sidecar = tmp_path / "validation_batch_compatibility_recovery.json"

    sidecar.write_text("existing", encoding="utf-8")
    with pytest.raises(export_v7_validation_batch.V7BatchExportError, match="sidecar exists"):
        export_v7_validation_batch._rebind_existing_batch(
            output,
            expected_metadata=expected,
            expected_batch_hash=content_hash,
            sidecar_path=sidecar,
        )
    assert sha256_file(output) == old_file_hash
    sidecar.unlink()

    assert export_v7_validation_batch._rebind_existing_batch(
        output,
        expected_metadata=expected,
        expected_batch_hash=content_hash,
        sidecar_path=sidecar,
    )
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert torch.equal(payload["inputs"], inputs)
    assert torch.equal(payload["targets"], targets)
    assert payload["metadata"]["git_commit"] == "c" * 40
    assert payload["metadata"]["freeze_manifest_sha256"] == "d" * 64
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["batch_content_sha256"] == content_hash
    assert record["base_batch_file_sha256"] == old_file_hash
    assert record["rebound_batch_file_sha256"] == sha256_file(output)


def _hardware() -> dict[str, str]:
    return {
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "driver_version": "570.195.03",
        "torch": "2.9.1+cu128",
        "cuda": "12.8",
    }


def _measurement(condition: str, batch_size: int) -> dict[str, Any]:
    mode = V7_OPERATION_MODE_BY_CONDITION[condition]
    return {
        "batch_size": batch_size,
        "latency_mean_ms": 1.0,
        "latency_sd_ms": 0.1,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 1.2,
        "peak_allocated_bytes": 1024,
        "incremental_peak_bytes": 512,
        "spike_rate": 0.25,
        "synaptic_operation_proxy_per_sample": 10.0,
        "dense_mac_equivalents_per_sample": 20.0,
        "shared_threshold_window_accesses_per_sample": (
            2.0 if mode == "shared_window" else 0.0
        ),
        "threshold_bank_accesses_per_sample": (
            2.0 if mode in {"time_indexed_bank", "count_indexed_bank"} else 0.0
        ),
        "spike_count_updates_per_sample": (
            1.0 if mode == "count_indexed_bank" else 0.0
        ),
        "neuron_operation_mode": mode,
        "operation_count_method": f"fixture;{mode}",
    }


def _result(
    *,
    run_id: str,
    protocol_hash: str,
    protocol_file_hash: str,
    matrix_hash: str,
    freeze_hash: str,
    checkpoint_hash: str,
    config_hash: str,
    environment_hash: str,
    split_hash: str,
    shared_hash: str,
    batch_file_hash: str,
    batch_hash: str,
) -> dict[str, Any]:
    condition = run_id.split("_t6_", 1)[1].rsplit("_s", 1)[0]
    return {
        "schema": RESULT_SCHEMA,
        "artifact_class": RESULT_ARTIFACT_CLASS,
        "status": "complete",
        "protocol_version": 7,
        "protocol_hash": protocol_hash,
        "git_commit": COMMIT,
        "tracked_clean": True,
        "protocol_path": "configs/protocol_v7_mechanism.yaml",
        "protocol_file_sha256": protocol_file_hash,
        "matrix_manifest_path": "configs/v7_mechanism_generated/matrix_manifest.json",
        "matrix_manifest_sha256": matrix_hash,
        "freeze_manifest_path": "FREEZE_MANIFEST_V7_MECHANISM.json",
        "freeze_manifest_sha256": freeze_hash,
        "run_id": run_id,
        "checkpoint_sha256": checkpoint_hash,
        "config_hash": config_hash,
        "training_environment_sha256": environment_hash,
        "split_manifest_sha256": split_hash,
        "shared_weight_sha256": shared_hash,
        "input": {
            "source_split": "validation",
            "test_data_accessed": False,
            "file_sha256": batch_file_hash,
            "batch_sha256": batch_hash,
            "input_seed": V7_BENCHMARK_CONTRACT["input_seed"],
            "batch_size": 128,
        },
        "benchmark_contract": dict(V7_BENCHMARK_CONTRACT),
        "hardware": _hardware(),
        "measurements": {
            "parameters": {"total": 100, "trainable": 100},
            "batch_1": _measurement(condition, 1),
            "batch_128": _measurement(condition, 128),
        },
        "energy": {"status": "not_measured", "claim": "forbidden"},
        "completed_at": "2026-08-11T12:00:00+08:00",
    }


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], Path]:
    protocol = load_protocol(PROTOCOL_SOURCE)
    protocol_path = tmp_path / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes(PROTOCOL_SOURCE.read_bytes())
    matrix_path = tmp_path / "configs" / "v7_mechanism_generated" / "matrix_manifest.json"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text('{"matrix":"v7"}\n', encoding="utf-8")
    freeze_path = tmp_path / "FREEZE_MANIFEST_V7_MECHANISM.json"
    freeze_path.write_text('{"freeze":"v7"}\n', encoding="utf-8")
    batch_path = expected_batch_path(protocol, tmp_path)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_bytes(b"fixed-v7-validation-batch")
    protocol_hash = stable_hash(protocol)
    protocol_file_hash = sha256_file(protocol_path)
    matrix_hash = sha256_file(matrix_path)
    freeze_hash = sha256_file(freeze_path)
    batch_file_hash = sha256_file(batch_path)
    batch_hash = _hash("batch-content")
    environment_hash = _hash("environment")
    split_hash = _hash("split")
    rows: list[dict[str, Any]] = []
    for run_id in expected_run_ids():
        seed = int(run_id.rsplit("_s", 1)[1])
        condition = run_id.split("_t6_", 1)[1].rsplit("_s", 1)[0]
        row = {
            "run_id": run_id,
            "seed": seed,
            "condition": condition,
            "checkpoint_sha256": _hash(f"checkpoint:{run_id}"),
            "config_hash": _hash(f"config:{run_id}"),
            "training_environment_sha256": environment_hash,
            "split_manifest_sha256": split_hash,
            "shared_weight_sha256": _hash(f"shared:{seed}"),
        }
        result = _result(
            run_id=run_id,
            protocol_hash=protocol_hash,
            protocol_file_hash=protocol_file_hash,
            matrix_hash=matrix_hash,
            freeze_hash=freeze_hash,
            checkpoint_hash=row["checkpoint_sha256"],
            config_hash=row["config_hash"],
            environment_hash=environment_hash,
            split_hash=split_hash,
            shared_hash=row["shared_weight_sha256"],
            batch_file_hash=batch_file_hash,
            batch_hash=batch_hash,
        )
        result_path = expected_result_path(protocol, tmp_path, run_id)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(row)
    monkeypatch.setattr(
        benchmark_v7,
        "repository_git_identity",
        lambda _root: (COMMIT, True),
    )
    receipt = build_receipt(
        project_root=tmp_path,
        protocol=protocol,
        protocol_hash=protocol_hash,
        git_commit=COMMIT,
        tracked_clean=True,
        protocol_file_sha256=protocol_file_hash,
        matrix_manifest_sha256=matrix_hash,
        freeze_manifest_sha256=freeze_hash,
        batch_file_sha256=batch_file_hash,
        batch_hash=batch_hash,
        hardware=_hardware(),
        results=rows,
        created_at="2026-08-11T12:00:00+08:00",
    )
    receipt_path = expected_receipt_path(protocol, tmp_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return protocol, rows, receipt, protocol_path


def _validate(
    tmp_path: Path,
    protocol: dict[str, Any],
    rows: list[dict[str, Any]],
    protocol_path: Path,
) -> dict[str, Any]:
    return validate_receipt(
        project_root=tmp_path,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_hash=stable_hash(protocol),
        expected_git_commit=COMMIT,
        expected_matrix_manifest_sha256=sha256_file(
            tmp_path / "configs" / "v7_mechanism_generated" / "matrix_manifest.json"
        ),
        expected_freeze_manifest_sha256=sha256_file(
            tmp_path / "FREEZE_MANIFEST_V7_MECHANISM.json"
        ),
        expected_formal_rows=rows,
    )


def test_v7_receipt_binds_exact_ordered_48_run_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, rows, _receipt, protocol_path = _fixture(tmp_path, monkeypatch)

    validated = _validate(tmp_path, protocol, rows, protocol_path)

    assert validated["run_count"] == 48
    assert [entry["run_id"] for entry in validated["runs"]] == list(expected_run_ids())


@pytest.mark.parametrize("mutation", ("47", "49", "duplicate", "order", "e6"))
def test_v7_receipt_rejects_nonexact_formal_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    protocol, rows, _receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    changed = copy.deepcopy(rows)
    if mutation == "47":
        changed.pop()
    elif mutation == "49":
        changed.append(copy.deepcopy(changed[-1]))
    elif mutation == "duplicate":
        changed[-1] = copy.deepcopy(changed[0])
    elif mutation == "order":
        changed[0], changed[1] = changed[1], changed[0]
    else:
        changed[0]["run_id"] = changed[0]["run_id"].replace("E9_", "E6_", 1)

    with pytest.raises(V7BenchmarkError, match="48-run|duplicate|order|matrix"):
        _validate(tmp_path, protocol, changed, protocol_path)


@pytest.mark.parametrize(
    ("commit", "clean"),
    (("b" * 40, True), (COMMIT, False)),
)
def test_v7_receipt_requires_current_clean_release_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit: str,
    clean: bool,
) -> None:
    protocol, rows, _receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        benchmark_v7,
        "repository_git_identity",
        lambda _root: (commit, clean),
    )

    with pytest.raises(V7BenchmarkError, match="clean release commit"):
        _validate(tmp_path, protocol, rows, protocol_path)


@pytest.mark.parametrize(
    "value",
    (
        "ABSOLUTE",
        "configs/../configs/protocol_v7_mechanism.yaml",
        "../protocol_v7_mechanism.yaml",
    ),
)
def test_v7_receipt_rejects_noncanonical_bound_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    protocol, rows, receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    receipt["protocol_path"] = (
        str(protocol_path.resolve()) if value == "ABSOLUTE" else value
    )
    expected_receipt_path(protocol, tmp_path).write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(V7BenchmarkError, match="repository-relative|canonical|escapes"):
        _validate(tmp_path, protocol, rows, protocol_path)


@pytest.mark.parametrize("artifact", ("protocol", "matrix", "freeze"))
def test_v7_receipt_rejects_bound_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    protocol, rows, _receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    paths = {
        "protocol": protocol_path,
        "matrix": tmp_path / "configs" / "v7_mechanism_generated" / "matrix_manifest.json",
        "freeze": tmp_path / "FREEZE_MANIFEST_V7_MECHANISM.json",
    }
    paths[artifact].write_text("drift\n", encoding="utf-8")

    with pytest.raises(V7BenchmarkError, match="hash|SHA|differs"):
        _validate(tmp_path, protocol, rows, protocol_path)


@pytest.mark.parametrize(
    "field",
    (
        "checkpoint_sha256",
        "config_hash",
        "training_environment_sha256",
        "split_manifest_sha256",
        "shared_weight_sha256",
    ),
)
def test_v7_receipt_rejects_formal_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    protocol, rows, _receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    rows[0][field] = _hash(f"drift:{field}")

    with pytest.raises(V7BenchmarkError, match=field):
        _validate(tmp_path, protocol, rows, protocol_path)


def test_v7_receipt_rejects_result_file_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, rows, _receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    expected_result_path(protocol, tmp_path, expected_run_ids()[0]).write_text(
        "{}\n", encoding="utf-8"
    )

    with pytest.raises(V7BenchmarkError, match="result file differs"):
        _validate(tmp_path, protocol, rows, protocol_path)


def test_v7_receipt_rejects_hardware_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, rows, receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    run_id = expected_run_ids()[0]
    result_path = expected_result_path(protocol, tmp_path, run_id)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["hardware"]["device_uuid"] = "GPU-different"
    result_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    receipt["runs"][0]["result_sha256"] = sha256_file(result_path)
    expected_receipt_path(protocol, tmp_path).write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(V7BenchmarkError, match="identity differs"):
        _validate(tmp_path, protocol, rows, protocol_path)


def test_v7_receipt_rejects_validation_batch_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, rows, _receipt, protocol_path = _fixture(tmp_path, monkeypatch)
    expected_batch_path(protocol, tmp_path).write_bytes(b"different-validation-batch")

    with pytest.raises(V7BenchmarkError, match="input hash"):
        _validate(tmp_path, protocol, rows, protocol_path)
