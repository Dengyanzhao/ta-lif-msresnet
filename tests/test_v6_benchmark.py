from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from talif_msresnet.benchmark_v6 import (
    RESULT_ARTIFACT_CLASS,
    RESULT_SCHEMA,
    V6BenchmarkError,
    build_receipt,
    expected_batch_path,
    expected_receipt_path,
    expected_result_path,
    expected_run_ids,
    validate_receipt,
    validate_result,
)
from talif_msresnet.config import load_protocol
from talif_msresnet.config_v6 import V6_BENCHMARK_CONTRACT, V6_OPERATION_MODE_BY_CONDITION
from talif_msresnet.ops import OperationCounter, activity_from_diagnostics
from talif_msresnet.utils import sha256_file, stable_hash


def _hash(token: str) -> str:
    return stable_hash({"fixture": token})


def _hardware() -> dict[str, str]:
    return {
        "device": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "driver_version": "570.195.03",
        "torch": "2.9.1+cu128",
        "cuda": "12.8",
    }


def _result(
    *,
    run_id: str,
    config_hash: str,
    environment_hash: str,
    split_hash: str,
    shared_hash: str,
    checkpoint_hash: str,
    batch_file_hash: str,
    batch_hash: str,
) -> dict[str, Any]:
    condition = run_id.split("_t6_", 1)[1].rsplit("_s", 1)[0]
    operation_mode = V6_OPERATION_MODE_BY_CONDITION[condition]
    shared_window_accesses = 2.0 if operation_mode == "shared_window" else 0.0
    bank_accesses = 2.0 if operation_mode == "time_indexed_bank" else 20.0
    if operation_mode not in {"time_indexed_bank", "count_indexed_bank"}:
        bank_accesses = 0.0
    measurement = {
        "batch_size": 1,
        "latency_mean_ms": 1.0,
        "latency_sd_ms": 0.1,
        "latency_p50_ms": 1.0,
        "latency_p95_ms": 1.2,
        "peak_allocated_bytes": 1024,
        "incremental_peak_bytes": 512,
        "spike_rate": 0.25,
        "synaptic_operation_proxy_per_sample": 10.0,
        "dense_mac_equivalents_per_sample": 20.0,
        "shared_threshold_window_accesses_per_sample": shared_window_accesses,
        "threshold_bank_accesses_per_sample": bank_accesses,
        "spike_count_updates_per_sample": 1.0 if operation_mode == "count_indexed_bank" else 0.0,
        "neuron_operation_mode": operation_mode,
        "operation_count_method": f"fixture;{operation_mode}",
    }
    second = dict(measurement)
    second["batch_size"] = 128
    protocol = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    return {
        "schema": RESULT_SCHEMA,
        "artifact_class": RESULT_ARTIFACT_CLASS,
        "status": "complete",
        "protocol_version": 6,
        "protocol_hash": stable_hash(protocol),
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
            "input_seed": 1169352105,
            "batch_size": 128,
        },
        "benchmark_contract": dict(V6_BENCHMARK_CONTRACT),
        "hardware": _hardware(),
        "measurements": {
            "parameters": {"total": 100, "trainable": 100},
            "batch_1": measurement,
            "batch_128": second,
        },
        "energy": {"status": "not_measured", "claim": "forbidden"},
        "completed_at": "2026-08-11T12:00:00+08:00",
    }


@pytest.mark.parametrize(
    ("mode", "shared", "bank", "count"),
    (
        ("none", 0.0, 0.0, 0.0),
        ("shared_window", 24.0, 0.0, 0.0),
        ("time_indexed_bank", 0.0, 24.0, 0.0),
        ("count_indexed_bank", 0.0, 200.0, 100.0),
    ),
)
def test_v6_control_operation_modes_are_mechanism_specific(
    mode: str,
    shared: float,
    bank: float,
    count: float,
) -> None:
    activity = activity_from_diagnostics(
        {
            "spike_count": 25.0,
            "elements": 100,
            "layers": {"stem": {}, "terminal": {}},
            "timesteps": 6,
        },
        neuron_operation_mode=mode,
    )

    assert activity["shared_threshold_window_accesses"] == shared
    assert activity["threshold_bank_accesses"] == bank
    assert activity["spike_count_updates"] == count
    assert activity["activity_method"].startswith(f"{mode};")


def test_legacy_talif_flag_still_means_count_indexed_accounting() -> None:
    activity = activity_from_diagnostics(
        {"spike_count": 25.0, "elements": 100},
        talif_active=True,
    )

    assert activity["threshold_bank_accesses"] == 200.0
    assert activity["spike_count_updates"] == 100.0


def test_shared_and_time_indexed_accounting_requires_layer_step_diagnostics() -> None:
    for mode in ("shared_window", "time_indexed_bank"):
        with pytest.raises(ValueError, match="neuron_layer_step_calls"):
            activity_from_diagnostics(
                {"spike_count": 25.0, "elements": 100},
                neuron_operation_mode=mode,
            )


def test_operation_counter_carries_the_explicit_mode_into_the_record() -> None:
    with OperationCounter(nn.Identity()) as counter:
        estimate = counter.result(
            batch_size=4,
            diagnostics={
                "spike_count": 25.0,
                "elements": 100,
                "layers": {"stem": {}, "terminal": {}},
                "timesteps": 6,
            },
            neuron_operation_mode="time_indexed_bank",
        )

    record = estimate.per_sample()
    assert record["neuron_operation_mode"] == "time_indexed_bank"
    assert record["shared_threshold_window_accesses_per_sample"] == 0.0
    assert record["threshold_bank_accesses_per_sample"] == 6.0
    assert record["spike_count_updates_per_sample"] == 0.0


def test_v6_benchmark_rejects_a_mode_mismatched_to_the_condition() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    run_id = next(identifier for identifier in expected_run_ids() if "_M2_" in identifier)
    value = _result(
        run_id=run_id,
        config_hash=_hash("c"),
        environment_hash=_hash("e"),
        split_hash=_hash("s"),
        shared_hash=_hash("w"),
        checkpoint_hash=_hash("k"),
        batch_file_hash=_hash("b"),
        batch_hash=_hash("h"),
    )
    value["measurements"]["batch_1"]["neuron_operation_mode"] = "count_indexed_bank"

    with pytest.raises(V6BenchmarkError, match="operation mode"):
        validate_result(
            value,
            run_id=run_id,
            protocol_hash=stable_hash(protocol),
            checkpoint_sha256=_hash("k"),
            config_hash=_hash("c"),
            training_environment_sha256=_hash("e"),
            split_manifest_sha256=_hash("s"),
            shared_weight_sha256=_hash("w"),
            batch_file_sha256=_hash("b"),
            batch_hash=_hash("h"),
        )


@pytest.mark.parametrize("condition", ("M2", "M3"))
def test_v6_benchmark_rejects_history_updates_for_non_history_conditions(condition: str) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    run_id = next(identifier for identifier in expected_run_ids() if f"_{condition}_" in identifier)
    value = _result(
        run_id=run_id,
        config_hash=_hash("c"),
        environment_hash=_hash("e"),
        split_hash=_hash("s"),
        shared_hash=_hash("w"),
        checkpoint_hash=_hash("k"),
        batch_file_hash=_hash("b"),
        batch_hash=_hash("h"),
    )
    value["measurements"]["batch_1"]["spike_count_updates_per_sample"] = 1.0

    with pytest.raises(V6BenchmarkError, match="bank/history|shared/history"):
        validate_result(
            value,
            run_id=run_id,
            protocol_hash=stable_hash(protocol),
            checkpoint_sha256=_hash("k"),
            config_hash=_hash("c"),
            training_environment_sha256=_hash("e"),
            split_manifest_sha256=_hash("s"),
            shared_weight_sha256=_hash("w"),
            batch_file_sha256=_hash("b"),
            batch_hash=_hash("h"),
        )


def _write_complete_receipt_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    protocol = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    protocol_path = tmp_path / "configs" / "protocol_v6_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_bytes((ROOT / "configs" / "protocol_v6_mechanism.yaml").read_bytes())
    batch_path = expected_batch_path(protocol, tmp_path)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    batch_path.write_bytes(b"fixed-validation-batch")
    batch_file_hash = sha256_file(batch_path)
    batch_hash = _hash("b")
    environment_hash = _hash("e")
    split_hash = _hash("s")
    rows: list[dict[str, Any]] = []
    for run_id in expected_run_ids():
        seed = int(run_id.rsplit("_s", 1)[1])
        condition = run_id.split("_t6_", 1)[1].rsplit("_s", 1)[0]
        shared_hash = f"{seed:064x}"[-64:]
        checkpoint_hash = stable_hash({"checkpoint": run_id})
        config_hash = stable_hash({"config": run_id})
        value = _result(
            run_id=run_id,
            config_hash=config_hash,
            environment_hash=environment_hash,
            split_hash=split_hash,
            shared_hash=shared_hash,
            checkpoint_hash=checkpoint_hash,
            batch_file_hash=batch_file_hash,
            batch_hash=batch_hash,
        )
        result_path = expected_result_path(protocol, tmp_path, run_id)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "run_id": run_id,
                "seed": seed,
                "condition": condition,
                "checkpoint_sha256": checkpoint_hash,
                "config_hash": config_hash,
                "training_environment_sha256": environment_hash,
                "split_manifest_sha256": split_hash,
                "shared_weight_sha256": shared_hash,
            }
        )
    receipt = build_receipt(
        project_root=tmp_path,
        protocol=protocol,
        protocol_hash=stable_hash(protocol),
        protocol_file_sha256=sha256_file(protocol_path),
        freeze_manifest_sha256=_hash("f"),
        batch_file_sha256=batch_file_hash,
        batch_hash=batch_hash,
        hardware=_hardware(),
        results=rows,
        created_at="2026-08-11T12:00:00+08:00",
    )
    receipt_path = expected_receipt_path(protocol, tmp_path)
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    return protocol, receipt, protocol_path


def test_v6_benchmark_receipt_binds_all_48_results(tmp_path: Path) -> None:
    protocol, receipt, protocol_path = _write_complete_receipt_fixture(tmp_path)

    validated = validate_receipt(
        project_root=tmp_path,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_hash=stable_hash(protocol),
        expected_freeze_manifest_sha256=_hash("f"),
    )

    assert validated["run_count"] == 48
    assert [entry["run_id"] for entry in validated["runs"]] == list(expected_run_ids())
    assert receipt["final_test_markers_at_completion"] == 0


def test_v6_benchmark_rejects_energy_claim(tmp_path: Path) -> None:
    protocol, receipt, protocol_path = _write_complete_receipt_fixture(tmp_path)
    entry = receipt["runs"][0]
    path = expected_result_path(protocol, tmp_path, entry["run_id"])
    result = json.loads(path.read_text(encoding="utf-8"))
    result["energy"] = {"status": "estimated", "claim": "forbidden"}
    path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    entry["result_sha256"] = sha256_file(path)
    expected_receipt_path(protocol, tmp_path).write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(V6BenchmarkError, match="energy"):
        validate_receipt(
            project_root=tmp_path,
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_hash=stable_hash(protocol),
            expected_freeze_manifest_sha256=_hash("f"),
        )


def test_v6_benchmark_rejects_protocol_file_drift(tmp_path: Path) -> None:
    protocol, _receipt, protocol_path = _write_complete_receipt_fixture(tmp_path)
    protocol_path.write_text("protocol_version: 6\n# drift\n", encoding="utf-8")

    with pytest.raises(V6BenchmarkError, match="protocol_file_sha256"):
        validate_receipt(
            project_root=tmp_path,
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_hash=stable_hash(protocol),
            expected_freeze_manifest_sha256=_hash("f"),
        )


def test_v6_benchmark_result_rejects_test_input() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    value = _result(
        run_id=expected_run_ids()[0],
        config_hash=_hash("c"),
        environment_hash=_hash("e"),
        split_hash=_hash("s"),
        shared_hash=_hash("w"),
        checkpoint_hash=_hash("k"),
        batch_file_hash=_hash("b"),
        batch_hash=_hash("h"),
    )
    value["input"]["test_data_accessed"] = True

    with pytest.raises(V6BenchmarkError, match="validation-only"):
        validate_result(
            value,
            run_id=expected_run_ids()[0],
            protocol_hash=stable_hash(protocol),
            checkpoint_sha256=_hash("k"),
            config_hash=_hash("c"),
            training_environment_sha256=_hash("e"),
            split_manifest_sha256=_hash("s"),
            shared_weight_sha256=_hash("w"),
            batch_file_sha256=_hash("b"),
            batch_hash=_hash("h"),
        )
