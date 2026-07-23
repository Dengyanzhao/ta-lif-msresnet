from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_checkpoint import (  # noqa: E402
    build_benchmark_identity,
    resolve_frozen_energy_constants,
    validate_frozen_benchmark_request,
)
from talif_msresnet.config import load_protocol  # noqa: E402


def _identity(**overrides):
    values = {
        "protocol_hash": "protocol",
        "checkpoint_sha256": "checkpoint",
        "input_file_sha256": "input-file",
        "input_batch_sha256": "input-batch",
        "input_kind": "representative_validation_batch",
        "device_identity": {
            "device": "cuda:0",
            "hardware_name": "GPU",
            "compute_capability": "8.0",
            "total_memory_bytes": 1,
            "multiprocessor_count": 1,
            "torch_version": "2.x",
            "cuda_version": "12.x",
        },
        "precision": "float32",
        "warmup_iterations": 25,
        "timed_iterations": 100,
        "energy_constants_sha256": "none",
    }
    values.update(overrides)
    return build_benchmark_identity(**values)


def test_benchmark_id_is_stable_and_binds_every_measurement_input() -> None:
    baseline = _identity()
    assert baseline == _identity()
    assert baseline["protocol_hash"] == "protocol"
    assert baseline["batch_sizes"] == [1, 128]
    changes = (
        {"protocol_hash": "other-protocol"},
        {"checkpoint_sha256": "other"},
        {"input_file_sha256": "other"},
        {"input_batch_sha256": "other"},
        {"device_identity": {**baseline["device_identity"], "hardware_name": "other"}},
        {"precision": "float16"},
        {"warmup_iterations": 26},
        {"timed_iterations": 101},
        {"energy_constants_sha256": "energy"},
    )
    for change in changes:
        assert _identity(**change)["benchmark_id"] != baseline["benchmark_id"]
    assert baseline["input_residency"] == "preloaded_on_device_before_warmup_and_timing"
    assert baseline["synchronization"] == "cuda_device_synchronize_after_each_timed_forward"


def test_benchmark_wrapper_rejects_protocol_overrides() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol.yaml")["benchmark"]
    validate_frozen_benchmark_request(
        protocol,
        device="cuda:1",
        precision="float32",
        warmup_iterations=25,
        timed_iterations=100,
    )
    import pytest

    with pytest.raises(ValueError, match="frozen study protocol"):
        validate_frozen_benchmark_request(
            protocol,
            device="cuda",
            precision="float16",
            warmup_iterations=25,
            timed_iterations=100,
        )


def test_energy_constants_cannot_be_added_when_protocol_says_not_assessed(tmp_path) -> None:
    protocol_path = ROOT / "configs" / "protocol.yaml"
    protocol = load_protocol(protocol_path)["benchmark"]
    constants = tmp_path / "energy.json"
    constants.write_text("{}", encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match="cannot add a model after freeze"):
        resolve_frozen_energy_constants(
            protocol,
            protocol_path=protocol_path,
            requested_path=constants,
        )
