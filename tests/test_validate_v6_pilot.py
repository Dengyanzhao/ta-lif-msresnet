from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_v6_pilot as validator

from talif_msresnet.config import load_protocol


def _history() -> list[dict[str, Any]]:
    return [
        {
            "epoch": index,
            "val_accuracy": 0.40 + index * 0.001,
            "all_zero_block_gradient_batch_fraction": 0.02,
        }
        for index in range(120)
    ]


def test_threshold_summary_passes_frozen_v6_pilot_contract() -> None:
    summary, integrity, thresholds = validator._threshold_summary(
        _history(),
        best_accuracy=0.519,
        required_best=0.20,
        required_gradient_coverage=0.95,
        late_window_epochs=10,
        minimum_late_to_best_ratio=0.75,
    )

    assert integrity == []
    assert thresholds == []
    assert summary["minimum_gradient_coverage"] == pytest.approx(0.98)
    assert summary["late_to_best_ratio"] is not None


def test_threshold_summary_separates_threshold_from_integrity_failure() -> None:
    history = _history()
    history[0]["all_zero_block_gradient_batch_fraction"] = 0.20
    summary, integrity, thresholds = validator._threshold_summary(
        history,
        best_accuracy=0.519,
        required_best=0.20,
        required_gradient_coverage=0.95,
        late_window_epochs=10,
        minimum_late_to_best_ratio=1.0,
    )

    assert integrity == []
    assert any("gradient coverage" in item for item in thresholds)
    assert any("late-to-best ratio" in item for item in thresholds)
    assert summary["minimum_gradient_coverage"] == pytest.approx(0.80)


def test_optimizer_group_contract_rejects_an_adaptive_group_for_m0() -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(condition="M0"),
        optimizer=SimpleNamespace(
            lr=0.025, ta_lr_scale=0.1, ta_weight_decay=0.0, weight_decay=0.0005
        ),
    )
    manifest = {
        "schema_version": 1,
        "groups": [
            {
                "role": "base_decay",
                "parameter_names": ["weight"],
                "parameter_count": 1,
                "parameter_numel": 1,
                "initial_lr": 0.025,
                "current_lr": 0.025,
                "weight_decay": 0.0005,
            },
            {
                "role": "base_no_decay",
                "parameter_names": ["bias"],
                "parameter_count": 1,
                "parameter_numel": 1,
                "initial_lr": 0.025,
                "current_lr": 0.025,
                "weight_decay": 0.0,
            },
            {
                "role": "adaptive",
                "parameter_names": ["forbidden"],
                "parameter_count": 1,
                "parameter_numel": 1,
                "initial_lr": 0.0025,
                "current_lr": 0.0025,
                "weight_decay": 0.0,
            },
        ],
    }

    failures = validator._optimizer_group_failures(
        manifest,
        config=config,
        adaptive_names=[],
    )

    assert failures
    assert any("optimizer roles" in item for item in failures)


def test_validate_pilot_aggregate_exposes_freeze_consumer_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    protocol_path = tmp_path / "configs" / "protocol_v6_mechanism.yaml"
    config_dir = tmp_path / "configs" / "v6_mechanism_pilot_generated"
    output_root = tmp_path / "results" / "pilot" / "v6_mechanism" / "cifar100_s22068314"
    protocol_path.parent.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    output_root.mkdir(parents=True)
    protocol_path.write_bytes((ROOT / "configs" / "protocol_v6_mechanism.yaml").read_bytes())
    manifest_path = config_dir / "matrix_manifest.json"
    csv_path = config_dir / "run_manifest.csv"
    manifest_path.write_text("{}\n", encoding="utf-8")
    csv_path.write_text("run_id\n", encoding="utf-8")
    conditions = list(validator.V6_ACTIVE_CONDITIONS)
    rows = [{"condition": condition} for condition in conditions]
    expected = {
        condition: SimpleNamespace(runtime=SimpleNamespace(run_id=f"E6_{condition}"))
        for condition in conditions
    }
    for config in expected.values():
        (output_root / config.runtime.run_id).mkdir()
    environment_hash = "a" * 64
    block = SimpleNamespace(
        dataset="cifar100",
        pilot_seed=22_068_314,
        health_seed=1_707_261_715,
        block_hash="b" * 64,
        pilot_output_root=output_root,
    )

    monkeypatch.setattr(validator, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(validator, "require_v6_author_freeze", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        validator, "artifact_paths_for_protocol", lambda _protocol: protocol["artifact_paths"]
    )
    monkeypatch.setattr(validator, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(
        validator,
        "_load_generated_matrix",
        lambda **_kwargs: (rows, expected, [], manifest_path, csv_path),
    )
    monkeypatch.setattr(
        validator,
        "_bound_evidence",
        lambda **_kwargs: ({"health": {}}, []),
    )
    monkeypatch.setattr(validator, "_aggregate_metrics_csv_failures", lambda *_args: [])

    def fake_validate_run(**kwargs: Any) -> tuple[dict[str, Any], str, str, str]:
        return (
            {"integrity_failures": [], "threshold_failures": []},
            environment_hash,
            "c" * 64,
            "d" * 64,
        )

    monkeypatch.setattr(validator, "_validate_run", fake_validate_run)
    report = validator.validate_pilot(
        protocol_path=protocol_path,
        config_dir=config_dir,
        repository_root=tmp_path,
    )

    assert report["status"] == "PASS"
    assert report["decision"] == validator.PASS_DECISION
    assert report["artifact_class"] == validator.ARTIFACT_CLASS
    assert report["training_environment_sha256"] == environment_hash
    dataset = report["datasets"]["cifar100"]
    assert dataset["environment_sha256"] == environment_hash
    assert list(dataset["runs"]) == conditions


def test_main_refuses_to_overwrite_canonical_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results" / "pilot" / "v6_mechanism" / "validation.json"
    output.parent.mkdir(parents=True)
    original = b'{"status":"PASS"}\n'
    output.write_bytes(original)
    monkeypatch.setattr(validator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        validator,
        "load_protocol",
        lambda _path: {
            "pilot_acceptance": {"validation_output": str(output.relative_to(tmp_path))}
        },
    )
    monkeypatch.setattr(
        validator,
        "validate_pilot",
        lambda **_kwargs: pytest.fail("existing output must block before validation"),
    )

    assert validator.main(["--protocol", str(tmp_path / "protocol.yaml")]) == 2
    assert output.read_bytes() == original
