from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_v7_pilot as validator

from talif_msresnet import pilot_v7
from talif_msresnet.config import load_protocol


def test_recovered_health_plan_binds_execution_and_recovery_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8")
    health_path = tmp_path / "health.json"
    receipt_path = tmp_path / "health.attempt.json"
    health_path.write_text("{}\n", encoding="utf-8")
    receipt_path.write_text("{}\n", encoding="utf-8")
    recovery = {
        "base_health_commit": "1" * 40,
        "recovery_commit": "2" * 40,
    }
    block = SimpleNamespace(
        protocol_hash="a" * 64,
        acceptance_hash="b" * 64,
        block_hash="c" * 64,
        health_output=health_path,
        attempt_receipt=receipt_path,
        dataset="cifar100",
        pilot_seed=1_673_127_435,
        health_seed=1_068_798_027,
        conditions=("M0", "M1", "M2", "M3", "M4", "PLIF"),
        pilot_output_root=tmp_path / "pilot",
        formal_output_root=tmp_path / "formal",
    )
    monkeypatch.setattr(pilot_v7, "validate_v7_protocol_from_path", lambda _path: {})
    monkeypatch.setattr(pilot_v7, "expected_pilot_configs", lambda _protocol: ())
    monkeypatch.setattr(
        pilot_v7,
        "validate_health_recovery_release",
        lambda *_args, **_kwargs: dict(recovery),
    )

    plan = pilot_v7.expected_pilot_plan_payload(
        block=block,
        protocol_path=protocol_path,
        config_dir=tmp_path / "configs",
        manifest_rows=(),
        health_report={
            "git_commit": recovery["base_health_commit"],
            "compatibility_recovery": recovery,
            "environment": {"training_environment_sha256": "d" * 64},
        },
        repository_root=tmp_path,
    )

    assert plan["git_commit"] == recovery["recovery_commit"]
    assert plan["health_execution_git_commit"] == recovery["base_health_commit"]
    assert plan["health_compatibility_recovery"] == recovery


def _history() -> list[dict[str, Any]]:
    return [
        {
            "epoch": index,
            "val_accuracy": 0.40 + index * 0.001,
            "all_zero_block_gradient_batch_fraction": 0.02,
        }
        for index in range(120)
    ]


def test_threshold_summary_passes_frozen_v7_pilot_contract() -> None:
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
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    acceptance = protocol["pilot_acceptance"]
    pilot_seed = int(acceptance["pilot_seed"])
    health_seed = int(acceptance["health_seed"])
    protocol_path = tmp_path / "configs" / "protocol_v7_mechanism.yaml"
    config_dir = tmp_path / "configs" / "v7_mechanism_pilot_generated"
    output_root = tmp_path / str(acceptance["pilot_output_root"])
    protocol_path.parent.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    output_root.mkdir(parents=True)
    protocol_path.write_bytes((ROOT / "configs" / "protocol_v7_mechanism.yaml").read_bytes())
    manifest_path = config_dir / "matrix_manifest.json"
    csv_path = config_dir / "run_manifest.csv"
    manifest_path.write_text("{}\n", encoding="utf-8")
    csv_path.write_text("run_id\n", encoding="utf-8")
    conditions = list(validator.V7_ACTIVE_CONDITIONS)
    rows = [{"condition": condition} for condition in conditions]
    expected = {
        condition: SimpleNamespace(
            runtime=SimpleNamespace(
                run_id=f"E9_cifar100_d20_t6_{condition}_s{pilot_seed}"
            )
        )
        for condition in conditions
    }
    for config in expected.values():
        (output_root / config.runtime.run_id).mkdir()
    environment_hash = "a" * 64
    block = SimpleNamespace(
        dataset="cifar100",
        pilot_seed=pilot_seed,
        health_seed=health_seed,
        block_hash="b" * 64,
        pilot_output_root=output_root,
    )
    recovery = {
        "schema": "ta-lif-msresnet-v7-health-compatibility-recovery-release-v1",
        "recovery_commit": "e" * 40,
    }

    monkeypatch.setattr(validator, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(validator, "require_v7_author_freeze", lambda *_args, **_kwargs: {})
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
        lambda **_kwargs: (
            {"health": {}, "health_compatibility_recovery": recovery},
            [],
        ),
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
    assert report["health_compatibility_recovery"] == recovery
    dataset = report["datasets"]["cifar100"]
    assert dataset["environment_sha256"] == environment_hash
    assert dataset["health_compatibility_recovery"] == recovery
    assert list(dataset["runs"]) == conditions


def test_orchestrator_evidence_requires_identical_health_recovery_binding() -> None:
    recovery = {"schema": "sealed-v7-test", "recovery_commit": "e" * 40}
    block = SimpleNamespace(
        protocol_hash="1" * 64,
        acceptance_hash="2" * 64,
        block_hash="3" * 64,
        dataset="cifar100",
        health_seed=1068798027,
        pilot_seed=1673127435,
    )
    bound = {
        "health_path": "results/pilot/v7_mechanism/health.json",
        "health_sha256": "4" * 64,
        "attempt_receipt_path": "results/pilot/v7_mechanism/health.attempt.json",
        "attempt_receipt_sha256": "5" * 64,
        "plan_path": "environment/v7-plan.json",
        "plan_sha256": "6" * 64,
        "plan": {"git_commit": "e" * 40},
        "health_compatibility_recovery": recovery,
    }
    evidence = {
        "protocol_version": 7,
        "execution_stage": "pilot",
        "plan_version": 7,
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "health_report": bound["health_path"],
        "health_report_sha256": bound["health_sha256"],
        "attempt_receipt": bound["attempt_receipt_path"],
        "attempt_receipt_sha256": bound["attempt_receipt_sha256"],
        "pilot_plan": bound["plan_path"],
        "pilot_plan_sha256": bound["plan_sha256"],
        "git_commit": "e" * 40,
        "runtime_context": {
            "pass": True,
            "protocol_version": 7,
            "dataset": block.dataset,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "block_hash": block.block_hash,
            "training_environment_sha256": "7" * 64,
            "health_compatibility_recovery": recovery,
        },
    }

    assert validator._orchestrator_failures(
        evidence,
        block=block,
        bound=bound,
        environment_hash="7" * 64,
    ) == []

    evidence["runtime_context"]["health_compatibility_recovery"] = {
        **recovery,
        "recovery_commit": "f" * 40,
    }
    failures = validator._orchestrator_failures(
        evidence,
        block=block,
        bound=bound,
        environment_hash="7" * 64,
    )
    assert any("health compatibility recovery" in item for item in failures)


def test_main_refuses_to_overwrite_canonical_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results" / "pilot" / "v7_mechanism" / "validation.json"
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
