from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_v7_pilot as validator

from talif_msresnet import pilot_v7
from talif_msresnet.config import load_protocol, validate_run_mapping
from talif_msresnet.config_v7 import generate_v7_pilot_matrix
from talif_msresnet.utils import stable_hash


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


def _v7_recovery_artifact_config(
    tmp_path: Path,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    raw = copy.deepcopy(generate_v7_pilot_matrix(protocol)[0])
    config = validate_run_mapping(raw, protocol)
    artifact = config.as_dict()
    artifact["runtime"]["device"] = "cuda:0"
    artifact["runtime"]["output_dir"] = str(
        tmp_path / config.runtime.output_dir
    )
    return protocol, config, artifact


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "cpu", None])
def test_device_recovery_rejects_any_unapproved_device_alias(
    tmp_path: Path, device: str | None
) -> None:
    protocol, config, artifact = _v7_recovery_artifact_config(tmp_path)
    artifact["runtime"]["device"] = device

    _hash, failures = validator._recovery_config_identity(
        artifact,
        label="artifact",
        protocol=protocol,
        repository_root=tmp_path,
        expected_config=config,
    )

    assert failures
    assert any("reviewed execution device" in failure for failure in failures)


def test_device_recovery_binds_raw_execution_hash_and_all_config_channels(
    tmp_path: Path,
) -> None:
    protocol, config, artifact = _v7_recovery_artifact_config(tmp_path)
    execution_hash = stable_hash(artifact)
    run_manifest = {"config": copy.deepcopy(artifact), "execution_hash": execution_hash}

    assert validator._stored_recovery_execution_identity_failures(
        run_manifest["config"],
        run_manifest["execution_hash"],
        label="run_manifest.json",
        protocol=protocol,
        repository_root=tmp_path,
        expected_config=config,
        reference_raw=artifact,
        reference_execution_hash=execution_hash,
    ) == []

    run_manifest["execution_hash"] = "0" * 64
    failures = validator._stored_recovery_execution_identity_failures(
        run_manifest["config"],
        run_manifest["execution_hash"],
        label="run_manifest.json",
        protocol=protocol,
        repository_root=tmp_path,
        expected_config=config,
        reference_raw=artifact,
        reference_execution_hash=execution_hash,
    )
    assert any("does not hash its raw config" in failure for failure in failures)

    run_manifest["execution_hash"] = execution_hash
    run_manifest["config"]["runtime"]["device"] = "cuda:1"
    failures = validator._stored_recovery_execution_identity_failures(
        run_manifest["config"],
        run_manifest["execution_hash"],
        label="run_manifest.json",
        protocol=protocol,
        repository_root=tmp_path,
        expected_config=config,
        reference_raw=artifact,
        reference_execution_hash=execution_hash,
    )
    assert failures


def test_device_recovery_binds_both_checkpoint_execution_identities(
    tmp_path: Path,
) -> None:
    protocol, config, artifact = _v7_recovery_artifact_config(tmp_path)
    execution_hash = stable_hash(artifact)
    resolved_artifact = json.loads(json.dumps(artifact))
    assert type(artifact["optimizer"]["milestones"]) is tuple
    assert type(resolved_artifact["optimizer"]["milestones"]) is list
    checkpoint = {
        "config": copy.deepcopy(artifact),
        "execution_hash": execution_hash,
    }
    best = tmp_path / "best.pt"
    last = tmp_path / "last.pt"
    torch.save(checkpoint, best)
    torch.save(checkpoint, last)

    for path in (best, last):
        assert validator._checkpoint_recovery_execution_identity_failures(
            path,
            label=path.name,
            protocol=protocol,
            repository_root=tmp_path,
            expected_config=config,
            reference_raw=resolved_artifact,
            reference_execution_hash=execution_hash,
        ) == []

    checkpoint["execution_hash"] = "0" * 64
    torch.save(checkpoint, last)
    failures = validator._checkpoint_recovery_execution_identity_failures(
        last,
        label=last.name,
        protocol=protocol,
        repository_root=tmp_path,
        expected_config=config,
        reference_raw=resolved_artifact,
        reference_execution_hash=execution_hash,
    )
    assert any("does not hash its raw config" in failure for failure in failures)


def test_checkpoint_serialization_equivalence_is_exactly_milestones_tuple_to_list(
    tmp_path: Path,
) -> None:
    protocol, config, artifact = _v7_recovery_artifact_config(tmp_path)
    resolved = json.loads(json.dumps(artifact))
    execution_hash = stable_hash(artifact)

    assert validator._checkpoint_milestones_container_equivalent(artifact, resolved)

    changed_value = copy.deepcopy(artifact)
    changed_value["optimizer"]["milestones"] = (
        *changed_value["optimizer"]["milestones"][:-1],
        changed_value["optimizer"]["milestones"][-1] + 1,
    )
    assert not validator._checkpoint_milestones_container_equivalent(
        changed_value, resolved
    )

    changed_elsewhere = copy.deepcopy(artifact)
    changed_elsewhere["analysis"]["unexpected_tuple"] = (1, 2)
    resolved_elsewhere = copy.deepcopy(resolved)
    resolved_elsewhere["analysis"]["unexpected_tuple"] = [1, 2]
    assert not validator._checkpoint_milestones_container_equivalent(
        changed_elsewhere, resolved_elsewhere
    )

    run_manifest_failures = validator._stored_recovery_execution_identity_failures(
        artifact,
        execution_hash,
        label="run_manifest.json",
        protocol=protocol,
        repository_root=tmp_path,
        expected_config=config,
        reference_raw=resolved,
        reference_execution_hash=execution_hash,
    )
    assert run_manifest_failures == [
        "run_manifest.json raw config differs from resolved_config.json"
    ]


@pytest.mark.parametrize(
    ("checkpoint_container", "resolved_container"),
    [
        (list, list),
        (tuple, tuple),
        (list, tuple),
        (set, list),
    ],
)
def test_checkpoint_serialization_rejects_every_other_container_direction(
    tmp_path: Path,
    checkpoint_container,
    resolved_container,
) -> None:
    _protocol, _config, artifact = _v7_recovery_artifact_config(tmp_path)
    checkpoint = copy.deepcopy(artifact)
    resolved = json.loads(json.dumps(artifact))
    values = artifact["optimizer"]["milestones"]
    checkpoint["optimizer"]["milestones"] = checkpoint_container(values)
    resolved["optimizer"]["milestones"] = resolved_container(values)

    assert not validator._checkpoint_milestones_container_equivalent(
        checkpoint, resolved
    )


@pytest.mark.parametrize("missing_from", ["checkpoint", "resolved"])
def test_checkpoint_serialization_requires_the_exact_milestones_path(
    tmp_path: Path, missing_from: str
) -> None:
    _protocol, _config, artifact = _v7_recovery_artifact_config(tmp_path)
    checkpoint = copy.deepcopy(artifact)
    resolved = json.loads(json.dumps(artifact))
    target = checkpoint if missing_from == "checkpoint" else resolved
    target["optimizer"].pop("milestones")

    assert not validator._checkpoint_milestones_container_equivalent(
        checkpoint, resolved
    )


def test_checkpoint_serialization_is_type_strict_outside_the_allowed_path(
    tmp_path: Path,
) -> None:
    _protocol, _config, artifact = _v7_recovery_artifact_config(tmp_path)
    checkpoint = copy.deepcopy(artifact)
    resolved = json.loads(json.dumps(artifact))
    checkpoint["analysis"]["test_access"] = True
    resolved["analysis"]["test_access"] = 1

    assert not validator._checkpoint_milestones_container_equivalent(
        checkpoint, resolved
    )


def test_checkpoint_entry_requires_tuple_to_list_even_when_raw_payloads_match(
    tmp_path: Path,
) -> None:
    protocol, config, artifact = _v7_recovery_artifact_config(tmp_path)
    checkpoint_raw = json.loads(json.dumps(artifact))
    execution_hash = stable_hash(checkpoint_raw)
    path = tmp_path / "list_checkpoint.pt"
    torch.save(
        {"config": checkpoint_raw, "execution_hash": execution_hash},
        path,
    )

    failures = validator._checkpoint_recovery_execution_identity_failures(
        path,
        label=path.name,
        protocol=protocol,
        repository_root=tmp_path,
        expected_config=config,
        reference_raw=copy.deepcopy(checkpoint_raw),
        reference_execution_hash=execution_hash,
    )

    assert failures == [
        "list_checkpoint.pt raw config differs from resolved_config.json"
    ]


def test_device_recovery_requires_exactly_one_trainer_start_on_cuda_zero(
    tmp_path: Path,
) -> None:
    _protocol, config, _artifact = _v7_recovery_artifact_config(tmp_path)
    start = {
        "event": "run_started",
        "device": "cuda:0",
        "dry_run": False,
        "config_hash": config.config_hash,
    }

    assert validator._trainer_recovery_launch_failures([start], expected_config=config) == []
    assert validator._trainer_recovery_launch_failures([], expected_config=config)
    assert validator._trainer_recovery_launch_failures(
        [start, start], expected_config=config
    )
    wrong_device = {**start, "device": "cuda:1"}
    assert any(
        "reviewed execution device" in failure
        for failure in validator._trainer_recovery_launch_failures(
            [wrong_device], expected_config=config
        )
    )


def test_device_recovery_matrix_audits_exact_cuda_command_and_uuid(
    tmp_path: Path,
) -> None:
    _protocol, config, _artifact = _v7_recovery_artifact_config(tmp_path)
    output = tmp_path / "results"
    output.mkdir()
    uuid = "GPU-test"
    environment_hash = "a" * 64
    context = {
        "device": "cuda:0",
        "device_uuid": uuid,
        "training_environment_sha256": environment_hash,
    }
    events = [
        {"event": "run_context_validated", "run_id": config.runtime.run_id, "context": context},
        {
            "event": "run_started",
            "run_id": config.runtime.run_id,
            "command": ["python", "-m", "talif_msresnet.train", "--device", "cuda:0"],
        },
    ]
    (output / "matrix_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    bound = {
        "health": {
            "environment": {
                "training_environment_sha256": environment_hash,
                "gpu_idle_precheck": {"device_uuid": uuid},
            }
        }
    }

    assert validator._matrix_recovery_launch_failures(
        output, expected_run_ids={config.runtime.run_id}, bound=bound
    ) == []

    events[1]["command"][-1] = "cuda:1"
    (output / "matrix_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8"
    )
    failures = validator._matrix_recovery_launch_failures(
        output, expected_run_ids={config.runtime.run_id}, bound=bound
    )
    assert any("--device cuda:0" in failure for failure in failures)
