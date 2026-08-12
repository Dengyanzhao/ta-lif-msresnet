from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pilot_health_gate_v7 as gate

from talif_msresnet import pilot_v7
from talif_msresnet.config import load_protocol
from talif_msresnet.config_v7 import (
    V7_CIFAR100_PROVENANCE_CONTRACT,
    V7_PILOT_ACCEPTANCE,
)
from talif_msresnet.pilot_v7 import (
    ATTEMPT_ARTIFACT_CLASS,
    ATTEMPT_SCHEMA_VERSION,
    ATTEMPT_STATUS,
    HEALTH_ARTIFACT_CLASS,
    HEALTH_SCHEMA_VERSION,
    HEALTH_SEED_DISPOSITION,
    PILOT_SEED_DISPOSITION,
    REPORTING_ELIGIBILITY,
    V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS,
    V7_HEALTH_RECOVERY_CORRECTED_FIELD,
    V7_HEALTH_RECOVERY_SCHEMA,
    V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT,
    V7_HEALTH_RUNTIME_SOURCE_PATHS,
    expected_pilot_configs,
    resolve_pilot_block,
    validate_attempt_receipt,
    validate_health_recovery_release,
    validate_health_report_payload,
)


def _development_config(condition: str) -> Any:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    config = next(
        item for item in expected_pilot_configs(protocol) if item.model.condition == condition
    )
    return dataclasses.replace(
        config,
        runtime=dataclasses.replace(
            config.runtime,
            seed=31_337,
            run_id=f"DEVELOPMENT_{condition}",
            output_dir="results/development/v7_health_gate_cpu",
        ),
    )


def _expected_updates() -> dict[str, bool]:
    return dict(
        V7_PILOT_ACCEPTANCE["health"]["expected_adaptive_update_by_condition"]
    )


@pytest.mark.parametrize("condition", ["M0", "M1", "M2", "M3", "M4", "PLIF"])
def test_v7_optimizer_manifest_uses_frozen_adaptive_role(condition: str) -> None:
    config = _development_config(condition)
    _model, optimizer, _scheduler, adaptive, names, activation = gate._build_objects(
        config, torch.device("cpu")
    )
    manifest = gate.optimizer_group_manifest(_model, optimizer)
    adaptive_groups = [
        group for group in manifest["groups"] if group.get("role") == "adaptive"
    ]

    assert activation == 5
    if _expected_updates()[condition]:
        assert adaptive
        assert names
        assert len(adaptive_groups) == 1
        assert adaptive_groups[0]["parameter_names"] == names
        assert adaptive_groups[0]["initial_lr"] == pytest.approx(0.0025)
        assert adaptive_groups[0]["weight_decay"] == 0.0
    else:
        assert adaptive == []
        assert names == []
        assert adaptive_groups == []


def _passing_condition_evidence(
    condition: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expects_adaptive = _expected_updates()[condition]
    fixed = {
        "pass": True,
        "finite": True,
        "adaptive_parameter_names": ["neuron.adaptive"] if expects_adaptive else [],
        "adaptive_update": {
            "pass": True,
            "nonzero": expects_adaptive,
            "parameters": {},
        },
    }
    schedule = {
        "pass": True,
        "finite": True,
        "adaptive_enabled_by_epoch": [False] * 5 + [expects_adaptive],
    }
    resume = {"pass": True, "resume_exact": True}
    return fixed, schedule, resume


def test_fixed_health_batch_is_seeded_and_restores_caller_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def random_loader(
        base: Any, *, seed: int, required_batch_size: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, str], dict[str, str]]:
        del seed
        return (
            base,
            torch.rand(required_batch_size, 3, 2, 2),
            torch.arange(required_batch_size),
            {"manifest_sha256": "a" * 64},
            {},
        )

    monkeypatch.setattr(gate.v3_gate, "load_fixed_real_batch", random_loader)
    config = _development_config("M0")

    with torch.random.fork_rng(devices=[]):
        initial = torch.Generator(device="cpu").manual_seed(901).get_state()
        torch.set_rng_state(initial)
        state_before_first = torch.get_rng_state().clone()
        first = gate._load_fixed_health_batch(config, seed=123, required_batch_size=4)
        assert torch.equal(torch.get_rng_state(), state_before_first)

        torch.rand(17)
        state_before_second = torch.get_rng_state().clone()
        second = gate._load_fixed_health_batch(config, seed=123, required_batch_size=4)
        assert torch.equal(torch.get_rng_state(), state_before_second)

    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_fixed_health_batch_seed_changes_random_augmentation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def random_loader(
        base: Any, *, seed: int, required_batch_size: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, str], dict[str, str]]:
        del seed
        return (
            base,
            torch.rand(required_batch_size, 3, 2, 2),
            torch.arange(required_batch_size),
            {"manifest_sha256": "a" * 64},
            {},
        )

    monkeypatch.setattr(gate.v3_gate, "load_fixed_real_batch", random_loader)
    config = _development_config("M0")

    first = gate._load_fixed_health_batch(config, seed=123, required_batch_size=4)
    second = gate._load_fixed_health_batch(config, seed=124, required_batch_size=4)

    assert not torch.equal(first[1], second[1])


def test_expected_adaptive_mapping_requires_exact_frozen_order_and_booleans() -> None:
    valid = {
        "expected_adaptive_update_by_condition": _expected_updates(),
    }
    assert gate._expected_adaptive_updates(valid) == _expected_updates()

    wrong_order = dict(reversed(tuple(_expected_updates().items())))
    with pytest.raises(gate.PilotV7Error, match="frozen condition order"):
        gate._expected_adaptive_updates(
            {"expected_adaptive_update_by_condition": wrong_order}
        )

    wrong_type = _expected_updates()
    wrong_type["M0"] = 0  # type: ignore[assignment]
    with pytest.raises(gate.PilotV7Error, match="must be boolean"):
        gate._expected_adaptive_updates(
            {"expected_adaptive_update_by_condition": wrong_type}
        )


@pytest.mark.parametrize("condition", ["M0", "M1"])
def test_nonadaptive_condition_passes_only_without_update(condition: str) -> None:
    fixed, schedule, resume = _passing_condition_evidence(condition)
    passing = gate._condition_health_summary(
        condition,
        fixed=fixed,
        schedule=schedule,
        resume=resume,
        expected_adaptive_update_by_condition=_expected_updates(),
    )
    unexpected_update = copy.deepcopy(fixed)
    unexpected_update["adaptive_update"]["nonzero"] = True
    failing = gate._condition_health_summary(
        condition,
        fixed=unexpected_update,
        schedule=schedule,
        resume=resume,
        expected_adaptive_update_by_condition=_expected_updates(),
    )

    assert passing["pass"] is True
    assert passing["adaptive_parameter_update_nonzero"] is False
    assert failing["pass"] is False


@pytest.mark.parametrize("condition", ["M2", "M3", "M4", "PLIF"])
def test_adaptive_condition_requires_nonzero_update(condition: str) -> None:
    fixed, schedule, resume = _passing_condition_evidence(condition)
    passing = gate._condition_health_summary(
        condition,
        fixed=fixed,
        schedule=schedule,
        resume=resume,
        expected_adaptive_update_by_condition=_expected_updates(),
    )
    missing_update = copy.deepcopy(fixed)
    missing_update["adaptive_update"]["nonzero"] = False
    failing = gate._condition_health_summary(
        condition,
        fixed=missing_update,
        schedule=schedule,
        resume=resume,
        expected_adaptive_update_by_condition=_expected_updates(),
    )

    assert passing["pass"] is True
    assert passing["adaptive_parameter_update_nonzero"] is True
    assert failing["pass"] is False


def test_condition_expectation_is_supplied_by_contract_not_hardcoded() -> None:
    fixed, schedule, resume = _passing_condition_evidence("M0")
    fixed["adaptive_parameter_names"] = ["neuron.adaptive"]
    fixed["adaptive_update"]["nonzero"] = True
    schedule["adaptive_enabled_by_epoch"] = [False] * 5 + [True]
    custom = _expected_updates()
    custom["M0"] = True

    report = gate._condition_health_summary(
        "M0",
        fixed=fixed,
        schedule=schedule,
        resume=resume,
        expected_adaptive_update_by_condition=custom,
    )

    assert report["pass"] is True
    assert report["expected_adaptive_parameter_update_nonzero"] is True


def test_preclaim_double_reconstruction_mismatch_blocks_before_seed_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _development_config("M0")
    block = SimpleNamespace(
        conditions=("M0",),
        health_seed=123,
        pilot_seed=config.runtime.seed,
        attempt_receipt=tmp_path / "health.attempt.json",
    )
    identity = {"git_commit": "a" * 40, "tracked_clean": True}
    calls = 0

    monkeypatch.setattr(gate, "_configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(
        gate.legacy_gate,
        "require_target_cuda",
        lambda *_args, **_kwargs: {"device": "cuda:0", "name": "RTX 5090"},
    )
    monkeypatch.setattr(gate.legacy_gate, "validate_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(gate, "validate_v7_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(
        gate.legacy_gate,
        "gpu_idle_precheck",
        lambda *_args, **_kwargs: {"device_uuid": "GPU-test"},
    )
    monkeypatch.setattr(
        gate,
        "_training_environment_identity",
        lambda *_args, **_kwargs: (
            '{"device":"cuda:0","hardware":{},"software":{},"precision":"float32"}',
            "b" * 64,
        ),
    )
    monkeypatch.setattr(gate, "repository_git_identity", lambda _root: dict(identity))
    monkeypatch.setattr(gate, "_runtime_source_hashes", dict)
    monkeypatch.setattr(
        gate,
        "validate_v7_cifar100_provenance_files",
        lambda *_args, **_kwargs: {},
    )

    def inconsistent_loader(
        _config: Any, *, seed: int, required_batch_size: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, str], dict[str, str]]:
        nonlocal calls
        del seed
        calls += 1
        inputs = torch.full((required_batch_size, 3, 2, 2), float(calls))
        return (
            config,
            inputs,
            torch.arange(required_batch_size, dtype=torch.long),
            {"manifest_sha256": "c" * 64},
            {"split_source_fingerprint": "d" * 64},
        )

    monkeypatch.setattr(gate, "_load_fixed_health_batch", inconsistent_loader)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    protocol = {
        "pilot_acceptance": {
            "environment": {
                "expected_gpu_substring": "RTX 5090",
                "cublas_workspace_config": ":4096:8",
            },
            "health": {"fixed_batch_size": 2},
        }
    }

    with pytest.raises(gate.PilotV7Error, match="reconstructed exactly"):
        gate._preclaim_gate(
            protocol=protocol,
            configs=(config,),
            block=block,
            device_name="cuda:0",
            git_identity=identity,
            runtime_sources={},
            source_binding={},
        )

    assert calls == 2
    assert block.attempt_receipt.exists() is False


@pytest.mark.parametrize("hash_matches", [False, True])
def test_run_gate_binds_preclaim_batch_before_condition_probes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hash_matches: bool,
) -> None:
    config = _development_config("M0")
    block = SimpleNamespace(
        conditions=("M0",),
        health_seed=123,
        pilot_seed=config.runtime.seed,
        health_output=tmp_path / "health.json",
    )
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    targets = torch.tensor([0, 1])
    observed_hash = gate.legacy_gate.tensor_batch_sha256(inputs, targets)
    expected_hash = observed_hash if hash_matches else "0" * 64
    probe_calls = 0

    monkeypatch.setattr(gate, "seed_everything", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        gate.legacy_gate,
        "require_target_cuda",
        lambda *_args, **_kwargs: {"device": "cpu", "name": "test"},
    )
    monkeypatch.setattr(gate.legacy_gate, "validate_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(gate, "validate_v7_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(
        gate.legacy_gate,
        "gpu_idle_precheck",
        lambda *_args, **_kwargs: {"device_uuid": "CPU-test"},
    )
    monkeypatch.setattr(
        gate,
        "_training_environment_identity",
        lambda *_args, **_kwargs: (
            '{"device":"cpu","hardware":{},"software":{},"precision":"float32"}',
            "a" * 64,
        ),
    )
    monkeypatch.setattr(
        gate,
        "_load_fixed_health_batch",
        lambda *_args, **_kwargs: (
            config,
            inputs,
            targets,
            {"manifest_sha256": "b" * 64},
            {},
        ),
    )
    monkeypatch.setattr(
        gate,
        "validate_v7_cifar100_provenance_files",
        lambda *_args, **_kwargs: _provenance_evidence(),
    )
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def condition_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal probe_calls
        probe_calls += 1
        raise RuntimeError("condition probe reached")

    monkeypatch.setattr(gate, "_initial_forward_evidence", condition_probe)
    health_contract = {
        "fixed_batch_size": 2,
        "expected_adaptive_update_by_condition": _expected_updates(),
    }
    protocol = {
        "pilot_acceptance": {
            "environment": {
                "expected_gpu_substring": "unused",
                "cublas_workspace_config": ":4096:8",
            },
            "health": health_contract,
        }
    }
    kwargs = {
        "protocol": protocol,
        "protocol_path": tmp_path / "protocol.yaml",
        "config_paths": (),
        "configs": (config,),
        "block": block,
        "device_name": "cpu",
        "git_identity": {"git_commit": "a" * 40, "tracked_clean": True},
        "runtime_sources": {},
        "source_binding": {},
        "attempt_receipt_sha256": "f" * 64,
        "expected_fixed_batch_sha256": expected_hash,
    }

    if hash_matches:
        with pytest.raises(RuntimeError, match="condition probe reached"):
            gate.run_gate(**kwargs)
        assert probe_calls == 1
    else:
        with pytest.raises(gate.PilotV7Error, match="fixed batch differs from preclaim"):
            gate.run_gate(**kwargs)
        assert probe_calls == 0


def test_precheck_only_creates_no_seed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = tmp_path / "health.attempt.json"
    health_output = tmp_path / "health.json"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    block = SimpleNamespace(
        health_output=health_output,
        attempt_receipt=receipt,
        pilot_plan=tmp_path / "plan.json",
        pilot_output_root=tmp_path / "pilot",
        dataset="cifar100",
        health_seed=1_068_798_027,
        pilot_seed=1_673_127_435,
    )

    monkeypatch.setattr(
        gate,
        "load_protocol",
        lambda _path: {
            "protocol_version": 7,
            "pilot_acceptance": {"validation_output": "validation"},
        },
    )
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(
        gate, "require_v7_author_freeze", lambda *_args, **_kwargs: {"signoff_sha256": "a"}
    )
    monkeypatch.setattr(
        gate,
        "artifact_paths_for_protocol",
        lambda _protocol: {"pilot_matrix": "matrix", "signoff": "signoff"},
    )
    monkeypatch.setattr(
        gate,
        "_repository_path",
        lambda value: config_dir if value == "matrix" else tmp_path / str(value),
    )
    monkeypatch.setattr(gate, "_load_exact_configs", lambda *_args: ((), ()))
    monkeypatch.setattr(
        gate,
        "repository_git_identity",
        lambda _root: {"git_commit": "a" * 40, "tracked_clean": True},
    )
    monkeypatch.setattr(gate, "_require_head_bound_sources", lambda _paths: None)
    monkeypatch.setattr(gate, "_runtime_source_hashes", dict)
    monkeypatch.setattr(gate, "health_source_binding", lambda **_kwargs: {})
    monkeypatch.setattr(
        gate,
        "_preclaim_gate",
        lambda **_kwargs: {
            "device": "cuda:0",
            "fixed_batch_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        gate,
        "attempt_receipt_payload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("precheck-only must not construct a receipt")
        ),
    )
    monkeypatch.setattr(
        gate,
        "run_gate",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("precheck-only must not execute health probes")
        ),
    )

    assert gate.main(["--protocol", str(tmp_path / "protocol.yaml"), "--precheck-only"]) == 0
    stdout = capsys.readouterr().out
    assert "V7_HEALTH_PRECHECK_ONLY_PASS" in stdout
    assert "NO_HEALTH_SEED_WAS_CLAIMED" in stdout
    assert receipt.exists() is False
    assert health_output.exists() is False


def test_pilot_compatibility_precheck_preserves_evidence_and_pilot_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = tmp_path / "health.attempt.json"
    health_output = tmp_path / "health.json"
    receipt_bytes = b'{"status":"ATTEMPT_CLAIMED_SEED_CONSUMED_NO_RETRY"}\n'
    health_bytes = b'{"status":"PASS","pass":true}\n'
    receipt.write_bytes(receipt_bytes)
    health_output.write_bytes(health_bytes)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    recovery = {"recovery_commit": "f" * 40}
    block = SimpleNamespace(
        health_output=health_output,
        attempt_receipt=receipt,
        pilot_plan=tmp_path / "plan.json",
        pilot_output_root=tmp_path / "pilot",
        dataset="cifar100",
        health_seed=1_068_798_027,
        pilot_seed=1_673_127_435,
    )
    protocol = {
        "protocol_version": 7,
        "pilot_acceptance": {"validation_output": "validation.json"},
    }

    monkeypatch.setattr(gate, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(
        gate, "require_v7_author_freeze", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        gate,
        "artifact_paths_for_protocol",
        lambda _protocol: {"pilot_matrix": "matrix", "signoff": "signoff"},
    )
    monkeypatch.setattr(
        gate,
        "_repository_path",
        lambda value: config_dir if value == "matrix" else tmp_path / str(value),
    )
    reference_config = SimpleNamespace()
    monkeypatch.setattr(
        gate, "_load_exact_configs", lambda *_args: ((), (reference_config,))
    )
    monkeypatch.setattr(gate, "_require_head_bound_sources", lambda _paths: None)
    monkeypatch.setattr(
        gate,
        "validate_health_report",
        lambda *_args, **_kwargs: {"compatibility_recovery": recovery},
    )
    monkeypatch.setattr(
        gate,
        "validate_current_runtime_against_health_report",
        lambda *_args, **_kwargs: {"health_compatibility_recovery": recovery},
    )
    monkeypatch.setattr(
        gate,
        "attempt_receipt_payload",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("compatibility precheck must not construct a receipt")
        ),
    )
    monkeypatch.setattr(
        gate,
        "run_gate",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("compatibility precheck must not execute health probes")
        ),
    )

    assert gate.main(
        [
            "--protocol",
            str(tmp_path / "protocol.yaml"),
            "--pilot-compatibility-precheck-only",
        ]
    ) == 0
    stdout = capsys.readouterr().out
    assert "V7_PILOT_COMPATIBILITY_PRECHECK_PASS" in stdout
    assert "NO_HEALTH_OR_PILOT_SEED_WAS_CLAIMED" in stdout
    assert receipt.read_bytes() == receipt_bytes
    assert health_output.read_bytes() == health_bytes
    assert block.pilot_plan.exists() is False
    assert block.pilot_output_root.exists() is False
    assert (tmp_path / "validation.json").exists() is False


def test_existing_v7_receipt_blocks_retry_before_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "health.attempt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    block = SimpleNamespace(
        health_output=tmp_path / "health.json",
        attempt_receipt=receipt,
        pilot_plan=tmp_path / "plan.json",
        pilot_output_root=tmp_path / "pilot",
    )
    invoked = False

    monkeypatch.setattr(
        gate,
        "load_protocol",
        lambda _path: {
            "protocol_version": 7,
            "pilot_acceptance": {"validation_output": "validation"},
        },
    )
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(gate, "require_v7_author_freeze", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        gate,
        "artifact_paths_for_protocol",
        lambda _protocol: {"pilot_matrix": "matrix"},
    )
    monkeypatch.setattr(gate, "_repository_path", lambda _value: config_dir)
    monkeypatch.setattr(gate, "_load_exact_configs", lambda *_args: ((), ()))

    def forbidden_gate(**_kwargs: Any) -> dict[str, Any]:
        nonlocal invoked
        invoked = True
        raise AssertionError("existing receipt must block before run_gate")

    monkeypatch.setattr(gate, "run_gate", forbidden_gate)

    assert gate.main(["--protocol", str(tmp_path / "protocol.yaml")]) == 2
    assert invoked is False


def test_postclaim_failure_writes_canonical_error_without_rereading_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "health.attempt.json"
    health_output = tmp_path / "health.json"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    conditions = tuple(_expected_updates())
    block = SimpleNamespace(
        health_output=health_output,
        attempt_receipt=receipt,
        pilot_plan=tmp_path / "plan.json",
        pilot_output_root=tmp_path / "pilot",
        dataset="cifar100",
        conditions=conditions,
        health_seed=1_068_798_027,
        pilot_seed=1_673_127_435,
        protocol_hash="1" * 64,
        acceptance_hash="2" * 64,
        block_hash="3" * 64,
    )
    configs = tuple(
        SimpleNamespace(
            model=SimpleNamespace(condition=condition),
            runtime=SimpleNamespace(seed=block.pilot_seed),
        )
        for condition in conditions
    )
    protocol = {
        "protocol_version": 7,
        "pilot_acceptance": {"validation_output": "validation"},
    }

    monkeypatch.setattr(gate, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(
        gate, "require_v7_author_freeze", lambda *_args, **_kwargs: {"signoff_sha256": "a"}
    )
    monkeypatch.setattr(
        gate,
        "artifact_paths_for_protocol",
        lambda _protocol: {"pilot_matrix": "matrix", "signoff": "signoff"},
    )
    monkeypatch.setattr(
        gate,
        "_repository_path",
        lambda value: config_dir if value == "matrix" else tmp_path / str(value),
    )
    monkeypatch.setattr(
        gate,
        "_load_exact_configs",
        lambda *_args: (
            tuple(tmp_path / f"{condition}.yaml" for condition in conditions),
            configs,
        ),
    )
    monkeypatch.setattr(
        gate,
        "repository_git_identity",
        lambda _root: {"git_commit": "a" * 40, "tracked_clean": True},
    )
    monkeypatch.setattr(gate, "_require_head_bound_sources", lambda _paths: None)
    monkeypatch.setattr(gate, "_runtime_source_hashes", lambda: {"source.py": "b" * 64})
    monkeypatch.setattr(gate, "health_source_binding", lambda **_kwargs: {"bound": True})
    monkeypatch.setattr(
        gate,
        "_preclaim_gate",
        lambda **_kwargs: {"device": "cuda:0", "fixed_batch_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        gate,
        "attempt_receipt_payload",
        lambda **_kwargs: {"receipt": "claimed"},
    )
    monkeypatch.setattr(
        gate,
        "run_gate",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("simulated post-claim failure")),
    )
    monkeypatch.setattr(
        gate,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("claimed receipt must not be reread on the fatal path")
        ),
    )

    assert (
        gate.main(
            [
                "--protocol",
                str(tmp_path / "protocol.yaml"),
                "--config-dir",
                str(config_dir),
                "--device",
                "cuda:0",
            ]
        )
        == 1
    )
    assert receipt.is_file()
    report = gate.json.loads(health_output.read_text(encoding="utf-8"))
    assert report["status"] == "ERROR"
    assert report["pass"] is False
    assert report["fatal_error"] == "OSError: simulated post-claim failure"
    assert report["attempt_receipt_sha256"] == gate.json_file_payload_sha256(
        {"receipt": "claimed"}
    )


def _source_binding(block: Any) -> dict[str, Any]:
    return {
        "protocol": "configs/protocol_v7_mechanism.yaml",
        "protocol_file_sha256": "1" * 64,
        "configs": [
            {
                "condition": condition,
                "path": f"configs/v7_mechanism_pilot_generated/{condition}.yaml",
                "file_sha256": f"{index:x}" * 64,
                "config_hash": f"{index + 6:x}" * 64,
            }
            for index, condition in enumerate(block.conditions, start=1)
        ],
        "runtime_sources_sha256": {
            path: "a" * 64 for path in V7_HEALTH_RUNTIME_SOURCE_PATHS
        },
    }


def _provenance_evidence() -> dict[str, Any]:
    contract = V7_CIFAR100_PROVENANCE_CONTRACT
    return {
        "cifar100_source_provenance": contract["source_provenance_path"],
        "cifar100_source_provenance_sha256": contract["source_provenance_sha256"],
        "cifar100_archive": contract["archive_path"],
        "cifar100_archive_sha256": contract["archive_sha256"],
        "cifar100_train_pickle": contract["train_pickle_path"],
        "cifar100_train_pickle_sha256": contract["train_pickle_sha256"],
        "cifar100_test_pickle": contract["test_pickle_path"],
        "cifar100_test_pickle_sha256": contract["test_pickle_sha256"],
        "cifar100_meta_pickle": contract["meta_pickle_path"],
        "cifar100_meta_pickle_sha256": contract["meta_pickle_sha256"],
        "cifar100_split_manifest": contract["split_manifest_path"],
        "cifar100_split_manifest_sha256": contract["split_manifest_sha256"],
    }


def _valid_receipt(block: Any) -> dict[str, Any]:
    return {
        "schema_version": ATTEMPT_SCHEMA_VERSION,
        "artifact_class": ATTEMPT_ARTIFACT_CLASS,
        "status": ATTEMPT_STATUS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "claimed_at": "2026-08-11T00:00:00+00:00",
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": "a" * 40,
        "tracked_clean": True,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
        "source": _source_binding(block),
    }


def _valid_report(block: Any) -> dict[str, Any]:
    updates = _expected_updates()
    by_condition: dict[str, Any] = {}
    for condition in block.conditions:
        expects_adaptive = updates[condition]
        by_condition[condition] = {
            "status": "PASS",
            "pass": True,
            "finite": True,
            "resume_exact": True,
            "adaptive_enabled_by_epoch": [False] * 5 + [expects_adaptive],
            "adaptive_parameter_names": ["adaptive"] if expects_adaptive else [],
            "expected_adaptive_parameter_update_nonzero": expects_adaptive,
            "adaptive_parameter_update_nonzero": expects_adaptive,
            "fixed_batch": {"pass": True},
            "formal_schedule_boundary": {"pass": True},
            "checkpoint_resume": {"pass": True},
        }
    batch_sha256 = "d" * 64
    split_sha256 = "e" * 64
    environment_sha256 = "c" * 64
    source_binding = _source_binding(block)
    provenance = _provenance_evidence()
    environment_identity = {"device": "cuda:0", "precision": "float32"}
    gpu_idle = {"device_uuid": "GPU-v7-test"}
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
        "git_commit": "a" * 40,
        "tracked_clean": True,
        "attempt_receipt": V7_PILOT_ACCEPTANCE["attempt_receipt"],
        "attempt_receipt_sha256": "b" * 64,
        "integrity_anomalies": [],
        "failures": [],
        "source": {
            **source_binding,
            "reference_config_hash": "9" * 64,
            "reference_config_seed": block.pilot_seed,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "fixed_batch_sha256": batch_sha256,
            "fixed_batch_shape": [32, 3, 32, 32],
            "fixed_batch_size": 32,
            "split_manifest_sha256": split_sha256,
            "split_source_fingerprint": V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT,
            "cifar100_provenance": provenance,
        },
        "preclaim": {
            "pass": True,
            "source": copy.deepcopy(source_binding),
            "cifar100_provenance": copy.deepcopy(provenance),
            "fixed_batch_sha256": batch_sha256,
            "fixed_batch_reconstruction_sha256": batch_sha256,
            "fixed_batch_reconstruction_count": 2,
            "split_manifest_sha256": split_sha256,
            "training_environment_sha256": environment_sha256,
            "training_environment_identity": copy.deepcopy(environment_identity),
            "gpu_idle_precheck": copy.deepcopy(gpu_idle),
        },
        "environment": {
            "contract": copy.deepcopy(V7_PILOT_ACCEPTANCE["environment"]),
            "training_environment_identity": environment_identity,
            "training_environment_sha256": environment_sha256,
            "gpu_idle_precheck": gpu_idle,
        },
        "checks": {
            "contract": copy.deepcopy(V7_PILOT_ACCEPTANCE["health"]),
            "expected_adaptive_update_by_condition": updates,
            "initial_forward_equivalent": True,
            "shared_initialization_equal": True,
            "initial_evidence": {
                "pass": True,
                "initial_forward_equivalent": True,
                "shared_initialization_equal": True,
                "output_sha256_by_condition": {
                    condition: "6" * 64 for condition in block.conditions
                },
                "shared_weight_sha256_by_condition": {
                    condition: "7" * 64 for condition in block.conditions
                },
            },
            "by_condition": by_condition,
        },
    }


def test_health_report_requires_protocol_frozen_adaptive_mapping() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    block = resolve_pilot_block(protocol, repository_root=ROOT)
    receipt = _valid_receipt(block)
    report = _valid_report(block)
    assert validate_health_report_payload(report, block=block, receipt=receipt)["pass"]

    report["checks"]["expected_adaptive_update_by_condition"]["M0"] = True
    with pytest.raises(gate.PilotV7Error, match="exact PASS"):
        validate_health_report_payload(report, block=block, receipt=receipt)


@pytest.mark.parametrize("invalid", ["8" * 64, "cifar100:49999", "cifar100"])
def test_health_report_accepts_only_exact_static_split_fingerprint(
    invalid: str,
) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    block = resolve_pilot_block(protocol, repository_root=ROOT)
    receipt = _valid_receipt(block)
    report = _valid_report(block)
    assert validate_health_report_payload(report, block=block, receipt=receipt)["pass"]

    report["source"]["split_source_fingerprint"] = invalid
    with pytest.raises(
        gate.PilotV7Error, match="source.split_source_fingerprint"
    ):
        validate_health_report_payload(report, block=block, receipt=receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixed_batch_sha256", "f" * 64),
        ("fixed_batch_reconstruction_sha256", "f" * 64),
        ("fixed_batch_reconstruction_count", 1),
        ("split_manifest_sha256", "f" * 64),
        ("training_environment_sha256", "f" * 64),
    ],
)
def test_health_report_requires_bound_preclaim_evidence(
    field: str, value: Any
) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    block = resolve_pilot_block(protocol, repository_root=ROOT)
    receipt = _valid_receipt(block)
    report = _valid_report(block)
    report["preclaim"][field] = value

    with pytest.raises(gate.PilotV7Error, match="exact PASS"):
        validate_health_report_payload(report, block=block, receipt=receipt)


def test_health_report_requires_shared_initialization_equality() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    block = resolve_pilot_block(protocol, repository_root=ROOT)
    receipt = _valid_receipt(block)
    report = _valid_report(block)
    report["checks"]["shared_initialization_equal"] = False

    with pytest.raises(gate.PilotV7Error, match="exact PASS"):
        validate_health_report_payload(report, block=block, receipt=receipt)


@pytest.mark.parametrize(
    "mutation",
    ["missing_source", "missing_config", "missing_runtime_source", "invalid_config_hash"],
)
def test_attempt_receipt_requires_complete_exact_source_binding(mutation: str) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    block = resolve_pilot_block(protocol, repository_root=ROOT)
    receipt = _valid_receipt(block)

    if mutation == "missing_source":
        receipt.pop("source")
    elif mutation == "missing_config":
        receipt["source"]["configs"].pop()
    elif mutation == "missing_runtime_source":
        receipt["source"]["runtime_sources_sha256"].pop(
            V7_HEALTH_RUNTIME_SOURCE_PATHS[0]
        )
    else:
        receipt["source"]["configs"][0]["config_hash"] = "not-a-sha256"

    with pytest.raises(gate.PilotV7Error, match="Invalid V7 health attempt receipt"):
        validate_attempt_receipt(receipt, block=block)


@pytest.mark.parametrize(
    "mutation",
    [
        "report_receipt_source_mismatch",
        "preclaim_receipt_source_mismatch",
        "condition_false_pass",
        "missing_initial_evidence",
        "git_commit_mismatch",
    ],
)
def test_health_report_rejects_incomplete_or_cross_bound_evidence(
    mutation: str,
) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    block = resolve_pilot_block(protocol, repository_root=ROOT)
    receipt = _valid_receipt(block)
    report = _valid_report(block)

    if mutation == "report_receipt_source_mismatch":
        report["source"]["protocol_file_sha256"] = "0" * 64
    elif mutation == "preclaim_receipt_source_mismatch":
        report["preclaim"]["source"]["protocol_file_sha256"] = "0" * 64
    elif mutation == "condition_false_pass":
        report["checks"]["by_condition"]["M0"]["pass"] = False
    elif mutation == "missing_initial_evidence":
        report["checks"].pop("initial_evidence")
    else:
        report["git_commit"] = "0" * 40

    with pytest.raises(gate.PilotV7Error, match="exact PASS"):
        validate_health_report_payload(report, block=block, receipt=receipt)


def test_canonical_health_validator_rejects_synchronized_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = ROOT / "configs" / "protocol_v7_mechanism.yaml"
    protocol = load_protocol(protocol_path)
    canonical = resolve_pilot_block(protocol, repository_root=ROOT)
    block = dataclasses.replace(
        canonical,
        health_output=tmp_path / "health.json",
        attempt_receipt=tmp_path / "health.attempt.json",
        pilot_output_root=tmp_path / "pilot",
        pilot_plan=tmp_path / "plan.json",
        formal_output_root=tmp_path / "formal",
    )
    current_source = _source_binding(block)
    receipt = _valid_receipt(block)
    receipt["source"] = copy.deepcopy(current_source)
    report = _valid_report(block)
    report["source"].update(copy.deepcopy(current_source))
    report["preclaim"]["source"] = copy.deepcopy(current_source)

    monkeypatch.setattr(
        pilot_v7, "resolve_pilot_block", lambda *_args, **_kwargs: block
    )
    monkeypatch.setattr(
        pilot_v7,
        "current_health_source_binding",
        lambda **_kwargs: copy.deepcopy(current_source),
    )

    pilot_v7.exclusive_create_json(block.attempt_receipt, receipt)
    report["attempt_receipt_sha256"] = gate.sha256_file(block.attempt_receipt)
    pilot_v7.exclusive_create_json(block.health_output, report)
    assert pilot_v7.validate_health_report(
        block.health_output,
        protocol_path,
        repository_root=ROOT,
    )["pass"]

    synchronized = "0" * 64
    receipt["source"]["protocol_file_sha256"] = synchronized
    report["source"]["protocol_file_sha256"] = synchronized
    report["preclaim"]["source"]["protocol_file_sha256"] = synchronized
    block.attempt_receipt.write_text(
        gate.json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report["attempt_receipt_sha256"] = gate.sha256_file(block.attempt_receipt)
    block.health_output.write_text(
        gate.json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(
        gate.PilotV7Error, match="differs from current frozen files"
    ):
        pilot_v7.validate_health_report(
            block.health_output,
            protocol_path,
            repository_root=ROOT,
        )


def test_v6_failure_evidence_cannot_authorize_v7_pilot() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    block = resolve_pilot_block(protocol, repository_root=ROOT)
    receipt = _valid_receipt(block)
    report = _valid_report(block)
    report.update(
        {
            "artifact_class": "NON_REPORTABLE_V6_MECHANISM_IMPLEMENTATION_HEALTH_GATE",
            "status": "FAIL",
            "pass": False,
            "health_seed": 1_707_261_715,
        }
    )

    with pytest.raises(gate.PilotV7Error, match="exact PASS"):
        validate_health_report_payload(report, block=block, receipt=receipt)


def test_v7_health_paths_and_seeds_are_isolated_from_v6() -> None:
    v6 = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    v7 = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    v7_block = resolve_pilot_block(v7, repository_root=ROOT)

    assert v7_block.health_seed != v6["pilot_acceptance"]["health_seed"]
    assert v7_block.pilot_seed != v6["pilot_acceptance"]["pilot_seed"]
    assert "v7_mechanism" in v7_block.health_output.as_posix()
    assert "v6_mechanism" not in v7_block.health_output.as_posix()
    assert all(config.runtime.run_id.startswith("E9_") for config in expected_pilot_configs(v7))


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _sealed_v7_health_recovery_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_implementation_path: bool = False,
    release_overrides: dict[str, Any] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    root = tmp_path / "recovery"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "V7 Recovery Test")
    _git(root, "config", "user.email", "v7-recovery@example.invalid")
    for relative in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"base:{relative}\n", encoding="utf-8")
    _git(root, "add", "--", *V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS)
    _git(root, "commit", "-m", "base health source")
    base_commit = _git(root, "rev-parse", "HEAD")

    receipt_path = root / "evidence" / "health.attempt.json"
    health_path = root / "evidence" / "health.json"
    protocol_hash = "c" * 64
    receipt = {"git_commit": base_commit, "protocol_hash": protocol_hash}
    pilot_v7.exclusive_create_json(receipt_path, receipt)
    attempt_sha256 = gate.sha256_file(receipt_path)
    report = {
        "status": "PASS",
        "pass": True,
        "git_commit": base_commit,
        "protocol_hash": protocol_hash,
        "attempt_receipt_sha256": attempt_sha256,
        "source": {
            "split_source_fingerprint": (
                V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT
            )
        },
    }
    pilot_v7.exclusive_create_json(health_path, report)
    health_sha256 = gate.sha256_file(health_path)

    monkeypatch.setattr(pilot_v7, "V7_HEALTH_RECOVERY_BASE_COMMIT", base_commit)
    monkeypatch.setattr(
        pilot_v7, "V7_HEALTH_RECOVERY_HEALTH_SHA256", health_sha256
    )
    monkeypatch.setattr(
        pilot_v7,
        "V7_HEALTH_RECOVERY_HEALTH_PATH",
        health_path.relative_to(root).as_posix(),
    )
    monkeypatch.setattr(
        pilot_v7, "V7_HEALTH_RECOVERY_ATTEMPT_SHA256", attempt_sha256
    )
    monkeypatch.setattr(
        pilot_v7,
        "V7_HEALTH_RECOVERY_ATTEMPT_PATH",
        receipt_path.relative_to(root).as_posix(),
    )
    monkeypatch.setattr(
        pilot_v7, "V7_HEALTH_RECOVERY_PROTOCOL_HASH", protocol_hash
    )

    for relative in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS:
        (root / relative).write_text(f"recovered:{relative}\n", encoding="utf-8")
    implementation_paths = list(V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS)
    if extra_implementation_path:
        unexpected = root / "unexpected_recovery_change.txt"
        unexpected.write_text("not authorized\n", encoding="utf-8")
        implementation_paths.append(unexpected.relative_to(root).as_posix())
    _git(root, "add", "--", *implementation_paths)
    _git(root, "commit", "-m", "implement health compatibility recovery")
    implementation_commit = _git(root, "rev-parse", "HEAD")
    implementation_tree = _git(root, "rev-parse", f"{implementation_commit}^{{tree}}")
    implementation_sha256 = {
        relative: pilot_v7.hashlib.sha256(
            _git_bytes(root, "show", f"{implementation_commit}:{relative}")
        ).hexdigest()
        for relative in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS
    }
    release = {
        "schema": V7_HEALTH_RECOVERY_SCHEMA,
        "base_health_commit": base_commit,
        "implementation_commit": implementation_commit,
        "implementation_tree": implementation_tree,
        "health_report_path": health_path.relative_to(root).as_posix(),
        "health_report_sha256": health_sha256,
        "attempt_receipt_path": receipt_path.relative_to(root).as_posix(),
        "attempt_receipt_sha256": attempt_sha256,
        "protocol_hash": protocol_hash,
        "corrected_field": V7_HEALTH_RECOVERY_CORRECTED_FIELD,
        "split_source_fingerprint": (
            V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT
        ),
        "allowed_changed_paths": list(V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS),
        "implementation_file_sha256": implementation_sha256,
    }
    if release_overrides is not None:
        release.update(release_overrides)
    release["record_sha256"] = pilot_v7.stable_hash(release)
    release_path = root / pilot_v7.V7_HEALTH_RECOVERY_RELEASE_RECORD
    pilot_v7.exclusive_create_json(release_path, release)
    _git(root, "add", "--", pilot_v7.V7_HEALTH_RECOVERY_RELEASE_RECORD)
    _git(root, "commit", "-m", "seal health compatibility recovery")
    recovery_commit = _git(root, "rev-parse", "HEAD")
    monkeypatch.setattr(
        pilot_v7, "V7_HEALTH_RECOVERY_SEAL_COMMIT", recovery_commit
    )
    return root, health_path, receipt_path, report, receipt


def test_health_recovery_release_accepts_only_sealed_two_commit_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, health_path, receipt_path, report, receipt = (
        _sealed_v7_health_recovery_fixture(tmp_path, monkeypatch)
    )

    binding = validate_health_recovery_release(
        root,
        health_path=health_path,
        attempt_receipt_path=receipt_path,
        report=report,
        receipt=receipt,
    )

    assert binding["base_health_commit"] == report["git_commit"]
    assert binding["recovery_commit"] == _git(root, "rev-parse", "HEAD")
    assert binding["observed_changes"] == [
        f"M\t{path}" for path in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS
    ]
    assert binding["tracked_clean"] is True


def test_health_recovery_release_rejects_changed_evidence_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, health_path, receipt_path, _report, _receipt = (
        _sealed_v7_health_recovery_fixture(tmp_path, monkeypatch)
    )
    health_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(pilot_v7.PilotV7Error, match="report SHA-256"):
        validate_health_recovery_release(
            root,
            health_path=health_path,
            attempt_receipt_path=receipt_path,
        )


def test_health_recovery_release_rejects_dirty_tracked_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _health_path, _receipt_path, _report, _receipt = (
        _sealed_v7_health_recovery_fixture(tmp_path, monkeypatch)
    )
    changed = root / V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS[0]
    changed.write_text("dirty\n", encoding="utf-8")

    with pytest.raises(pilot_v7.PilotV7Error, match="clean tracked Git worktree"):
        validate_health_recovery_release(root)


def test_health_recovery_release_rejects_extra_implementation_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _health_path, _receipt_path, _report, _receipt = (
        _sealed_v7_health_recovery_fixture(
            tmp_path,
            monkeypatch,
            extra_implementation_path=True,
        )
    )

    with pytest.raises(pilot_v7.PilotV7Error, match="exact modification-only allowlist"):
        validate_health_recovery_release(root)


def test_health_recovery_release_rejects_tampered_sealed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _health_path, _receipt_path, _report, _receipt = (
        _sealed_v7_health_recovery_fixture(
            tmp_path,
            monkeypatch,
            release_overrides={"corrected_field": "source.unrelated_field"},
        )
    )

    with pytest.raises(pilot_v7.PilotV7Error, match="corrected_field"):
        validate_health_recovery_release(root)


def test_health_recovery_release_accepts_authorized_descendant_after_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _health_path, _receipt_path, _report, _receipt = (
        _sealed_v7_health_recovery_fixture(tmp_path, monkeypatch)
    )
    _git(root, "commit", "--allow-empty", "-m", "unauthorized commit after seal")

    binding = validate_health_recovery_release(root)
    assert binding["recovery_commit"] == pilot_v7.V7_HEALTH_RECOVERY_SEAL_COMMIT


def test_device_recovery_release_uses_git_object_identity_after_crlf_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Windows text checkout must not invalidate a sealed JSON record."""
    root = tmp_path / "device-recovery"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "V7 Recovery Test")
    _git(root, "config", "user.email", "v7-recovery@example.invalid")
    record = root / pilot_v7.V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD
    record.write_bytes(b'{"record_sha256":"x"}\n')
    _git(root, "add", "--", record.name)
    _git(root, "commit", "-m", "seal device recovery record")
    seal = _git(root, "rev-parse", "HEAD")
    record.write_bytes(b'{"record_sha256":"x"}\r\n')
    assert pilot_v7._git_blob_id(root, seal, record.name) == pilot_v7._git_blob_id(
        root, "HEAD", record.name
    )
