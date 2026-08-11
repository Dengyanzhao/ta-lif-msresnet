from __future__ import annotations

import dataclasses
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

import pilot_health_gate_v6 as gate

from talif_msresnet.config import load_protocol
from talif_msresnet.pilot_v6 import expected_pilot_configs


def _development_config(condition: str) -> Any:
    protocol = load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")
    config = next(
        item for item in expected_pilot_configs(protocol) if item.model.condition == condition
    )
    return dataclasses.replace(
        config,
        runtime=dataclasses.replace(
            config.runtime,
            seed=31_337,
            run_id=f"DEVELOPMENT_{condition}",
            output_dir="results/development/v6_health_gate_cpu",
        ),
    )


def _passing_condition_evidence(
    condition: str,
    *,
    adaptive_update: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    expects_adaptive = condition in gate.ADAPTIVE_CONDITIONS
    fixed = {
        "pass": True,
        "finite": True,
        "adaptive_parameter_names": ["neuron.adaptive"] if expects_adaptive else [],
        "adaptive_update": {"pass": True, "nonzero": adaptive_update, "parameters": {}},
    }
    schedule = {
        "pass": True,
        "finite": True,
        "adaptive_enabled_by_epoch": [False] * 5 + [expects_adaptive],
    }
    resume = {"pass": True, "resume_exact": True}
    return fixed, schedule, resume


def test_v6_preclaim_resolves_frozen_split_manifest_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "data" / "cifar100"
    dataset_root.mkdir(parents=True)
    manifest_path = tmp_path / "data" / "manifests" / "cifar100_seed2024.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}\n", encoding="utf-8")
    config = _development_config("M0")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            root="data/cifar100",
            split_manifest="data/manifests/{dataset}_seed{split_seed}.json",
        ),
    )
    monkeypatch.setattr(gate.v3_gate, "REPOSITORY_ROOT", tmp_path)

    resolved = gate.v3_gate._resolved_data_config(config)

    assert Path(resolved.data.root) == dataset_root.resolve()
    assert Path(resolved.data.split_manifest) == manifest_path.resolve()
    assert "{" not in resolved.data.split_manifest


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
        caller_state_before_first = torch.get_rng_state().clone()
        first = gate._load_fixed_health_batch(config, seed=123, required_batch_size=4)
        assert torch.equal(torch.get_rng_state(), caller_state_before_first)

        torch.rand(17)
        caller_state_before_second = torch.get_rng_state().clone()
        second = gate._load_fixed_health_batch(config, seed=123, required_batch_size=4)
        assert torch.equal(torch.get_rng_state(), caller_state_before_second)

    assert torch.equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert gate.legacy_gate.tensor_batch_sha256(first[1], first[2]) == (
        gate.legacy_gate.tensor_batch_sha256(second[1], second[2])
    )


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


@pytest.mark.parametrize("condition", ["M0", "M1"])
def test_nonadaptive_condition_summary_reports_no_update_and_passes(condition: str) -> None:
    fixed, schedule, resume = _passing_condition_evidence(
        condition,
        adaptive_update=False,
    )

    report = gate._condition_health_summary(
        condition,
        fixed=fixed,
        schedule=schedule,
        resume=resume,
    )

    assert report["status"] == "PASS"
    assert report["pass"] is True
    assert report["adaptive_parameter_names"] == []
    assert report["adaptive_parameter_update_nonzero"] is False


@pytest.mark.parametrize("condition", ["M2", "M3", "M4", "PLIF"])
def test_adaptive_condition_summary_requires_nonzero_update(condition: str) -> None:
    fixed, schedule, resume = _passing_condition_evidence(
        condition,
        adaptive_update=True,
    )
    passing = gate._condition_health_summary(
        condition,
        fixed=fixed,
        schedule=schedule,
        resume=resume,
    )

    fixed_without_update = {
        **fixed,
        "adaptive_update": {**fixed["adaptive_update"], "nonzero": False},
    }
    failing = gate._condition_health_summary(
        condition,
        fixed=fixed_without_update,
        schedule=schedule,
        resume=resume,
    )

    assert passing["pass"] is True
    assert passing["adaptive_parameter_update_nonzero"] is True
    assert failing["pass"] is False
    assert failing["adaptive_parameter_update_nonzero"] is False


def test_v6_initial_forward_is_bitwise_equivalent_on_cpu() -> None:
    configs = [_development_config(condition) for condition in gate.V6_ACTIVE_CONDITIONS]
    generator = torch.Generator().manual_seed(99)
    batch = (torch.randn(1, 3, 32, 32, generator=generator), torch.tensor([0]))

    report = gate._initial_forward_evidence(
        configs,
        device=torch.device("cpu"),
        batch=batch,
    )

    assert report["pass"] is True
    assert report["initial_forward_equivalent"] is True
    assert report["shared_initialization_equal"] is True
    assert set(report["output_sha256_by_condition"]) == set(gate.V6_ACTIVE_CONDITIONS)


@pytest.mark.parametrize("condition", ["M0", "M4", "PLIF"])
def test_v6_fixed_probe_uses_only_development_seed(condition: str) -> None:
    config = _development_config(condition)
    report = gate._fixed_probe(
        config,
        device=torch.device("cpu"),
        batch=(torch.randn(1, 3, 32, 32), torch.tensor([0])),
        steps=1,
    )

    assert config.runtime.seed not in {1_707_261_715, 22_068_314}
    assert report["pass"] is True
    assert report["optimizer_group_contract_pass"] is True
    if condition in gate.ADAPTIVE_CONDITIONS:
        assert report["adaptive_parameter_names"]
        assert report["adaptive_update"]["pass"] is True
    else:
        assert report["adaptive_parameter_names"] == []


def test_optimizer_state_report_requires_populated_finite_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    parameter.sum().backward()
    optimizer.step()

    report = gate._optimizer_state_report(optimizer)

    assert report["pass"] is True
    assert report["populated_parameter_state_count"] == 1


def test_schedule_and_resume_boundary_hold_on_cpu(tmp_path: Path) -> None:
    config = _development_config("M4")
    config = dataclasses.replace(
        config,
        optimizer=dataclasses.replace(
            config.optimizer,
            epochs=2,
            ta_start_fraction=0.5,
            warmup_epochs=2,
            milestones=(),
        ),
    )
    generator = torch.Generator().manual_seed(99)
    batch = (torch.randn(1, 3, 32, 32, generator=generator), torch.tensor([0]))

    schedule = gate._schedule_probe(
        config,
        device=torch.device("cpu"),
        batch=batch,
        epochs=2,
        train_batches=1,
        validation_batches=1,
    )
    resume = gate._checkpoint_resume_check(
        config,
        device=torch.device("cpu"),
        batch=batch,
        checkpoint_path=tmp_path / "last.pt",
        training_environment_sha256="a" * 64,
        boundary_epoch=0,
    )

    assert schedule["pass"] is True
    assert schedule["adaptive_enabled_by_epoch"] == [False, True]
    assert resume["resume_exact"] is True


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
    monkeypatch.setattr(gate, "validate_v6_runtime_environment", lambda *_args: None)
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
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def condition_probe(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal probe_calls
        probe_calls += 1
        raise RuntimeError("condition probe reached")

    monkeypatch.setattr(gate, "_initial_forward_evidence", condition_probe)
    protocol = {
        "pilot_acceptance": {
            "environment": {
                "expected_gpu_substring": "unused",
                "cublas_workspace_config": ":4096:8",
            },
            "health": {"fixed_batch_size": 2},
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
        "expected_fixed_batch_sha256": expected_hash,
    }

    if hash_matches:
        with pytest.raises(RuntimeError, match="condition probe reached"):
            gate.run_gate(**kwargs)
        assert probe_calls == 1
    else:
        with pytest.raises(gate.PilotV6Error, match="fixed batch differs from preclaim"):
            gate.run_gate(**kwargs)
        assert probe_calls == 0


def test_pilot_runtime_reconstructs_fixed_batch_with_seeded_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _development_config("M0")
    block = SimpleNamespace(
        dataset="cifar100",
        health_seed=123,
        pilot_seed=config.runtime.seed,
        block_hash="c" * 64,
    )
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    targets = torch.tensor([0, 1])
    batch_hash = gate.legacy_gate.tensor_batch_sha256(inputs, targets)
    environment_identity = {
        "device": "cpu",
        "hardware": {},
        "software": {},
        "precision": "float32",
    }
    environment_hash = "d" * 64
    runtime_sources = {"scripts/pilot_health_gate_v6.py": "e" * 64}
    helper_calls: list[tuple[int, int]] = []

    def fixed_batch_helper(
        _config: Any, *, seed: int, required_batch_size: int
    ) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, str], dict[str, str]]:
        helper_calls.append((seed, required_batch_size))
        return (
            config,
            inputs,
            targets,
            {"manifest_sha256": "f" * 64},
            {"split_source_fingerprint": "1" * 64},
        )

    monkeypatch.setattr(
        gate,
        "repository_git_identity",
        lambda _root: {"git_commit": "a" * 40, "tracked_clean": True},
    )
    monkeypatch.setattr(gate, "seed_everything", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        gate.legacy_gate,
        "require_target_cuda",
        lambda *_args, **_kwargs: {"device": "cpu", "name": "test"},
    )
    monkeypatch.setattr(gate.legacy_gate, "validate_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(gate, "validate_v6_runtime_environment", lambda *_args: None)
    monkeypatch.setattr(
        gate,
        "_training_environment_identity",
        lambda *_args, **_kwargs: (json.dumps(environment_identity), environment_hash),
    )
    monkeypatch.setattr(
        gate.legacy_gate,
        "gpu_idle_precheck",
        lambda *_args, **_kwargs: {"device_uuid": "CPU-test"},
    )
    monkeypatch.setattr(gate, "_load_fixed_health_batch", fixed_batch_helper)
    monkeypatch.setattr(gate, "_runtime_source_hashes", lambda: dict(runtime_sources))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    report = {
        "status": "PASS",
        "pass": True,
        "git_commit": "a" * 40,
        "environment": {
            "training_environment_sha256": environment_hash,
            "training_environment_identity": environment_identity,
            "gpu_idle_precheck": {"device_uuid": "CPU-test"},
        },
        "source": {
            "split_manifest_sha256": "f" * 64,
            "fixed_batch_sha256": batch_hash,
            "fixed_batch_shape": list(inputs.shape),
            "fixed_batch_size": 2,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "split_source_fingerprint": "1" * 64,
            "runtime_sources_sha256": runtime_sources,
        },
    }
    protocol = {
        "pilot_acceptance": {
            "environment": {
                "expected_gpu_substring": "unused",
                "cublas_workspace_config": ":4096:8",
            },
            "health": {"fixed_batch_size": 2},
        }
    }

    validated = gate.validate_current_runtime_against_health_report(
        report,
        protocol=protocol,
        reference_config=config,
        block=block,
        device="cpu",
    )

    assert validated["pass"] is True
    assert validated["fixed_batch_sha256"] == batch_hash
    assert helper_calls == [(block.health_seed, 2)]


def test_existing_attempt_receipt_blocks_before_run_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = tmp_path / "health.attempt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    block = SimpleNamespace(
        health_output=tmp_path / "health.json",
        attempt_receipt=receipt,
        pilot_plan=tmp_path / "plan.json",
        pilot_output_root=tmp_path / "pilot",
        dataset="cifar100",
        health_seed=1_707_261_715,
        pilot_seed=22_068_314,
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    invoked = False

    monkeypatch.setattr(
        gate,
        "load_protocol",
        lambda _path: {
            "protocol_version": 6,
            "pilot_acceptance": {"validation_output": "validation"},
        },
    )
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(gate, "require_v6_author_freeze", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        gate, "artifact_paths_for_protocol", lambda _protocol: {"pilot_matrix": "ignored"}
    )
    monkeypatch.setattr(gate, "_repository_path", lambda _value: config_dir)
    monkeypatch.setattr(gate, "_load_exact_configs", lambda *_args: ((), ()))

    def forbidden_run_gate(**_kwargs: Any) -> dict[str, Any]:
        nonlocal invoked
        invoked = True
        raise AssertionError("existing receipt must block before run_gate")

    monkeypatch.setattr(gate, "run_gate", forbidden_run_gate)

    assert (
        gate.main(["--protocol", str(tmp_path / "protocol.yaml"), "--config-dir", str(config_dir)])
        == 2
    )
    assert invoked is False


def test_precheck_only_does_not_construct_receipt_or_claim_seed(
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
        health_seed=1_707_261_715,
        pilot_seed=22_068_314,
    )

    monkeypatch.setattr(
        gate,
        "load_protocol",
        lambda _path: {
            "protocol_version": 6,
            "pilot_acceptance": {"validation_output": "validation"},
        },
    )
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(
        gate, "require_v6_author_freeze", lambda *_args, **_kwargs: {"signoff_sha256": "a"}
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
    monkeypatch.setattr(
        gate,
        "_preclaim_gate",
        lambda **_kwargs: {"device": "cuda:0", "fixed_batch_sha256": "a" * 64},
    )

    def forbidden_receipt(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("precheck-only must not construct an attempt receipt")

    def forbidden_run_gate(**_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("precheck-only must not execute the health gate")

    monkeypatch.setattr(gate, "attempt_receipt_payload", forbidden_receipt)
    monkeypatch.setattr(gate, "run_gate", forbidden_run_gate)

    assert (
        gate.main(
            [
                "--protocol",
                str(tmp_path / "protocol.yaml"),
                "--config-dir",
                str(config_dir),
                "--precheck-only",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out
    assert "V6_HEALTH_PRECHECK_ONLY_PASS" in stdout
    assert "NO_HEALTH_SEED_WAS_CLAIMED" in stdout
    assert receipt.exists() is False
    assert health_output.exists() is False


@pytest.mark.parametrize(
    "failure",
    [
        "hardware preclaim failure",
        "provenance preclaim failure",
        "loader preclaim failure",
    ],
)
def test_preclaim_failure_does_not_claim_seed_or_run_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    receipt = tmp_path / "health.attempt.json"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    conditions = tuple(gate.V6_ACTIVE_CONDITIONS)
    block = SimpleNamespace(
        health_output=tmp_path / "health.json",
        attempt_receipt=receipt,
        pilot_plan=tmp_path / "plan.json",
        pilot_output_root=tmp_path / "pilot",
        dataset="cifar100",
        conditions=conditions,
        health_seed=1_707_261_715,
        pilot_seed=22_068_314,
    )
    configs = tuple(
        SimpleNamespace(
            model=SimpleNamespace(condition=condition),
            runtime=SimpleNamespace(seed=22_068_314, deterministic=True, amp=False),
            data=SimpleNamespace(dataset="cifar100"),
        )
        for condition in conditions
    )
    run_gate_called = False
    receipt_called = False

    monkeypatch.setattr(
        gate,
        "load_protocol",
        lambda _path: {
            "protocol_version": 6,
            "pilot_acceptance": {
                "validation_output": "validation",
                "environment": {
                    "expected_gpu_substring": "RTX 5090",
                    "pytorch_version": "unused",
                    "cuda_runtime": "unused",
                    "precision": "float32",
                    "deterministic": True,
                    "cublas_workspace_config": ":4096:8",
                },
                "health": {"fixed_batch_size": 8},
            },
        },
    )
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(
        gate, "require_v6_author_freeze", lambda *_args, **_kwargs: {"signoff_sha256": "a"}
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
        lambda *_args: (tuple(tmp_path / f"{index}.yaml" for index in range(6)), configs),
    )
    monkeypatch.setattr(
        gate,
        "repository_git_identity",
        lambda _root: {"git_commit": "a" * 40, "tracked_clean": True},
    )
    monkeypatch.setattr(gate, "_require_head_bound_sources", lambda _paths: None)
    monkeypatch.setattr(gate, "_runtime_source_hashes", dict)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if failure.startswith("hardware"):
        monkeypatch.setattr(
            gate.legacy_gate,
            "require_target_cuda",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(gate.PilotV6Error(failure)),
        )
    else:
        monkeypatch.setattr(
            gate.legacy_gate,
            "require_target_cuda",
            lambda *_args, **_kwargs: {"device": "cuda:0", "name": "RTX 5090"},
        )
        monkeypatch.setattr(gate.legacy_gate, "validate_runtime_environment", lambda *_args: None)
        monkeypatch.setattr(gate, "validate_v6_runtime_environment", lambda *_args: "unused")
        monkeypatch.setattr(
            gate.legacy_gate,
            "gpu_idle_precheck",
            lambda *_args, **_kwargs: {"device_uuid": "GPU-test"},
        )
        monkeypatch.setattr(
            gate,
            "_training_environment_identity",
            lambda *_args, **_kwargs: (
                '{"device":"cuda:0","hardware":{},"software":{},"precision":"float32","determinism":{}}',
                "a" * 64,
            ),
        )
        if failure.startswith("provenance"):
            monkeypatch.setattr(
                gate,
                "validate_v6_cifar100_provenance_files",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(failure)),
            )
            monkeypatch.setattr(
                gate.v3_gate,
                "load_fixed_real_batch",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("provenance failure must block before the loader")
                ),
            )
        else:
            monkeypatch.setattr(
                gate.v3_gate,
                "load_fixed_real_batch",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(gate.PilotV6Error(failure)),
            )

    def forbidden_receipt(**_kwargs: Any) -> dict[str, Any]:
        nonlocal receipt_called
        receipt_called = True
        raise AssertionError("preclaim failure must happen before receipt construction")

    def forbidden_run_gate(**_kwargs: Any) -> dict[str, Any]:
        nonlocal run_gate_called
        run_gate_called = True
        raise AssertionError("preclaim failure must block before run_gate")

    monkeypatch.setattr(gate, "attempt_receipt_payload", forbidden_receipt)
    monkeypatch.setattr(gate, "run_gate", forbidden_run_gate)

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
        == 2
    )
    assert receipt.exists() is False
    assert receipt_called is False
    assert run_gate_called is False
