from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_matrix as matrix_runner

import talif_msresnet.freeze as freeze_module
import talif_msresnet.train as trainer
from talif_msresnet.freeze import FreezeGateError
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.utils import sha256_file, stable_hash


def _load_freeze_tool():
    path = ROOT / "scripts" / "create_freeze_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "create_freeze_manifest_v5_execution_tests", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze_tool = _load_freeze_tool()


def test_v5_manifest_contract_lists_every_gate_source() -> None:
    assert set(freeze_tool.V5_REQUIRED_SOURCE_PATHS) == freeze_module.V5_GATE_SOURCE_PATHS
    assert {
        "src/talif_msresnet/pilot_v3.py",
        "src/talif_msresnet/pilot_v4.py",
        "scripts/calibrate_v5_health.py",
        "scripts/pilot_health_gate.py",
        "scripts/pilot_health_gate_v3.py",
        "scripts/evaluate_checkpoints.py",
        "scripts/validate_v5_pilot.py",
    } <= freeze_module.V5_GATE_SOURCE_PATHS


def test_runtime_v5_gate_rejects_manifest_without_pilot_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = tmp_path / "configs" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 5\n", encoding="utf-8")
    matrix_dir = tmp_path / "configs" / "matrix"
    matrix_dir.mkdir()
    stored = {
        "protocol": {
            "path": "configs/protocol.yaml",
            "canonical_sha256": "a" * 64,
        },
        "generated_matrix": {"directory": "configs/matrix"},
    }
    monkeypatch.setattr(
        freeze_module,
        "load_protocol",
        lambda _path: {"protocol_version": 5},
    )
    monkeypatch.setattr(
        freeze_module,
        "_load_manifest_tool",
        lambda _root: SimpleNamespace(verify_manifest=lambda **_kwargs: stored),
    )

    with pytest.raises(FreezeGateError, match="no bound aggregate pilot validation"):
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST.json",
        )


def test_v5_formal_evidence_binds_one_shared_pilot_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = {"protocol_version": 5, "artifact_paths": {"freeze_manifest": "freeze.json"}}
    manifest_path = tmp_path / "freeze.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    environment_sha256 = "e" * 64
    pilot = {
        "status": "PASS",
        "pass": True,
        "protocol_hash": stable_hash(protocol),
        "sha256": "b" * 64,
        "path": "results/pilot/v5_talif_only/validation.json",
        "training_environment_sha256": environment_sha256,
        "datasets": {
            "cifar100": {"environment_sha256": environment_sha256},
            "cifar10dvs": {"environment_sha256": environment_sha256},
        },
    }
    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        trainer,
        "artifact_paths_for_protocol",
        lambda _protocol: {"freeze_manifest": "freeze.json"},
    )

    evidence = trainer._formal_execution_evidence(
        protocol,
        {"pilot_validation": pilot},
        protocol_path=tmp_path / "protocol.yaml",
    )

    assert evidence["protocol_version"] == 5
    assert evidence["execution_stage"] == "formal"
    assert evidence["training_environment_sha256"] == environment_sha256


def test_v5_matrix_launch_requires_c1_c2_with_pilot_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    health = tmp_path / "health.json"
    health.write_text("{}\n", encoding="utf-8")
    block = SimpleNamespace(
        dataset="cifar100",
        conditions=("C1", "C2"),
        health_seed=101,
        pilot_seed=202,
        pilot_output_root=tmp_path / "pilot",
        pilot_plan=tmp_path / "plan.json",
        health_output=health,
    )
    monkeypatch.setattr(matrix_runner, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(matrix_runner, "resolve_v5_pilot_block", lambda *_a, **_k: block)
    monkeypatch.setattr(
        matrix_runner,
        "validate_v5_health_report",
        lambda *_a, **_k: {"status": "PASS", "pass": True},
    )
    rows = [
        {
            "experiment": "E1",
            "dataset": "cifar100",
            "depth": "20",
            "time_steps": "6",
            "seed": "202",
            "condition": condition,
        }
        for condition in ("C1", "C2")
    ]

    resolved, output_root, plan, report = matrix_runner._validate_v5_pilot_launch_contract(
        protocol={"protocol_version": 5},
        protocol_path=tmp_path / "protocol.yaml",
        selected_rows=rows,
        dataset="cifar100",
        output_root=None,
        pilot_plan=None,
    )

    assert resolved is block
    assert output_root == block.pilot_output_root
    assert plan == block.pilot_plan
    assert report["pass"] is True
    for row in rows:
        row["seed"] = str(block.health_seed)
    with pytest.raises(ValueError, match="pilot-seed binding"):
        matrix_runner._validate_v5_pilot_launch_contract(
            protocol={"protocol_version": 5},
            protocol_path=tmp_path / "protocol.yaml",
            selected_rows=rows,
            dataset="cifar100",
            output_root=None,
            pilot_plan=None,
        )


def test_trainer_revalidates_exact_v5_two_run_pilot_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text("protocol_version: 5\n", encoding="utf-8")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "run-c1.yaml"
    config_path.write_text("run: c1\n", encoding="utf-8")
    c2_path = config_dir / "run-c2.yaml"
    c2_path.write_text("run: c2\n", encoding="utf-8")
    with (config_dir / "run_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "condition", "config_file"),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"dataset": "cifar100", "condition": "C1", "config_file": config_path.name},
                {"dataset": "cifar100", "condition": "C2", "config_file": c2_path.name},
            )
        )
    pilot_root = tmp_path / "pilot"
    formal_root = tmp_path / "formal"
    plan_path = tmp_path / "plan.json"
    health_path = tmp_path / "health.json"
    health_path.write_text("{}\n", encoding="utf-8")
    block = SimpleNamespace(
        dataset="cifar100",
        seed=202,
        pilot_seed=202,
        health_seed=101,
        pilot_plan=plan_path.resolve(),
        health_output=health_path,
    )
    protocol = {
        "protocol_version": 5,
        "study_stage": "pilot_and_formal",
        "protocol_status": {"frozen": True},
        "output_root": str(formal_root),
    }
    protocol_hash = "p" * 64
    payload = {
        "version": 5,
        "artifact_class": "NON_REPORTABLE_V5_PILOT_PLAN",
        "non_reportable": True,
        "dataset": "cifar100",
        "health_seed": 101,
        "pilot_seed": 202,
        "protocol_hash": protocol_hash,
        "protocol_path": artifact_path_reference(protocol_path, ROOT),
        "pilot_output_root": artifact_path_reference(pilot_root, ROOT),
        "formal_output_root": artifact_path_reference(formal_root, ROOT),
        "acceptance_hash": "a" * 64,
        "development_probe_hash": "d" * 64,
        "block_hash": "b" * 64,
        "git_commit": "c" * 40,
        "health_report": artifact_path_reference(health_path, ROOT),
        "health_report_sha256": sha256_file(health_path),
        "attempt_receipt": "receipt.json",
        "attempt_receipt_sha256": "r" * 64,
        "runs": [
            {
                "condition": "C1",
                "run_id": "run-c1",
                "config_file": artifact_path_reference(config_path, ROOT),
                "config_file_sha256": sha256_file(config_path),
                "config_hash": "config-hash",
            },
            {
                "condition": "C2",
                "run_id": "run-c2",
                "config_file": artifact_path_reference(c2_path, ROOT),
                "config_file_sha256": sha256_file(c2_path),
                "config_hash": "other-hash",
            },
        ],
    }
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    config = SimpleNamespace(
        data=SimpleNamespace(dataset="cifar100"),
        runtime=SimpleNamespace(
            seed=202,
            output_dir=str(pilot_root),
            run_id="run-c1",
            device="cuda:0",
        ),
        config_hash="config-hash",
    )
    monkeypatch.setattr(trainer, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(trainer, "resolve_v5_pilot_block", lambda *_a, **_k: block)
    monkeypatch.setattr(trainer, "require_v5_author_freeze", lambda *_a, **_k: {})
    monkeypatch.setattr(
        trainer,
        "validate_v5_health_report",
        lambda *_a, **_k: {"status": "PASS", "pass": True},
    )
    monkeypatch.setattr(
        trainer,
        "expected_v5_pilot_plan_payload",
        lambda **_kwargs: dict(payload),
    )
    monkeypatch.setattr(
        trainer,
        "_load_pilot_runtime_validator",
        lambda _version: lambda *_a, **_k: {"pass": True, "pilot_seed": 202},
    )
    monkeypatch.setattr(
        trainer,
        "load_run_config",
        lambda *_a, **_k: SimpleNamespace(),
    )

    evidence = trainer._validate_orchestrator_pilot_plan(
        plan_path,
        config=config,
        config_path=config_path,
        protocol_path=protocol_path,
        protocol_hash=protocol_hash,
        output_dir_was_explicit=True,
    )

    assert evidence["protocol_version"] == 5
    assert evidence["execution_stage"] == "pilot"
    assert evidence["health_seed"] == 101
    assert evidence["pilot_seed"] == 202
    assert evidence["runtime_context"]["pass"] is True
