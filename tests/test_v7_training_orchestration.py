from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import talif_msresnet.train as trainer
from talif_msresnet.utils import sha256_file, stable_hash


def test_v7_run_requires_validated_orchestrator_evidence() -> None:
    config = SimpleNamespace(protocol_version=7)

    with pytest.raises(ValueError, match="requires validated pilot/formal execution evidence"):
        trainer.run(config)


def test_v7_formal_evidence_accepts_one_bound_cifar100_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        trainer,
        "artifact_paths_for_protocol",
        lambda _protocol: {"freeze_manifest": "FREEZE_MANIFEST_V7_MECHANISM.json"},
    )
    manifest_path = tmp_path / "FREEZE_MANIFEST_V7_MECHANISM.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    protocol_path = tmp_path / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8")
    protocol = {"protocol_version": 7}
    environment_sha256 = "a" * 64
    freeze_manifest = {
        "pilot_validation": {
            "status": "PASS",
            "pass": True,
            "protocol_hash": stable_hash(protocol),
            "sha256": "b" * 64,
            "training_environment_sha256": environment_sha256,
            "datasets": {
                "cifar100": {"environment_sha256": environment_sha256},
            },
        }
    }

    evidence = trainer._formal_execution_evidence(
        protocol,
        freeze_manifest,
        protocol_path=protocol_path,
    )

    assert evidence["protocol_version"] == 7
    assert evidence["execution_stage"] == "formal"
    assert evidence["training_environment_sha256"] == environment_sha256
    assert evidence["freeze_manifest_sha256"] == sha256_file(manifest_path)


def test_v7_pilot_plan_dispatch_builds_trainer_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        trainer,
        "PILOT_PLAN_PATH",
        tmp_path / "environment" / "unfrozen_pilot_plan.json",
    )
    protocol_path = tmp_path / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8")
    protocol = {
        "protocol_version": 7,
        "study_stage": "health_pilot_formal",
        "output_root": "results/formal_v7_mechanism",
    }
    protocol_hash = "c" * 64
    config_dir = tmp_path / "configs" / "v7_mechanism_pilot_generated"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "E9_cifar100_d20_t6_M0_s1673127435.yaml"
    config_path.write_text("run: v7-pilot\n", encoding="utf-8")
    conditions = ("M0", "M1", "M2", "M3", "M4", "PLIF")
    with (config_dir / "run_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("dataset", "condition", "config_file"))
        writer.writeheader()
        for condition in conditions:
            writer.writerow(
                {
                    "dataset": "cifar100",
                    "condition": condition,
                    "config_file": config_path.name,
                }
            )

    pilot_root = tmp_path / "results" / "pilot" / "v7_mechanism" / "cifar100_s1673127435"
    formal_root = tmp_path / "results" / "formal_v7_mechanism"
    plan_path = tmp_path / "environment" / "v7_mechanism_pilot_cifar100_s1673127435.json"
    plan_path.parent.mkdir(parents=True)
    health_path = tmp_path / "results" / "pilot" / "v7_mechanism" / "health.json"
    block = SimpleNamespace(
        pilot_plan=plan_path.resolve(),
        health_output=health_path,
        pilot_output_root=pilot_root,
        formal_output_root=formal_root,
        dataset="cifar100",
        seed=1673127435,
        health_seed=1068798027,
        pilot_seed=1673127435,
        conditions=conditions,
    )
    run_id = "E9_cifar100_d20_t6_M0_s1673127435"
    config = SimpleNamespace(
        config_hash="d" * 64,
        data=SimpleNamespace(dataset="cifar100"),
        runtime=SimpleNamespace(
            run_id=run_id,
            seed=block.pilot_seed,
            output_dir=str(pilot_root),
            device="cuda:0",
        ),
    )
    runs = []
    for condition in conditions:
        item = {
            "condition": condition,
            "run_id": f"E9_cifar100_d20_t6_{condition}_s{block.pilot_seed}",
            "config_hash": "unused",
            "config_file": config_path.relative_to(tmp_path).as_posix(),
            "config_file_sha256": sha256_file(config_path),
        }
        if condition == "M0":
            item["config_hash"] = config.config_hash
        runs.append(item)
    payload = {
        "version": 7,
        "artifact_class": "NON_REPORTABLE_V7_MECHANISM_PILOT_PLAN",
        "non_reportable": True,
        "protocol_hash": protocol_hash,
        "protocol_path": protocol_path.relative_to(tmp_path).as_posix(),
        "acceptance_hash": "e" * 64,
        "block_hash": "f" * 64,
        "dataset": "cifar100",
        "git_commit": "1" * 40,
        "health_report": health_path.relative_to(tmp_path).as_posix(),
        "health_report_sha256": "2" * 64,
        "attempt_receipt": "results/pilot/v7_mechanism/health.attempt.json",
        "attempt_receipt_sha256": "3" * 64,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "pilot_output_root": pilot_root.relative_to(tmp_path).as_posix(),
        "formal_output_root": formal_root.relative_to(tmp_path).as_posix(),
        "runs": runs,
    }
    plan_path.write_text(json.dumps(payload), encoding="utf-8")
    health_report = {"status": "PASS"}
    runtime_context = {
        "pass": True,
        "protocol_version": 7,
        "dataset": "cifar100",
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "block_hash": payload["block_hash"],
        "training_environment_sha256": "4" * 64,
    }

    monkeypatch.setattr(trainer, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(trainer, "resolve_v7_pilot_block", lambda *_args, **_kwargs: block)
    monkeypatch.setattr(trainer, "require_v7_author_freeze", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        trainer,
        "validate_v7_health_report",
        lambda *_args, **_kwargs: health_report,
    )
    monkeypatch.setattr(
        trainer,
        "expected_v7_pilot_plan_payload",
        lambda **_kwargs: dict(payload),
    )
    monkeypatch.setattr(trainer, "load_run_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        trainer,
        "_load_pilot_runtime_validator",
        lambda version: (
            lambda *_args, **_kwargs: dict(runtime_context)
            if version == 7
            else pytest.fail("wrong pilot runtime validator version")
        ),
    )

    evidence = trainer._validate_orchestrator_pilot_plan(
        plan_path,
        config=config,
        config_path=config_path,
        protocol_path=protocol_path,
        protocol_hash=protocol_hash,
        output_dir_was_explicit=True,
    )

    assert evidence["protocol_version"] == 7
    assert evidence["execution_stage"] == "pilot"
    assert evidence["health_seed"] == block.health_seed
    assert evidence["pilot_seed"] == block.pilot_seed
    assert evidence["runtime_context"] == runtime_context
