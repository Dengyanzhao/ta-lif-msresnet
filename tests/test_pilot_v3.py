from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pilot_health_gate_v3 as health_gate  # noqa: E402
import run_matrix as matrix_runner  # noqa: E402
import create_freeze_manifest as freeze_manifest  # noqa: E402
import talif_msresnet.train as trainer  # noqa: E402
from talif_msresnet.config import generate_v3_pilot_matrix, load_protocol  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.utils import stable_hash  # noqa: E402
from talif_msresnet.pilot_v3 import (  # noqa: E402
    PilotV3Error,
    exclusive_create_json,
    require_v3_author_freeze,
    resolve_pilot_block,
    validate_fixed_batch_shape,
)


PROTOCOL_PATH = ROOT / "configs" / "protocol_v3_talif_only.yaml"


def _unsigned_protocol() -> dict:
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    status = protocol["protocol_status"]
    status["frozen"] = False
    status["confirmed_by"] = None
    status["confirmed_at"] = None
    status["confirmations"] = {
        field: False for field in status["confirmations"]
    }
    return protocol


def test_v3_health_shape_contract_covers_static_and_event_inputs() -> None:
    assert validate_fixed_batch_shape(
        "cifar100", (64, 3, 32, 32), minimum_batch_size=64, time_steps=6, in_channels=3
    ) == [64, 3, 32, 32]
    assert validate_fixed_batch_shape(
        "cifar10dvs", (64, 10, 2, 48, 48), minimum_batch_size=64, time_steps=10, in_channels=2
    ) == [64, 10, 2, 48, 48]

    with pytest.raises(PilotV3Error, match="CIFAR-100"):
        validate_fixed_batch_shape(
            "cifar100", (64, 3, 32), minimum_batch_size=64, time_steps=6, in_channels=3
        )
    with pytest.raises(PilotV3Error, match="CIFAR10-DVS"):
        validate_fixed_batch_shape(
            "cifar10dvs", (64, 2, 48, 48), minimum_batch_size=64, time_steps=10, in_channels=2
        )


def test_v3_author_freeze_is_required_before_claiming_a_health_seed(tmp_path: Path) -> None:
    protocol = _unsigned_protocol()
    block = resolve_pilot_block(protocol, "cifar100", repository_root=tmp_path)
    with pytest.raises(PilotV3Error, match="frozen=true"):
        require_v3_author_freeze(protocol, repository_root=tmp_path)
    assert not block.attempt_receipt.exists()


def test_health_gate_main_refuses_unsigned_protocol_before_receipt_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "protocol_v3_unsigned.yaml"
    protocol_path.write_text(
        yaml.safe_dump(_unsigned_protocol(), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(health_gate, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        health_gate,
        "exclusive_create_json",
        lambda *_args, **_kwargs: pytest.fail("unsigned v3 must not claim a seed"),
    )

    result = health_gate.main(
        ["--protocol", str(protocol_path), "--dataset", "cifar100", "--device", "cuda:0"]
    )

    assert result == 2
    assert not (tmp_path / "results" / "pilot" / "v3_talif_only").exists()


def test_health_attempt_receipt_is_exclusive_and_preserves_first_claim(tmp_path: Path) -> None:
    destination = tmp_path / "attempt.json"
    first = {"seed": 474123945, "status": "ATTEMPT_CLAIMED_SEED_CONSUMED_NO_RETRY"}
    exclusive_create_json(destination, first)
    with pytest.raises(FileExistsError):
        exclusive_create_json(destination, {"seed": 999, "status": "replacement"})
    assert json.loads(destination.read_text(encoding="utf-8")) == first


def test_v3_pilot_launch_selection_is_one_dataset_c1_c2_block() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    rows = [
        {
            "experiment": run["experiment"],
            "dataset": run["data"]["dataset"],
            "depth": str(run["model"]["depth"]),
            "time_steps": str(run["model"]["time_steps"]),
            "condition": run["condition"],
            "seed": str(run["seed"]),
        }
        for run in generate_v3_pilot_matrix(protocol)
    ]
    with pytest.raises(ValueError, match="exactly one complete C1/C2 block"):
        matrix_runner._validate_v3_pilot_launch_contract(
            protocol=protocol,
            protocol_path=PROTOCOL_PATH,
            selected_rows=rows[:1],
            dataset="cifar100",
            output_root=None,
            pilot_plan=None,
        )
    with pytest.raises(ValueError, match="differs from the protocol binding"):
        matrix_runner._validate_v3_pilot_launch_contract(
            protocol=protocol,
            protocol_path=PROTOCOL_PATH,
            selected_rows=rows[2:],
            dataset="cifar100",
            output_root=None,
            pilot_plan=None,
        )


def test_v3_matrix_and_trainer_reject_training_dry_runs() -> None:
    with pytest.raises(ValueError, match="v3 matrix dry-runs are disabled"):
        matrix_runner.run_matrix(
            config_dir=None,
            protocol_path=PROTOCOL_PATH,
            dry_run=True,
            device="cpu",
        )
    with pytest.raises(SystemExit, match="v3 training dry-runs are disabled"):
        trainer.main(
            [
                "--config",
                str(ROOT / "configs" / "v3_talif_only_generated" / "missing.yaml"),
                "--protocol",
                str(PROTOCOL_PATH),
                "--dry-run",
                "--device",
                "cpu",
            ]
        )


def test_v3_trainer_revalidates_author_freeze_before_pilot_health(tmp_path: Path, monkeypatch) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    block = resolve_pilot_block(protocol, "cifar100", repository_root=tmp_path)
    block.pilot_plan.parent.mkdir(parents=True)
    payload = {
        "version": 3,
        "artifact_class": "NON_REPORTABLE_V3_PILOT_PLAN",
        "non_reportable": True,
        "dataset": "cifar100",
        "protocol_hash": "test-protocol-hash",
        "protocol_path": artifact_path_reference(PROTOCOL_PATH, tmp_path),
    }
    block.pilot_plan.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    config = SimpleNamespace(runtime=SimpleNamespace(run_id="run"), config_hash="config")
    with pytest.raises(ValueError, match="requires author freeze"):
        trainer._validate_orchestrator_pilot_plan(
            block.pilot_plan,
            config=config,
            config_path=tmp_path / "config.yaml",
            protocol_path=PROTOCOL_PATH,
            protocol_hash="test-protocol-hash",
            output_dir_was_explicit=True,
        )


def test_phase_b_binds_only_a_revalidated_aggregate_pilot_pass(tmp_path: Path) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    acceptance = protocol["pilot_acceptance"]
    validation_path = tmp_path / acceptance["validation_output"]
    validation_path.parent.mkdir(parents=True)
    hashes = {
        "health_report_sha256": "1" * 64,
        "attempt_receipt_sha256": "2" * 64,
        "pilot_plan_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
    }
    report = {
        "schema_version": 1,
        "artifact_class": "NON_REPORTABLE_V3_TALIF_ONLY_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V3_TALIF_ONLY_120_EPOCH_PILOT",
        "exit_code": 0,
        "validated_at": "2026-07-28T12:00:00+08:00",
        "protocol_hash": stable_hash(protocol),
        "acceptance_hash": stable_hash(dict(acceptance)),
        "datasets": {
            dataset: {"status": "PASS", "pass": True, **hashes}
            for dataset in ("cifar100", "cifar10dvs")
        },
        "integrity_failures": [],
        "threshold_failures": [],
    }
    validation_path.write_text(json.dumps(report), encoding="utf-8")
    validator_path = tmp_path / "scripts" / "validate_v3_pilot.py"
    validator_path.parent.mkdir()
    validator_path.write_text(
        "REPORT = " + repr(report) + "\n"
        "def validate_pilot(**_kwargs):\n"
        "    return dict(REPORT)\n",
        encoding="utf-8",
    )

    binding = freeze_manifest._validate_v3_pilot_acceptance(
        protocol, PROTOCOL_PATH, tmp_path
    )

    assert binding["path"] == acceptance["validation_output"]
    assert binding["protocol_hash"] == stable_hash(protocol)
    assert binding["datasets"]["cifar100"] == hashes
    assert binding["datasets"]["cifar10dvs"] == hashes

    report["status"] = "FAIL"
    report["pass"] = False
    validation_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(
        freeze_manifest.FreezeManifestError,
        match="not an exact aggregate PASS",
    ):
        freeze_manifest._validate_v3_pilot_acceptance(
            protocol, PROTOCOL_PATH, tmp_path
        )
