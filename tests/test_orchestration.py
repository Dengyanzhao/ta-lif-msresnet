from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

V2R1_PROTOCOL = ROOT / "configs" / "protocol_v2_pilot.yaml"
V2R1_CONFIG_DIR = ROOT / "configs" / "v2_pilot_generated"
V2R2_PROTOCOL = ROOT / "configs" / "protocol_v2r2_seed88_of80_e120.yaml"
V2R2_CONFIG_DIR = ROOT / "configs" / "v2r2_seed88_of80_e120_generated"

import evaluate_checkpoints as final_eval  # noqa: E402
import run_matrix as matrix_runner  # noqa: E402
import talif_msresnet.train as trainer  # noqa: E402
from talif_msresnet.train import (  # noqa: E402
    SEED_METRIC_FIELDS,
    _validate_resume_run_environment,
    validate_resume_checkpoint_path,
)
from talif_msresnet.config import load_protocol  # noqa: E402
from talif_msresnet.utils import sha256_file  # noqa: E402


def test_resume_accepts_only_last_checkpoint() -> None:
    assert validate_resume_checkpoint_path("some/run/last.pt").name == "last.pt"
    with pytest.raises(ValueError, match="only supported from last.pt"):
        validate_resume_checkpoint_path("some/run/best.pt")
    with pytest.raises(ValueError, match="only supported from last.pt"):
        validate_resume_checkpoint_path("some/run/failed.pt")


def test_resume_rejects_a_different_training_environment(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run",
                "config_hash": "config-hash",
                "environment": {
                    "training_environment_sha256": "same-environment",
                }
            }
        ),
        encoding="utf-8",
    )
    expected_identity = {
        "expected_run_id": "run",
        "expected_config_hash": "config-hash",
    }

    previous = _validate_resume_run_environment(
        run_dir, checkpoint, "same-environment", **expected_identity
    )
    assert previous["environment"]["training_environment_sha256"] == "same-environment"
    with pytest.raises(RuntimeError, match="Cross-environment"):
        _validate_resume_run_environment(
            run_dir, checkpoint, "other-environment", **expected_identity
        )
    foreign = tmp_path / "foreign" / "last.pt"
    foreign.parent.mkdir()
    foreign.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="current run directory"):
        _validate_resume_run_environment(
            run_dir, foreign, "same-environment", **expected_identity
        )


def test_unfrozen_pilot_requires_a_separate_output_root() -> None:
    with pytest.raises(ValueError, match="explicit separate"):
        matrix_runner._validate_pilot_output_root(None, "results/runs")
    with pytest.raises(ValueError, match="outside the formal results root"):
        matrix_runner._validate_pilot_output_root("results/runs", "results/runs")
    with pytest.raises(ValueError, match="outside the formal results root"):
        matrix_runner._validate_pilot_output_root("results/runs/pilot", "results/runs")
    with pytest.raises(ValueError, match="outside the formal results root"):
        matrix_runner._validate_pilot_output_root("results", "results/runs")
    assert matrix_runner._validate_pilot_output_root(
        "results/pilot", "results/runs"
    ) == (ROOT / "results" / "pilot").resolve()


def test_pilot_plan_uses_portable_paths_and_trainer_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unfrozen_protocol_path: Path,
) -> None:
    config_dir = tmp_path / "external-configs"
    config_dir.mkdir()
    rows: list[dict[str, str]] = []
    for condition in ("C1", "C2", "C3", "C4"):
        config_file = config_dir / f"{condition}.yaml"
        config_file.write_text(f"condition: {condition}\n", encoding="utf-8")
        rows.append(
            {
                "experiment": "E1",
                "dataset": "cifar10",
                "depth": "20",
                "time_steps": "6",
                "seed": "11",
                "condition": condition,
                "run_id": f"run-{condition}",
                "config_file": config_file.name,
            }
        )

    monkeypatch.setattr(
        matrix_runner,
        "load_run_config",
        lambda path, protocol: SimpleNamespace(
            runtime=SimpleNamespace(run_id=f"run-{Path(path).stem}"),
            config_hash=f"hash-{Path(path).stem}",
        ),
    )
    monkeypatch.setattr(matrix_runner, "PILOT_PLAN_PATH", tmp_path / "plan.json")
    with pytest.raises(ValueError, match="complete derived C1-C4"):
        matrix_runner._prepare_unfrozen_pilot_plan(
            all_rows=rows,
            selected_rows=rows[:1],
            config_dir=config_dir,
            protocol_path=unfrozen_protocol_path,
            protocol_hash="protocol-hash",
            output_root=tmp_path / "pilot",
            formal_root="results/runs",
        )

    def fake_load(path: Path, _protocol: Path) -> SimpleNamespace:
        condition = Path(path).stem
        return SimpleNamespace(
            runtime=SimpleNamespace(run_id=f"run-{condition}"),
            config_hash=f"hash-{condition}",
        )

    plan_path = tmp_path / "unfrozen_pilot_plan.json"
    monkeypatch.setattr(matrix_runner, "load_run_config", fake_load)
    monkeypatch.setattr(matrix_runner, "PILOT_PLAN_PATH", plan_path)
    monkeypatch.setattr(trainer, "PILOT_PLAN_PATH", plan_path)
    pilot_root = tmp_path / "external-pilot"
    matrix_runner._prepare_unfrozen_pilot_plan(
        all_rows=rows,
        selected_rows=rows,
        config_dir=config_dir,
        protocol_path=unfrozen_protocol_path,
        protocol_hash="protocol-hash",
        output_root=pilot_root,
        formal_root="results/runs",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["protocol_path"].startswith("external:file-sha256:")
    assert payload["formal_output_root"] == "results/runs"
    assert payload["pilot_output_root"].startswith("external:path-sha256:")
    assert all(
        run["config_file"].startswith("external:file-sha256:")
        for run in payload["runs"]
    )
    assert str(tmp_path) not in json.dumps(payload)

    first_config = config_dir / "C1.yaml"
    config = SimpleNamespace(
        runtime=SimpleNamespace(run_id="run-C1", output_dir=str(pilot_root)),
        config_hash="hash-C1",
    )
    trainer._validate_orchestrator_pilot_plan(
        plan_path,
        config=config,
        config_path=first_config,
        protocol_path=unfrozen_protocol_path,
        protocol_hash="protocol-hash",
        output_dir_was_explicit=True,
    )

    protocol = yaml.safe_load(unfrozen_protocol_path.read_text(encoding="utf-8"))
    protocol["protocol_status"]["frozen"] = True
    monkeypatch.setattr(trainer, "load_protocol", lambda _path: protocol)
    with pytest.raises(ValueError, match="frozen protocol cannot use"):
        trainer._validate_orchestrator_pilot_plan(
            plan_path,
            config=config,
            config_path=first_config,
            protocol_path=unfrozen_protocol_path,
            protocol_hash="protocol-hash",
            output_dir_was_explicit=True,
        )


def test_pilot_plan_rejects_partial_condition_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unfrozen_protocol_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    rows: list[dict[str, str]] = []
    for condition in ("C1", "C2", "C3", "C4"):
        path = config_dir / f"{condition}.yaml"
        path.write_text(condition, encoding="utf-8")
        rows.append(
            {
                "experiment": "E1",
                "dataset": "cifar100",
                "depth": "20",
                "time_steps": "6",
                "seed": "77",
                "condition": condition,
                "run_id": f"run-{condition}",
                "config_file": path.name,
            }
        )


def test_v2_pilot_launch_contract_uses_only_protocol_bound_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = V2R1_PROTOCOL
    protocol = load_protocol(protocol_path)
    with (V2R1_CONFIG_DIR / "run_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    health_report = {
        "status": "PASS",
        "acceptance_hash": "acceptance-hash",
        "git_commit": "a" * 40,
    }
    monkeypatch.setattr(
        matrix_runner,
        "validate_pilot_health_report",
        lambda *args, **kwargs: health_report,
    )

    output, plan, health, observed = matrix_runner._validate_v2_pilot_launch_contract(
        protocol=protocol,
        protocol_path=protocol_path,
        selected_rows=rows,
        output_root=None,
        pilot_plan=None,
    )

    acceptance = protocol["pilot_acceptance"]
    assert output == (ROOT / acceptance["pilot_output_root"]).resolve()
    assert plan == (ROOT / acceptance["pilot_plan"]).resolve()
    assert health == (ROOT / acceptance["health_output"]).resolve()
    assert observed is health_report
    matrix_manifest = json.loads(
        (V2R1_CONFIG_DIR / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    matrix_runner._validate_v2_generated_matrix(
        protocol=protocol,
        protocol_path=protocol_path,
        config_dir=V2R1_CONFIG_DIR,
        manifest_rows=rows,
        matrix_manifest=matrix_manifest,
    )

    tampered_dir = tmp_path / "v2_pilot_generated"
    shutil.copytree(V2R1_CONFIG_DIR, tampered_dir)
    tampered_path = tampered_dir / rows[0]["config_file"]
    tampered = yaml.safe_load(tampered_path.read_text(encoding="utf-8"))
    tampered["runtime"]["log_every"] = 99
    tampered_path.write_text(
        yaml.safe_dump(tampered, sort_keys=False), encoding="utf-8"
    )
    tampered_rows = [dict(row) for row in rows]
    tampered_rows[0]["config_file_sha256"] = sha256_file(tampered_path)
    with pytest.raises(RuntimeError, match="in-memory protocol matrix"):
        matrix_runner._validate_v2_generated_matrix(
            protocol=protocol,
            protocol_path=protocol_path,
            config_dir=tampered_dir,
            manifest_rows=tampered_rows,
            matrix_manifest=matrix_manifest,
        )

    with pytest.raises(ValueError, match="complete four-run"):
        matrix_runner._validate_v2_pilot_launch_contract(
            protocol=protocol,
            protocol_path=protocol_path,
            selected_rows=rows[:3],
            output_root=None,
            pilot_plan=None,
        )
    with pytest.raises(ValueError, match="cannot override"):
        matrix_runner._validate_v2_pilot_launch_contract(
            protocol=protocol,
            protocol_path=protocol_path,
            selected_rows=rows,
            output_root="results/pilot/changed",
            pilot_plan=None,
        )


def test_v2r2_launch_contract_isolated_from_seed77_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol(V2R2_PROTOCOL)
    acceptance = protocol["pilot_acceptance"]
    legacy_acceptance = load_protocol(V2R1_PROTOCOL)["pilot_acceptance"]
    with (V2R2_CONFIG_DIR / "run_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))

    observed_health_calls: list[tuple[Path, Path]] = []
    health_report = {
        "status": "PASS",
        "acceptance_hash": "v2r2-acceptance-hash",
        "git_commit": "b" * 40,
    }

    def validate_bound_health(
        health_path: Path,
        protocol_path: Path,
        **_kwargs: object,
    ) -> dict[str, object]:
        observed_health_calls.append((health_path.resolve(), protocol_path.resolve()))
        return health_report

    monkeypatch.setattr(
        matrix_runner,
        "validate_pilot_health_report",
        validate_bound_health,
    )

    output, plan, health, observed = matrix_runner._validate_v2_pilot_launch_contract(
        protocol=protocol,
        protocol_path=V2R2_PROTOCOL,
        selected_rows=rows,
        output_root=None,
        pilot_plan=None,
    )

    assert [int(row["seed"]) for row in rows] == [88, 88, 88, 88]
    assert acceptance["overfit"]["steps"] == 80
    assert output == (ROOT / acceptance["pilot_output_root"]).resolve()
    assert plan == (ROOT / acceptance["pilot_plan"]).resolve()
    assert health == (ROOT / acceptance["health_output"]).resolve()
    assert observed is health_report
    assert observed_health_calls == [(health, V2R2_PROTOCOL.resolve())]
    assert health != (ROOT / legacy_acceptance["health_output"]).resolve()

    matrix_manifest = json.loads(
        (V2R2_CONFIG_DIR / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    matrix_runner._validate_v2_generated_matrix(
        protocol=protocol,
        protocol_path=V2R2_PROTOCOL,
        config_dir=V2R2_CONFIG_DIR,
        manifest_rows=rows,
        matrix_manifest=matrix_manifest,
    )

    with pytest.raises(ValueError, match="output-root cannot override"):
        matrix_runner._validate_v2_pilot_launch_contract(
            protocol=protocol,
            protocol_path=V2R2_PROTOCOL,
            selected_rows=rows,
            output_root=legacy_acceptance["pilot_output_root"],
            pilot_plan=None,
        )
    with pytest.raises(ValueError, match="pilot-plan cannot override"):
        matrix_runner._validate_v2_pilot_launch_contract(
            protocol=protocol,
            protocol_path=V2R2_PROTOCOL,
            selected_rows=rows,
            output_root=None,
            pilot_plan=legacy_acceptance["pilot_plan"],
        )

    with (V2R1_CONFIG_DIR / "run_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        legacy_rows = list(csv.DictReader(handle))
    legacy_manifest = json.loads(
        (V2R1_CONFIG_DIR / "matrix_manifest.json").read_text(encoding="utf-8")
    )
    with pytest.raises(RuntimeError, match="differs from the protocol"):
        matrix_runner._validate_v2_generated_matrix(
            protocol=protocol,
            protocol_path=V2R2_PROTOCOL,
            config_dir=V2R1_CONFIG_DIR,
            manifest_rows=legacy_rows,
            matrix_manifest=legacy_manifest,
        )


def test_matrix_continuation_preserves_attempts_and_skips_complete(tmp_path: Path) -> None:
    assert matrix_runner._project_path("results/runs").is_absolute()
    matrix_runner._validate_run_id("E1_cifar10_C1_s11", "E1_cifar10_C1_s11")
    with pytest.raises(RuntimeError, match="do not rename"):
        matrix_runner._validate_run_id("renamed", "E1_cifar10_C1_s11")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert matrix_runner._continuation_action(run_dir, "run", "hash", False) == ("fresh", None)
    (run_dir / "attempt_001.stdout.log").write_text("failed\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--resume-matrix"):
        matrix_runner._continuation_action(run_dir, "run", "hash", False)
    attempt, stdout, stderr = matrix_runner._next_attempt(run_dir)
    assert attempt == 2
    assert stdout.name == "attempt_002.stdout.log"
    assert stderr.name == "attempt_002.stderr.log"

    metrics = {"run_id": "run", "config_hash": "hash", "status": "failed", "failed": 1}
    (run_dir / "seed_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "last.pt").write_bytes(b"checkpoint")
    action, checkpoint = matrix_runner._continuation_action(run_dir, "run", "hash", True)
    assert action == "resume"
    assert checkpoint == (run_dir / "last.pt").resolve()

    metrics.update({"status": "complete", "failed": 0})
    (run_dir / "seed_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"best")
    assert matrix_runner._continuation_action(run_dir, "run", "hash", True) == ("skip", None)


def _metric_row(run_id: str, index: int) -> dict[str, object]:
    row: dict[str, object] = {field: "" for field in SEED_METRIC_FIELDS}
    condition = f"C{index % 4 + 1}"
    row.update({
        "run_id": run_id,
        "experiment": "E1",
        "dataset": "cifar100" if index < 20 else "cifar10dvs",
        "depth": 20,
        "time_steps": 6 if index < 20 else 10,
        "condition": condition,
        "topology": "spiking_resnet" if condition in {"C1", "C2"} else "ms_resnet",
        "neuron": "lif" if condition in {"C1", "C3"} else "ta_lif",
        "seed": index % 5,
        "config_hash": f"hash-{index}",
        "protocol_hash": "protocol-hash",
        "split_manifest_sha256": f"split-{index}",
        "shared_weight_sha256": f"weights-{index}",
        "status": "complete",
        "failed": 0,
    })
    return row


def test_final_test_requires_exact_consolidated_plan(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    metrics: list[dict[str, object]] = []
    for index in range(40):
        run_id = f"run-{index:03d}"
        metric = _metric_row(run_id, index)
        metrics.append(metric)
        rows.append({key: str(metric[key]) for key in (
            "run_id", "experiment", "dataset", "depth", "time_steps", "condition", "topology",
            "neuron", "seed", "config_hash", "protocol_hash",
        )})
        run_dir = tmp_path / run_id
        run_dir.mkdir()
        (run_dir / "seed_metrics.json").write_text(json.dumps(metric), encoding="utf-8")
        (run_dir / "run_manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "config_hash": metric["config_hash"],
            "split_manifest_sha256": metric["split_manifest_sha256"],
            "shared_weight_sha256": metric["shared_weight_sha256"],
            "config": {"analysis": {"protocol_hash": metric["protocol_hash"]}},
        }), encoding="utf-8")
    with (tmp_path / "seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    audited = final_eval._audit_consolidated_metrics(rows, tmp_path)
    assert len(audited) == 40

    metrics[-1]["config_hash"] = "wrong"
    with (tmp_path / "seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    with pytest.raises(RuntimeError, match="config hashes differ"):
        final_eval._audit_consolidated_metrics(rows, tmp_path)

    metrics[-1]["config_hash"] = rows[-1]["config_hash"]
    metrics[-1]["protocol_hash"] = "wrong-protocol"
    with (tmp_path / "seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    with pytest.raises(RuntimeError, match="protocol_hash"):
        final_eval._audit_consolidated_metrics(rows, tmp_path)

    metrics[-1]["protocol_hash"] = rows[-1]["protocol_hash"]
    metrics[-1]["test_accuracy"] = 0.5
    with (tmp_path / "seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metrics)
    with pytest.raises(RuntimeError, match="test_accuracy"):
        final_eval._audit_consolidated_metrics(rows, tmp_path)


def test_results_ready_journal_recovers_without_test_loader(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"frozen checkpoint bytes")
    metric = _metric_row("run", 0)
    metric["config_hash"] = "config-hash"
    (run_dir / "seed_metrics.json").write_text(json.dumps(metric), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": "run"}), encoding="utf-8")
    with (tmp_path / "seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_METRIC_FIELDS)
        writer.writeheader()
        writer.writerow(metric)
    journal = {
        "format_version": 1,
        "status": "in_progress",
        "stage": "results_ready",
        "run_id": "run",
        "config_hash": "config-hash",
        "protocol_hash": "protocol-hash",
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch_zero_based": 4,
        "evaluated_at": "2026-07-19T12:00:00+00:00",
        "device": "cpu",
        "test": {"loss": 1.25, "accuracy": 0.75, "samples": 8},
        "test_source": {"test_source": "tiny", "test_source_sha256": "tiny-hash"},
    }
    (run_dir / final_eval.JOURNAL_NAME).write_text(json.dumps(journal), encoding="utf-8")
    (run_dir / final_eval.LOCK_NAME).write_text("stale lock\n", encoding="utf-8")

    status = final_eval._recover_one(run_dir, tmp_path, "protocol-hash")
    assert status == "recovered_without_test_access"
    assert (run_dir / "final_test.json").exists()
    marker = json.loads((run_dir / "final_test.json").read_text(encoding="utf-8"))
    assert marker["checkpoint"] == (
        f"external:checkpoint-sha256:{sha256_file(checkpoint)}"
    )
    assert str(tmp_path) not in marker["checkpoint"]
    assert not (run_dir / final_eval.JOURNAL_NAME).exists()
    assert not (run_dir / final_eval.LOCK_NAME).exists()


def test_ambiguous_test_access_is_not_retried(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    journal = {
        "run_id": "run",
        "stage": "test_access_started",
        "protocol_hash": "protocol-hash",
        "checkpoint_sha256": sha256_file(checkpoint),
    }
    (run_dir / final_eval.JOURNAL_NAME).write_text(json.dumps(journal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="test access is ambiguous"):
        final_eval._recover_one(run_dir, tmp_path, "protocol-hash")


def test_dvs_test_source_uses_index_hash_for_external_directory(tmp_path: Path) -> None:
    source = tmp_path / "external-dvs"
    source.mkdir()
    index = source / "index.csv"
    index.write_text("sample_id\n1\n", encoding="utf-8")
    config = SimpleNamespace(
        data=SimpleNamespace(dataset="cifar10dvs", test_frames_path=str(source))
    )

    record = final_eval._test_source_record(config)

    assert record["test_source"] == (
        f"external:index-sha256:{sha256_file(index)}"
    )
    assert str(tmp_path) not in record["test_source"]
