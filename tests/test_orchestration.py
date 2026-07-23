from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_checkpoints as final_eval  # noqa: E402
import run_matrix as matrix_runner  # noqa: E402
import talif_msresnet.train as trainer  # noqa: E402
from talif_msresnet.train import SEED_METRIC_FIELDS, validate_resume_checkpoint_path  # noqa: E402
from talif_msresnet.utils import sha256_file  # noqa: E402


def test_resume_accepts_only_last_checkpoint() -> None:
    assert validate_resume_checkpoint_path("some/run/last.pt").name == "last.pt"
    with pytest.raises(ValueError, match="only supported from last.pt"):
        validate_resume_checkpoint_path("some/run/best.pt")
    with pytest.raises(ValueError, match="only supported from last.pt"):
        validate_resume_checkpoint_path("some/run/failed.pt")


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
        selected_rows=rows[:1],
        config_dir=config_dir,
        protocol_path=ROOT / "configs" / "protocol.yaml",
        protocol_hash="protocol-hash",
        output_root=pilot_root,
        formal_root="results/runs",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["protocol_path"] == "configs/protocol.yaml"
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
        protocol_path=ROOT / "configs" / "protocol.yaml",
        protocol_hash="protocol-hash",
        output_dir_was_explicit=True,
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
