from __future__ import annotations

import copy
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_checkpoints as final_eval  # noqa: E402
from talif_msresnet.config import (  # noqa: E402
    generate_run_matrix,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.train import SEED_METRIC_FIELDS  # noqa: E402
from talif_msresnet.utils import sha256_file  # noqa: E402


def _source() -> dict[str, str]:
    return {
        "test_source": "data/cifar100/cifar-100-python/test",
        "test_source_sha256": "a" * 64,
        "cifar100_source_provenance": "environment/CIFAR100_SOURCE_PROVENANCE.json",
        "cifar100_source_provenance_sha256": "b" * 64,
        "cifar100_archive": "data/cifar100/cifar-100-python.tar.gz",
        "cifar100_archive_sha256": "c" * 64,
        "cifar100_train_pickle": "data/cifar100/cifar-100-python/train",
        "cifar100_train_pickle_sha256": "d" * 64,
        "cifar100_test_pickle": "data/cifar100/cifar-100-python/test",
        "cifar100_test_pickle_sha256": "a" * 64,
        "cifar100_meta_pickle": "data/cifar100/cifar-100-python/meta",
        "cifar100_meta_pickle_sha256": "e" * 64,
        "cifar100_split_manifest": "data/manifests/cifar100_seed2024.json",
        "cifar100_split_manifest_sha256": "f" * 64,
    }


def _binding() -> dict[str, str]:
    return {
        "benchmark_receipt": "results/benchmark/v7_mechanism/benchmark_completion.json",
        "benchmark_receipt_sha256": "d" * 64,
    }


def test_v7_source_binding_requires_the_complete_expanded_provenance() -> None:
    source = _source()
    assert set(source) == final_eval._V7_FINAL_TEST_SOURCE_KEYS
    assert final_eval._require_v7_test_source_binding(source) is source

    incomplete = dict(source)
    incomplete.pop("cifar100_archive_sha256")
    with pytest.raises(RuntimeError, match="cifar100_archive_sha256"):
        final_eval._require_v7_test_source_binding(incomplete)


def _metric(run_id: str, config_hash: str, protocol_hash: str) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in SEED_METRIC_FIELDS}
    row.update(
        {
            "run_id": run_id,
            "experiment": "E9",
            "dataset": "cifar100",
            "depth": 20,
            "time_steps": 6,
            "condition": "M0",
            "topology": "ms_resnet",
            "neuron": "lif",
            "seed": 1,
            "config_hash": config_hash,
            "protocol_hash": protocol_hash,
            "split_manifest_sha256": "c" * 64,
            "shared_weight_sha256": "e" * 64,
            "training_environment_sha256": "f" * 64,
            "status": "complete",
            "failed": False,
        }
    )
    return row


def _write_commit_files(
    root: Path,
    run_id: str,
    *,
    config_hash: str,
    protocol_hash: str,
) -> tuple[Path, dict[str, Any]]:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"v7-checkpoint")
    metric = _metric(run_id, config_hash, protocol_hash)
    (run_dir / "seed_metrics.json").write_text(json.dumps(metric), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": run_id}), encoding="utf-8"
    )
    with (root / "seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_METRIC_FIELDS)
        writer.writeheader()
        writer.writerow(metric)
    return run_dir, metric


def _results_ready_journal(
    run_dir: Path,
    *,
    config_hash: str,
    protocol_hash: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "status": "in_progress",
        "stage": "results_ready",
        "run_id": run_dir.name,
        "config_hash": config_hash,
        "protocol_hash": protocol_hash,
        "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
        "checkpoint_epoch_zero_based": 4,
        "started_at": "2026-08-11T12:00:00+08:00",
        "evaluated_at": "2026-08-11T12:01:00+08:00",
        "device": "cuda:0",
        "test_source": _source(),
        **_binding(),
        "test": {"loss": 1.25, "accuracy": 0.75, "samples": 8},
    }


def test_v7_commit_binds_exact_receipt_to_marker_manifest_and_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")
    raw_config = generate_run_matrix(protocol)[0]
    resolved = validate_run_mapping(raw_config, protocol)
    run_id = resolved.runtime.run_id
    run_dir, _metric_row = _write_commit_files(
        tmp_path,
        run_id,
        config_hash=resolved.config_hash,
        protocol_hash=str(resolved.analysis["protocol_hash"]),
    )
    payload = {
        "config": copy.deepcopy(raw_config),
        "config_hash": resolved.config_hash,
        "epoch": 4,
        "model_state": {},
    }
    events: list[str] = []
    journal_snapshots: list[dict[str, Any]] = []
    original_atomic_write_json = final_eval.atomic_write_json

    def receipt_validator() -> dict[str, str]:
        events.append("receipt")
        return dict(_binding())

    def spy_atomic_write_json(path: Path, value: Any) -> None:
        if Path(path).name == final_eval.JOURNAL_NAME:
            journal_snapshots.append(copy.deepcopy(dict(value)))
        original_atomic_write_json(path, value)

    class FakeModel:
        def to(self, _device: Any) -> "FakeModel":
            return self

        def load_state_dict(self, _state: Any, *, strict: bool) -> None:
            assert strict is True

    monkeypatch.setattr(final_eval, "load_checkpoint", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(final_eval, "resolve_device", lambda _name: torch.device("cpu"))
    monkeypatch.setattr(final_eval, "build_model", lambda _config: FakeModel())
    monkeypatch.setattr(final_eval, "_test_source_record", lambda *_args: dict(_source()))
    monkeypatch.setattr(final_eval, "atomic_write_json", spy_atomic_write_json)

    def fake_loader(*_args: Any, **_kwargs: Any) -> object:
        events.append("loader")
        return object()

    monkeypatch.setattr(final_eval, "build_test_loader", fake_loader)
    monkeypatch.setattr(
        final_eval,
        "evaluate",
        lambda *_args, **_kwargs: {"loss": 1.25, "accuracy": 0.75, "samples": 8},
    )

    status = final_eval._evaluate_one(
        run_dir,
        "cpu",
        tmp_path,
        protocol,
        v7_test_source=_source(),
        v7_receipt_validator=receipt_validator,
    )

    assert status == "evaluated"
    assert events == ["receipt", "loader", "receipt"]
    assert journal_snapshots
    assert all(
        {key: snapshot[key] for key in final_eval.V7_BENCHMARK_BINDING_KEYS}
        == _binding()
        for snapshot in journal_snapshots
    )
    marker = json.loads((run_dir / "final_test.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert {key: marker[key] for key in final_eval.V7_BENCHMARK_BINDING_KEYS} == _binding()
    assert {
        key: manifest["final_test"][key]
        for key in final_eval.V7_BENCHMARK_BINDING_KEYS
    } == _binding()


def test_v7_receipt_failure_blocks_before_test_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loader_called = False

    def loader(*_args: Any, **_kwargs: Any) -> object:
        nonlocal loader_called
        loader_called = True
        return object()

    monkeypatch.setattr(final_eval, "build_test_loader", loader)

    with pytest.raises(RuntimeError, match="receipt tampered"):
        final_eval._evaluate_one(
            tmp_path / "not-read",
            "cuda:0",
            tmp_path,
            {"protocol_version": 7},
            v7_test_source=_source(),
            v7_receipt_validator=lambda: (_ for _ in ()).throw(
                RuntimeError("receipt tampered")
            ),
        )

    assert loader_called is False


def test_v7_receipt_tampering_immediately_before_commit_writes_nothing(
    tmp_path: Path,
) -> None:
    run_dir, metric = _write_commit_files(
        tmp_path,
        "run",
        config_hash="config-hash",
        protocol_hash="protocol-hash",
    )
    journal = _results_ready_journal(
        run_dir,
        config_hash="config-hash",
        protocol_hash="protocol-hash",
    )

    with pytest.raises(RuntimeError, match="receipt tampered"):
        final_eval._commit_final_test(
            run_dir,
            tmp_path,
            journal,
            v7_receipt_validator=lambda: (_ for _ in ()).throw(
                RuntimeError("receipt tampered")
            ),
        )

    assert not (run_dir / "final_test.json").exists()
    assert json.loads((run_dir / "seed_metrics.json").read_text(encoding="utf-8")) == metric


def test_v7_recovery_revalidates_receipt_again_at_commit_boundary(
    tmp_path: Path,
) -> None:
    run_dir, _metric_row = _write_commit_files(
        tmp_path,
        "run",
        config_hash="config-hash",
        protocol_hash="protocol-hash",
    )
    journal = _results_ready_journal(
        run_dir,
        config_hash="config-hash",
        protocol_hash="protocol-hash",
    )
    (run_dir / final_eval.JOURNAL_NAME).write_text(json.dumps(journal), encoding="utf-8")
    (run_dir / final_eval.LOCK_NAME).write_text("locked\n", encoding="utf-8")
    calls = 0

    def validator() -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return dict(_binding())
        raise RuntimeError("receipt changed during recovery")

    with pytest.raises(RuntimeError, match="changed during recovery"):
        final_eval._recover_one(
            run_dir,
            tmp_path,
            "protocol-hash",
            v7_test_source=_source(),
            v7_receipt_validator=validator,
        )

    assert calls == 2
    assert (run_dir / final_eval.JOURNAL_NAME).exists()
    assert (run_dir / final_eval.LOCK_NAME).exists()
    assert not (run_dir / "final_test.json").exists()


def test_v7_recovery_rejects_journal_receipt_mismatch_without_cleanup(
    tmp_path: Path,
) -> None:
    run_dir, _metric_row = _write_commit_files(
        tmp_path,
        "run",
        config_hash="config-hash",
        protocol_hash="protocol-hash",
    )
    journal = _results_ready_journal(
        run_dir,
        config_hash="config-hash",
        protocol_hash="protocol-hash",
    )
    journal["benchmark_receipt_sha256"] = "0" * 64
    (run_dir / final_eval.JOURNAL_NAME).write_text(json.dumps(journal), encoding="utf-8")
    (run_dir / final_eval.LOCK_NAME).write_text("locked\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="journal benchmark binding"):
        final_eval._recover_one(
            run_dir,
            tmp_path,
            "protocol-hash",
            v7_test_source=_source(),
            v7_receipt_validator=lambda: dict(_binding()),
        )

    assert (run_dir / final_eval.JOURNAL_NAME).exists()
    assert (run_dir / final_eval.LOCK_NAME).exists()
