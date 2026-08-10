from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import v6_monitor_cn as monitor

from talif_msresnet.config import load_protocol


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _protocol() -> dict[str, object]:
    return load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")


def test_formal_monitor_estimates_from_running_epochs_and_completed_runs(
    tmp_path: Path, monkeypatch
) -> None:
    protocol = _protocol()
    monkeypatch.setattr(monitor, "PROJECT_ROOT", tmp_path)
    now = datetime.fromtimestamp(1_000, tz=timezone.utc)
    root = tmp_path / "results" / "formal_v6_mechanism"
    first = "E6_cifar100_d20_t6_M0_s1587277406"
    second = "E6_cifar100_d20_t6_M1_s1587277406"
    _write_jsonl(
        root / "matrix_events.jsonl",
        [
            {"event": "run_started", "run_id": first, "timestamp": 900.0},
            {
                "event": "run_finished",
                "run_id": second,
                "duration_s": 400.0,
                "timestamp": 950.0,
            },
        ],
    )
    epoch_records = [
        {
            "event": "epoch_completed",
            "timestamp": f"2026-08-11T00:{minute:02d}:00+00:00",
            "train": {"seconds": 10.0},
        }
        for minute in range(10)
    ]
    _write_jsonl(root / first / "events.jsonl", epoch_records)
    (root / second).mkdir(parents=True, exist_ok=True)
    (root / second / "seed_metrics.json").write_text(
        json.dumps({"status": "complete", "duration_s": 400.0}), encoding="utf-8"
    )

    snapshot = monitor.collect_snapshot(protocol, stage="formal", now=now, include_gpu=False)

    assert snapshot.runs[0].state == "running"
    assert snapshot.runs[0].epoch_completed == 10
    assert snapshot.runs[0].expected_total_seconds == 1200.0
    assert snapshot.runs[1].state == "completed"
    assert snapshot.runs[1].expected_total_seconds == 400.0
    assert snapshot.overall_remaining_seconds is not None
    assert snapshot.overall_finish_time is not None


def test_failed_run_withholds_total_eta(tmp_path: Path, monkeypatch) -> None:
    protocol = _protocol()
    monkeypatch.setattr(monitor, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "results" / "formal_v6_mechanism"
    run_id = "E6_cifar100_d20_t6_M0_s1587277406"
    _write_jsonl(
        root / "matrix_events.jsonl",
        [{"event": "run_blocked", "run_id": run_id, "message": "fixture", "timestamp": 1.0}],
    )

    snapshot = monitor.collect_snapshot(
        protocol,
        stage="formal",
        now=datetime.fromtimestamp(10, tz=timezone.utc),
        include_gpu=False,
    )

    assert snapshot.runs[0].state == "failed"
    assert snapshot.overall_remaining_seconds is None
    assert snapshot.overall_finish_time is None
