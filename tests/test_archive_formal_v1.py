from __future__ import annotations

import hashlib
import importlib.util
import json
import signal
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "archive_formal_v1", ROOT / "scripts" / "archive_formal_v1.py"
)
assert SPEC is not None and SPEC.loader is not None
tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tool
SPEC.loader.exec_module(tool)


def _record(pid: int, *, scope: str, kind: str = "trainer") -> object:
    return tool.ProcessRecord(
        pid=pid,
        ppid=1,
        elapsed_seconds=10,
        kind=kind,
        scope=scope,
        match_reason="test",
        cwd="/project",
        declared_output="results/runs",
        resolved_output="/project/results/runs",
        command="python -m talif_msresnet.train",
    )


def test_process_discovery_distinguishes_formal_pilot_and_ambiguous(tmp_path: Path) -> None:
    project = tmp_path / "project"
    formal = project / "results" / "runs"
    cwd_by_pid = {
        10: project,
        11: project,
        12: project,
    }
    ps = "\n".join(
        [
            "10 1 100 python scripts/run_matrix.py --output-root results/runs --device cuda:0",
            (
                "11 10 90 python -m talif_msresnet.train --output-dir "
                f"{formal.as_posix()} --device cuda:0"
            ),
            "12 1 80 python scripts/run_matrix.py --output-root results/pilot --device cuda:0",
            "13 1 70 python scripts/run_matrix.py --output-root results/runs --device cuda:0",
            "14 1 60 python unrelated.py --output-root results/runs",
        ]
    )
    records = tool.parse_process_table(
        ps,
        project_root=project,
        results_root=formal,
        cwd_resolver=lambda pid: cwd_by_pid.get(pid),
    )
    assert [(record.pid, record.scope) for record in records] == [
        (10, "formal_v1"),
        (11, "formal_v1"),
        (12, "other"),
        (13, "ambiguous"),
    ]


def test_stop_requires_confirmation_and_never_signals_other_scope() -> None:
    records = [
        _record(10, scope="formal_v1", kind="trainer"),
        _record(11, scope="other", kind="matrix"),
        _record(12, scope="formal_v1", kind="matrix"),
    ]
    calls: list[tuple[int, int]] = []
    with pytest.raises(tool.IncidentArchiveError, match="--confirm-stop"):
        tool.signal_formal_processes(records, confirmation=None, signaler=calls.append)
    assert calls == []

    def signaler(pid: int, selected_signal: int) -> None:
        calls.append((pid, selected_signal))

    signaled = tool.signal_formal_processes(
        records,
        confirmation=tool.STOP_CONFIRMATION,
        signaler=signaler,
    )
    assert signaled == [12, 10]
    assert calls == [(12, signal.SIGTERM), (10, signal.SIGTERM)]


def test_ambiguous_process_blocks_all_stop_signals() -> None:
    calls: list[tuple[int, int]] = []
    records = [_record(10, scope="formal_v1"), _record(11, scope="ambiguous")]
    with pytest.raises(tool.IncidentArchiveError, match="manual review"):
        tool.signal_formal_processes(
            records,
            confirmation=tool.STOP_CONFIRMATION,
            signaler=lambda pid, selected_signal: calls.append((pid, selected_signal)),
        )
    assert calls == []


def test_stop_receipt_uses_exclusive_create(tmp_path: Path) -> None:
    receipt = tmp_path / "stop.json"
    tool._exclusive_json(receipt, {"first": True})
    with pytest.raises(tool.IncidentArchiveError, match="overwrite"):
        tool._exclusive_json(receipt, {"second": True})
    assert json.loads(receipt.read_text()) == {"first": True}


def _project(tmp_path: Path) -> tuple[Path, str, str, str]:
    project = tmp_path / "project"
    run = project / "results" / "runs" / "E1_cifar100_d20_t6_C1_s11"
    generated = project / "configs" / "generated"
    environment = project / "environment"
    scripts = project / "scripts"
    for directory in (run, generated, environment, scripts):
        directory.mkdir(parents=True, exist_ok=True)

    commit = "a" * 40
    protocol_hash = "b" * 64
    protocol_path = project / "configs" / "protocol.yaml"
    signoff_path = project / "PREREGISTRATION_SIGNOFF.md"
    config_path = generated / "example.yaml"
    run_manifest_path = generated / "run_manifest.csv"
    matrix_manifest_path = generated / "matrix_manifest.json"
    protocol_path.write_text("protocol_version: v1\n")
    signoff_path.write_text("signed\n")
    config_path.write_text("run_id: example\n")
    run_manifest_path.write_text("run_id\nexample\n")
    matrix_manifest = {
        "protocol_hash": protocol_hash,
        "run_count": 1,
        "runs": [
            {
                "run_id": "example",
                "protocol_hash": protocol_hash,
                "config_file": config_path.name,
                "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            }
        ],
    }
    matrix_manifest_path.write_text(json.dumps(matrix_manifest, sort_keys=True) + "\n")
    manifest = {
        "schema": "ta-lif-msresnet-freeze-manifest-v1",
        "repository": {"freeze_commit": commit},
        "protocol": {
            "path": "configs/protocol.yaml",
            "file_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            "canonical_sha256": protocol_hash,
        },
        "signed_record": {
            "path": "PREREGISTRATION_SIGNOFF.md",
            "file_sha256": hashlib.sha256(signoff_path.read_bytes()).hexdigest(),
        },
        "generated_matrix": {
            "directory": "configs/generated",
            "run_count": 1,
            "matrix_manifest": {
                "path": matrix_manifest_path.name,
                "sha256": hashlib.sha256(matrix_manifest_path.read_bytes()).hexdigest(),
            },
            "run_manifest": {
                "path": run_manifest_path.name,
                "sha256": hashlib.sha256(run_manifest_path.read_bytes()).hexdigest(),
            },
        },
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    (project / "FREEZE_MANIFEST.json").write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    (project / "FORMAL_V1_INCIDENT.md").write_text("incident\n")
    (scripts / "archive_formal_v1.py").write_text("tool\n")
    (environment / "pip-freeze.txt").write_text("torch==test\n")
    (run / "events.jsonl").write_text('{"epoch": 1}\n{"epoch": 2}')
    (run / "attempt_001.stdout.log").write_text("started\ncomplete\n")
    metrics = {
        "run_id": run.name,
        "protocol_hash": protocol_hash,
        "status": "complete",
    }
    (run / "seed_metrics.json").write_text(json.dumps(metrics))
    (run / "last.pt").write_bytes(b"checkpoint")
    (run / "best.pt").write_bytes(b"best")
    return project, commit, protocol_hash, manifest_sha256


def _idle_snapshot() -> dict[str, object]:
    return {"captured_at": "2026-07-27T00:00:00+08:00", "discovery_error": None, "processes": []}


def test_archive_is_exclusive_complete_and_non_destructive(tmp_path: Path) -> None:
    project, commit, protocol_hash, manifest_sha256 = _project(tmp_path)
    output = tmp_path / "formal-v1.tar.gz"
    before = (project / "results" / "runs" / "E1_cifar100_d20_t6_C1_s11" / "last.pt").read_bytes()
    result = tool.create_archive(
        project_root=project,
        results_root=project / "results" / "runs",
        output=output,
        process_snapshot=_idle_snapshot(),
        expected_commit=commit,
        expected_protocol_hash=protocol_hash,
        expected_manifest_sha256=manifest_sha256,
        expected_run_count=1,
    )
    assert output.is_file()
    assert result["run_count"] == 1
    assert result["log_count"] == 2
    preserved_checkpoint = project / "results" / "runs" / "E1_cifar100_d20_t6_C1_s11" / "last.pt"
    assert preserved_checkpoint.read_bytes() == before

    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
        assert "formal_v1_snapshot/results/runs/E1_cifar100_d20_t6_C1_s11/last.pt" in names
        assert "formal_v1_snapshot/_audit/inventory.json" in names
        assert "formal_v1_snapshot/_audit/processes.json" in names
        assert "formal_v1_snapshot/_audit/log_metadata.json" in names
        assert "formal_v1_snapshot/_audit/SHA256SUMS" in names
        log_member = archive.extractfile("formal_v1_snapshot/_audit/log_metadata.json")
        assert log_member is not None
        log_metadata = json.load(log_member)
        event = next(item for item in log_metadata["logs"] if item["path"].endswith("events.jsonl"))
        assert event["line_count"] == 2
        assert event["ends_with_newline"] is False

    with pytest.raises(tool.IncidentArchiveError, match="overwrite"):
        tool.create_archive(
            project_root=project,
            results_root=project / "results" / "runs",
            output=output,
            process_snapshot=_idle_snapshot(),
            expected_commit=commit,
            expected_protocol_hash=protocol_hash,
            expected_manifest_sha256=manifest_sha256,
            expected_run_count=1,
        )


def test_archive_rejects_live_writer_and_protocol_mix(tmp_path: Path) -> None:
    project, commit, protocol_hash, manifest_sha256 = _project(tmp_path)
    output = tmp_path / "formal-v1.tar.gz"
    live = {
        "captured_at": "2026-07-27T00:00:00+08:00",
        "discovery_error": None,
        "processes": [tool.asdict(_record(10, scope="formal_v1"))],
    }
    with pytest.raises(tool.IncidentArchiveError, match="writers"):
        tool.create_archive(
            project_root=project,
            results_root=project / "results" / "runs",
            output=output,
            process_snapshot=live,
            expected_commit=commit,
            expected_protocol_hash=protocol_hash,
            expected_manifest_sha256=manifest_sha256,
            expected_run_count=1,
        )
    assert not output.exists()

    metrics_path = project / "results" / "runs" / "E1_cifar100_d20_t6_C1_s11" / "seed_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["protocol_hash"] = "c" * 64
    metrics_path.write_text(json.dumps(metrics))
    with pytest.raises(tool.IncidentArchiveError, match="identities"):
        tool.create_archive(
            project_root=project,
            results_root=project / "results" / "runs",
            output=output,
            process_snapshot=_idle_snapshot(),
            expected_commit=commit,
            expected_protocol_hash=protocol_hash,
            expected_manifest_sha256=manifest_sha256,
            expected_run_count=1,
        )
    assert not output.exists()
