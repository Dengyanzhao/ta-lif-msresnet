#!/usr/bin/env python3
"""Inspect, stop, or immutably archive the withdrawn formal-v1 experiment.

With no action flag this command only prints a process snapshot. Stopping
requires both ``--stop`` and the exact confirmation token. Archive creation is
exclusive, rejects live writers and identity conflicts, and never removes its
inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_V1_FREEZE_COMMIT = "369a96b142532e9084bffbae008942477e05fcfd"
FORMAL_V1_PROTOCOL_HASH = "bec83ee5145518f2dd81e02bdf3d3499021c5c86489b5d54655ff9f2f57fcf1e"
FORMAL_V1_FREEZE_MANIFEST_SHA256 = (
    "3137389accdbfddfecd50ff1cfa74ce4f1179ccfc41d2a0b2e8b9001e924778d"
)
FORMAL_V1_RUN_COUNT = 40
STOP_CONFIRMATION = "STOP_FORMAL_V1_369A96B"
ARCHIVE_SCHEMA = "ta-lif-msresnet-formal-v1-incident-archive-v1"
ARCHIVE_ROOT = "formal_v1_snapshot"
PROCESS_MARKERS = ("run_matrix.py", "talif_msresnet.train")
LOG_NAMES = {"events.jsonl", "matrix_events.jsonl"}


class IncidentArchiveError(RuntimeError):
    """Raised when a safe stop or internally consistent archive is impossible."""


@dataclass(frozen=True)
class ProcessRecord:
    pid: int
    ppid: int
    elapsed_seconds: int
    kind: str
    scope: str
    match_reason: str
    cwd: str | None
    declared_output: str | None
    resolved_output: str | None
    command: str


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _inside(root: Path, value: str | Path, label: str, *, must_exist: bool) -> Path:
    root = root.resolve()
    raw = Path(value).expanduser()
    path = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise IncidentArchiveError(f"{label} must stay inside project root: {path}") from exc
    if must_exist and not path.exists():
        raise IncidentArchiveError(f"{label} does not exist: {path}")
    return path


def _option_value(tokens: Sequence[str], names: set[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token in names:
            return tokens[index + 1] if index + 1 < len(tokens) else None
        for name in names:
            prefix = f"{name}="
            if token.startswith(prefix):
                return token[len(prefix) :]
    return None


def _process_kind(tokens: Sequence[str], command: str) -> str | None:
    if any(Path(token).name == "run_matrix.py" for token in tokens):
        return "matrix"
    if any(token == "talif_msresnet.train" for token in tokens) or (
        "talif_msresnet.train" in command
    ):
        return "trainer"
    return None


def _default_cwd_resolver(pid: int) -> Path | None:
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd")).resolve()
    except (FileNotFoundError, OSError):
        return None


def parse_process_table(
    text: str,
    *,
    project_root: Path,
    results_root: Path,
    cwd_resolver: Callable[[int], Path | None] = _default_cwd_resolver,
) -> list[ProcessRecord]:
    """Classify experiment processes without trusting a stored PID."""

    project_root = project_root.resolve()
    results_root = results_root.resolve()
    records: list[ProcessRecord] = []
    for raw in text.splitlines():
        fields = raw.strip().split(maxsplit=3)
        if len(fields) != 4:
            continue
        try:
            pid, ppid, elapsed = (int(fields[index]) for index in range(3))
        except ValueError:
            continue
        command = fields[3]
        if not any(marker in command for marker in PROCESS_MARKERS):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        kind = _process_kind(tokens, command)
        if kind is None:
            continue
        cwd = cwd_resolver(pid)
        declared = _option_value(
            tokens,
            {"--output-root"} if kind == "matrix" else {"--output-dir"},
        )
        resolved: Path | None = None
        if declared:
            raw_output = Path(declared).expanduser()
            if raw_output.is_absolute():
                resolved = raw_output.resolve()
            elif cwd is not None:
                resolved = (cwd / raw_output).resolve()

        in_project = cwd is not None and (cwd == project_root or project_root in cwd.parents)
        if resolved == results_root:
            scope, reason = "formal_v1", "explicit formal output path"
        elif resolved is not None:
            scope, reason = "other", "explicit different output path"
        elif declared is not None:
            scope, reason = "ambiguous", "relative output path but process cwd is unavailable"
        elif in_project:
            scope, reason = "formal_v1", "project process using the protocol default output"
        else:
            scope, reason = "ambiguous", "experiment process has no resolvable output path"
        records.append(
            ProcessRecord(
                pid=pid,
                ppid=ppid,
                elapsed_seconds=elapsed,
                kind=kind,
                scope=scope,
                match_reason=reason,
                cwd=str(cwd) if cwd is not None else None,
                declared_output=declared,
                resolved_output=str(resolved) if resolved is not None else None,
                command=command,
            )
        )
    return sorted(records, key=lambda item: item.pid)


def inspect_processes(project_root: Path, results_root: Path) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,etimes=,args="],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        return {
            "captured_at": _now(),
            "discovery_error": str(exc),
            "processes": [],
        }
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"ps returned {completed.returncode}"
        return {"captured_at": _now(), "discovery_error": detail, "processes": []}
    records = parse_process_table(
        completed.stdout,
        project_root=project_root,
        results_root=results_root,
    )
    return {
        "captured_at": _now(),
        "discovery_error": None,
        "processes": [asdict(record) for record in records],
    }


def _records(snapshot: Mapping[str, Any], scope: str) -> list[dict[str, Any]]:
    processes = snapshot.get("processes", [])
    if not isinstance(processes, list):
        raise IncidentArchiveError("Malformed process snapshot")
    return [
        dict(item) for item in processes if isinstance(item, Mapping) and item.get("scope") == scope
    ]


def signal_formal_processes(
    records: Sequence[ProcessRecord],
    *,
    confirmation: str | None,
    signaler: Callable[[int, int], None] = os.kill,
) -> list[int]:
    """Send SIGTERM only to dynamically matched formal-v1 processes."""

    if confirmation != STOP_CONFIRMATION:
        raise IncidentArchiveError(
            f"Stopping requires --confirm-stop {STOP_CONFIRMATION}; no signal was sent"
        )
    ambiguous = [record.pid for record in records if record.scope == "ambiguous"]
    if ambiguous:
        raise IncidentArchiveError(
            f"Ambiguous experiment processes require manual review before stopping: {ambiguous}"
        )
    matched = [record for record in records if record.scope == "formal_v1"]
    ordered = sorted(matched, key=lambda item: (item.kind != "matrix", item.pid))
    signaled: list[int] = []
    for record in ordered:
        try:
            signaler(record.pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        signaled.append(record.pid)
    return signaled


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise IncidentArchiveError(f"Refusing to overwrite existing receipt: {path}") from exc


def controlled_stop(
    *,
    project_root: Path,
    results_root: Path,
    confirmation: str | None,
    receipt_path: Path,
    wait_seconds: float,
) -> dict[str, Any]:
    before = inspect_processes(project_root, results_root)
    if before.get("discovery_error"):
        raise IncidentArchiveError(f"Cannot safely discover processes: {before['discovery_error']}")
    records = [ProcessRecord(**item) for item in before["processes"]]
    signaled = signal_formal_processes(records, confirmation=confirmation)
    deadline = time.monotonic() + max(0.0, wait_seconds)
    after = before
    while signaled and time.monotonic() < deadline:
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        after = inspect_processes(project_root, results_root)
        remaining = [*_records(after, "formal_v1"), *_records(after, "ambiguous")]
        if after.get("discovery_error") or not remaining:
            break
    if not signaled:
        after = inspect_processes(project_root, results_root)
    receipt = {
        "schema": "ta-lif-msresnet-formal-v1-stop-receipt-v1",
        "formal_v1_freeze_commit": FORMAL_V1_FREEZE_COMMIT,
        "formal_v1_protocol_hash": FORMAL_V1_PROTOCOL_HASH,
        "signal": "SIGTERM",
        "forced_kill_used": False,
        "signaled_pids": signaled,
        "before": before,
        "after": after,
        "completed_at": _now(),
    }
    _exclusive_json(receipt_path, receipt)
    return receipt


def _validate_freeze_identity(
    project_root: Path,
    *,
    expected_commit: str,
    expected_protocol_hash: str,
    expected_manifest_sha256: str,
    expected_run_count: int,
) -> dict[str, Any]:
    path = project_root / "FREEZE_MANIFEST.json"
    if not path.is_file():
        raise IncidentArchiveError(f"Missing formal-v1 freeze manifest: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_manifest_sha256:
        raise IncidentArchiveError(
            "FREEZE_MANIFEST.json SHA-256 differs from the formal-v1 external receipt"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IncidentArchiveError(f"Cannot read freeze manifest: {exc}") from exc
    if not isinstance(value, Mapping):
        raise IncidentArchiveError("Freeze manifest root must be an object")
    repository = value.get("repository")
    protocol = value.get("protocol")
    signed_record = value.get("signed_record")
    generated = value.get("generated_matrix")
    bindings = (repository, protocol, signed_record, generated)
    if not all(isinstance(binding, Mapping) for binding in bindings):
        raise IncidentArchiveError("Freeze manifest is missing a required identity object")
    assert isinstance(repository, Mapping)
    assert isinstance(protocol, Mapping)
    assert isinstance(signed_record, Mapping)
    assert isinstance(generated, Mapping)
    if repository.get("freeze_commit") != expected_commit:
        raise IncidentArchiveError("Freeze manifest commit is not formal v1")
    if protocol.get("canonical_sha256") != expected_protocol_hash:
        raise IncidentArchiveError("Freeze manifest protocol hash is not formal v1")

    def verify_bound_file(root: Path, binding: Mapping[str, Any], label: str) -> Path:
        bound_path = binding.get("path")
        bound_sha256 = binding.get("sha256", binding.get("file_sha256"))
        if not isinstance(bound_path, str) or not isinstance(bound_sha256, str):
            raise IncidentArchiveError(f"Freeze manifest has an invalid {label} binding")
        selected = _inside(root, bound_path, label, must_exist=True)
        if not selected.is_file():
            raise IncidentArchiveError(f"Bound {label} is not a file: {selected}")
        if _sha256_file(selected) != bound_sha256:
            raise IncidentArchiveError(f"Bound {label} SHA-256 differs from Phase B")
        return selected

    protocol_path = verify_bound_file(project_root, protocol, "protocol")
    signoff_path = verify_bound_file(project_root, signed_record, "signed record")

    matrix_directory = generated.get("directory")
    if not isinstance(matrix_directory, str):
        raise IncidentArchiveError("Freeze manifest has no generated-matrix directory")
    matrix_dir = _inside(project_root, matrix_directory, "generated matrix", must_exist=True)
    if not matrix_dir.is_dir():
        raise IncidentArchiveError(f"Generated-matrix binding is not a directory: {matrix_dir}")
    matrix_binding = generated.get("matrix_manifest")
    run_binding = generated.get("run_manifest")
    if not isinstance(matrix_binding, Mapping) or not isinstance(run_binding, Mapping):
        raise IncidentArchiveError("Freeze manifest has invalid generated-matrix file bindings")
    matrix_manifest_path = verify_bound_file(matrix_dir, matrix_binding, "matrix manifest")
    run_manifest_path = verify_bound_file(matrix_dir, run_binding, "run manifest")

    try:
        matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IncidentArchiveError(f"Cannot read bound matrix manifest: {exc}") from exc
    if not isinstance(matrix_manifest, Mapping):
        raise IncidentArchiveError("Bound matrix manifest root must be an object")
    if matrix_manifest.get("protocol_hash") != expected_protocol_hash:
        raise IncidentArchiveError("Bound matrix manifest protocol hash is not formal v1")
    if matrix_manifest.get("run_count") != expected_run_count:
        raise IncidentArchiveError(
            f"Formal v1 requires {expected_run_count} generated configurations"
        )
    if generated.get("run_count") != expected_run_count:
        raise IncidentArchiveError("Freeze manifest generated run count is not formal v1")
    rows = matrix_manifest.get("runs")
    if not isinstance(rows, list) or len(rows) != expected_run_count:
        raise IncidentArchiveError("Bound matrix manifest has an invalid run list")
    expected_configs: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise IncidentArchiveError("Bound matrix manifest contains a malformed run row")
        filename = row.get("config_file")
        config_sha256 = row.get("config_file_sha256")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".yaml")
            or not isinstance(config_sha256, str)
        ):
            raise IncidentArchiveError("Bound matrix manifest has an invalid config binding")
        config_path = _inside(matrix_dir, filename, "generated config", must_exist=True)
        if not config_path.is_file() or config_path.parent != matrix_dir:
            raise IncidentArchiveError(f"Generated config path is invalid: {config_path}")
        if filename in expected_configs:
            raise IncidentArchiveError(f"Duplicate generated config binding: {filename}")
        if row.get("protocol_hash") != expected_protocol_hash:
            raise IncidentArchiveError(f"Generated config row is not formal v1: {filename}")
        if _sha256_file(config_path) != config_sha256:
            raise IncidentArchiveError(f"Generated config SHA-256 differs from Phase B: {filename}")
        expected_configs.add(filename)
    actual_entries = {item.name for item in matrix_dir.iterdir()}
    expected_files = {
        *expected_configs,
        matrix_manifest_path.name,
        run_manifest_path.name,
    }
    if actual_entries != expected_files:
        raise IncidentArchiveError("Generated-matrix directory has missing or unexpected entries")
    return {
        "path": "FREEZE_MANIFEST.json",
        "sha256": actual_sha256,
        "protocol_path": protocol_path.relative_to(project_root).as_posix(),
        "signed_record_path": signoff_path.relative_to(project_root).as_posix(),
        "matrix_directory": matrix_dir.relative_to(project_root).as_posix(),
        "config_count": len(expected_configs),
    }


def _git(project_root: Path, *arguments: str) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _repository_metadata(project_root: Path) -> dict[str, Any]:
    return {
        "captured_at": _now(),
        "head": _git(project_root, "rev-parse", "HEAD"),
        "formal_v1_commit_object": _git(
            project_root, "cat-file", "-e", f"{FORMAL_V1_FREEZE_COMMIT}^{{commit}}"
        ),
        "formal_v1_commit": _git(
            project_root,
            "show",
            "-s",
            "--format=%H%n%T%n%cI%n%an <%ae>%n%s",
            FORMAL_V1_FREEZE_COMMIT,
        ),
        "origin": _git(project_root, "config", "--get", "remote.origin.url"),
        "tracked_worktree_status": _git(
            project_root, "status", "--porcelain=v1", "--untracked-files=no"
        ),
    }


def _read_protocol_hash(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    direct = value.get("protocol_hash")
    if isinstance(direct, str):
        return direct
    analysis = value.get("analysis")
    if isinstance(analysis, Mapping) and isinstance(analysis.get("protocol_hash"), str):
        return str(analysis["protocol_hash"])
    config = value.get("config")
    if isinstance(config, Mapping):
        nested = config.get("analysis")
        if isinstance(nested, Mapping) and isinstance(nested.get("protocol_hash"), str):
            return str(nested["protocol_hash"])
    return None


def _result_summary(results_root: Path, expected_protocol_hash: str) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    conflicts: list[dict[str, str]] = []
    protocol_hashes: set[str] = set()
    for run_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
        identities: list[dict[str, str]] = []
        for name in ("seed_metrics.json", "run_manifest.json", "resolved_config.json"):
            path = run_dir / name
            if not path.is_file():
                continue
            protocol_hash = _read_protocol_hash(path)
            if protocol_hash is not None:
                protocol_hashes.add(protocol_hash)
                identities.append({"path": name, "protocol_hash": protocol_hash})
                if protocol_hash != expected_protocol_hash:
                    conflicts.append(
                        {
                            "run_id": run_dir.name,
                            "path": path.relative_to(results_root.parent.parent).as_posix(),
                            "protocol_hash": protocol_hash,
                        }
                    )
        status = "incomplete"
        metrics_path = run_dir / "seed_metrics.json"
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if isinstance(metrics, Mapping):
                    status = str(metrics.get("status", "unknown"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                status = "unreadable_seed_metrics"
        runs.append(
            {
                "run_id": run_dir.name,
                "status": status,
                "identity_records": identities,
                "has_best_checkpoint": (run_dir / "best.pt").is_file(),
                "has_last_checkpoint": (run_dir / "last.pt").is_file(),
                "has_failure_record": (run_dir / "failure.json").is_file(),
            }
        )
    counts: dict[str, int] = {}
    for run in runs:
        status = str(run["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "results_root": results_root.name,
        "run_count": len(runs),
        "status_counts": counts,
        "observed_protocol_hashes": sorted(protocol_hashes),
        "identity_conflicts": conflicts,
        "runs": runs,
    }


def _source_paths(project_root: Path, results_root: Path) -> list[Path]:
    required = [
        results_root,
        project_root / "FREEZE_MANIFEST.json",
        project_root / "configs" / "protocol.yaml",
        project_root / "configs" / "generated",
        project_root / "PREREGISTRATION_SIGNOFF.md",
    ]
    optional = [
        project_root / "environment",
        project_root / "FORMAL_V1_INCIDENT.md",
        project_root / "scripts" / "archive_formal_v1.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise IncidentArchiveError(f"Required archive inputs are missing: {missing}")
    selected = [*required, *(path for path in optional if path.exists())]
    return sorted({path.resolve() for path in selected}, key=lambda item: item.as_posix())


def _walk_sources(project_root: Path, sources: Sequence[Path]) -> list[Path]:
    entries: set[Path] = set()
    for source in sources:
        candidates: Iterable[Path] = (
            chain((source,), source.rglob("*")) if source.is_dir() else (source,)
        )
        for path in candidates:
            resolved_parent = path.parent.resolve()
            try:
                resolved_parent.relative_to(project_root)
            except ValueError as exc:
                raise IncidentArchiveError(f"Archive input escapes project root: {path}") from exc
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise IncidentArchiveError(
                    f"Symlinks are not accepted in an incident archive: {path}"
                )
            if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise IncidentArchiveError(f"Unsupported archive input type: {path}")
            entries.add(path)
    return sorted(entries, key=lambda item: item.relative_to(project_root).as_posix())


class _HashingReader:
    def __init__(self, handle: Any) -> None:
        self.handle = handle
        self.digest = hashlib.sha256()
        self.line_count = 0
        self.last_byte: int | None = None

    def read(self, size: int = -1) -> bytes:
        block = self.handle.read(size)
        if block:
            self.digest.update(block)
            self.line_count += block.count(b"\n")
            self.last_byte = block[-1]
        return block


def _add_bytes(tar: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(value)
    info.mode = 0o444
    info.mtime = int(time.time())
    tar.addfile(info, io.BytesIO(value))


def _publish_exclusive(temporary: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary, output)
    except FileExistsError as exc:
        raise IncidentArchiveError(f"Refusing to overwrite existing archive: {output}") from exc
    except OSError:
        try:
            with temporary.open("rb") as source, output.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
        except FileExistsError as exc:
            raise IncidentArchiveError(f"Refusing to overwrite existing archive: {output}") from exc
        except Exception:
            output.unlink(missing_ok=True)
            raise


def create_archive(
    *,
    project_root: Path,
    results_root: Path,
    output: Path,
    process_snapshot: Mapping[str, Any] | None = None,
    expected_commit: str = FORMAL_V1_FREEZE_COMMIT,
    expected_protocol_hash: str = FORMAL_V1_PROTOCOL_HASH,
    expected_manifest_sha256: str = FORMAL_V1_FREEZE_MANIFEST_SHA256,
    expected_run_count: int = FORMAL_V1_RUN_COUNT,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    results_root = _inside(project_root, results_root, "Formal results root", must_exist=True)
    output = output.expanduser().resolve()
    if output.exists():
        raise IncidentArchiveError(f"Refusing to overwrite existing archive: {output}")
    sources = _source_paths(project_root, results_root)
    for source in sources:
        if output == source or source in output.parents or output in source.parents:
            raise IncidentArchiveError("Archive output must be outside every archived input")

    supplied_snapshot = process_snapshot is not None
    before_snapshot = (
        dict(process_snapshot)
        if supplied_snapshot
        else inspect_processes(project_root, results_root)
    )
    if before_snapshot.get("discovery_error"):
        raise IncidentArchiveError(
            f"Process discovery failed; archive not created: {before_snapshot['discovery_error']}"
        )
    live = _records(before_snapshot, "formal_v1")
    ambiguous = _records(before_snapshot, "ambiguous")
    if live or ambiguous:
        pids = sorted(int(item["pid"]) for item in [*live, *ambiguous])
        raise IncidentArchiveError(f"Live or ambiguous experiment writers block archive: {pids}")

    freeze = _validate_freeze_identity(
        project_root,
        expected_commit=expected_commit,
        expected_protocol_hash=expected_protocol_hash,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_run_count=expected_run_count,
    )
    summary = _result_summary(results_root, expected_protocol_hash)
    if summary["identity_conflicts"]:
        raise IncidentArchiveError(
            "Formal results contain protocol identities other than formal v1; archive separately"
        )
    entries = _walk_sources(project_root, sources)
    initial_stats = {path: path.lstat() for path in entries}
    repository = _repository_metadata(project_root)
    incident = {
        "schema": ARCHIVE_SCHEMA,
        "created_at": _now(),
        "disposition": "withdrawn_non_reportable_audit_only",
        "formal_v1_freeze_commit": expected_commit,
        "formal_v1_protocol_hash": expected_protocol_hash,
        "formal_v1_freeze_manifest_sha256": expected_manifest_sha256,
        "mix_with_later_results": "forbidden",
        "source_removed_after_archive": False,
        "archive_sha256_note": (
            "The archive cannot contain its own SHA-256; retain the value printed by the command."
        ),
        "freeze_manifest": {"path": freeze["path"], "sha256": freeze["sha256"]},
        "bound_config_count": freeze["config_count"],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    inventory: list[dict[str, Any]] = []
    log_metadata: list[dict[str, Any]] = []
    try:
        with tarfile.open(temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as tar:
            for path in entries:
                before = initial_stats[path]
                current = path.lstat()
                if (before.st_mode, before.st_size, before.st_mtime_ns) != (
                    current.st_mode,
                    current.st_size,
                    current.st_mtime_ns,
                ):
                    raise IncidentArchiveError(f"Archive input changed before capture: {path}")
                relative = path.relative_to(project_root).as_posix()
                archive_name = f"{ARCHIVE_ROOT}/{relative}"
                info = tar.gettarinfo(str(path), arcname=archive_name)
                record: dict[str, Any] = {
                    "path": relative,
                    "type": "directory" if path.is_dir() else "file",
                    "size_bytes": current.st_size,
                    "mtime_ns": current.st_mtime_ns,
                    "mode": oct(stat.S_IMODE(current.st_mode)),
                }
                if path.is_dir():
                    tar.addfile(info)
                else:
                    with path.open("rb") as handle:
                        reader = _HashingReader(handle)
                        tar.addfile(info, reader)
                    after = path.lstat()
                    if (current.st_size, current.st_mtime_ns) != (
                        after.st_size,
                        after.st_mtime_ns,
                    ):
                        raise IncidentArchiveError(f"Archive input changed during capture: {path}")
                    record["sha256"] = reader.digest.hexdigest()
                    is_log = path.suffix == ".log" or path.name in LOG_NAMES
                    if is_log:
                        line_count = reader.line_count + int(
                            current.st_size > 0 and reader.last_byte != ord("\n")
                        )
                        log_metadata.append(
                            {
                                "path": relative,
                                "size_bytes": current.st_size,
                                "sha256": record["sha256"],
                                "line_count": line_count,
                                "ends_with_newline": (
                                    current.st_size == 0 or reader.last_byte == ord("\n")
                                ),
                            }
                        )
                inventory.append(record)

            final_entries = _walk_sources(project_root, sources)
            if final_entries != entries:
                raise IncidentArchiveError("Archive input tree changed during capture")
            for path, before in initial_stats.items():
                after = path.lstat()
                if (before.st_mode, before.st_size, before.st_mtime_ns) != (
                    after.st_mode,
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise IncidentArchiveError(f"Archive input changed during capture: {path}")

            after_snapshot = (
                before_snapshot
                if supplied_snapshot
                else inspect_processes(project_root, results_root)
            )
            if after_snapshot.get("discovery_error"):
                raise IncidentArchiveError(
                    "Final process discovery failed; archive not created: "
                    f"{after_snapshot['discovery_error']}"
                )
            final_writers = [
                *_records(after_snapshot, "formal_v1"),
                *_records(after_snapshot, "ambiguous"),
            ]
            if final_writers:
                pids = sorted(int(item["pid"]) for item in final_writers)
                raise IncidentArchiveError(
                    f"An experiment writer appeared during archive capture: {pids}"
                )

            checksum_lines = [
                f"{item['sha256']}  {item['path']}" for item in inventory if item["type"] == "file"
            ]
            audit = {
                "incident.json": incident,
                "processes.json": {
                    "before_capture": before_snapshot,
                    "after_capture": after_snapshot,
                },
                "repository.json": repository,
                "result_summary.json": summary,
                "inventory.json": {"entries": inventory},
                "log_metadata.json": {"logs": log_metadata},
            }
            for filename, value in audit.items():
                _add_bytes(tar, f"{ARCHIVE_ROOT}/_audit/{filename}", _json_bytes(value))
            _add_bytes(
                tar,
                f"{ARCHIVE_ROOT}/_audit/SHA256SUMS",
                ("\n".join(checksum_lines) + "\n").encode("utf-8"),
            )
        _publish_exclusive(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "archive": str(output),
        "sha256": _sha256_file(output),
        "file_count": sum(item["type"] == "file" for item in inventory),
        "directory_count": sum(item["type"] == "directory" for item in inventory),
        "log_count": len(log_metadata),
        "run_count": summary["run_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--results-root", type=Path, default=Path("results/runs"))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--stop", action="store_true", help="Send SIGTERM after explicit confirmation"
    )
    actions.add_argument("--archive", type=Path, help="Create a new immutable tar.gz snapshot")
    parser.add_argument(
        "--confirm-stop",
        help=f"Exact token required with --stop: {STOP_CONFIRMATION}",
    )
    parser.add_argument("--stop-receipt", type=Path, help="New JSON receipt required with --stop")
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    try:
        results_root = _inside(
            project_root, args.results_root, "Formal results root", must_exist=True
        )
        if args.stop:
            if args.stop_receipt is None:
                raise IncidentArchiveError("--stop-receipt is required with --stop")
            receipt = controlled_stop(
                project_root=project_root,
                results_root=results_root,
                confirmation=args.confirm_stop,
                receipt_path=args.stop_receipt.expanduser().resolve(),
                wait_seconds=args.wait_seconds,
            )
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
            remaining = [
                *_records(receipt["after"], "formal_v1"),
                *_records(receipt["after"], "ambiguous"),
            ]
            if receipt["after"].get("discovery_error") or remaining:
                print("STOP_INCOMPLETE: no forced kill was attempted", file=sys.stderr)
                return 3
            print("FORMAL_V1_CONTROLLED_STOP_PASS")
            return 0
        if args.archive is not None:
            result = create_archive(
                project_root=project_root,
                results_root=results_root,
                output=args.archive,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            print("FORMAL_V1_ARCHIVE_PASS")
            print("Source files were not removed.")
            return 0
        snapshot = inspect_processes(project_root, results_root)
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        if snapshot.get("discovery_error"):
            print("READ_ONLY_INSPECTION_FAILED", file=sys.stderr)
            return 2
        print("READ_ONLY_INSPECTION_PASS")
        return 0
    except (IncidentArchiveError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
