#!/usr/bin/env python3
"""Archive complete V6 evidence without changing any source or result input."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyze_v6_results import collect_final_test_rows

from talif_msresnet.benchmark_v6 import validate_receipt
from talif_msresnet.config import artifact_paths_for_protocol, load_protocol
from talif_msresnet.config_v6 import (
    V6_ACTIVE_CONDITIONS,
    V6_FORMAL_SEEDS,
    validate_v6_cifar100_provenance_files,
)
from talif_msresnet.freeze import verify_formal_freeze
from talif_msresnet.utils import sha256_file, stable_hash

ARCHIVE_ROOT_NAME = "ta-lif-msresnet-v6-evidence"
ARCHIVE_SCHEMA = "ta-lif-msresnet-v6-evidence-archive-v1"
ARCHIVE_STATE_SCHEMA = "ta-lif-msresnet-v6-evidence-state-v1"
ANALYSIS_JSON = "v6_mechanism_accuracy_analysis.json"
ARCHIVE_STATE_FILE = "ARCHIVE_STATE.json"
_CIFAR100_PROVENANCE_STATE_KEYS = (
    "cifar100_source_provenance",
    "cifar100_source_provenance_sha256",
    "cifar100_test_pickle",
    "cifar100_test_pickle_sha256",
    "cifar100_split_manifest",
    "cifar100_split_manifest_sha256",
)


class V6EvidenceArchiveError(RuntimeError):
    """Raised when the reportable V6 evidence is incomplete or still changing."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New .tar.gz file to create outside the repository.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "protocol_v6_mechanism.yaml",
    )
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V6EvidenceArchiveError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V6EvidenceArchiveError(f"{label} must be a JSON object: {path}")
    return value


def _inside(root: Path, path: str | Path, label: str) -> Path:
    raw = Path(path)
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise V6EvidenceArchiveError(f"{label} escapes the project root: {resolved}") from exc
    return resolved


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise V6EvidenceArchiveError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _active_v6_processes() -> list[str]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "args="],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode:
        return []
    markers = (
        "talif_msresnet.train",
        "scripts/run_matrix.py",
        "scripts/run_v6_benchmarks.py",
        "scripts/evaluate_checkpoints.py",
        "scripts/analyze_v6_results.py",
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if "protocol_v6_mechanism" in line and any(marker in line for marker in markers)
    ]


def _copy_input(root: Path, staging: Path, path: Path, records: list[dict[str, Any]]) -> None:
    relative = path.relative_to(root)
    destination = staging / relative
    if path.is_dir():
        shutil.copytree(path, destination, dirs_exist_ok=False)
        files = sorted(item for item in path.rglob("*") if item.is_file())
        for item in files:
            item_relative = item.relative_to(root).as_posix()
            records.append(
                {"path": item_relative, "bytes": item.stat().st_size, "sha256": sha256_file(item)}
            )
    elif path.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records.append(
            {"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    else:
        raise V6EvidenceArchiveError(f"Archive input is missing: {path}")


def _write_hash_manifest(staging: Path, records: list[dict[str, Any]]) -> None:
    records.sort(key=lambda item: str(item["path"]))
    payload = {
        "schema": ARCHIVE_SCHEMA,
        "files": records,
        "file_set_sha256": stable_hash(records),
    }
    (staging / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _archive_state_payload(state: Mapping[str, Any], *, source_commit: str) -> dict[str, Any]:
    missing = [key for key in _CIFAR100_PROVENANCE_STATE_KEYS if key not in state]
    if missing:
        raise V6EvidenceArchiveError(
            "V6 archive state is missing frozen CIFAR-100 provenance fields: "
            + ", ".join(missing)
        )
    return {
        "schema": ARCHIVE_STATE_SCHEMA,
        "source_commit": source_commit,
        **dict(state),
    }


def _write_archive(staging: Path, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            archive.add(staging, arcname=ARCHIVE_ROOT_NAME, recursive=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_reportable_state(protocol_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    artifacts = artifact_paths_for_protocol(protocol)
    try:
        cifar100_binding = validate_v6_cifar100_provenance_files(
            protocol,
            project_root=PROJECT_ROOT,
        )
    except ValueError as exc:
        raise V6EvidenceArchiveError(
            f"Frozen CIFAR-100 provenance validation failed: {exc}"
        ) from exc
    manifest_path = _inside(PROJECT_ROOT, artifacts["freeze_manifest"], "V6 freeze manifest")
    if not manifest_path.is_file():
        raise V6EvidenceArchiveError("V6 freeze manifest is missing")
    verify_formal_freeze(
        project_root=PROJECT_ROOT,
        protocol_path=protocol_path,
        matrix_dir=PROJECT_ROOT / artifacts["formal_matrix"],
    )
    freeze_sha256 = sha256_file(manifest_path)
    formal_root = _inside(PROJECT_ROOT, artifacts["formal_results"], "V6 formal results")
    _frame, final_test_audit = collect_final_test_rows(
        formal_root,
        protocol=protocol,
        expected_protocol_hash=stable_hash(protocol),
    )
    receipt = validate_receipt(
        project_root=PROJECT_ROOT,
        protocol=protocol,
        protocol_path=protocol_path,
        protocol_hash=stable_hash(protocol),
        expected_freeze_manifest_sha256=freeze_sha256,
        expected_formal_rows=final_test_audit,
    )
    expected_ids = {
        f"E6_cifar100_d20_t6_{condition}_s{seed}"
        for seed in V6_FORMAL_SEEDS
        for condition in V6_ACTIVE_CONDITIONS
    }
    observed_ids = {path.name for path in formal_root.iterdir() if path.is_dir()}
    if observed_ids != expected_ids:
        raise V6EvidenceArchiveError(
            "V6 formal results directory set is incomplete or contains extras"
        )
    for run_id in sorted(expected_ids):
        run_dir = formal_root / run_id
        marker = run_dir / "final_test.json"
        if not marker.is_file() or (run_dir / "final_test.in_progress.json").exists():
            raise V6EvidenceArchiveError(f"{run_id}: final-test marker is missing or interrupted")
        value = _read_json(marker, f"{run_id} final-test marker")
        if value.get("status") != "complete" or value.get("run_id") != run_id:
            raise V6EvidenceArchiveError(f"{run_id}: final-test marker is not complete")
    analysis_root = _inside(PROJECT_ROOT, artifacts["analysis_results"], "V6 analysis results")
    analysis_path = analysis_root / ANALYSIS_JSON
    analysis = _read_json(analysis_path, "V6 analysis")
    if (
        analysis.get("protocol_hash") != stable_hash(protocol)
        or analysis.get("n_final_test_markers") != 48
    ):
        raise V6EvidenceArchiveError("V6 analysis is not bound to all 48 final-test markers")
    return {
        "freeze_manifest_sha256": freeze_sha256,
        "benchmark_receipt_sha256": sha256_file(
            _inside(PROJECT_ROOT, artifacts["benchmark_results"], "V6 benchmark results")
            / "benchmark_completion.json"
        ),
        "analysis_sha256": sha256_file(analysis_path),
        "training_environment_sha256": receipt["training_environment_sha256"],
        **cifar100_binding,
    }


def create_archive(*, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    output = output.expanduser().resolve()
    if output.exists() or Path(f"{output}.sha256").exists():
        raise V6EvidenceArchiveError(
            "Refusing to overwrite an existing V6 evidence archive or sidecar"
        )
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise V6EvidenceArchiveError("Evidence archive output must end in .tar.gz")
    active = _active_v6_processes()
    if active:
        raise V6EvidenceArchiveError("V6 processes are still running; archive only after they stop")
    protocol = load_protocol(protocol_path)
    if int(protocol.get("protocol_version", 0)) != 6:
        raise V6EvidenceArchiveError("Evidence archive accepts only protocol v6")
    state = _validate_reportable_state(protocol_path, protocol)
    artifacts = artifact_paths_for_protocol(protocol)
    source_commit = _git("rev-parse", "HEAD")
    archive_state = _archive_state_payload(state, source_commit=source_commit)
    staging_parent = Path(tempfile.mkdtemp(prefix=".v6-evidence-", dir=output.parent))
    try:
        staging = staging_parent / ARCHIVE_ROOT_NAME
        staging.mkdir(parents=True)
        records: list[dict[str, Any]] = []
        relative_inputs = (
            artifacts["protocol"],
            artifacts["signoff"],
            artifacts["freeze_manifest"],
            artifacts["formal_matrix"],
            artifacts["pilot_matrix"],
            artifacts["pilot_results"],
            artifacts["formal_results"],
            artifacts["benchmark_results"],
            artifacts["analysis_results"],
            "V6_STATISTICAL_ANALYSIS_AUDIT.md",
            state["cifar100_split_manifest"],
            state["cifar100_source_provenance"],
        )
        for relative in relative_inputs:
            _copy_input(
                PROJECT_ROOT, staging, _inside(PROJECT_ROOT, relative, "V6 archive input"), records
            )
        source_bundle = staging / "source" / "ta-lif-msresnet-v6.bundle"
        source_bundle.parent.mkdir(parents=True, exist_ok=True)
        _git("bundle", "create", str(source_bundle), "HEAD")
        records.append(
            {
                "path": "source/ta-lif-msresnet-v6.bundle",
                "bytes": source_bundle.stat().st_size,
                "sha256": sha256_file(source_bundle),
            }
        )
        (staging / "ARCHIVE_README.txt").write_text(
            "V6 evidence archive. V5 evidence remains outside this archive.\n"
            "All listed hashes bind the copied files; the archive is read-only evidence.\n",
            encoding="utf-8",
            newline="\n",
        )
        archive_state_path = staging / ARCHIVE_STATE_FILE
        archive_state_path.write_text(
            json.dumps(archive_state, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        records.append(
            {
                "path": ARCHIVE_STATE_FILE,
                "bytes": archive_state_path.stat().st_size,
                "sha256": sha256_file(archive_state_path),
            }
        )
        _write_hash_manifest(staging, records)
        _write_archive(staging, output)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)
    archive_sha256 = sha256_file(output)
    sidecar = Path(f"{output}.sha256")
    sidecar.write_text(f"{archive_sha256}  {output.name}\n", encoding="utf-8", newline="\n")
    return {
        "output": str(output),
        "sha256": archive_sha256,
        "source_commit": source_commit,
        **state,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = create_archive(protocol_path=args.protocol, output=args.output)
    except (OSError, ValueError, V6EvidenceArchiveError) as exc:
        print(f"V6_EVIDENCE_ARCHIVE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"V6_EVIDENCE_ARCHIVE_CREATED={result['output']}")
    print(f"V6_EVIDENCE_ARCHIVE_SHA256={result['sha256']}")
    print(f"V6_EVIDENCE_ARCHIVE_SOURCE={result['source_commit']}")
    print(f"V6_EVIDENCE_ARCHIVE_TRAINING_ENV={result['training_environment_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
