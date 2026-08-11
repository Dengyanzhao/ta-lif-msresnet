#!/usr/bin/env python3
"""Archive complete V7 evidence without changing any source or result input."""

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

from analyze_v7_results import collect_final_test_rows

from talif_msresnet.benchmark_v7 import (
    V7BenchmarkError,
    canonical_bound_paths,
    expected_receipt_path,
    expected_run_ids,
    repository_git_identity,
    validate_receipt,
)
from talif_msresnet.config import artifact_paths_for_protocol, load_protocol
from talif_msresnet.config_v7 import validate_v7_cifar100_provenance_files
from talif_msresnet.freeze import verify_formal_freeze
from talif_msresnet.utils import sha256_file, stable_hash

ARCHIVE_ROOT_NAME = "ta-lif-msresnet-v7-evidence"
ARCHIVE_SCHEMA = "ta-lif-msresnet-v7-evidence-archive-v1"
ARCHIVE_STATE_SCHEMA = "ta-lif-msresnet-v7-evidence-state-v1"
ANALYSIS_JSON = "v7_mechanism_accuracy_analysis.json"
RAW_ACCURACIES_CSV = "v7_seed_condition_test_accuracies.csv"
RAW_DIFFERENCES_CSV = "v7_paired_seed_differences.csv"
ARCHIVE_STATE_FILE = "ARCHIVE_STATE.json"
_CIFAR100_PROVENANCE_STATE_KEYS = (
    "test_source",
    "test_source_sha256",
    "cifar100_source_provenance",
    "cifar100_source_provenance_sha256",
    "cifar100_archive",
    "cifar100_archive_sha256",
    "cifar100_train_pickle",
    "cifar100_train_pickle_sha256",
    "cifar100_test_pickle",
    "cifar100_test_pickle_sha256",
    "cifar100_meta_pickle",
    "cifar100_meta_pickle_sha256",
    "cifar100_split_manifest",
    "cifar100_split_manifest_sha256",
)


class V7EvidenceArchiveError(RuntimeError):
    """Raised when the reportable V7 evidence is incomplete or still changing."""


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
        default=PROJECT_ROOT / "configs" / "protocol_v7_mechanism.yaml",
    )
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V7EvidenceArchiveError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V7EvidenceArchiveError(f"{label} must be a JSON object: {path}")
    return value


def _inside(root: Path, path: str | Path, label: str) -> Path:
    raw = Path(path)
    resolved = (raw if raw.is_absolute() else root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise V7EvidenceArchiveError(f"{label} escapes the project root: {resolved}") from exc
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
        raise V7EvidenceArchiveError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _active_v7_processes() -> list[str]:
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
        "scripts/run_v7_benchmarks.py",
        "scripts/evaluate_checkpoints.py",
        "scripts/analyze_v7_results.py",
    )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if "protocol_v7_mechanism" in line and any(marker in line for marker in markers)
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
        raise V7EvidenceArchiveError(f"Archive input is missing: {path}")


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
        raise V7EvidenceArchiveError(
            "V7 archive state is missing frozen CIFAR-100 provenance fields: "
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
        cifar100_binding = validate_v7_cifar100_provenance_files(
            protocol,
            project_root=PROJECT_ROOT,
        )
    except ValueError as exc:
        raise V7EvidenceArchiveError(
            f"Frozen CIFAR-100 provenance validation failed: {exc}"
        ) from exc
    final_test_source_binding = {
        "test_source": cifar100_binding["cifar100_test_pickle"],
        "test_source_sha256": cifar100_binding["cifar100_test_pickle_sha256"],
        **cifar100_binding,
    }
    bound_paths = canonical_bound_paths(protocol, PROJECT_ROOT)
    manifest_path = bound_paths["freeze_manifest"]
    if not manifest_path.is_file():
        raise V7EvidenceArchiveError("V7 freeze manifest is missing")
    verify_formal_freeze(
        project_root=PROJECT_ROOT,
        protocol_path=protocol_path,
        matrix_dir=PROJECT_ROOT / artifacts["formal_matrix"],
    )
    freeze_sha256 = sha256_file(manifest_path)
    matrix_manifest_path = bound_paths["matrix_manifest"]
    if not matrix_manifest_path.is_file():
        raise V7EvidenceArchiveError("V7 formal matrix manifest is missing")
    matrix_manifest_sha256 = sha256_file(matrix_manifest_path)
    source_commit, tracked_clean = repository_git_identity(PROJECT_ROOT)
    if tracked_clean is not True:
        raise V7EvidenceArchiveError(
            "V7 evidence archive requires the current clean tracked release commit"
        )
    receipt_path = expected_receipt_path(protocol, PROJECT_ROOT)
    if not receipt_path.is_file():
        raise V7EvidenceArchiveError("V7 benchmark completion receipt is missing")
    receipt_sha256_before = sha256_file(receipt_path)
    receipt_reference = receipt_path.relative_to(PROJECT_ROOT).as_posix()
    benchmark_binding = {
        "benchmark_receipt": receipt_reference,
        "benchmark_receipt_sha256": receipt_sha256_before,
    }
    formal_root = _inside(PROJECT_ROOT, artifacts["formal_results"], "V7 formal results")
    _frame, final_test_audit = collect_final_test_rows(
        formal_root,
        protocol=protocol,
        expected_protocol_hash=stable_hash(protocol),
        expected_benchmark_binding=benchmark_binding,
    )
    try:
        receipt = validate_receipt(
            project_root=PROJECT_ROOT,
            protocol=protocol,
            protocol_path=protocol_path,
            protocol_hash=stable_hash(protocol),
            expected_git_commit=source_commit,
            expected_matrix_manifest_sha256=matrix_manifest_sha256,
            expected_freeze_manifest_sha256=freeze_sha256,
            expected_formal_rows=final_test_audit,
        )
    except V7BenchmarkError as exc:
        raise V7EvidenceArchiveError(
            f"V7 benchmark receipt validation failed: {exc}"
        ) from exc
    receipt_sha256_after = sha256_file(receipt_path)
    if receipt_sha256_after != receipt_sha256_before:
        raise V7EvidenceArchiveError(
            "V7 benchmark receipt changed while archive inputs were validated"
        )
    expected_ids = set(expected_run_ids())
    observed_ids = {
        path.name
        for path in formal_root.iterdir()
        if path.is_dir() and path.name != "logs"
    }
    if observed_ids != expected_ids:
        raise V7EvidenceArchiveError(
            "V7 formal results directory set is incomplete or contains extras"
        )
    for run_id in sorted(expected_ids):
        run_dir = formal_root / run_id
        marker = run_dir / "final_test.json"
        if not marker.is_file() or (run_dir / "final_test.in_progress.json").exists():
            raise V7EvidenceArchiveError(f"{run_id}: final-test marker is missing or interrupted")
        value = _read_json(marker, f"{run_id} final-test marker")
        if value.get("status") != "complete" or value.get("run_id") != run_id:
            raise V7EvidenceArchiveError(f"{run_id}: final-test marker is not complete")
    analysis_root = _inside(PROJECT_ROOT, artifacts["analysis_results"], "V7 analysis results")
    analysis_path = analysis_root / ANALYSIS_JSON
    accuracies_path = analysis_root / RAW_ACCURACIES_CSV
    differences_path = analysis_root / RAW_DIFFERENCES_CSV
    for path, label in (
        (accuracies_path, "V7 raw accuracies CSV"),
        (differences_path, "V7 paired differences CSV"),
    ):
        if not path.is_file():
            raise V7EvidenceArchiveError(f"{label} is missing: {path}")
    analysis = _read_json(analysis_path, "V7 analysis")
    if (
        analysis.get("protocol_version") != 7
        or analysis.get("protocol_hash") != stable_hash(protocol)
        or analysis.get("n_final_test_markers") != 48
        or analysis.get("source_git_commit") != source_commit
        or analysis.get("tracked_clean") is not True
        or analysis.get("formal_matrix_manifest_sha256")
        != matrix_manifest_sha256
        or analysis.get("freeze_manifest_sha256") != freeze_sha256
        or analysis.get("benchmark_receipt") != receipt_reference
        or analysis.get("benchmark_receipt_sha256") != receipt_sha256_after
        or analysis.get("benchmark_receipt_schema") != receipt.get("schema")
        or analysis.get("input_artifact_audit") != final_test_audit
        or analysis.get("cifar100_final_test_source_binding")
        != final_test_source_binding
    ):
        raise V7EvidenceArchiveError(
            "V7 analysis is not bound to the release, benchmark receipt, and all 48 final-test markers"
        )
    return {
        "freeze_manifest_sha256": freeze_sha256,
        "formal_matrix_manifest_sha256": matrix_manifest_sha256,
        "benchmark_receipt_sha256": receipt_sha256_after,
        "analysis_sha256": sha256_file(analysis_path),
        "training_environment_sha256": receipt["training_environment_sha256"],
        **final_test_source_binding,
    }


def create_archive(*, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    output = output.expanduser().resolve()
    if output.exists() or Path(f"{output}.sha256").exists():
        raise V7EvidenceArchiveError(
            "Refusing to overwrite an existing V7 evidence archive or sidecar"
        )
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise V7EvidenceArchiveError("Evidence archive output must end in .tar.gz")
    active = _active_v7_processes()
    if active:
        raise V7EvidenceArchiveError("V7 processes are still running; archive only after they stop")
    protocol = load_protocol(protocol_path)
    if int(protocol.get("protocol_version", 0)) != 7:
        raise V7EvidenceArchiveError("Evidence archive accepts only protocol v7")
    state = _validate_reportable_state(protocol_path, protocol)
    artifacts = artifact_paths_for_protocol(protocol)
    source_commit = _git("rev-parse", "HEAD")
    archive_state = _archive_state_payload(state, source_commit=source_commit)
    staging_parent = Path(tempfile.mkdtemp(prefix=".v7-evidence-", dir=output.parent))
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
            "V7_STATISTICAL_ANALYSIS_AUDIT.md",
            state["cifar100_split_manifest"],
            state["cifar100_source_provenance"],
        )
        for relative in relative_inputs:
            _copy_input(
                PROJECT_ROOT, staging, _inside(PROJECT_ROOT, relative, "V7 archive input"), records
            )
        source_bundle = staging / "source" / "ta-lif-msresnet-v7.bundle"
        source_bundle.parent.mkdir(parents=True, exist_ok=True)
        _git("bundle", "create", str(source_bundle), "HEAD")
        records.append(
            {
                "path": "source/ta-lif-msresnet-v7.bundle",
                "bytes": source_bundle.stat().st_size,
                "sha256": sha256_file(source_bundle),
            }
        )
        (staging / "ARCHIVE_README.txt").write_text(
            "V7 evidence archive. V5 evidence remains outside this archive.\n"
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
    except (OSError, ValueError, V7EvidenceArchiveError) as exc:
        print(f"V7_EVIDENCE_ARCHIVE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"V7_EVIDENCE_ARCHIVE_CREATED={result['output']}")
    print(f"V7_EVIDENCE_ARCHIVE_SHA256={result['sha256']}")
    print(f"V7_EVIDENCE_ARCHIVE_SOURCE={result['source_commit']}")
    print(f"V7_EVIDENCE_ARCHIVE_TRAINING_ENV={result['training_environment_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
