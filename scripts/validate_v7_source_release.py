#!/usr/bin/env python3
"""Fail closed before packaging the committed V7 source for a blank GPU host."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for index, directory in enumerate((SRC_ROOT, SCRIPTS_ROOT)):
    value = str(directory)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(index, value)

from create_v7_instance_package import (
    PACKAGE_ROOT_NAME,
    PACKAGE_SCHEMA,
    V7_RELEASE_BRANCH,
    _required_data_paths,
    _verify_provenance,
)

from talif_msresnet.config import (
    RunConfig,
    artifact_paths_for_protocol,
    generate_run_matrix,
    load_protocol,
    load_run_config,
    validate_run_mapping,
)
from talif_msresnet.config_v7 import (
    V7_ARTIFACT_PATHS,
    generate_v7_pilot_matrix,
    validate_v7_cifar100_provenance_files,
    validate_v7_protocol,
)
from talif_msresnet.utils import sha256_file, stable_hash

V7_RELEASE_SOURCE_PATHS = (
    "PREREGISTRATION_SIGNOFF_V7_MECHANISM.md",
    "V6_MECHANISM_TERMINATION.md",
    "V7_STATISTICAL_ANALYSIS_AUDIT.md",
    "configs/protocol_v7_mechanism.yaml",
    "scripts/analyze_v7_results.py",
    "scripts/archive_v7_evidence.py",
    "scripts/create_freeze_manifest.py",
    "scripts/create_v7_instance_package.py",
    "scripts/evaluate_checkpoints.py",
    "scripts/export_v7_validation_batch.py",
    "scripts/generate_run_configs.py",
    "scripts/pilot_health_gate.py",
    "scripts/pilot_health_gate_v3.py",
    "scripts/pilot_health_gate_v4.py",
    "scripts/pilot_health_gate_v5.py",
    "scripts/pilot_health_gate_v6.py",
    "scripts/pilot_health_gate_v7.py",
    "scripts/preflight.py",
    "scripts/run_matrix.py",
    "scripts/run_v7_benchmarks.py",
    "scripts/validate_v3_pilot.py",
    "scripts/validate_v4_pilot.py",
    "scripts/validate_v5_pilot.py",
    "scripts/validate_v6_pilot.py",
    "scripts/validate_v7_pilot.py",
    "scripts/validate_v7_source_release.py",
    "scripts/v7_monitor_cn.py",
    "src/talif_msresnet/benchmark.py",
    "src/talif_msresnet/benchmark_v7.py",
    "src/talif_msresnet/config.py",
    "src/talif_msresnet/config_v6.py",
    "src/talif_msresnet/config_v7.py",
    "src/talif_msresnet/data.py",
    "src/talif_msresnet/diagnostics.py",
    "src/talif_msresnet/freeze.py",
    "src/talif_msresnet/models.py",
    "src/talif_msresnet/neurons.py",
    "src/talif_msresnet/ops.py",
    "src/talif_msresnet/pathing.py",
    "src/talif_msresnet/pilot_v3.py",
    "src/talif_msresnet/pilot_v4.py",
    "src/talif_msresnet/pilot_v5.py",
    "src/talif_msresnet/pilot_v6.py",
    "src/talif_msresnet/pilot_v7.py",
    "src/talif_msresnet/preflight.py",
    "src/talif_msresnet/train.py",
    "src/talif_msresnet/utils.py",
    "src/talif_msresnet/v7_statistics.py",
    "tests/test_v7_archive.py",
    "tests/test_v7_benchmark.py",
    "tests/test_v7_final_test_binding.py",
    "tests/test_v7_freeze_gates.py",
    "tests/test_v7_health_gate.py",
    "tests/test_v7_monitor_cn.py",
    "tests/test_v7_orchestration.py",
    "tests/test_v7_protocol.py",
    "tests/test_v7_provenance.py",
    "tests/test_v7_runtime.py",
    "tests/test_v7_source_release.py",
    "tests/test_v7_statistics.py",
    "tests/test_v7_training_orchestration.py",
    "tests/test_validate_v7_pilot.py",
    "V7_MECHANISM_5090_RUNBOOK.md",
)


class V7SourceReleaseError(RuntimeError):
    """Raised when a V7 source package would not be reproducibly executable."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / V7_ARTIFACT_PATHS["protocol"],
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Also verify the local CIFAR-100 inputs that will enter the offline package.",
    )
    parser.add_argument(
        "--package",
        type=Path,
        help="Also verify a package produced by create_v7_instance_package.py.",
    )
    return parser


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
        raise V7SourceReleaseError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _sha256_stream(handle: Any) -> str:
    digest = hashlib.sha256()
    while block := handle.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _assert_tracked_clean() -> str:
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise V7SourceReleaseError("V7 source release requires a completely clean worktree")
    branch = _git("branch", "--show-current")
    if branch != V7_RELEASE_BRANCH:
        raise V7SourceReleaseError(
            "V7 source release requires the exact release branch "
            f"{V7_RELEASE_BRANCH!r}, got {branch or '<detached HEAD>'!r}"
        )
    commit = _git("rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise V7SourceReleaseError("Git HEAD is not a full lowercase SHA-1 commit")
    return commit


def _assert_sources_tracked() -> None:
    missing: list[str] = []
    changed: list[str] = []
    for relative in V7_RELEASE_SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            missing.append(relative)
            continue
        changed_check = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=False,
        )
        if changed_check.returncode == 1:
            changed.append(relative)
        elif changed_check.returncode:
            raise V7SourceReleaseError(f"Cannot check committed source: {relative}")
    if missing:
        raise V7SourceReleaseError(
            "Required V7 source paths are untracked:\n- " + "\n- ".join(missing)
        )
    if changed:
        raise V7SourceReleaseError(
            "Required V7 source paths differ from HEAD:\n- " + "\n- ".join(changed)
        )


def _resolved_runs(protocol: Mapping[str, Any], stage: str) -> list[RunConfig]:
    raw = (
        generate_v7_pilot_matrix(protocol)
        if stage == "pilot"
        else generate_run_matrix(protocol)
    )
    return [validate_run_mapping(item, protocol) for item in raw]


def _validate_matrix(
    protocol: Mapping[str, Any], protocol_path: Path, stage: str
) -> dict[str, Any]:
    artifacts = artifact_paths_for_protocol(protocol)
    directory = (
        PROJECT_ROOT / artifacts["pilot_matrix" if stage == "pilot" else "formal_matrix"]
    ).resolve()
    if not directory.is_dir():
        raise V7SourceReleaseError(f"V7 {stage} matrix directory is missing: {directory}")
    expected = _resolved_runs(protocol, stage)
    expected_names = {
        "matrix_manifest.json",
        "run_manifest.csv",
        *(f"{run.runtime.run_id}.yaml" for run in expected),
    }
    actual_names = {item.name for item in directory.iterdir() if item.is_file()}
    nested = [item.name for item in directory.iterdir() if item.is_dir()]
    if actual_names != expected_names or nested:
        raise V7SourceReleaseError(
            f"V7 {stage} matrix is not an exact file set: "
            f"missing={sorted(expected_names - actual_names)[:5]}, "
            f"extra={sorted(actual_names - expected_names)[:5]}, nested={nested[:5]}"
        )
    rows: list[dict[str, Any]] = []
    for run in expected:
        path = directory / f"{run.runtime.run_id}.yaml"
        loaded = load_run_config(path, protocol_path)
        if loaded.as_dict() != run.as_dict():
            raise V7SourceReleaseError(f"V7 {stage} generated config differs: {path.name}")
        rows.append(
            {
                "run_id": run.runtime.run_id,
                "config_hash": run.config_hash,
                "config_file": path.name,
                "config_file_sha256": sha256_file(path),
            }
        )
    try:
        manifest = json.loads((directory / "matrix_manifest.json").read_text(encoding="utf-8"))
        with (directory / "run_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        raise V7SourceReleaseError(f"Cannot read V7 {stage} matrix manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise V7SourceReleaseError(f"V7 {stage} matrix_manifest.json is not an object")
    expected_hash = stable_hash(protocol)
    if manifest.get("protocol_hash") != expected_hash or manifest.get("run_count") != len(expected):
        raise V7SourceReleaseError(f"V7 {stage} matrix manifest has an invalid protocol binding")
    manifest_rows = manifest.get("runs")
    if not isinstance(manifest_rows, list):
        raise V7SourceReleaseError(f"V7 {stage} matrix manifest has no run list")
    manifest_projection = [
        {
            "run_id": item.get("run_id"),
            "config_hash": item.get("config_hash"),
            "config_file": item.get("config_file"),
            "config_file_sha256": item.get("config_file_sha256"),
        }
        for item in manifest_rows
        if isinstance(item, Mapping)
    ]
    if manifest_projection != rows:
        raise V7SourceReleaseError(f"V7 {stage} matrix manifest differs from its generated configs")
    csv_projection = [
        {
            "run_id": row.get("run_id"),
            "config_hash": row.get("config_hash"),
            "config_file": row.get("config_file"),
            "config_file_sha256": row.get("config_file_sha256"),
        }
        for row in csv_rows
    ]
    if csv_projection != rows:
        raise V7SourceReleaseError(f"V7 {stage} run_manifest.csv differs from generated configs")
    return {
        "directory": directory,
        "run_count": len(expected),
        "matrix_hash": manifest.get("matrix_hash"),
    }


def _assert_pristine_artifact_paths(protocol: Mapping[str, Any]) -> None:
    artifacts = protocol["artifact_paths"]
    acceptance = protocol["pilot_acceptance"]
    paths = {
        "health receipt": acceptance["attempt_receipt"],
        "health report": acceptance["health_output"],
        "pilot results": acceptance["pilot_output_root"],
        "pilot plan": acceptance["pilot_plan"],
        "pilot validation": acceptance["validation_output"],
        "formal results": artifacts["formal_results"],
        "freeze manifest": artifacts["freeze_manifest"],
        "benchmark results": artifacts["benchmark_results"],
        "analysis results": artifacts["analysis_results"],
    }
    present = [
        f"{label}: {PROJECT_ROOT / str(relative)}"
        for label, relative in paths.items()
        if (PROJECT_ROOT / str(relative)).exists()
    ]
    if present:
        raise V7SourceReleaseError(
            "Pre-rental V7 source must not contain experiment artifacts:\n- " + "\n- ".join(present)
        )


def _package_data_records(
    *,
    project_root: Path | None,
    data_paths: Sequence[Path] | None,
) -> dict[str, dict[str, Any]] | None:
    if project_root is None or data_paths is None:
        return None
    return {
        path.relative_to(project_root).as_posix(): {
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in data_paths
    }


def _verify_package(
    path: Path,
    *,
    expected_commit: str | None = None,
    expected_branch: str | None = None,
    expected_protocol: Mapping[str, Any] | None = None,
    expected_protocol_path: Path | None = None,
    expected_data_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    package = path.resolve()
    sidecar = Path(f"{package}.sha256")
    if not package.is_file() or not sidecar.is_file():
        raise V7SourceReleaseError("V7 package or its .sha256 sidecar is missing")
    expected_line = sidecar.read_text(encoding="utf-8").strip().split()
    if (
        len(expected_line) != 2
        or expected_line[1] != package.name
        or expected_line[0] != sha256_file(package)
    ):
        raise V7SourceReleaseError("V7 package checksum sidecar differs from the archive")
    try:
        with tarfile.open(package, mode="r:gz") as archive:
            file_members = [member for member in archive.getmembers() if member.isfile()]
            members = {member.name: member for member in file_members}
            if len(members) != len(file_members):
                raise V7SourceReleaseError("V7 package contains duplicate file entries")
            prefix = f"{PACKAGE_ROOT_NAME}/"
            if any(not member.name.startswith(prefix) for member in file_members):
                raise V7SourceReleaseError("V7 package contains a file outside its root directory")
            required = {
                f"{prefix}README.txt",
                f"{prefix}PACKAGE_MANIFEST.json",
                f"{prefix}SHA256SUMS.txt",
                f"{prefix}source/ta-lif-msresnet-v7.bundle",
                f"{prefix}payload/data/cifar100/cifar-100-python.tar.gz",
                f"{prefix}payload/data/cifar100/cifar-100-python/train",
                f"{prefix}payload/data/cifar100/cifar-100-python/test",
                f"{prefix}payload/data/cifar100/cifar-100-python/meta",
                f"{prefix}payload/data/manifests/cifar100_seed2024.json",
                f"{prefix}payload/environment/CIFAR100_SOURCE_PROVENANCE.json",
            }
            missing = sorted(required - set(members))
            if missing:
                raise V7SourceReleaseError(
                    "V7 package is missing required entries: " + ", ".join(missing)
                )
            sums_handle = archive.extractfile(members[f"{prefix}SHA256SUMS.txt"])
            if sums_handle is None:
                raise V7SourceReleaseError("V7 package has no readable SHA256SUMS.txt")
            expected_hashes = {
                line.rsplit("  ", 1)[1]: line.rsplit("  ", 1)[0]
                for line in sums_handle.read().decode("utf-8").splitlines()
                if "  " in line
            }
            for member_name, member in members.items():
                relative = member_name.removeprefix(prefix)
                if relative == "SHA256SUMS.txt":
                    continue
                stream = archive.extractfile(member)
                if stream is None or expected_hashes.get(relative) != _sha256_stream(stream):
                    raise V7SourceReleaseError(f"V7 package checksum mismatch: {relative}")
            manifest_handle = archive.extractfile(members[f"{prefix}PACKAGE_MANIFEST.json"])
            if manifest_handle is None:
                raise V7SourceReleaseError("V7 package has no readable PACKAGE_MANIFEST.json")
            try:
                manifest = json.loads(manifest_handle.read().decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise V7SourceReleaseError(
                    f"V7 package manifest is not valid JSON: {exc}"
                ) from exc
            if not isinstance(manifest, Mapping):
                raise V7SourceReleaseError("V7 package manifest must be an object")
            required_manifest_keys = {
                "schema",
                "source",
                "protocol",
                "payload_root",
                "data_files",
                "excluded_artifact_classes",
            }
            if set(manifest) != required_manifest_keys or manifest.get("schema") != PACKAGE_SCHEMA:
                raise V7SourceReleaseError("V7 package manifest has an unexpected schema")
            source = manifest.get("source")
            protocol_record = manifest.get("protocol")
            if not isinstance(source, Mapping) or not isinstance(protocol_record, Mapping):
                raise V7SourceReleaseError("V7 package manifest source/protocol binding is malformed")
            bundle_relative = "source/ta-lif-msresnet-v7.bundle"
            if (
                set(source) != {"branch", "commit", "bundle", "bundle_sha256"}
                or source.get("bundle") != bundle_relative
                or source.get("bundle_sha256")
                != expected_hashes.get(bundle_relative)
            ):
                raise V7SourceReleaseError("V7 package source bundle binding is invalid")
            if expected_commit is not None and source.get("commit") != expected_commit:
                raise V7SourceReleaseError("V7 package source commit differs from the current release")
            if expected_branch is not None and source.get("branch") != expected_branch:
                raise V7SourceReleaseError("V7 package source branch differs from the current release")
            if (
                set(protocol_record) != {"path", "canonical_sha256", "file_sha256"}
                or manifest.get("payload_root") != "payload"
            ):
                raise V7SourceReleaseError("V7 package protocol/payload binding is invalid")
            if expected_protocol is not None:
                if protocol_record.get("canonical_sha256") != stable_hash(expected_protocol):
                    raise V7SourceReleaseError("V7 package protocol hash differs from the current release")
                if expected_protocol_path is None:
                    raise V7SourceReleaseError("V7 package protocol path binding is missing")
                if (
                    protocol_record.get("path")
                    != expected_protocol_path.relative_to(PROJECT_ROOT).as_posix()
                    or protocol_record.get("file_sha256") != sha256_file(expected_protocol_path)
                ):
                    raise V7SourceReleaseError(
                        "V7 package protocol file differs from the current release"
                    )
            data_files = manifest.get("data_files")
            if not isinstance(data_files, list):
                raise V7SourceReleaseError("V7 package data file manifest is malformed")
            observed_data_records: dict[str, dict[str, Any]] = {}
            for record in data_files:
                if not isinstance(record, Mapping) or set(record) != {"path", "bytes", "sha256"}:
                    raise V7SourceReleaseError("V7 package data file manifest has an invalid row")
                relative = record.get("path")
                if not isinstance(relative, str) or not relative.startswith("data/") and not relative.startswith("environment/"):
                    raise V7SourceReleaseError("V7 package data file path is invalid")
                if relative in observed_data_records:
                    raise V7SourceReleaseError("V7 package data file manifest has a duplicate path")
                if (
                    not isinstance(record.get("bytes"), int)
                    or isinstance(record.get("bytes"), bool)
                    or record["bytes"] < 0
                    or not isinstance(record.get("sha256"), str)
                    or len(record["sha256"]) != 64
                ):
                    raise V7SourceReleaseError("V7 package data file manifest has an invalid hash row")
                if expected_hashes.get(f"payload/{relative}") != record["sha256"]:
                    raise V7SourceReleaseError(
                        f"V7 package data file hash is not bound to payload: {relative}"
                    )
                observed_data_records[relative] = {
                    "bytes": record["bytes"],
                    "sha256": record["sha256"],
                }
            if expected_data_records is not None and observed_data_records != dict(
                expected_data_records
            ):
                raise V7SourceReleaseError(
                    "V7 package data inputs differ from the verified local inputs"
                )
            excluded = manifest.get("excluded_artifact_classes")
            if excluded != [
                "health",
                "pilot",
                "formal_training",
                "validation_benchmark",
                "final_test",
                "analysis",
            ]:
                raise V7SourceReleaseError("V7 package artifact exclusion declaration is invalid")
    except (OSError, tarfile.TarError, UnicodeError) as exc:
        raise V7SourceReleaseError(f"Cannot verify V7 package: {exc}") from exc
    return {"path": package, "sha256": sha256_file(package)}


def validate_source_release(
    *, protocol_path: Path, require_data: bool, package: Path | None
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    expected_protocol = (PROJECT_ROOT / V7_ARTIFACT_PATHS["protocol"]).resolve()
    if protocol_path != expected_protocol:
        raise V7SourceReleaseError(
            f"V7 release requires the canonical protocol path: {expected_protocol}"
        )
    protocol = validate_v7_protocol(load_protocol(protocol_path))
    commit = _assert_tracked_clean()
    _assert_sources_tracked()
    pilot = _validate_matrix(protocol, protocol_path, "pilot")
    formal = _validate_matrix(protocol, protocol_path, "formal")
    _assert_pristine_artifact_paths(protocol)
    data_paths: tuple[Path, ...] | None = None
    if require_data:
        data_paths = _required_data_paths(PROJECT_ROOT)
        _verify_provenance(PROJECT_ROOT, data_paths)
        validate_v7_cifar100_provenance_files(protocol, project_root=PROJECT_ROOT)
    package_summary = (
        _verify_package(
            package,
            expected_commit=commit,
            expected_branch=V7_RELEASE_BRANCH,
            expected_protocol=protocol,
            expected_protocol_path=protocol_path,
            expected_data_records=_package_data_records(
                project_root=PROJECT_ROOT if data_paths is not None else None,
                data_paths=data_paths,
            ),
        )
        if package is not None
        else None
    )
    return {
        "commit": commit,
        "protocol_hash": stable_hash(protocol),
        "pilot_matrix": {**pilot, "directory": str(pilot["directory"])},
        "formal_matrix": {**formal, "directory": str(formal["directory"])},
        "local_data_verified": require_data,
        "package": package_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_source_release(
            protocol_path=args.protocol,
            require_data=args.require_data,
            package=args.package,
        )
    except (OSError, ValueError, V7SourceReleaseError) as exc:
        print(f"V7_SOURCE_RELEASE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"V7_SOURCE_RELEASE_PASS commit={report['commit']}")
    print(f"V7_PROTOCOL_HASH={report['protocol_hash']}")
    print(
        "V7_MATRICES_PASS "
        f"pilot={report['pilot_matrix']['run_count']} formal={report['formal_matrix']['run_count']}"
    )
    if report["local_data_verified"]:
        print("V7_LOCAL_CIFAR100_PACKAGE_INPUTS_PASS")
    if report["package"]:
        print(f"V7_OFFLINE_PACKAGE_PASS sha256={report['package']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
