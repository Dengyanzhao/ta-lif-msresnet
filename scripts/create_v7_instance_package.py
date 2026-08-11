#!/usr/bin/env python3
"""Create the offline source-and-data package for a blank V7 RTX 5090 instance.

The package intentionally contains only the committed V7 source history and
the CIFAR-100 training/validation inputs.  It never includes pilot, formal,
benchmark, or final-test artifacts, which must be created on the new instance
under the frozen execution gates.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import load_protocol
from talif_msresnet.config_v7 import (
    V7_ARTIFACT_PATHS,
    validate_v7_cifar100_provenance_files,
    validate_v7_protocol,
)
from talif_msresnet.utils import sha256_file, stable_hash

PACKAGE_ROOT_NAME = "ta-lif-msresnet-v7-instance-input"
PACKAGE_SCHEMA = "ta-lif-msresnet-v7-instance-package-v1"
V7_RELEASE_BRANCH = "v7-mechanism-study"


class V7InstancePackageError(RuntimeError):
    """Raised when an offline V7 package cannot be safely created."""


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
        default=PROJECT_ROOT / V7_ARTIFACT_PATHS["protocol"],
    )
    return parser


def _git(*args: str, cwd: Path = PROJECT_ROOT) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise V7InstancePackageError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _assert_release_branch(project_root: Path) -> str:
    branch = _git("branch", "--show-current", cwd=project_root)
    if branch != V7_RELEASE_BRANCH:
        raise V7InstancePackageError(
            "V7 source package requires the exact release branch "
            f"{V7_RELEASE_BRANCH!r}, got {branch or '<detached HEAD>'!r}"
        )
    return branch


def _inside(root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise V7InstancePackageError(f"{label} escapes the repository: {resolved}") from exc
    return resolved


def _required_data_paths(project_root: Path) -> tuple[Path, ...]:
    relative_paths = (
        "data/cifar100/cifar-100-python.tar.gz",
        "data/cifar100/cifar-100-python/train",
        "data/cifar100/cifar-100-python/test",
        "data/cifar100/cifar-100-python/meta",
        "data/manifests/cifar100_seed2024.json",
        "environment/CIFAR100_SOURCE_PROVENANCE.json",
    )
    paths = tuple(
        _inside(project_root, project_root / item, "V7 package input") for item in relative_paths
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise V7InstancePackageError(
            "Required CIFAR-100 package input is missing:\n- " + "\n- ".join(missing)
        )
    return paths


def _verify_provenance(project_root: Path, paths: Iterable[Path]) -> None:
    provenance_path = project_root / "environment" / "CIFAR100_SOURCE_PROVENANCE.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V7InstancePackageError(f"Cannot read CIFAR-100 provenance: {exc}") from exc
    if not isinstance(provenance, dict):
        raise V7InstancePackageError("CIFAR-100 provenance must be a JSON object")
    archive = provenance.get("local_archive_binding")
    extracted = provenance.get("extracted_binding")
    split = provenance.get("development_split_binding")
    if (
        not isinstance(archive, dict)
        or not isinstance(extracted, dict)
        or not isinstance(split, dict)
    ):
        raise V7InstancePackageError("CIFAR-100 provenance is incomplete")
    expected: dict[str, str] = {
        "data/cifar100/cifar-100-python.tar.gz": str(archive.get("sha256", "")),
        "data/cifar100/cifar-100-python/train": str(
            extracted.get("files", {}).get("train", {}).get("sha256", "")
        ),
        "data/cifar100/cifar-100-python/test": str(
            extracted.get("files", {}).get("test", {}).get("sha256", "")
        ),
        "data/cifar100/cifar-100-python/meta": str(
            extracted.get("files", {}).get("meta", {}).get("sha256", "")
        ),
        "data/manifests/cifar100_seed2024.json": str(split.get("file_sha256", "")),
    }
    requested = {path.relative_to(project_root).as_posix() for path in paths}
    for relative, expected_hash in expected.items():
        if relative not in requested:
            continue
        if len(expected_hash) != 64:
            raise V7InstancePackageError(f"CIFAR-100 provenance lacks a valid hash for {relative}")
        observed = sha256_file(project_root / relative)
        if observed != expected_hash:
            raise V7InstancePackageError(
                f"CIFAR-100 provenance mismatch for {relative}: {observed} != {expected_hash}"
            )


def _assert_fresh_v7_artifact_state(project_root: Path, protocol: dict[str, Any]) -> None:
    artifacts = protocol["artifact_paths"]
    forbidden = (
        artifacts["formal_results"],
        artifacts["pilot_results"],
        artifacts["benchmark_results"],
        artifacts["analysis_results"],
        artifacts["freeze_manifest"],
    )
    present = [
        str(project_root / relative) for relative in forbidden if (project_root / relative).exists()
    ]
    if present:
        raise V7InstancePackageError(
            "Refusing to package a non-pristine V7 artifact state:\n- " + "\n- ".join(present)
        )


def _copy_data_inputs(
    project_root: Path, staging_payload: Path, paths: Iterable[Path]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in paths:
        relative = source.relative_to(project_root)
        destination = staging_payload / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": int(source.stat().st_size),
                "sha256": sha256_file(source),
            }
        )
    return records


def _write_readme(path: Path) -> None:
    path.write_text(
        "This directory is the payload of a V7 blank-instance package.\n\n"
        "1. Verify SHA256SUMS.txt before extracting or using its contents.\n"
        "2. Clone source/ta-lif-msresnet-v7.bundle into /hy-tmp/ta-lif-msresnet "
        f"and switch to {V7_RELEASE_BRANCH}.\n"
        "3. Copy data/ and environment/CIFAR100_SOURCE_PROVENANCE.json into that clone.\n"
        "4. Follow V7_MECHANISM_5090_RUNBOOK.md from the committed source.\n\n"
        "The package excludes all V7 experiment outputs by design.\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256sum_lines(root: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "SHA256SUMS.txt":
            continue
        lines.append(f"{sha256_file(path)}  {relative}")
    return lines


def _write_archive(staging_root: Path, output: Path) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with tarfile.open(temporary, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            archive.add(staging_root, arcname=PACKAGE_ROOT_NAME, recursive=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def create_package(*, project_root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    protocol_path = _inside(project_root, protocol_path, "V7 protocol")
    output = output.expanduser().resolve()
    if output.exists():
        raise V7InstancePackageError(f"Refusing to overwrite existing package: {output}")
    sidecar = Path(f"{output}.sha256")
    if sidecar.exists():
        raise V7InstancePackageError(f"Refusing to overwrite existing checksum sidecar: {sidecar}")
    try:
        output.relative_to(project_root)
    except ValueError:
        pass
    else:
        raise V7InstancePackageError("Package output must be outside the repository")
    if output.suffixes[-2:] != [".tar", ".gz"]:
        raise V7InstancePackageError("V7 instance package output must end in .tar.gz")
    if not output.parent.is_dir():
        raise V7InstancePackageError(f"Package parent directory is missing: {output.parent}")

    protocol = validate_v7_protocol(load_protocol(protocol_path))
    validate_v7_cifar100_provenance_files(protocol, project_root=project_root)
    expected_protocol = (project_root / V7_ARTIFACT_PATHS["protocol"]).resolve()
    if protocol_path != expected_protocol:
        raise V7InstancePackageError(f"V7 protocol path must be {expected_protocol}")
    if _git("status", "--porcelain=v1", cwd=project_root):
        raise V7InstancePackageError(
            "V7 source package requires a clean tracked and untracked worktree"
        )
    branch = _assert_release_branch(project_root)
    commit = _git("rev-parse", "HEAD", cwd=project_root)
    _assert_fresh_v7_artifact_state(project_root, protocol)
    data_paths = _required_data_paths(project_root)
    _verify_provenance(project_root, data_paths)

    temporary_parent = Path(tempfile.mkdtemp(prefix=".v7-instance-package-", dir=output.parent))
    try:
        staging_root = temporary_parent / PACKAGE_ROOT_NAME
        source_dir = staging_root / "source"
        payload_dir = staging_root / "payload"
        source_dir.mkdir(parents=True)
        payload_dir.mkdir(parents=True)
        bundle = source_dir / "ta-lif-msresnet-v7.bundle"
        _git("bundle", "create", str(bundle), "HEAD", f"refs/heads/{branch}", cwd=project_root)
        _git("bundle", "verify", str(bundle), cwd=project_root)
        data_records = _copy_data_inputs(project_root, payload_dir, data_paths)
        _write_readme(staging_root / "README.txt")
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "source": {
                "branch": branch,
                "commit": commit,
                "bundle": "source/ta-lif-msresnet-v7.bundle",
                "bundle_sha256": sha256_file(bundle),
            },
            "protocol": {
                "path": protocol_path.relative_to(project_root).as_posix(),
                "canonical_sha256": stable_hash(protocol),
                "file_sha256": sha256_file(protocol_path),
            },
            "payload_root": "payload",
            "data_files": data_records,
            "excluded_artifact_classes": [
                "health",
                "pilot",
                "formal_training",
                "validation_benchmark",
                "final_test",
                "analysis",
            ],
        }
        (staging_root / "PACKAGE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging_root / "SHA256SUMS.txt").write_text(
            "\n".join(_sha256sum_lines(staging_root)) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_archive(staging_root, output)
    finally:
        shutil.rmtree(temporary_parent, ignore_errors=True)
    archive_sha256 = sha256_file(output)
    sidecar.write_text(f"{archive_sha256}  {output.name}\n", encoding="utf-8", newline="\n")
    return {
        "output": str(output),
        "archive_sha256": archive_sha256,
        "checksum_sidecar": str(sidecar),
        "commit": commit,
        "branch": branch,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = create_package(
            project_root=PROJECT_ROOT,
            protocol_path=args.protocol.resolve(),
            output=args.output,
        )
    except (OSError, ValueError, V7InstancePackageError) as exc:
        print(f"V7_INSTANCE_PACKAGE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"V7_INSTANCE_PACKAGE_CREATED={result['output']}")
    print(f"V7_INSTANCE_PACKAGE_SHA256={result['archive_sha256']}")
    print(f"V7_INSTANCE_PACKAGE_SIDECAR={result['checksum_sidecar']}")
    print(f"V7_INSTANCE_PACKAGE_SOURCE={result['branch']}@{result['commit']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
