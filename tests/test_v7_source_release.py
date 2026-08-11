from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import create_v7_instance_package as instance_package
import validate_v7_source_release as source_release


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_package(path: Path) -> None:
    prefix = f"{source_release.PACKAGE_ROOT_NAME}/"
    contents = {
        "README.txt": b"readme\n",
        "source/ta-lif-msresnet-v7.bundle": b"bundle\n",
        "payload/data/cifar100/cifar-100-python.tar.gz": b"archive\n",
        "payload/data/cifar100/cifar-100-python/train": b"train\n",
        "payload/data/cifar100/cifar-100-python/test": b"test\n",
        "payload/data/cifar100/cifar-100-python/meta": b"meta\n",
        "payload/data/manifests/cifar100_seed2024.json": b"{}\n",
        "payload/environment/CIFAR100_SOURCE_PROVENANCE.json": b"{}\n",
    }
    data_records = [
        {
            "path": name.removeprefix("payload/"),
            "bytes": len(value),
            "sha256": _sha256(value),
        }
        for name, value in sorted(contents.items())
        if name.startswith("payload/")
    ]
    contents["PACKAGE_MANIFEST.json"] = (
        json.dumps(
            {
                "schema": source_release.PACKAGE_SCHEMA,
                "source": {
                    "branch": "v7-mechanism-study",
                    "commit": "a" * 40,
                    "bundle": "source/ta-lif-msresnet-v7.bundle",
                    "bundle_sha256": _sha256(contents["source/ta-lif-msresnet-v7.bundle"]),
                },
                "protocol": {
                    "path": "configs/protocol_v7_mechanism.yaml",
                    "canonical_sha256": "b" * 64,
                    "file_sha256": "c" * 64,
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
            },
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    checksums = (
        "\n".join(f"{_sha256(value)}  {name}" for name, value in sorted(contents.items())) + "\n"
    )
    contents["SHA256SUMS.txt"] = checksums.encode("utf-8")
    with tarfile.open(path, mode="w:gz") as archive:
        for name, value in contents.items():
            info = tarfile.TarInfo(f"{prefix}{name}")
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{_sha256(path.read_bytes())}  {path.name}\n", encoding="utf-8"
    )


def test_offline_package_validation_checks_embedded_hashes(tmp_path: Path) -> None:
    package = tmp_path / "v7-input.tar.gz"
    _write_package(package)

    result = source_release._verify_package(package)

    assert result["path"] == package.resolve()
    assert result["sha256"] == _sha256(package.read_bytes())


def test_offline_package_validation_rejects_bad_sidecar(tmp_path: Path) -> None:
    package = tmp_path / "v7-input.tar.gz"
    _write_package(package)
    package.with_suffix(package.suffix + ".sha256").write_text(
        f"{'0' * 64}  {package.name}\n", encoding="utf-8"
    )

    with pytest.raises(source_release.V7SourceReleaseError, match="checksum sidecar"):
        source_release._verify_package(package)


def test_offline_package_validation_binds_current_source_commit(tmp_path: Path) -> None:
    package = tmp_path / "v7-input.tar.gz"
    _write_package(package)

    with pytest.raises(source_release.V7SourceReleaseError, match="source commit"):
        source_release._verify_package(package, expected_commit="d" * 40)


def test_offline_package_validation_binds_exact_release_branch(tmp_path: Path) -> None:
    package = tmp_path / "v7-input.tar.gz"
    _write_package(package)

    with pytest.raises(source_release.V7SourceReleaseError, match="source branch"):
        source_release._verify_package(package, expected_branch="v6-mechanism-study")


def test_source_release_rejects_wrong_checked_out_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
        ("branch", "--show-current"): "v6-mechanism-study",
    }
    monkeypatch.setattr(source_release, "_git", lambda *args: responses[args])

    with pytest.raises(source_release.V7SourceReleaseError, match="exact release branch"):
        source_release._assert_tracked_clean()


def test_instance_package_rejects_wrong_checked_out_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        instance_package,
        "_git",
        lambda *args, cwd: "v6-mechanism-study",
    )

    with pytest.raises(instance_package.V7InstancePackageError, match="exact release branch"):
        instance_package._assert_release_branch(tmp_path)
