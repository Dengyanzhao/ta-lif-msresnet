from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from talif_msresnet.pathing import artifact_path_reference  # noqa: E402


def test_repository_path_is_posix_relative(tmp_path: Path) -> None:
    root = tmp_path / "project"
    artifact = root / "results" / "run" / "best.pt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"checkpoint")

    assert artifact_path_reference(artifact, root) == "results/run/best.pt"


def test_external_file_uses_content_hash_without_host_path(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    artifact = tmp_path / "external" / "best.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"checkpoint")
    expected = hashlib.sha256(b"checkpoint").hexdigest()

    reference = artifact_path_reference(artifact, root)

    assert reference == f"external:file-sha256:{expected}"
    assert str(tmp_path) not in reference
    assert "/" not in reference and "\\" not in reference


def test_external_directory_uses_explicit_opaque_path_id(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "external-results"
    external.mkdir()

    first = artifact_path_reference(external, root)
    second = artifact_path_reference(external, root)

    assert first == second
    assert first.startswith("external:path-sha256:")
    assert len(first.removeprefix("external:path-sha256:")) == 64
    assert str(tmp_path) not in first


def test_domain_external_identifier_is_path_free(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    external = tmp_path / "checkpoint.pt"

    assert artifact_path_reference(
        external,
        root,
        external_identifier="checkpoint-sha256:abc",
    ) == "external:checkpoint-sha256:abc"
    with pytest.raises(ValueError, match="path-free"):
        artifact_path_reference(external, root, external_identifier="outside/file")
