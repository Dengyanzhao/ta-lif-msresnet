from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_checkpoint import _portable_artifact_reference  # noqa: E402


def test_diagnostic_external_artifact_reference_has_no_host_path(tmp_path: Path) -> None:
    external = tmp_path / "private-data" / "batch.pt"
    external.parent.mkdir()

    reference = _portable_artifact_reference(external, "abc123", "input-file")

    assert reference == "external:input-file-sha256:abc123"
    assert str(tmp_path) not in reference
    assert "/" not in reference and "\\" not in reference


def test_diagnostic_repository_artifact_reference_remains_relative() -> None:
    artifact = ROOT / "results" / "run" / "best.pt"

    assert _portable_artifact_reference(artifact, "abc123", "checkpoint") == (
        "results/run/best.pt"
    )
