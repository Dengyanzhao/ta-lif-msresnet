from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from talif_msresnet.freeze import FreezeGateError, verify_formal_freeze  # noqa: E402

from test_freeze_manifest import _complete_project, _create  # noqa: E402


def test_formal_gate_accepts_bound_protocol_and_matrix(tmp_path: Path) -> None:
    project, commit, matrix_dir = _complete_project(tmp_path)
    _create(project, commit, matrix_dir)

    manifest = verify_formal_freeze(
        project_root=project,
        protocol_path=project / "configs" / "protocol.yaml",
        matrix_dir=matrix_dir,
    )

    assert manifest["generated_matrix"]["run_count"] == 40


def test_formal_gate_rejects_a_different_matrix_directory(tmp_path: Path) -> None:
    project, commit, matrix_dir = _complete_project(tmp_path)
    _create(project, commit, matrix_dir)
    other = project / "configs" / "other"
    other.mkdir()

    with pytest.raises(FreezeGateError, match="configuration directory"):
        verify_formal_freeze(
            project_root=project,
            protocol_path=project / "configs" / "protocol.yaml",
            matrix_dir=other,
        )
