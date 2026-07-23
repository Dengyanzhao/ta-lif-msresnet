from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_run_configs import generate  # noqa: E402


def test_matrix_manifest_uses_portable_protocol_path(tmp_path: Path) -> None:
    output = tmp_path / "generated"

    runs = generate(ROOT / "configs" / "protocol.yaml", output)
    manifest = json.loads((output / "matrix_manifest.json").read_text(encoding="utf-8"))

    assert len(runs) == 40
    assert manifest["run_count"] == 40
    assert len(list(output.glob("*.yaml"))) == 40
    assert manifest["protocol"] == "configs/protocol.yaml"
    assert not Path(manifest["protocol"]).is_absolute()
