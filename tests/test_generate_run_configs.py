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


def test_v2_pilot_generation_is_one_ordered_c1_c4_seed_block(tmp_path: Path) -> None:
    output = tmp_path / "v2-pilot-generated"

    runs = generate(ROOT / "configs" / "protocol_v2_pilot.yaml", output)
    manifest = json.loads((output / "matrix_manifest.json").read_text(encoding="utf-8"))
    csv_path = output / "run_manifest.csv"
    csv_bytes = csv_path.read_bytes()

    assert len(runs) == 4
    assert manifest["run_count"] == 4
    assert manifest["seeds"] == [77]
    assert [run["condition"] for run in runs] == ["C1", "C2", "C3", "C4"]
    assert len(list(output.glob("*.yaml"))) == 4
    assert manifest["protocol"] == "configs/protocol_v2_pilot.yaml"
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in csv_bytes
    assert csv_bytes.count(b"\n") == 5

    generate(ROOT / "configs" / "protocol_v2_pilot.yaml", output)
    assert csv_path.read_bytes() == csv_bytes


def test_v2r2_generation_is_one_isolated_seed88_block(tmp_path: Path) -> None:
    output = tmp_path / "v2r2-seed88-generated"
    protocol_path = ROOT / "configs" / "protocol_v2r2_seed88_of80_e120.yaml"

    runs = generate(protocol_path, output)
    manifest = json.loads((output / "matrix_manifest.json").read_text(encoding="utf-8"))
    csv_path = output / "run_manifest.csv"
    csv_bytes = csv_path.read_bytes()

    assert len(runs) == 4
    assert manifest["run_count"] == 4
    assert manifest["seeds"] == [88]
    assert [run["condition"] for run in runs] == ["C1", "C2", "C3", "C4"]
    assert [run["run_id"] for run in runs] == [
        "E1_cifar100_d20_t6_C1_s88",
        "E1_cifar100_d20_t6_C2_s88",
        "E1_cifar100_d20_t6_C3_s88",
        "E1_cifar100_d20_t6_C4_s88",
    ]
    assert len(list(output.glob("*.yaml"))) == 4
    assert manifest["protocol"] == "configs/protocol_v2r2_seed88_of80_e120.yaml"
    assert csv_bytes.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in csv_bytes
    assert csv_bytes.count(b"\n") == 5

    generate(protocol_path, output)
    assert csv_path.read_bytes() == csv_bytes
