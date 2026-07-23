from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_checkpoint import validate_frozen_diagnostic_settings  # noqa: E402
from run_diagnostics import _validate_diagnostic_protocol  # noqa: E402
from talif_msresnet.config import load_protocol  # noqa: E402


def test_diagnostic_wrapper_is_bound_to_frozen_analysis_settings(tmp_path: Path) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol.yaml")
    values = {
        "device": "cuda",
        "batch_size": 8,
        "probes": 8,
        "probe_seed": 20_260_719,
        "time_index": -1,
    }
    validate_frozen_diagnostic_settings(values, protocol["analysis"])
    path = tmp_path / "diagnostic.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    _validate_diagnostic_protocol(path, protocol)

    values["batch_size"] = 16
    path.write_text(json.dumps(values), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen study protocol"):
        _validate_diagnostic_protocol(path, protocol)


def test_diagnostic_wrapper_rejects_unknown_protocol_keys(tmp_path: Path) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol.yaml")
    path = tmp_path / "diagnostic.json"
    path.write_text(
        json.dumps(
            {
                "device": "cuda",
                "batch_size": 8,
                "probes": 8,
                "probe_seed": 20_260_719,
                "time_index": -1,
                "unregistered_override": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown diagnostic protocol keys"):
        _validate_diagnostic_protocol(path, protocol)
