from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight import build_parser  # noqa: E402


def test_preflight_cli_exposes_every_supported_mode() -> None:
    parser = build_parser()

    for mode in ("smoke", "pilot", "full", "final-test"):
        assert parser.parse_args(["--mode", mode]).mode == mode
