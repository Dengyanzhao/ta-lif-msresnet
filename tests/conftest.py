from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def unfrozen_protocol_path(tmp_path: Path) -> Path:
    protocol = yaml.safe_load(
        (ROOT / "configs" / "protocol.yaml").read_text(encoding="utf-8")
    )
    status = protocol["protocol_status"]
    status["frozen"] = False
    status["confirmed_by"] = None
    status["confirmed_at"] = None
    status["confirmations"] = {
        field: False for field in status["confirmations"]
    }

    path = tmp_path / "protocol.yaml"
    path.write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return path
