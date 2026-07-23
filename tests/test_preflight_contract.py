from __future__ import annotations

from pathlib import Path

import yaml

from talif_msresnet.preflight import check_protocol
import talif_msresnet.preflight as preflight_module


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "protocol.yaml"


def test_pilot_mode_does_not_bypass_scientific_data_checks(tmp_path: Path) -> None:
    report = check_protocol(
        PROTOCOL,
        mode="pilot",
        project_root=tmp_path,
        check_dependencies=False,
    )
    assert not any("protocol_status.frozen" in error for error in report.errors)
    assert not any("protocol_status.confirmed" in error for error in report.errors)
    assert not any("protocol_status.confirmations" in error for error in report.errors)
    assert any("CIFAR10-DVS" in error for error in report.errors)


def test_full_mode_still_requires_author_freeze(
    unfrozen_protocol_path: Path,
) -> None:
    report = check_protocol(
        unfrozen_protocol_path,
        mode="full",
        project_root=ROOT,
        check_dependencies=False,
    )
    assert not report.ok
    assert any("protocol_status.frozen" in error for error in report.errors)
    assert any("protocol_status.confirmed_by" in error for error in report.errors)
    assert any("protocol_status.confirmed_at" in error for error in report.errors)
    assert any("protocol_status.confirmations" in error for error in report.errors)
    assert any("unsigned and unfrozen" in warning for warning in report.warnings)
    assert not any("Phase B freeze manifest" in error for error in report.errors)


def test_full_mode_checks_phase_b_manifest_after_author_freeze(
    monkeypatch,
) -> None:
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    status = protocol["protocol_status"]
    status["frozen"] = True
    status["confirmed_by"] = "Yanzhao Deng; Peng Yan; Song Wang"
    status["confirmed_at"] = "2026-07-21T23:00:00+08:00"
    status["confirmations"] = {
        field: True for field in preflight_module.PROTOCOL_CONFIRMATION_FIELDS
    }
    monkeypatch.setattr(preflight_module, "load_protocol", lambda _path: protocol)

    def reject_manifest(**_kwargs):
        raise preflight_module.FreezeGateError("manifest missing")

    monkeypatch.setattr(preflight_module, "verify_formal_freeze", reject_manifest)
    report = check_protocol(PROTOCOL, mode="full", project_root=ROOT, check_dependencies=False)
    assert any("Phase B freeze manifest verification failed" in error for error in report.errors)
