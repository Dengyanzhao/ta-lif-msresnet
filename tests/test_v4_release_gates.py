from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import talif_msresnet.freeze as freeze_module
import talif_msresnet.preflight as preflight_module
from talif_msresnet.freeze import FreezeGateError


ROOT = Path(__file__).resolve().parents[1]
V4_PROTOCOL_PATH = ROOT / "configs" / "protocol_v4_talif_only.yaml"
V4_SIGNOFF_PATH = ROOT / "PREREGISTRATION_SIGNOFF_V4_TALIF_ONLY.md"


def _load_freeze_tool():
    path = ROOT / "scripts" / "create_freeze_manifest.py"
    spec = importlib.util.spec_from_file_location("create_freeze_manifest_v4_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_freeze_tool()


@pytest.mark.parametrize(
    "script_name",
    (
        "preflight.py",
        "generate_run_configs.py",
        "pilot_health_gate_v4.py",
        "run_matrix.py",
        "validate_v4_pilot.py",
        "create_freeze_manifest.py",
        "evaluate_checkpoints.py",
        "analyze_v4_results.py",
    ),
)
def test_v4_entrypoint_prefers_reviewed_checkout_over_stale_package(
    tmp_path: Path, script_name: str
) -> None:
    stale_root = tmp_path / "stale"
    stale_package = stale_root / "talif_msresnet"
    stale_package.mkdir(parents=True)
    (stale_package / "__init__.py").write_text("", encoding="utf-8")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(stale_root), str(ROOT / "src")))

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script_name), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _v4_protocol() -> dict[str, Any]:
    return copy.deepcopy(tool.load_protocol(V4_PROTOCOL_PATH))


def _skip_dvs_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight_module, "_check_dvs_provenance", lambda *_args: [])


def test_v4_pilot_preflight_requires_accountable_author_freeze(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _v4_protocol()
    protocol["protocol_status"] = {
        "frozen": False,
        "confirmed_by": None,
        "confirmed_at": None,
        "confirmations": {},
    }
    monkeypatch.setattr(preflight_module, "load_protocol", lambda _path: protocol)
    _skip_dvs_scan(monkeypatch)

    report = preflight_module.check_protocol(
        V4_PROTOCOL_PATH,
        mode="pilot",
        project_root=ROOT,
        check_dependencies=False,
    )

    assert any("protocol_status.frozen" in error for error in report.errors)
    assert any("protocol_status.confirmed_by" in error for error in report.errors)
    assert any("protocol_status.confirmed_at" in error for error in report.errors)
    assert any("protocol_status.confirmations" in error for error in report.errors)


def test_v4_pilot_uses_paired_talif_analysis_without_phase_b_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _v4_protocol()
    monkeypatch.setattr(preflight_module, "load_protocol", lambda _path: protocol)
    _skip_dvs_scan(monkeypatch)

    def unexpected_phase_b(**_kwargs):
        raise AssertionError("pilot preflight must not request the formal freeze")

    monkeypatch.setattr(preflight_module, "verify_formal_freeze", unexpected_phase_b)
    report = preflight_module.check_protocol(
        V4_PROTOCOL_PATH,
        mode="pilot",
        project_root=ROOT,
        check_dependencies=False,
    )

    assert not any("interaction_practical_threshold" in error for error in report.errors)
    assert not any("efficiency_noninferiority_margins" in error for error in report.errors)
    assert not any("primary_accuracy_test" in error for error in report.errors)


def test_v4_signoff_binds_only_the_accountable_author() -> None:
    protocol = _v4_protocol()
    signoff = V4_SIGNOFF_PATH.read_text(encoding="utf-8")

    confirmation = tool._validate_protocol_status(protocol)
    tool._validate_signoff(signoff, protocol)

    assert confirmation["responsible_author"] == "Yanzhao Deng"
    tampered = copy.deepcopy(protocol)
    tampered["protocol_status"]["confirmed_by"] += "; Peng Yan"
    with pytest.raises(tool.FreezeManifestError, match="unrecorded co-author approval"):
        tool._validate_protocol_status(tampered)


def test_v4_signoff_rejects_fabricated_coauthor_approval() -> None:
    protocol = _v4_protocol()
    signoff = V4_SIGNOFF_PATH.read_text(encoding="utf-8").replace(
        "- Independent approval from Peng Yan: not asserted in this record",
        "- Peng Yan - approval evidence/location: `invented`; date: `2026-07-29`",
    )

    with pytest.raises(tool.FreezeManifestError, match="avoid asserting approval from Peng Yan"):
        tool._validate_signoff(signoff, protocol)


def _v4_pass_report(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_class": "NON_REPORTABLE_V4_TALIF_ONLY_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V4_TALIF_ONLY_120_EPOCH_PILOTS_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "validated_at": "2026-07-29T12:00:00+00:00",
        "protocol_hash": tool._stable_hash(protocol),
        "acceptance_hash": tool._stable_hash(dict(protocol["pilot_acceptance"])),
        "datasets": {
            "cifar100": {
                "status": "PASS",
                "pass": True,
                "health_report_sha256": "1" * 64,
            },
            "cifar10dvs": {
                "status": "PASS",
                "pass": True,
                "health_report_sha256": "2" * 64,
            },
        },
        "integrity_failures": [],
        "threshold_failures": [],
    }


def _write_fake_v4_validator(project: Path, report: dict[str, Any]) -> Path:
    path = project / "scripts" / "validate_v4_pilot.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "REPORT = " + repr(report) + "\n\n"
        "def validate_pilot(**_kwargs):\n"
        "    return REPORT\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_v4_pilot_acceptance_is_revalidated_before_formal_release(
    tmp_path: Path,
) -> None:
    protocol = _v4_protocol()
    protocol_path = tmp_path / "configs" / "protocol_v4_talif_only.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 4\n", encoding="utf-8")
    report = _v4_pass_report(protocol)
    validation_path = tmp_path / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        tool.canonical_json(report) + "\n", encoding="utf-8", newline="\n"
    )
    validator_path = _write_fake_v4_validator(tmp_path, report)

    bound = tool._validate_v4_pilot_acceptance(protocol, protocol_path, tmp_path)

    assert bound["status"] == "PASS"
    assert bound["pass"] is True
    assert bound["validator"]["path"] == "scripts/validate_v4_pilot.py"
    assert bound["validator"]["sha256"] == tool._sha256_file(validator_path)
    assert set(bound["datasets"]) == {"cifar100", "cifar10dvs"}


def test_v4_pilot_nonpass_blocks_formal_release(tmp_path: Path) -> None:
    protocol = _v4_protocol()
    protocol_path = tmp_path / "configs" / "protocol_v4_talif_only.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 4\n", encoding="utf-8")
    report = _v4_pass_report(protocol)
    report["status"] = "FAIL"
    report["pass"] = False
    report["exit_code"] = 2
    report["datasets"]["cifar10dvs"]["status"] = "FAIL"
    report["datasets"]["cifar10dvs"]["pass"] = False
    report["threshold_failures"] = ["cifar10dvs pilot threshold failed"]
    validation_path = tmp_path / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(tool.canonical_json(report), encoding="utf-8")
    _write_fake_v4_validator(tmp_path, report)

    with pytest.raises(tool.FreezeManifestError, match="exact aggregate PASS"):
        tool._validate_v4_pilot_acceptance(protocol, protocol_path, tmp_path)


def test_runtime_v4_gate_rejects_manifest_without_pilot_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = tmp_path / "configs" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 4\n", encoding="utf-8")
    matrix_dir = tmp_path / "configs" / "matrix"
    matrix_dir.mkdir()
    stored = {
        "protocol": {
            "path": "configs/protocol.yaml",
            "canonical_sha256": "a" * 64,
        },
        "generated_matrix": {"directory": "configs/matrix"},
    }
    monkeypatch.setattr(
        freeze_module,
        "load_protocol",
        lambda _path: {"protocol_version": 4},
    )
    monkeypatch.setattr(
        freeze_module,
        "_load_manifest_tool",
        lambda _root: SimpleNamespace(verify_manifest=lambda **_kwargs: stored),
    )

    with pytest.raises(FreezeGateError, match="no bound aggregate pilot validation"):
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST.json",
        )


def test_v4_manifest_contract_lists_every_gate_source() -> None:
    assert set(tool.V4_REQUIRED_SOURCE_PATHS) == freeze_module.V4_GATE_SOURCE_PATHS
