from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import talif_msresnet.freeze as freeze_module
from talif_msresnet.config_v6 import V6_ACTIVE_CONDITIONS
from talif_msresnet.freeze import FreezeGateError


def _load_freeze_tool():
    path = ROOT / "scripts" / "create_freeze_manifest.py"
    spec = importlib.util.spec_from_file_location("create_freeze_manifest_v6_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_freeze_tool()
V6_PROTOCOL_PATH = ROOT / "configs" / "protocol_v6_mechanism.yaml"


def _v6_protocol() -> dict[str, Any]:
    return copy.deepcopy(tool.load_protocol(V6_PROTOCOL_PATH))


def _v6_pass_report(protocol: dict[str, Any]) -> dict[str, Any]:
    acceptance = protocol["pilot_acceptance"]
    environment = "e" * 64
    return {
        "schema_version": 1,
        "protocol_version": 6,
        "artifact_class": "NON_REPORTABLE_V6_MECHANISM_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V6_SIX_CONDITION_120_EPOCH_PILOT_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "validated_at": "2026-08-11T12:00:00+08:00",
        "protocol_path": "configs/protocol_v6_mechanism.yaml",
        "protocol_file_sha256": "1" * 64,
        "protocol_hash": tool._stable_hash(protocol),
        "acceptance_hash": tool._stable_hash(dict(acceptance)),
        "config_dir": protocol["artifact_paths"]["pilot_matrix"],
        "matrix_manifest_sha256": "2" * 64,
        "run_manifest_csv_sha256": "3" * 64,
        "required_epochs": 120,
        "thresholds": {
            "minimum_best_validation_accuracy": 0.20,
            "required_gradient_coverage": 0.95,
            "late_window_epochs": 10,
            "minimum_late_to_best_ratio": 0.75,
        },
        "training_environment_sha256": environment,
        "failure_action": acceptance["run_handling"]["failure_action"],
        "fallback": None,
        "datasets": {
            "cifar100": {
                "status": "PASS",
                "pass": True,
                "seed": acceptance["pilot_seed"],
                "health_seed": acceptance["health_seed"],
                "pilot_seed": acceptance["pilot_seed"],
                "block_hash": "4" * 64,
                "results_root": acceptance["pilot_output_root"],
                "health_report_path": acceptance["health_output"],
                "health_report_sha256": "5" * 64,
                "attempt_receipt_path": acceptance["attempt_receipt"],
                "attempt_receipt_sha256": "6" * 64,
                "pilot_plan_path": acceptance["pilot_plan"],
                "pilot_plan_sha256": "7" * 64,
                "environment_sha256": environment,
                "shared_weight_sha256": "8" * 64,
                "split_manifest_sha256": "9" * 64,
                "runs": {
                    condition: {
                        "run_id": f"E6_cifar100_d20_t6_{condition}_s{acceptance['pilot_seed']}",
                        "condition": condition,
                        "training_environment_sha256": environment,
                        "integrity_failures": [],
                        "threshold_failures": [],
                    }
                    for condition in V6_ACTIVE_CONDITIONS
                },
                "integrity_failures": [],
                "threshold_failures": [],
            }
        },
        "integrity_failures": [],
        "threshold_failures": [],
    }


def _write_fake_v6_validator(project: Path, report: dict[str, Any]) -> Path:
    path = project / "scripts" / "validate_v6_pilot.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "REPORT = " + repr(report) + "\n\n"
        "def validate_pilot(**_kwargs):\n"
        "    return REPORT\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_v6_validation_fixture(
    tmp_path: Path,
    protocol: dict[str, Any],
    report: dict[str, Any],
) -> tuple[Path, Path]:
    project = tmp_path / "project"
    project.mkdir()
    protocol_path = project / "configs" / "protocol_v6_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 6\n", encoding="utf-8")
    validation_path = project / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        json.dumps(report, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_fake_v6_validator(project, report)
    return project, protocol_path


def test_v6_signoff_uses_the_accountable_author_contract() -> None:
    protocol = _v6_protocol()
    signoff = (ROOT / "PREREGISTRATION_SIGNOFF_V6_MECHANISM.md").read_text(
        encoding="utf-8"
    )

    confirmation = tool._validate_protocol_status(protocol)
    tool._validate_signoff(signoff, protocol)

    assert confirmation["responsible_author"] == "Yanzhao Deng"


def test_v6_pilot_acceptance_is_revalidated_and_bound(
    tmp_path: Path,
) -> None:
    protocol = _v6_protocol()
    report = _v6_pass_report(protocol)
    project, protocol_path = _write_v6_validation_fixture(tmp_path, protocol, report)

    bound = tool._validate_v6_pilot_acceptance(protocol, protocol_path, project)

    assert bound["status"] == "PASS"
    assert bound["conditions"] == list(V6_ACTIVE_CONDITIONS)
    assert bound["training_environment_sha256"] == "e" * 64
    assert set(bound["datasets"]) == {"cifar100"}
    assert bound["validator"]["path"] == "scripts/validate_v6_pilot.py"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["datasets"]["cifar100"]["runs"]["M3"].update(
                {"integrity_failures": ["tampered"]}
            ),
            "cifar100.runs.M3",
        ),
        (
            lambda report: report.update({"training_environment_sha256": "bad"}),
            "training_environment_sha256",
        ),
        (
            lambda report: report["datasets"].pop("cifar100"),
            "datasets",
        ),
    ],
)
def test_v6_pilot_acceptance_rejects_incomplete_evidence(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    protocol = _v6_protocol()
    report = _v6_pass_report(protocol)
    mutate(report)
    project, protocol_path = _write_v6_validation_fixture(tmp_path, protocol, report)

    with pytest.raises(tool.FreezeManifestError, match=message):
        tool._validate_v6_pilot_acceptance(protocol, protocol_path, project)


def _v6_freeze_record(protocol: dict[str, Any]) -> dict[str, Any]:
    report = _v6_pass_report(protocol)
    evidence = report["datasets"]["cifar100"]
    validation = {
        "artifact_class": report["artifact_class"],
        "status": "PASS",
        "pass": True,
        "path": "results/pilot/v6/validation.json",
        "sha256": "a" * 64,
        "protocol_hash": report["protocol_hash"],
        "acceptance_hash": report["acceptance_hash"],
        "validated_at": report["validated_at"],
        "training_environment_sha256": report["training_environment_sha256"],
        "conditions": list(V6_ACTIVE_CONDITIONS),
        "validator": {"path": "scripts/validate_v6_pilot.py", "sha256": "b" * 64},
        "datasets": {
            "cifar100": {
                key: evidence[key]
                for key in (
                    "status",
                    "pass",
                    "health_seed",
                    "pilot_seed",
                    "block_hash",
                    "environment_sha256",
                    "shared_weight_sha256",
                    "split_manifest_sha256",
                    "health_report_sha256",
                    "attempt_receipt_sha256",
                    "pilot_plan_sha256",
                )
            }
        },
    }
    return {
        "protocol": {
            "path": "configs/protocol_v6_mechanism.yaml",
            "canonical_sha256": report["protocol_hash"],
        },
        "generated_matrix": {"directory": "configs/v6-formal"},
        "pilot_validation": validation,
        "gate_sources": {
            path: {
                "path": path,
                "file_sha256": (
                    "b" * 64 if path == "scripts/validate_v6_pilot.py" else "c" * 64
                ),
            }
            for path in freeze_module.V6_GATE_SOURCE_PATHS
        },
    }


def _install_v6_runtime_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    protocol = _v6_protocol()
    protocol_path = tmp_path / "configs" / "protocol_v6_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 6\n", encoding="utf-8")
    matrix_dir = tmp_path / "configs" / "v6-formal"
    matrix_dir.mkdir()
    monkeypatch.setattr(freeze_module, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        freeze_module,
        "_load_manifest_tool",
        lambda _root: SimpleNamespace(verify_manifest=lambda **_kwargs: stored),
    )
    return protocol_path, matrix_dir, protocol


def test_runtime_v6_formal_gate_requires_exact_pilot_and_gate_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _v6_protocol()
    stored = _v6_freeze_record(protocol)
    protocol_path, matrix_dir, _ = _install_v6_runtime_freeze(
        tmp_path, monkeypatch, stored
    )

    assert freeze_module.verify_formal_freeze(
        project_root=tmp_path,
        protocol_path=protocol_path,
        matrix_dir=matrix_dir,
        manifest_path=tmp_path / "FREEZE_MANIFEST_V6.json",
    ) is stored


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda stored: stored["pilot_validation"].pop(
                "training_environment_sha256"
            ),
            "no valid pilot-bound",
        ),
        (
            lambda stored: stored["pilot_validation"].update({"conditions": ["M0"]}),
            "ordered six-condition",
        ),
        (
            lambda stored: stored["pilot_validation"]["datasets"]["cifar100"].update(
                {"pilot_plan_sha256": "not-a-hash"}
            ),
            "inconsistent six-condition",
        ),
        (
            lambda stored: stored["gate_sources"].pop("scripts/validate_v6_pilot.py"),
            "complete v6 gate source set",
        ),
    ],
)
def test_runtime_v6_formal_gate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    protocol = _v6_protocol()
    stored = _v6_freeze_record(protocol)
    mutate(stored)
    protocol_path, matrix_dir, _ = _install_v6_runtime_freeze(
        tmp_path, monkeypatch, stored
    )

    with pytest.raises(FreezeGateError, match=message):
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST_V6.json",
        )


def test_v6_manifest_and_runtime_gate_share_the_complete_source_set() -> None:
    assert set(tool.V6_REQUIRED_SOURCE_PATHS) == freeze_module.V6_GATE_SOURCE_PATHS
