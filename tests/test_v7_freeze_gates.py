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
from talif_msresnet.config_v7 import V7_ACTIVE_CONDITIONS, V7_FORMAL_SEEDS
from talif_msresnet.freeze import FreezeGateError


def _load_freeze_tool():
    path = ROOT / "scripts" / "create_freeze_manifest.py"
    spec = importlib.util.spec_from_file_location(
        "create_freeze_manifest_v7_tests", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_freeze_tool()
V7_PROTOCOL_PATH = ROOT / "configs" / "protocol_v7_mechanism.yaml"


def _v7_protocol() -> dict[str, Any]:
    return copy.deepcopy(tool.load_protocol(V7_PROTOCOL_PATH))


def _v7_pass_report(
    protocol: dict[str, Any], *, protocol_file_sha256: str
) -> dict[str, Any]:
    acceptance = protocol["pilot_acceptance"]
    environment = "e" * 64
    return {
        "schema_version": 1,
        "protocol_version": 7,
        "artifact_class": "NON_REPORTABLE_V7_MECHANISM_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V7_SIX_CONDITION_120_EPOCH_PILOT_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "validated_at": "2026-08-11T12:00:00+08:00",
        "protocol_path": "configs/protocol_v7_mechanism.yaml",
        "protocol_file_sha256": protocol_file_sha256,
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
                        "run_id": (
                            f"E9_cifar100_d20_t6_{condition}_s"
                            f"{acceptance['pilot_seed']}"
                        ),
                        "condition": condition,
                        "training_environment_sha256": environment,
                        "integrity_failures": [],
                        "threshold_failures": [],
                    }
                    for condition in V7_ACTIVE_CONDITIONS
                },
                "integrity_failures": [],
                "threshold_failures": [],
            }
        },
        "integrity_failures": [],
        "threshold_failures": [],
    }


def _write_fake_v7_validator(project: Path, report: dict[str, Any]) -> Path:
    path = project / "scripts" / "validate_v7_pilot.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "REPORT = " + repr(report) + "\n\n"
        "def validate_pilot(**_kwargs):\n"
        "    return REPORT\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def _write_v7_validation_fixture(
    tmp_path: Path,
    protocol: dict[str, Any],
    *,
    stored_report: dict[str, Any] | None = None,
    current_report: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    project = tmp_path / "project"
    project.mkdir()
    protocol_path = project / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8", newline="\n")
    report = stored_report or _v7_pass_report(
        protocol, protocol_file_sha256=tool._sha256_file(protocol_path)
    )
    validation_path = project / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        json.dumps(report, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_fake_v7_validator(project, current_report or report)
    return project, protocol_path, report


def test_v7_signoff_uses_the_accountable_author_contract() -> None:
    protocol = _v7_protocol()
    signoff = (ROOT / "PREREGISTRATION_SIGNOFF_V7_MECHANISM.md").read_text(
        encoding="utf-8"
    )

    confirmation = tool._validate_protocol_status(protocol)
    tool._validate_signoff(signoff, protocol)

    assert confirmation["responsible_author"] == "Yanzhao Deng"


def test_v7_pilot_acceptance_is_revalidated_and_bound(tmp_path: Path) -> None:
    protocol = _v7_protocol()
    project, protocol_path, report = _write_v7_validation_fixture(tmp_path, protocol)

    bound = tool._validate_v7_pilot_acceptance(protocol, protocol_path, project)

    assert bound["status"] == "PASS"
    assert bound["conditions"] == list(V7_ACTIVE_CONDITIONS)
    assert bound["protocol_file_sha256"] == report["protocol_file_sha256"]
    assert bound["pilot_matrix_manifest_sha256"] == "2" * 64
    assert bound["pilot_run_manifest_csv_sha256"] == "3" * 64
    assert set(bound["datasets"]) == {"cifar100"}


def test_v7_pilot_acceptance_rejects_missing_validation(tmp_path: Path) -> None:
    protocol = _v7_protocol()
    project = tmp_path / "project"
    project.mkdir()
    protocol_path = project / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8")

    with pytest.raises(tool.FreezeManifestError, match="V7 pilot validation"):
        tool._validate_v7_pilot_acceptance(protocol, protocol_path, project)


def test_v7_pilot_acceptance_rejects_modified_validation(tmp_path: Path) -> None:
    protocol = _v7_protocol()
    project = tmp_path / "project"
    project.mkdir()
    protocol_path = project / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8", newline="\n")
    current = _v7_pass_report(
        protocol, protocol_file_sha256=tool._sha256_file(protocol_path)
    )
    stored = copy.deepcopy(current)
    stored["matrix_manifest_sha256"] = "a" * 64
    validation_path = project / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(json.dumps(stored) + "\n", encoding="utf-8")
    _write_fake_v7_validator(project, current)

    with pytest.raises(tool.FreezeManifestError, match="differs from the current"):
        tool._validate_v7_pilot_acceptance(protocol, protocol_path, project)


def _v7_freeze_record(protocol: dict[str, Any]) -> dict[str, Any]:
    protocol_hash = tool._stable_hash(protocol)
    environment = "e" * 64
    pilot = {
        "artifact_class": "NON_REPORTABLE_V7_MECHANISM_PILOT_ACCEPTANCE",
        "status": "PASS",
        "pass": True,
        "path": protocol["pilot_acceptance"]["validation_output"],
        "sha256": "a" * 64,
        "protocol_file_sha256": "1" * 64,
        "protocol_hash": protocol_hash,
        "acceptance_hash": tool._stable_hash(dict(protocol["pilot_acceptance"])),
        "validated_at": "2026-08-11T12:00:00+08:00",
        "training_environment_sha256": environment,
        "conditions": list(V7_ACTIVE_CONDITIONS),
        "pilot_matrix_manifest_sha256": "2" * 64,
        "pilot_run_manifest_csv_sha256": "3" * 64,
        "validator": {"path": "scripts/validate_v7_pilot.py", "sha256": "b" * 64},
        "datasets": {
            "cifar100": {
                "status": "PASS",
                "pass": True,
                "health_seed": protocol["pilot_acceptance"]["health_seed"],
                "pilot_seed": protocol["pilot_acceptance"]["pilot_seed"],
                "block_hash": "4" * 64,
                "environment_sha256": environment,
                "shared_weight_sha256": "5" * 64,
                "split_manifest_sha256": "6" * 64,
                "health_report_sha256": "7" * 64,
                "attempt_receipt_sha256": "8" * 64,
                "pilot_plan_sha256": "9" * 64,
            }
        },
    }
    return {
        "repository": {"freeze_commit": "f" * 40},
        "protocol": {
            "path": "configs/protocol_v7_mechanism.yaml",
            "file_sha256": "1" * 64,
            "canonical_sha256": protocol_hash,
            "version": 7,
        },
        "generated_matrix": {
            "directory": "configs/v7-formal",
            "matrix_hash": "c" * 64,
            "run_count": 48,
            "matrix_manifest": {"path": "matrix_manifest.json", "sha256": "d" * 64},
            "run_manifest": {"path": "run_manifest.csv", "sha256": "e" * 64},
            "config_count": 48,
            "config_set_sha256": "f" * 64,
            "conditions": list(V7_ACTIVE_CONDITIONS),
            "seeds": list(V7_FORMAL_SEEDS),
        },
        "pilot_validation": pilot,
        "gate_sources": {
            path: {
                "path": path,
                "file_sha256": (
                    "b" * 64 if path == "scripts/validate_v7_pilot.py" else "c" * 64
                ),
            }
            for path in freeze_module.V7_GATE_SOURCE_PATHS
        },
    }


def _install_v7_runtime_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stored: dict[str, Any],
) -> tuple[Path, Path]:
    protocol = _v7_protocol()
    protocol_path = tmp_path / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8")
    matrix_dir = tmp_path / "configs" / "v7-formal"
    matrix_dir.mkdir()
    monkeypatch.setattr(freeze_module, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        freeze_module,
        "_load_manifest_tool",
        lambda _root: SimpleNamespace(verify_manifest=lambda **_kwargs: stored),
    )
    return protocol_path, matrix_dir


def test_runtime_v7_formal_gate_requires_exact_pilot_matrix_and_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _v7_protocol()
    stored = _v7_freeze_record(protocol)
    protocol_path, matrix_dir = _install_v7_runtime_freeze(
        tmp_path, monkeypatch, stored
    )

    assert freeze_module.verify_formal_freeze(
        project_root=tmp_path,
        protocol_path=protocol_path,
        matrix_dir=matrix_dir,
        manifest_path=tmp_path / "FREEZE_MANIFEST_V7.json",
    ) is stored


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda stored: stored["generated_matrix"].update({"run_count": 47}),
            "exact 48-run formal matrix",
        ),
        (
            lambda stored: stored["pilot_validation"].update({"conditions": ["M0"]}),
            "ordered six-condition",
        ),
        (
            lambda stored: stored["gate_sources"].pop("scripts/validate_v7_pilot.py"),
            "complete v7 gate source set",
        ),
        (
            lambda stored: stored["gate_sources"][
                "src/talif_msresnet/config_v7.py"
            ].update({"file_sha256": "modified"}),
            "malformed gate source hashes",
        ),
        (
            lambda stored: stored["pilot_validation"].update(
                {"pilot_matrix_manifest_sha256": "bad"}
            ),
            "pilot validation/matrix hash",
        ),
    ],
)
def test_runtime_v7_formal_gate_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    protocol = _v7_protocol()
    stored = _v7_freeze_record(protocol)
    mutate(stored)
    protocol_path, matrix_dir = _install_v7_runtime_freeze(
        tmp_path, monkeypatch, stored
    )

    with pytest.raises(FreezeGateError, match=message):
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST_V7.json",
        )


def test_v6_pilot_evidence_cannot_authorize_v7(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _v7_protocol()
    stored = _v7_freeze_record(protocol)
    stored["pilot_validation"]["artifact_class"] = (
        "NON_REPORTABLE_V6_MECHANISM_PILOT_ACCEPTANCE"
    )
    protocol_path, matrix_dir = _install_v7_runtime_freeze(
        tmp_path, monkeypatch, stored
    )

    with pytest.raises(FreezeGateError, match="V7 six-condition pilot artifact"):
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST_V7.json",
        )


def test_runtime_v7_rejects_missing_or_unverifiable_freeze(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _v7_protocol()
    protocol_path = tmp_path / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8")
    monkeypatch.setattr(freeze_module, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        freeze_module,
        "_load_manifest_tool",
        lambda _root: SimpleNamespace(
            verify_manifest=lambda **_kwargs: (_ for _ in ()).throw(
                FileNotFoundError("freeze absent")
            )
        ),
    )

    with pytest.raises(FreezeGateError, match="freeze absent"):
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            manifest_path=tmp_path / "FREEZE_MANIFEST_V7.json",
        )


def test_v7_manifest_and_runtime_gate_share_the_complete_source_set() -> None:
    assert set(tool.V7_REQUIRED_SOURCE_PATHS) == freeze_module.V7_GATE_SOURCE_PATHS
