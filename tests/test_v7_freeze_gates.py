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
from talif_msresnet.pilot_v7 import (
    V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS,
    V7_HEALTH_RECOVERY_ATTEMPT_PATH,
    V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
    V7_HEALTH_RECOVERY_BASE_COMMIT,
    V7_HEALTH_RECOVERY_CORRECTED_FIELD,
    V7_HEALTH_RECOVERY_HEALTH_PATH,
    V7_HEALTH_RECOVERY_HEALTH_SHA256,
    V7_HEALTH_RECOVERY_PROTOCOL_HASH,
    V7_HEALTH_RECOVERY_RELEASE_RECORD,
    V7_HEALTH_RECOVERY_SCHEMA,
    V7_HEALTH_RECOVERY_SEAL_COMMIT,
    V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT,
    V7_PILOT_DEVICE_RECOVERY_ALLOWED_CHANGED_PATHS,
    V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE,
    V7_PILOT_DEVICE_RECOVERY_FROZEN_DEVICE,
    V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION,
    V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256,
    V7_PILOT_DEVICE_RECOVERY_OUTPUT,
    V7_PILOT_DEVICE_RECOVERY_PROTOCOL_HASH,
    V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD,
    V7_PILOT_DEVICE_RECOVERY_SCHEMA,
)


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


def _v7_recovery(*, recovery_commit: str = V7_HEALTH_RECOVERY_SEAL_COMMIT) -> dict[str, Any]:
    return {
        "schema": V7_HEALTH_RECOVERY_SCHEMA,
        "base_health_commit": V7_HEALTH_RECOVERY_BASE_COMMIT,
        "implementation_commit": "d" * 40,
        "implementation_tree": "a" * 40,
        "recovery_commit": recovery_commit,
        "health_report_path": V7_HEALTH_RECOVERY_HEALTH_PATH,
        "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
        "attempt_receipt_path": V7_HEALTH_RECOVERY_ATTEMPT_PATH,
        "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
        "protocol_hash": V7_HEALTH_RECOVERY_PROTOCOL_HASH,
        "corrected_field": V7_HEALTH_RECOVERY_CORRECTED_FIELD,
        "split_source_fingerprint": V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT,
        "allowed_changed_paths": list(V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS),
        "observed_changes": [
            f"M\t{path}" for path in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS
        ],
        "implementation_file_sha256": {
            path: "9" * 64 for path in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS
        },
        "release_record": V7_HEALTH_RECOVERY_RELEASE_RECORD,
        "record_sha256": "c" * 64,
        "release_record_sha256": "b" * 64,
        "tracked_clean": True,
    }


def _v7_device_recovery(*, recovery_commit: str = "f" * 40) -> dict[str, Any]:
    return {
        "schema": V7_PILOT_DEVICE_RECOVERY_SCHEMA,
        "base_health_recovery_seal_commit": V7_HEALTH_RECOVERY_SEAL_COMMIT,
        "pilot_execution_commit": V7_HEALTH_RECOVERY_SEAL_COMMIT,
        "implementation_commit": "d" * 40,
        "implementation_tree": "a" * 40,
        "recovery_commit": recovery_commit,
        "original_validation_path": V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION,
        "original_validation_sha256": V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256,
        "recovery_output_path": V7_PILOT_DEVICE_RECOVERY_OUTPUT,
        "protocol_hash": V7_PILOT_DEVICE_RECOVERY_PROTOCOL_HASH,
        "frozen_runtime_device": V7_PILOT_DEVICE_RECOVERY_FROZEN_DEVICE,
        "accepted_execution_device": V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE,
        "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
        "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
        "allowed_changed_paths": list(V7_PILOT_DEVICE_RECOVERY_ALLOWED_CHANGED_PATHS),
        "observed_changes": [
            f"M\t{path}" for path in V7_PILOT_DEVICE_RECOVERY_ALLOWED_CHANGED_PATHS
        ],
        "implementation_file_sha256": {
            path: "9" * 64 for path in V7_PILOT_DEVICE_RECOVERY_ALLOWED_CHANGED_PATHS
        },
        "release_record": V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD,
        "record_sha256": "c" * 64,
        "release_record_sha256": "b" * 64,
        "tracked_clean": True,
    }


def _v7_pass_report(
    protocol: dict[str, Any], *, protocol_file_sha256: str
) -> dict[str, Any]:
    acceptance = protocol["pilot_acceptance"]
    environment = "e" * 64
    recovery = _v7_recovery()
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
        "health_compatibility_recovery": recovery,
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
                "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
                "attempt_receipt_path": acceptance["attempt_receipt"],
                "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
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
                "health_compatibility_recovery": recovery,
            }
        },
        "integrity_failures": [],
        "threshold_failures": [],
    }


def _v7_legacy_invalid_report(recovered: dict[str, Any]) -> dict[str, Any]:
    """Model the immutable pilot verdict before the device-equivalence repair."""

    legacy = copy.deepcopy(recovered)
    device_failures = [
        "v7 run runtime.device must be frozen as 'auto', got 'cuda:0'"
        for _ in V7_ACTIVE_CONDITIONS
    ]
    aggregate_failures = [
        f"{condition}: {failure}"
        for condition, failure in zip(V7_ACTIVE_CONDITIONS, device_failures)
    ] + [
        "training environments are not identical across the six conditions: []",
        "shared-weight hashes are not identical across the six conditions: []",
        "split-manifest hashes are not identical across the six conditions: []",
    ]
    legacy.update(
        {
            "status": "INVALID",
            "pass": False,
            "decision": "BLOCK_V7_AND_INVESTIGATE_PILOT_EVIDENCE_INTEGRITY",
            "exit_code": 2,
            "training_environment_sha256": None,
            "integrity_failures": aggregate_failures,
            "threshold_failures": [],
        }
    )
    dataset = legacy["datasets"]["cifar100"]
    dataset.update(
        {
            "status": "INVALID",
            "pass": False,
            "environment_sha256": None,
            "shared_weight_sha256": None,
            "split_manifest_sha256": None,
            "integrity_failures": aggregate_failures,
            "threshold_failures": [],
        }
    )
    for condition, failure in zip(V7_ACTIVE_CONDITIONS, device_failures):
        run = dataset["runs"][condition]
        run["training_environment_sha256"] = None
        run["integrity_failures"] = [failure]
        run["threshold_failures"] = []
    return legacy


def _v7_device_recovery_sidecar(
    recovered: dict[str, Any], legacy: dict[str, Any]
) -> dict[str, Any]:
    device_recovery = _v7_device_recovery()
    return {
        "schema_version": 1,
        "artifact_class": (
            "NON_REPORTABLE_V7_PILOT_VALIDATION_DEVICE_COMPATIBILITY_RECOVERY"
        ),
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "RECOVER_V7_PILOT_PASS_AFTER_DEVICE_EXECUTION_EQUIVALENCE_FIX",
        "exit_code": 0,
        "validated_at": recovered["validated_at"],
        "protocol_hash": recovered["protocol_hash"],
        "acceptance_hash": recovered["acceptance_hash"],
        "pilot_execution_commit": V7_HEALTH_RECOVERY_SEAL_COMMIT,
        "recovery_validator_commit": device_recovery["recovery_commit"],
        "release_delta": device_recovery,
        "recovery_validator": {
            "path": "scripts/validate_v7_pilot.py",
            "sha256": "a" * 64,
        },
        "original_validation": {
            "path": V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION,
            "sha256": V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256,
            "status": "INVALID",
            "decision": legacy["decision"],
            "legacy_reproduction": "EXACT_EXCEPT_VALIDATED_AT",
            "legacy_reproduction_sha256": "b" * 64,
        },
        "health_compatibility_recovery": recovered[
            "health_compatibility_recovery"
        ],
        "recovered_validation": recovered,
        "output": V7_PILOT_DEVICE_RECOVERY_OUTPUT,
    }


def _write_fake_v7_validator(
    project: Path, *, recovered: dict[str, Any], legacy: dict[str, Any]
) -> Path:
    path = project / "scripts" / "validate_v7_pilot.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "RECOVERED = " + repr(recovered) + "\n"
        "LEGACY = " + repr(legacy) + "\n\n"
        "def validate_pilot(*, allow_device_execution_equivalence=False, **_kwargs):\n"
        "    return RECOVERED if allow_device_execution_equivalence else LEGACY\n",
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
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    project = tmp_path / "project"
    project.mkdir()
    protocol_path = project / "configs" / "protocol_v7_mechanism.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 7\n", encoding="utf-8", newline="\n")
    recovered = stored_report or _v7_pass_report(
        protocol, protocol_file_sha256=tool._sha256_file(protocol_path)
    )
    legacy = _v7_legacy_invalid_report(recovered)
    sidecar = _v7_device_recovery_sidecar(recovered, legacy)
    validation_path = project / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        json.dumps(legacy, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    recovery_path = project / V7_PILOT_DEVICE_RECOVERY_OUTPUT
    recovery_path.write_text(
        json.dumps(sidecar, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_fake_v7_validator(
        project,
        recovered=current_report or recovered,
        legacy=legacy,
    )
    return project, protocol_path, recovered, sidecar


def _patch_v7_device_recovery_bindings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    project: Path,
    protocol: dict[str, Any],
    recovered: dict[str, Any],
    device_recovery: dict[str, Any] | None = None,
    health_recovery: dict[str, Any] | None = None,
) -> None:
    validation_path = (
        project / protocol["pilot_acceptance"]["validation_output"]
    ).resolve()
    real_sha256 = tool._sha256_file
    monkeypatch.setattr(
        tool,
        "_sha256_file",
        lambda path: (
            V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256
            if Path(path).resolve() == validation_path
            else real_sha256(path)
        ),
    )
    monkeypatch.setattr(
        tool,
        "validate_health_recovery_release",
        lambda *_args, **_kwargs: health_recovery
        or recovered["health_compatibility_recovery"],
    )
    monkeypatch.setattr(
        tool,
        "validate_pilot_device_recovery_release",
        lambda *_args, **_kwargs: device_recovery or _v7_device_recovery(),
    )


def test_v7_signoff_uses_the_accountable_author_contract() -> None:
    protocol = _v7_protocol()
    signoff = (ROOT / "PREREGISTRATION_SIGNOFF_V7_MECHANISM.md").read_text(
        encoding="utf-8"
    )

    confirmation = tool._validate_protocol_status(protocol)
    tool._validate_signoff(signoff, protocol)

    assert confirmation["responsible_author"] == "Yanzhao Deng"


def test_v7_pilot_acceptance_is_revalidated_and_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _v7_protocol()
    project, protocol_path, report, sidecar = _write_v7_validation_fixture(
        tmp_path, protocol
    )
    _patch_v7_device_recovery_bindings(
        monkeypatch,
        project=project,
        protocol=protocol,
        recovered=report,
    )

    bound = tool._validate_v7_pilot_acceptance(protocol, protocol_path, project)

    assert bound["status"] == "PASS"
    assert bound["conditions"] == list(V7_ACTIVE_CONDITIONS)
    assert bound["protocol_file_sha256"] == report["protocol_file_sha256"]
    assert bound["pilot_matrix_manifest_sha256"] == "2" * 64
    assert bound["pilot_run_manifest_csv_sha256"] == "3" * 64
    assert bound["health_compatibility_recovery"] == report[
        "health_compatibility_recovery"
    ]
    assert bound["artifact_class"] == sidecar["artifact_class"]
    assert bound["device_compatibility_recovery"] == sidecar["release_delta"]
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


def test_v7_pilot_acceptance_rejects_modified_recovered_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _v7_protocol()
    project, protocol_path, report, _sidecar = _write_v7_validation_fixture(
        tmp_path, protocol
    )
    current = copy.deepcopy(report)
    current["matrix_manifest_sha256"] = "a" * 64
    _write_fake_v7_validator(
        project,
        recovered=current,
        legacy=_v7_legacy_invalid_report(report),
    )
    _patch_v7_device_recovery_bindings(
        monkeypatch,
        project=project,
        protocol=protocol,
        recovered=report,
    )

    with pytest.raises(
        tool.FreezeManifestError,
        match="differs from the current pilot artifacts",
    ):
        tool._validate_v7_pilot_acceptance(protocol, protocol_path, project)


def test_v7_pilot_acceptance_rejects_unsealed_recovery_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _v7_protocol()
    project, protocol_path, report, _sidecar = _write_v7_validation_fixture(
        tmp_path, protocol
    )
    verified = copy.deepcopy(report["health_compatibility_recovery"])
    verified["record_sha256"] = "0" * 64
    _patch_v7_device_recovery_bindings(
        monkeypatch,
        project=project,
        protocol=protocol,
        recovered=report,
        health_recovery=verified,
    )

    with pytest.raises(tool.FreezeManifestError, match="differs from the sealed"):
        tool._validate_v7_pilot_acceptance(protocol, protocol_path, project)


def _v7_freeze_record(protocol: dict[str, Any]) -> dict[str, Any]:
    protocol_hash = tool._stable_hash(protocol)
    environment = "e" * 64
    recovery = _v7_recovery()
    device_recovery = _v7_device_recovery()
    pilot = {
        "artifact_class": (
            "NON_REPORTABLE_V7_PILOT_VALIDATION_DEVICE_COMPATIBILITY_RECOVERY"
        ),
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
        "health_compatibility_recovery": recovery,
        "device_compatibility_recovery": device_recovery,
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
                "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
                "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
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
                    "b" * 64
                    if path
                    in {
                        "scripts/validate_v7_pilot.py",
                        V7_HEALTH_RECOVERY_RELEASE_RECORD,
                        V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD,
                    }
                    else "c" * 64
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
        (
            lambda stored: stored["pilot_validation"][
                "health_compatibility_recovery"
            ].update({"health_report_sha256": "0" * 64}),
            "sealed health compatibility recovery",
        ),
        (
            lambda stored: stored["pilot_validation"][
                "health_compatibility_recovery"
            ].update({"corrected_field": "source.other"}),
            "sealed health compatibility recovery",
        ),
        (
            lambda stored: stored["pilot_validation"][
                "device_compatibility_recovery"
            ].update({"accepted_execution_device": "cuda:1"}),
            "device compatibility recovery",
        ),
        (
            lambda stored: stored["pilot_validation"][
                "device_compatibility_recovery"
            ].update({"frozen_runtime_device": "cuda:0"}),
            "device compatibility recovery",
        ),
        (
            lambda stored: stored["pilot_validation"][
                "device_compatibility_recovery"
            ].update({"original_validation_sha256": "0" * 64}),
            "device compatibility recovery",
        ),
        (
            lambda stored: stored["gate_sources"][
                V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD
            ].update({"file_sha256": "0" * 64}),
            "device compatibility recovery",
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
