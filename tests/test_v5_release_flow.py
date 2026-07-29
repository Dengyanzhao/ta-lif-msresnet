from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import talif_msresnet.freeze as freeze_module
import talif_msresnet.train as train_module
from talif_msresnet.config import V5_ARTIFACT_PATHS, load_protocol
from talif_msresnet.freeze import FreezeGateError

ROOT = Path(__file__).resolve().parents[1]
V5_PROTOCOL_PATH = ROOT / "configs" / "protocol_v5_talif_only.yaml"


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"v5_release_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


freeze_tool = _load_script("create_freeze_manifest.py")
evaluate_tool = _load_script("evaluate_checkpoints.py")


def _v5_protocol() -> dict[str, Any]:
    return copy.deepcopy(load_protocol(V5_PROTOCOL_PATH))


def _v5_pass_report(protocol: dict[str, Any]) -> dict[str, Any]:
    environment_sha256 = "e" * 64
    return {
        "schema_version": 1,
        "artifact_class": "NON_REPORTABLE_V5_TALIF_ONLY_PILOT_ACCEPTANCE",
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "status": "PASS",
        "pass": True,
        "decision": "ACCEPT_V5_TALIF_ONLY_120_EPOCH_PILOTS_RELEASE_FORMAL_FREEZE",
        "exit_code": 0,
        "validated_at": "2026-07-29T12:00:00+00:00",
        "protocol_hash": freeze_tool._stable_hash(protocol),
        "acceptance_hash": freeze_tool._stable_hash(
            dict(protocol["pilot_acceptance"])
        ),
        "training_environment_sha256": environment_sha256,
        "datasets": {
            "cifar100": {
                "status": "PASS",
                "pass": True,
                "health_seed": 1,
                "pilot_seed": 2,
                "environment_sha256": environment_sha256,
                "health_report_sha256": "1" * 64,
            },
            "cifar10dvs": {
                "status": "PASS",
                "pass": True,
                "health_seed": 3,
                "pilot_seed": 4,
                "environment_sha256": environment_sha256,
                "health_report_sha256": "2" * 64,
            },
        },
        "integrity_failures": [],
        "threshold_failures": [],
    }


def _write_fake_v5_validator(project: Path, report: dict[str, Any]) -> Path:
    path = project / "scripts" / "validate_v5_pilot.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "REPORT = "
        + repr(report)
        + "\n\ndef validate_pilot(**_kwargs):\n    return REPORT\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def test_v5_artifact_paths_are_distinct_from_v3_v4_and_legacy() -> None:
    values = set(V5_ARTIFACT_PATHS.values())

    assert len(values) == len(V5_ARTIFACT_PATHS)
    assert V5_ARTIFACT_PATHS["protocol"] == "configs/protocol_v5_talif_only.yaml"
    assert V5_ARTIFACT_PATHS["formal_matrix"] == "configs/v5_talif_only_generated"
    assert V5_ARTIFACT_PATHS["pilot_matrix"] == "configs/v5_talif_only_pilot_generated"
    assert V5_ARTIFACT_PATHS["freeze_manifest"] == "FREEZE_MANIFEST_V5_TALIF_ONLY.json"
    assert all("v3" not in value and "v4" not in value for value in values)


def test_v5_freeze_binds_aggregate_pass_and_training_environment(
    tmp_path: Path,
) -> None:
    protocol = _v5_protocol()
    protocol_path = tmp_path / V5_ARTIFACT_PATHS["protocol"]
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 5\n", encoding="utf-8")
    report = _v5_pass_report(protocol)
    validation_path = tmp_path / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        freeze_tool.canonical_json(report) + "\n", encoding="utf-8", newline="\n"
    )
    _write_fake_v5_validator(tmp_path, report)

    bound = freeze_tool._validate_v5_pilot_acceptance(
        protocol, protocol_path, tmp_path
    )

    assert bound["status"] == "PASS"
    assert bound["training_environment_sha256"] == "e" * 64
    assert {
        evidence["environment_sha256"] for evidence in bound["datasets"].values()
    } == {bound["training_environment_sha256"]}


def test_v5_freeze_rejects_cross_dataset_environment_mismatch(tmp_path: Path) -> None:
    protocol = _v5_protocol()
    protocol_path = tmp_path / V5_ARTIFACT_PATHS["protocol"]
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 5\n", encoding="utf-8")
    report = _v5_pass_report(protocol)
    report["datasets"]["cifar10dvs"]["environment_sha256"] = "f" * 64
    validation_path = tmp_path / protocol["pilot_acceptance"]["validation_output"]
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(freeze_tool.canonical_json(report), encoding="utf-8")
    _write_fake_v5_validator(tmp_path, report)

    with pytest.raises(
        freeze_tool.FreezeManifestError,
        match="dataset training environment bindings",
    ):
        freeze_tool._validate_v5_pilot_acceptance(protocol, protocol_path, tmp_path)


def _runtime_manifest(version: int) -> dict[str, Any]:
    validator_path = f"scripts/validate_v{version}_pilot.py"
    required_sources = (
        freeze_module.V4_GATE_SOURCE_PATHS
        if version == 4
        else freeze_module.V5_GATE_SOURCE_PATHS
    )
    pilot: dict[str, Any] = {
        "status": "PASS",
        "pass": True,
        "protocol_hash": "a" * 64,
        "validator": {"path": validator_path, "sha256": "b" * 64},
    }
    if version == 5:
        pilot.update(
            {
                "artifact_class": "NON_REPORTABLE_V5_TALIF_ONLY_PILOT_ACCEPTANCE",
                "training_environment_sha256": "e" * 64,
                "datasets": {
                    dataset: {
                        "status": "PASS",
                        "pass": True,
                        "environment_sha256": "e" * 64,
                    }
                    for dataset in ("cifar100", "cifar10dvs")
                },
            }
        )
    return {
        "protocol": {
            "path": "configs/protocol.yaml",
            "canonical_sha256": "a" * 64,
        },
        "generated_matrix": {"directory": "configs/matrix"},
        "pilot_validation": pilot,
        "gate_sources": {
            path: {
                "path": path,
                "file_sha256": "b" * 64 if path == validator_path else "c" * 64,
            }
            for path in required_sources
        },
    }


def _install_runtime_freeze_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    version: int,
    stored: dict[str, Any],
) -> None:
    monkeypatch.setattr(
        freeze_module, "load_protocol", lambda _path: {"protocol_version": version}
    )
    monkeypatch.setattr(
        freeze_module,
        "_load_manifest_tool",
        lambda _root: SimpleNamespace(verify_manifest=lambda **_kwargs: stored),
    )


def test_runtime_v5_freeze_gate_rejects_missing_environment_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "configs" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 5\n", encoding="utf-8")
    matrix_dir = tmp_path / "configs" / "matrix"
    matrix_dir.mkdir()
    stored = _runtime_manifest(5)
    stored["pilot_validation"].pop("training_environment_sha256")
    _install_runtime_freeze_stubs(monkeypatch, version=5, stored=stored)

    with pytest.raises(FreezeGateError, match="training_environment_sha256"):
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST.json",
        )


def test_runtime_v5_freeze_gate_accepts_exact_aggregate_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "configs" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 5\n", encoding="utf-8")
    matrix_dir = tmp_path / "configs" / "matrix"
    matrix_dir.mkdir()
    stored = _runtime_manifest(5)
    _install_runtime_freeze_stubs(monkeypatch, version=5, stored=stored)

    assert (
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST.json",
        )
        is stored
    )


def test_runtime_v4_freeze_gate_keeps_legacy_pilot_record_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol_path = tmp_path / "configs" / "protocol.yaml"
    protocol_path.parent.mkdir(parents=True)
    protocol_path.write_text("protocol_version: 4\n", encoding="utf-8")
    matrix_dir = tmp_path / "configs" / "matrix"
    matrix_dir.mkdir()
    stored = _runtime_manifest(4)
    _install_runtime_freeze_stubs(monkeypatch, version=4, stored=stored)

    assert (
        freeze_module.verify_formal_freeze(
            project_root=tmp_path,
            protocol_path=protocol_path,
            matrix_dir=matrix_dir,
            manifest_path=tmp_path / "FREEZE_MANIFEST.json",
        )
        is stored
    )


def test_v5_formal_evidence_requires_top_level_environment_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    protocol = _v5_protocol()
    pilot = _runtime_manifest(5)["pilot_validation"]
    pilot.update({"sha256": "d" * 64, "protocol_hash": train_module.stable_hash(protocol)})
    manifest_path = tmp_path / V5_ARTIFACT_PATHS["freeze_manifest"]
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(train_module, "PROJECT_ROOT", tmp_path)

    evidence = train_module._formal_execution_evidence(
        protocol,
        {"pilot_validation": pilot},
        protocol_path=tmp_path / V5_ARTIFACT_PATHS["protocol"],
    )
    assert evidence["training_environment_sha256"] == "e" * 64

    pilot["training_environment_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="aggregate pilot training_environment_sha256"):
        train_module._formal_execution_evidence(
            protocol,
            {"pilot_validation": pilot},
            protocol_path=tmp_path / V5_ARTIFACT_PATHS["protocol"],
        )


def _install_evaluate_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    version: int,
    observed_environment_sha256: str = "e" * 64,
) -> list[str]:
    protocol = {
        "protocol_version": version,
        "artifact_paths": {
            "formal_results": "results/formal",
            "formal_matrix": "configs/formal",
        },
    }
    (tmp_path / "results" / "formal").mkdir(parents=True)
    (tmp_path / "configs" / "formal").mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(evaluate_tool, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(evaluate_tool.os, "chdir", lambda _path: None)
    monkeypatch.setattr(
        evaluate_tool,
        "check_protocol",
        lambda *_args, **_kwargs: SimpleNamespace(ok=True, errors=[], protocol_hash="a" * 64),
    )
    monkeypatch.setattr(evaluate_tool, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        evaluate_tool, "artifact_paths_for_protocol", lambda _protocol: protocol["artifact_paths"]
    )
    monkeypatch.setattr(evaluate_tool, "expected_run_count_for_protocol", lambda _protocol: 0)
    monkeypatch.setattr(
        evaluate_tool,
        "verify_formal_freeze",
        lambda **_kwargs: calls.append("freeze") or {"pilot_validation": {}},
    )
    monkeypatch.setattr(
        evaluate_tool,
        "_formal_execution_evidence",
        lambda *_args, **_kwargs: calls.append("evidence")
        or {"training_environment_sha256": "e" * 64},
    )
    monkeypatch.setattr(evaluate_tool, "seed_everything", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate_tool, "resolve_device", lambda _device: "cpu")
    monkeypatch.setattr(
        evaluate_tool,
        "_training_environment_identity",
        lambda *_args, **_kwargs: ({}, observed_environment_sha256),
    )
    monkeypatch.setattr(
        evaluate_tool,
        "_read_expected",
        lambda *_args, **_kwargs: calls.append("read_expected") or [],
    )
    monkeypatch.setattr(
        evaluate_tool,
        "_audit_consolidated_metrics",
        lambda *_args, **kwargs: calls.append(
            "audit:"
            + str(kwargs.get("expected_training_environment_sha256"))
        ),
    )
    monkeypatch.setattr(evaluate_tool, "_audit_all_frozen", lambda *_args, **_kwargs: None)
    return calls


@pytest.mark.parametrize("version", (4, 5))
def test_evaluate_checkpoints_v4_v5_share_freeze_and_environment_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    calls = _install_evaluate_stubs(monkeypatch, tmp_path, version=version)

    assert evaluate_tool.main(["--protocol", str(tmp_path / "protocol.yaml")]) == 0
    assert calls == [
        "freeze",
        "evidence",
        "read_expected",
        f"audit:{'e' * 64}",
    ]


def test_evaluate_checkpoints_v5_blocks_environment_drift_before_result_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_evaluate_stubs(
        monkeypatch,
        tmp_path,
        version=5,
        observed_environment_sha256="f" * 64,
    )

    with pytest.raises(SystemExit, match="final-test environment differs"):
        evaluate_tool.main(["--protocol", str(tmp_path / "protocol.yaml")])
    assert calls == ["freeze", "evidence"]


def test_evaluate_checkpoints_v3_preserves_pre_v4_release_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_evaluate_stubs(monkeypatch, tmp_path, version=3)

    assert evaluate_tool.main(["--protocol", str(tmp_path / "protocol.yaml")]) == 0
    assert calls == ["read_expected", "audit:None"]


def _write_v5_final_test_audit_fixture(
    results_root: Path,
    environment_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, object], Path]:
    run_id = "v5-run"
    metric: dict[str, object] = {
        field: "" for field in train_module.SEED_METRIC_FIELDS
    }
    metric.update(
        {
            "run_id": run_id,
            "experiment": "E1",
            "dataset": "cifar100",
            "depth": 20,
            "time_steps": 6,
            "condition": "C1",
            "topology": "spiking_resnet",
            "neuron": "lif",
            "seed": 101,
            "config_hash": "config-hash",
            "protocol_hash": "protocol-hash",
            "split_manifest_sha256": "split-hash",
            "shared_weight_sha256": "weight-hash",
            "training_environment_sha256": environment_sha256,
            "status": "complete",
            "failed": 0,
        }
    )
    plan_fields = (
        "run_id",
        "experiment",
        "dataset",
        "depth",
        "time_steps",
        "condition",
        "topology",
        "neuron",
        "seed",
        "config_hash",
        "protocol_hash",
    )
    rows = [{field: str(metric[field]) for field in plan_fields}]
    run_dir = results_root / run_id
    run_dir.mkdir()
    (run_dir / "seed_metrics.json").write_text(
        json.dumps(metric), encoding="utf-8"
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config_hash": metric["config_hash"],
                "split_manifest_sha256": metric["split_manifest_sha256"],
                "shared_weight_sha256": metric["shared_weight_sha256"],
                "config": {
                    "analysis": {"protocol_hash": metric["protocol_hash"]}
                },
                "environment": {
                    "training_environment_sha256": environment_sha256
                },
            }
        ),
        encoding="utf-8",
    )
    _write_v5_consolidated_metric(results_root, metric)
    return rows, metric, run_dir


def _write_v5_consolidated_metric(
    results_root: Path, metric: dict[str, object]
) -> None:
    with (results_root / "seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=train_module.SEED_METRIC_FIELDS
        )
        writer.writeheader()
        writer.writerow(metric)


@pytest.mark.parametrize(
    ("tampered_source", "message"),
    (
        ("consolidated", "consolidated seed_metrics.csv differs"),
        ("per_run", "per-run seed_metrics.json differs"),
        (
            "manifest",
            "run_manifest.json environment.training_environment_sha256 differs",
        ),
    ),
)
def test_v5_final_test_audit_binds_each_training_environment_hash_to_freeze(
    tmp_path: Path,
    tampered_source: str,
    message: str,
) -> None:
    frozen_environment_sha256 = "e" * 64
    rows, metric, run_dir = _write_v5_final_test_audit_fixture(
        tmp_path, frozen_environment_sha256
    )
    assert len(
        evaluate_tool._audit_consolidated_metrics(
            rows,
            tmp_path,
            expected_training_environment_sha256=frozen_environment_sha256,
        )
    ) == 1

    if tampered_source == "consolidated":
        metric["training_environment_sha256"] = "f" * 64
        _write_v5_consolidated_metric(tmp_path, metric)
    elif tampered_source == "per_run":
        per_run = json.loads(
            (run_dir / "seed_metrics.json").read_text(encoding="utf-8")
        )
        per_run["training_environment_sha256"] = "f" * 64
        (run_dir / "seed_metrics.json").write_text(
            json.dumps(per_run), encoding="utf-8"
        )
    else:
        manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        manifest["environment"]["training_environment_sha256"] = "f" * 64
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    with pytest.raises(RuntimeError, match=message):
        evaluate_tool._audit_consolidated_metrics(
            rows,
            tmp_path,
            expected_training_environment_sha256=frozen_environment_sha256,
        )


def test_v5_final_test_audit_rejects_joint_environment_hash_rebinding(
    tmp_path: Path,
) -> None:
    frozen_environment_sha256 = "e" * 64
    rows, metric, run_dir = _write_v5_final_test_audit_fixture(
        tmp_path, frozen_environment_sha256
    )
    rebound_environment_sha256 = "f" * 64
    metric["training_environment_sha256"] = rebound_environment_sha256
    _write_v5_consolidated_metric(tmp_path, metric)

    per_run = json.loads(
        (run_dir / "seed_metrics.json").read_text(encoding="utf-8")
    )
    per_run["training_environment_sha256"] = rebound_environment_sha256
    (run_dir / "seed_metrics.json").write_text(
        json.dumps(per_run), encoding="utf-8"
    )
    manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    manifest["environment"][
        "training_environment_sha256"
    ] = rebound_environment_sha256
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    with pytest.raises(RuntimeError) as exc_info:
        evaluate_tool._audit_consolidated_metrics(
            rows,
            tmp_path,
            expected_training_environment_sha256=frozen_environment_sha256,
        )
    message = str(exc_info.value)
    assert "consolidated seed_metrics.csv differs" in message
    assert "per-run seed_metrics.json differs" in message
    assert (
        "run_manifest.json environment.training_environment_sha256 differs"
        in message
    )
