"""Runtime gate for the two-stage experiment freeze."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import (
    active_conditions_for_protocol,
    artifact_paths_for_protocol,
    load_protocol,
)
from .pilot_v7 import (
    V7_CHECKPOINT_SERIALIZATION_RECOVERY_ALLOWED_CHANGED_PATHS,
    V7_CHECKPOINT_SERIALIZATION_RECOVERY_ALLOWED_PATH,
    V7_CHECKPOINT_SERIALIZATION_RECOVERY_BASE_SEAL_COMMIT,
    V7_CHECKPOINT_SERIALIZATION_RECOVERY_RELEASE_RECORD,
    V7_CHECKPOINT_SERIALIZATION_RECOVERY_SCHEMA,
    V7_CHECKPOINT_SERIALIZATION_RECOVERY_SOURCE_TYPE,
    V7_CHECKPOINT_SERIALIZATION_RECOVERY_TARGET_TYPE,
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

V4_GATE_SOURCE_PATHS = frozenset(
    {
        "src/talif_msresnet/pilot_v4.py",
        "src/talif_msresnet/statistics_v3.py",
        "src/talif_msresnet/statistics_v4.py",
        "scripts/analyze_v3_results.py",
        "scripts/analyze_v4_results.py",
        "scripts/pilot_health_gate_v4.py",
        "scripts/validate_v4_pilot.py",
    }
)
V5_GATE_SOURCE_PATHS = frozenset(
    {
        "src/talif_msresnet/pilot_v3.py",
        "src/talif_msresnet/pilot_v4.py",
        "src/talif_msresnet/pilot_v5.py",
        "src/talif_msresnet/statistics_v3.py",
        "src/talif_msresnet/statistics_v5.py",
        "scripts/analyze_v3_results.py",
        "scripts/analyze_v5_results.py",
        "scripts/calibrate_v5_health.py",
        "scripts/evaluate_checkpoints.py",
        "scripts/pilot_health_gate.py",
        "scripts/pilot_health_gate_v3.py",
        "scripts/pilot_health_gate_v5.py",
        "scripts/validate_v3_pilot.py",
        "scripts/validate_v5_pilot.py",
    }
)
V6_GATE_SOURCE_PATHS = frozenset(
    {
        "src/talif_msresnet/benchmark.py",
        "src/talif_msresnet/benchmark_v6.py",
        "src/talif_msresnet/config_v6.py",
        "src/talif_msresnet/models.py",
        "src/talif_msresnet/neurons.py",
        "src/talif_msresnet/pathing.py",
        "src/talif_msresnet/pilot_v3.py",
        "src/talif_msresnet/pilot_v6.py",
        "src/talif_msresnet/utils.py",
        "src/talif_msresnet/v6_statistics.py",
        "scripts/analyze_v6_results.py",
        "scripts/archive_v6_evidence.py",
        "scripts/benchmark_checkpoint.py",
        "scripts/export_v6_validation_batch.py",
        "scripts/evaluate_checkpoints.py",
        "scripts/pilot_health_gate.py",
        "scripts/pilot_health_gate_v3.py",
        "scripts/pilot_health_gate_v6.py",
        "scripts/run_benchmarks.py",
        "scripts/run_v6_benchmarks.py",
        "scripts/validate_v3_pilot.py",
        "scripts/validate_v6_pilot.py",
    }
)
V7_GATE_SOURCE_PATHS = frozenset(
    {
        V7_CHECKPOINT_SERIALIZATION_RECOVERY_RELEASE_RECORD,
        V7_HEALTH_RECOVERY_RELEASE_RECORD,
        V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD,
        "V7_MECHANISM_5090_RUNBOOK.md",
        "V7_STATISTICAL_ANALYSIS_AUDIT.md",
        "src/talif_msresnet/benchmark.py",
        "src/talif_msresnet/benchmark_v7.py",
        "src/talif_msresnet/config.py",
        "src/talif_msresnet/config_v6.py",
        "src/talif_msresnet/config_v7.py",
        "src/talif_msresnet/data.py",
        "src/talif_msresnet/diagnostics.py",
        "src/talif_msresnet/freeze.py",
        "src/talif_msresnet/models.py",
        "src/talif_msresnet/neurons.py",
        "src/talif_msresnet/ops.py",
        "src/talif_msresnet/pathing.py",
        "src/talif_msresnet/pilot_v3.py",
        "src/talif_msresnet/pilot_v4.py",
        "src/talif_msresnet/pilot_v5.py",
        "src/talif_msresnet/pilot_v6.py",
        "src/talif_msresnet/pilot_v7.py",
        "src/talif_msresnet/preflight.py",
        "src/talif_msresnet/train.py",
        "src/talif_msresnet/utils.py",
        "src/talif_msresnet/v7_statistics.py",
        "scripts/analyze_v7_results.py",
        "scripts/archive_v7_evidence.py",
        "scripts/create_freeze_manifest.py",
        "scripts/create_v7_instance_package.py",
        "scripts/evaluate_checkpoints.py",
        "scripts/export_v7_validation_batch.py",
        "scripts/generate_run_configs.py",
        "scripts/pilot_health_gate.py",
        "scripts/pilot_health_gate_v3.py",
        "scripts/pilot_health_gate_v4.py",
        "scripts/pilot_health_gate_v5.py",
        "scripts/pilot_health_gate_v6.py",
        "scripts/pilot_health_gate_v7.py",
        "scripts/preflight.py",
        "scripts/run_matrix.py",
        "scripts/run_v7_benchmarks.py",
        "scripts/validate_v3_pilot.py",
        "scripts/validate_v4_pilot.py",
        "scripts/validate_v5_pilot.py",
        "scripts/validate_v6_pilot.py",
        "scripts/validate_v7_pilot.py",
        "scripts/validate_v7_source_release.py",
        "scripts/v7_monitor_cn.py",
        "tests/test_v7_archive.py",
        "tests/test_v7_benchmark.py",
        "tests/test_v7_final_test_binding.py",
        "tests/test_v7_freeze_gates.py",
        "tests/test_v7_health_gate.py",
        "tests/test_v7_monitor_cn.py",
        "tests/test_v7_orchestration.py",
        "tests/test_v7_protocol.py",
        "tests/test_v7_provenance.py",
        "tests/test_v7_runtime.py",
        "tests/test_v7_source_release.py",
        "tests/test_v7_statistics.py",
        "tests/test_v7_training_orchestration.py",
        "tests/test_validate_v7_pilot.py",
    }
)
_LOWER_HEX = frozenset("0123456789abcdef")


class FreezeGateError(RuntimeError):
    """Raised when a formal run is not bound to the verified freeze manifest."""


def _load_manifest_tool(project_root: Path) -> ModuleType:
    tool_path = project_root / "scripts" / "create_freeze_manifest.py"
    if not tool_path.is_file():
        raise FreezeGateError(f"Freeze-manifest verifier is missing: {tool_path}")
    spec = importlib.util.spec_from_file_location("talif_runtime_freeze_manifest", tool_path)
    if spec is None or spec.loader is None:
        raise FreezeGateError(f"Cannot load freeze-manifest verifier: {tool_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise FreezeGateError(
            f"Cannot import freeze-manifest verifier: {type(exc).__name__}: {exc}"
        ) from exc
    if not callable(getattr(module, "verify_manifest", None)):
        raise FreezeGateError("Freeze-manifest verifier has no verify_manifest entry point")
    return module


def _bound_path(project_root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FreezeGateError(f"Freeze manifest has no {label} path")
    path = Path(value)
    resolved = (path if path.is_absolute() else project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise FreezeGateError(f"Freeze manifest {label} path escapes the project root") from exc
    return resolved


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_LOWER_HEX)
    )


def verify_formal_freeze(
    *,
    project_root: str | Path,
    protocol_path: str | Path,
    matrix_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Verify Phase B and bind the requested formal inputs to its artifacts."""

    root = Path(project_root).resolve()
    protocol = Path(protocol_path)
    protocol = (protocol if protocol.is_absolute() else root / protocol).resolve()
    protocol_mapping = load_protocol(protocol)
    if manifest_path is None:
        manifest_value = artifact_paths_for_protocol(protocol_mapping)["freeze_manifest"]
        manifest = root / manifest_value
    else:
        manifest = Path(manifest_path)
    manifest = (manifest if manifest.is_absolute() else root / manifest).resolve()
    tool = _load_manifest_tool(root)
    try:
        stored = tool.verify_manifest(project_root=root, manifest_path=manifest)
    except Exception as exc:
        raise FreezeGateError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(stored, Mapping):
        raise FreezeGateError("Freeze-manifest verifier returned a non-mapping result")

    protocol_record = stored.get("protocol")
    generated_record = stored.get("generated_matrix")
    if not isinstance(protocol_record, Mapping) or not isinstance(generated_record, Mapping):
        raise FreezeGateError("Freeze manifest is missing protocol/generated_matrix records")
    bound_protocol = _bound_path(root, protocol_record.get("path"), "protocol")
    if protocol != bound_protocol:
        raise FreezeGateError(
            f"Requested protocol is not the manifest-bound protocol: {protocol} != {bound_protocol}"
        )
    if matrix_dir is not None:
        requested_matrix = Path(matrix_dir)
        requested_matrix = (
            requested_matrix if requested_matrix.is_absolute() else root / requested_matrix
        ).resolve()
        bound_matrix = _bound_path(root, generated_record.get("directory"), "matrix")
        if requested_matrix != bound_matrix:
            raise FreezeGateError(
                "Requested configuration directory is not the manifest-bound matrix: "
                f"{requested_matrix} != {bound_matrix}"
            )
    version = int(protocol_mapping.get("protocol_version", 1))
    if version in (4, 5, 6, 7):
        pilot_record = stored.get("pilot_validation")
        gate_sources = stored.get("gate_sources")
        required_sources = (
            V4_GATE_SOURCE_PATHS
            if version == 4
            else V5_GATE_SOURCE_PATHS
            if version == 5
            else V6_GATE_SOURCE_PATHS
            if version == 6
            else V7_GATE_SOURCE_PATHS
        )
        validator_path = f"scripts/validate_v{version}_pilot.py"
        if not isinstance(pilot_record, Mapping):
            raise FreezeGateError(
                f"Protocol v{version} formal release has no bound aggregate pilot validation"
            )
        if pilot_record.get("status") != "PASS" or pilot_record.get("pass") is not True:
            raise FreezeGateError(
                f"Protocol v{version} formal release is not bound to an aggregate pilot PASS"
            )
        if pilot_record.get("protocol_hash") != protocol_record.get("canonical_sha256"):
            raise FreezeGateError(
                f"Protocol v{version} pilot validation is bound to a different protocol hash"
            )
        if version == 5:
            expected_artifact_class = "NON_REPORTABLE_V5_TALIF_ONLY_PILOT_ACCEPTANCE"
            if pilot_record.get("artifact_class") != expected_artifact_class:
                raise FreezeGateError(
                    "Protocol v5 formal release is not bound to the aggregate pilot artifact"
                )
            training_environment_sha256 = pilot_record.get(
                "training_environment_sha256"
            )
            if not _is_sha256(training_environment_sha256):
                raise FreezeGateError(
                    "Protocol v5 freeze manifest has no valid pilot-bound "
                    "training_environment_sha256"
                )
            datasets = pilot_record.get("datasets")
            if not isinstance(datasets, Mapping) or set(datasets) != {
                "cifar100",
                "cifar10dvs",
            }:
                raise FreezeGateError(
                    "Protocol v5 freeze manifest does not bind both pilot datasets"
                )
            for dataset, evidence in datasets.items():
                if (
                    not isinstance(evidence, Mapping)
                    or evidence.get("status") != "PASS"
                    or evidence.get("pass") is not True
                    or evidence.get("environment_sha256")
                    != training_environment_sha256
                ):
                    raise FreezeGateError(
                        "Protocol v5 freeze manifest has an inconsistent aggregate "
                        f"pilot/environment binding for {dataset}"
                    )
            recovery = pilot_record.get("recovery")
            if recovery is not None:
                original = recovery.get("original_validation") if isinstance(
                    recovery, Mapping
                ) else None
                repository = stored.get("repository")
                if (
                    not isinstance(recovery, Mapping)
                    or recovery.get("artifact_class")
                    != "NON_REPORTABLE_V5_PILOT_VALIDATION_RECOVERY"
                    or not _is_sha256(recovery.get("sha256"))
                    or recovery.get("pilot_execution_commit")
                    != "6eeadd6389677347fe46ffa8d3bdec8c75455b44"
                    or not isinstance(repository, Mapping)
                    or recovery.get("recovery_validator_commit")
                    != repository.get("freeze_commit")
                    or not isinstance(original, Mapping)
                    or original.get("status") != "INVALID"
                    or not _is_sha256(original.get("sha256"))
                    or original.get("sha256")
                    != "5cc85680923b6eee1e454dcf4cf7667f4271dc531d5837a529e57bfcaccd7e86"
                ):
                    raise FreezeGateError(
                        "Protocol v5 freeze manifest has a malformed pilot recovery binding"
                    )
        elif version == 6:
            expected_artifact_class = "NON_REPORTABLE_V6_MECHANISM_PILOT_ACCEPTANCE"
            if pilot_record.get("artifact_class") != expected_artifact_class:
                raise FreezeGateError(
                    "Protocol v6 formal release is not bound to the six-condition pilot artifact"
                )
            training_environment_sha256 = pilot_record.get(
                "training_environment_sha256"
            )
            if not _is_sha256(training_environment_sha256):
                raise FreezeGateError(
                    "Protocol v6 freeze manifest has no valid pilot-bound "
                    "training_environment_sha256"
                )
            expected_conditions = list(active_conditions_for_protocol(protocol_mapping))
            if pilot_record.get("conditions") != expected_conditions:
                raise FreezeGateError(
                    "Protocol v6 freeze manifest does not bind the ordered six-condition pilot"
                )
            datasets = pilot_record.get("datasets")
            if not isinstance(datasets, Mapping) or set(datasets) != {"cifar100"}:
                raise FreezeGateError(
                    "Protocol v6 freeze manifest does not bind the CIFAR-100 pilot"
                )
            evidence = datasets["cifar100"]
            required_hashes = (
                "block_hash",
                "health_report_sha256",
                "attempt_receipt_sha256",
                "pilot_plan_sha256",
                "shared_weight_sha256",
                "split_manifest_sha256",
            )
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("status") != "PASS"
                or evidence.get("pass") is not True
                or evidence.get("environment_sha256")
                != training_environment_sha256
                or any(not _is_sha256(evidence.get(key)) for key in required_hashes)
            ):
                raise FreezeGateError(
                    "Protocol v6 freeze manifest has an inconsistent six-condition "
                    "pilot/environment binding"
                )
        elif version == 7:
            expected_artifact_class = (
                "NON_REPORTABLE_V7_PILOT_VALIDATION_DEVICE_COMPATIBILITY_RECOVERY"
            )
            if pilot_record.get("artifact_class") != expected_artifact_class:
                raise FreezeGateError(
                    "Protocol v7 formal release is not bound to the V7 six-condition pilot artifact"
                )
            if protocol_record.get("version") != 7:
                raise FreezeGateError(
                    "Protocol v7 freeze manifest protocol identity is not version 7"
                )
            protocol_file_sha256 = pilot_record.get("protocol_file_sha256")
            if (
                not _is_sha256(protocol_file_sha256)
                or protocol_file_sha256 != protocol_record.get("file_sha256")
            ):
                raise FreezeGateError(
                    "Protocol v7 pilot validation is not bound to the frozen protocol file"
                )
            training_environment_sha256 = pilot_record.get(
                "training_environment_sha256"
            )
            if not _is_sha256(training_environment_sha256):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest has no valid pilot-bound "
                    "training_environment_sha256"
                )
            expected_conditions = list(active_conditions_for_protocol(protocol_mapping))
            if expected_conditions != ["M0", "M1", "M2", "M3", "M4", "PLIF"] or (
                pilot_record.get("conditions") != expected_conditions
            ):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest does not bind the ordered six-condition pilot"
                )
            for key in (
                "sha256",
                "pilot_matrix_manifest_sha256",
                "pilot_run_manifest_csv_sha256",
            ):
                if not _is_sha256(pilot_record.get(key)):
                    raise FreezeGateError(
                        "Protocol v7 freeze manifest has an invalid pilot validation/matrix hash"
                    )
            datasets = pilot_record.get("datasets")
            if not isinstance(datasets, Mapping) or set(datasets) != {"cifar100"}:
                raise FreezeGateError(
                    "Protocol v7 freeze manifest does not bind the CIFAR-100 pilot"
                )
            evidence = datasets["cifar100"]
            repository = stored.get("repository")
            recovery = pilot_record.get("health_compatibility_recovery")
            device_recovery = pilot_record.get("device_compatibility_recovery")
            serialization_recovery = pilot_record.get(
                "checkpoint_serialization_recovery"
            )
            recovery_source = gate_sources.get(V7_HEALTH_RECOVERY_RELEASE_RECORD)
            device_recovery_source = gate_sources.get(
                V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD
            )
            serialization_recovery_source = gate_sources.get(
                V7_CHECKPOINT_SERIALIZATION_RECOVERY_RELEASE_RECORD
            )
            expected_recovery_values = {
                "schema": V7_HEALTH_RECOVERY_SCHEMA,
                "base_health_commit": V7_HEALTH_RECOVERY_BASE_COMMIT,
                "health_report_path": V7_HEALTH_RECOVERY_HEALTH_PATH,
                "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
                "attempt_receipt_path": V7_HEALTH_RECOVERY_ATTEMPT_PATH,
                "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
                "protocol_hash": V7_HEALTH_RECOVERY_PROTOCOL_HASH,
                "corrected_field": V7_HEALTH_RECOVERY_CORRECTED_FIELD,
                "split_source_fingerprint": (
                    V7_HEALTH_RECOVERY_SPLIT_SOURCE_FINGERPRINT
                ),
                "allowed_changed_paths": list(
                    V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS
                ),
                "observed_changes": [
                    f"M\t{path}"
                    for path in V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS
                ],
                "release_record": V7_HEALTH_RECOVERY_RELEASE_RECORD,
                "tracked_clean": True,
            }
            if (
                not isinstance(repository, Mapping)
                or not isinstance(recovery, Mapping)
                or any(
                    recovery.get(key) != value
                    for key, value in expected_recovery_values.items()
                )
                or recovery.get("recovery_commit")
                != V7_HEALTH_RECOVERY_SEAL_COMMIT
                or not isinstance(recovery.get("implementation_commit"), str)
                or len(recovery["implementation_commit"]) != 40
                or not set(recovery["implementation_commit"]) <= _LOWER_HEX
                or not isinstance(recovery.get("implementation_tree"), str)
                or len(recovery["implementation_tree"]) != 40
                or not set(recovery["implementation_tree"]) <= _LOWER_HEX
                or not _is_sha256(recovery.get("release_record_sha256"))
                or not _is_sha256(recovery.get("record_sha256"))
                or not isinstance(recovery_source, Mapping)
                or recovery_source.get("file_sha256")
                != recovery.get("release_record_sha256")
                or not isinstance(recovery.get("implementation_file_sha256"), Mapping)
                or set(recovery["implementation_file_sha256"])
                != set(V7_HEALTH_RECOVERY_ALLOWED_CHANGED_PATHS)
                or any(
                    not _is_sha256(value)
                    for value in recovery["implementation_file_sha256"].values()
                )
            ):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest has a malformed sealed health "
                    "compatibility recovery binding"
                )
            expected_device_recovery = {
                "schema": V7_PILOT_DEVICE_RECOVERY_SCHEMA,
                "base_health_recovery_seal_commit": V7_HEALTH_RECOVERY_SEAL_COMMIT,
                "pilot_execution_commit": V7_HEALTH_RECOVERY_SEAL_COMMIT,
                "original_validation_path": V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION,
                "original_validation_sha256": (
                    V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256
                ),
                "recovery_output_path": V7_PILOT_DEVICE_RECOVERY_OUTPUT,
                "protocol_hash": V7_PILOT_DEVICE_RECOVERY_PROTOCOL_HASH,
                "frozen_runtime_device": V7_PILOT_DEVICE_RECOVERY_FROZEN_DEVICE,
                "accepted_execution_device": V7_PILOT_DEVICE_RECOVERY_EXECUTION_DEVICE,
                "health_report_sha256": V7_HEALTH_RECOVERY_HEALTH_SHA256,
                "attempt_receipt_sha256": V7_HEALTH_RECOVERY_ATTEMPT_SHA256,
                "allowed_changed_paths": list(
                    V7_PILOT_DEVICE_RECOVERY_ALLOWED_CHANGED_PATHS
                ),
                "observed_changes": [
                    f"M\t{path}"
                    for path in V7_PILOT_DEVICE_RECOVERY_ALLOWED_CHANGED_PATHS
                ],
                "release_record": V7_PILOT_DEVICE_RECOVERY_RELEASE_RECORD,
                "tracked_clean": True,
            }
            if (
                not isinstance(device_recovery, Mapping)
                or any(
                    device_recovery.get(key) != value
                    for key, value in expected_device_recovery.items()
                )
                or not isinstance(device_recovery.get("implementation_commit"), str)
                or len(device_recovery["implementation_commit"]) != 40
                or not set(device_recovery["implementation_commit"]) <= _LOWER_HEX
                or not isinstance(device_recovery.get("implementation_tree"), str)
                or len(device_recovery["implementation_tree"]) != 40
                or not set(device_recovery["implementation_tree"]) <= _LOWER_HEX
                or not _is_sha256(device_recovery.get("release_record_sha256"))
                or not _is_sha256(device_recovery.get("record_sha256"))
                or not isinstance(device_recovery_source, Mapping)
                or device_recovery_source.get("file_sha256")
                != device_recovery.get("release_record_sha256")
                or not isinstance(
                    device_recovery.get("implementation_file_sha256"), Mapping
                )
                or set(device_recovery["implementation_file_sha256"])
                != set(V7_PILOT_DEVICE_RECOVERY_ALLOWED_CHANGED_PATHS)
                or any(
                    not _is_sha256(value)
                    for value in device_recovery["implementation_file_sha256"].values()
                )
            ):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest has a malformed sealed pilot "
                    "device compatibility recovery binding"
                )
            expected_serialization_recovery = {
                "schema": V7_CHECKPOINT_SERIALIZATION_RECOVERY_SCHEMA,
                "base_device_recovery_seal_commit": (
                    V7_CHECKPOINT_SERIALIZATION_RECOVERY_BASE_SEAL_COMMIT
                ),
                "pilot_execution_commit": V7_HEALTH_RECOVERY_SEAL_COMMIT,
                "original_validation_path": (
                    V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION
                ),
                "original_validation_sha256": (
                    V7_PILOT_DEVICE_RECOVERY_ORIGINAL_VALIDATION_SHA256
                ),
                "recovery_output_path": V7_PILOT_DEVICE_RECOVERY_OUTPUT,
                "protocol_hash": V7_PILOT_DEVICE_RECOVERY_PROTOCOL_HASH,
                "allowed_container_path": (
                    V7_CHECKPOINT_SERIALIZATION_RECOVERY_ALLOWED_PATH
                ),
                "checkpoint_container_type": (
                    V7_CHECKPOINT_SERIALIZATION_RECOVERY_SOURCE_TYPE
                ),
                "json_container_type": (
                    V7_CHECKPOINT_SERIALIZATION_RECOVERY_TARGET_TYPE
                ),
                "canonical_payload_must_match": True,
                "execution_hash_must_match": True,
                "scientific_hash_must_match": True,
                "allowed_changed_paths": list(
                    V7_CHECKPOINT_SERIALIZATION_RECOVERY_ALLOWED_CHANGED_PATHS
                ),
                "observed_changes": [
                    f"M\t{path}"
                    for path in V7_CHECKPOINT_SERIALIZATION_RECOVERY_ALLOWED_CHANGED_PATHS
                ],
                "release_record": (
                    V7_CHECKPOINT_SERIALIZATION_RECOVERY_RELEASE_RECORD
                ),
                "tracked_clean": True,
            }
            if not isinstance(serialization_recovery, Mapping):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest has a malformed sealed checkpoint "
                    "serialization compatibility recovery binding"
                )
            if pilot_record.get("recovery_validator_commit") != (
                serialization_recovery.get("recovery_commit")
            ):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest checkpoint serialization "
                    "compatibility recovery validator commit does not match its seal"
                )
            if (
                any(
                    serialization_recovery.get(key) != value
                    for key, value in expected_serialization_recovery.items()
                )
                or not isinstance(
                    serialization_recovery.get("implementation_commit"), str
                )
                or len(serialization_recovery["implementation_commit"]) != 40
                or not set(serialization_recovery["implementation_commit"])
                <= _LOWER_HEX
                or not isinstance(
                    serialization_recovery.get("implementation_tree"), str
                )
                or len(serialization_recovery["implementation_tree"]) != 40
                or not set(serialization_recovery["implementation_tree"])
                <= _LOWER_HEX
                or not _is_sha256(
                    serialization_recovery.get("release_record_sha256")
                )
                or not _is_sha256(serialization_recovery.get("record_sha256"))
                or not isinstance(serialization_recovery_source, Mapping)
                or serialization_recovery_source.get("file_sha256")
                != serialization_recovery.get("release_record_sha256")
                or not isinstance(
                    serialization_recovery.get("implementation_file_sha256"),
                    Mapping,
                )
                or set(serialization_recovery["implementation_file_sha256"])
                != set(V7_CHECKPOINT_SERIALIZATION_RECOVERY_ALLOWED_CHANGED_PATHS)
                or any(
                    not _is_sha256(value)
                    for value in serialization_recovery[
                        "implementation_file_sha256"
                    ].values()
                )
            ):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest has a malformed sealed checkpoint "
                    "serialization compatibility recovery binding"
                )
            required_hashes = (
                "block_hash",
                "health_report_sha256",
                "attempt_receipt_sha256",
                "pilot_plan_sha256",
                "shared_weight_sha256",
                "split_manifest_sha256",
            )
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("status") != "PASS"
                or evidence.get("pass") is not True
                or evidence.get("health_seed")
                != protocol_mapping["pilot_acceptance"]["health_seed"]
                or evidence.get("pilot_seed")
                != protocol_mapping["pilot_acceptance"]["pilot_seed"]
                or evidence.get("environment_sha256")
                != training_environment_sha256
                or evidence.get("health_report_sha256")
                != recovery.get("health_report_sha256")
                or evidence.get("attempt_receipt_sha256")
                != recovery.get("attempt_receipt_sha256")
                or any(not _is_sha256(evidence.get(key)) for key in required_hashes)
            ):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest has an inconsistent six-condition "
                    "pilot/environment binding"
                )
            expected_seeds = list(protocol_mapping.get("seeds", ()))
            matrix_manifest = generated_record.get("matrix_manifest")
            run_manifest = generated_record.get("run_manifest")
            if (
                generated_record.get("run_count") != 48
                or generated_record.get("config_count") != 48
                or generated_record.get("conditions") != expected_conditions
                or len(expected_seeds) != 8
                or generated_record.get("seeds") != expected_seeds
                or not _is_sha256(generated_record.get("matrix_hash"))
                or not _is_sha256(generated_record.get("config_set_sha256"))
                or not isinstance(matrix_manifest, Mapping)
                or matrix_manifest.get("path") != "matrix_manifest.json"
                or not _is_sha256(matrix_manifest.get("sha256"))
                or not isinstance(run_manifest, Mapping)
                or run_manifest.get("path") != "run_manifest.csv"
                or not _is_sha256(run_manifest.get("sha256"))
            ):
                raise FreezeGateError(
                    "Protocol v7 freeze manifest does not bind the exact 48-run formal matrix"
                )
        if not isinstance(gate_sources, Mapping) or set(gate_sources) != required_sources:
            raise FreezeGateError(
                f"Protocol v{version} freeze manifest does not bind the complete "
                f"v{version} gate source set"
            )
        malformed_sources = [
            path
            for path in required_sources
            if not isinstance(gate_sources.get(path), Mapping)
            or gate_sources[path].get("path") != path
            or not _is_sha256(gate_sources[path].get("file_sha256"))
        ]
        if malformed_sources:
            raise FreezeGateError(
                f"Protocol v{version} freeze manifest has malformed gate source hashes: "
                + ", ".join(sorted(malformed_sources))
            )
        validator = pilot_record.get("validator")
        validator_source = gate_sources.get(validator_path)
        if not isinstance(validator, Mapping) or not isinstance(
            validator_source, Mapping
        ):
            raise FreezeGateError(
                f"Protocol v{version} validator source binding is malformed"
            )
        if (
            validator.get("path") != validator_path
            or validator.get("sha256") != validator_source.get("file_sha256")
        ):
            raise FreezeGateError(
                f"Protocol v{version} pilot validator differs from the manifest-bound source"
            )
    return stored
