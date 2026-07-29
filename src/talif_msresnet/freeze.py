"""Runtime gate for the two-stage experiment freeze."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from .config import artifact_paths_for_protocol, load_protocol


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
    if version in (4, 5):
        pilot_record = stored.get("pilot_validation")
        gate_sources = stored.get("gate_sources")
        required_sources = (
            V4_GATE_SOURCE_PATHS if version == 4 else V5_GATE_SOURCE_PATHS
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
        if not isinstance(gate_sources, Mapping) or set(gate_sources) != required_sources:
            raise FreezeGateError(
                f"Protocol v{version} freeze manifest does not bind the complete "
                f"v{version} gate source set"
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
