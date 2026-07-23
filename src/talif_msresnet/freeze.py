"""Runtime gate for the two-stage experiment freeze."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


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
    manifest = Path(manifest_path or root / "FREEZE_MANIFEST.json")
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
    return stored
