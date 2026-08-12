#!/usr/bin/env python
"""Run generated configurations sequentially and preserve failures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)

from talif_msresnet.config import (  # noqa: E402
    artifact_paths_for_protocol,
    expected_run_count_for_protocol,
    generate_run_matrix,
    generate_v3_pilot_matrix,
    generate_v4_pilot_matrix,
    generate_v5_pilot_matrix,
    load_protocol,
    load_run_config,
    validate_run_mapping,
)
from talif_msresnet.config_v6 import generate_v6_pilot_matrix  # noqa: E402
from talif_msresnet.config_v7 import generate_v7_pilot_matrix  # noqa: E402
from talif_msresnet.pilot_v3 import (  # noqa: E402
    PilotBlock,
    PilotV3Error,
    expected_pilot_configs,
    expected_pilot_plan_payload,
    require_v3_author_freeze,
    resolve_pilot_block,
    validate_health_report as validate_v3_health_report,
)
from talif_msresnet.pilot_v4 import (  # noqa: E402
    PilotBlock as PilotV4Block,
    PilotV4Error,
    expected_pilot_configs as expected_v4_pilot_configs,
    expected_pilot_plan_payload as expected_v4_pilot_plan_payload,
    require_v4_author_freeze,
    resolve_pilot_block as resolve_v4_pilot_block,
    validate_health_report as validate_v4_health_report,
)
from talif_msresnet.pilot_v5 import (
    PilotBlock as PilotV5Block,
    PilotV5Error,
    expected_pilot_configs as expected_v5_pilot_configs,
    expected_pilot_plan_payload as expected_v5_pilot_plan_payload,
    require_v5_author_freeze,
    resolve_pilot_block as resolve_v5_pilot_block,
    validate_health_report as validate_v5_health_report,
)
from talif_msresnet.pilot_v6 import (  # noqa: E402
    PilotBlock as PilotV6Block,
    PilotV6Error,
    expected_pilot_configs as expected_v6_pilot_configs,
    expected_pilot_plan_payload as expected_v6_pilot_plan_payload,
    require_v6_author_freeze,
    resolve_pilot_block as resolve_v6_pilot_block,
    validate_health_report as validate_v6_health_report,
)
from talif_msresnet.pilot_v7 import (  # noqa: E402
    PilotBlock as PilotV7Block,
    PilotV7Error,
    expected_pilot_configs as expected_v7_pilot_configs,
    expected_pilot_plan_payload as expected_v7_pilot_plan_payload,
    require_v7_author_freeze,
    resolve_pilot_block as resolve_v7_pilot_block,
    validate_health_report as validate_v7_health_report,
)
from talif_msresnet.freeze import FreezeGateError, verify_formal_freeze  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.utils import (  # noqa: E402
    DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
    atomic_write_json,
    file_lock,
    sha256_file,
)
from pilot_health_gate import (  # noqa: E402
    HealthGateError,
    validate_current_runtime_against_health_report,
    validate_pilot_health_report,
)
from pilot_health_gate_v3 import (  # noqa: E402
    validate_current_runtime_against_health_report as validate_v3_current_runtime,
)
from pilot_health_gate_v4 import (  # noqa: E402
    validate_current_runtime_against_health_report as validate_v4_current_runtime,
)
from pilot_health_gate_v5 import (
    validate_current_runtime_against_health_report as validate_v5_current_runtime,
)
from pilot_health_gate_v6 import (
    validate_current_runtime_against_health_report as validate_v6_current_runtime,
)
from pilot_health_gate_v7 import (
    validate_current_runtime_against_health_report as validate_v7_current_runtime,
)

PILOT_PLAN_VERSION = 1
PILOT_PLAN_PATH = PROJECT_ROOT / "environment" / "unfrozen_pilot_plan.json"


def _validated_pilot_plan_path(value: str | Path) -> Path:
    path = _project_path(value)
    default = PILOT_PLAN_PATH.resolve()
    if path != default and (
        path.parent != default.parent
        or not path.name.startswith("unfrozen_pilot_plan_")
        or path.suffix.lower() != ".json"
    ):
        raise ValueError(
            "Pilot plans must use environment/unfrozen_pilot_plan.json or a versioned "
            "environment/unfrozen_pilot_plan_*.json sibling"
        )
    return path


def _read_manifest(config_dir: Path) -> list[dict[str, str]]:
    manifest = config_dir / "run_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; run generate_run_configs.py first")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _event(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _validate_pilot_output_root(output_root: str | Path | None, formal_root: str | Path) -> Path:
    if output_root is None:
        raise ValueError("Unfrozen pilots require an explicit separate --output-root")
    selected = _project_path(output_root)
    formal = _project_path(formal_root)
    if selected == formal or formal in selected.parents or selected in formal.parents:
        raise ValueError(
            "Pilot output must be outside the formal results root so it cannot enter analysis"
        )
    return selected


def _pilot_block(row: dict[str, str]) -> tuple[str, str, int, int, int]:
    required = ("experiment", "dataset", "depth", "time_steps", "seed")
    missing = [key for key in required if not str(row.get(key, "")).strip()]
    if missing:
        raise ValueError(f"Pilot manifest row is missing block fields: {missing}")
    return (
        str(row["experiment"]),
        str(row["dataset"]),
        int(row["depth"]),
        int(row["time_steps"]),
        int(row["seed"]),
    )


def _is_v2_pilot(protocol: Mapping[str, Any]) -> bool:
    return protocol.get("protocol_version") == 2 and protocol.get("study_stage") == "pilot"


def _is_v3_pilot(protocol: Mapping[str, Any]) -> bool:
    return (
        protocol.get("protocol_version") == 3
        and protocol.get("study_stage") == "pilot_and_formal"
    )


def _is_v4_pilot(protocol: Mapping[str, Any]) -> bool:
    return (
        protocol.get("protocol_version") == 4
        and protocol.get("study_stage") == "pilot_and_formal"
    )


def _is_v5_pilot(protocol: Mapping[str, Any]) -> bool:
    return (
        protocol.get("protocol_version") == 5
        and protocol.get("study_stage") == "pilot_and_formal"
    )


def _is_v6_pilot(protocol: Mapping[str, Any]) -> bool:
    return (
        protocol.get("protocol_version") == 6
        and protocol.get("study_stage") == "health_pilot_formal"
    )


def _is_v7_pilot(protocol: Mapping[str, Any]) -> bool:
    return (
        protocol.get("protocol_version") == 7
        and protocol.get("study_stage") == "health_pilot_formal"
    )


def _validate_v2_pilot_launch_contract(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    selected_rows: Sequence[dict[str, str]],
    output_root: str | Path | None,
    pilot_plan: str | Path | None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Bind the v2 quartet to its protocol paths and prior PASS health report."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise ValueError("The v2 pilot protocol has no pilot_acceptance mapping")
    if len(selected_rows) != 4:
        raise ValueError("The v2 pilot must launch one complete four-run C1-C4 block")
    conditions = [str(row.get("condition")) for row in selected_rows]
    if conditions != ["C1", "C2", "C3", "C4"]:
        raise ValueError("The v2 pilot launch order must be exactly C1, C2, C3, C4")
    if len({_pilot_block(row) for row in selected_rows}) != 1:
        raise ValueError("The v2 pilot rows must belong to one dataset/depth/T/seed block")

    expected_output = _project_path(str(acceptance.get("pilot_output_root", "")))
    if output_root is not None and _project_path(output_root) != expected_output:
        raise ValueError(
            "--output-root cannot override pilot_acceptance.pilot_output_root"
        )
    expected_plan = _validated_pilot_plan_path(
        str(acceptance.get("pilot_plan", ""))
    )
    if pilot_plan is not None and _validated_pilot_plan_path(pilot_plan) != expected_plan:
        raise ValueError("--pilot-plan cannot override pilot_acceptance.pilot_plan")
    health_path = _project_path(str(acceptance.get("health_output", "")))
    try:
        health_report = validate_pilot_health_report(
            health_path,
            protocol_path,
            repository_root=PROJECT_ROOT,
            require_current_tracked_clean=True,
        )
    except HealthGateError as exc:
        raise RuntimeError(f"V2 pilot health-report gate blocked execution: {exc}") from exc
    return expected_output, expected_plan, health_path, health_report


def _validate_v3_pilot_launch_contract(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    selected_rows: Sequence[dict[str, str]],
    dataset: str | None,
    output_root: str | Path | None,
    pilot_plan: str | Path | None,
) -> tuple[PilotBlock, Path, Path, dict[str, Any]]:
    """Bind one exact v3 dataset C1/C2 block to its one-time health report."""

    if dataset is None:
        raise ValueError("Protocol v3 pilot execution requires --dataset")
    try:
        block = resolve_pilot_block(
            protocol, dataset, repository_root=PROJECT_ROOT
        )
    except PilotV3Error as exc:
        raise ValueError(f"Invalid v3 pilot block: {exc}") from exc
    if len(selected_rows) != 2:
        raise ValueError("The v3 pilot must launch exactly one complete C1/C2 block")
    conditions = [str(row.get("condition")) for row in selected_rows]
    if conditions != list(block.conditions):
        raise ValueError("The v3 pilot launch order must be exactly C1, C2")
    if len({_pilot_block(row) for row in selected_rows}) != 1:
        raise ValueError("The v3 pilot rows must belong to one dataset/depth/T/seed block")
    selected_block = _pilot_block(selected_rows[0])
    if selected_block[1] != block.dataset or selected_block[4] != block.seed:
        raise ValueError("The selected v3 pilot block differs from the protocol binding")
    if output_root is not None and _project_path(output_root) != block.pilot_output_root:
        raise ValueError("--output-root cannot override the v3 dataset pilot output root")
    if pilot_plan is not None and _project_path(pilot_plan) != block.pilot_plan:
        raise ValueError("--pilot-plan cannot override the v3 dataset pilot plan")
    try:
        report = validate_v3_health_report(
            block.health_output,
            protocol_path,
            block.dataset,
            repository_root=PROJECT_ROOT,
            require_current_tracked_clean=True,
        )
    except PilotV3Error as exc:
        raise RuntimeError(f"V3 pilot health-report gate blocked execution: {exc}") from exc
    return block, block.pilot_output_root, block.pilot_plan, report


def _validate_v4_pilot_launch_contract(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    selected_rows: Sequence[dict[str, str]],
    dataset: str | None,
    output_root: str | Path | None,
    pilot_plan: str | Path | None,
) -> tuple[PilotV4Block, Path, Path, dict[str, Any]]:
    """Bind one exact v4 dataset C1/C2 block to its one-shot health PASS."""

    if dataset is None:
        raise ValueError("Protocol v4 pilot execution requires --dataset")
    try:
        block = resolve_v4_pilot_block(
            protocol, dataset, repository_root=PROJECT_ROOT
        )
    except PilotV4Error as exc:
        raise ValueError(f"Invalid v4 pilot block: {exc}") from exc
    if len(selected_rows) != 2:
        raise ValueError("The v4 pilot must launch exactly one complete C1/C2 block")
    conditions = [str(row.get("condition")) for row in selected_rows]
    if conditions != list(block.conditions):
        raise ValueError("The v4 pilot launch order must be exactly C1, C2")
    if len({_pilot_block(row) for row in selected_rows}) != 1:
        raise ValueError("The v4 pilot rows must belong to one dataset/depth/T/seed block")
    selected_block = _pilot_block(selected_rows[0])
    if selected_block[1] != block.dataset or selected_block[4] != block.seed:
        raise ValueError("The selected v4 pilot block differs from the protocol binding")
    if output_root is not None and _project_path(output_root) != block.pilot_output_root:
        raise ValueError("--output-root cannot override the v4 dataset pilot output root")
    if pilot_plan is not None and _project_path(pilot_plan) != block.pilot_plan:
        raise ValueError("--pilot-plan cannot override the v4 dataset pilot plan")
    try:
        report = validate_v4_health_report(
            block.health_output,
            protocol_path,
            block.dataset,
            repository_root=PROJECT_ROOT,
            require_current_tracked_clean=True,
        )
    except PilotV4Error as exc:
        raise RuntimeError(f"V4 pilot health-report gate blocked execution: {exc}") from exc
    return block, block.pilot_output_root, block.pilot_plan, report


def _validate_v5_pilot_launch_contract(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    selected_rows: Sequence[dict[str, str]],
    dataset: str | None,
    output_root: str | Path | None,
    pilot_plan: str | Path | None,
) -> tuple[PilotV5Block, Path, Path, dict[str, Any]]:
    """Bind a v5 pilot-seed C1/C2 block to its disjoint health-seed PASS."""

    if dataset is None:
        raise ValueError("Protocol v5 pilot execution requires --dataset")
    try:
        block = resolve_v5_pilot_block(
            protocol, dataset, repository_root=PROJECT_ROOT
        )
    except PilotV5Error as exc:
        raise ValueError(f"Invalid v5 pilot block: {exc}") from exc
    if len(selected_rows) != 2:
        raise ValueError("The v5 pilot must launch exactly one complete C1/C2 block")
    conditions = [str(row.get("condition")) for row in selected_rows]
    if conditions != list(block.conditions):
        raise ValueError("The v5 pilot launch order must be exactly C1, C2")
    if len({_pilot_block(row) for row in selected_rows}) != 1:
        raise ValueError("The v5 pilot rows must belong to one dataset/depth/T/seed block")
    selected_block = _pilot_block(selected_rows[0])
    if selected_block[1] != block.dataset or selected_block[4] != block.pilot_seed:
        raise ValueError("The selected v5 pilot block differs from the pilot-seed binding")
    if output_root is not None and _project_path(output_root) != block.pilot_output_root:
        raise ValueError("--output-root cannot override the v5 dataset pilot output root")
    if pilot_plan is not None and _project_path(pilot_plan) != block.pilot_plan:
        raise ValueError("--pilot-plan cannot override the v5 dataset pilot plan")
    try:
        report = validate_v5_health_report(
            block.health_output,
            protocol_path,
            block.dataset,
            repository_root=PROJECT_ROOT,
            require_current_tracked_clean=True,
        )
    except PilotV5Error as exc:
        raise RuntimeError(f"V5 pilot health-report gate blocked execution: {exc}") from exc
    return block, block.pilot_output_root, block.pilot_plan, report


def _validate_v6_pilot_launch_contract(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    selected_rows: Sequence[dict[str, str]],
    dataset: str | None,
    output_root: str | Path | None,
    pilot_plan: str | Path | None,
) -> tuple[PilotV6Block, Path, Path, dict[str, Any]]:
    """Bind the exact six-condition V6 pilot to its disjoint health PASS."""

    try:
        block = resolve_v6_pilot_block(protocol, repository_root=PROJECT_ROOT)
    except PilotV6Error as exc:
        raise ValueError(f"Invalid V6 pilot block: {exc}") from exc
    if dataset is None:
        raise ValueError("Protocol V6 pilot execution requires --dataset cifar100")
    if dataset != block.dataset:
        raise ValueError(
            f"Protocol V6 pilot dataset must be {block.dataset!r}, got {dataset!r}"
        )
    if len(selected_rows) != len(block.conditions):
        raise ValueError("The V6 pilot must launch exactly one complete six-condition block")
    conditions = [str(row.get("condition")) for row in selected_rows]
    if conditions != list(block.conditions):
        raise ValueError(
            "The V6 pilot launch order must be exactly " + ", ".join(block.conditions)
        )
    if len({_pilot_block(row) for row in selected_rows}) != 1:
        raise ValueError(
            "The V6 pilot rows must belong to one dataset/depth/T/seed block"
        )
    selected_block = _pilot_block(selected_rows[0])
    if selected_block[1] != block.dataset or selected_block[4] != block.pilot_seed:
        raise ValueError("The selected V6 pilot block differs from the pilot-seed binding")
    if output_root is not None and _project_path(output_root) != block.pilot_output_root:
        raise ValueError("--output-root cannot override the V6 pilot output root")
    if pilot_plan is not None and _project_path(pilot_plan) != block.pilot_plan:
        raise ValueError("--pilot-plan cannot override the V6 pilot plan")
    try:
        report = validate_v6_health_report(
            block.health_output,
            protocol_path,
            repository_root=PROJECT_ROOT,
            require_current_tracked_clean=True,
        )
    except PilotV6Error as exc:
        raise RuntimeError(f"V6 pilot health-report gate blocked execution: {exc}") from exc
    return block, block.pilot_output_root, block.pilot_plan, report


def _validate_v7_pilot_launch_contract(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    selected_rows: Sequence[dict[str, str]],
    dataset: str | None,
    output_root: str | Path | None,
    pilot_plan: str | Path | None,
) -> tuple[PilotV7Block, Path, Path, dict[str, Any]]:
    """Bind the exact six-condition V7 pilot to its disjoint health PASS."""

    try:
        block = resolve_v7_pilot_block(protocol, repository_root=PROJECT_ROOT)
    except PilotV7Error as exc:
        raise ValueError(f"Invalid V7 pilot block: {exc}") from exc
    if dataset is None:
        raise ValueError("Protocol V7 pilot execution requires --dataset cifar100")
    if dataset != block.dataset:
        raise ValueError(
            f"Protocol V7 pilot dataset must be {block.dataset!r}, got {dataset!r}"
        )
    if len(selected_rows) != len(block.conditions):
        raise ValueError("The V7 pilot must launch exactly one complete six-condition block")
    conditions = [str(row.get("condition")) for row in selected_rows]
    if conditions != list(block.conditions):
        raise ValueError(
            "The V7 pilot launch order must be exactly " + ", ".join(block.conditions)
        )
    if len({_pilot_block(row) for row in selected_rows}) != 1:
        raise ValueError(
            "The V7 pilot rows must belong to one dataset/depth/T/seed block"
        )
    selected_block = _pilot_block(selected_rows[0])
    if selected_block[1] != block.dataset or selected_block[4] != block.pilot_seed:
        raise ValueError("The selected V7 pilot block differs from the pilot-seed binding")
    if output_root is not None and _project_path(output_root) != block.pilot_output_root:
        raise ValueError("--output-root cannot override the V7 pilot output root")
    if pilot_plan is not None and _project_path(pilot_plan) != block.pilot_plan:
        raise ValueError("--pilot-plan cannot override the V7 pilot plan")
    try:
        report = validate_v7_health_report(
            block.health_output,
            protocol_path,
            repository_root=PROJECT_ROOT,
            require_current_tracked_clean=True,
        )
    except PilotV7Error as exc:
        raise RuntimeError(f"V7 pilot health-report gate blocked execution: {exc}") from exc
    return block, block.pilot_output_root, block.pilot_plan, report


def _prepare_v3_pilot_plan(
    *,
    block: PilotBlock,
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
    health_report: Mapping[str, Any],
) -> Path:
    expected = expected_pilot_plan_payload(
        block=block,
        protocol_path=protocol_path,
        config_dir=config_dir,
        manifest_rows=manifest_rows,
        health_report=health_report,
        repository_root=PROJECT_ROOT,
    )
    with file_lock(block.pilot_plan):
        if block.pilot_plan.exists():
            existing = json.loads(block.pilot_plan.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    "A different v3 dataset pilot plan already exists; preserve it and "
                    "create a new protocol generation rather than overwriting it"
                )
        else:
            atomic_write_json(block.pilot_plan, expected)
    return block.pilot_plan.resolve()


def _prepare_v4_pilot_plan(
    *,
    block: PilotV4Block,
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
    health_report: Mapping[str, Any],
) -> Path:
    expected = expected_v4_pilot_plan_payload(
        block=block,
        protocol_path=protocol_path,
        config_dir=config_dir,
        manifest_rows=manifest_rows,
        health_report=health_report,
        repository_root=PROJECT_ROOT,
    )
    with file_lock(block.pilot_plan):
        if block.pilot_plan.exists():
            existing = json.loads(block.pilot_plan.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    "A different v4 dataset pilot plan already exists; preserve it and "
                    "create a new protocol generation rather than overwriting it"
                )
        else:
            atomic_write_json(block.pilot_plan, expected)
    return block.pilot_plan.resolve()


def _prepare_v5_pilot_plan(
    *,
    block: PilotV5Block,
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
    health_report: Mapping[str, Any],
) -> Path:
    expected = expected_v5_pilot_plan_payload(
        block=block,
        protocol_path=protocol_path,
        config_dir=config_dir,
        manifest_rows=manifest_rows,
        health_report=health_report,
        repository_root=PROJECT_ROOT,
    )
    with file_lock(block.pilot_plan):
        if block.pilot_plan.exists():
            existing = json.loads(block.pilot_plan.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    "A different v5 dataset pilot plan already exists; preserve it and "
                    "create a new protocol generation rather than overwriting it"
                )
        else:
            atomic_write_json(block.pilot_plan, expected)
    return block.pilot_plan.resolve()


def _prepare_v6_pilot_plan(
    *,
    block: PilotV6Block,
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
    health_report: Mapping[str, Any],
) -> Path:
    expected = expected_v6_pilot_plan_payload(
        block=block,
        protocol_path=protocol_path,
        config_dir=config_dir,
        manifest_rows=manifest_rows,
        health_report=health_report,
        repository_root=PROJECT_ROOT,
    )
    with file_lock(block.pilot_plan):
        if block.pilot_plan.exists():
            existing = json.loads(block.pilot_plan.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    "A different V6 pilot plan already exists; preserve it and create "
                    "a new protocol generation rather than overwriting it"
                )
        else:
            atomic_write_json(block.pilot_plan, expected)
    return block.pilot_plan.resolve()


def _prepare_v7_pilot_plan(
    *,
    block: PilotV7Block,
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
    health_report: Mapping[str, Any],
) -> Path:
    expected = expected_v7_pilot_plan_payload(
        block=block,
        protocol_path=protocol_path,
        config_dir=config_dir,
        manifest_rows=manifest_rows,
        health_report=health_report,
        repository_root=PROJECT_ROOT,
    )
    with file_lock(block.pilot_plan):
        if block.pilot_plan.exists():
            existing = json.loads(block.pilot_plan.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    "A different V7 pilot plan already exists; preserve it and create "
                    "a new protocol generation rather than overwriting it"
                )
        else:
            atomic_write_json(block.pilot_plan, expected)
    return block.pilot_plan.resolve()


def _validate_v2_generated_matrix(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
    matrix_manifest: Mapping[str, Any],
    expected_raw_runs: Sequence[Mapping[str, Any]] | None = None,
    label: str = "V2",
) -> None:
    """Require a disk matrix to equal its protocol-generated rows exactly."""

    expected = [
        validate_run_mapping(raw, protocol)
        for raw in (
            expected_raw_runs
            if expected_raw_runs is not None
            else generate_run_matrix(protocol)
        )
    ]
    expected_ids = [config.runtime.run_id for config in expected]
    observed_ids = [str(row.get("run_id")) for row in manifest_rows]
    if observed_ids != expected_ids:
        raise RuntimeError(f"{label} generated run manifest differs from the protocol")
    json_rows = matrix_manifest.get("runs")
    if not isinstance(json_rows, list) or [
        str(row.get("run_id")) for row in json_rows if isinstance(row, Mapping)
    ] != expected_ids:
        raise RuntimeError(
            f"{label} matrix manifest differs from the protocol-generated order"
        )

    for expected_config, row in zip(expected, manifest_rows):
        run_id = expected_config.runtime.run_id
        expected_file = f"{run_id}.yaml"
        if row.get("config_file") != expected_file:
            raise RuntimeError(f"{run_id}: generated config filename is not canonical")
        config_path = config_dir / expected_file
        observed_config = load_run_config(config_path, protocol_path)
        if observed_config.as_dict() != expected_config.as_dict():
            raise RuntimeError(
                f"{run_id}: generated YAML differs from the in-memory protocol matrix"
            )
        expected_fields = {
            "experiment": expected_config.experiment,
            "dataset": expected_config.data.dataset,
            "depth": str(expected_config.model.depth),
            "time_steps": str(expected_config.model.time_steps),
            "condition": expected_config.model.condition,
            "topology": expected_config.model.topology,
            "neuron": expected_config.model.neuron,
            "seed": str(expected_config.runtime.seed),
            "config_hash": expected_config.config_hash,
            "protocol_hash": expected_config.analysis["protocol_hash"],
        }
        for key, expected_value in expected_fields.items():
            if str(row.get(key)) != str(expected_value):
                raise RuntimeError(
                    f"{run_id}: run_manifest.csv {key} differs from the protocol matrix"
                )
        if row.get("config_file_sha256") != sha256_file(config_path):
            raise RuntimeError(f"{run_id}: run-manifest config file SHA-256 mismatch")


def _prepare_unfrozen_pilot_plan(
    *,
    all_rows: Sequence[dict[str, str]],
    selected_rows: Sequence[dict[str, str]],
    config_dir: Path,
    protocol_path: Path,
    protocol_hash: str,
    output_root: Path,
    formal_root: str | Path,
    plan_path: Path | None = None,
    health_report_path: Path | None = None,
    health_report: Mapping[str, Any] | None = None,
) -> Path:
    """Create or verify the global four-cell non-reportable pilot plan."""

    plan_path = _validated_pilot_plan_path(plan_path or PILOT_PLAN_PATH)
    blocks = {_pilot_block(row) for row in selected_rows}
    if len(blocks) != 1:
        raise ValueError("An unfrozen pilot invocation must select one dataset/depth/T/seed block")
    block = next(iter(blocks))
    block_rows = [row for row in all_rows if _pilot_block(row) == block]
    by_condition = {str(row.get("condition")): row for row in block_rows}
    if set(by_condition) != {"C1", "C2", "C3", "C4"} or len(block_rows) != 4:
        raise ValueError("The pilot plan requires exactly one C1-C4 quartet for the selected seed")
    runs: list[dict[str, str]] = []
    for condition in ("C1", "C2", "C3", "C4"):
        row = by_condition[condition]
        config_path = (config_dir / row["config_file"]).resolve()
        resolved = load_run_config(config_path, protocol_path)
        if resolved.runtime.run_id != row["run_id"]:
            raise ValueError(f"Pilot run_id differs from generated config: {config_path}")
        runs.append(
            {
                "condition": condition,
                "run_id": resolved.runtime.run_id,
                "config_file": artifact_path_reference(config_path, PROJECT_ROOT),
                "config_file_sha256": sha256_file(config_path),
                "config_hash": resolved.config_hash,
            }
        )
    selected_ids = {str(row["run_id"]) for row in selected_rows}
    planned_ids = {run["run_id"] for run in runs}
    if len(selected_rows) != 4 or selected_ids != planned_ids:
        raise ValueError("Pilot execution must select the complete derived C1-C4 plan")
    formal = _project_path(formal_root)
    expected = {
        "version": PILOT_PLAN_VERSION,
        "non_reportable": True,
        "protocol_path": artifact_path_reference(protocol_path, PROJECT_ROOT),
        "protocol_hash": protocol_hash,
        "pilot_output_root": artifact_path_reference(output_root, PROJECT_ROOT),
        "formal_output_root": artifact_path_reference(formal, PROJECT_ROOT),
        "block": {
            "experiment": block[0],
            "dataset": block[1],
            "depth": block[2],
            "time_steps": block[3],
            "seed": block[4],
        },
        "runs": runs,
    }
    if (health_report_path is None) != (health_report is None):
        raise ValueError("Pilot health-report path and payload must be supplied together")
    if health_report_path is not None and health_report is not None:
        expected.update(
            {
                "version": 2,
                "acceptance_hash": health_report.get("acceptance_hash"),
                "git_commit": health_report.get("git_commit"),
                "health_report": artifact_path_reference(
                    health_report_path, PROJECT_ROOT
                ),
                "health_report_sha256": sha256_file(health_report_path),
            }
        )
    with file_lock(plan_path):
        if plan_path.exists():
            existing = json.loads(plan_path.read_text(encoding="utf-8"))
            if existing != expected:
                raise ValueError(
                    "A different unfrozen pilot plan already exists; archive it with the study "
                    "record instead of changing seed, block, protocol, or output root"
                )
        else:
            atomic_write_json(plan_path, expected)
    return plan_path.resolve()


def _load_metrics(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "seed_metrics.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Malformed terminal metrics: {path}")
    return value


def _validate_run_id(selected_run_id: str, configured_run_id: str) -> None:
    if configured_run_id != selected_run_id:
        raise RuntimeError(
            f"Selected file/manifest run_id={selected_run_id!r} differs from YAML runtime.run_id="
            f"{configured_run_id!r}; do not rename generated config files"
        )


def _next_attempt(run_dir: Path) -> tuple[int, Path, Path]:
    attempts: list[int] = []
    for path in run_dir.glob("attempt_*.*.log"):
        try:
            attempts.append(int(path.name.split("_")[1].split(".")[0]))
        except (IndexError, ValueError):
            continue
    number = max(attempts, default=0) + 1
    return (
        number,
        run_dir / f"attempt_{number:03d}.stdout.log",
        run_dir / f"attempt_{number:03d}.stderr.log",
    )


def _continuation_action(
    run_dir: Path,
    run_id: str,
    planned_config_hash: str,
    resume_matrix: bool,
    forbid_fresh_after_attempt: bool = False,
) -> tuple[str, Path | None]:
    """Return fresh/resume/skip without discarding an earlier attempt."""

    metrics = _load_metrics(run_dir)
    if metrics is not None:
        if str(metrics.get("run_id")) != run_id:
            raise RuntimeError(f"{run_id}: seed_metrics.json belongs to another run")
        if str(metrics.get("config_hash")) != planned_config_hash:
            raise RuntimeError(f"{run_id}: existing metrics use a different scientific config hash")
        if metrics.get("status") == "complete":
            if (
                metrics.get("failed") not in (0, "0", False)
                or not (run_dir / "best.pt").exists()
            ):
                raise RuntimeError(f"{run_id}: terminal metrics claim completion but artifacts are inconsistent")
            if resume_matrix:
                return "skip", None
            raise FileExistsError(
                f"{run_id} is already complete; use --resume-matrix to skip completed runs safely"
            )

    old_logs = list(run_dir.glob("attempt_*.*.log"))
    old_logs.extend(
        path for path in (run_dir / "matrix_stdout.log", run_dir / "matrix_stderr.log")
        if path.exists()
    )
    scientific_artifacts = [
        run_dir / "resolved_config.json",
        run_dir / "run_manifest.json",
        run_dir / "seed_metrics.json",
        run_dir / "best.pt",
        run_dir / "last.pt",
        run_dir / "failed.pt",
    ]
    if not resume_matrix:
        if old_logs or any(path.exists() for path in scientific_artifacts):
            raise FileExistsError(
                f"{run_id} already has an attempt; rerun with --resume-matrix to preserve it"
            )
        return "fresh", None

    last_checkpoint = run_dir / "last.pt"
    if last_checkpoint.exists():
        return "resume", last_checkpoint.resolve()
    if any(path.exists() for path in scientific_artifacts):
        raise RuntimeError(
            f"{run_id} has scientific artifacts but no last.pt. Safe continuation is impossible; "
            "archive the entire run directory, document the failed attempt, and restart it explicitly."
        )
    if forbid_fresh_after_attempt and old_logs:
        raise RuntimeError(
            f"{run_id} has an earlier attempt but no last.pt. The frozen TA-LIF "
            "protocol forbids a fresh retry under the consumed run identity"
        )
    return "fresh", None


def run_matrix(
    config_dir: str | Path | None,
    config_path: str | Path | None = None,
    output_root: str | Path | None = None,
    device: str | None = None,
    protocol_path: str | Path = PROJECT_ROOT / "configs" / "protocol.yaml",
    allow_unfrozen_pilot: bool = False,
    pilot_plan: str | Path | None = None,
    resume_matrix: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    limit_batches: int | None = None,
    experiment: str | None = None,
    condition: str | None = None,
    dataset: str | None = None,
    stop_on_error: bool = False,
) -> int:
    protocol_path = _project_path(protocol_path)
    protocol = load_protocol(protocol_path)
    protocol_version = int(protocol.get("protocol_version", 1))
    if protocol_version in (3, 4, 5, 6, 7) and dry_run:
        raise ValueError(
            f"Protocol v{protocol_version} matrix dry-runs are disabled; use "
            "generate_run_configs.py --dry-run and reserve training commands for "
            "the author-frozen health/pilot gates"
        )
    pilot_active = bool(allow_unfrozen_pilot)
    v3_pilot = bool(pilot_active and _is_v3_pilot(protocol))
    v4_pilot = bool(pilot_active and _is_v4_pilot(protocol))
    v5_pilot = bool(pilot_active and _is_v5_pilot(protocol))
    v6_pilot = bool(pilot_active and _is_v6_pilot(protocol))
    v7_pilot = bool(pilot_active and _is_v7_pilot(protocol))
    v7_formal = bool(not pilot_active and _is_v7_pilot(protocol))
    if v7_formal and device not in (None, "auto"):
        raise ValueError(
            "Protocol V7 formal execution must retain runtime.device='auto'; "
            "set CUDA_VISIBLE_DEVICES=0 and omit --device (or use --device auto)"
        )
    if config_dir is None and config_path is None:
        if protocol_version in (3, 4, 5, 6, 7):
            matrix_key = "pilot_matrix" if pilot_active else "formal_matrix"
            config_dir = PROJECT_ROOT / artifact_paths_for_protocol(protocol)[matrix_key]
        else:
            config_dir = PROJECT_ROOT / "configs" / "generated"
    config_dir = _project_path(config_dir) if config_dir is not None else None
    if config_path is not None:
        selected = _project_path(config_path)
        config_dir = selected.parent
        all_rows = [{"run_id": selected.stem, "config_file": selected.name}]
        rows = list(all_rows)
    elif config_dir is not None:
        all_rows = _read_manifest(config_dir)
        rows = list(all_rows)
    else:
        raise ValueError("Either config_dir or config_path is required")
    if experiment:
        rows = [row for row in rows if row.get("experiment") == experiment]
    if condition:
        rows = [row for row in rows if row.get("condition") == condition]
    if dataset:
        rows = [row for row in rows if row.get("dataset") == dataset]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        rows = rows[:limit]
    if not rows:
        raise ValueError("No run configurations selected")
    v2_pilot = bool(pilot_active and _is_v2_pilot(protocol))
    pilot_output_root: Path | None = None
    selected_pilot_plan: Path | None = (
        None
        if v3_pilot or v4_pilot or v5_pilot or v6_pilot or v7_pilot
        else _validated_pilot_plan_path(pilot_plan or PILOT_PLAN_PATH)
    )
    pilot_health_path: Path | None = None
    pilot_health_report: dict[str, Any] | None = None
    pilot_reference_config = None
    pilot_launch_context: dict[str, Any] | None = None
    v3_block: PilotBlock | None = None
    v4_block: PilotV4Block | None = None
    v5_block: PilotV5Block | None = None
    v6_block: PilotV6Block | None = None
    v7_block: PilotV7Block | None = None
    if pilot_active:
        if config_path is not None:
            raise ValueError("Unfrozen pilots require --config-dir so the full plan is auditable")
        if dry_run:
            raise ValueError("--allow-unfrozen-pilot cannot be combined with --dry-run")
        if v7_pilot:
            try:
                require_v7_author_freeze(protocol, repository_root=PROJECT_ROOT)
            except PilotV7Error as exc:
                raise ValueError(f"Protocol V7 pilot requires author freeze: {exc}") from exc
            if condition is not None or limit is not None or experiment is not None:
                raise ValueError(
                    "Protocol V7 pilot selection permits only --dataset cifar100; "
                    "condition, limit, and experiment filters are forbidden"
                )
            assert config_dir is not None
            expected_config_dir = _project_path(
                artifact_paths_for_protocol(protocol)["pilot_matrix"]
            )
            if config_dir != expected_config_dir:
                raise ValueError(
                    "Protocol V7 pilot execution must use artifact_paths.pilot_matrix"
                )
            (
                v7_block,
                pilot_output_root,
                selected_pilot_plan,
                pilot_health_report,
            ) = _validate_v7_pilot_launch_contract(
                protocol=protocol,
                protocol_path=protocol_path,
                selected_rows=rows,
                dataset=dataset,
                output_root=output_root,
                pilot_plan=pilot_plan,
            )
            pilot_health_path = v7_block.health_output
        elif v6_pilot:
            try:
                require_v6_author_freeze(protocol, repository_root=PROJECT_ROOT)
            except PilotV6Error as exc:
                raise ValueError(f"Protocol V6 pilot requires author freeze: {exc}") from exc
            if condition is not None or limit is not None or experiment is not None:
                raise ValueError(
                    "Protocol V6 pilot selection permits only --dataset cifar100; "
                    "condition, limit, and experiment filters are forbidden"
                )
            assert config_dir is not None
            expected_config_dir = _project_path(
                artifact_paths_for_protocol(protocol)["pilot_matrix"]
            )
            if config_dir != expected_config_dir:
                raise ValueError(
                    "Protocol V6 pilot execution must use artifact_paths.pilot_matrix"
                )
            (
                v6_block,
                pilot_output_root,
                selected_pilot_plan,
                pilot_health_report,
            ) = _validate_v6_pilot_launch_contract(
                protocol=protocol,
                protocol_path=protocol_path,
                selected_rows=rows,
                dataset=dataset,
                output_root=output_root,
                pilot_plan=pilot_plan,
            )
            pilot_health_path = v6_block.health_output
        elif v5_pilot:
            try:
                require_v5_author_freeze(protocol, repository_root=PROJECT_ROOT)
            except PilotV5Error as exc:
                raise ValueError(f"Protocol v5 pilot requires author freeze: {exc}") from exc
            if condition is not None or limit is not None or experiment is not None:
                raise ValueError(
                    "Protocol v5 pilot selection permits only --dataset; condition, limit, "
                    "and experiment filters are forbidden"
                )
            assert config_dir is not None
            expected_config_dir = _project_path(
                artifact_paths_for_protocol(protocol)["pilot_matrix"]
            )
            if config_dir != expected_config_dir:
                raise ValueError(
                    "Protocol v5 pilot execution must use artifact_paths.pilot_matrix"
                )
            (
                v5_block,
                pilot_output_root,
                selected_pilot_plan,
                pilot_health_report,
            ) = _validate_v5_pilot_launch_contract(
                protocol=protocol,
                protocol_path=protocol_path,
                selected_rows=rows,
                dataset=dataset,
                output_root=output_root,
                pilot_plan=pilot_plan,
            )
            pilot_health_path = v5_block.health_output
        elif v4_pilot:
            try:
                require_v4_author_freeze(protocol, repository_root=PROJECT_ROOT)
            except PilotV4Error as exc:
                raise ValueError(f"Protocol v4 pilot requires author freeze: {exc}") from exc
            if condition is not None or limit is not None or experiment is not None:
                raise ValueError(
                    "Protocol v4 pilot selection permits only --dataset; condition, limit, "
                    "and experiment filters are forbidden"
                )
            assert config_dir is not None
            expected_config_dir = _project_path(
                artifact_paths_for_protocol(protocol)["pilot_matrix"]
            )
            if config_dir != expected_config_dir:
                raise ValueError(
                    "Protocol v4 pilot execution must use artifact_paths.pilot_matrix"
                )
            (
                v4_block,
                pilot_output_root,
                selected_pilot_plan,
                pilot_health_report,
            ) = _validate_v4_pilot_launch_contract(
                protocol=protocol,
                protocol_path=protocol_path,
                selected_rows=rows,
                dataset=dataset,
                output_root=output_root,
                pilot_plan=pilot_plan,
            )
            pilot_health_path = v4_block.health_output
        elif v3_pilot:
            try:
                require_v3_author_freeze(protocol, repository_root=PROJECT_ROOT)
            except PilotV3Error as exc:
                raise ValueError(f"Protocol v3 pilot requires author freeze: {exc}") from exc
            if condition is not None or limit is not None or experiment is not None:
                raise ValueError(
                    "Protocol v3 pilot selection permits only --dataset; condition, limit, "
                    "and experiment filters are forbidden"
                )
            assert config_dir is not None
            expected_config_dir = _project_path(
                artifact_paths_for_protocol(protocol)["pilot_matrix"]
            )
            if config_dir != expected_config_dir:
                raise ValueError(
                    "Protocol v3 pilot execution must use artifact_paths.pilot_matrix"
                )
            (
                v3_block,
                pilot_output_root,
                selected_pilot_plan,
                pilot_health_report,
            ) = _validate_v3_pilot_launch_contract(
                protocol=protocol,
                protocol_path=protocol_path,
                selected_rows=rows,
                dataset=dataset,
                output_root=output_root,
                pilot_plan=pilot_plan,
            )
            pilot_health_path = v3_block.health_output
        elif v2_pilot:
            if protocol.get("protocol_status", {}).get("frozen") is True:
                raise ValueError("The v2 pilot protocol must remain unfrozen")
            (
                pilot_output_root,
                selected_pilot_plan,
                pilot_health_path,
                pilot_health_report,
            ) = _validate_v2_pilot_launch_contract(
                protocol=protocol,
                protocol_path=protocol_path,
                selected_rows=rows,
                output_root=output_root,
                pilot_plan=pilot_plan,
            )
            _validate_pilot_output_root(pilot_output_root, protocol["output_root"])
        else:
            if protocol.get("protocol_status", {}).get("frozen") is True:
                raise ValueError("The protocol is frozen; run formal training without the pilot flag")
            if limit != 4:
                raise ValueError("Unfrozen pilots require an explicit --limit 4")
            pilot_output_root = _validate_pilot_output_root(
                output_root, protocol["output_root"]
            )
    report = check_protocol(
        protocol_path,
        mode="smoke" if dry_run else ("pilot" if pilot_active else "full"),
        project_root=PROJECT_ROOT,
        check_dependencies=True,
    )
    if not report.ok:
        details = "\n".join(f"- {item}" for item in report.errors)
        raise RuntimeError(f"Preflight blocked matrix execution:\n{details}")
    if not dry_run and not pilot_active:
        assert config_dir is not None
        if protocol_version in (3, 4, 5, 6, 7):
            artifacts = artifact_paths_for_protocol(protocol)
            expected_config_dir = _project_path(artifacts["formal_matrix"])
            expected_output_root = _project_path(artifacts["formal_results"])
            if config_dir != expected_config_dir:
                raise ValueError(
                    f"Protocol v{protocol_version} formal execution must use "
                    "artifact_paths.formal_matrix"
                )
            if output_root is not None and _project_path(output_root) != expected_output_root:
                raise ValueError(
                    f"Protocol v{protocol_version} formal execution must use "
                    "artifact_paths.formal_results"
                )
            if config_path is not None or any(
                value is not None for value in (limit, experiment, condition, dataset)
            ):
                raise ValueError(
                    f"Protocol v{protocol_version} formal execution requires the complete "
                    f"frozen {expected_run_count_for_protocol(protocol)}-run matrix; "
                    "partial selection is forbidden"
                )
        try:
            verify_formal_freeze(
                project_root=PROJECT_ROOT,
                protocol_path=protocol_path,
                matrix_dir=config_dir,
            )
        except FreezeGateError as exc:
            raise RuntimeError(f"Formal freeze-manifest gate blocked execution: {exc}") from exc
    if pilot_active:
        if v7_pilot:
            conditions_label = "/".join(v7_block.conditions) if v7_block else "six-condition"
        elif v6_pilot:
            conditions_label = "/".join(v6_block.conditions) if v6_block else "six-condition"
        else:
            conditions_label = "C1/C2" if v3_pilot or v4_pilot or v5_pilot else "C1-C4"
        print(
            f"WARNING: running the fixed non-reportable {conditions_label} pilot plan; "
            "do not report these results"
        )

    matrix_manifest = config_dir / "matrix_manifest.json"
    if config_path is None:
        if not matrix_manifest.exists():
            raise FileNotFoundError(f"Missing {matrix_manifest}; regenerate configurations")
        metadata = json.loads(matrix_manifest.read_text(encoding="utf-8"))
        if metadata.get("protocol_hash") != report.protocol_hash:
            raise RuntimeError("Generated configs do not match the current protocol; regenerate them")
        if v2_pilot:
            _validate_v2_generated_matrix(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                manifest_rows=all_rows,
                matrix_manifest=metadata,
            )
            assert pilot_health_report is not None
            reference_row = next(
                row for row in all_rows if row.get("condition") == "C1"
            )
            pilot_reference_config = load_run_config(
                config_dir / reference_row["config_file"], protocol_path
            )
            try:
                pilot_launch_context = validate_current_runtime_against_health_report(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    device=str(device or ""),
                )
            except HealthGateError as exc:
                raise RuntimeError(
                    f"V2 pilot current-runtime/data gate blocked execution: {exc}"
                ) from exc
            print("V2_PILOT_CURRENT_CONTEXT_PASS")
        elif protocol_version in (3, 4, 5, 6, 7) and not dry_run and not pilot_active:
            _validate_v2_generated_matrix(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                manifest_rows=all_rows,
                matrix_manifest=metadata,
                label=f"V{protocol_version} formal",
            )
        elif v7_pilot:
            assert v7_block is not None
            assert pilot_health_report is not None
            _validate_v2_generated_matrix(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                manifest_rows=all_rows,
                matrix_manifest=metadata,
                expected_raw_runs=generate_v7_pilot_matrix(protocol),
                label="V7 pilot",
            )
            pilot_reference_config = expected_v7_pilot_configs(protocol)[0]
            try:
                pilot_launch_context = validate_v7_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v7_block,
                    device=str(device or ""),
                )
            except PilotV7Error as exc:
                raise RuntimeError(
                    f"V7 pilot current-runtime/data gate blocked execution: {exc}"
                ) from exc
            print(f"V7_PILOT_CURRENT_CONTEXT_PASS dataset={v7_block.dataset}")
        elif v6_pilot:
            assert v6_block is not None
            assert pilot_health_report is not None
            _validate_v2_generated_matrix(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                manifest_rows=all_rows,
                matrix_manifest=metadata,
                expected_raw_runs=generate_v6_pilot_matrix(protocol),
                label="V6 pilot",
            )
            pilot_reference_config = expected_v6_pilot_configs(protocol)[0]
            try:
                pilot_launch_context = validate_v6_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v6_block,
                    device=str(device or ""),
                )
            except PilotV6Error as exc:
                raise RuntimeError(
                    f"V6 pilot current-runtime/data gate blocked execution: {exc}"
                ) from exc
            print(f"V6_PILOT_CURRENT_CONTEXT_PASS dataset={v6_block.dataset}")
        elif v5_pilot:
            assert v5_block is not None
            assert pilot_health_report is not None
            _validate_v2_generated_matrix(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                manifest_rows=all_rows,
                matrix_manifest=metadata,
                expected_raw_runs=generate_v5_pilot_matrix(protocol),
                label="V5 pilot",
            )
            pilot_reference_config = expected_v5_pilot_configs(
                protocol, v5_block.dataset
            )[0]
            try:
                pilot_launch_context = validate_v5_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v5_block,
                    device=str(device or ""),
                )
            except PilotV5Error as exc:
                raise RuntimeError(
                    f"V5 pilot current-runtime/data gate blocked execution: {exc}"
                ) from exc
            print(f"V5_PILOT_CURRENT_CONTEXT_PASS dataset={v5_block.dataset}")
        elif v4_pilot:
            assert v4_block is not None
            assert pilot_health_report is not None
            _validate_v2_generated_matrix(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                manifest_rows=all_rows,
                matrix_manifest=metadata,
                expected_raw_runs=generate_v4_pilot_matrix(protocol),
                label="V4 pilot",
            )
            pilot_reference_config = expected_v4_pilot_configs(
                protocol, v4_block.dataset
            )[0]
            try:
                pilot_launch_context = validate_v4_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v4_block,
                    device=str(device or ""),
                )
            except PilotV4Error as exc:
                raise RuntimeError(
                    f"V4 pilot current-runtime/data gate blocked execution: {exc}"
                ) from exc
            print(f"V4_PILOT_CURRENT_CONTEXT_PASS dataset={v4_block.dataset}")
        elif v3_pilot:
            assert v3_block is not None
            assert pilot_health_report is not None
            _validate_v2_generated_matrix(
                protocol=protocol,
                protocol_path=protocol_path,
                config_dir=config_dir,
                manifest_rows=all_rows,
                matrix_manifest=metadata,
                expected_raw_runs=generate_v3_pilot_matrix(protocol),
                label="V3 pilot",
            )
            pilot_reference_config = expected_pilot_configs(
                protocol, v3_block.dataset
            )[0]
            try:
                pilot_launch_context = validate_v3_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v3_block,
                    device=str(device or ""),
                )
            except PilotV3Error as exc:
                raise RuntimeError(
                    f"V3 pilot current-runtime/data gate blocked execution: {exc}"
                ) from exc
            print(f"V3_PILOT_CURRENT_CONTEXT_PASS dataset={v3_block.dataset}")

    if dry_run:
        training_output_root = _project_path(output_root or PROJECT_ROOT / "results")
        result_root = training_output_root / "smoke"
    elif pilot_active:
        assert pilot_output_root is not None
        training_output_root = pilot_output_root
        result_root = training_output_root
    else:
        if protocol_version in (3, 4, 5, 6, 7):
            training_output_root = _project_path(
                artifact_paths_for_protocol(protocol)["formal_results"]
            )
        else:
            training_output_root = _project_path(
                output_root or PROJECT_ROOT / "results" / "runs"
            )
        result_root = training_output_root
    if v7_pilot:
        assert v7_block is not None
        assert pilot_health_report is not None
        assert config_dir is not None
        pilot_plan_path = _prepare_v7_pilot_plan(
            block=v7_block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=all_rows,
            health_report=pilot_health_report,
        )
    elif v6_pilot:
        assert v6_block is not None
        assert pilot_health_report is not None
        assert config_dir is not None
        pilot_plan_path = _prepare_v6_pilot_plan(
            block=v6_block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=all_rows,
            health_report=pilot_health_report,
        )
    elif v5_pilot:
        assert v5_block is not None
        assert pilot_health_report is not None
        assert config_dir is not None
        pilot_plan_path = _prepare_v5_pilot_plan(
            block=v5_block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=all_rows,
            health_report=pilot_health_report,
        )
    elif v4_pilot:
        assert v4_block is not None
        assert pilot_health_report is not None
        assert config_dir is not None
        pilot_plan_path = _prepare_v4_pilot_plan(
            block=v4_block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=all_rows,
            health_report=pilot_health_report,
        )
    elif v3_pilot:
        assert v3_block is not None
        assert pilot_health_report is not None
        assert config_dir is not None
        pilot_plan_path = _prepare_v3_pilot_plan(
            block=v3_block,
            protocol_path=protocol_path,
            config_dir=config_dir,
            manifest_rows=all_rows,
            health_report=pilot_health_report,
        )
    else:
        pilot_plan_path = (
            _prepare_unfrozen_pilot_plan(
                all_rows=all_rows,
                selected_rows=rows,
                config_dir=config_dir,
                protocol_path=protocol_path,
                protocol_hash=report.protocol_hash,
                output_root=pilot_output_root,
                formal_root=protocol["output_root"],
                plan_path=selected_pilot_plan,
                health_report_path=pilot_health_path,
                health_report=pilot_health_report,
            )
            if pilot_active and pilot_output_root is not None
            else None
        )
    event_path = result_root / "matrix_events.jsonl"
    if pilot_launch_context is not None:
        _event(
            event_path,
            {
                "event": "pilot_context_validated",
                "context": pilot_launch_context,
                "timestamp": time.time(),
            },
        )
    failures = 0
    attempted = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        run_id = row["run_id"]
        selected_config = (config_dir / row["config_file"]).resolve()
        expected_file_hash = row.get("config_file_sha256")
        if expected_file_hash and sha256_file(selected_config) != expected_file_hash:
            raise RuntimeError(f"Generated config file hash mismatch: {selected_config}")
        planned_config = load_run_config(selected_config, protocol_path)
        _validate_run_id(run_id, planned_config.runtime.run_id)
        planned_hash = planned_config.config_hash
        if row.get("config_hash") and row["config_hash"] != planned_hash:
            raise RuntimeError(f"Generated config scientific hash mismatch: {selected_config}")
        log_dir = result_root / run_id
        try:
            action, resume_checkpoint = _continuation_action(
                log_dir,
                run_id,
                planned_hash,
                resume_matrix=resume_matrix,
                forbid_fresh_after_attempt=protocol_version in (3, 4, 5, 6, 7)
                and not dry_run,
            )
        except Exception as exc:
            failures += 1
            _event(event_path, {
                "event": "run_blocked", "index": index, "total": len(rows),
                "run_id": run_id, "message": str(exc), "timestamp": time.time(),
            })
            print(f"[{index}/{len(rows)}] {run_id}: BLOCKED: {exc}", file=sys.stderr)
            if stop_on_error:
                break
            continue
        if action == "skip":
            skipped += 1
            _event(event_path, {
                "event": "run_skipped_complete", "index": index, "total": len(rows),
                "run_id": run_id, "timestamp": time.time(),
            })
            print(f"[{index}/{len(rows)}] {run_id}: skipped (complete)")
            continue

        if v2_pilot:
            assert pilot_health_report is not None
            assert pilot_reference_config is not None
            try:
                run_context = validate_current_runtime_against_health_report(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    device=str(device or ""),
                )
            except HealthGateError as exc:
                failures += 1
                _event(
                    event_path,
                    {
                        "event": "run_blocked",
                        "index": index,
                        "total": len(rows),
                        "run_id": run_id,
                        "message": f"current runtime/data mismatch: {exc}",
                        "timestamp": time.time(),
                    },
                )
                print(
                    f"[{index}/{len(rows)}] {run_id}: BLOCKED: {exc}",
                    file=sys.stderr,
                )
                if stop_on_error:
                    break
                continue
            _event(
                event_path,
                {
                    "event": "run_context_validated",
                    "index": index,
                    "total": len(rows),
                    "run_id": run_id,
                    "context": run_context,
                    "timestamp": time.time(),
                },
            )
        elif v7_pilot:
            assert pilot_health_report is not None
            assert pilot_reference_config is not None
            assert v7_block is not None
            try:
                run_context = validate_v7_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v7_block,
                    device=str(device or ""),
                )
            except PilotV7Error as exc:
                failures += 1
                _event(
                    event_path,
                    {
                        "event": "run_blocked",
                        "index": index,
                        "total": len(rows),
                        "run_id": run_id,
                        "message": f"current runtime/data mismatch: {exc}",
                        "timestamp": time.time(),
                    },
                )
                print(f"[{index}/{len(rows)}] {run_id}: BLOCKED: {exc}", file=sys.stderr)
                if stop_on_error:
                    break
                continue
            _event(
                event_path,
                {
                    "event": "run_context_validated",
                    "index": index,
                    "total": len(rows),
                    "run_id": run_id,
                    "context": run_context,
                    "timestamp": time.time(),
                },
            )
        elif v6_pilot:
            assert pilot_health_report is not None
            assert pilot_reference_config is not None
            assert v6_block is not None
            try:
                run_context = validate_v6_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v6_block,
                    device=str(device or ""),
                )
            except PilotV6Error as exc:
                failures += 1
                _event(
                    event_path,
                    {
                        "event": "run_blocked",
                        "index": index,
                        "total": len(rows),
                        "run_id": run_id,
                        "message": f"current runtime/data mismatch: {exc}",
                        "timestamp": time.time(),
                    },
                )
                print(f"[{index}/{len(rows)}] {run_id}: BLOCKED: {exc}", file=sys.stderr)
                if stop_on_error:
                    break
                continue
            _event(
                event_path,
                {
                    "event": "run_context_validated",
                    "index": index,
                    "total": len(rows),
                    "run_id": run_id,
                    "context": run_context,
                    "timestamp": time.time(),
                },
            )
        elif v5_pilot:
            assert pilot_health_report is not None
            assert pilot_reference_config is not None
            assert v5_block is not None
            try:
                run_context = validate_v5_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v5_block,
                    device=str(device or ""),
                )
            except PilotV5Error as exc:
                failures += 1
                _event(
                    event_path,
                    {
                        "event": "run_blocked",
                        "index": index,
                        "total": len(rows),
                        "run_id": run_id,
                        "message": f"current runtime/data mismatch: {exc}",
                        "timestamp": time.time(),
                    },
                )
                print(f"[{index}/{len(rows)}] {run_id}: BLOCKED: {exc}", file=sys.stderr)
                if stop_on_error:
                    break
                continue
            _event(
                event_path,
                {
                    "event": "run_context_validated",
                    "index": index,
                    "total": len(rows),
                    "run_id": run_id,
                    "context": run_context,
                    "timestamp": time.time(),
                },
            )
        elif v4_pilot:
            assert pilot_health_report is not None
            assert pilot_reference_config is not None
            assert v4_block is not None
            try:
                run_context = validate_v4_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v4_block,
                    device=str(device or ""),
                )
            except PilotV4Error as exc:
                failures += 1
                _event(
                    event_path,
                    {
                        "event": "run_blocked",
                        "index": index,
                        "total": len(rows),
                        "run_id": run_id,
                        "message": f"current runtime/data mismatch: {exc}",
                        "timestamp": time.time(),
                    },
                )
                print(f"[{index}/{len(rows)}] {run_id}: BLOCKED: {exc}", file=sys.stderr)
                if stop_on_error:
                    break
                continue
            _event(
                event_path,
                {
                    "event": "run_context_validated",
                    "index": index,
                    "total": len(rows),
                    "run_id": run_id,
                    "context": run_context,
                    "timestamp": time.time(),
                },
            )
        elif v3_pilot:
            assert pilot_health_report is not None
            assert pilot_reference_config is not None
            assert v3_block is not None
            try:
                run_context = validate_v3_current_runtime(
                    pilot_health_report,
                    protocol=protocol,
                    reference_config=pilot_reference_config,
                    block=v3_block,
                    device=str(device or ""),
                )
            except PilotV3Error as exc:
                failures += 1
                _event(
                    event_path,
                    {
                        "event": "run_blocked",
                        "index": index,
                        "total": len(rows),
                        "run_id": run_id,
                        "message": f"current runtime/data mismatch: {exc}",
                        "timestamp": time.time(),
                    },
                )
                print(f"[{index}/{len(rows)}] {run_id}: BLOCKED: {exc}", file=sys.stderr)
                if stop_on_error:
                    break
                continue
            _event(
                event_path,
                {
                    "event": "run_context_validated",
                    "index": index,
                    "total": len(rows),
                    "run_id": run_id,
                    "context": run_context,
                    "timestamp": time.time(),
                },
            )

        command = [sys.executable, "-m", "talif_msresnet.train", "--config", str(selected_config)]
        command.extend(["--protocol", str(protocol_path)])
        if dry_run:
            command.append("--dry-run")
        if device is not None:
            command.extend(["--device", device])
        if pilot_plan_path is not None:
            command.extend(["--orchestrator-pilot-plan", str(pilot_plan_path)])
        if limit_batches is not None:
            command.extend(["--limit-batches", str(limit_batches)])
        command.extend(["--output-dir", str(training_output_root)])
        if resume_checkpoint is not None:
            command.extend(["--resume", str(resume_checkpoint)])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_ROOT), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        if planned_config.runtime.deterministic:
            environment.setdefault(
                "CUBLAS_WORKSPACE_CONFIG", DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
            )
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt, stdout_path, stderr_path = _next_attempt(log_dir)
        started = time.time()
        _event(event_path, {
            "event": "run_started", "index": index, "total": len(rows), "run_id": run_id,
            "attempt": attempt, "action": action, "command": command, "timestamp": time.time(),
        })
        with stdout_path.open("x", encoding="utf-8", newline="\n") as stdout, stderr_path.open("x", encoding="utf-8", newline="\n") as stderr:
            completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, stdout=stdout, stderr=stderr, check=False)
        elapsed = time.time() - started
        attempted += 1
        event = {
            "event": "run_finished" if completed.returncode == 0 else "run_failed",
            "index": index,
            "total": len(rows),
            "run_id": run_id,
            "attempt": attempt,
            "action": action,
            "returncode": completed.returncode,
            "duration_s": elapsed,
            "timestamp": time.time(),
        }
        _event(event_path, event)
        print(
            f"[{index}/{len(rows)}] {run_id}: attempt={attempt} action={action} "
            f"returncode={completed.returncode} ({elapsed:.1f}s)"
        )
        if completed.returncode != 0:
            failures += 1
            if stop_on_error:
                break
    print(
        f"Matrix complete: {attempted} attempted, {skipped} skipped complete, "
        f"{failures} failed/blocked"
    )
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir")
    parser.add_argument("--config", help="Run one generated configuration instead of a directory")
    parser.add_argument("--output-root", help="Override the result root in every selected config")
    parser.add_argument("--device", help="Device override; dry-run defaults to CPU in the trainer")
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument(
        "--allow-unfrozen-pilot",
        action="store_true",
        help=(
            "Run the complete protocol-bound non-reportable pilot block; frozen V4-V7 "
            "pilots still require their author-freeze and health PASS"
        ),
    )
    parser.add_argument(
        "--pilot-plan",
        help=(
            "Immutable pilot-plan path; use a new path for a new protocol generation "
            "and preserve earlier plans"
        ),
    )
    parser.add_argument(
        "--resume-matrix",
        action="store_true",
        help="Skip completed runs and continue interrupted runs only from last.pt; preserve attempt logs",
    )
    parser.add_argument("--limit", type=int, help="Run only the first N selected configurations")
    parser.add_argument("--dry-run", action="store_true", help="One epoch and a limited number of batches")
    parser.add_argument("--limit-batches", type=int, help="Pass a batch limit to each trainer")
    parser.add_argument("--experiment", choices=("E1", "E6"))
    parser.add_argument(
        "--condition",
        choices=("C1", "C2", "C3", "C4", "M0", "M1", "M2", "M3", "M4", "PLIF"),
    )
    parser.add_argument("--dataset")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit_batches is not None and args.limit_batches < 1:
        raise SystemExit("--limit-batches must be >= 1")
    if args.limit_batches is not None and not args.dry_run:
        raise SystemExit("--limit-batches is only permitted together with --dry-run")
    if args.resume_matrix and args.dry_run:
        raise SystemExit("--resume-matrix is only supported for full/pilot training, not smoke runs")
    return run_matrix(
        None if args.config else args.config_dir, config_path=args.config, output_root=args.output_root,
        device=args.device, protocol_path=args.protocol,
        allow_unfrozen_pilot=args.allow_unfrozen_pilot,
        pilot_plan=args.pilot_plan,
        resume_matrix=args.resume_matrix,
        limit=args.limit, dry_run=args.dry_run,
        limit_batches=args.limit_batches, experiment=args.experiment,
        condition=args.condition, dataset=args.dataset, stop_on_error=args.stop_on_error,
    )


if __name__ == "__main__":
    raise SystemExit(main())
