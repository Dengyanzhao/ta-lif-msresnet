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
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import (  # noqa: E402
    generate_run_matrix,
    load_protocol,
    load_run_config,
    validate_run_mapping,
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


def _validate_v2_generated_matrix(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_dir: Path,
    manifest_rows: Sequence[dict[str, str]],
    matrix_manifest: Mapping[str, Any],
) -> None:
    """Require the disk matrix to equal the protocol-generated quartet exactly."""

    expected = [
        validate_run_mapping(raw, protocol) for raw in generate_run_matrix(protocol)
    ]
    expected_ids = [config.runtime.run_id for config in expected]
    observed_ids = [str(row.get("run_id")) for row in manifest_rows]
    if observed_ids != expected_ids:
        raise RuntimeError("V2 generated run manifest differs from the protocol")
    json_rows = matrix_manifest.get("runs")
    if not isinstance(json_rows, list) or [
        str(row.get("run_id")) for row in json_rows if isinstance(row, Mapping)
    ] != expected_ids:
        raise RuntimeError("V2 matrix manifest differs from the protocol-generated order")

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
                or not (run_dir / "last.pt").exists()
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
    protocol = load_protocol(protocol_path)
    pilot_active = bool(allow_unfrozen_pilot)
    v2_pilot = bool(pilot_active and _is_v2_pilot(protocol))
    pilot_output_root: Path | None = None
    selected_pilot_plan = _validated_pilot_plan_path(pilot_plan or PILOT_PLAN_PATH)
    pilot_health_path: Path | None = None
    pilot_health_report: dict[str, Any] | None = None
    pilot_reference_config = None
    pilot_launch_context: dict[str, Any] | None = None
    if pilot_active:
        if config_path is not None:
            raise ValueError("Unfrozen pilots require --config-dir so the full C1-C4 plan is auditable")
        if dry_run:
            raise ValueError("--allow-unfrozen-pilot cannot be combined with --dry-run")
        if protocol.get("protocol_status", {}).get("frozen") is True:
            raise ValueError("The protocol is frozen; run formal training without the pilot flag")
        if v2_pilot:
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
        try:
            verify_formal_freeze(
                project_root=PROJECT_ROOT,
                protocol_path=protocol_path,
                matrix_dir=config_dir,
            )
        except FreezeGateError as exc:
            raise RuntimeError(f"Formal freeze-manifest gate blocked execution: {exc}") from exc
    if pilot_active:
        print("WARNING: running the fixed non-reportable C1-C4 pilot plan; do not report these results")

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

    if dry_run:
        training_output_root = _project_path(output_root or PROJECT_ROOT / "results")
        result_root = training_output_root / "smoke"
    elif pilot_active:
        assert pilot_output_root is not None
        training_output_root = pilot_output_root
        result_root = training_output_root
    else:
        training_output_root = _project_path(output_root or PROJECT_ROOT / "results" / "runs")
        result_root = training_output_root
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
                log_dir, run_id, planned_hash, resume_matrix=resume_matrix,
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
    parser.add_argument("--config-dir", default=str(PROJECT_ROOT / "configs" / "generated"))
    parser.add_argument("--config", help="Run one generated configuration instead of a directory")
    parser.add_argument("--output-root", help="Override the result root in every selected config")
    parser.add_argument("--device", help="Device override; dry-run defaults to CPU in the trainer")
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument(
        "--allow-unfrozen-pilot",
        action="store_true",
        help="Permit one complete non-reportable C1-C4 pilot block before protocol freeze",
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
    parser.add_argument("--experiment", choices=("E1",))
    parser.add_argument("--condition", choices=("C1", "C2", "C3", "C4"))
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
