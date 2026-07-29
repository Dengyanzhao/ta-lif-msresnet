#!/usr/bin/env python3
"""Run one threshold-free, one-shot C1/C2 v5 implementation health gate."""

from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)
SCRIPTS_PATH = str(SCRIPTS_ROOT)
if SCRIPTS_PATH not in sys.path:
    sys.path.insert(1, SCRIPTS_PATH)

import calibrate_v5_health as calibration
import pilot_health_gate as legacy_gate
import pilot_health_gate_v3 as v3_gate

from talif_msresnet.config import (
    RunConfig,
    artifact_paths_for_protocol,
    load_protocol,
    load_run_config,
)
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.pilot_v5 import (
    HEALTH_ARTIFACT_CLASS,
    HEALTH_DECISION_BASIS,
    HEALTH_SCHEMA_VERSION,
    HEALTH_SEED_DISPOSITION,
    LEARNING_METRICS_ROLE,
    PILOT_SEED_DISPOSITION,
    REPORTING_ELIGIBILITY,
    SCHEDULE_HEALTH_ROLE,
    TIMING_HEALTH_ROLE,
    PilotBlock,
    PilotV5Error,
    attempt_receipt_payload,
    exclusive_create_json,
    expected_pilot_configs,
    implementation_health_anomalies,
    json_safe,
    repository_git_identity,
    require_v5_author_freeze,
    resolve_pilot_block,
    validate_development_probe_evidence,
)
from talif_msresnet.utils import environment_manifest, sha256_file, utc_now

RUNTIME_SOURCE_PATHS = (
    Path(__file__),
    REPOSITORY_ROOT / "scripts" / "calibrate_v5_health.py",
    REPOSITORY_ROOT / "scripts" / "pilot_health_gate.py",
    REPOSITORY_ROOT / "scripts" / "pilot_health_gate_v3.py",
    REPOSITORY_ROOT / "src" / "talif_msresnet" / "pilot_v3.py",
    REPOSITORY_ROOT / "src" / "talif_msresnet" / "pilot_v4.py",
    REPOSITORY_ROOT / "src" / "talif_msresnet" / "pilot_v5.py",
    REPOSITORY_ROOT / "src" / "talif_msresnet" / "config.py",
    REPOSITORY_ROOT / "src" / "talif_msresnet" / "data.py",
    REPOSITORY_ROOT / "src" / "talif_msresnet" / "train.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "protocol_v5_talif_only.yaml",
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--dataset", required=True, choices=("cifar100", "cifar10dvs"))
    parser.add_argument("--device", default="cuda:0")
    return parser


def _health_verdict(status: str, dataset: str) -> str:
    return f"V5_HEALTH_GATE_{status} dataset={dataset}"


def _repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _runtime_source_hashes() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in RUNTIME_SOURCE_PATHS:
        resolved = path.resolve()
        if not resolved.is_file():
            raise PilotV5Error(f"Required v5 runtime source is missing: {resolved}")
        values[artifact_path_reference(resolved, REPOSITORY_ROOT)] = sha256_file(resolved)
    return values


def _require_runtime_sources_unchanged(
    expected: Mapping[str, str], expected_git: Mapping[str, Any]
) -> dict[str, Any]:
    current = repository_git_identity(REPOSITORY_ROOT)
    if (
        current.get("git_commit") != expected_git.get("git_commit")
        or current.get("tracked_clean") is not True
    ):
        raise PilotV5Error("Git identity changed during the one-shot health execution")
    observed = _runtime_source_hashes()
    if observed != dict(expected):
        raise PilotV5Error("A bound runtime source changed during health execution")
    return current


def _reference_config(
    protocol: Mapping[str, Any], protocol_path: Path, config_dir: Path, dataset: str
) -> tuple[Path, RunConfig]:
    expected = expected_pilot_configs(protocol, dataset)[0]
    path = (config_dir / f"{expected.runtime.run_id}.yaml").resolve()
    if path.parent != config_dir.resolve() or not path.is_file():
        raise PilotV5Error(f"Canonical C1 pilot config is missing: {path}")
    observed = load_run_config(path, protocol_path)
    if observed.as_dict() != expected.as_dict():
        raise PilotV5Error("C1 pilot YAML differs from the protocol-generated config")
    return path, observed


def run_gate(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_path: Path,
    config: RunConfig,
    block: PilotBlock,
    device_name: str,
    git_identity: Mapping[str, Any],
    runtime_sources: Mapping[str, str],
) -> dict[str, Any]:
    """Execute only deterministic mechanism and numerical-integrity checks."""

    started_at = utc_now()
    started_clock = time.perf_counter()
    acceptance = protocol["pilot_acceptance"]
    implementation = acceptance["implementation_health"]
    fixed_contract = implementation["fixed_batch"]
    formal_contract = implementation["formal_schedule"]
    checkpoint_contract = implementation["checkpoint_resume"]
    environment_binding = acceptance["environment"]
    if not config.runtime.deterministic or config.runtime.amp:
        raise PilotV5Error("v5 health requires deterministic float32 execution with amp=false")
    if config.runtime.seed != block.pilot_seed:
        raise PilotV5Error("The reference config must retain the disjoint pilot seed")

    # The exclusive attempt receipt has already been created by main before this call.
    legacy_gate.seed_everything(block.health_seed, deterministic=True)
    device = torch.device(device_name)
    hardware = legacy_gate.require_target_cuda(
        device, str(environment_binding["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_binding, hardware)
    idle = legacy_gate.gpu_idle_precheck(device)

    batch_size = int(fixed_contract["batch_size"])
    resolved, inputs, targets, split_manifest, source_identity = v3_gate.load_fixed_real_batch(
        config,
        seed=block.health_seed,
        required_batch_size=batch_size,
    )
    batch_inputs = inputs[:batch_size]
    batch_targets = targets[:batch_size]
    if int(batch_inputs.shape[0]) != batch_size or int(batch_targets.shape[0]) != batch_size:
        raise PilotV5Error("The real-data loader did not provide the exact fixed batch")
    fixed_batch_sha256 = legacy_gate.tensor_batch_sha256(batch_inputs, batch_targets)
    fixed_batch = (
        batch_inputs.to(device, non_blocking=False),
        batch_targets.to(device, non_blocking=False),
    )
    fixed_settings = {
        **dict(fixed_contract),
        "learning_metrics_role": LEARNING_METRICS_ROLE,
    }
    formal_settings = {
        **dict(formal_contract),
        "learning_metrics_role": LEARNING_METRICS_ROLE,
        "use_frozen_v4_optimizer_schedule": True,
    }

    fixed_reports: dict[str, Any] = {}
    formal_reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix=f"v5-health-{block.dataset}-", dir=str(block.health_output.parent)
    ) as temporary:
        temporary_root = Path(temporary)
        for condition in block.conditions:
            fixed_reports[condition] = calibration.run_fixed_batch_condition(
                resolved,
                condition=condition,
                seed=block.health_seed,
                device=device,
                output_parent=block.health_output.parent,
                batch=fixed_batch,
                settings=fixed_settings,
                checkpoint_path=temporary_root / condition / "last.pt",
            )
            gc.collect()
            torch.cuda.empty_cache()
            formal_reports[condition] = calibration.run_formal_schedule_condition(
                config,
                condition=condition,
                seed=block.health_seed,
                device=device,
                output_parent=block.health_output.parent,
                settings=formal_settings,
                expected_split_manifest_sha256=str(split_manifest["manifest_sha256"]),
            )
            gc.collect()
            torch.cuda.empty_cache()

    anomalies = implementation_health_anomalies(
        fixed_reports,
        formal_reports,
        block=block,
        protocol=protocol,
        reference_config=config,
    )
    _require_runtime_sources_unchanged(runtime_sources, git_identity)
    status = "PASS" if not anomalies else "FAIL"
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v5-pilot threshold-free implementation health validation",
        "status": status,
        "pass": status == "PASS",
        "thresholds_evaluated": [],
        "decision_basis": HEALTH_DECISION_BASIS,
        "learning_metrics_role": LEARNING_METRICS_ROLE,
        "dataset": block.dataset,
        "seed": block.health_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "development_probe_hash": block.development_probe_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity["git_commit"],
        "tracked_clean": git_identity["tracked_clean"],
        "attempt_receipt": artifact_path_reference(block.attempt_receipt, REPOSITORY_ROOT),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
        "evidence": {
            "development_probe": dict(block.development_probe_binding),
        },
        "source": {
            "protocol": artifact_path_reference(protocol_path, REPOSITORY_ROOT),
            "protocol_file_sha256": sha256_file(protocol_path),
            "reference_config": artifact_path_reference(config_path, REPOSITORY_ROOT),
            "reference_config_file_sha256": sha256_file(config_path),
            "reference_config_hash": config.config_hash,
            "reference_config_seed": config.runtime.seed,
            "dataset": block.dataset,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "fixed_batch_sha256": fixed_batch_sha256,
            "fixed_batch_shape": list(batch_inputs.shape),
            "fixed_batch_size": batch_size,
            "runtime_sources_sha256": dict(runtime_sources),
            **source_identity,
        },
        "environment": {
            **environment_manifest(),
            "contract": dict(environment_binding),
            "hardware": hardware,
            "python_implementation": platform.python_implementation(),
            "precision": "float32",
            "amp": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu_idle_precheck": idle,
        },
        "implementation_health": {
            "decision_basis": HEALTH_DECISION_BASIS,
            "thresholds_evaluated": [],
            "learning_metrics_role": LEARNING_METRICS_ROLE,
            "fixed_batch": {
                "contract": dict(fixed_contract),
                "by_condition": fixed_reports,
            },
            "formal_schedule": {
                "contract": dict(formal_contract),
                "by_condition": formal_reports,
            },
            "checkpoint_resume": {
                "contract": dict(checkpoint_contract),
                "by_condition": {
                    condition: fixed_reports[condition]["checkpoint_boundary"]
                    for condition in block.conditions
                },
            },
            "integrity_anomalies": anomalies,
        },
        "timing": {
            "contract": dict(acceptance["timing"]),
            "health_role": TIMING_HEALTH_ROLE,
            "evaluated": False,
        },
        "schedule": {
            "contract": dict(acceptance["schedule"]),
            "health_role": SCHEDULE_HEALTH_ROLE,
            "evaluated": False,
        },
        "integrity_anomalies": anomalies,
        "failures": anomalies,
    }


def _fatal_report(
    *,
    block: PilotBlock,
    git_identity: Mapping[str, Any],
    runtime_sources: Mapping[str, str],
    exc: BaseException,
) -> dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v5-pilot threshold-free implementation health validation",
        "status": "ERROR",
        "pass": False,
        "thresholds_evaluated": [],
        "decision_basis": HEALTH_DECISION_BASIS,
        "learning_metrics_role": LEARNING_METRICS_ROLE,
        "dataset": block.dataset,
        "seed": block.health_seed,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "development_probe_hash": block.development_probe_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity.get("git_commit"),
        "tracked_clean": git_identity.get("tracked_clean"),
        "attempt_receipt": artifact_path_reference(block.attempt_receipt, REPOSITORY_ROOT),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "finished_at": utc_now(),
        "fatal_error": message,
        "traceback": traceback.format_exc(),
        "evidence": {"development_probe": dict(block.development_probe_binding)},
        "source": {"runtime_sources_sha256": dict(runtime_sources)},
        "integrity_anomalies": ["fatal execution error"],
        "failures": [message],
        "seed_disposition": HEALTH_SEED_DISPOSITION,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
    }


def validate_current_runtime_against_health_report(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
    block: PilotBlock,
    device: str,
) -> dict[str, Any]:
    """Rebuild the health-seed batch while authorizing only the pilot seed."""

    if report.get("status") != "PASS" or report.get("pass") is not True:
        raise PilotV5Error("Only a canonical v5 PASS can authorize pilot launch")
    if reference_config.data.dataset != block.dataset:
        raise PilotV5Error("Pilot reference config dataset differs from the health block")
    if reference_config.runtime.seed != block.pilot_seed:
        raise PilotV5Error("Pilot launch config must use the disjoint pilot seed")
    if not device or device == "auto":
        raise PilotV5Error("The v5 pilot requires an explicit CUDA device")
    identity = repository_git_identity(REPOSITORY_ROOT)
    if identity.get("tracked_clean") is not True or identity.get("git_commit") != report.get(
        "git_commit"
    ):
        raise PilotV5Error("Current Git identity differs from the v5 health report")

    acceptance = protocol["pilot_acceptance"]
    environment_binding = acceptance["environment"]
    fixed_contract = acceptance["implementation_health"]["fixed_batch"]
    legacy_gate.seed_everything(block.health_seed, deterministic=True)
    selected_device = torch.device(device)
    hardware = legacy_gate.require_target_cuda(
        selected_device, str(environment_binding["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_binding, hardware)
    report_environment = report.get("environment")
    if not isinstance(report_environment, Mapping) or hardware != report_environment.get(
        "hardware"
    ):
        raise PilotV5Error("Current CUDA hardware differs from the health report")

    current_environment = {
        **environment_manifest(),
        "precision": "float32",
        "amp": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    stable_keys = (
        "hostname",
        "platform",
        "python",
        "pytorch",
        "numpy",
        "cuda_available",
        "cuda_version",
        "cuda_device",
        "precision",
        "amp",
        "cublas_workspace_config",
        "torch_deterministic",
        "torch_deterministic_warn_only",
        "cudnn_benchmark",
        "cudnn_deterministic",
    )
    mismatches = [
        key
        for key in stable_keys
        if current_environment.get(key) != report_environment.get(key)
    ]
    if mismatches:
        raise PilotV5Error(
            "Current environment differs from the health report at: " + ", ".join(mismatches)
        )
    idle = legacy_gate.gpu_idle_precheck(selected_device)
    report_idle = report_environment.get("gpu_idle_precheck")
    if not isinstance(report_idle, Mapping) or idle.get("device_uuid") != report_idle.get(
        "device_uuid"
    ):
        raise PilotV5Error("Current CUDA device UUID differs from the health report")

    required = int(fixed_contract["batch_size"])
    _resolved, inputs, targets, manifest, source_identity = v3_gate.load_fixed_real_batch(
        reference_config,
        seed=block.health_seed,
        required_batch_size=required,
    )
    batch_inputs = inputs[:required]
    batch_targets = targets[:required]
    source = report.get("source")
    if not isinstance(source, Mapping):
        raise PilotV5Error("Health report source evidence is missing")
    observed = {
        "split_manifest_sha256": manifest["manifest_sha256"],
        "fixed_batch_sha256": legacy_gate.tensor_batch_sha256(batch_inputs, batch_targets),
        "fixed_batch_shape": list(batch_inputs.shape),
        "fixed_batch_size": required,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        **source_identity,
    }
    mismatches = [key for key, value in observed.items() if source.get(key) != value]
    expected_runtime_sources = source.get("runtime_sources_sha256")
    if not isinstance(expected_runtime_sources, Mapping) or _runtime_source_hashes() != dict(
        expected_runtime_sources
    ):
        mismatches.append("runtime_sources_sha256")
    if mismatches:
        raise PilotV5Error(
            "Current dataset/runtime identity differs from the health report at: "
            + ", ".join(dict.fromkeys(mismatches))
        )
    return {
        "pass": True,
        "protocol_version": 5,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "block_hash": block.block_hash,
        "device": hardware["device"],
        "device_uuid": idle["device_uuid"],
        "hardware": hardware,
        "stable_environment": {key: current_environment.get(key) for key in stable_keys},
        **observed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol_path = args.protocol.resolve()
        protocol = load_protocol(protocol_path)
        block = resolve_pilot_block(protocol, args.dataset, repository_root=REPOSITORY_ROOT)
        author_freeze = require_v5_author_freeze(protocol, repository_root=REPOSITORY_ROOT)
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else _repository_path(artifact_paths_for_protocol(protocol)["pilot_matrix"])
        )
        config_path, config = _reference_config(protocol, protocol_path, config_dir, block.dataset)
        if block.health_output.exists():
            raise PilotV5Error("Canonical v5 health output exists; no retry is allowed")
        if block.attempt_receipt.exists():
            raise PilotV5Error("This dataset health seed is consumed; no retry is allowed")
        validate_development_probe_evidence(protocol, repository_root=REPOSITORY_ROOT)
        git_identity = repository_git_identity(REPOSITORY_ROOT)
        if git_identity["tracked_clean"] is not True:
            raise PilotV5Error("v5 health evidence requires a clean tracked worktree")
        if not config.runtime.deterministic or config.runtime.amp:
            raise PilotV5Error("reference config must use deterministic float32 execution")
        calibration.require_head_bound_sources(
            (*RUNTIME_SOURCE_PATHS, protocol_path, config_path)
        )
        runtime_sources = _runtime_source_hashes()
        receipt = attempt_receipt_payload(
            block=block,
            protocol_path=protocol_path,
            config_path=config_path,
            config=config,
            git_identity=git_identity,
            repository_root=REPOSITORY_ROOT,
            runtime_sources=runtime_sources,
        )
        exclusive_create_json(block.attempt_receipt, receipt)
        print(f"V5_AUTHOR_FREEZE_PASS signoff_sha256={author_freeze['signoff_sha256']}")
        print(
            f"V5_HEALTH_ATTEMPT_CLAIMED dataset={block.dataset} "
            f"health_seed={block.health_seed} pilot_seed={block.pilot_seed}"
        )
        print("HEALTH_SEED_EXECUTION_BEGINS")
    except Exception as exc:  # noqa: BLE001 - normalize all preflight failures.
        print(f"V5_HEALTH_GATE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        report = run_gate(
            protocol=protocol,
            protocol_path=protocol_path,
            config_path=config_path,
            config=config,
            block=block,
            device_name=args.device,
            git_identity=git_identity,
            runtime_sources=runtime_sources,
        )
    except BaseException as exc:  # noqa: BLE001 - preserve consumed-seed evidence.
        report = _fatal_report(
            block=block,
            git_identity=git_identity,
            runtime_sources=runtime_sources,
            exc=exc,
        )
    if block.health_output.exists():
        print("V5_HEALTH_GATE_BLOCKED: output appeared during execution", file=sys.stderr)
        return 2
    safe_report = json_safe(report)
    try:
        exclusive_create_json(block.health_output, safe_report)
    except FileExistsError:
        print("V5_HEALTH_GATE_BLOCKED: output appeared during execution", file=sys.stderr)
        return 2
    print(_health_verdict(str(report["status"]), block.dataset))
    print(f"NON_REPORTING_OUTPUT={block.health_output}")
    print(f"HEALTH_SEED_{block.health_seed}_CONSUMED_DO_NOT_RETRY")
    print(f"PILOT_SEED_{block.pilot_seed}_REMAINS_UNCONSUMED")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
