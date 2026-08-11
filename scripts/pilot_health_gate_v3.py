#!/usr/bin/env python3
"""Run one non-reporting C1/C2 v3 health gate for a bound dataset block."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import math
import os
import platform
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pilot_health_gate as legacy_gate  # noqa: E402
from talif_msresnet.config import (  # noqa: E402
    RunConfig,
    artifact_paths_for_protocol,
    load_protocol,
    load_run_config,
)
from talif_msresnet.data import build_loaders  # noqa: E402
from talif_msresnet.pilot_v3 import (  # noqa: E402
    HEALTH_ARTIFACT_CLASS,
    HEALTH_SCHEMA_VERSION,
    REPORTING_ELIGIBILITY,
    PilotBlock,
    PilotV3Error,
    attempt_receipt_payload,
    exclusive_create_json,
    expected_pilot_configs,
    repository_git_identity,
    require_v3_author_freeze,
    resolve_pilot_block,
    validate_fixed_batch_shape,
)
from talif_msresnet.utils import (  # noqa: E402
    atomic_write_json,
    environment_manifest,
    sha256_file,
    utc_now,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "protocol_v3_talif_only.yaml",
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--dataset", required=True, choices=("cifar100", "cifar10dvs"))
    parser.add_argument("--device", default="cuda:0")
    return parser


def _repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _resolved_data_config(config: RunConfig) -> RunConfig:
    data = config.data

    def resolved(value: str | None) -> str | None:
        return str(_repository_path(value)) if value else None

    root = _repository_path(data.root)
    try:
        split_manifest_value = str(data.split_manifest).format(
            dataset=data.dataset,
            split_seed=data.split_seed,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise PilotV3Error(
            f"Invalid split manifest template {data.split_manifest!r}: {exc}"
        ) from exc
    split_manifest = _repository_path(split_manifest_value)
    if not root.exists():
        raise PilotV3Error(f"Dataset root is missing: {root}")
    if not split_manifest.is_file():
        raise PilotV3Error(f"Existing split manifest is required: {split_manifest}")
    if data.dataset == "cifar10dvs":
        frames_path = resolved(data.frames_path)
        if frames_path is None or not Path(frames_path).exists():
            raise PilotV3Error(f"CIFAR10-DVS frames path is missing: {frames_path}")
    else:
        frames_path = data.frames_path
    resolved_data = dataclasses.replace(
        data,
        root=str(root),
        split_manifest=str(split_manifest),
        frames_path=frames_path,
        labels_path=resolved(data.labels_path),
        test_frames_path=resolved(data.test_frames_path),
        test_labels_path=resolved(data.test_labels_path),
        download=False,
        num_workers=0,
    )
    return dataclasses.replace(config, data=resolved_data)


def load_fixed_real_batch(
    base: RunConfig,
    *,
    seed: int,
    required_batch_size: int,
) -> tuple[RunConfig, torch.Tensor, torch.Tensor, Mapping[str, Any], dict[str, Any]]:
    """Load the dataset's real training path once and hold the tensors fixed."""

    resolved = _resolved_data_config(base)
    optimizer = dataclasses.replace(
        resolved.optimizer, batch_size=int(required_batch_size)
    )
    runtime = dataclasses.replace(resolved.runtime, seed=int(seed))
    resolved = dataclasses.replace(resolved, optimizer=optimizer, runtime=runtime)
    loaders = build_loaders(resolved, final_test=False, seed=seed)
    try:
        batch = next(iter(loaders["train"]))
    except StopIteration as exc:
        raise PilotV3Error("The dataset training loader produced no batch") from exc
    inputs = torch.as_tensor(batch[0]).detach().cpu().contiguous()
    targets = torch.as_tensor(batch[1]).detach().cpu().long().contiguous()
    if inputs.shape[0] != targets.shape[0]:
        raise PilotV3Error("Fixed input and target batch sizes differ")
    validate_fixed_batch_shape(
        resolved.data.dataset,
        list(inputs.shape),
        minimum_batch_size=required_batch_size,
        time_steps=resolved.model.time_steps,
        in_channels=resolved.model.in_channels,
    )
    if not torch.isfinite(inputs).all().item():
        raise PilotV3Error("Fixed real batch contains non-finite input values")
    manifest = loaders.get("_manifest")
    if not isinstance(manifest, Mapping) or not manifest.get("manifest_sha256"):
        raise PilotV3Error("The loader did not return a bound split manifest")
    source: dict[str, Any] = {
        "split_source_fingerprint": manifest.get("source_fingerprint"),
    }
    if resolved.data.dataset == "cifar10dvs":
        frames = Path(str(resolved.data.frames_path))
        index = frames / "index.csv" if frames.is_dir() else frames
        if index.name.lower() != "index.csv" or not index.is_file():
            raise PilotV3Error("The v3 DVS health gate requires indexed folder data")
        source["dvs_index_sha256"] = sha256_file(index)
        if source["dvs_index_sha256"] != manifest.get("source_fingerprint"):
            raise PilotV3Error("DVS split manifest is not bound to the current index.csv")
    return resolved, inputs, targets, manifest, source


def _timed_c2_rounds(
    config: RunConfig,
    *,
    device: torch.device,
    batch: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iterations: int,
    maximum_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    round_orders = ((False, True), (True, False))
    rounds: list[dict[str, Any]] = []
    samples: dict[str, list[float]] = {"ta_frozen": [], "ta_enabled": []}
    peaks: dict[str, list[int]] = {"ta_frozen": [], "ta_enabled": []}
    round_ratios: list[float] = []
    for round_index, order in enumerate(round_orders, start=1):
        row: dict[str, Any] = {
            "round": round_index,
            "order": ["ta_enabled" if enabled else "ta_frozen" for enabled in order],
        }
        for enabled in order:
            label = "ta_enabled" if enabled else "ta_frozen"
            result = legacy_gate._timed_mode(
                config,
                device=device,
                batch=batch,
                ta_enabled=enabled,
                warmup=warmup,
                iterations=iterations,
            )
            row[label] = result
            samples[label].extend(float(value) for value in result["cuda_step_ms"])
            peaks[label].append(int(result["peak_allocated_bytes"]))
            gc.collect()
            torch.cuda.empty_cache()
        frozen = float(row["ta_frozen"]["cuda_step_median_ms"])
        enabled = float(row["ta_enabled"]["cuda_step_median_ms"])
        ratio = enabled / frozen if frozen > 0.0 else math.inf
        row["ta_enabled_over_frozen"] = ratio
        row["pass"] = bool(math.isfinite(ratio) and ratio <= maximum_ratio)
        round_ratios.append(ratio)
        rounds.append(row)
    frozen_combined = float(statistics.median(samples["ta_frozen"]))
    enabled_combined = float(statistics.median(samples["ta_enabled"]))
    combined_ratio = (
        enabled_combined / frozen_combined if frozen_combined > 0.0 else math.inf
    )
    conservative_ratio = max(round_ratios)
    passed = bool(
        all(math.isfinite(value) and value <= maximum_ratio for value in round_ratios)
        and math.isfinite(combined_ratio)
        and combined_ratio <= maximum_ratio
    )
    condition = {
        "design": "two fresh-model rounds with reversed AB/BA order",
        "rounds": rounds,
        "aggregate": {
            "ta_frozen_cuda_step_median_ms": frozen_combined,
            "ta_enabled_cuda_step_median_ms": enabled_combined,
            "ta_frozen_peak_allocated_bytes": max(peaks["ta_frozen"]),
            "ta_enabled_peak_allocated_bytes": max(peaks["ta_enabled"]),
            "combined_ta_enabled_over_frozen": combined_ratio,
            "conservative_ta_enabled_over_frozen": conservative_ratio,
        },
    }
    ratio = {
        "ta_enabled_over_frozen": conservative_ratio,
        "combined_ta_enabled_over_frozen": combined_ratio,
        "round_ratios": round_ratios,
        "maximum_allowed": maximum_ratio,
        "pass": passed,
    }
    return condition, ratio


def run_cuda_timing_gate(
    base: RunConfig,
    *,
    seed: int,
    device: torch.device,
    output_parent: Path,
    batch: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iterations: int,
    maximum_ratio: float,
) -> dict[str, Any]:
    """Time C1 plus exactly one C2 frozen/enabled comparison; never build C3/C4."""

    c1 = legacy_gate.diagnostic_config(
        base, condition="C1", seed=seed, device=device, output_parent=output_parent
    )
    c1_timing = legacy_gate._timed_mode(
        c1,
        device=device,
        batch=batch,
        ta_enabled=False,
        warmup=warmup,
        iterations=iterations,
    )
    gc.collect()
    torch.cuda.empty_cache()
    c2 = legacy_gate.diagnostic_config(
        base, condition="C2", seed=seed, device=device, output_parent=output_parent
    )
    c2_timing, c2_ratio = _timed_c2_rounds(
        c2,
        device=device,
        batch=batch,
        warmup=warmup,
        iterations=iterations,
        maximum_ratio=maximum_ratio,
    )
    return {
        "pass": c2_ratio["pass"],
        "method": (
            "synchronized CUDA events around real single-batch training steps; "
            "C2 uses two fresh-model rounds in reversed AB/BA order"
        ),
        "conditions": {"C1": {"lif_standard": c1_timing}, "C2": c2_timing},
        "ta_enabled_over_frozen_ratios": {"C2": c2_ratio},
    }


def run_gate(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_path: Path,
    config: RunConfig,
    block: PilotBlock,
    device_name: str,
    git_identity: Mapping[str, Any],
) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.perf_counter()
    acceptance = protocol["pilot_acceptance"]
    overfit = acceptance["overfit"]
    timing = acceptance["timing"]
    environment_binding = acceptance["environment"]
    if not config.runtime.deterministic or config.runtime.amp:
        raise PilotV3Error("v3 health requires deterministic float32 execution with amp=false")
    legacy_gate.seed_everything(block.seed, deterministic=True)
    device = torch.device(device_name)
    hardware = legacy_gate.require_target_cuda(
        device, str(environment_binding["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_binding, hardware)

    required_batch_size = max(int(overfit["batch_size"]), int(timing["batch_size"]))
    resolved, all_inputs, all_targets, split_manifest, source_identity = (
        load_fixed_real_batch(
            config, seed=block.seed, required_batch_size=required_batch_size
        )
    )
    overfit_batch = (
        all_inputs[: int(overfit["batch_size"])].to(device, non_blocking=False),
        all_targets[: int(overfit["batch_size"])].to(device, non_blocking=False),
    )
    timing_batch = (
        all_inputs[: int(timing["batch_size"])].to(device, non_blocking=False),
        all_targets[: int(timing["batch_size"])].to(device, non_blocking=False),
    )

    condition_reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix=f"v3-health-{block.dataset}-", dir=str(block.health_output.parent)
    ) as temporary:
        root = Path(temporary)
        for condition in block.conditions:
            condition_reports[condition] = legacy_gate.run_condition_health(
                resolved,
                condition=condition,
                seed=block.seed,
                device=device,
                output_parent=block.health_output.parent,
                batch=overfit_batch,
                overfit_steps=int(overfit["steps"]),
                minimum_accuracy=float(overfit["minimum_accuracy"]),
                maximum_loss_fraction=float(overfit["maximum_loss_fraction"]),
                minimum_ta_coverage=float(
                    overfit["minimum_ta_routed_gradient_coverage"]
                ),
                checkpoint_path=root / condition / "last.pt",
            )
            gc.collect()
            torch.cuda.empty_cache()

    idle = legacy_gate.gpu_idle_precheck(device)
    timing_report = run_cuda_timing_gate(
        resolved,
        seed=block.seed,
        device=device,
        output_parent=block.health_output.parent,
        batch=timing_batch,
        warmup=int(timing["warmup_steps"]),
        iterations=int(timing["timed_steps"]),
        maximum_ratio=float(timing["maximum_ta_enabled_over_frozen_ratio"]),
    )
    failures = [
        f"{condition}: functional/gradient/checkpoint/overfit gate failed"
        for condition, item in condition_reports.items()
        if not item["pass"]
    ]
    c2_ratio = timing_report["ta_enabled_over_frozen_ratios"]["C2"]
    if not c2_ratio["pass"]:
        failures.append(
            "C2: TA step ratio "
            f"{c2_ratio['ta_enabled_over_frozen']:.4f} exceeds "
            f"{c2_ratio['maximum_allowed']:.4f}"
        )
    passed = not failures
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v3-pilot dataset-specific implementation health validation",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "dataset": block.dataset,
        "seed": block.seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity["git_commit"],
        "tracked_clean": git_identity["tracked_clean"],
        "attempt_receipt": str(block.attempt_receipt),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "source": {
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "config_hash": config.config_hash,
            "protocol": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "dataset": block.dataset,
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "fixed_batch_sha256": legacy_gate.tensor_batch_sha256(
                all_inputs, all_targets
            ),
            "fixed_batch_shape": list(all_inputs.shape),
            "overfit_batch_size": int(overfit["batch_size"]),
            "timing_batch_size": int(timing["batch_size"]),
            **source_identity,
        },
        "environment": {
            **environment_manifest(),
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
        "thresholds": {
            "overfit_steps": int(overfit["steps"]),
            "minimum_overfit_accuracy": float(overfit["minimum_accuracy"]),
            "maximum_overfit_loss_fraction": float(
                overfit["maximum_loss_fraction"]
            ),
            "minimum_ta_routed_gradient_coverage": float(
                overfit["minimum_ta_routed_gradient_coverage"]
            ),
            "maximum_ta_enabled_over_frozen_step_ratio": float(
                timing["maximum_ta_enabled_over_frozen_ratio"]
            ),
        },
        "functional_health_by_condition": condition_reports,
        "cuda_train_step_timing": timing_report,
        "failures": failures,
    }


def _fatal_report(
    *,
    block: PilotBlock,
    git_identity: Mapping[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v3-pilot dataset-specific implementation health validation",
        "status": "FAIL",
        "pass": False,
        "dataset": block.dataset,
        "seed": block.seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity.get("git_commit"),
        "tracked_clean": git_identity.get("tracked_clean"),
        "attempt_receipt": str(block.attempt_receipt),
        "attempt_receipt_sha256": sha256_file(block.attempt_receipt),
        "finished_at": utc_now(),
        "fatal_error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "seed_disposition": "CONSUMED_NONREPORTING_V3_HEALTH_NEVER_RETRY",
    }


def _reference_config(
    protocol: Mapping[str, Any], protocol_path: Path, config_dir: Path, dataset: str
) -> tuple[Path, RunConfig]:
    expected = expected_pilot_configs(protocol, dataset)[0]
    path = (config_dir / f"{expected.runtime.run_id}.yaml").resolve()
    if path.parent != config_dir.resolve() or not path.is_file():
        raise PilotV3Error(f"Canonical C1 pilot config is missing: {path}")
    observed = load_run_config(path, protocol_path)
    if observed.as_dict() != expected.as_dict():
        raise PilotV3Error("C1 pilot YAML differs from the protocol-generated config")
    return path, observed


def validate_current_runtime_against_health_report(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
    block: PilotBlock,
    device: str,
) -> dict[str, Any]:
    """Recheck the matching dataset, environment, and fixed batch before pilot launch."""

    if not device or device == "auto":
        raise PilotV3Error("The v3 pilot requires an explicit CUDA device")
    acceptance = protocol["pilot_acceptance"]
    environment_binding = acceptance["environment"]
    overfit = acceptance["overfit"]
    timing = acceptance["timing"]
    legacy_gate.seed_everything(block.seed, deterministic=True)
    selected_device = torch.device(device)
    hardware = legacy_gate.require_target_cuda(
        selected_device, str(environment_binding["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_binding, hardware)
    report_environment = report.get("environment")
    if not isinstance(report_environment, Mapping) or hardware != report_environment.get(
        "hardware"
    ):
        raise PilotV3Error("Current CUDA hardware differs from the health report")
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
        raise PilotV3Error(
            "Current environment differs from the health report at: "
            + ", ".join(mismatches)
        )
    idle = legacy_gate.gpu_idle_precheck(selected_device)
    report_idle = report_environment.get("gpu_idle_precheck")
    if not isinstance(report_idle, Mapping) or idle.get("device_uuid") != report_idle.get(
        "device_uuid"
    ):
        raise PilotV3Error("Current CUDA device UUID differs from the health report")
    required = max(int(overfit["batch_size"]), int(timing["batch_size"]))
    _resolved, inputs, targets, manifest, source_identity = load_fixed_real_batch(
        reference_config, seed=block.seed, required_batch_size=required
    )
    source = report.get("source")
    if not isinstance(source, Mapping):
        raise PilotV3Error("Health report source evidence is missing")
    observed = {
        "split_manifest_sha256": manifest["manifest_sha256"],
        "fixed_batch_sha256": legacy_gate.tensor_batch_sha256(inputs, targets),
        "fixed_batch_shape": list(inputs.shape),
        **source_identity,
    }
    mismatches = [key for key, value in observed.items() if source.get(key) != value]
    if mismatches:
        raise PilotV3Error(
            "Current dataset identity differs from the health report at: "
            + ", ".join(mismatches)
        )
    return {
        "pass": True,
        "dataset": block.dataset,
        "seed": block.seed,
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
        block = resolve_pilot_block(
            protocol, args.dataset, repository_root=REPOSITORY_ROOT
        )
        author_freeze = require_v3_author_freeze(
            protocol, repository_root=REPOSITORY_ROOT
        )
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else _repository_path(artifact_paths_for_protocol(protocol)["pilot_matrix"])
        )
        config_path, config = _reference_config(
            protocol, protocol_path, config_dir, block.dataset
        )
        if block.health_output.exists():
            raise PilotV3Error("Canonical v3 health output already exists; no retry is allowed")
        if block.attempt_receipt.exists():
            raise PilotV3Error("This dataset health seed is already consumed; no retry is allowed")
        git_identity = repository_git_identity(REPOSITORY_ROOT)
        if git_identity["tracked_clean"] is not True:
            raise PilotV3Error("v3 health evidence requires a clean tracked worktree")
        receipt = attempt_receipt_payload(
            block=block,
            protocol_path=protocol_path,
            config_path=config_path,
            config=config,
            git_identity=git_identity,
            repository_root=REPOSITORY_ROOT,
        )
        exclusive_create_json(block.attempt_receipt, receipt)
        print(
            "V3_AUTHOR_FREEZE_PASS "
            f"signoff_sha256={author_freeze['signoff_sha256']}"
        )
        print(f"V3_HEALTH_ATTEMPT_CLAIMED dataset={block.dataset} seed={block.seed}")
        print("SEED_EXECUTION_BEGINS")
    except Exception as exc:
        print(f"V3_HEALTH_GATE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
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
        )
    except BaseException as exc:
        report = _fatal_report(block=block, git_identity=git_identity, exc=exc)
    if block.health_output.exists():
        print("V3_HEALTH_GATE_BLOCKED: output appeared during execution", file=sys.stderr)
        return 2
    atomic_write_json(block.health_output, report)
    print(f"V3_HEALTH_GATE_{report['status']}")
    print(f"NON_REPORTING_OUTPUT={block.health_output}")
    print(f"SEED_{block.seed}_CONSUMED_DO_NOT_RETRY")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
