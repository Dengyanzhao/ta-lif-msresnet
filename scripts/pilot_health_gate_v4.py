#!/usr/bin/env python3
"""Run one non-reporting C1/C2 v4 health gate for a bound dataset block."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import itertools
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)

import pilot_health_gate as legacy_gate  # noqa: E402
import pilot_health_gate_v3 as v3_gate  # noqa: E402
from talif_msresnet.config import (  # noqa: E402
    RunConfig,
    artifact_paths_for_protocol,
    load_protocol,
    load_run_config,
)
from talif_msresnet.data import build_loaders  # noqa: E402
from talif_msresnet.pilot_v4 import (  # noqa: E402
    HEALTH_ARTIFACT_CLASS,
    HEALTH_SCHEMA_VERSION,
    LONGITUDINAL_LOSS_REDUCTION_DEFINITION,
    LONGITUDINAL_RESIDUAL_COVERAGE_DEFINITION,
    LONGITUDINAL_SURROGATE_COVERAGE_DEFINITION,
    REPORTING_ELIGIBILITY,
    PilotBlock,
    PilotV4Error,
    attempt_receipt_payload,
    exclusive_create_json,
    expected_pilot_configs,
    repository_git_identity,
    require_v4_author_freeze,
    resolve_pilot_block,
)
from talif_msresnet.train import evaluate, train_one_epoch  # noqa: E402
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
        default=REPOSITORY_ROOT / "configs" / "protocol_v4_talif_only.yaml",
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--dataset", required=True, choices=("cifar100", "cifar10dvs"))
    parser.add_argument("--device", default="cuda:0")
    return parser


def _health_verdict(status: str, dataset: str) -> str:
    return f"V4_HEALTH_GATE_{status} dataset={dataset}"


def _repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


class _CountedIterable:
    def __init__(self, source: Iterable[Any], limit: int) -> None:
        self.source = source
        self.limit = int(limit)
        self.count = 0

    def __iter__(self) -> Iterator[Any]:
        for item in itertools.islice(self.source, self.limit):
            self.count += 1
            yield item


class _CaptureLogger:
    def __init__(self) -> None:
        self.batch_events: list[dict[str, Any]] = []

    def log(self, event: str, **values: Any) -> None:
        if event == "train_batch":
            self.batch_events.append(dict(values))


def _parameter_norm(model: torch.nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        norm = float(parameter.detach().double().norm(2).item())
        squared += norm * norm
    return math.sqrt(squared)


def _finite_required_values(
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
) -> list[float]:
    diagnostics = train_metrics.get("diagnostics")
    values = [
        train_metrics.get("loss"),
        train_metrics.get("accuracy"),
        train_metrics.get("gradient_mean"),
        train_metrics.get("nonzero_block_gradient_fraction_mean"),
        val_metrics.get("loss"),
        val_metrics.get("accuracy"),
        diagnostics.get("surrogate_coverage_min")
        if isinstance(diagnostics, Mapping)
        else None,
    ]
    return [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]


def run_longitudinal_condition(
    base: RunConfig,
    *,
    condition: str,
    seed: int,
    device: torch.device,
    output_parent: Path,
    acceptance: Mapping[str, Any],
    expected_split_manifest_sha256: str,
) -> dict[str, Any]:
    """Run the exact short real-data trajectory frozen by v4."""

    legacy_gate.seed_everything(seed, deterministic=True)
    config = legacy_gate.diagnostic_config(
        base,
        condition=condition,
        seed=seed,
        device=device,
        output_parent=output_parent,
    )
    config = dataclasses.replace(
        config,
        runtime=dataclasses.replace(
            config.runtime,
            limit_batches=None,
            log_every=1,
        ),
        analysis={**config.analysis, "collect_activity": True},
    )
    loaders = build_loaders(config, final_test=False, seed=seed)
    manifest = loaders.get("_manifest")
    if not isinstance(manifest, Mapping) or (
        manifest.get("manifest_sha256") != expected_split_manifest_sha256
    ):
        raise PilotV4Error("Longitudinal loader split differs from the fixed-batch split")

    model, optimizer, scheduler, ta_parameters = legacy_gate._build_training_objects(
        config, device, ta_enabled=False
    )
    initial_norm = _parameter_norm(model)
    if not math.isfinite(initial_norm) or initial_norm <= 0.0:
        raise PilotV4Error(f"{condition} initial parameter norm is invalid")
    epochs = int(acceptance["epochs"])
    train_limit = int(acceptance["train_batches_per_epoch"])
    val_limit = int(acceptance["validation_batches_per_epoch"])
    activation_epoch = int(math.ceil(config.optimizer.epochs * config.optimizer.ta_start_fraction))
    logger = _CaptureLogger()
    scaler = legacy_gate._grad_scaler()
    epoch_rows: list[dict[str, Any]] = []
    nonfinite = 0
    fixed_validation_batches = list(itertools.islice(loaders["val"], val_limit))
    if len(fixed_validation_batches) != val_limit:
        raise PilotV4Error(
            f"{condition} validation loader did not provide {val_limit} fixed batches"
        )

    for epoch in range(epochs):
        sampler = getattr(loaders["train"], "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        ta_enabled = bool(condition == "C2" and epoch >= activation_epoch)
        legacy_gate._set_ta_enabled(ta_parameters, ta_enabled)
        logger.batch_events.clear()
        train_batches = _CountedIterable(loaders["train"], train_limit)
        train_metrics = train_one_epoch(
            model,
            train_batches,
            optimizer,
            device,
            epoch,
            config,
            logger,  # type: ignore[arg-type]
            scaler,
        )
        val_batches = _CountedIterable(fixed_validation_batches, val_limit)
        val_metrics = evaluate(model, val_batches, device, config, "val")
        if train_batches.count != train_limit or len(logger.batch_events) != train_limit:
            raise PilotV4Error(
                f"{condition} epoch {epoch} did not execute {train_limit} train batches"
            )
        if val_batches.count != val_limit:
            raise PilotV4Error(
                f"{condition} epoch {epoch} did not execute {val_limit} validation batches"
            )
        required_values = _finite_required_values(train_metrics, val_metrics)
        if len(required_values) != 7:
            nonfinite += 1
        nonfinite += sum(not math.isfinite(value) for value in required_values)
        residual_coverage = float(train_metrics["nonzero_block_gradient_fraction_mean"])
        diagnostics = train_metrics.get("diagnostics")
        surrogate_coverage = (
            float(diagnostics.get("surrogate_coverage_min", float("nan")))
            if isinstance(diagnostics, Mapping)
            else float("nan")
        )
        norm = _parameter_norm(model)
        if not math.isfinite(norm):
            nonfinite += 1
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_batches": train_batches.count,
                "validation_batches": val_batches.count,
                "ta_enabled": ta_enabled,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "validation_loss": float(val_metrics["loss"]),
                "validation_accuracy": float(val_metrics["accuracy"]),
                "residual_gradient_batch_coverage": residual_coverage,
                "surrogate_support_coverage": surrogate_coverage,
                "parameter_norm": norm,
            }
        )
        scheduler.step()

    final_norm = _parameter_norm(model)
    first_loss = float(epoch_rows[0]["validation_loss"])
    final_loss = float(epoch_rows[-1]["validation_loss"])
    loss_reduction = (first_loss - final_loss) / first_loss if first_loss > 0.0 else -math.inf
    parameter_ratio = final_norm / initial_norm
    minimum_residual = min(float(row["residual_gradient_batch_coverage"]) for row in epoch_rows)
    minimum_surrogate = min(float(row["surrogate_support_coverage"]) for row in epoch_rows)
    passed = bool(
        minimum_residual >= float(acceptance["minimum_residual_gradient_batch_coverage"])
        and minimum_surrogate >= float(acceptance["minimum_surrogate_support_coverage"])
        and loss_reduction >= float(acceptance["minimum_loss_reduction_fraction"])
        and float(acceptance["parameter_norm_ratio_minimum"])
        <= parameter_ratio
        <= float(acceptance["parameter_norm_ratio_maximum"])
        and nonfinite <= int(acceptance["maximum_nonfinite_observations"])
    )
    return {
        "pass": passed,
        "condition": condition,
        "epochs": epochs,
        "train_batches_per_epoch": train_limit,
        "validation_batches_per_epoch": val_limit,
        "ta_activation_epoch_zero_based": activation_epoch,
        "epoch_metrics": epoch_rows,
        "residual_gradient_batch_coverage_definition": (
            LONGITUDINAL_RESIDUAL_COVERAGE_DEFINITION
        ),
        "minimum_epoch_residual_gradient_batch_coverage": minimum_residual,
        "surrogate_support_coverage_definition": (
            LONGITUDINAL_SURROGATE_COVERAGE_DEFINITION
        ),
        "minimum_epoch_surrogate_support_coverage": minimum_surrogate,
        "loss_reduction_definition": LONGITUDINAL_LOSS_REDUCTION_DEFINITION,
        "initial_validation_loss": first_loss,
        "final_validation_loss": final_loss,
        "loss_reduction_fraction": loss_reduction,
        "initial_parameter_norm": initial_norm,
        "final_parameter_norm": final_norm,
        "parameter_norm_ratio": parameter_ratio,
        "nonfinite_observations": nonfinite,
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
    longitudinal = acceptance["longitudinal_health"]
    timing = acceptance["timing"]
    environment_binding = acceptance["environment"]
    if not config.runtime.deterministic or config.runtime.amp:
        raise PilotV4Error("v4 health requires deterministic float32 execution with amp=false")
    legacy_gate.seed_everything(block.seed, deterministic=True)
    device = torch.device(device_name)
    hardware = legacy_gate.require_target_cuda(
        device, str(environment_binding["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_binding, hardware)

    required_batch_size = max(int(overfit["batch_size"]), int(timing["batch_size"]))
    resolved, all_inputs, all_targets, split_manifest, source_identity = (
        v3_gate.load_fixed_real_batch(
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
    longitudinal_reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix=f"v4-health-{block.dataset}-", dir=str(block.health_output.parent)
    ) as temporary:
        temporary_root = Path(temporary)
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
                minimum_ta_coverage=float(overfit["minimum_ta_routed_gradient_coverage"]),
                checkpoint_path=temporary_root / condition / "last.pt",
            )
            gc.collect()
            torch.cuda.empty_cache()
            longitudinal_reports[condition] = run_longitudinal_condition(
                resolved,
                condition=condition,
                seed=block.seed,
                device=device,
                output_parent=block.health_output.parent,
                acceptance=longitudinal,
                expected_split_manifest_sha256=str(split_manifest["manifest_sha256"]),
            )
            gc.collect()
            torch.cuda.empty_cache()

    idle = legacy_gate.gpu_idle_precheck(device)
    timing_report = v3_gate.run_cuda_timing_gate(
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
    failures.extend(
        f"{condition}: longitudinal health gate failed"
        for condition, item in longitudinal_reports.items()
        if not item["pass"]
    )
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
        "purpose": "pre-v4-pilot dataset-specific implementation health validation",
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
            "fixed_batch_sha256": legacy_gate.tensor_batch_sha256(all_inputs, all_targets),
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
            "overfit": dict(overfit),
            "longitudinal_health": dict(longitudinal),
            "timing": dict(timing),
        },
        "functional_health_by_condition": condition_reports,
        "longitudinal_health_by_condition": longitudinal_reports,
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
        "purpose": "pre-v4-pilot dataset-specific implementation health validation",
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
        "seed_disposition": "CONSUMED_NONREPORTING_V4_HEALTH_NEVER_RETRY",
    }


def _reference_config(
    protocol: Mapping[str, Any], protocol_path: Path, config_dir: Path, dataset: str
) -> tuple[Path, RunConfig]:
    expected = expected_pilot_configs(protocol, dataset)[0]
    path = (config_dir / f"{expected.runtime.run_id}.yaml").resolve()
    if path.parent != config_dir.resolve() or not path.is_file():
        raise PilotV4Error(f"Canonical C1 pilot config is missing: {path}")
    observed = load_run_config(path, protocol_path)
    if observed.as_dict() != expected.as_dict():
        raise PilotV4Error("C1 pilot YAML differs from the protocol-generated config")
    return path, observed


def validate_current_runtime_against_health_report(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
    block: PilotBlock,
    device: str,
) -> dict[str, Any]:
    """Recheck the dataset and runtime identity immediately before a v4 pilot."""

    try:
        result = v3_gate.validate_current_runtime_against_health_report(
            report,
            protocol=protocol,
            reference_config=reference_config,
            block=block,  # type: ignore[arg-type]
            device=device,
        )
    except Exception as exc:
        raise PilotV4Error(str(exc)) from exc
    return {**result, "protocol_version": 4}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol_path = args.protocol.resolve()
        protocol = load_protocol(protocol_path)
        block = resolve_pilot_block(protocol, args.dataset, repository_root=REPOSITORY_ROOT)
        author_freeze = require_v4_author_freeze(protocol, repository_root=REPOSITORY_ROOT)
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else _repository_path(artifact_paths_for_protocol(protocol)["pilot_matrix"])
        )
        config_path, config = _reference_config(protocol, protocol_path, config_dir, block.dataset)
        if block.health_output.exists():
            raise PilotV4Error("Canonical v4 health output exists; no retry is allowed")
        if block.attempt_receipt.exists():
            raise PilotV4Error("This dataset health seed is consumed; no retry is allowed")
        git_identity = repository_git_identity(REPOSITORY_ROOT)
        if git_identity["tracked_clean"] is not True:
            raise PilotV4Error("v4 health evidence requires a clean tracked worktree")
        receipt = attempt_receipt_payload(
            block=block,
            protocol_path=protocol_path,
            config_path=config_path,
            config=config,
            git_identity=git_identity,
            repository_root=REPOSITORY_ROOT,
        )
        exclusive_create_json(block.attempt_receipt, receipt)
        print(f"V4_AUTHOR_FREEZE_PASS signoff_sha256={author_freeze['signoff_sha256']}")
        print(f"V4_HEALTH_ATTEMPT_CLAIMED dataset={block.dataset} seed={block.seed}")
        print("SEED_EXECUTION_BEGINS")
    except Exception as exc:
        print(f"V4_HEALTH_GATE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
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
        print("V4_HEALTH_GATE_BLOCKED: output appeared during execution", file=sys.stderr)
        return 2
    atomic_write_json(block.health_output, report)
    print(_health_verdict(str(report["status"]), block.dataset))
    print(f"NON_REPORTING_OUTPUT={block.health_output}")
    print(f"SEED_{block.seed}_CONSUMED_DO_NOT_RETRY")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
