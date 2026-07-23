#!/usr/bin/env python3
"""Run independent E3 diagnostics on one fixed representative batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from talif_msresnet.diagnostics import (  # noqa: E402
    DiagnosticProtocol,
    diagnose_model,
    diagnostic_id,
    representative_batch_sha256,
    write_diagnostic_outputs,
)
from talif_msresnet.config import load_protocol, validate_run_mapping  # noqa: E402
from talif_msresnet.data import (  # noqa: E402
    load_representative_batch_artifact,
    validate_representative_batch_artifact,
)
from talif_msresnet.models import build_model  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.train import SEED_METRIC_FIELDS  # noqa: E402
from talif_msresnet.utils import atomic_write_json, stable_hash, upsert_csv_row  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_artifact_reference(path: Path, digest: str, label: str) -> str:
    """Keep diagnostic metadata independent of the host filesystem layout."""

    return artifact_path_reference(
        path,
        ROOT,
        external_identifier=f"{label}-sha256:{digest}",
    )


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--input-batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol", type=Path, help="Optional JSON protocol")
    parser.add_argument(
        "--study-protocol",
        required=True,
        type=Path,
        help="Frozen study protocol that binds the diagnostic settings and checkpoint.",
    )
    parser.add_argument("--device", help="cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, help="Use the first N fixed samples")
    parser.add_argument("--probes", type=int, help="Rademacher probes per block")
    parser.add_argument("--probe-seed", type=int)
    parser.add_argument("--time-index", type=int, help="Block invocation/time index; default -1")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument(
        "--no-update-seed-metrics",
        action="store_true",
        help="Write diagnostic artifacts without merging their scalar summaries into the run row",
    )
    return parser


def _finite_or_blank(value: Any) -> float | str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return ""
    return parsed if math.isfinite(parsed) else ""


def _update_seed_metrics(checkpoint_path: Path, summary: Mapping[str, Any]) -> None:
    run_dir = checkpoint_path.parent
    metrics_path = run_dir / "seed_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Cannot merge diagnostics; missing {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "complete":
        raise RuntimeError("Diagnostics can only be merged into a completed full-training run")
    if str(metrics.get("config_hash")) != str(summary.get("checkpoint_config_hash")):
        raise RuntimeError("Diagnostic checkpoint config hash does not match seed_metrics.json")
    if summary.get("diagnostic_status") != "complete":
        raise RuntimeError("Incomplete block Jacobians cannot be merged into seed metrics")
    activity = summary.get("activity_global", {})
    activity = activity if isinstance(activity, Mapping) else {}
    protocol = summary.get("protocol", {})
    protocol = protocol if isinstance(protocol, Mapping) else {}
    protocol_hash = stable_hash(
        {
            "protocol": dict(protocol),
            "jacobian_method": summary.get("jacobian_method", ""),
            "state_and_jacobian_boundary": summary.get("state_and_jacobian_boundary", ""),
        }
    )
    metrics.update(
        {
            "gradient_cv": _finite_or_blank(summary.get("gradient_cv_across_blocks")),
            "gradient_cv_method": summary.get("gradient_cv_method", ""),
            "jacobian_phi": _finite_or_blank(summary.get("jacobian_phi_mean_across_blocks")),
            "jacobian_varphi": _finite_or_blank(
                summary.get("jacobian_varphi_mean_across_blocks")
            ),
            "spike_rate": _finite_or_blank(activity.get("spike_rate")),
            "diagnostic_id": summary.get("diagnostic_id", ""),
            "diagnostic_batch_sha256": summary.get("input_batch_sha256", ""),
            "diagnostic_protocol_hash": protocol_hash,
            "jacobian_method": summary.get("jacobian_method", ""),
            "jacobian_probes": protocol.get("probes", ""),
            "jacobian_probe_seed": protocol.get("probe_seed", ""),
            "jacobian_time_index": protocol.get("time_index", ""),
        }
    )
    metrics_csv = run_dir.parent / "seed_metrics.csv"
    upsert_csv_row(
        metrics_csv,
        metrics,
        SEED_METRIC_FIELDS,
        key_field="run_id",
        require_existing=True,
    )
    atomic_write_json(metrics_path, metrics)


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if args.protocol:
        values = dict(_mapping(json.loads(args.protocol.read_text(encoding="utf-8")), "protocol"))
    overrides = {
        "device": args.device,
        "batch_size": args.batch_size,
        "probes": args.probes,
        "probe_seed": args.probe_seed,
        "time_index": args.time_index,
    }
    for key, value in overrides.items():
        if value is not None:
            values[key] = value
    allowed = {"device", "batch_size", "probes", "probe_seed", "time_index"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown diagnostic protocol keys: {unknown}")
    return values


def validate_frozen_diagnostic_settings(
    settings: Mapping[str, Any], analysis_protocol: Mapping[str, Any]
) -> None:
    expected = {
        "batch_size": int(analysis_protocol["diagnostic_batch_size"]),
        "probes": int(analysis_protocol["jacobian_probes"]),
        "probe_seed": int(analysis_protocol["jacobian_seed"]),
        "time_index": int(analysis_protocol["jacobian_time_index"]),
    }
    observed = {key: settings.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            "Diagnostic settings differ from the frozen study protocol: "
            f"observed={observed} expected={expected}"
        )


def _checkpoint_split_manifest_sha256(
    checkpoint_path: Path, expected_config_hash: str
) -> str:
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest beside checkpoint: {manifest_path}")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "run manifest")
    if manifest.get("config_hash") != expected_config_hash:
        raise ValueError(f"Run manifest config hash differs from checkpoint: {manifest_path}")
    value = manifest.get("split_manifest_sha256")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Run manifest has no split-manifest hash: {manifest_path}")
    return value


def _load_batch(
    path: Path,
    batch_size: int | None,
    *,
    checkpoint_path: Path,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    inputs, targets, metadata = load_representative_batch_artifact(path)
    validate_representative_batch_artifact(
        inputs,
        targets,
        metadata,
        expected_dataset=config.data.dataset,
        expected_protocol_hash=str(config.analysis.get("protocol_hash", "")),
        expected_split_manifest_sha256=_checkpoint_split_manifest_sha256(
            checkpoint_path, config.config_hash
        ),
        expected_in_channels=config.model.in_channels,
        expected_time_steps=config.model.time_steps,
        expected_num_classes=config.model.num_classes,
    )
    size = int(batch_size or inputs.shape[0])
    if size < 1 or size > inputs.shape[0]:
        raise ValueError(f"batch_size must be between 1 and {inputs.shape[0]}")
    return inputs[:size].contiguous(), targets[:size].contiguous(), metadata


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = _settings(args)
    report = check_protocol(
        args.study_protocol,
        mode="full",
        project_root=ROOT,
        check_dependencies=False,
    )
    if not report.ok:
        raise SystemExit(
            "Diagnostic preflight blocked:\n"
            + "\n".join(f"- {error}" for error in report.errors)
        )
    study_protocol = load_protocol(args.study_protocol)
    analysis_protocol = study_protocol["analysis"]
    try:
        validate_frozen_diagnostic_settings(settings, analysis_protocol)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    checkpoint_path = args.checkpoint.resolve()
    input_path = args.input_batch.resolve()
    checkpoint = _mapping(_torch_load(checkpoint_path), "checkpoint")
    if "model_state" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state and config")
    config = _mapping(checkpoint["config"], "checkpoint config")
    resolved_config = validate_run_mapping(config)
    if checkpoint.get("config_hash") != resolved_config.config_hash:
        raise ValueError("Checkpoint scientific config hash is invalid")
    if resolved_config.analysis.get("protocol_hash") != report.protocol_hash:
        raise ValueError("Checkpoint protocol hash differs from the frozen study protocol")
    model_config = _mapping(config.get("model", config), "model config")
    inputs, targets, batch_metadata = _load_batch(
        input_path,
        settings.get("batch_size"),
        checkpoint_path=checkpoint_path,
        config=resolved_config,
    )
    device_name = str(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)

    protocol = DiagnosticProtocol(
        probes=int(settings.get("probes", 8)),
        probe_seed=int(settings.get("probe_seed", 20_260_719)),
        time_index=int(settings.get("time_index", -1)),
    )
    checkpoint_hash = _sha256_file(checkpoint_path)
    input_file_hash = _sha256_file(input_path)
    batch_hash = representative_batch_sha256(inputs, targets)
    run_id = diagnostic_id(checkpoint_hash, batch_hash, protocol)

    existing_summary = args.output / "diagnostic_summary.json"
    if existing_summary.exists() and not args.replace:
        existing = json.loads(existing_summary.read_text(encoding="utf-8"))
        if existing.get("diagnostic_id") == run_id:
            if not args.no_update_seed_metrics:
                _update_seed_metrics(checkpoint_path, existing)
            print(f"already complete: {args.output}")
            return 0
        raise FileExistsError("Output contains another diagnostic run; pass --replace explicitly")

    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), Mapping) else {}
    data = config.get("data", {}) if isinstance(config.get("data"), Mapping) else {}
    metadata = {
        "diagnostic_id": run_id,
        "checkpoint_path": _portable_artifact_reference(
            checkpoint_path, checkpoint_hash, "checkpoint"
        ),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_config_hash": checkpoint.get("config_hash", ""),
        "input_batch_path": _portable_artifact_reference(
            input_path, input_file_hash, "input-file"
        ),
        "input_file_sha256": input_file_hash,
        "input_artifact_batch_sha256": batch_metadata.get(
            "representative_batch_sha256", ""
        ),
        "input_batch_sha256": batch_hash,
        "device": str(device),
        "run_id": runtime.get("run_id", checkpoint_path.parent.name),
        "experiment": config.get("experiment", ""),
        "dataset": data.get("dataset", ""),
        "condition": model_config.get("condition", ""),
        "depth": model_config.get("depth", ""),
        "time_steps": model_config.get("time_steps", model_config.get("timesteps", "")),
        "seed": runtime.get("seed", ""),
        "selected_epoch": (
            int(checkpoint["epoch"]) + 1 if checkpoint.get("epoch") is not None else ""
        ),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch", ""),
        "pytorch": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    result = diagnose_model(model, inputs, targets, protocol=protocol, metadata=metadata)
    paths = write_diagnostic_outputs(result, args.output, replace=args.replace)
    if result.summary.get("diagnostic_status") != "complete":
        raise RuntimeError(
            "One or more block Jacobians failed; long tables were retained but scalar metrics were not merged"
        )
    if not args.no_update_seed_metrics:
        _update_seed_metrics(checkpoint_path, result.summary)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
