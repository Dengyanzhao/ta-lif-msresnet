#!/usr/bin/env python3
"""Benchmark one trained checkpoint with synchronized CUDA events."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.benchmark import benchmark_model  # noqa: E402
from talif_msresnet.config import load_protocol, validate_run_mapping  # noqa: E402
from talif_msresnet.data import (  # noqa: E402
    load_representative_batch_artifact,
    validate_representative_batch_artifact,
)
from talif_msresnet.models import build_model  # noqa: E402
from talif_msresnet.ops import EnergyConstants  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.utils import sha256_file  # noqa: E402


BENCHMARK_ID_VERSION = 1


def benchmark_device_identity(device_name: str) -> dict[str, Any]:
    """Resolve a CUDA request to the hardware identity bound into benchmark_id."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to resolve the benchmark hardware identity")
    requested = torch.device(device_name)
    if requested.type != "cuda":
        raise ValueError(f"Benchmark device must be CUDA, got {requested}")
    index = torch.cuda.current_device() if requested.index is None else int(requested.index)
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": f"cuda:{index}",
        "hardware_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def build_benchmark_identity(
    *,
    protocol_hash: str,
    checkpoint_sha256: str,
    input_file_sha256: str,
    input_batch_sha256: str,
    input_kind: str,
    device_identity: Mapping[str, Any],
    precision: str,
    warmup_iterations: int,
    timed_iterations: int,
    energy_constants_sha256: str,
) -> dict[str, Any]:
    """Return the canonical protocol payload and deterministic benchmark_id."""

    payload: dict[str, Any] = {
        "benchmark_id_version": BENCHMARK_ID_VERSION,
        "protocol_hash": protocol_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "input_file_sha256": input_file_sha256,
        "input_batch_sha256": input_batch_sha256,
        "input_kind": input_kind,
        "device_identity": dict(device_identity),
        "precision": precision,
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "energy_constants_sha256": energy_constants_sha256,
        "method": "synchronized_cuda_events_b1_b128",
        "synchronization": "cuda_device_synchronize_after_each_timed_forward",
        "input_residency": "preloaded_on_device_before_warmup_and_timing",
        "batch_sizes": [1, 128],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    payload["benchmark_id"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return payload


def validate_frozen_benchmark_request(
    benchmark_protocol: Mapping[str, Any],
    *,
    device: str,
    precision: str,
    warmup_iterations: int,
    timed_iterations: int,
) -> None:
    observed = {
        "device_type": str(device).lower().split(":", 1)[0],
        "precision": precision,
        "batch_sizes": [1, 128],
        "warmup_iterations": int(warmup_iterations),
        "timed_iterations": int(timed_iterations),
        "timing_method": "synchronized_cuda_events",
        "input_residency": "preloaded_on_device_before_warmup_and_timing",
    }
    expected = {key: benchmark_protocol[key] for key in observed}
    if observed != expected:
        raise ValueError(
            "Benchmark settings differ from the frozen study protocol: "
            f"observed={observed} expected={expected}"
        )


def resolve_frozen_energy_constants(
    benchmark_protocol: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    requested_path: Path | None,
) -> tuple[Path | None, str]:
    """Resolve the protocol-bound energy model and reject CLI substitution."""

    energy = benchmark_protocol.get("energy_model")
    if not isinstance(energy, Mapping):
        raise ValueError("Frozen benchmark protocol is missing energy_model")
    status = str(energy.get("status", ""))
    if status == "not_assessed":
        if requested_path is not None:
            raise ValueError(
                "--energy-constants cannot add a model after freeze; energy_model is not_assessed"
            )
        return None, "none"
    if status != "modeled":
        raise ValueError(f"Unknown frozen energy_model status: {status!r}")
    project_root = Path(protocol_path).resolve().parents[1]
    configured = (project_root / str(energy.get("constants_path", ""))).resolve()
    if project_root != configured and project_root not in configured.parents:
        raise ValueError("Frozen energy constants path escapes the project root")
    if requested_path is not None and requested_path.resolve() != configured:
        raise ValueError("--energy-constants differs from the path frozen in the protocol")
    if not configured.is_file():
        raise ValueError(f"Frozen energy constants file is missing: {configured}")
    digest = sha256_file(configured)
    if digest != energy.get("constants_sha256"):
        raise ValueError("Frozen energy constants SHA-256 differs from the protocol")
    return configured, digest


def checkpoint_split_manifest_sha256(
    checkpoint_path: Path, expected_config_hash: str | None = None
) -> str:
    manifest_path = checkpoint_path.parent / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest beside checkpoint: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Malformed run manifest: {manifest_path}")
    if expected_config_hash is not None and manifest.get("config_hash") != expected_config_hash:
        raise ValueError(f"Run manifest config hash differs from checkpoint: {manifest_path}")
    value = manifest.get("split_manifest_sha256")
    if not isinstance(value, str) or not value:
        raise ValueError(f"Run manifest has no split-manifest hash: {manifest_path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "protocol.yaml",
        help="Frozen study protocol that binds the formal benchmark.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--input-tensor",
        type=Path,
        help=(
            "The auditable validation artifact from export_representative_batch.py "
            "with at least 128 samples. "
            "Without it, activity/SyOP/energy fields are withheld."
        ),
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        metavar=("C", "H", "W"),
        help="Synthetic-input C H W shape; static CIFAR defaults to configured C x 32 x 32.",
    )
    parser.add_argument(
        "--temporal-input",
        action="store_true",
        help="Create synthetic [B,T,C,H,W] input rather than [B,C,H,W].",
    )
    parser.add_argument(
        "--energy-constants",
        type=Path,
        help="Exact sourced constants JSON already bound by the frozen protocol.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing row with the same complete benchmark identity.",
    )
    parser.add_argument("--expected-benchmark-id", help=argparse.SUPPRESS)
    return parser


def _torch_load(path: Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint {label} must be a mapping")
    return value


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _load_energy_constants(path: Path | None) -> EnergyConstants | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return EnergyConstants.from_mapping(_mapping(payload, "energy constants"))


def _representative_factory(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    largest_batch: int,
    device: str,
):
    if tensor.ndim not in (4, 5):
        raise ValueError("--input-tensor must have shape [N,C,H,W] or [N,T,C,H,W]")
    if tensor.shape[0] < largest_batch:
        raise ValueError(
            f"--input-tensor contains {tensor.shape[0]} samples; at least {largest_batch} are required"
        )
    if not tensor.is_floating_point():
        tensor = tensor.float()
    tensor = tensor.to(dtype=dtype, device=device)

    def factory(batch_size: int) -> torch.Tensor:
        return tensor[:batch_size]

    return factory


def _synthetic_factory(
    shape: tuple[int, int, int],
    *,
    time_steps: int,
    temporal: bool,
    dtype: torch.dtype,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20_260_719)

    def factory(batch_size: int) -> torch.Tensor:
        full_shape = (
            (batch_size, time_steps, *shape) if temporal else (batch_size, *shape)
        )
        return torch.randn(full_shape, generator=generator, dtype=torch.float32).to(dtype=dtype)

    return factory


def _flatten_record(
    result: Any,
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    parameter_report: Mapping[str, Any],
    precision: str,
    input_source: str,
    representative_activity: bool,
    benchmark_identity: Mapping[str, Any],
) -> dict[str, Any]:
    batch1 = result.batches[1]
    batch128 = result.batches[128]
    operations = dict(result.operations[1])
    energy = dict(result.energy[1])
    runtime = config.get("runtime", {}) if isinstance(config.get("runtime"), Mapping) else {}
    record: dict[str, Any] = {
        "benchmark_id": benchmark_identity["benchmark_id"],
        "benchmark_id_version": benchmark_identity["benchmark_id_version"],
        "protocol_hash": benchmark_identity["protocol_hash"],
        "run_id": runtime.get("run_id", checkpoint_path.parent.name),
        "experiment": config.get("experiment", ""),
        "dataset": data_config.get("dataset", data_config.get("name", "")),
        "depth": model_config.get("depth"),
        "time_steps": model_config.get("time_steps", model_config.get("timesteps")),
        "condition": model_config.get("condition"),
        "topology": model_config.get("topology"),
        "neuron": model_config.get("neuron"),
        "seed": runtime.get("seed", ""),
        "selected_epoch": (
            int(checkpoint["epoch"]) + 1 if checkpoint.get("epoch") is not None else ""
        ),
        "checkpoint_epoch_zero_based": checkpoint.get("epoch", ""),
        "config_hash": checkpoint.get("config_hash", ""),
        "checkpoint_path": artifact_path_reference(
            checkpoint_path,
            REPOSITORY_ROOT,
            external_identifier=f"checkpoint-sha256:{checkpoint_sha256}",
        ),
        "checkpoint_sha256": checkpoint_sha256,
        "precision": precision,
        "input_source": input_source,
        "input_file_sha256": benchmark_identity["input_file_sha256"],
        "input_batch_sha256": benchmark_identity["input_batch_sha256"],
        "energy_constants_sha256": benchmark_identity["energy_constants_sha256"],
        "benchmark_device_identity": json.dumps(
            benchmark_identity["device_identity"], sort_keys=True, separators=(",", ":")
        ),
        "benchmark_method": benchmark_identity["method"],
        "benchmark_synchronization": benchmark_identity["synchronization"],
        "benchmark_input_residency": benchmark_identity["input_residency"],
        "input_shape": "",
        "activity_status": (
            "representative_validation_input" if representative_activity else "withheld_synthetic_input"
        ),
        "warmup_iterations": batch1.warmup_iterations,
        "timed_iterations": batch1.iterations,
        "latency_b1_ms": batch1.latency_mean_ms,
        "latency_b1_sd_ms": batch1.latency_sd_ms,
        "latency_b1_p50_ms": batch1.latency_p50_ms,
        "latency_b1_p95_ms": batch1.latency_p95_ms,
        "latency_b128_ms": batch128.latency_mean_ms,
        "latency_b128_sd_ms": batch128.latency_sd_ms,
        "latency_b128_p50_ms": batch128.latency_p50_ms,
        "latency_b128_p95_ms": batch128.latency_p95_ms,
        "latency_b1_per_sample_ms": batch1.latency_mean_ms,
        "latency_b128_per_sample_ms": batch128.latency_mean_ms / 128.0,
        "peak_allocated_b1_bytes": batch1.peak_allocated_bytes,
        "incremental_peak_b1_bytes": batch1.incremental_peak_bytes,
        "peak_allocated_b128_bytes": batch128.peak_allocated_bytes,
        "incremental_peak_b128_bytes": batch128.incremental_peak_bytes,
        "timing_boundary": result.timing_boundary,
        "reset_state_boundary": result.reset_state_boundary,
        "hardware": torch.cuda.get_device_name(torch.device(result.device)),
        "software": f"torch={torch.__version__};cuda={torch.version.cuda}",
    }
    for name, value in parameter_report.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            record[name] = value
    record["dense_mac_equivalents_per_sample"] = operations.get(
        "dense_mac_equivalents_per_sample"
    )
    record["threshold_accesses_per_sample"] = operations.get(
        "threshold_bank_accesses_per_sample"
    )
    record["count_updates_per_sample"] = operations.get("spike_count_updates_per_sample")
    record["operation_count_method"] = operations.get("method", "")
    if representative_activity:
        record["firing_rate"] = operations.get("firing_rate")
        record["syops_per_sample"] = operations.get("syops_per_sample")
        record["macs_per_sample"] = operations.get("macs_per_sample")
        record.update(energy)
    else:
        record.update(
            {
                "firing_rate": np.nan,
                "syops_per_sample": np.nan,
                "macs_per_sample": np.nan,
                "energy_j": np.nan,
                "energy_j_per_sample": np.nan,
                "energy_status": "not_estimated_synthetic_input",
                "energy_model_source": "",
                "energy_model_technology": "",
                "energy_model_precision": "",
            }
        )
    return record


def _write_row(path: Path, row: Mapping[str, Any], replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([dict(row)])
    if path.exists():
        existing = pd.read_csv(path)
        if "benchmark_id" not in existing.columns:
            raise ValueError(
                f"Existing benchmark file predates benchmark_id; archive or migrate it: {path}"
            )
        duplicate = existing["benchmark_id"].astype(str) == str(row["benchmark_id"])
        if duplicate.any() and not replace:
            raise FileExistsError(
                "This complete benchmark identity already exists; pass --replace to rerun it"
            )
        existing = existing.loc[~duplicate]
        incoming = pd.concat([existing, incoming], ignore_index=True, sort=False)
    incoming.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be >= 0 and --iterations must be >= 1")
    if args.precision == "bfloat16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise SystemExit("The selected CUDA device does not support bfloat16")
    report = check_protocol(
        args.protocol,
        mode="full",
        project_root=REPOSITORY_ROOT,
        check_dependencies=False,
    )
    if not report.ok:
        raise SystemExit(
            "Benchmark preflight blocked:\n"
            + "\n".join(f"- {error}" for error in report.errors)
        )
    study_protocol = load_protocol(args.protocol)
    benchmark_protocol = study_protocol["benchmark"]
    try:
        validate_frozen_benchmark_request(
            benchmark_protocol,
            device=args.device,
            precision=args.precision,
            warmup_iterations=args.warmup,
            timed_iterations=args.iterations,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        energy_path, energy_constants_sha256 = resolve_frozen_energy_constants(
            benchmark_protocol,
            protocol_path=args.protocol,
            requested_path=args.energy_constants,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.input_tensor is None:
        raise SystemExit("Formal frozen benchmark requires representative --input-tensor")

    checkpoint_path = args.checkpoint.resolve()
    payload = _mapping(_torch_load(checkpoint_path), "payload")
    if "model_state" not in payload or "config" not in payload:
        raise ValueError("Checkpoint must contain model_state and config")
    config = _mapping(payload["config"], "config")
    resolved_config = validate_run_mapping(config)
    if payload.get("config_hash") != resolved_config.config_hash:
        raise ValueError("Checkpoint scientific config hash is invalid")
    if resolved_config.analysis.get("protocol_hash") != report.protocol_hash:
        raise ValueError("Checkpoint protocol hash differs from the frozen study protocol")
    model_config = _mapping(config.get("model", config), "config.model")
    data_config = config.get("data", {})
    data_config = _mapping(data_config, "config.data") if data_config else {}
    model = build_model(model_config)
    model.load_state_dict(payload["model_state"], strict=True)
    dtype = _dtype(args.precision)
    model.to(dtype=dtype)
    device_identity = benchmark_device_identity(args.device)

    dataset = str(data_config.get("dataset", "")).lower().replace("-", "").replace("_", "")
    time_steps = int(model_config.get("time_steps", model_config.get("timesteps", 1)))
    in_channels = int(model_config.get("in_channels", data_config.get("in_channels", 3)))
    representative = args.input_tensor is not None
    if representative:
        input_path = args.input_tensor.resolve()
        tensor, targets, batch_metadata = load_representative_batch_artifact(input_path)
        input_batch_sha256 = validate_representative_batch_artifact(
            tensor,
            targets,
            batch_metadata,
            expected_dataset=resolved_config.data.dataset,
            expected_protocol_hash=str(resolved_config.analysis.get("protocol_hash", "")),
            expected_split_manifest_sha256=checkpoint_split_manifest_sha256(
                checkpoint_path, str(payload.get("config_hash", ""))
            ),
            expected_in_channels=resolved_config.model.in_channels,
            expected_time_steps=resolved_config.model.time_steps,
            expected_num_classes=resolved_config.model.num_classes,
        )
        input_file_sha256 = sha256_file(input_path)
        input_kind = "representative_validation_batch"
        input_factory = _representative_factory(
            tensor,
            dtype=dtype,
            largest_batch=128,
            device=str(device_identity["device"]),
        )
        input_source = artifact_path_reference(
            input_path,
            REPOSITORY_ROOT,
            external_identifier=f"input-sha256:{input_file_sha256}",
        )
        input_shape = tuple(int(value) for value in tensor.shape[1:])
    else:
        if "dvs" in dataset and args.input_shape is None:
            raise ValueError("CIFAR10-DVS synthetic timing requires explicit --input-shape C H W")
        shape = tuple(args.input_shape or (in_channels, 32, 32))
        if shape[0] != in_channels:
            raise ValueError(
                f"Input shape channels ({shape[0]}) do not match model in_channels ({in_channels})"
            )
        input_factory = _synthetic_factory(
            shape,
            time_steps=time_steps,
            temporal=bool(args.temporal_input),
            dtype=dtype,
        )
        input_source = "synthetic_normal_latency_only"
        input_shape = ((time_steps, *shape) if args.temporal_input else shape)
        synthetic_spec = {
            "source": input_source,
            "shape": input_shape,
            "seed": 20_260_719,
            "dtype": args.precision,
        }
        encoded_spec = json.dumps(
            synthetic_spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        input_batch_sha256 = hashlib.sha256(encoded_spec.encode("utf-8")).hexdigest()
        input_file_sha256 = ""
        input_kind = "synthetic_latency_only"

    constants = _load_energy_constants(energy_path)
    if constants is not None and constants.source != benchmark_protocol["energy_model"]["source"]:
        raise ValueError("Energy constants source differs from the source frozen in the protocol")
    benchmark_identity = build_benchmark_identity(
        protocol_hash=report.protocol_hash,
        checkpoint_sha256=_sha256(checkpoint_path),
        input_file_sha256=input_file_sha256,
        input_batch_sha256=input_batch_sha256,
        input_kind=input_kind,
        device_identity=device_identity,
        precision=args.precision,
        warmup_iterations=args.warmup,
        timed_iterations=args.iterations,
        energy_constants_sha256=energy_constants_sha256,
    )
    if (
        args.expected_benchmark_id is not None
        and args.expected_benchmark_id != benchmark_identity["benchmark_id"]
    ):
        raise RuntimeError("Wrapper and single-checkpoint benchmark identities disagree")
    result = benchmark_model(
        model,
        input_factory,
        batch_sizes=(1, 128),
        warmup_iterations=args.warmup,
        iterations=args.iterations,
        device=device_identity["device"],
        talif_active=str(model_config.get("neuron", "")).replace("-", "_") == "ta_lif",
        energy_constants=constants,
    )
    parameter_report = model.parameter_report() if hasattr(model, "parameter_report") else {}
    checkpoint_hash = str(benchmark_identity["checkpoint_sha256"])
    row = _flatten_record(
        result,
        checkpoint=payload,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        config=config,
        model_config=model_config,
        data_config=data_config,
        parameter_report=parameter_report,
        precision=args.precision,
        input_source=input_source,
        representative_activity=representative,
        benchmark_identity=benchmark_identity,
    )
    row["input_shape"] = "x".join(str(value) for value in input_shape)
    _write_row(args.output, row, replace=args.replace)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
