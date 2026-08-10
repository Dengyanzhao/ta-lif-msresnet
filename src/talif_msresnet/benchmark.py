"""CUDA timing and resource benchmark helpers.

Latency is measured only on CUDA with synchronized CUDA events.  There is no
CPU fallback because a CPU wall-clock value is not comparable to the planned
GPU table.  Callers may still use the statistics/operation helpers on CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .ops import EnergyConstants, OperationCounter, estimate_energy


class CudaBenchmarkUnavailable(RuntimeError):
    """Raised when a CUDA event benchmark was requested without CUDA."""


@dataclass(frozen=True)
class BatchBenchmark:
    batch_size: int
    iterations: int
    warmup_iterations: int
    latency_mean_ms: float
    latency_sd_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_min_ms: float
    latency_max_ms: float
    peak_allocated_bytes: int
    baseline_allocated_bytes: int
    incremental_peak_bytes: int


@dataclass
class BenchmarkResult:
    device: str
    batches: dict[int, BatchBenchmark]
    operations: dict[int, dict[str, Any]]
    energy: dict[int, dict[str, Any]]
    timing_boundary: str = "CUDA events around synchronized model forward; input already on device"
    reset_state_boundary: str = (
        "external model.reset_state() call is before the CUDA event; any reset performed "
        "inside model.forward is necessarily included in measured latency"
    )

    def to_records(self, *, metadata: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Flatten one result into one CSV-friendly row per batch size."""

        base = dict(metadata or {})
        rows: list[dict[str, Any]] = []
        for batch_size, batch in sorted(self.batches.items()):
            record = {
                **base,
                "device": self.device,
                "batch_size": batch_size,
                **asdict(batch),
                "timing_boundary": self.timing_boundary,
                "reset_state_boundary": self.reset_state_boundary,
            }
            operations = self.operations.get(batch_size, {})
            energy = self.energy.get(batch_size, {})
            record.update(operations)
            record.update(energy)
            rows.append(record)
        return rows


def _reset_state(model: nn.Module) -> None:
    reset = getattr(model, "reset_state", None)
    if reset is not None:
        reset()


def _forward_with_diagnostics(model: nn.Module, sample: torch.Tensor) -> tuple[Any, Mapping[str, Any] | None]:
    try:
        result = model(sample, collect_activity=True)
    except TypeError as error:
        if "collect_activity" not in str(error):
            raise
        result = model(sample)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], result[1]
    diagnostics = getattr(result, "diagnostics", None)
    return result, diagnostics if isinstance(diagnostics, Mapping) else None


def _forward_logits(model: nn.Module, sample: torch.Tensor) -> Any:
    try:
        return model(sample, collect_activity=False)
    except TypeError as error:
        if "collect_activity" not in str(error):
            raise
        return model(sample)


def _to_device(sample: Any, device: torch.device) -> torch.Tensor:
    if not isinstance(sample, torch.Tensor):
        raise TypeError("input_factory must return a torch.Tensor")
    return sample if sample.device == device else sample.to(device=device, non_blocking=False)


def _timed_batch(
    model: nn.Module,
    input_factory: Callable[[int], torch.Tensor],
    *,
    batch_size: int,
    warmup_iterations: int,
    iterations: int,
    device: torch.device,
) -> BatchBenchmark:
    if warmup_iterations < 0 or iterations < 1:
        raise ValueError("warmup_iterations must be >= 0 and iterations must be positive")

    sample = _to_device(input_factory(batch_size), device)
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _reset_state(model)
            _forward_logits(model, sample)
        torch.cuda.synchronize(device)

        # Inputs are allocated before the measurement.  This keeps host/device
        # transfer out of the reported model-forward latency.
        torch.cuda.reset_peak_memory_stats(device)
        baseline = int(torch.cuda.memory_allocated(device))
        timings: list[float] = []
        for _ in range(iterations):
            _reset_state(model)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _forward_logits(model, sample)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(float(start.elapsed_time(end)))
        peak = int(torch.cuda.max_memory_allocated(device))

    values = np.asarray(timings, dtype=float)
    return BatchBenchmark(
        batch_size=batch_size,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        latency_mean_ms=float(values.mean()),
        latency_sd_ms=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        latency_p50_ms=float(np.quantile(values, 0.50)),
        latency_p95_ms=float(np.quantile(values, 0.95)),
        latency_min_ms=float(values.min()),
        latency_max_ms=float(values.max()),
        peak_allocated_bytes=peak,
        baseline_allocated_bytes=baseline,
        incremental_peak_bytes=max(0, peak - baseline),
    )


def _operation_batch(
    model: nn.Module,
    input_factory: Callable[[int], torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    talif_active: bool | None,
    neuron_operation_mode: str | None,
) -> tuple[dict[str, Any], Mapping[str, Any] | None]:
    sample = _to_device(input_factory(batch_size), device)
    with torch.inference_mode(), OperationCounter(model) as counter:
        _reset_state(model)
        _, diagnostics = _forward_with_diagnostics(model, sample)
        torch.cuda.synchronize(device)
        estimate = counter.result(
            batch_size=batch_size,
            diagnostics=diagnostics,
            talif_active=talif_active,
            neuron_operation_mode=neuron_operation_mode,
        )
    return estimate.per_sample(), diagnostics


def benchmark_model(
    model: nn.Module,
    input_factory: Callable[[int], torch.Tensor],
    *,
    batch_sizes: Sequence[int] = (1, 128),
    warmup_iterations: int = 25,
    iterations: int = 100,
    device: str | torch.device = "cuda",
    talif_active: bool | None = None,
    neuron_operation_mode: str | None = None,
    energy_constants: EnergyConstants | None = None,
) -> BenchmarkResult:
    """Benchmark B=1 and B=128 (or explicit sizes) using synchronized CUDA events."""

    if not torch.cuda.is_available():
        raise CudaBenchmarkUnavailable(
            "CUDA is required for latency/memory measurements; no CPU fallback is provided"
        )
    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        raise CudaBenchmarkUnavailable(f"Expected a CUDA device, got {cuda_device}")
    if not batch_sizes or any(int(size) < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")

    model_was_training = model.training
    model.eval()
    model.to(cuda_device)
    batches: dict[int, BatchBenchmark] = {}
    operations: dict[int, dict[str, Any]] = {}
    energy: dict[int, dict[str, Any]] = {}
    try:
        with torch.cuda.device(cuda_device):
            for size in batch_sizes:
                size = int(size)
                batches[size] = _timed_batch(
                    model,
                    input_factory,
                    batch_size=size,
                    warmup_iterations=warmup_iterations,
                    iterations=iterations,
                    device=cuda_device,
                )
                operation_record, _ = _operation_batch(
                    model,
                    input_factory,
                    batch_size=size,
                    device=cuda_device,
                    talif_active=talif_active,
                    neuron_operation_mode=neuron_operation_mode,
                )
                operations[size] = operation_record
                # Reconstruct the aggregate count fields needed by the energy
                # calculation from per-sample records.
                operation_estimate = _operation_estimate_from_record(operation_record, size)
                energy[size] = estimate_energy(operation_estimate, energy_constants)
    finally:
        model.train(model_was_training)
    return BenchmarkResult(
        device=str(cuda_device),
        batches=batches,
        operations=operations,
        energy=energy,
    )


def _operation_estimate_from_record(record: Mapping[str, Any], batch_size: int) -> Any:
    """Reconstruct an OperationEstimate-like object for energy calculation."""

    from .ops import OperationEstimate

    def value(name: str) -> float:
        return float(record.get(f"{name}_per_sample", 0.0)) * batch_size

    return OperationEstimate(
        batch_size=batch_size,
        dense_mac_equivalents=value("dense_mac_equivalents"),
        macs=value("macs"),
        syops=value("syops"),
        binary_layer_calls=int(record.get("binary_layer_calls", 0)),
        analog_layer_calls=int(record.get("analog_layer_calls", 0)),
        shared_threshold_window_accesses=value("shared_threshold_window_accesses"),
        threshold_bank_accesses=value("threshold_bank_accesses"),
        spike_count_updates=value("spike_count_updates"),
        firing_rate=record.get("firing_rate"),
        activity_elements=(
            value("activity_elements") if record.get("activity_elements") is not None else None
        ),
    )
