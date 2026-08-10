"""Operation and activity accounting for inference benchmarks.

Counts in this module are estimates, not hardware energy measurements.  Dense
MAC-equivalents are obtained from executed Conv2d/Linear calls.  Binary-input
calls are separated into activity-scaled synaptic additions (SyOPs); analog
calls remain MACs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Mapping

import torch
from torch import nn


NEURON_OPERATION_MODES = frozenset(
    {
        "none",
        "shared_window",
        "time_indexed_bank",
        "count_indexed_bank",
    }
)


def resolve_neuron_operation_mode(
    *,
    talif_active: bool | None = None,
    neuron_operation_mode: str | None = None,
) -> str:
    """Resolve explicit V6 control accounting with legacy TA-LIF support.

    ``talif_active`` predates the V6 ablations and denotes the original
    count-indexed TA-LIF implementation.  New callers must use
    ``neuron_operation_mode`` so a shared window or time-indexed bank is not
    accidentally charged for per-neuron spike-history state.
    """

    if neuron_operation_mode is not None:
        mode = str(neuron_operation_mode).strip().lower()
        if mode not in NEURON_OPERATION_MODES:
            raise ValueError(
                "neuron_operation_mode must be one of "
                f"{sorted(NEURON_OPERATION_MODES)}, got {neuron_operation_mode!r}"
            )
        if talif_active is not None:
            legacy_mode = "count_indexed_bank" if talif_active else "none"
            if mode != legacy_mode:
                raise ValueError(
                    "talif_active conflicts with neuron_operation_mode: "
                    f"{talif_active!r} implies {legacy_mode!r}, got {mode!r}"
                )
        return mode
    return "count_indexed_bank" if talif_active else "none"


@dataclass(frozen=True)
class OperationEstimate:
    batch_size: int
    dense_mac_equivalents: float
    macs: float
    syops: float
    binary_layer_calls: int
    analog_layer_calls: int
    shared_threshold_window_accesses: float
    threshold_bank_accesses: float
    spike_count_updates: float
    firing_rate: float | None
    activity_elements: float | None
    neuron_operation_mode: str = "none"
    method: str = "forward_hooks_activity_scaled"

    def per_sample(self) -> dict[str, float | int | str | None]:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        record = asdict(self)
        for name in (
            "dense_mac_equivalents",
            "macs",
            "syops",
            "shared_threshold_window_accesses",
            "threshold_bank_accesses",
            "spike_count_updates",
            "activity_elements",
        ):
            value = record[name]
            if value is not None:
                record[f"{name}_per_sample"] = float(value) / self.batch_size
        return record


@dataclass(frozen=True)
class EnergyConstants:
    """Explicit joules-per-operation constants for a labeled energy model."""

    mac_j: float
    syop_j: float
    threshold_access_j: float
    count_update_j: float
    source: str
    technology: str
    precision: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "EnergyConstants":
        required = (
            "mac_j",
            "syop_j",
            "threshold_access_j",
            "count_update_j",
            "source",
            "technology",
            "precision",
        )
        missing = [name for name in required if name not in values]
        if missing:
            raise ValueError(f"Energy constants are incomplete: missing {missing}")
        constants = cls(
            mac_j=float(values["mac_j"]),
            syop_j=float(values["syop_j"]),
            threshold_access_j=float(values["threshold_access_j"]),
            count_update_j=float(values["count_update_j"]),
            source=str(values["source"]).strip(),
            technology=str(values["technology"]).strip(),
            precision=str(values["precision"]).strip(),
        )
        numeric = (
            constants.mac_j,
            constants.syop_j,
            constants.threshold_access_j,
            constants.count_update_j,
        )
        if any(not torch.isfinite(torch.tensor(value)).item() or value < 0 for value in numeric):
            raise ValueError("Energy constants must be finite and non-negative")
        if not constants.source or not constants.technology or not constants.precision:
            raise ValueError("Energy source, technology, and precision labels are required")
        return constants


def estimate_energy(
    operations: OperationEstimate,
    constants: EnergyConstants | None,
) -> dict[str, Any]:
    """Estimate energy only when a complete, sourced constant set is supplied."""

    if constants is None:
        return {
            "energy_j": None,
            "energy_j_per_sample": None,
            "energy_status": "not_estimated_missing_constants",
            "energy_model_source": "",
            "energy_model_technology": "",
            "energy_model_precision": "",
        }
    operation_counts = (
        operations.macs,
        operations.syops,
        operations.shared_threshold_window_accesses,
        operations.threshold_bank_accesses,
        operations.spike_count_updates,
    )
    if any(not torch.isfinite(torch.tensor(value)).item() for value in operation_counts):
        return {
            "energy_j": None,
            "energy_j_per_sample": None,
            "energy_status": "not_estimated_incomplete_operation_counts",
            "energy_model_source": constants.source,
            "energy_model_technology": constants.technology,
            "energy_model_precision": constants.precision,
        }
    energy = (
        operations.macs * constants.mac_j
        + operations.syops * constants.syop_j
        + operations.shared_threshold_window_accesses * constants.threshold_access_j
        + operations.threshold_bank_accesses * constants.threshold_access_j
        + operations.spike_count_updates * constants.count_update_j
    )
    return {
        "energy_j": float(energy),
        "energy_j_per_sample": float(energy / operations.batch_size),
        "energy_status": "modeled_from_explicit_constants",
        "energy_model_source": constants.source,
        "energy_model_technology": constants.technology,
        "energy_model_precision": constants.precision,
    }


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, Mapping):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _is_binary_zero_one(tensor: torch.Tensor) -> bool:
    if tensor.numel() == 0 or not tensor.is_floating_point():
        return False
    detached = tensor.detach()
    return bool(torch.all((detached == 0) | (detached == 1)).item())


def _activity_fraction(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(torch.count_nonzero(tensor.detach()).item() / tensor.numel())


def _numeric_diagnostic(diagnostics: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name not in diagnostics or diagnostics[name] is None:
            continue
        value = diagnostics[name]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            return float(value.detach().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _neuron_layer_step_calls(diagnostics: Mapping[str, Any]) -> float | None:
    """Read the number of logical neuron-layer evaluations from diagnostics."""

    reported = _numeric_diagnostic(diagnostics, "neuron_layer_step_calls")
    if reported is not None:
        return reported
    layers = diagnostics.get("layers")
    timesteps = _numeric_diagnostic(diagnostics, "timesteps")
    if isinstance(layers, Mapping) and timesteps is not None:
        return float(len(layers) * timesteps)
    return None


def activity_from_diagnostics(
    diagnostics: Mapping[str, Any] | None,
    *,
    talif_active: bool | None = None,
    neuron_operation_mode: str | None = None,
) -> dict[str, float | None | str]:
    """Extract activity and explicit neuron-control operation estimates.

    Shared and time-indexed windows select two scalar boundaries once for each
    neuron-layer evaluation.  The full TA-LIF count-indexed bank selects two
    boundaries and updates history for each neuron state.  These are logical
    control-operation proxies, not an instruction-level or memory-traffic
    trace; measured CUDA latency and memory remain separate observations.
    """

    mode = resolve_neuron_operation_mode(
        talif_active=talif_active,
        neuron_operation_mode=neuron_operation_mode,
    )

    if not diagnostics:
        shared_accesses = None if mode == "shared_window" else 0.0
        bank_accesses = None if mode in {"time_indexed_bank", "count_indexed_bank"} else 0.0
        count_updates = None if mode == "count_indexed_bank" else 0.0
        return {
            "firing_rate": None,
            "activity_elements": None,
            "shared_threshold_window_accesses": shared_accesses,
            "threshold_bank_accesses": bank_accesses,
            "spike_count_updates": count_updates,
            "activity_method": f"{mode};diagnostics_unavailable",
        }
    spike_count = _numeric_diagnostic(diagnostics, "spike_count", "spikes")
    elements = _numeric_diagnostic(diagnostics, "elements", "activity_elements", "neuron_updates")
    firing_rate = _numeric_diagnostic(diagnostics, "spike_rate", "firing_rate")
    if firing_rate is None and spike_count is not None and elements and elements > 0:
        firing_rate = spike_count / elements

    shared_accesses = 0.0
    threshold_accesses = 0.0
    count_updates = 0.0
    if mode in {"shared_window", "time_indexed_bank"}:
        layer_steps = _neuron_layer_step_calls(diagnostics)
        if layer_steps is None:
            raise ValueError(
                f"{mode} operation accounting requires neuron_layer_step_calls diagnostics"
            )
        if mode == "shared_window":
            shared_accesses = 2.0 * layer_steps
            method = "estimated_two_shared_window_boundary_reads_per_neuron_layer_step"
        else:
            threshold_accesses = 2.0 * layer_steps
            method = "estimated_two_time_indexed_bank_boundary_reads_per_neuron_layer_step"
    elif mode == "count_indexed_bank":
        if elements is None:
            threshold_accesses = None
            count_updates = None
        else:
            threshold_accesses = 2.0 * elements
            count_updates = elements
        method = "estimated_two_count_indexed_bank_boundary_reads_and_one_count_update_per_state"
    else:
        method = "fixed_window_no_adaptive_routing_state"

    return {
        "firing_rate": firing_rate,
        "activity_elements": elements,
        "shared_threshold_window_accesses": shared_accesses,
        "threshold_bank_accesses": threshold_accesses,
        "spike_count_updates": count_updates,
        "activity_method": f"{mode};{method}",
    }


class OperationCounter:
    """Context manager that counts executed Conv2d/Linear operations."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._handles: list[Any] = []
        self.dense_mac_equivalents = 0.0
        self.macs = 0.0
        self.syops = 0.0
        self.binary_layer_calls = 0
        self.analog_layer_calls = 0

    def __enter__(self) -> "OperationCounter":
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self._handles.append(module.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        input_tensor = _first_tensor(inputs)
        output_tensor = _first_tensor(output)
        if input_tensor is None or output_tensor is None:
            return
        if isinstance(module, nn.Conv2d):
            operations_per_output = (module.in_channels // module.groups) * prod(module.kernel_size)
        elif isinstance(module, nn.Linear):
            operations_per_output = module.in_features
        else:
            return
        dense = float(output_tensor.numel() * operations_per_output)
        self.dense_mac_equivalents += dense
        if _is_binary_zero_one(input_tensor):
            self.binary_layer_calls += 1
            self.syops += dense * _activity_fraction(input_tensor)
        else:
            self.analog_layer_calls += 1
            self.macs += dense

    def result(
        self,
        *,
        batch_size: int,
        diagnostics: Mapping[str, Any] | None,
        talif_active: bool | None = None,
        neuron_operation_mode: str | None = None,
    ) -> OperationEstimate:
        mode = resolve_neuron_operation_mode(
            talif_active=talif_active,
            neuron_operation_mode=neuron_operation_mode,
        )
        activity = activity_from_diagnostics(
            diagnostics,
            neuron_operation_mode=mode,
        )
        shared_accesses = activity["shared_threshold_window_accesses"]
        threshold_accesses = activity["threshold_bank_accesses"]
        count_updates = activity["spike_count_updates"]
        return OperationEstimate(
            batch_size=batch_size,
            dense_mac_equivalents=self.dense_mac_equivalents,
            macs=self.macs,
            syops=self.syops,
            binary_layer_calls=self.binary_layer_calls,
            analog_layer_calls=self.analog_layer_calls,
            shared_threshold_window_accesses=(
                float(shared_accesses) if shared_accesses is not None else float("nan")
            ),
            threshold_bank_accesses=(
                float(threshold_accesses) if threshold_accesses is not None else float("nan")
            ),
            spike_count_updates=(
                float(count_updates) if count_updates is not None else float("nan")
            ),
            firing_rate=(
                float(activity["firing_rate"]) if activity["firing_rate"] is not None else None
            ),
            activity_elements=(
                float(activity["activity_elements"])
                if activity["activity_elements"] is not None
                else None
            ),
            neuron_operation_mode=mode,
            method=f"forward_hooks_activity_scaled;{activity['activity_method']}",
        )
