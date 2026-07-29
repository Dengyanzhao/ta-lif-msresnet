"""Independent checkpoint diagnostics for the E3 mechanism analysis.

Local block Jacobians are evaluated on tensors captured during one fixed full
sequence forward pass.  For invocation t, J means d(block_output_t) /
d(block_input_t), conditional on the SNN state that existed immediately before
that invocation.  Temporal predecessors are therefore held fixed; these values
must not be described as end-to-end temporal Jacobian moments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn


JACOBIAN_METHOD = "Hutchinson/Rademacher reverse-over-reverse VJP"
DIAGNOSTIC_SCHEMA_VERSION = 2
JACOBIAN_BOUNDARY = (
    "local J=d(block_output_t)/d(block_input_t), conditional on the pre-invocation "
    "SNN state captured in one fixed full-sequence forward; temporal predecessors fixed"
)


@dataclass(frozen=True)
class DiagnosticProtocol:
    probes: int = 8
    probe_seed: int = 20_260_719
    time_index: int = -1

    def validate(self) -> None:
        if self.probes < 1:
            raise ValueError("probes must be positive")


@dataclass
class DiagnosticResult:
    block_gradients: pd.DataFrame
    block_jacobians: pd.DataFrame
    layer_activity: pd.DataFrame
    summary: dict[str, Any]


def tensor_sha256(tensor: Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous CPU bytes."""

    detached = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(detached.dtype).encode("ascii"))
    digest.update(json.dumps(list(detached.shape), separators=(",", ":")).encode("ascii"))
    digest.update(detached.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def representative_batch_sha256(inputs: Tensor, targets: Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(tensor_sha256(inputs).encode("ascii"))
    digest.update(tensor_sha256(targets).encode("ascii"))
    return digest.hexdigest()


def diagnostic_id(
    checkpoint_sha256: str,
    input_sha256: str,
    protocol: DiagnosticProtocol,
) -> str:
    payload = {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "input_sha256": input_sha256,
        "protocol": asdict(protocol),
        "method": JACOBIAN_METHOD,
        "boundary": JACOBIAN_BOUNDARY,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_residual_block(module: nn.Module) -> bool:
    name = module.__class__.__name__.lower()
    if name.endswith("basicblock") or name.endswith("residualblock"):
        return True
    return all(hasattr(module, attribute) for attribute in ("conv1", "conv2", "shortcut"))


def residual_blocks(model: nn.Module) -> list[tuple[str, nn.Module]]:
    blocks = [(name, module) for name, module in model.named_modules() if name and _is_residual_block(module)]
    if not blocks:
        raise ValueError("No residual blocks were found for E3 diagnostics")
    return blocks


def _tensor_output(value: Any) -> Tensor | None:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            result = _tensor_output(item)
            if result is not None:
                return result
    return None


def _capture_block_invocations(model: nn.Module):
    captures: dict[str, list[tuple[Tensor, Tensor]]] = {name: [] for name, _ in residual_blocks(model)}
    handles = []
    for name, module in residual_blocks(model):
        def hook(_module: nn.Module, args: tuple[Any, ...], output: Any, *, block_name: str = name) -> None:
            input_tensor = _tensor_output(args)
            output_tensor = _tensor_output(output)
            if input_tensor is None or output_tensor is None:
                raise TypeError(f"Residual block {block_name} did not expose tensor input/output")
            captures[block_name].append((input_tensor, output_tensor))

        handles.append(module.register_forward_hook(hook))
    return captures, handles


def _select_invocation(
    captures: Mapping[str, Sequence[tuple[Tensor, Tensor]]], time_index: int
) -> list[tuple[str, int, int, Tensor, Tensor]]:
    selected: list[tuple[str, int, int, Tensor, Tensor]] = []
    for block_name, invocations in captures.items():
        if not invocations:
            raise RuntimeError(f"Residual block {block_name} was not called")
        resolved = time_index if time_index >= 0 else len(invocations) + time_index
        if not 0 <= resolved < len(invocations):
            raise IndexError(
                f"time_index={time_index} is invalid for {block_name} with {len(invocations)} calls"
            )
        x, y = invocations[resolved]
        selected.append((block_name, resolved, len(invocations), x, y))
    return selected


def _forward_activity(model: nn.Module, inputs: Tensor) -> tuple[Tensor, Mapping[str, Any]]:
    try:
        result = model(inputs, collect_activity=True)
    except TypeError as error:
        if "collect_activity" not in str(error):
            raise
        result = model(inputs)
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
        return result[0], result[1]
    return result, {}


def _global_block_gradients(
    loss: Tensor,
    selected: Sequence[tuple[str, int, int, Tensor, Tensor]],
) -> tuple[pd.DataFrame, float]:
    inputs = [item[3] for item in selected]
    gradients = torch.autograd.grad(loss, inputs, retain_graph=True, allow_unused=True)
    rows: list[dict[str, Any]] = []
    rms_values: list[float] = []
    for (block_name, invocation, invocation_count, x, y), gradient in zip(selected, gradients):
        if gradient is None:
            l2 = rms = maximum = float("nan")
            status = "missing_disconnected"
        else:
            detached = gradient.detach().float()
            l2 = float(torch.linalg.vector_norm(detached).item())
            rms = float(torch.sqrt(torch.mean(detached.square())).item())
            maximum = float(detached.abs().max().item())
            status = "measured"
            if math.isfinite(rms):
                rms_values.append(rms)
        rows.append(
            {
                "block": block_name,
                "invocation_index": invocation,
                "invocation_count": invocation_count,
                "input_shape": "x".join(map(str, x.shape)),
                "output_shape": "x".join(map(str, y.shape)),
                "gradient_l2": l2,
                "gradient_rms": rms,
                "gradient_abs_max": maximum,
                "status": status,
                "objective": "mean cross-entropy on fixed representative batch",
            }
        )
    if len(rms_values) > 1 and float(np.mean(rms_values)) != 0.0:
        gradient_cv = float(np.std(rms_values, ddof=1) / np.mean(rms_values))
    else:
        gradient_cv = float("nan")
    return pd.DataFrame.from_records(rows), gradient_cv


def _rademacher_like(reference: Tensor, generator: torch.Generator) -> Tensor:
    values = torch.empty(reference.shape, dtype=reference.dtype, device=reference.device)
    values.bernoulli_(0.5, generator=generator)
    return values.mul_(2.0).sub_(1.0)


def _jacobian_moments(
    selected: Sequence[tuple[str, int, int, Tensor, Tensor]],
    protocol: DiagnosticProtocol,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for block_index, (block_name, invocation, invocation_count, x, y) in enumerate(selected):
        block_seed = protocol.probe_seed + block_index
        generator = torch.Generator(device=y.device)
        generator.manual_seed(block_seed)
        phi_samples: list[float] = []
        phi2_samples: list[float] = []
        status = "estimated"
        error_message = ""
        try:
            output_elements = y.numel()
            for _ in range(protocol.probes):
                probe = _rademacher_like(y, generator)
                jt_probe = torch.autograd.grad(
                    y, x, grad_outputs=probe, retain_graph=True, allow_unused=False
                )[0]
                phi_samples.append(float(jt_probe.detach().square().sum().item() / output_elements))

                dummy = torch.zeros_like(y, requires_grad=True)
                jt_dummy = torch.autograd.grad(
                    y,
                    x,
                    grad_outputs=dummy,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False,
                )[0]
                a_probe = torch.autograd.grad(
                    jt_dummy,
                    dummy,
                    grad_outputs=jt_probe.detach(),
                    retain_graph=True,
                    allow_unused=False,
                )[0]
                phi2_samples.append(float(a_probe.detach().square().sum().item() / output_elements))
            phi = float(np.mean(phi_samples))
            phi2 = float(np.mean(phi2_samples))
            varphi = phi2 - phi**2
        except (RuntimeError, NotImplementedError) as error:
            raise RuntimeError(
                f"Jacobian diagnostic failed for residual block {block_name!r}; "
                "partial-block summaries are forbidden"
            ) from error
        rows.append(
            {
                "block": block_name,
                "invocation_index": invocation,
                "invocation_count": invocation_count,
                "input_shape": "x".join(map(str, x.shape)),
                "output_shape": "x".join(map(str, y.shape)),
                "output_elements": y.numel(),
                "phi_jjt": phi,
                "phi_jjt_probe_sd": (
                    float(np.std(phi_samples, ddof=1)) if len(phi_samples) > 1 else 0.0
                ),
                "phi_jjt_squared": phi2,
                "phi_jjt_squared_probe_sd": (
                    float(np.std(phi2_samples, ddof=1)) if len(phi2_samples) > 1 else 0.0
                ),
                "varphi_jjt": varphi,
                "probes": protocol.probes,
                "probe_distribution": "Rademacher {-1,+1}",
                "probe_seed": block_seed,
                "method": JACOBIAN_METHOD,
                "normalization": "trace divided by block-output element count",
                "state_and_jacobian_boundary": JACOBIAN_BOUNDARY,
                "status": status,
                "error": error_message,
            }
        )
    return pd.DataFrame.from_records(rows)


def _json_value(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _layer_activity(model: nn.Module, diagnostics: Mapping[str, Any]) -> pd.DataFrame:
    layers = diagnostics.get("layers", {})
    if not isinstance(layers, Mapping):
        layers = {}
    modules = dict(model.named_modules())
    names = sorted(set(layers) | {name for name, module in modules.items() if hasattr(module, "membrane")})
    rows: list[dict[str, Any]] = []
    for name in names:
        values = layers.get(name, {})
        values = values if isinstance(values, Mapping) else {}
        module = modules.get(name)
        row: dict[str, Any] = {
            "layer": name,
            "spike_rate": values.get("spike_rate"),
            "spike_count": values.get("spike_count"),
            "elements": values.get("elements"),
            "surrogate_coverage": values.get("surrogate_coverage"),
            "surrogate_active": values.get("surrogate_active"),
            "surrogate_elements": values.get("surrogate_elements"),
            "c_pre_max": values.get("c_pre_max"),
            "window_v1_json": _json_value(values.get("window_v1")) if "window_v1" in values else "",
            "window_v2_json": _json_value(values.get("window_v2")) if "window_v2" in values else "",
            "window_status": "reported" if "window_v1" in values and "window_v2" in values else "missing_not_exposed_or_lif",
        }
        membrane = getattr(module, "last_membrane", None) if module is not None else None
        if isinstance(membrane, Tensor) and membrane.numel():
            flat_membrane = membrane.detach().float().reshape(-1).cpu()
            quantiles = torch.quantile(
                flat_membrane,
                torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]),
            )
            for label, value in zip(("q01", "q05", "q25", "q50", "q75", "q95", "q99"), quantiles):
                row[f"membrane_{label}"] = float(value.item())
            row["membrane_status"] = "measured_final_time_state"
        else:
            for label in ("q01", "q05", "q25", "q50", "q75", "q95", "q99"):
                row[f"membrane_{label}"] = np.nan
            row["membrane_status"] = "missing_model_did_not_expose_state"

        count = getattr(module, "count", None) if module is not None else None
        if isinstance(count, Tensor) and count.numel():
            flattened = count.detach().long().reshape(-1).cpu()
            occupancy = torch.bincount(flattened, minlength=int(flattened.max().item()) + 1)
            occupancy = occupancy.float() / occupancy.sum()
            row["count_occupancy_json"] = _json_value(occupancy.tolist())
            row["count_occupancy_status"] = "measured_final_time_state"
        else:
            row["count_occupancy_json"] = ""
            row["count_occupancy_status"] = "missing_model_did_not_expose_count_or_lif"
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def diagnose_model(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    *,
    protocol: DiagnosticProtocol | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> DiagnosticResult:
    """Run fixed-batch gradient, local Jacobian, and activity diagnostics."""

    protocol = protocol or DiagnosticProtocol()
    protocol.validate()
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same batch dimension")
    if inputs.shape[0] < 1:
        raise ValueError("representative batch cannot be empty")
    if not inputs.is_floating_point():
        inputs = inputs.float()
    inputs = inputs.detach().requires_grad_(True)
    targets = targets.detach().long()
    model.eval()

    captures, handles = _capture_block_invocations(model)
    try:
        logits, activity = _forward_activity(model, inputs)
    finally:
        for handle in handles:
            handle.remove()
    if logits.ndim != 2 or logits.shape[0] != targets.shape[0]:
        raise ValueError("model must return [batch, classes] logits")
    loss = F.cross_entropy(logits, targets, reduction="mean")
    selected = _select_invocation(captures, protocol.time_index)
    gradients, gradient_cv = _global_block_gradients(loss, selected)
    jacobians = _jacobian_moments(selected, protocol)
    layer_activity = _layer_activity(model, activity)
    measured_jacobians = jacobians.loc[jacobians["status"] == "estimated"]
    jacobian_complete = (
        len(measured_jacobians) == len(jacobians)
        and np.isfinite(measured_jacobians["phi_jjt"]).all()
        and np.isfinite(measured_jacobians["varphi_jjt"]).all()
    )
    jacobian_phi = (
        float(measured_jacobians["phi_jjt"].mean()) if jacobian_complete else float("nan")
    )
    jacobian_varphi = (
        float(measured_jacobians["varphi_jjt"].mean()) if jacobian_complete else float("nan")
    )
    summary = {
        "diagnostic_schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        **dict(metadata or {}),
        "batch_size": int(inputs.shape[0]),
        "input_shape": list(inputs.shape),
        "input_batch_sha256": representative_batch_sha256(inputs, targets),
        "objective": "mean cross-entropy on fixed representative batch",
        "loss": float(loss.detach().item()),
        "gradient_norm_definition": "RMS over every block-input tensor element",
        "gradient_cv_across_blocks": gradient_cv,
        "gradient_cv_method": "CV of residual-block input-gradient RMS on fixed batch",
        "jacobian_phi_mean_across_blocks": jacobian_phi,
        "jacobian_varphi_mean_across_blocks": jacobian_varphi,
        "diagnostic_status": "complete" if jacobian_complete else "incomplete_jacobian_blocks",
        "jacobian_blocks_expected": len(jacobians),
        "jacobian_blocks_measured": len(measured_jacobians),
        "block_count": len(selected),
        "protocol": asdict(protocol),
        "jacobian_method": JACOBIAN_METHOD,
        "state_and_jacobian_boundary": JACOBIAN_BOUNDARY,
        "phi_definition": "trace(JJ^T)/block-output-elements",
        "varphi_definition": "phi((JJ^T)^2)-phi(JJ^T)^2; finite-probe estimate may be negative",
        "activity_global": {key: value for key, value in activity.items() if key != "layers"},
        "missing_policy": "unexposed membrane/count/window diagnostics are explicitly marked missing",
    }
    identity_keys = (
        "diagnostic_id",
        "checkpoint_sha256",
        "checkpoint_config_hash",
        "input_file_sha256",
        "input_batch_sha256",
        "dataset",
        "condition",
        "depth",
        "time_steps",
        "seed",
        "selected_epoch",
    )
    identity = {key: summary.get(key, "") for key in identity_keys}
    for frame in (gradients, jacobians, layer_activity):
        for key, value in reversed(tuple(identity.items())):
            frame.insert(0, key, value)
    return DiagnosticResult(gradients, jacobians, layer_activity, summary)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        frame.to_csv(temporary_path, index=False, lineterminator="\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_diagnostic_outputs(
    result: DiagnosticResult,
    output_directory: str | Path,
    *,
    replace: bool = False,
) -> dict[str, Path]:
    """Atomically write CSV/JSON outputs with diagnostic-id idempotence guard."""

    output = Path(output_directory)
    summary_path = output / "diagnostic_summary.json"
    incoming_id = str(result.summary.get("diagnostic_id", ""))
    if not incoming_id:
        raise ValueError("summary must contain diagnostic_id")
    if summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("diagnostic_id") == incoming_id and not replace:
            existing_paths = {
                "gradients": output / "block_gradients.csv",
                "jacobians": output / "block_jacobian_moments.csv",
                "activity": output / "layer_activity.csv",
                "summary": summary_path,
            }
            missing = [str(path) for path in existing_paths.values() if not path.exists()]
            if missing:
                raise FileExistsError(
                    f"Matching diagnostic summary is incomplete ({missing}); rerun with replace=True"
                )
            return existing_paths
        if not replace:
            raise FileExistsError(
                "Output contains a different diagnostic run; pass replace=True/--replace explicitly"
            )
    paths = {
        "gradients": output / "block_gradients.csv",
        "jacobians": output / "block_jacobian_moments.csv",
        "activity": output / "layer_activity.csv",
        "summary": summary_path,
    }
    _atomic_csv(paths["gradients"], result.block_gradients)
    _atomic_csv(paths["jacobians"], result.block_jacobians)
    _atomic_csv(paths["activity"], result.layer_activity)
    output.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".diagnostic_summary.", suffix=".tmp", dir=output)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(result.summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, summary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return paths
