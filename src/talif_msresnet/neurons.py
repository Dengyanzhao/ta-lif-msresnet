"""Stateful LIF and spike-history-driven TA-LIF neurons.

The implementation deliberately keeps the hard, binary spike in the forward
pass.  :class:`RectangularSpike` only changes the backward pass and therefore
does not turn a residual branch into a continuous activation.  The stateful
modules are small enough to be used both by the model in ``models.py`` and by
standalone experiment/smoke-test code.

The centre/width parameterisation and the minimum width safeguard are an
implementation choice for the joint experiment.  The dissertation specifies
the rectangular derivatives and the history-indexed threshold bank, but does
not prescribe this numerical parameterisation.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

__all__ = [
    "RectangularSpike",
    "RectangularSpikeFn",
    "rectangular_spike",
    "surrogate_spike",
    "BaseNeuron",
    "LIFNeuron",
    "TALIFNeuron",
    "LIF",
    "TALIF",
]


class RectangularSpike(torch.autograd.Function):
    """Binary threshold with the TA-LIF rectangular surrogate derivative.

    For an interval ``[v1, v2]`` the forward value is
    ``1[u > (v1 + v2) / 2]``.  In the backward pass the derivatives are the
    dissertation equations (5)--(7):

    ``dS/dU = 1[interval] / (v2-v1)``
    ``dS/dv1 = 1[interval] * (U-v2)/(v2-v1)^2``
    ``dS/dv2 = 1[interval] * (v1-U)/(v2-v1)^2``.

    The function accepts broadcastable tensors.  It is public so tests and
    analysis scripts can verify the derivative without constructing a model.
    """

    @staticmethod
    def forward(ctx, u: Tensor, v1: Tensor, v2: Tensor) -> Tensor:  # type: ignore[override]
        # Inputs are broadcast before this function is called in the helper,
        # but broadcasting here too makes direct ``.apply`` use well behaved.
        ctx.input_shapes = (u.shape, v1.shape, v2.shape)
        u, v1, v2 = torch.broadcast_tensors(u, v1, v2)
        width = v2 - v1
        # A non-positive width is a configuration error.  Clamping here keeps
        # a malformed experimental config from producing NaNs in a smoke test;
        # TA-LIF itself guarantees positivity through softplus + delta_min.
        eps = torch.finfo(u.dtype).eps
        safe_width = width.clamp_min(eps)
        interval = (u >= v1) & (u <= v2)
        midpoint = 0.5 * (v1 + v2)
        spike = (u > midpoint).to(dtype=u.dtype)
        ctx.save_for_backward(u, v1, v2, interval.to(dtype=u.dtype), safe_width)
        return spike

    @staticmethod
    def backward(ctx, grad_output: Optional[Tensor]):  # type: ignore[override]
        if grad_output is None:
            return None, None, None
        u, v1, v2, interval, width = ctx.saved_tensors
        inv_width = interval / width
        grad_u = grad_output * inv_width
        denom = width.square()
        grad_v1 = grad_output * interval * (u - v2) / denom
        grad_v2 = grad_output * interval * (v1 - u) / denom
        # The helper passes expanded views, so these usually are already the
        # right shape.  Reducing here also supports direct scalar ``.apply``.
        shape_u, shape_v1, shape_v2 = ctx.input_shapes
        return (
            grad_u.sum_to_size(shape_u),
            grad_v1.sum_to_size(shape_v1),
            grad_v2.sum_to_size(shape_v2),
        )


# A conventional alias is useful for code that refers to a Function class by
# its ``Fn`` suffix.
RectangularSpikeFn = RectangularSpike


def rectangular_spike(u: Tensor, v1: Tensor, v2: Tensor) -> Tensor:
    """Apply the deterministic binary spike and rectangular surrogate."""

    if not (u.is_floating_point() and v1.is_floating_point() and v2.is_floating_point()):
        raise TypeError("rectangular_spike expects floating point tensors")
    v1 = v1.to(device=u.device, dtype=u.dtype)
    v2 = v2.to(device=u.device, dtype=u.dtype)
    u, v1, v2 = torch.broadcast_tensors(u, v1, v2)
    return RectangularSpike.apply(u, v1, v2)


surrogate_spike = rectangular_spike


class _HistoryWindowSelect(torch.autograd.Function):
    """Select two history banks with a deterministic, bounded-cost backward.

    ``bank[index]`` is inexpensive in the forward pass, but its CUDA backward
    aggregates many repeated indices with an indexed write.  Under strict
    deterministic algorithms that path can be prohibitively slow for dense
    feature maps and tiny TA-LIF banks.  The derivative with respect to bank
    entry ``k`` is simply the sum of output gradients whose index is ``k``.
    Reducing one masked gradient tensor per bank slot avoids indexed writes and
    the large ``N x bank_size`` indicator matrix while preserving the exact
    mathematical operation and a deterministic reduction path.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        bank_v1: Tensor,
        bank_v2: Tensor,
        index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        ctx.bank_size = bank_v1.shape[0]
        ctx.save_for_backward(index)
        return bank_v1[index], bank_v2[index]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx,
        grad_v1: Optional[Tensor],
        grad_v2: Optional[Tensor],
    ) -> Tuple[Optional[Tensor], Optional[Tensor], None]:
        (index,) = ctx.saved_tensors
        flat_index = index.reshape(-1)

        def reduce_gradient(gradient: Optional[Tensor]) -> Optional[Tensor]:
            if gradient is None:
                return None
            flat_gradient = gradient.reshape(-1)
            zero = torch.zeros((), device=gradient.device, dtype=gradient.dtype)
            return torch.stack(
                [
                    torch.where(flat_index == slot, flat_gradient, zero).sum()
                    for slot in range(ctx.bank_size)
                ]
            )

        return reduce_gradient(grad_v1), reduce_gradient(grad_v2), None


def _select_history_windows(
    bank_v1: Tensor,
    bank_v2: Tensor,
    index: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _HistoryWindowSelect.apply(bank_v1, bank_v2, index)


def _inverse_softplus(value: Tensor) -> Tensor:
    """Stable inverse used only to initialise TA-LIF widths."""

    # log(expm1(x)) loses precision for large x and is undefined at x=0.
    return value + torch.log(-torch.expm1(-value))


class BaseNeuron(nn.Module):
    """Common state and diagnostics handling for one spiking layer."""

    def __init__(
        self,
        tau: float = 0.5,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 < float(tau) <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        self.tau = float(tau)
        self.v_rest = float(v_rest)
        self.v_reset = float(v_reset)
        # Dynamic states intentionally are not buffers.  They belong to the
        # current sample and should never be serialized in a checkpoint.
        self._mem: Optional[Tensor] = None
        self._count: Optional[Tensor] = None
        self._spike_sum: Optional[Tensor] = None
        self._element_count = 0
        self._surrogate_active_sum: Optional[Tensor] = None
        self._surrogate_element_count = 0
        # Standalone neurons collect by default.  The full model explicitly
        # disables this flag for normal training to avoid one reduction per
        # layer and time step.
        self._collect_activity = True
        self.last_c_pre: Optional[Tensor] = None
        self.last_spike: Optional[Tensor] = None
        self.last_membrane: Optional[Tensor] = None

    @property
    def membrane(self) -> Optional[Tensor]:
        """Current membrane state (kept attached for temporal BPTT)."""

        return self._mem

    @property
    def count(self) -> Optional[Tensor]:
        """Current cumulative count, after the most recent decision."""

        return self._count

    def reset_state(self, clear_stats: bool = True) -> None:
        """Reset temporal state at an input boundary.

        ``clear_stats=False`` is useful when a caller wants to aggregate
        activity over several chunks of one logical sample.
        """

        self._mem = None
        self._count = None
        self.last_c_pre = None
        self.last_spike = None
        self.last_membrane = None
        if clear_stats:
            self._spike_sum = None
            self._element_count = 0
            self._surrogate_active_sum = None
            self._surrogate_element_count = 0

    # Common spelling used by several SNN toolkits.
    reset = reset_state

    def _prepare_membrane(self, x: Tensor) -> Tensor:
        if (
            self._mem is None
            or self._mem.shape != x.shape
            or self._mem.device != x.device
            or self._mem.dtype != x.dtype
        ):
            self._mem = torch.full_like(x, self.v_rest)
        return self._mem

    def set_collect_activity(self, enabled: bool) -> None:
        self._collect_activity = bool(enabled)

    def _record(
        self,
        spike: Tensor,
        membrane: Tensor,
        surrogate_active: Optional[Tensor] = None,
    ) -> None:
        if not self._collect_activity:
            self.last_spike = None
            self.last_membrane = None
            return
        detached = spike.detach()
        step_sum = detached.sum()
        self._spike_sum = step_sum if self._spike_sum is None else self._spike_sum + step_sum
        self._element_count += detached.numel()
        if surrogate_active is not None:
            active = surrogate_active.detach().sum()
            self._surrogate_active_sum = (
                active
                if self._surrogate_active_sum is None
                else self._surrogate_active_sum + active
            )
            self._surrogate_element_count += surrogate_active.numel()
        self.last_spike = detached
        self.last_membrane = membrane.detach()

    def diagnostics(self) -> Dict[str, object]:
        spike_sum = float(self._spike_sum.item()) if self._spike_sum is not None else 0.0
        rate = spike_sum / self._element_count if self._element_count else 0.0
        surrogate_active = (
            float(self._surrogate_active_sum.item())
            if self._surrogate_active_sum is not None
            else 0.0
        )
        surrogate_coverage = (
            surrogate_active / self._surrogate_element_count
            if self._surrogate_element_count
            else 0.0
        )
        out: Dict[str, object] = {
            "spike_rate": rate,
            "spike_count": spike_sum,
            "elements": self._element_count,
            "surrogate_coverage": surrogate_coverage,
            "surrogate_active": surrogate_active,
            "surrogate_elements": self._surrogate_element_count,
        }
        if self.last_c_pre is not None:
            out["c_pre_max"] = int(self.last_c_pre.max().item())
        return out

    def extra_repr(self) -> str:
        return f"tau={self.tau}, v_rest={self.v_rest}, v_reset={self.v_reset}"


class LIFNeuron(BaseNeuron):
    """LIF baseline with a fixed rectangular surrogate window.

    The reset indicator is detached in both branches. This is the source
    ablation's original-LIF baseline: reset gradients are isolated, while the
    additional non-spiking membrane path in dissertation Eq. (2.21) remains a
    TA-LIF intervention. The threshold and window are fixed.
    """

    def __init__(
        self,
        tau: float = 0.5,
        threshold: float = 1.0,
        surrogate_width: float = 1.0,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
    ) -> None:
        super().__init__(tau=tau, v_rest=v_rest, v_reset=v_reset)
        if surrogate_width <= 0:
            raise ValueError("surrogate_width must be positive")
        self.threshold = float(threshold)
        self.surrogate_width = float(surrogate_width)

    def _window(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        width = torch.as_tensor(self.surrogate_width, device=x.device, dtype=x.dtype)
        threshold = torch.as_tensor(self.threshold, device=x.device, dtype=x.dtype)
        return threshold - 0.5 * width, threshold + 0.5 * width

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if not x.is_floating_point():
            x = x.float()
        previous = self._prepare_membrane(x)
        u = previous + x
        v1, v2 = self._window(u)
        spike = rectangular_spike(u, v1, v2)
        # ``where`` selects the same branch as the hard forward spike.  The
        # condition and reset indicator are detached, so a spike contributes
        # no reset derivative; the quiet branch retains dS/dU.
        fired = spike.detach().bool()
        fired_branch = self.tau * u * (1.0 - spike.detach()) + self.v_reset
        quiet_branch = self.v_rest + self.tau * (u - self.v_rest) * (1.0 - spike.detach())
        membrane = torch.where(fired, fired_branch, quiet_branch)
        self._mem = membrane
        self.last_c_pre = None
        surrogate_active = (u >= v1) & (u <= v2) if self._collect_activity else None
        self._record(spike, membrane, surrogate_active)
        return spike


class TALIFNeuron(BaseNeuron):
    """Spike-history-driven threshold-adaptive LIF neuron.

    ``center`` and ``raw_width`` each have ``steps`` entries.  A bank entry is
    selected by the per-element cumulative spike count *before* the current
    decision.  Counts are integer state, clipped at the final bank entry;
    this makes direct use with event sequences longer than the configured
    horizon well defined while preserving the intended first ``T`` steps.
    """

    def __init__(
        self,
        steps: int = 6,
        T: Optional[int] = None,
        time_steps: Optional[int] = None,
        tau: float = 0.5,
        threshold: float = 1.0,
        width: float = 1.0,
        delta_min: float = 1e-3,
        v_rest: float = 0.0,
        v_reset: float = 0.0,
    ) -> None:
        if time_steps is not None:
            steps = time_steps
        if T is not None:
            steps = T
        if int(steps) < 1:
            raise ValueError("steps/T must be >= 1")
        if float(delta_min) <= 0:
            raise ValueError("delta_min must be positive")
        if float(width) <= float(delta_min):
            raise ValueError("initial width must exceed delta_min")
        super().__init__(tau=tau, v_rest=v_rest, v_reset=v_reset)
        self.steps = int(steps)
        self.T = self.steps  # explicit alias used in experiment configs
        self.delta_min = float(delta_min)
        self.center = nn.Parameter(torch.full((self.steps,), float(threshold)))
        initial = torch.full((self.steps,), float(width) - self.delta_min)
        self.raw_width = nn.Parameter(_inverse_softplus(initial))

    @property
    def width(self) -> Tensor:
        return self.delta_min + F.softplus(self.raw_width)

    @property
    def v1(self) -> Tensor:
        return self.center - 0.5 * self.width

    @property
    def v2(self) -> Tensor:
        return self.center + 0.5 * self.width

    @property
    def threshold_bank(self) -> Tensor:
        return self.center

    def threshold_windows(self) -> Tuple[Tensor, Tensor]:
        """Return the current bank as ``(V1, V2)``."""

        return self.v1, self.v2

    def _selected_windows(self, count: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
        # Count is per neuron (same shape as U), unlike a layer-wide scalar.
        index = count.clamp(min=0, max=self.steps - 1).long()
        bank_v1 = self.v1.to(device=x.device, dtype=x.dtype)
        bank_v2 = self.v2.to(device=x.device, dtype=x.dtype)
        return _select_history_windows(bank_v1, bank_v2, index)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if not x.is_floating_point():
            x = x.float()
        previous = self._prepare_membrane(x)
        if (
            self._count is None
            or self._count.shape != x.shape
            or self._count.device != x.device
        ):
            self._count = torch.zeros(x.shape, device=x.device, dtype=torch.long)
        c_pre = self._count
        u = previous + x
        v1, v2 = self._selected_windows(c_pre, u)
        spike = rectangular_spike(u, v1, v2)
        fired = spike.detach().bool()
        fired_branch = self.tau * u * (1.0 - spike.detach()) + self.v_reset
        quiet_branch = self.v_rest + self.tau * (u - self.v_rest) * (1.0 - spike)
        membrane = torch.where(fired, fired_branch, quiet_branch)
        self._mem = membrane
        self.last_c_pre = c_pre.detach().clone()
        # History is a routing/index state, never a differentiable quantity.
        self._count = (c_pre + spike.detach().to(dtype=torch.long)).clamp(max=self.steps - 1)
        surrogate_active = (u >= v1) & (u <= v2) if self._collect_activity else None
        self._record(spike, membrane, surrogate_active)
        return spike

    def extra_repr(self) -> str:
        return f"steps={self.steps}, delta_min={self.delta_min}, {super().extra_repr()}"


# Short aliases keep configuration-driven code concise and preserve common
# naming conventions used by SNN repositories.
LIF = LIFNeuron
TALIF = TALIFNeuron
