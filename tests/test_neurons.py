"""CPU-only mathematical tests for LIF and TA-LIF."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from talif_msresnet.neurons import LIFNeuron, TALIFNeuron, rectangular_spike  # noqa: E402


def test_rectangular_spike_is_binary_and_deterministic() -> None:
    u = torch.tensor([-0.1, 0.49, 0.5, 2.0])
    output = rectangular_spike(u, torch.tensor(0.0), torch.tensor(1.0))
    assert torch.equal(output, torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert set(output.tolist()) <= {0.0, 1.0}


def test_rectangular_surrogate_matches_paper_derivatives() -> None:
    u = torch.tensor([0.25, 0.75], requires_grad=True)
    v1 = torch.tensor(0.0, requires_grad=True)
    v2 = torch.tensor(1.0, requires_grad=True)
    rectangular_spike(u, v1, v2).sum().backward()

    # dS/dU=1; summed dS/dV1=(-.75-.25) and
    # dS/dV2=(-.25-.75) for this unit-width interval.
    assert torch.allclose(u.grad, torch.ones_like(u))
    assert torch.allclose(v1.grad, torch.tensor(-1.0))
    assert torch.allclose(v2.grad, torch.tensor(-1.0))


def test_surrogate_has_zero_gradient_outside_window() -> None:
    u = torch.tensor([-1.0, 2.0], requires_grad=True)
    rectangular_spike(u, torch.tensor(0.0), torch.tensor(1.0)).sum().backward()
    assert torch.equal(u.grad, torch.zeros_like(u))


def test_lif_baseline_has_plain_decay_derivative_and_detached_reset() -> None:
    quiet_input = torch.tensor([0.75], requires_grad=True)
    quiet = LIFNeuron(tau=0.5, threshold=1.0, surrogate_width=1.0)
    quiet(quiet_input)
    assert quiet.membrane is not None
    quiet_grad = torch.autograd.grad(quiet.membrane.sum(), quiet_input)[0]
    assert torch.allclose(quiet_grad, torch.tensor([0.5]))

    firing_input = torch.tensor([1.25], requires_grad=True)
    firing = LIFNeuron(tau=0.5, threshold=1.0, surrogate_width=1.0)
    firing(firing_input)
    assert firing.membrane is not None
    reset_grad = torch.autograd.grad(firing.membrane.sum(), firing_input)[0]
    assert torch.equal(reset_grad, torch.zeros_like(firing_input))


def test_talif_retains_nonfiring_surrogate_membrane_path() -> None:
    value = torch.tensor([0.75], requires_grad=True)
    neuron = TALIFNeuron(steps=2, tau=0.5, threshold=1.0, width=1.0, delta_min=0.05)
    neuron(value)
    assert neuron.membrane is not None
    gradient = torch.autograd.grad(neuron.membrane.sum(), value)[0]
    # Eq. (2.21): tau - tau * (U-v_rest) * dS/dU.
    assert torch.allclose(gradient, torch.tensor([0.125]))


def test_talif_width_is_positive_and_boundary_parameters_receive_gradients() -> None:
    neuron = TALIFNeuron(steps=3, threshold=1.0, width=1.0, delta_min=0.05)
    assert neuron.center.shape == (3,)
    assert neuron.raw_width.shape == (3,)
    assert torch.all(neuron.width > 0.05)
    assert torch.all(neuron.v2 > neuron.v1)

    x = torch.tensor([0.75], requires_grad=True)
    neuron(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert neuron.center.grad is not None and neuron.center.grad[0].abs() > 0
    assert neuron.raw_width.grad is not None and neuron.raw_width.grad[0].abs() > 0
    # Unselected history entries must not receive a gradient on the first step.
    assert torch.equal(neuron.center.grad[1:], torch.zeros_like(neuron.center.grad[1:]))


def test_talif_uses_per_neuron_c_pre_before_current_decision() -> None:
    neuron = TALIFNeuron(steps=3, threshold=0.5, width=0.2, delta_min=0.01)
    with torch.no_grad():
        neuron.center.copy_(torch.tensor([0.5, 1.5, 2.5]))

    first = neuron(torch.tensor([1.0, 0.0]))
    assert torch.equal(first, torch.tensor([1.0, 0.0]))
    assert torch.equal(neuron.count, torch.tensor([1, 0]))

    second = neuron(torch.tensor([1.0, 1.0]))
    # The first element uses bank 1 (threshold 1.5) while the second still
    # uses bank 0 (threshold 0.5).  A layer-wide count would fail this test.
    assert torch.equal(neuron.last_c_pre, torch.tensor([1, 0]))
    assert torch.equal(second, torch.tensor([0.0, 1.0]))
    assert torch.equal(neuron.count, torch.tensor([1, 1]))


def test_reset_state_clears_membrane_and_history() -> None:
    neuron = TALIFNeuron(steps=2)
    neuron(torch.ones(2, 3))
    assert neuron.membrane is not None
    assert neuron.count is not None
    neuron.reset_state()
    assert neuron.membrane is None
    assert neuron.count is None
    assert neuron.last_c_pre is None
