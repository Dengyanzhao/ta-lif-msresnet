"""Equivalence tests for deterministic TA-LIF history-bank selection."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from talif_msresnet import neurons  # noqa: E402


def _loss_and_gradients(
    bank_v1: "torch.Tensor",
    bank_v2: "torch.Tensor",
    index: "torch.Tensor",
    weight_v1: "torch.Tensor",
    weight_v2: "torch.Tensor",
    *,
    optimized: bool,
) -> tuple[tuple["torch.Tensor", "torch.Tensor"], tuple["torch.Tensor", "torch.Tensor"]]:
    if optimized:
        selected_v1, selected_v2 = neurons._select_history_windows(bank_v1, bank_v2, index)
    else:
        selected_v1, selected_v2 = bank_v1[index], bank_v2[index]
    loss = (selected_v1 * weight_v1).sum() + (selected_v2 * weight_v2).sum()
    gradients = torch.autograd.grad(loss, (bank_v1, bank_v2))
    return (selected_v1, selected_v2), gradients


def test_history_window_selection_matches_index_forward_and_gradients() -> None:
    index = torch.tensor(
        [
            [[0, 1, 1, 3], [2, 0, 3, 3]],
            [[3, 2, 0, 1], [1, 3, 2, 0]],
        ],
        dtype=torch.long,
    )
    weight_v1 = torch.linspace(-1.5, 2.0, index.numel(), dtype=torch.float64).reshape(index.shape)
    weight_v2 = torch.linspace(0.75, -2.25, index.numel(), dtype=torch.float64).reshape(index.shape)

    legacy_banks = (
        torch.tensor([-1.0, -0.25, 0.5, 1.25], dtype=torch.float64, requires_grad=True),
        torch.tensor([0.0, 0.75, 1.5, 2.25], dtype=torch.float64, requires_grad=True),
    )
    optimized_banks = tuple(bank.detach().clone().requires_grad_() for bank in legacy_banks)

    legacy_output, legacy_gradients = _loss_and_gradients(
        *legacy_banks, index, weight_v1, weight_v2, optimized=False
    )
    optimized_output, optimized_gradients = _loss_and_gradients(
        *optimized_banks, index, weight_v1, weight_v2, optimized=True
    )

    for actual, expected in zip(optimized_output, legacy_output, strict=True):
        assert torch.equal(actual, expected)
        assert actual.shape == index.shape
    for actual, expected in zip(optimized_gradients, legacy_gradients, strict=True):
        assert torch.allclose(actual, expected, rtol=1e-14, atol=1e-14)
        assert actual.shape == (4,)


def test_history_window_selection_leaves_unused_bank_entries_at_zero() -> None:
    bank_v1 = torch.arange(5.0, requires_grad=True)
    bank_v2 = torch.arange(5.0, 10.0, requires_grad=True)
    index = torch.tensor([[0, 2, 2], [0, 2, 0]])

    selected_v1, selected_v2 = neurons._select_history_windows(bank_v1, bank_v2, index)
    (selected_v1.sum() + 2.0 * selected_v2.sum()).backward()

    assert torch.equal(bank_v1.grad, torch.tensor([3.0, 0.0, 3.0, 0.0, 0.0]))
    assert torch.equal(bank_v2.grad, torch.tensor([6.0, 0.0, 6.0, 0.0, 0.0]))


def test_history_window_selection_is_repeatable_with_strict_determinism() -> None:
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        index = torch.tensor([[3, 0, 1, 3], [2, 3, 1, 0]])
        weight_v1 = torch.tensor([[0.2, -0.1, 0.3, 0.7], [-0.4, 0.9, 1.1, -0.8]])
        weight_v2 = weight_v1.flip(1)
        results = []
        for _ in range(3):
            banks = (
                torch.tensor([-0.5, 0.0, 0.5, 1.0], requires_grad=True),
                torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True),
            )
            results.append(_loss_and_gradients(*banks, index, weight_v1, weight_v2, optimized=True))

        reference_output, reference_gradients = results[0]
        for output, gradients in results[1:]:
            for actual, expected in zip(output, reference_output, strict=True):
                assert torch.equal(actual, expected)
            for actual, expected in zip(gradients, reference_gradients, strict=True):
                assert torch.equal(actual, expected)
    finally:
        torch.use_deterministic_algorithms(was_deterministic)


def test_history_window_selection_passes_double_precision_gradcheck() -> None:
    index = torch.tensor([[0, 2, 1], [2, 2, 0]], dtype=torch.long)
    bank_v1 = torch.tensor([-0.75, 0.1, 0.9], dtype=torch.float64, requires_grad=True)
    bank_v2 = torch.tensor([0.25, 1.1, 1.9], dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        lambda first, second: neurons._select_history_windows(first, second, index),
        (bank_v1, bank_v2),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
    )
