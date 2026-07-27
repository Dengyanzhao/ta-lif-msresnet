"""CPU tests for the non-reporting TA history reduction benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import benchmark_ta_history_reduction as benchmark  # noqa: E402


def test_masked_where_selector_matches_direct_indexing() -> None:
    index = torch.tensor([[0, 2, 2], [5, 1, 0]], dtype=torch.long)
    weight_v1 = torch.tensor([[0.5, -0.25, 0.75], [1.5, -2.0, 0.125]])
    weight_v2 = weight_v1.flip(1)

    direct_banks = (
        torch.linspace(-1.0, 1.5, 6, requires_grad=True),
        torch.linspace(0.0, 2.5, 6, requires_grad=True),
    )
    masked_banks = tuple(bank.detach().clone().requires_grad_() for bank in direct_banks)

    direct = (direct_banks[0][index], direct_banks[1][index])
    masked = benchmark.masked_where_select(*masked_banks, index)
    direct_loss = (direct[0] * weight_v1).sum() + (direct[1] * weight_v2).sum()
    masked_loss = (masked[0] * weight_v1).sum() + (masked[1] * weight_v2).sum()
    direct_gradients = torch.autograd.grad(direct_loss, direct_banks)
    masked_gradients = torch.autograd.grad(masked_loss, masked_banks)

    for actual, expected in zip(masked, direct, strict=True):
        assert torch.equal(actual, expected)
    for actual, expected in zip(masked_gradients, direct_gradients, strict=True):
        assert torch.equal(actual, expected)


def test_masked_where_selector_is_repeatable_under_strict_determinism() -> None:
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        reports = [
            benchmark.probe_selector(benchmark.masked_where_select, torch.device("cpu"))
            for _ in range(3)
        ]
        assert all(report["pass"] for report in reports)
        assert reports[0] == reports[1] == reports[2]
    finally:
        torch.use_deterministic_algorithms(was_deterministic)
