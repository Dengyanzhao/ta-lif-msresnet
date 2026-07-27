#!/usr/bin/env python3
"""Compare deterministic TA history-gradient reductions on the target GPU.

This is a non-reporting engineering diagnostic. It monkeypatches the history
selector in memory, uses one fixed real CIFAR-100 batch, prints one JSON report
to stdout, and does not write checkpoints, protocol files, or result artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPOSITORY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import pilot_health_gate as gate  # noqa: E402
from talif_msresnet import neurons  # noqa: E402
from talif_msresnet.config import load_protocol, load_run_config  # noqa: E402
from talif_msresnet.utils import environment_manifest, stable_hash, utc_now  # noqa: E402


ARTIFACT_CLASS = "NON_REPORTING_TA_HISTORY_REDUCTION_BENCHMARK"
Selector = Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]


class _MaskedWhereHistoryWindowSelect(torch.autograd.Function):
    """Use fixed-order per-slot reductions instead of an N-by-bank indicator."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        bank_v1: Tensor,
        bank_v2: Tensor,
        index: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        ctx.bank_size = int(bank_v1.shape[0])
        ctx.save_for_backward(index)
        return bank_v1[index], bank_v2[index]

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
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


def masked_where_select(
    bank_v1: Tensor,
    bank_v2: Tensor,
    index: Tensor,
) -> Tuple[Tensor, Tensor]:
    return _MaskedWhereHistoryWindowSelect.apply(bank_v1, bank_v2, index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs"
        / "v2_pilot_generated"
        / "E1_cifar100_d20_t6_C1_s77.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "protocol_v2_pilot.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser


def _cpu_reference() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    index = torch.tensor(
        [[0, 1, 1, 3], [2, 0, 3, 3], [5, 4, 2, 0]],
        dtype=torch.long,
    )
    weight_v1 = torch.linspace(-1.5, 2.0, index.numel(), dtype=torch.float64).reshape(index.shape)
    weight_v2 = torch.linspace(0.75, -2.25, index.numel(), dtype=torch.float64).reshape(
        index.shape
    )
    bank_v1 = torch.linspace(-1.0, 1.5, 6, dtype=torch.float64, requires_grad=True)
    bank_v2 = torch.linspace(0.0, 2.5, 6, dtype=torch.float64, requires_grad=True)
    selected_v1, selected_v2 = bank_v1[index], bank_v2[index]
    loss = (selected_v1 * weight_v1).sum() + (selected_v2 * weight_v2).sum()
    gradients = torch.autograd.grad(loss, (bank_v1, bank_v2))
    return selected_v1, selected_v2, gradients[0], gradients[1], index


def probe_selector(selector: Selector, device: torch.device) -> dict[str, Any]:
    reference_v1, reference_v2, reference_grad_v1, reference_grad_v2, cpu_index = (
        _cpu_reference()
    )
    weight_v1 = torch.linspace(
        -1.5, 2.0, cpu_index.numel(), dtype=torch.float64, device=device
    ).reshape(cpu_index.shape)
    weight_v2 = torch.linspace(
        0.75, -2.25, cpu_index.numel(), dtype=torch.float64, device=device
    ).reshape(cpu_index.shape)
    index = cpu_index.to(device)
    repetitions: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
    for _ in range(3):
        bank_v1 = torch.linspace(
            -1.0, 1.5, 6, dtype=torch.float64, device=device, requires_grad=True
        )
        bank_v2 = torch.linspace(
            0.0, 2.5, 6, dtype=torch.float64, device=device, requires_grad=True
        )
        selected_v1, selected_v2 = selector(bank_v1, bank_v2, index)
        loss = (selected_v1 * weight_v1).sum() + (selected_v2 * weight_v2).sum()
        grad_v1, grad_v2 = torch.autograd.grad(loss, (bank_v1, bank_v2))
        repetitions.append(
            tuple(
                value.detach().cpu()
                for value in (selected_v1, selected_v2, grad_v1, grad_v2)
            )
        )

    expected = (reference_v1, reference_v2, reference_grad_v1, reference_grad_v2)
    first = repetitions[0]
    forward_exact = torch.equal(first[0], expected[0]) and torch.equal(first[1], expected[1])
    gradient_close = torch.allclose(first[2], expected[2], rtol=1e-12, atol=1e-12) and (
        torch.allclose(first[3], expected[3], rtol=1e-12, atol=1e-12)
    )
    repeatable = all(
        all(torch.equal(actual, reference) for actual, reference in zip(row, first, strict=True))
        for row in repetitions[1:]
    )
    return {
        "pass": bool(forward_exact and gradient_close and repeatable),
        "forward_exact": forward_exact,
        "gradient_close_to_cpu_float64_reference": gradient_close,
        "three_run_bitwise_repeatable": repeatable,
        "maximum_absolute_gradient_error": max(
            float((first[2] - expected[2]).abs().max().item()),
            float((first[3] - expected[3]).abs().max().item()),
        ),
    }


def _timing_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    conditions = report["conditions"]
    ratios = report["ta_enabled_over_frozen_ratios"]
    result: dict[str, Any] = {
        "pass_under_protocol_limit": bool(report["pass"]),
        "lif_cuda_step_median_ms": {
            condition: conditions[condition]["lif_standard"]["cuda_step_median_ms"]
            for condition in ("C1", "C3")
        },
        "ta_conditions": {},
    }
    for condition in ("C2", "C4"):
        item = conditions[condition]
        result["ta_conditions"][condition] = {
            "pass_under_protocol_limit": ratios[condition]["pass"],
            "conservative_enabled_over_frozen_ratio": ratios[condition][
                "ta_enabled_over_frozen"
            ],
            "combined_enabled_over_frozen_ratio": ratios[condition][
                "combined_ta_enabled_over_frozen"
            ],
            "round_ratios": ratios[condition]["round_ratios"],
            "maximum_allowed": ratios[condition]["maximum_allowed"],
            "aggregate": item["aggregate"],
            "rounds": [
                {
                    "round": row["round"],
                    "order": row["order"],
                    "frozen_cuda_step_median_ms": row["ta_frozen"][
                        "cuda_step_median_ms"
                    ],
                    "enabled_cuda_step_median_ms": row["ta_enabled"][
                        "cuda_step_median_ms"
                    ],
                    "ratio": row["ta_enabled_over_frozen"],
                }
                for row in item["rounds"]
            ],
        }
    return result


def benchmark_selector(
    *,
    selector: Selector,
    base_config: Any,
    device: torch.device,
    batch: tuple[Tensor, Tensor],
    output_parent: Path,
    seed: int,
    warmup: int,
    iterations: int,
    maximum_ratio: float,
) -> dict[str, Any]:
    original = neurons._select_history_windows
    try:
        neurons._select_history_windows = selector
        correctness = probe_selector(selector, device)
        if not correctness["pass"]:
            return {"status": "CORRECTNESS_FAIL", "correctness": correctness}
        gc.collect()
        torch.cuda.empty_cache()
        timing = gate.run_cuda_timing_gate(
            base_config,
            seed=seed,
            device=device,
            output_parent=output_parent,
            batch=batch,
            warmup=warmup,
            iterations=iterations,
            maximum_ratio=maximum_ratio,
        )
        return {
            "status": "PASS" if timing["pass"] else "TIMING_LIMIT_FAIL",
            "correctness": correctness,
            "timing": _timing_summary(timing),
        }
    finally:
        neurons._select_history_windows = original
        gc.collect()
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be >= 0 and --iterations must be >= 1")

    protocol = load_protocol(args.protocol)
    acceptance = protocol["pilot_acceptance"]
    timing_acceptance = acceptance["timing"]
    environment_binding = acceptance["environment"]
    base_config = load_run_config(args.config, args.protocol)
    seed = int(acceptance["seed"])
    device = torch.device(args.device)

    gate.seed_everything(seed, deterministic=True)
    hardware = gate.require_target_cuda(device, str(environment_binding["expected_gpu_substring"]))
    gate.validate_runtime_environment(environment_binding, hardware)
    git_identity = gate.repository_git_identity(REPOSITORY_ROOT)
    if not git_identity["tracked_clean"]:
        raise SystemExit("Tracked worktree must be clean for the engineering benchmark")

    timing_batch_size = int(timing_acceptance["batch_size"])
    base_config, inputs, targets, split_manifest = gate.load_fixed_real_batch(
        base_config,
        seed=seed,
        required_batch_size=timing_batch_size,
    )
    batch = (
        inputs[:timing_batch_size].to(device, non_blocking=False),
        targets[:timing_batch_size].to(device, non_blocking=False),
    )
    idle = gate.gpu_idle_precheck(device)

    candidates: dict[str, Any] = {}
    selectors: tuple[tuple[str, Selector], ...] = (
        ("repository_indicator_matmul", neurons._select_history_windows),
        ("fixed_order_masked_where_sum", masked_where_select),
    )
    for name, selector in selectors:
        try:
            candidates[name] = benchmark_selector(
                selector=selector,
                base_config=base_config,
                device=device,
                batch=batch,
                output_parent=REPOSITORY_ROOT / "results" / "pilot",
                seed=seed,
                warmup=args.warmup,
                iterations=args.iterations,
                maximum_ratio=float(
                    timing_acceptance["maximum_ta_enabled_over_frozen_ratio"]
                ),
            )
        except Exception as exc:
            candidates[name] = {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }

    report = {
        "schema_version": 1,
        "artifact_class": ARTIFACT_CLASS,
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "purpose": "select a mathematically equivalent deterministic TA reduction",
        "created_at": utc_now(),
        "git_commit": git_identity["git_commit"],
        "tracked_clean": git_identity["tracked_clean"],
        "protocol_hash": stable_hash(protocol),
        "acceptance_hash": stable_hash(dict(acceptance)),
        "source": {
            "config": str(args.config.resolve()),
            "protocol": str(args.protocol.resolve()),
            "dataset": base_config.data.dataset,
            "seed": seed,
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "fixed_batch_sha256": gate.tensor_batch_sha256(inputs, targets),
            "fixed_batch_shape": list(inputs.shape),
            "batch_size": timing_batch_size,
            "warmup_steps": args.warmup,
            "timed_steps": args.iterations,
        },
        "environment": {
            **environment_manifest(),
            "hardware": hardware,
            "gpu_idle_precheck": idle,
        },
        "protocol_maximum_ratio": timing_acceptance[
            "maximum_ta_enabled_over_frozen_ratio"
        ],
        "candidates": candidates,
    }
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))

    current = candidates.get("repository_indicator_matmul", {})
    proposed = candidates.get("fixed_order_masked_where_sum", {})
    diagnostic_ok = current.get("correctness", {}).get("pass") is True and (
        proposed.get("correctness", {}).get("pass") is True
    )
    marker = (
        "TA_HISTORY_REDUCTION_BENCHMARK_PASS"
        if diagnostic_ok
        else "TA_HISTORY_REDUCTION_BENCHMARK_FAIL"
    )
    print(marker)
    return 0 if diagnostic_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
