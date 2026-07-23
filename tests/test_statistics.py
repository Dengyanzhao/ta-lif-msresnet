from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import pytest
torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from talif_msresnet.aggregate import build_table4, build_table5, build_table6, standardize_columns  # noqa: E402
from talif_msresnet.benchmark import benchmark_model  # noqa: E402
from talif_msresnet.statistics import (  # noqa: E402
    CONFIRMATORY_SEEDS,
    CONDITION_FACTORS,
    CONDITION_LABELS,
    INTERACTION_SESOI_PP,
    IncompleteSeedBlockError,
    PRIMARY_ACCURACY_GROUPS,
    analyze_seed_level_interaction,
    fit_seed_blocked_accuracy,
    grouped_seed_bootstrap,
    holm_adjust,
    validate_complete_seed_blocks,
)


def synthetic_group(
    dataset: str = "cifar10",
    depth: int = 20,
    time_steps: int = 6,
    *,
    neuron_effect: float = 1.6,
    topology_effect: float = 0.8,
    interaction: float = 0.6,
) -> pd.DataFrame:
    rows = []
    seed_offsets = {11: -0.4, 22: 0.1, 33: 0.0, 44: 0.3, 55: -0.1}
    interaction_offsets = {11: -0.20, 22: 0.10, 33: 0.00, 44: 0.25, 55: -0.15}
    for seed, seed_offset in seed_offsets.items():
        for condition, (neuron, topology) in CONDITION_FACTORS.items():
            value = (
                80.0
                + seed_offset
                + neuron_effect * neuron
                + topology_effect * topology
                + (interaction + interaction_offsets[seed]) * neuron * topology
            )
            rows.append(
                {
                    "dataset": dataset,
                    "depth": depth,
                    "time_steps": time_steps,
                    "condition": condition,
                    "neuron": CONDITION_LABELS[condition][0],
                    "topology": CONDITION_LABELS[condition][1],
                    "seed": seed,
                    "status": "complete",
                    "accuracy": value,
                    "protocol_hash": "protocol-v1",
                    "split_manifest_sha256": f"split-{dataset}",
                    "shared_weight_sha256": (
                        f"shared-{dataset}-{depth}-{time_steps}-{seed}"
                    ),
                    "test_checkpoint_sha256": (
                        f"checkpoint-{dataset}-{depth}-{time_steps}-{condition}-{seed}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_complete_seed_block_validation_rejects_missing_condition() -> None:
    frame = synthetic_group()
    validate_complete_seed_blocks(frame)
    incomplete = frame.loc[~((frame["seed"] == 33) & (frame["condition"] == "C4"))]
    with pytest.raises(IncompleteSeedBlockError, match=r"seed=33.*C4"):
        validate_complete_seed_blocks(incomplete)


def test_complete_seed_block_validation_rejects_duplicate() -> None:
    frame = synthetic_group()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(IncompleteSeedBlockError, match="Duplicate"):
        validate_complete_seed_blocks(duplicated)


def test_confirmatory_seed_validation_requires_the_exact_frozen_set() -> None:
    frame = synthetic_group()
    wrong_seed = frame.copy()
    wrong_seed.loc[wrong_seed["seed"] == 55, "seed"] = 66
    with pytest.raises(IncompleteSeedBlockError, match="confirmatory seeds differ"):
        validate_complete_seed_blocks(wrong_seed, expected_seeds=CONFIRMATORY_SEEDS)


def test_effect_coded_model_recovers_main_effects_and_diff_in_diff() -> None:
    result = fit_seed_blocked_accuracy(
        synthetic_group(), bootstrap_resamples=500, bootstrap_seed=7
    )
    assert result.neuron_effect.estimate == pytest.approx(1.6, abs=1e-10)
    assert result.topology_effect.estimate == pytest.approx(0.8, abs=1e-10)
    assert result.interaction_effect.estimate == pytest.approx(0.6, abs=1e-10)
    assert result.interaction_difference_in_differences == pytest.approx(0.6, abs=1e-10)
    assert result.formula == "accuracy ~ neuron * topology + C(seed)"


def test_grouped_bootstrap_is_reproducible_and_uses_10000_resamples() -> None:
    frame = synthetic_group()
    first = grouped_seed_bootstrap(frame, n_resamples=10_000, random_seed=123)
    second = grouped_seed_bootstrap(frame, n_resamples=10_000, random_seed=123)
    assert first == second
    assert first.n_resamples == 10_000
    assert first.n_seed_blocks == 5
    assert first.ci_low <= first.estimate <= first.ci_high


def test_holm_adjustment_preserves_order() -> None:
    raw = np.array([0.01, 0.04, 0.03, 0.002])
    adjusted = holm_adjust(raw)
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert np.all(adjusted >= raw)


def test_holm_adjustment_fails_closed_on_nonfinite_family() -> None:
    with pytest.raises(ValueError, match="missing or non-finite"):
        holm_adjust([0.01, 0.02, np.nan, 0.04])


def test_table4_contains_two_holm_adjusted_primary_interactions() -> None:
    metrics = pd.concat(
        [
            synthetic_group("cifar100", 20, 6, interaction=0.7),
            synthetic_group("cifar10dvs", 20, 10, interaction=0.9),
        ],
        ignore_index=True,
    )
    table = build_table4(metrics)
    assert len(table) == 2
    assert table["interaction_p_sesoi_holm"].notna().all()
    assert np.allclose(table["interaction_estimate"], [0.7, 0.9])
    assert table["interaction_practical_threshold_pp"].eq(INTERACTION_SESOI_PP).all()
    assert table["bootstrap_resamples"].eq(10_000).all()
    assert table["bootstrap_seed"].eq(20_260_719).all()
    assert table["bootstrap_role"].eq("sensitivity_only_not_confirmatory").all()


def test_seed_level_interaction_uses_five_did_values_and_threshold_null() -> None:
    result = analyze_seed_level_interaction(synthetic_group(interaction=0.9))
    assert tuple(seed for seed, _ in result.seed_contrasts) == CONFIRMATORY_SEEDS
    assert result.estimate == pytest.approx(0.9)
    assert result.degrees_of_freedom == 4
    assert result.sesoi_pp == INTERACTION_SESOI_PP
    assert result.p_value_one_sided_sesoi < 0.05
    assert result.ci_low < result.estimate < result.ci_high


def test_bundled_fractional_test_accuracy_is_converted_to_percent() -> None:
    frame = pd.DataFrame({"test_accuracy": [0.81, 0.82]})
    normalized = standardize_columns(frame)
    assert normalized["accuracy"].tolist() == pytest.approx([81.0, 82.0])


def with_complete_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["diagnostic_id"] = "diagnostic-" + result["test_checkpoint_sha256"]
    result["diagnostic_batch_sha256"] = (
        "representative-batch-" + result["dataset"].astype(str)
    )
    result["diagnostic_protocol_hash"] = "diagnostic-protocol-v1"
    result["gradient_cv_method"] = "CV of residual-block input-gradient RMS on fixed batch"
    result["jacobian_method"] = "Hutchinson/Rademacher reverse-over-reverse VJP"
    result["jacobian_probes"] = 8
    result["jacobian_probe_seed"] = 20_260_719
    result["jacobian_time_index"] = -1
    result["loss_auc"] = 100.0 + result["seed"] / 100.0
    result["gradient_cv"] = 0.1 + result["seed"] / 10_000.0
    result["jacobian_phi"] = 1.0 + result["seed"] / 10_000.0
    result["jacobian_varphi"] = 0.05 + result["seed"] / 100_000.0
    result["converged"] = 1
    return result


def complete_primary_metrics() -> pd.DataFrame:
    return pd.concat(
        [
            synthetic_group(dataset, depth, time_steps)
            for dataset, depth, time_steps in PRIMARY_ACCURACY_GROUPS
        ],
        ignore_index=True,
    )


def complete_primary_diagnostics() -> pd.DataFrame:
    return with_complete_diagnostics(complete_primary_metrics())


def efficiency_seed_metrics() -> pd.DataFrame:
    frame = complete_primary_metrics()
    factor = frame["condition"].map({"C1": 1.0, "C2": 1.05, "C3": 1.0, "C4": 1.2})
    frame["peak_training_memory_bytes"] = factor * 1_000_000 + frame["seed"]
    frame["train_epoch_seconds"] = factor * 10.0 + frame["seed"] / 1000.0
    identity = json.dumps(
        {
            "device": "cuda:0",
            "hardware": {"name": "test-gpu"},
            "software": {"pytorch": "test", "cuda_version": "test"},
            "precision": "float32",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    frame["training_environment_identity"] = identity
    frame["training_environment_sha256"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return frame


def efficiency_benchmarks(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = metrics[["dataset", "depth", "time_steps", "condition", "seed"]].copy()
    factor = keys["condition"].map({"C1": 1.0, "C2": 1.05, "C3": 1.0, "C4": 1.2})
    keys["total_parameters"] = 100_000 + keys["condition"].isin(["C2", "C4"]).astype(int) * 100
    keys["threshold_parameters"] = keys["condition"].isin(["C2", "C4"]).astype(int) * 100
    keys["latency_b1_ms"] = factor * 1.0 + keys["seed"] / 100_000.0
    keys["latency_b128_ms"] = factor * 10.0 + keys["seed"] / 10_000.0
    keys["peak_allocated_b1_bytes"] = factor * 100_000
    keys["peak_allocated_b128_bytes"] = factor * 1_000_000
    keys["firing_rate"] = 0.1 * factor
    keys["syops_per_sample"] = 1_000 * factor
    keys["macs_per_sample"] = 2_000 * factor
    keys["threshold_accesses_per_sample"] = 100 * factor
    keys["count_updates_per_sample"] = 50 * factor
    keys["energy_j_per_sample"] = factor
    keys["energy_status"] = "modeled_from_explicit_constants"
    keys["energy_model_source"] = "synthetic-test-constants"
    keys["benchmark_id"] = (
        "benchmark-"
        + keys["dataset"].astype(str)
        + "-"
        + keys["condition"].astype(str)
        + "-t"
        + keys["time_steps"].astype(str)
        + "-s"
        + keys["seed"].astype(str)
    )
    keys["protocol_hash"] = "protocol-v1"
    keys["hardware"] = "test-gpu"
    keys["benchmark_device_identity"] = (
        '{"compute_capability":"8.0","cuda_version":"12.x","device":"cuda:0",'
        '"hardware_name":"test-gpu","multiprocessor_count":1,"torch_version":"2.x",'
        '"total_memory_bytes":1}'
    )
    keys["precision"] = "float32"
    keys["input_file_sha256"] = "representative-file-" + keys["dataset"].astype(str)
    keys["input_batch_sha256"] = "representative-batch-" + keys["dataset"].astype(str)
    keys["warmup_iterations"] = 25
    keys["timed_iterations"] = 100
    keys["software"] = "torch=test"
    keys["energy_constants_sha256"] = "energy-constants-v1"
    return keys


EFFICIENCY_MARGINS = {
    "latency_b1_percent": 10.0,
    "latency_b128_percent": 10.0,
    "peak_memory_percent": 5.0,
    "modeled_energy_percent": 10.0,
}


@pytest.mark.parametrize(
    ("column", "replacement", "message"),
    [
        ("protocol_hash", "other-protocol", "multiple protocol"),
        ("split_manifest_sha256", "other-split", "multiple train/validation split"),
        ("shared_weight_sha256", "other-shared-weight", "shared initialization"),
    ],
)
def test_seed_block_audit_rejects_mixed_provenance(
    column: str, replacement: str, message: str
) -> None:
    frame = synthetic_group()
    frame.loc[0, column] = replacement
    with pytest.raises(IncompleteSeedBlockError, match=message):
        validate_complete_seed_blocks(frame)


def test_table5_requires_complete_homogeneous_diagnostics() -> None:
    metrics = complete_primary_diagnostics()
    table = build_table5(metrics)
    assert len(table) == 8
    assert (table["n_seed_rows"] == 5).all()
    assert (table["jacobian_phi_n"] == 5).all()
    assert set(table["diagnostic_batch_sha256"]) == {
        "representative-batch-cifar100",
        "representative-batch-cifar10dvs",
    }
    assert table["missing_optional_metrics"].eq("").all()


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("diagnostic_id", "", "diagnostic_id"),
        ("diagnostic_batch_sha256", "other-batch", "diagnostic_batch_sha256"),
        ("diagnostic_protocol_hash", "other-protocol", "diagnostic_protocol_hash"),
        ("gradient_cv_method", "other-gradient-method", "gradient_cv_method"),
        ("jacobian_method", "other-jacobian-method", "jacobian_method"),
        ("jacobian_probes", 16, "jacobian_probes"),
        ("jacobian_probe_seed", 7, "jacobian_probe_seed"),
        ("jacobian_time_index", 0, "jacobian_time_index"),
    ],
)
def test_table5_rejects_missing_or_mixed_diagnostic_contract(
    column: str, value: object, message: str
) -> None:
    metrics = complete_primary_diagnostics()
    metrics.loc[0, column] = value
    with pytest.raises(IncompleteSeedBlockError, match=message):
        build_table5(metrics)


def test_table5_rejects_duplicate_diagnostic_ids() -> None:
    metrics = complete_primary_diagnostics()
    metrics.loc[1, "diagnostic_id"] = metrics.loc[0, "diagnostic_id"]
    with pytest.raises(IncompleteSeedBlockError, match="unique"):
        build_table5(metrics)


def test_table5_rejects_incomplete_seed_condition_block() -> None:
    metrics = complete_primary_diagnostics().iloc[:-1]
    with pytest.raises(IncompleteSeedBlockError, match="C1-C4|missing"):
        build_table5(metrics)


def test_table6_emits_both_primary_groups_and_applies_margins() -> None:
    metrics = efficiency_seed_metrics()
    benchmarks = efficiency_benchmarks(metrics)
    benchmarks = benchmarks.loc[
        ~(
            (benchmarks["dataset"] == "cifar10dvs")
            & (benchmarks["condition"] == "C2")
        )
    ]
    table = build_table6(metrics, benchmarks, efficiency_margins=EFFICIENCY_MARGINS)
    assert len(table) == 8
    assert set(zip(table["dataset"], table["condition"], table["time_steps"])) == {
        (dataset, condition, time_steps)
        for dataset, _depth, time_steps in PRIMARY_ACCURACY_GROUPS
        for condition in ("C1", "C2", "C3", "C4")
    }

    c2_t6 = table.loc[
        (table["dataset"] == "cifar100")
        & (table["condition"] == "C2")
        & (table["time_steps"] == 6)
    ].iloc[0]
    assert c2_t6["peak_training_memory_bytes_n"] == 5
    assert c2_t6["peak_training_memory_bytes_sd"] > 0
    assert "peak_training_memory_bytes" not in c2_t6["missing_optional_metrics"]
    assert c2_t6["latency_b1_paired_n"] == 5
    assert c2_t6["latency_b1_geometric_overhead_percent"] == pytest.approx(5.0, abs=0.02)
    assert c2_t6["latency_b1_tolerance_status"] == "supported_within_tolerance"
    assert c2_t6["efficiency_tolerance_status"] == "supported_within_tolerance"
    assert c2_t6["n_benchmarks"] == 5
    assert c2_t6["timed_iterations"] == 100
    assert c2_t6["benchmark_replication_unit"] == "checkpoint_seed"

    c4_t6 = table.loc[
        (table["dataset"] == "cifar100")
        & (table["condition"] == "C4")
        & (table["time_steps"] == 6)
    ].iloc[0]
    assert c4_t6["efficiency_tolerance_status"] == "exceeds_tolerance"
    assert c4_t6["latency_b1_tolerance_status"] == "exceeds_tolerance"

    missing = table.loc[
        (table["dataset"] == "cifar10dvs")
        & (table["condition"] == "C2")
        & (table["time_steps"] == 10)
    ].iloc[0]
    assert missing["n_benchmarks"] == 0
    assert missing["latency_b1_ms_n"] == 0
    assert "latency_b1_ms" in missing["missing_optional_metrics"]
    assert missing["latency_b1_paired_n"] == 0
    assert missing["efficiency_tolerance_status"] == "not_assessed"


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("protocol_hash", "other-protocol"),
        ("hardware", "other-gpu"),
        ("benchmark_device_identity", '{"device":"cuda:1"}'),
        ("precision", "float16"),
        ("input_batch_sha256", "other-input"),
        ("warmup_iterations", 26),
        ("timed_iterations", 101),
        ("software", "torch=other"),
        ("energy_constants_sha256", "other-energy-constants"),
    ],
)
def test_table6_blocks_mixed_benchmark_contract(column: str, replacement: object) -> None:
    metrics = efficiency_seed_metrics()
    benchmarks = efficiency_benchmarks(metrics)
    benchmarks.loc[0, column] = replacement
    with pytest.raises(IncompleteSeedBlockError, match=column):
        build_table6(metrics, benchmarks, efficiency_margins=EFFICIENCY_MARGINS)


def test_synergy_uses_holm_adjusted_sesoi_tests() -> None:
    metrics = pd.concat(
        [
            synthetic_group("cifar100", 20, 6, interaction=1.0),
            synthetic_group("cifar10dvs", 20, 10, interaction=1.0),
        ],
        ignore_index=True,
    )
    table = build_table4(metrics)
    assert table["interaction_exceeds_practical_threshold"].all()
    assert (table["interaction_p_sesoi_holm"] < 0.05).all()
    assert table["synergy_supported"].all()
    assert not table["complementary_supported"].any()
    assert table["c4_minus_c3_ci95_low"].notna().all()
    assert table["c4_minus_c2_ci95_low"].notna().all()
    assert not table["inconclusive"].any()


def test_additive_and_complementary_wording_gates_are_computed_from_seed_pairs() -> None:
    metrics = pd.concat(
        [
            synthetic_group("cifar100", 20, 6, interaction=0.0),
            synthetic_group("cifar10dvs", 20, 10, interaction=0.0),
        ],
        ignore_index=True,
    )
    table = build_table4(metrics)
    assert not table["synergy_supported"].any()
    assert table["complementary_supported"].all()
    assert table["interaction_equivalent_within_0_50pp"].all()
    assert table["c4_minus_c3_noninferior_0_50pp"].all()
    assert table["c4_minus_c2_noninferior_0_50pp"].all()
    assert table["composable_additive_supported"].all()
    assert not table["inconclusive"].any()


def test_wording_gate_marks_unsupported_composition_inconclusive() -> None:
    metrics = pd.concat(
        [
            synthetic_group(
                "cifar100", 20, 6, neuron_effect=-1.0, topology_effect=-1.0,
                interaction=0.0,
            ),
            synthetic_group(
                "cifar10dvs", 20, 10, neuron_effect=-1.0, topology_effect=-1.0,
                interaction=0.0,
            ),
        ],
        ignore_index=True,
    )
    table = build_table4(metrics)
    assert not table["synergy_supported"].any()
    assert not table["complementary_supported"].any()
    assert not table["composable_additive_supported"].any()
    assert table["inconclusive"].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA event benchmark requires a GPU")
def test_cuda_event_benchmark_smoke() -> None:
    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
            self.fc = nn.Linear(4, 2)

        def reset_state(self) -> None:
            return None

        def forward(self, x: torch.Tensor, collect_activity: bool = False):
            features = self.conv(x).mean(dim=(2, 3))
            logits = self.fc(features)
            if collect_activity:
                diagnostics = {
                    "spike_rate": 0.25,
                    "spike_count": float(x.shape[0]),
                    "elements": float(4 * x.shape[0]),
                }
                return logits, diagnostics
            return logits

    result = benchmark_model(
        TinyModel(),
        lambda batch: torch.randn(batch, 3, 8, 8),
        batch_sizes=(1, 128),
        warmup_iterations=1,
        iterations=2,
        device="cuda",
    )
    assert set(result.batches) == {1, 128}
    assert result.batches[1].latency_mean_ms > 0
    assert result.batches[128].peak_allocated_bytes > 0
