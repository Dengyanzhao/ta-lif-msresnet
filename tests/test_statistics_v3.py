from __future__ import annotations

import pandas as pd
import pytest

from talif_msresnet.statistics_v3 import (
    V3_BOOTSTRAP_SEEDS,
    V3_FORMAL_SEEDS,
    V3PairedAnalysisError,
    analyze_v3_paired_accuracy,
    complete_pair_bootstrap,
    exhaustive_sign_flip,
    paired_t_test,
    validate_v3_paired_blocks,
)


PROTOCOL_HASH = "v3-protocol-hash"


def v3_metrics(
    *,
    cifar100_differences: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    dvs_differences: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0, 3.0),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("cifar100", 20, 6, 0.70, cifar100_differences),
        ("cifar10dvs", 20, 10, 0.65, dvs_differences),
    )
    for dataset, depth, time_steps, base_accuracy, differences in specifications:
        for seed, difference in zip(V3_FORMAL_SEEDS, differences, strict=True):
            shared_hash = f"shared-{dataset}-{seed}"
            for condition in ("C1", "C2"):
                accuracy = base_accuracy + (difference / 100.0 if condition == "C2" else 0.0)
                rows.append(
                    {
                        "run_id": (
                            f"E1_{dataset}_d{depth}_t{time_steps}_{condition}_s{seed}"
                        ),
                        "experiment": "E1",
                        "dataset": dataset,
                        "depth": depth,
                        "time_steps": time_steps,
                        "condition": condition,
                        "neuron": "lif" if condition == "C1" else "ta_lif",
                        "topology": "spiking_resnet",
                        "seed": seed,
                        "status": "complete",
                        "failed": 0,
                        "test_accuracy": accuracy,
                        "protocol_hash": PROTOCOL_HASH,
                        "config_hash": f"config-{dataset}-{condition}-{seed}",
                        "split_manifest_sha256": f"split-{dataset}",
                        "shared_weight_sha256": shared_hash,
                        "test_checkpoint_sha256": (
                            f"checkpoint-{dataset}-{condition}-{seed}"
                        ),
                        "test_evaluated_at": "2026-08-01T00:00:00+00:00",
                    }
                )
    return pd.DataFrame(rows)


def test_complete_v3_analysis_uses_frozen_pairs_and_percentage_points() -> None:
    result = analyze_v3_paired_accuracy(
        v3_metrics(),
        expected_protocol_hash=PROTOCOL_HASH,
    )

    primary = result.datasets["cifar100"]
    assert tuple(item.seed for item in primary.raw_differences) == V3_FORMAL_SEEDS
    assert [item.difference_pp for item in primary.raw_differences] == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert primary.paired_t.estimable
    assert primary.paired_t.degrees_of_freedom == 4
    assert primary.paired_t.mean_difference_pp == pytest.approx(3.0)
    assert primary.paired_t.t_statistic == pytest.approx(4.242640687119285)
    assert primary.paired_t.p_value_one_sided_greater == pytest.approx(
        0.006617799781841345
    )
    assert primary.paired_t.ci95_low_pp == pytest.approx(1.0367568385224428)
    assert primary.paired_t.ci95_high_pp == pytest.approx(4.963243161477557)
    assert result.claim_gates.primary_improvement_allowed
    assert result.claim_gates.primary_result == "TA-LIF improved accuracy on CIFAR-100"


def test_sign_flip_and_bootstrap_match_frozen_deterministic_contract() -> None:
    differences = [1.0, 2.0, 3.0, 4.0, 5.0]
    sign_flip = exhaustive_sign_flip(differences)
    first = complete_pair_bootstrap(differences, dataset="cifar100")
    second = complete_pair_bootstrap(differences, dataset="cifar100")

    assert sign_flip.assignments == 32
    assert sign_flip.extreme_assignments == 1
    assert sign_flip.p_value_one_sided_greater == pytest.approx(1 / 32)
    assert first == second
    assert first.n_resamples == 10_000
    assert first.random_seed == V3_BOOTSTRAP_SEEDS["cifar100"]
    assert first.n_complete_pairs == 5
    assert first.ci95_low_pp == pytest.approx(1.8)
    assert first.ci95_high_pp == pytest.approx(4.2)


def test_dataset_specific_bootstrap_seed_has_independent_fixed_output() -> None:
    result = complete_pair_bootstrap([-1.0, 0.0, 1.0, 2.0, 3.0], dataset="cifar10dvs")

    assert result.random_seed == V3_BOOTSTRAP_SEEDS["cifar10dvs"]
    assert result.ci95_low_pp == pytest.approx(-0.2)
    assert result.ci95_high_pp == pytest.approx(2.2)


def test_zero_variance_is_not_estimable_and_primary_is_inconclusive() -> None:
    direct = paired_t_test([1.0] * 5)
    result = analyze_v3_paired_accuracy(
        v3_metrics(cifar100_differences=(1.0,) * 5),
        expected_protocol_hash=PROTOCOL_HASH,
    )

    assert not direct.estimable
    assert direct.t_statistic is None
    assert direct.p_value_one_sided_greater is None
    assert direct.ci95_low_pp is None
    assert not result.claim_gates.primary_improvement_allowed
    assert result.claim_gates.primary_result == "inconclusive"


def test_dvs_replication_cannot_rescue_failed_primary() -> None:
    result = analyze_v3_paired_accuracy(
        v3_metrics(
            cifar100_differences=(-5.0, -4.0, -3.0, -2.0, -1.0),
            dvs_differences=(1.0, 2.0, 3.0, 4.0, 5.0),
        ),
        expected_protocol_hash=PROTOCOL_HASH,
    )

    assert not result.claim_gates.primary_improvement_allowed
    assert result.claim_gates.dvs_replication_improvement_allowed
    assert result.claim_gates.dvs_directional_support
    assert not result.claim_gates.cross_domain_improvement_allowed
    assert result.claim_gates.dvs_cannot_rescue_primary
    assert result.claim_gates.primary_result == "inconclusive"


def test_validation_rejects_partial_pair_instead_of_dropping_it() -> None:
    frame = v3_metrics()
    incomplete = frame.loc[
        ~(
            (frame["dataset"] == "cifar100")
            & (frame["seed"] == V3_FORMAL_SEEDS[0])
            & (frame["condition"] == "C2")
        )
    ]
    with pytest.raises(V3PairedAnalysisError, match="exactly 20 rows"):
        validate_v3_paired_blocks(incomplete, expected_protocol_hash=PROTOCOL_HASH)


def test_validation_rejects_c3_even_when_row_count_remains_twenty() -> None:
    frame = v3_metrics()
    frame.loc[0, "condition"] = "C3"
    with pytest.raises(V3PairedAnalysisError, match="conditions must be exactly C1/C2"):
        validate_v3_paired_blocks(frame, expected_protocol_hash=PROTOCOL_HASH)


def test_validation_rejects_seed_substitution_and_protocol_mismatch() -> None:
    wrong_seed = v3_metrics()
    mask = wrong_seed["seed"] == V3_FORMAL_SEEDS[-1]
    wrong_seed.loc[mask, "seed"] = 12345
    with pytest.raises(V3PairedAnalysisError, match="formal seeds differ"):
        validate_v3_paired_blocks(wrong_seed, expected_protocol_hash=PROTOCOL_HASH)

    with pytest.raises(V3PairedAnalysisError, match="protocol hashes differ"):
        validate_v3_paired_blocks(
            v3_metrics(),
            expected_protocol_hash="different-v3-protocol-hash",
        )


def test_validation_rejects_test_checkpoint_reuse() -> None:
    frame = v3_metrics()
    frame.loc[1, "test_checkpoint_sha256"] = frame.loc[0, "test_checkpoint_sha256"]
    with pytest.raises(V3PairedAnalysisError, match="test checkpoint SHA-256 is reused"):
        validate_v3_paired_blocks(frame, expected_protocol_hash=PROTOCOL_HASH)
