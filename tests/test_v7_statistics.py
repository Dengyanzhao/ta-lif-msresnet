from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from talif_msresnet.config_v7 import (
    V7_ANALYSIS_CONTRACT,
    V7_BOOTSTRAP_SEEDS,
    V7_FORMAL_SEEDS,
    validate_v7_protocol,
)
from talif_msresnet.v7_statistics import (
    V7_BOOTSTRAP_RESAMPLES,
    V7_CONDITIONS,
    V7_HOLM_CONTRASTS,
    V7_PLANNED_SEED_BLOCKS,
    V7_REPLICATION_CONTRAST,
    V7_SECONDARY_CONTRASTS,
    V7_SESOI_PP,
    V7_SIGN_FLIP_ASSIGNMENTS,
    V7StatisticsError,
    analyze_v7_mechanism_accuracy,
    classify_route_contribution,
    classify_study,
    complete_block_bootstrap,
    exhaustive_sign_flip,
    mdes_grid,
    mdes_sensitivity,
    paired_contrast,
    validate_v7_accuracy_blocks,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = V7_FORMAL_SEEDS


def mechanism_frame(
    *,
    route_differences: tuple[float, ...] | None = None,
    m4_shift_pp: float = 1.60,
) -> pd.DataFrame:
    """Build eight complete blocks with non-zero variance in every contrast."""

    if route_differences is None:
        route_differences = (0.76, 0.83, 0.79, 0.86, 0.81, 0.77, 0.84, 0.80)
    if len(route_differences) != len(SEEDS):
        raise ValueError("route_differences must contain exactly eight values")

    perturbation = np.asarray((-0.06, 0.02, -0.03, 0.07, -0.01, 0.04, -0.04, 0.01))
    rows: list[dict[str, object]] = []
    for index, seed in enumerate(SEEDS):
        base = 50.0 + (index - 3.5) * 0.11
        values = {
            "M0": base,
            "M1": base + route_differences[index],
            "M2": base + 1.00 + 0.25 * perturbation[index],
            "M3": base + 0.70 - 0.20 * perturbation[index],
            "M4": base + m4_shift_pp + perturbation[index],
            "PLIF": base + 0.40 - 0.30 * perturbation[index],
        }
        for condition, accuracy in values.items():
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "test_accuracy_pp": accuracy,
                }
            )
    return pd.DataFrame(rows)


def test_frozen_contract_has_six_conditions_eight_blocks_and_direct_contrasts() -> None:
    assert V7_CONDITIONS == ("M0", "M1", "M2", "M3", "M4", "PLIF")
    assert V7_PLANNED_SEED_BLOCKS == 8
    assert (
        V7_REPLICATION_CONTRAST.name,
        V7_REPLICATION_CONTRAST.treatment,
        V7_REPLICATION_CONTRAST.reference,
    ) == ("M4-M0", "M4", "M0")
    assert tuple(
        (spec.name, spec.treatment, spec.reference) for spec in V7_HOLM_CONTRASTS
    ) == (
        ("M4-M2", "M4", "M2"),
        ("M4-M3", "M4", "M3"),
    )
    assert tuple(spec.name for spec in V7_SECONDARY_CONTRASTS) == (
        "M1-M0",
        "PLIF-M0",
        "M4-PLIF",
    )


def test_protocol_config_and_statistics_share_one_frozen_analysis_contract() -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs" / "protocol_v7_mechanism.yaml").read_text(encoding="utf-8")
    )
    validated = validate_v7_protocol(protocol)

    assert validated["protocol_version"] == 7
    assert validated["analysis"] == V7_ANALYSIS_CONTRACT
    assert tuple(validated["seeds"]) == V7_FORMAL_SEEDS
    assert validated["analysis"]["holm_family"]["contrasts"] == ["M4-M2", "M4-M3"]
    assert validated["analysis"]["replication"]["contrast"] == "M4-M0"
    assert tuple(validated["analysis"]["secondary_contrasts"]) == (
        "M1-M0",
        "PLIF-M0",
        "M4-PLIF",
    )
    assert validated["analysis"]["bootstrap"]["seeds"] == V7_BOOTSTRAP_SEEDS
    assert validated["analysis"]["power_planning"]["paired_seed_count"] == 8


def test_v7_requires_exact_complete_seed_blocks() -> None:
    frame = mechanism_frame()
    normalized = validate_v7_accuracy_blocks(frame, expected_seeds=SEEDS)
    assert len(normalized) == len(SEEDS) * len(V7_CONDITIONS) == 48

    incomplete = frame.loc[
        ~((frame["seed"] == SEEDS[0]) & (frame["condition"] == "M4"))
    ]
    with pytest.raises(V7StatisticsError, match="Incomplete V7 seed blocks"):
        validate_v7_accuracy_blocks(incomplete, expected_seeds=SEEDS)


def test_v7_rejects_duplicate_wrong_seed_and_invalid_accuracy() -> None:
    frame = mechanism_frame()
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(V7StatisticsError, match="Duplicate V7 seed-condition rows"):
        validate_v7_accuracy_blocks(duplicate, expected_seeds=SEEDS)

    wrong_seed = frame.copy()
    wrong_seed.loc[wrong_seed["seed"] == SEEDS[-1], "seed"] = 9999
    with pytest.raises(V7StatisticsError, match="V7 seeds differ from the frozen set"):
        validate_v7_accuracy_blocks(wrong_seed, expected_seeds=SEEDS)

    invalid_accuracy = frame.copy()
    invalid_accuracy.loc[0, "test_accuracy_pp"] = np.nan
    with pytest.raises(V7StatisticsError, match="finite percentage points"):
        validate_v7_accuracy_blocks(invalid_accuracy, expected_seeds=SEEDS)

    with pytest.raises(ValueError, match="exact frozen V7 formal seed order"):
        validate_v7_accuracy_blocks(frame, expected_seeds=tuple(range(8)))


def test_supported_analysis_uses_only_the_two_member_holm_family() -> None:
    analysis = analyze_v7_mechanism_accuracy(mechanism_frame(), expected_seeds=SEEDS)

    assert analysis.n_seed_blocks == 8
    assert analysis.replication_passed
    assert analysis.replication.name == "M4-M0"
    assert tuple(result.name for result in analysis.holm_family) == ("M4-M2", "M4-M3")
    assert all(result.superiority_p_holm < 0.05 for result in analysis.holm_family)
    assert analysis.study_conclusion.decision == "mechanism_supported"
    assert analysis.study_conclusion.holm_rejections == ("M4-M2", "M4-M3")
    assert tuple(result.name for result in analysis.secondary_estimates) == (
        "M1-M0",
        "PLIF-M0",
        "M4-PLIF",
    )
    assert analysis.route_conclusion.decision == "route_positive_evidence"
    assert tuple(result.name for result in analysis.sensitivity_analyses) == (
        "M4-M0",
        "M4-M2",
        "M4-M3",
        "M1-M0",
        "PLIF-M0",
        "M4-PLIF",
    )
    assert analysis.to_dict()["study_conclusion"]["decision"] == "mechanism_supported"


def test_m4_minus_m3_is_the_bank_matched_history_index_contrast() -> None:
    frame = validate_v7_accuracy_blocks(mechanism_frame(), expected_seeds=SEEDS)
    result = paired_contrast(frame, V7_HOLM_CONTRASTS[1])

    assert result.name == "M4-M3"
    assert result.treatment == "M4"
    assert result.reference == "M3"
    assert result.estimate_pp > V7_SESOI_PP
    assert tuple(seed for seed, _ in result.seed_differences_pp) == SEEDS


def test_route_branch_can_establish_practical_equivalence() -> None:
    route_differences = (0.03, 0.07, 0.04, 0.06, 0.02, 0.08, 0.05, 0.04)
    frame = validate_v7_accuracy_blocks(
        mechanism_frame(route_differences=route_differences), expected_seeds=SEEDS
    )
    route_result = paired_contrast(frame, V7_SECONDARY_CONTRASTS[0])
    conclusion = classify_route_contribution(route_result)

    assert route_result.estimate_pp < V7_SESOI_PP
    assert conclusion.decision == "route_practical_equivalence"
    assert conclusion.tost.equivalent
    assert conclusion.tost.ci90_low_pp > -V7_SESOI_PP
    assert conclusion.tost.ci90_high_pp < V7_SESOI_PP


def test_non_significant_route_difference_is_not_called_no_effect() -> None:
    route_differences = (-0.80, 0.90, -0.70, 1.00, -0.90, 0.80, -0.60, 0.70)
    frame = validate_v7_accuracy_blocks(
        mechanism_frame(route_differences=route_differences), expected_seeds=SEEDS
    )
    route_result = paired_contrast(frame, V7_SECONDARY_CONTRASTS[0])
    conclusion = classify_route_contribution(route_result)

    assert route_result.superiority_p_raw > 0.05
    assert not conclusion.tost.equivalent
    assert conclusion.decision == "route_contribution_inconclusive"
    assert "non-significance is not no effect" in conclusion.claim


def test_failed_replication_blocks_mechanism_wording() -> None:
    analysis = analyze_v7_mechanism_accuracy(
        mechanism_frame(m4_shift_pp=-0.20), expected_seeds=SEEDS
    )

    assert not analysis.replication_passed
    assert analysis.study_conclusion.decision == (
        "replication_not_confirmed_mechanism_claim_blocked"
    )
    assert not analysis.study_conclusion.replication_passed


def test_study_gate_requires_complete_adjusted_holm_family() -> None:
    analysis = analyze_v7_mechanism_accuracy(mechanism_frame(), expected_seeds=SEEDS)
    unadjusted = replace(analysis.holm_family[0], superiority_p_holm=None)

    with pytest.raises(V7StatisticsError, match="complete two-member Holm family"):
        classify_study(
            replication_passed=True,
            holm_family=(unadjusted, analysis.holm_family[1]),
            route_decision="route_positive_evidence",
        )

    wrong_identity = replace(
        analysis.holm_family[0],
        name="M1-M0",
        treatment="M1",
        reference="M0",
    )
    with pytest.raises(V7StatisticsError, match="frozen contrast identity"):
        classify_study(
            replication_passed=True,
            holm_family=(wrong_identity, analysis.holm_family[1]),
            route_decision="route_positive_evidence",
        )

    with pytest.raises(V7StatisticsError, match="Unknown V7 route decision"):
        classify_study(
            replication_passed=True,
            holm_family=analysis.holm_family,
            route_decision="unregistered_decision",
        )


def test_zero_variance_fails_closed_instead_of_inventing_infinite_t() -> None:
    frame = mechanism_frame()
    m3 = frame.loc[frame["condition"] == "M3", "test_accuracy_pp"].to_numpy()
    frame.loc[frame["condition"] == "M4", "test_accuracy_pp"] = m3 + 1.0
    normalized = validate_v7_accuracy_blocks(frame, expected_seeds=SEEDS)

    with pytest.raises(V7StatisticsError, match="zero paired variance"):
        paired_contrast(normalized, V7_HOLM_CONTRASTS[1])


def test_sign_flip_and_bootstrap_are_frozen_reproducible_sensitivities() -> None:
    frame = validate_v7_accuracy_blocks(mechanism_frame(), expected_seeds=SEEDS)
    result = paired_contrast(frame, V7_REPLICATION_CONTRAST)
    sign_flip = exhaustive_sign_flip(result)
    first = complete_block_bootstrap(result)
    second = complete_block_bootstrap(result)

    assert sign_flip.assignments == V7_SIGN_FLIP_ASSIGNMENTS == 256
    assert sign_flip.extreme_assignments >= 1
    assert sign_flip.p_value_one_sided_greater == pytest.approx(
        sign_flip.extreme_assignments / 256.0
    )
    assert sign_flip.role == "sensitivity_only_cannot_rescue_prespecified_t_tests"
    assert first == second
    assert first.n_resamples == V7_BOOTSTRAP_RESAMPLES == 10_000
    assert first.n_complete_blocks == 8
    assert first.ci95_low_pp <= first.observed_mean_pp <= first.ci95_high_pp
    assert first.role == V7_ANALYSIS_CONTRACT["bootstrap"]["role"]
    assert first.rng == V7_ANALYSIS_CONTRACT["bootstrap"]["rng"]
    assert first.interval == V7_ANALYSIS_CONTRACT["bootstrap"]["interval"]
    assert first.quantile_method == V7_ANALYSIS_CONTRACT["bootstrap"]["quantile_method"]


def test_analysis_rejects_post_freeze_alpha_change() -> None:
    with pytest.raises(ValueError, match="alpha must remain frozen"):
        analyze_v7_mechanism_accuracy(mechanism_frame(), alpha=0.10)


def test_mdes_uses_eight_pairs_two_test_holm_threshold_and_sign_flip_floor() -> None:
    reference = mdes_sensitivity(paired_sd_pp=0.3844867)
    noisier = mdes_sensitivity(paired_sd_pp=1.0)
    grid = mdes_grid()

    assert reference.n_pairs == 8
    assert reference.family_size == 2
    assert reference.conservative_local_alpha == pytest.approx(0.025)
    assert reference.detectable_difference_pp == pytest.approx(0.444, abs=0.001)
    assert reference.one_sided_sign_flip_min_p == pytest.approx(1.0 / 256.0)
    assert reference.conservative_holm_sign_flip_floor == pytest.approx(2.0 / 256.0)
    assert noisier.detectable_difference_pp == pytest.approx(1.156, abs=0.001)
    assert [item.paired_sd_pp for item in grid] == pytest.approx(
        [0.3844867, 0.50, 0.75, 1.00]
    )
