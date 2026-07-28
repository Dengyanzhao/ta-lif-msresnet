from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_v3_results import (  # noqa: E402
    V3_ANALYSIS_RESULTS_REFERENCE,
    V3_FORMAL_RESULTS_REFERENCE,
    V3_PROTOCOL_REFERENCE,
    isolated_analysis_paths,
    validate_v3_protocol_contract,
)
from talif_msresnet.statistics_v3 import (  # noqa: E402
    V3_BOOTSTRAP_SEEDS,
    V3_FORMAL_SEEDS,
)


def v3_protocol_contract() -> dict[str, object]:
    return {
        "protocol_version": 3,
        "active_conditions": ["C1", "C2"],
        "seeds": list(V3_FORMAL_SEEDS),
        "output_root": V3_FORMAL_RESULTS_REFERENCE,
        "artifact_paths": {
            "protocol": V3_PROTOCOL_REFERENCE,
            "signoff": "PREREGISTRATION_SIGNOFF_V3_TALIF_ONLY.md",
            "formal_matrix": "configs/v3_talif_only_generated",
            "pilot_matrix": "configs/v3_talif_only_pilot_generated",
            "freeze_manifest": "FREEZE_MANIFEST_V3_TALIF_ONLY.json",
            "formal_results": V3_FORMAL_RESULTS_REFERENCE,
            "pilot_results": "results/pilot/v3_talif_only",
            "analysis_results": V3_ANALYSIS_RESULTS_REFERENCE,
        },
        "matrix": {
            "primary": [
                {
                    "experiment": "E1",
                    "dataset": "cifar100",
                    "depth": 20,
                    "time_steps": 6,
                },
                {
                    "experiment": "E1",
                    "dataset": "cifar10dvs",
                    "depth": 20,
                    "time_steps": 10,
                },
            ]
        },
        "analysis": {
            "accuracy_scale": "proportion",
            "alpha": 0.05,
            "confidence_level": 0.95,
            "primary_dataset": "cifar100",
            "replication_dataset": "cifar10dvs",
            "primary_accuracy_test": {
                "estimand": "mean_seed_paired_c2_minus_c1_test_accuracy_pp",
                "blocking_factor": "seed",
                "test_statistic": "one_sample_t_over_seed_level_differences",
                "null_hypothesis": "mean_delta_le_0",
                "alternative_hypothesis": "mean_delta_gt_0",
                "sidedness": "one_sided_greater",
                "estimation_interval": "two_sided_95_percent_t",
                "degrees_of_freedom": 4,
                "decision_rule": "p_lt_0_05_and_mean_delta_gt_0",
                "normality_pretest_switch": "forbidden",
                "zero_variance_rule": "not_estimable_primary_inconclusive",
            },
            "replication_analysis": {
                "cannot_rescue_primary": True,
                "cross_dataset_multiplicity": "none_single_confirmatory_primary",
            },
            "sign_flip": {
                "assignments": 32,
                "enumeration": "all_2_power_5_seed_level_sign_flips",
                "sidedness": "one_sided_greater",
                "role": "sensitivity_only_cannot_rescue_primary",
            },
            "bootstrap": {
                "resamples": 10_000,
                "resampling_unit": "complete_seed_pair",
                "cellwise_resampling": "forbidden",
                "rng": "numpy_generator_pcg64",
                "index_draw": "integers_0_5_size_10000_by_5_endpoint_false",
                "statistic": "mean_seed_paired_c2_minus_c1_test_accuracy_pp",
                "interval": "two_sided_95_percent_percentile",
                "quantile_method": "linear",
                "seeds": dict(V3_BOOTSTRAP_SEEDS),
                "role": "sensitivity_only_cannot_rescue_primary",
            },
            "run_handling": {
                "incomplete_seed_pair": "unresolved_no_partial_analysis",
                "seed_substitution": "forbidden",
                "pilot_in_reportable_analysis": "forbidden",
            },
            "wording_gates": {
                "cifar100_improved": "primary_p_lt_0_05_and_mean_delta_gt_0",
                "primary_not_passed": "inconclusive_no_equivalence_or_no_effect_claim",
                "dvs_cannot_rescue_primary": True,
                "dvs_directional_support": "mean_delta_gt_0",
                "dvs_replication_claim": "one_sided_p_lt_0_05_and_mean_delta_gt_0",
                "cross_domain_improvement": "both_dataset_tests_pass_and_both_means_gt_0",
            },
        },
    }


def test_cli_protocol_contract_accepts_only_frozen_v3_analysis_design() -> None:
    validate_v3_protocol_contract(v3_protocol_contract())

    changed = deepcopy(v3_protocol_contract())
    changed["analysis"]["bootstrap"]["resamples"] = 9999  # type: ignore[index]
    with pytest.raises(ValueError, match="analysis.bootstrap"):
        validate_v3_protocol_contract(changed)


def test_cli_paths_are_isolated_from_legacy_and_pilot_artifacts() -> None:
    protocol = v3_protocol_contract()
    root = Path("C:/isolated-v3-analysis-test-root")
    protocol_path = root / V3_PROTOCOL_REFERENCE

    input_path, output_path = isolated_analysis_paths(
        protocol_path,
        protocol,
        None,
        None,
        repository_root=root,
    )
    assert input_path == (root / V3_FORMAL_RESULTS_REFERENCE).resolve()
    assert output_path == (root / V3_ANALYSIS_RESULTS_REFERENCE).resolve()

    with pytest.raises(ValueError, match="isolated formal-results root"):
        isolated_analysis_paths(
            protocol_path,
            protocol,
            root / "results" / "runs",
            None,
            repository_root=root,
        )
    with pytest.raises(ValueError, match="isolated analysis-results root"):
        isolated_analysis_paths(
            protocol_path,
            protocol,
            None,
            root / "results" / "pilot" / "v3_talif_only",
            repository_root=root,
        )


def test_cli_rejects_legacy_protocol_path_even_with_v3_mapping() -> None:
    root = Path("C:/isolated-v3-analysis-test-root")
    legacy = root / "configs" / "protocol.yaml"
    with pytest.raises(ValueError, match="requires protocol"):
        isolated_analysis_paths(
            legacy,
            v3_protocol_contract(),
            None,
            None,
            repository_root=root,
        )
