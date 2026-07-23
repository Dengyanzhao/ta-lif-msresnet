from __future__ import annotations

import copy
import dataclasses
import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from talif_msresnet.config import (  # noqa: E402
    CONDITION_SPECS,
    ConfigError,
    PRESPECIFIED_BENCHMARK,
    PRESPECIFIED_EFFICIENCY_COMPARISON,
    PRESPECIFIED_PRIMARY_GROUPS,
    PRESPECIFIED_SEEDS,
    SUPPORTED_CONDITIONS,
    canonical_json,
    generate_run_matrix,
    load_protocol,
    validate_protocol,
    validate_run_mapping,
)


@pytest.fixture(scope="module")
def protocol():
    return load_protocol(ROOT / "configs" / "protocol.yaml")


def test_protocol_is_strict_and_has_five_unique_seeds(protocol):
    assert protocol["protocol_version"] == 1
    assert tuple(protocol["seeds"]) == PRESPECIFIED_SEEDS
    assert protocol["analysis"]["validation_accuracy_thresholds"] == {
        "cifar10": 0.70,
        "cifar100": 0.60,
        "cifar10dvs": 0.60,
    }
    assert protocol["analysis"]["interaction_practical_threshold_pp"] == 0.50
    assert protocol["analysis"]["primary_interaction_test"]["sidedness"] == "one_sided_greater"
    assert protocol["analysis"]["multiplicity_correction"] == "holm_two_confirmatory_tests"
    assert protocol["analysis"]["multiplicity_family"] == "two_confirmatory_interaction_tests"
    assert protocol["analysis"]["bootstrap"] == {
        "resamples": 10_000,
        "seed": 20_260_719,
        "role": "sensitivity_only",
        "resampling_unit": "complete_seed_block",
        "cellwise_resampling": "forbidden",
        "interval": "two_sided_95_percent_percentile",
    }
    assert protocol["analysis"]["efficiency_assessment"] == "descriptive"
    assert protocol["analysis"]["efficiency_comparison"] == PRESPECIFIED_EFFICIENCY_COMPARISON
    assert protocol["analysis"]["wording_gates"] == {
        "synergy_precedence": True,
        "complementary_and_composable_may_both_apply": True,
        "synergistic": "sesoi_shifted_one_sided_holm_reject",
        "complementary": (
            "synergy_not_supported_and_c4_minus_c3_95ci_lower_gt_0_and_"
            "c4_minus_c2_95ci_lower_gt_0"
        ),
        "composable_additive": (
            "interaction_90ci_within_plus_minus_0_50pp_and_c4_vs_c2_and_c3_"
            "one_sided_95ci_lower_gt_minus_0_50pp"
        ),
        "inconclusive": "no_applicable_gate_supported_or_incomplete_block",
    }
    assert protocol["analysis"]["diagnostic_batch_size"] == 8
    assert protocol["analysis"]["jacobian_time_index"] == -1
    assert {key: protocol["benchmark"][key] for key in PRESPECIFIED_BENCHMARK} == PRESPECIFIED_BENCHMARK
    assert protocol["benchmark"]["energy_model"] == {
        "status": "not_assessed",
        "constants_path": None,
        "constants_sha256": "none",
        "source": "none",
    }
    assert tuple(
        (slot["experiment"], slot["dataset"], slot["depth"], slot["time_steps"])
        for slot in protocol["matrix"]["primary"]
    ) == PRESPECIFIED_PRIMARY_GROUPS


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    (
        (("seeds",), [11, 22, 33, 44, 66], "seeds"),
        (("analysis", "validation_accuracy_thresholds", "cifar10"), 0.71, "cifar10"),
        (("analysis", "interaction_practical_threshold_pp"), 0.40, "interaction"),
        (("analysis", "alpha"), 0.10, "alpha"),
        (("analysis", "primary_interaction_test", "sidedness"), "two_sided", "sidedness"),
        (("analysis", "primary_interaction_test", "estimation_interval"), "wald", "interaction"),
        (("analysis", "bootstrap", "resamples"), 9999, "bootstrap"),
        (("analysis", "bootstrap", "resampling_unit"), "cell", "bootstrap"),
        (("analysis", "efficiency_comparison", "effect_scale"), "raw_ratio", "efficiency"),
        (("analysis", "efficiency_comparison", "mixed_training_environment"), "ignore", "efficiency"),
        (("analysis", "run_handling", "nonconvergence_excludes_run"), True, "run_handling"),
        (("analysis", "jacobian_time_index"), 0, "jacobian_time_index"),
        (("benchmark", "precision"), "float16", "precision"),
        (("benchmark", "warmup_iterations"), 24, "warmup"),
        (("benchmark", "energy_model", "status"), "modeled", "constants_path"),
    ),
)
def test_prespecified_protocol_values_cannot_drift(protocol, path, replacement, message):
    bad = copy.deepcopy(protocol)
    target = bad
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(ConfigError, match=message):
        validate_protocol(bad)


def test_protocol_hash_binds_analysis_and_benchmark(protocol):
    expected = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
    runs = generate_run_matrix(protocol)
    assert {run["analysis"]["protocol_hash"] for run in runs} == {expected}
    changed = copy.deepcopy(protocol)
    changed["benchmark"]["timed_iterations"] += 1
    changed_hash = hashlib.sha256(canonical_json(changed).encode("utf-8")).hexdigest()
    assert changed_hash != expected


def test_matrix_has_40_unique_runs(protocol):
    runs = generate_run_matrix(protocol)
    assert len(runs) == 40
    keys = {
        (r["experiment"], r["data"]["dataset"], r["model"]["depth"], r["model"]["time_steps"], r["condition"], r["seed"])
        for r in runs
    }
    assert len(keys) == 40
    assert len({r["run_id"] for r in runs}) == 40


def test_conditions_are_fixed_pairs(protocol):
    for run in generate_run_matrix(protocol):
        assert run["condition"] in SUPPORTED_CONDITIONS
        assert run["model"]["topology"] == CONDITION_SPECS[run["condition"]]["topology"]
        assert run["model"]["neuron"] == CONDITION_SPECS[run["condition"]]["neuron"]


def test_matrix_contains_only_complete_e1_primary_blocks(protocol):
    runs = generate_run_matrix(protocol)
    assert {r["experiment"] for r in runs} == {"E1"}
    observed_groups = {
        (r["experiment"], r["data"]["dataset"], r["model"]["depth"], r["model"]["time_steps"])
        for r in runs
    }
    assert observed_groups == set(PRESPECIFIED_PRIMARY_GROUPS)
    for group in observed_groups:
        block = [
            r
            for r in runs
            if (
                r["experiment"],
                r["data"]["dataset"],
                r["model"]["depth"],
                r["model"]["time_steps"],
            )
            == group
        ]
        assert len(block) == 20
        assert {r["condition"] for r in block} == set(SUPPORTED_CONDITIONS)
        assert {r["seed"] for r in block} == set(PRESPECIFIED_SEEDS)


def test_unknown_top_level_key_is_rejected(protocol):
    bad = copy.deepcopy(generate_run_matrix(protocol)[0])
    bad["typoed_key"] = True
    with pytest.raises(ConfigError, match="Unknown key"):
        validate_run_mapping(bad, protocol)


def test_mismatched_condition_is_rejected(protocol):
    bad = copy.deepcopy(generate_run_matrix(protocol)[0])
    bad["model"]["neuron"] = "ta_lif"
    with pytest.raises(ConfigError, match="requires topology"):
        validate_run_mapping(bad, protocol)


def test_invalid_neuron_decay_is_rejected_before_model_construction(protocol):
    bad = copy.deepcopy(protocol)
    bad["model"]["neuron_cfg"]["tau"] = 2.0
    from talif_msresnet.config import validate_protocol

    with pytest.raises(ConfigError, match=r"tau.*<= 1"):
        validate_protocol(bad)


def test_scientific_hash_ignores_resume_path_but_not_seed(protocol):
    config = validate_run_mapping(generate_run_matrix(protocol)[0], protocol)
    resumed = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, resume="results/last.pt", device="cuda:1"),
    )
    different_seed = dataclasses.replace(
        config,
        runtime=dataclasses.replace(config.runtime, seed=config.runtime.seed + 1),
    )
    assert resumed.config_hash == config.config_hash
    assert resumed.execution_hash != config.execution_hash
    assert different_seed.config_hash != config.config_hash
