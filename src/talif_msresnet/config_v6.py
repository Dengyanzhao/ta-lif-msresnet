"""Strict protocol-v6 contracts for the TA-LIF mechanism study.

Protocol v6 is intentionally isolated from the historical C1--C4 studies.  It
adds six mechanism conditions on CIFAR-100 and prospectively records a gated
CIFAR-10 extension, while leaving every v1--v5 constant and parser path intact.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .config import (
    V3_FORMAL_SEEDS,
    V3_RETIRED_SEEDS,
    V4_FORMAL_SEEDS,
    V4_PILOT_SEEDS,
    V4_RETIRED_SEEDS,
    V5_BOOTSTRAP_SEEDS,
    V5_DEVELOPMENT_SEEDS,
    V5_FORMAL_SEEDS,
    V5_HEALTH_SEEDS,
    V5_PILOT_SEEDS,
    V5_RETIRED_SEEDS,
    ConfigError,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    canonical_json,
)
from .pathing import artifact_path_reference
from .utils import sha256_file

V6_PROFILE = "cifar100_mechanism_core"
V6_ACTIVE_CONDITIONS: tuple[str, ...] = (
    "M0",
    "M1",
    "M2",
    "M3",
    "M4",
    "PLIF",
)
V6_CONDITION_SPECS: dict[str, dict[str, str]] = {
    "M0": {"topology": "spiking_resnet", "neuron": "lif"},
    "M1": {"topology": "spiking_resnet", "neuron": "route_matched_lif"},
    "M2": {"topology": "spiking_resnet", "neuron": "shared_window_ta_lif"},
    "M3": {"topology": "spiking_resnet", "neuron": "time_indexed_ta_lif"},
    "M4": {"topology": "spiking_resnet", "neuron": "ta_lif"},
    "PLIF": {"topology": "spiking_resnet", "neuron": "plif_style"},
}
V6_TA_PARAMETER_CONDITIONS = frozenset({"M2", "M3", "M4"})
V6_ROUTE_MATCHED_CONDITIONS = frozenset({"M1", "M2", "M3", "M4"})
V6_OPERATION_MODE_BY_CONDITION: dict[str, str] = {
    "M0": "none",
    "M1": "none",
    "M2": "shared_window",
    "M3": "time_indexed_bank",
    "M4": "count_indexed_bank",
    "PLIF": "none",
}

V6_SEED_DERIVATION_NAMESPACE = "ta-lif-msresnet/v6-mechanism-study/2026-08-10"
V6_HEALTH_SEED = 1707261715
V6_PILOT_SEED = 22068314
V6_FORMAL_SEEDS: tuple[int, ...] = (
    1587277406,
    187544621,
    990604905,
    969985360,
    1718581689,
    465608478,
    1670924910,
    1825076945,
)
V6_BOOTSTRAP_SEEDS: dict[str, int] = {
    "M4-M0": 1128314682,
    "M4-M2": 65044423,
    "M4-M3": 774715468,
    "M1-M0": 508189787,
    "PLIF-M0": 1956664330,
    "M4-PLIF": 367144660,
}
V6_BENCHMARK_SEED = 1169352105

V6_CIFAR10_HEALTH_SEED = 1537480873
V6_CIFAR10_PILOT_SEED = 2048258242
V6_CIFAR10_FORMAL_SEEDS: tuple[int, ...] = (
    786414499,
    1691781719,
    897483192,
    2138200647,
    1641516093,
)
V6_CIFAR10_BOOTSTRAP_SEEDS: dict[str, int] = {
    "M4-M0": 1811716963,
    "M4-M3": 1134185216,
}
V6_CIFAR10_BENCHMARK_SEED = 38481167

V6_OUTPUT_ROOT = "results/formal_v6_mechanism"
V6_ARTIFACT_PATHS: dict[str, str] = {
    "protocol": "configs/protocol_v6_mechanism.yaml",
    "signoff": "PREREGISTRATION_SIGNOFF_V6_MECHANISM.md",
    "formal_matrix": "configs/v6_mechanism_generated",
    "pilot_matrix": "configs/v6_mechanism_pilot_generated",
    "freeze_manifest": "FREEZE_MANIFEST_V6_MECHANISM.json",
    "formal_results": V6_OUTPUT_ROOT,
    "pilot_results": "results/pilot/v6_mechanism",
    "analysis_results": "results/analysis/v6_mechanism",
    "benchmark_results": "results/benchmark/v6_mechanism",
}
V6_PROTOCOL_CONFIRMATION_FIELDS: tuple[str, ...] = (
    "v5_evidence_remains_immutable_and_out_of_scope",
    "six_condition_mechanism_design",
    "condition_level_confound_audit",
    "plif_style_parameterization_and_optimizer_policy",
    "time_index_definition_k_equals_min_t_tminus1",
    "paired_seed_and_shared_initialization_design",
    "health_pilot_and_formal_run_handling",
    "holm_family_equivalence_and_wording_gates",
    "conditional_cifar10_extension",
    "validation_only_benchmark_and_one_time_test_access",
)

V6_MODEL_CONTRACT: dict[str, Any] = {
    "base_channels": 16,
    "terminal_neuron_mode": "topology_required",
    "neuron_cfg": {
        "tau": 0.5,
        "threshold": 1.0,
        "v_rest": 0.0,
        "v_reset": 0.0,
        "width": 1.0,
        "delta_min": 0.05,
    },
}
V6_OPTIMIZER_CONTRACT: dict[str, Any] = {
    "epochs": 120,
    "batch_size": 64,
    "lr": 0.025,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "milestones": [75, 90, 105],
    "gamma": 0.1,
    "ta_start_fraction": 5.0 / 120.0,
    "ta_lr_scale": 0.1,
    "ta_weight_decay": 0.0,
    "grad_clip": 1.0,
    "warmup_epochs": 5,
    "exclude_norm_and_bias_from_weight_decay": True,
    "label_smoothing": 0.0,
    "cutmix_alpha": 0.0,
    "cutmix_probability": 0.0,
}
V6_RUNTIME_CONTRACT: dict[str, Any] = {
    "device": "auto",
    "checkpoint_every": 1,
    "log_every": 100,
    "amp": False,
    "deterministic": True,
    "dry_run": False,
    "limit_batches": None,
    "resume": None,
}
V6_DATA_CONTRACT: dict[str, Any] = {
    "root": "data",
    "val_fraction": 0.1,
    "split_seed": 2024,
    "split_manifest": "data/manifests/{dataset}_seed{split_seed}.json",
    "normalize": True,
    "augment": True,
    "autoaugment": False,
    "download": True,
    "num_workers": 0,
    "pin_memory": True,
}
V6_CIFAR100_CONTRACT: dict[str, Any] = {
    "dataset": "cifar100",
    "root": "data/cifar100",
    "num_classes": 100,
    "in_channels": 3,
}
V6_CIFAR100_PROVENANCE_CONTRACT: dict[str, Any] = {
    "source_provenance_path": "environment/CIFAR100_SOURCE_PROVENANCE.json",
    "source_provenance_schema": "cifar100-source-provenance-v1",
    "source_provenance_sha256": (
        "32f1bfe88102f728f969499399790e678b4ade5602bfef72706069a1a84d5c45"
    ),
    "test_pickle_path": "data/cifar100/cifar-100-python/test",
    "test_pickle_bytes": 31049707,
    "test_pickle_sha256": (
        "4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864"
    ),
    "split_manifest_path": "data/manifests/cifar100_seed2024.json",
    "split_manifest_sha256": (
        "ade5f378ea8864b8c36711c4c3bc0e4e7016adb1eaded30d2dc5cf7b0d998f26"
    ),
}
V6_MATRIX_CONTRACT: dict[str, Any] = {
    "primary": [
        {
            "experiment": "E6",
            "dataset": "cifar100",
            "depth": 20,
            "time_steps": 6,
            "seeds": list(V6_FORMAL_SEEDS),
        }
    ]
}

V6_BENCHMARK_CONTRACT: dict[str, Any] = {
    "role": "descriptive_activity_and_operation_proxy_not_measured_energy",
    "dataset_partition": "validation_only_never_test",
    "device_type": "cuda",
    "precision": "float32",
    "batch_sizes": [1, 128],
    "warmup_iterations": 25,
    "timed_iterations": 100,
    "timing_method": "synchronized_cuda_events",
    "input_residency": "preloaded_on_device_before_warmup_and_timing",
    "input_seed": V6_BENCHMARK_SEED,
    "checkpoint_scope": "all_48_frozen_best_checkpoints",
    "energy_claim": "forbidden",
}

V6_PILOT_ACCEPTANCE: dict[str, Any] = {
    "identity": "v6_cifar100_six_condition_e120_mechanism",
    "artifact_class": "non_reportable_pilot",
    "dataset": "cifar100",
    "conditions": list(V6_ACTIVE_CONDITIONS),
    "health_seed": V6_HEALTH_SEED,
    "pilot_seed": V6_PILOT_SEED,
    "health_output": (
        "results/pilot/v6_mechanism/health_cifar100_s1707261715.json"
    ),
    "attempt_receipt": (
        "results/pilot/v6_mechanism/health_cifar100_s1707261715.attempt.json"
    ),
    "pilot_output_root": "results/pilot/v6_mechanism/cifar100_s22068314",
    "pilot_plan": "environment/v6_mechanism_pilot_cifar100_s22068314.json",
    "validation_output": "results/pilot/v6_mechanism/validation.json",
    "environment": {
        "expected_gpu_substring": "RTX 5090",
        "pytorch_version": "2.9.1+cu128",
        "torchvision_version": "0.24.1+cu128",
        "cuda_runtime": "12.8",
        "precision": "float32",
        "deterministic": True,
        "cublas_workspace_config": ":4096:8",
    },
    "health": {
        "decision_basis": "mechanism_gradient_state_resume_and_numerical_integrity",
        "performance_thresholds": "none",
        "fixed_batch_size": 8,
        "fixed_batch_steps": 80,
        "schedule_epochs": 6,
        "train_batches_per_epoch": 16,
        "validation_batches_per_epoch": 4,
        "required_ta_activation_epoch_zero_based": 5,
        "required_checkpoint_resume_boundary": "epoch_4_to_epoch_5_zero_based",
        "all_conditions_required": True,
        "nonfinite_tolerance": 0,
    },
    "pilot": {
        "epochs": 120,
        "all_six_conditions_required": True,
        "required_best_checkpoint": True,
        "required_finite_metrics": True,
        "required_gradient_coverage": 0.95,
        "minimum_best_validation_accuracy": 0.20,
        "late_window_epochs": 10,
        "minimum_late_to_best_ratio": 0.75,
        "comparative_performance_role": "non_reportable_no_model_selection",
    },
    "run_handling": {
        "technical_interruption_policy": (
            "same_environment_last_checkpoint_resume_permitted"
        ),
        "cross_environment_resume": "forbidden",
        "failed_or_interrupted_fresh_retry": "forbidden",
        "failure_action": "new_protocol_version_and_new_unused_seed",
    },
}

V6_ANALYSIS_CONTRACT: dict[str, Any] = {
    "collect_activity": False,
    "accuracy_scale": "proportion",
    "validation_accuracy_thresholds": {"cifar100": 0.20},
    "final_outcome": "independent_test_accuracy_of_validation_selected_best_checkpoint",
    "pairing_unit": "seed_with_shared_conv_bn_classifier_initialization",
    "complete_block_rule": "all_six_conditions_required_no_partial_analysis",
    "alpha": 0.05,
    "confidence_level": 0.95,
    "sesoi_pp": 0.50,
    "replication": {
        "contrast": "M4-M0",
        "role": "confirmatory_independent_replication_of_v5_cifar100_direction",
        "test": "one_sided_paired_t_greater",
        "decision_rule": "p_lt_0_05_and_mean_delta_gt_0",
        "cannot_be_rescued_by_other_contrasts": True,
    },
    "holm_family": {
        "contrasts": ["M4-M2", "M4-M3"],
        "hypotheses": [
            "count_indexed_bank_outperforms_shared_trainable_window",
            "count_indexed_bank_outperforms_time_indexed_bank",
        ],
        "tests": "one_sided_seed_paired_t_greater",
        "familywise_alpha": 0.05,
        "method": "holm_step_down",
        "ordered_thresholds": [0.025, 0.05],
        "direct_contrast_required": True,
        "significance_vs_nonsignificance_comparison": "forbidden",
    },
    "secondary_contrasts": {
        "M1-M0": "gradient_route_contribution_estimate",
        "PLIF-M0": "generic_learnable_decay_baseline_estimate",
        "M4-PLIF": "full_talif_vs_learnable_decay_estimate",
    },
    "secondary_branching": {
        "route_positive_evidence": (
            "m1_minus_m0_two_sided_95ci_lower_gt_0_and_mean_ge_0_50pp"
        ),
        "route_practical_equivalence": (
            "m1_minus_m0_tost_alpha_0_05_within_plus_minus_0_50pp"
        ),
        "otherwise": "route_contribution_inconclusive",
        "secondary_evidence_cannot_rescue_holm_family": True,
    },
    "equivalence": {
        "method": "paired_tost",
        "alpha": 0.05,
        "bounds_pp": [-0.50, 0.50],
        "interval": "two_sided_90_percent_t",
        "decision_rule": "both_one_sided_tests_reject",
        "nonsignificant_difference_is_not_equivalence": True,
    },
    "sign_flip": {
        "assignments": 256,
        "enumeration": "all_2_power_8_seed_level_sign_flips",
        "role": "sensitivity_only_cannot_rescue_prespecified_t_tests",
    },
    "bootstrap": {
        "resamples": 10000,
        "resampling_unit": "complete_six_condition_seed_block",
        "rng": "numpy_generator_pcg64",
        "interval": "two_sided_95_percent_percentile",
        "quantile_method": "linear",
        "seeds": dict(V6_BOOTSTRAP_SEEDS),
        "role": "sensitivity_only",
    },
    "power_planning": {
        "paired_seed_count": 8,
        "one_sided_family_worst_case_alpha": 0.025,
        "power": 0.80,
        "reference_sd_pp": 0.3844867,
        "reference_mdes_pp": 0.444,
        "sensitivity_sd_pp": [0.50, 0.75, 1.00],
        "prospective_not_posthoc_power_claim": True,
    },
    "model_selection": {
        "checkpoint": "best.pt",
        "primary_order": "maximum_validation_accuracy",
        "first_tie_breaker": "minimum_validation_loss",
        "exact_tie_breaker": "earliest_epoch",
        "test_based_selection": "forbidden",
    },
    "test_access": {
        "training_config_final_test": False,
        "when": "after_all_48_runs_and_best_checkpoints_pass_frozen_audit",
        "access_count_per_checkpoint": 1,
        "reselection_or_repeated_evaluation": "forbidden",
        "ambiguous_interruption": "stop_document_and_obtain_author_decision",
    },
    "run_handling": {
        "health_and_pilot_in_reportable_analysis": "forbidden",
        "failed_run_records": "retain",
        "seed_substitution": "forbidden",
        "outlier_exclusion": "forbidden",
        "incomplete_seed_block": "unresolved_no_partial_confirmatory_analysis",
        "cross_environment_checkpoint_resume": "forbidden",
        "protocol_change_after_first_reportable_run": "forbidden",
    },
    "wording_gates": {
        "mechanism_supported": (
            "both_holm_contrasts_reject_positive_and_replication_passes"
        ),
        "partial_mechanism_support": (
            "exactly_one_holm_contrast_rejects_positive_and_replication_passes"
        ),
        "mechanism_inconclusive": "neither_holm_contrast_rejects",
        "equivalence_requires_tost": True,
        "route_contribution_requires_secondary_branch_rule": True,
        "plif_style_not_exact_fang_plif": True,
    },
    "efficiency": {
        "role": "descriptive_only",
        "metrics": [
            "parameter_count",
            "cuda_latency_b1",
            "cuda_latency_b128",
            "peak_memory",
            "spike_rate",
            "synaptic_operation_proxy",
        ],
        "validation_only_input": True,
        "energy": "not_measured_no_energy_claim",
    },
}

V6_CONDITIONAL_EXTENSION: dict[str, Any] = {
    "status": "preregistered_inactive_until_core_gate_receipt_passes",
    "dataset": "cifar10",
    "scientific_role": "static_modality_boundary_triangulation_not_causal_modality_test",
    "conditions": ["M0", "M3", "M4"],
    "depth": 20,
    "time_steps": 6,
    "formal_seeds": list(V6_CIFAR10_FORMAL_SEEDS),
    "health_seed": V6_CIFAR10_HEALTH_SEED,
    "pilot_seed": V6_CIFAR10_PILOT_SEED,
    "bootstrap_seeds": dict(V6_CIFAR10_BOOTSTRAP_SEEDS),
    "benchmark_seed": V6_CIFAR10_BENCHMARK_SEED,
    "activation_gate": {
        "core_integrity_status": "PASS",
        "m4_minus_m0_replication": "pass_one_sided_p_lt_0_05_and_mean_gt_0",
        "holm_mechanism_requirement": "at_least_one_positive_holm_rejection",
        "manual_condition_selection_after_results": "forbidden",
    },
    "run_count": 15,
    "new_freeze_manifest_required": True,
    "results_root": "results/formal_v6_cifar10_boundary",
}


def _exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ConfigError(f"{name} must be frozen as {expected!r}, got {value!r}")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{name} must be a sequence")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise ConfigError(f"Unknown key(s) in {name}: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"Missing key(s) in {name}: {', '.join(missing)}")


def _record(label: str, counter: int = 0) -> dict[str, Any]:
    payload = f"{V6_SEED_DERIVATION_NAMESPACE}|{label}|{counter}".encode()
    digest = hashlib.sha256(payload).digest()
    return {
        "label": label,
        "counter": counter,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "value": int.from_bytes(digest[:8], "big") & 0x7FFFFFFF,
    }


def expected_v6_seed_records() -> list[dict[str, Any]]:
    labels = ["health|cifar100|1", "pilot|cifar100|1"]
    labels.extend(f"formal|cifar100|{index}" for index in range(1, 9))
    labels.extend(
        f"bootstrap|cifar100|{name}"
        for name in ("m4-m0", "m4-m2", "m4-m3", "m1-m0", "plif-m0", "m4-plif")
    )
    labels.extend(["benchmark|cifar100|1", "health|cifar10|1", "pilot|cifar10|1"])
    labels.extend(f"formal|cifar10|{index}" for index in range(1, 6))
    labels.extend(
        f"bootstrap|cifar10|{name}" for name in ("m4-m0", "m4-m3")
    )
    labels.append("benchmark|cifar10|1")
    return [_record(label) for label in labels]


def _all_historical_seeds() -> frozenset[int]:
    return frozenset(
        {
            *V3_FORMAL_SEEDS,
            *V3_RETIRED_SEEDS,
            *V4_FORMAL_SEEDS,
            *V4_PILOT_SEEDS.values(),
            *V4_RETIRED_SEEDS,
            *V5_FORMAL_SEEDS,
            *V5_HEALTH_SEEDS.values(),
            *V5_PILOT_SEEDS.values(),
            *V5_BOOTSTRAP_SEEDS.values(),
            *V5_DEVELOPMENT_SEEDS,
            *V5_RETIRED_SEEDS,
        }
    )


def _validate_seed_ledger(value: Any) -> None:
    ledger = _mapping(value, "protocol.seed_ledger")
    expected_keys = {
        "namespace",
        "algorithm",
        "collision_policy",
        "historical_seed_policy",
        "records",
    }
    _keys(ledger, expected_keys, "protocol.seed_ledger")
    _exact(ledger["namespace"], V6_SEED_DERIVATION_NAMESPACE, "seed_ledger.namespace")
    _exact(
        ledger["algorithm"],
        "sha256_first_eight_bytes_big_endian_mask_positive_31_bit",
        "seed_ledger.algorithm",
    )
    _exact(
        ledger["collision_policy"],
        "reject_zero_historical_or_previous_then_increment_decimal_counter",
        "seed_ledger.collision_policy",
    )
    _exact(
        ledger["historical_seed_policy"],
        "all_v1_to_v5_health_pilot_formal_bootstrap_and_development_seeds_excluded",
        "seed_ledger.historical_seed_policy",
    )
    records = list(_sequence(ledger["records"], "protocol.seed_ledger.records"))
    expected = expected_v6_seed_records()
    _exact(records, expected, "protocol.seed_ledger.records")
    values = [int(record["value"]) for record in records]
    if not all(values) or len(values) != len(set(values)):
        raise ConfigError("V6 seed ledger contains zero or duplicate values")
    overlap = sorted(set(values) & _all_historical_seeds())
    if overlap:
        raise ConfigError(f"V6 seed ledger reuses historical seeds: {overlap}")


def _validate_status(value: Any) -> None:
    status = _mapping(value, "protocol.protocol_status")
    _keys(
        status,
        {"frozen", "confirmed_by", "confirmed_at", "notes", "confirmations"},
        "protocol.protocol_status",
    )
    _exact(status["frozen"], True, "protocol_status.frozen")
    for field in ("confirmed_by", "confirmed_at", "notes"):
        if not isinstance(status[field], str) or not status[field].strip():
            raise ConfigError(f"protocol_status.{field} must be a non-empty string")
    confirmations = _mapping(
        status["confirmations"], "protocol.protocol_status.confirmations"
    )
    _keys(
        confirmations,
        set(V6_PROTOCOL_CONFIRMATION_FIELDS),
        "protocol.protocol_status.confirmations",
    )
    for field in V6_PROTOCOL_CONFIRMATION_FIELDS:
        _exact(confirmations[field], True, f"protocol_status.confirmations.{field}")


def validate_v6_protocol(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete, author-frozen v6 core protocol."""

    raw = _mapping(raw, "protocol")
    expected_top = {
        "protocol_version",
        "study_profile",
        "study_stage",
        "active_conditions",
        "seeds",
        "output_root",
        "artifact_paths",
        "protocol_status",
        "seed_ledger",
        "pilot_acceptance",
        "data",
        "datasets",
        "cifar100_provenance",
        "conditions",
        "model",
        "optimizer",
        "runtime",
        "experiments",
        "analysis",
        "benchmark",
        "matrix",
        "conditional_extension",
    }
    _keys(raw, expected_top, "protocol")
    _exact(raw["protocol_version"], 6, "protocol_version")
    _exact(raw["study_profile"], V6_PROFILE, "study_profile")
    _exact(raw["study_stage"], "health_pilot_formal", "study_stage")
    _exact(tuple(raw["active_conditions"]), V6_ACTIVE_CONDITIONS, "active_conditions")
    _exact(tuple(raw["seeds"]), V6_FORMAL_SEEDS, "seeds")
    _exact(raw["output_root"], V6_OUTPUT_ROOT, "output_root")
    _exact(dict(raw["artifact_paths"]), V6_ARTIFACT_PATHS, "artifact_paths")
    _validate_status(raw["protocol_status"])
    _validate_seed_ledger(raw["seed_ledger"])
    _exact(dict(raw["pilot_acceptance"]), V6_PILOT_ACCEPTANCE, "pilot_acceptance")
    _exact(dict(raw["data"]), V6_DATA_CONTRACT, "data")
    datasets = _mapping(raw["datasets"], "protocol.datasets")
    _keys(datasets, {"cifar100"}, "protocol.datasets")
    _exact(dict(datasets["cifar100"]), V6_CIFAR100_CONTRACT, "datasets.cifar100")
    _exact(
        dict(_mapping(raw["cifar100_provenance"], "protocol.cifar100_provenance")),
        V6_CIFAR100_PROVENANCE_CONTRACT,
        "cifar100_provenance",
    )
    conditions = _mapping(raw["conditions"], "protocol.conditions")
    _keys(conditions, set(V6_ACTIVE_CONDITIONS), "protocol.conditions")
    _exact(
        {name: dict(value) for name, value in conditions.items()},
        V6_CONDITION_SPECS,
        "conditions",
    )
    _exact(dict(raw["model"]), V6_MODEL_CONTRACT, "model")
    _exact(dict(raw["optimizer"]), V6_OPTIMIZER_CONTRACT, "optimizer")
    _exact(dict(raw["runtime"]), V6_RUNTIME_CONTRACT, "runtime")
    _exact(
        dict(raw["experiments"]),
        {
            "E6": "paired six-condition CIFAR-100 mechanism study",
            "E7": "prespecified paired mechanism analysis",
            "E8": "validation-only descriptive resource audit",
        },
        "experiments",
    )
    _exact(dict(raw["analysis"]), V6_ANALYSIS_CONTRACT, "analysis")
    _exact(dict(raw["benchmark"]), V6_BENCHMARK_CONTRACT, "benchmark")
    _exact(dict(raw["matrix"]), V6_MATRIX_CONTRACT, "matrix")
    _exact(
        dict(raw["conditional_extension"]),
        V6_CONDITIONAL_EXTENSION,
        "conditional_extension",
    )
    return copy.deepcopy(dict(raw))


def _v6_provenance_file(
    project_root: Path,
    relative_path: str,
    label: str,
) -> Path:
    path = (project_root / relative_path).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ConfigError(f"V6 {label} escapes the project root: {path}") from exc
    if not path.is_file():
        raise ConfigError(f"V6 {label} is missing: {path}")
    return path


def validate_v6_cifar100_provenance_files(
    protocol: Mapping[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, str]:
    """Verify the frozen CIFAR-100 source, test, and split byte identities.

    This is deliberately a file-integrity gate, not a test-loader construction
    path.  It makes the held-out test pickle identifiable without exposing any
    examples or labels to model selection.
    """

    validated = validate_v6_protocol(protocol)
    root = Path(project_root).resolve()
    contract = validated["cifar100_provenance"]
    provenance_path = _v6_provenance_file(
        root,
        str(contract["source_provenance_path"]),
        "CIFAR-100 source provenance",
    )
    test_path = _v6_provenance_file(
        root,
        str(contract["test_pickle_path"]),
        "CIFAR-100 test pickle",
    )
    split_path = _v6_provenance_file(
        root,
        str(contract["split_manifest_path"]),
        "CIFAR-100 split manifest",
    )
    observed = {
        "source_provenance_sha256": sha256_file(provenance_path),
        "test_pickle_sha256": sha256_file(test_path),
        "split_manifest_sha256": sha256_file(split_path),
    }
    for key, value in observed.items():
        if value != contract[key]:
            raise ConfigError(
                f"V6 CIFAR-100 {key} differs from the frozen protocol: "
                f"{value} != {contract[key]}"
            )
    if test_path.stat().st_size != int(contract["test_pickle_bytes"]):
        raise ConfigError(
            "V6 CIFAR-100 test pickle byte count differs from the frozen protocol"
        )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot parse frozen V6 CIFAR-100 provenance input: {exc}") from exc
    if not isinstance(provenance, Mapping) or not isinstance(split, Mapping):
        raise ConfigError("Frozen V6 CIFAR-100 provenance and split inputs must be JSON objects")
    if provenance.get("schema") != contract["source_provenance_schema"]:
        raise ConfigError("V6 CIFAR-100 source provenance schema differs from the protocol")
    extracted = provenance.get("extracted_binding")
    files = extracted.get("files") if isinstance(extracted, Mapping) else None
    test_record = files.get("test") if isinstance(files, Mapping) else None
    if not isinstance(test_record, Mapping) or (
        test_record.get("sha256") != contract["test_pickle_sha256"]
        or test_record.get("bytes") != contract["test_pickle_bytes"]
    ):
        raise ConfigError("V6 CIFAR-100 source provenance does not bind the frozen test pickle")
    split_record = provenance.get("development_split_binding")
    if not isinstance(split_record, Mapping) or (
        split_record.get("path") != contract["split_manifest_path"]
        or split_record.get("file_sha256") != contract["split_manifest_sha256"]
    ):
        raise ConfigError("V6 CIFAR-100 source provenance does not bind the frozen split manifest")
    internal_hash = split_record.get("internal_manifest_sha256")
    if not isinstance(internal_hash, str) or split.get("manifest_sha256") != internal_hash:
        raise ConfigError("V6 CIFAR-100 split manifest does not match its provenance record")
    return {
        "cifar100_source_provenance": artifact_path_reference(provenance_path, root),
        "cifar100_source_provenance_sha256": observed["source_provenance_sha256"],
        "cifar100_test_pickle": artifact_path_reference(test_path, root),
        "cifar100_test_pickle_sha256": observed["test_pickle_sha256"],
        "cifar100_split_manifest": artifact_path_reference(split_path, root),
        "cifar100_split_manifest_sha256": observed["split_manifest_sha256"],
    }


def _merge_data(protocol: Mapping[str, Any], override: Mapping[str, Any]) -> DataConfig:
    unknown = set(override) - set(dataclasses.asdict(DataConfig()))
    if unknown:
        raise ConfigError(f"Unknown key(s) in run.data: {', '.join(sorted(unknown))}")
    values = dataclasses.asdict(DataConfig())
    values.update(protocol["data"])
    values.update(protocol["datasets"]["cifar100"])
    values.update(override)
    if values["dataset"] != "cifar100":
        raise ConfigError("V6 core run dataset must be cifar100")
    frozen = dict(V6_DATA_CONTRACT)
    frozen.update(V6_CIFAR100_CONTRACT)
    for key, expected in frozen.items():
        _exact(values[key], expected, f"v6 run data.{key}")
    return DataConfig(**values)


def _merge_model(
    protocol: Mapping[str, Any],
    override: Mapping[str, Any],
    condition: str,
    data: DataConfig,
) -> ModelConfig:
    allowed = set(dataclasses.asdict(ModelConfig()))
    unknown = set(override) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in run.model: {', '.join(sorted(unknown))}")
    values = dataclasses.asdict(ModelConfig())
    values.update(protocol["model"])
    values.update(V6_CONDITION_SPECS[condition])
    values.update({"condition": condition, "depth": 20, "time_steps": 6})
    values.update({"num_classes": data.num_classes, "in_channels": data.in_channels})
    values.update(override)
    expected = {
        "condition": condition,
        **V6_CONDITION_SPECS[condition],
        "depth": 20,
        "time_steps": 6,
        "num_classes": 100,
        "in_channels": 3,
        **V6_MODEL_CONTRACT,
    }
    for key, frozen in expected.items():
        _exact(values[key], frozen, f"v6 run model.{key}")
    return ModelConfig(**values)


def _merge_optimizer(protocol: Mapping[str, Any], override: Mapping[str, Any]) -> OptimizerConfig:
    allowed = set(dataclasses.asdict(OptimizerConfig()))
    unknown = set(override) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in run.optimizer: {', '.join(sorted(unknown))}")
    values = dataclasses.asdict(OptimizerConfig())
    values.update(protocol["optimizer"])
    values.update(override)
    normalized = dict(values)
    normalized["milestones"] = list(normalized["milestones"])
    _exact(normalized, V6_OPTIMIZER_CONTRACT, "v6 run optimizer")
    values["milestones"] = tuple(values["milestones"])
    return OptimizerConfig(**values)


def _merge_runtime(
    protocol: Mapping[str, Any],
    override: Mapping[str, Any],
    *,
    run_id: str,
    seed: int,
    expected_output: str,
) -> RuntimeConfig:
    allowed = set(dataclasses.asdict(RuntimeConfig()))
    unknown = set(override) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in run.runtime: {', '.join(sorted(unknown))}")
    values = dataclasses.asdict(RuntimeConfig())
    values.update(protocol["runtime"])
    values.update({"run_id": run_id, "seed": seed, "output_dir": expected_output})
    values.update(override)
    frozen = {**V6_RUNTIME_CONTRACT, "run_id": run_id, "seed": seed, "output_dir": expected_output}
    for key, expected in frozen.items():
        _exact(values[key], expected, f"v6 run runtime.{key}")
    return RuntimeConfig(**values)


def validate_v6_run_mapping(
    raw: Mapping[str, Any], protocol: Mapping[str, Any] | None
) -> RunConfig:
    """Resolve one strict v6 formal or pilot run configuration."""

    if protocol is None:
        raise ConfigError("Protocol v6 run validation requires its protocol mapping")
    protocol = validate_v6_protocol(protocol)
    raw = _mapping(raw, "run")
    allowed = {
        "protocol_version",
        "experiment",
        "run_id",
        "condition",
        "seed",
        "data",
        "model",
        "optimizer",
        "runtime",
        "final_test",
        "analysis",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in run: {', '.join(unknown)}")
    _exact(raw.get("protocol_version"), 6, "run.protocol_version")
    experiment = str(raw.get("experiment", ""))
    _exact(experiment, "E6", "run.experiment")
    model_raw = dict(_mapping(raw.get("model", {}), "run.model"))
    runtime_raw = dict(_mapping(raw.get("runtime", {}), "run.runtime"))
    condition = str(raw.get("condition", model_raw.get("condition", "")))
    if condition not in V6_ACTIVE_CONDITIONS:
        raise ConfigError(f"V6 run condition must be one of {V6_ACTIVE_CONDITIONS}")
    seed_value = raw.get("seed", runtime_raw.get("seed"))
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ConfigError("V6 run seed must be an integer")
    seed = int(seed_value)
    if seed in V6_FORMAL_SEEDS:
        expected_output = V6_OUTPUT_ROOT
    elif seed == V6_PILOT_SEED:
        expected_output = str(V6_PILOT_ACCEPTANCE["pilot_output_root"])
    else:
        raise ConfigError("V6 run seed is neither a frozen formal nor pilot seed")
    expected_run_id = f"E6_cifar100_d20_t6_{condition}_s{seed}"
    run_id = str(raw.get("run_id", runtime_raw.get("run_id", "")))
    _exact(run_id, expected_run_id, "v6 run run_id")
    data = _merge_data(protocol, _mapping(raw.get("data", {}), "run.data"))
    model = _merge_model(protocol, model_raw, condition, data)
    optimizer = _merge_optimizer(
        protocol, _mapping(raw.get("optimizer", {}), "run.optimizer")
    )
    runtime = _merge_runtime(
        protocol,
        runtime_raw,
        run_id=run_id,
        seed=seed,
        expected_output=expected_output,
    )
    _exact(raw.get("final_test", False), False, "v6 run final_test")
    analysis = dict(_mapping(raw.get("analysis", {}), "run.analysis"))
    expected_protocol_hash = hashlib.sha256(
        canonical_json(protocol).encode("utf-8")
    ).hexdigest()
    matrix_key = ["cifar100", 20, 6, condition, seed]
    expected_analysis = copy.deepcopy(V6_ANALYSIS_CONTRACT)
    expected_analysis.update(
        {"matrix_key": matrix_key, "protocol_hash": expected_protocol_hash}
    )
    _exact(analysis, expected_analysis, "v6 run analysis")
    return RunConfig(
        protocol_version=6,
        experiment=experiment,
        data=data,
        model=model,
        optimizer=optimizer,
        runtime=runtime,
        final_test=False,
        analysis=analysis,
    )


def _raw_v6_run(
    protocol: Mapping[str, Any], *, condition: str, seed: int, output_dir: str
) -> dict[str, Any]:
    protocol_hash = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
    run_id = f"E6_cifar100_d20_t6_{condition}_s{seed}"
    analysis = copy.deepcopy(V6_ANALYSIS_CONTRACT)
    analysis.update(
        {
            "matrix_key": ["cifar100", 20, 6, condition, seed],
            "protocol_hash": protocol_hash,
        }
    )
    return {
        "protocol_version": 6,
        "experiment": "E6",
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "data": dict(V6_CIFAR100_CONTRACT),
        "model": {
            "condition": condition,
            **V6_CONDITION_SPECS[condition],
            "depth": 20,
            "time_steps": 6,
        },
        "optimizer": copy.deepcopy(V6_OPTIMIZER_CONTRACT),
        "runtime": {"output_dir": output_dir},
        "final_test": False,
        "analysis": analysis,
    }


def generate_v6_formal_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = validate_v6_protocol(protocol)
    runs = [
        _raw_v6_run(
            protocol,
            condition=condition,
            seed=seed,
            output_dir=V6_OUTPUT_ROOT,
        )
        for seed in V6_FORMAL_SEEDS
        for condition in V6_ACTIVE_CONDITIONS
    ]
    if len(runs) != 48 or len({run["run_id"] for run in runs}) != 48:
        raise ConfigError("V6 formal matrix must contain exactly 48 unique runs")
    return runs


def generate_v6_pilot_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = validate_v6_protocol(protocol)
    output = str(V6_PILOT_ACCEPTANCE["pilot_output_root"])
    runs = [
        _raw_v6_run(
            protocol,
            condition=condition,
            seed=V6_PILOT_SEED,
            output_dir=output,
        )
        for condition in V6_ACTIVE_CONDITIONS
    ]
    if len(runs) != 6 or len({run["run_id"] for run in runs}) != 6:
        raise ConfigError("V6 pilot matrix must contain exactly six unique runs")
    return runs


def canonicalize_v6_artifact_run_mapping(
    raw: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    project_root: str | Path,
    expected_output_dir: str = V6_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Normalize only a path-equivalent V6 runtime artifact for revalidation.

    Generated YAML files retain the exact repository-relative string required
    by :func:`validate_v6_run_mapping`.  Runtime manifests and checkpoints can
    legitimately contain the absolute form injected by the orchestrator.  This
    helper accepts that single representational difference only when both paths
    resolve to the same protocol-bound directory; it never relaxes the strict
    run validator itself.
    """

    validated = validate_v6_protocol(protocol)
    value = copy.deepcopy(dict(_mapping(raw, "V6 runtime artifact run")))
    _exact(value.get("protocol_version"), 6, "artifact run.protocol_version")
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ConfigError("V6 runtime artifact run.runtime must be a mapping")
    runtime_value = copy.deepcopy(dict(runtime))
    observed_value = runtime_value.get("output_dir")
    if not isinstance(observed_value, str) or not observed_value.strip():
        raise ConfigError("V6 runtime artifact output_dir must be a non-empty string")

    root = Path(project_root).resolve()
    expected = Path(expected_output_dir)
    if expected.is_absolute():
        raise ConfigError("V6 canonical output_dir must remain repository-relative")
    expected_path = (root / expected).resolve()
    try:
        expected_path.relative_to(root)
    except ValueError as exc:
        raise ConfigError("V6 canonical output_dir escapes the repository") from exc

    observed = Path(observed_value)
    observed_path = (
        observed.resolve() if observed.is_absolute() else (root / observed).resolve()
    )
    if observed_path != expected_path:
        raise ConfigError(
            "V6 runtime artifact output_dir is not path-equivalent to the frozen "
            f"directory: {observed_value!r} != {expected_output_dir!r}"
        )
    runtime_value["output_dir"] = expected_output_dir
    value["runtime"] = runtime_value

    # Resolve here as well so callers cannot use normalization as a substitute
    # for the full protocol/run contract.
    validate_v6_run_mapping(value, validated)
    return value


def v6_path(project_root: str | Path, key: str) -> Path:
    """Resolve one frozen v6 artifact path inside the repository."""

    root = Path(project_root).resolve()
    if key not in V6_ARTIFACT_PATHS:
        raise ConfigError(f"Unknown V6 artifact path key: {key}")
    resolved = (root / V6_ARTIFACT_PATHS[key]).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"V6 artifact path escapes the repository: {key}") from exc
    return resolved
