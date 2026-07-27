"""Configuration loading and experiment-matrix generation.

The configuration layer deliberately has a small, explicit schema.  A typo in a
run file must fail before a GPU job is started; silently accepting unknown YAML
keys makes a large multi-seed study difficult to reproduce.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:  # PyYAML is part of the experiment environment, but keep the import lazy-friendly.
    import yaml
except ImportError:  # pragma: no cover - a useful error is raised by load_yaml.
    yaml = None


class ConfigError(ValueError):
    """Raised when a protocol or run configuration violates the schema."""


SUPPORTED_CONDITIONS: Tuple[str, ...] = ("C1", "C2", "C3", "C4")
CONDITION_SPECS: Dict[str, Dict[str, str]] = {
    "C1": {"topology": "spiking_resnet", "neuron": "lif"},
    "C2": {"topology": "spiking_resnet", "neuron": "ta_lif"},
    "C3": {"topology": "ms_resnet", "neuron": "lif"},
    "C4": {"topology": "ms_resnet", "neuron": "ta_lif"},
}

PROTOCOL_CONFIRMATION_FIELDS: Tuple[str, ...] = (
    "reference_implementation_reviewed",
    "neuron_parameters",
    "static_augmentation_package",
    "optimizer_and_schedule",
    "cifar10dvs_preprocessing_and_split",
    "outcome_thresholds_and_margins",
)

PRESPECIFIED_SEEDS: Tuple[int, ...] = (11, 22, 33, 44, 55)
PRESPECIFIED_PRIMARY_GROUPS: Tuple[Tuple[str, str, int, int], ...] = (
    ("E1", "cifar100", 20, 6),
    ("E1", "cifar10dvs", 20, 10),
)
V2_PILOT_SEEDS: Tuple[int, ...] = (77,)
V2_PILOT_PRIMARY_GROUPS: Tuple[Tuple[str, str, int, int], ...] = (
    ("E1", "cifar100", 20, 6),
)
V2_PILOT_ACCEPTANCE: Dict[str, Any] = {
    "artifact_class": "non_reportable_pilot",
    "seed": 77,
    "dataset": "cifar100",
    "conditions": ["C1", "C2", "C3", "C4"],
    "health_output": "results/pilot/v2_seed77_e120_health.json",
    "validation_output": "results/pilot/v2_seed77_e120_validation.json",
    "pilot_output_root": "results/pilot/v2_seed77_e120",
    "pilot_plan": "environment/unfrozen_pilot_plan_v2_seed77_e120.json",
    "environment": {
        "expected_gpu_substring": "RTX 5090",
        "pytorch_version": "2.9.1+cu128",
        "cuda_runtime": "12.8",
        "precision": "float32",
        "deterministic": True,
    },
    "overfit": {
        "batch_size": 8,
        "steps": 40,
        "minimum_accuracy": 0.50,
        "maximum_loss_fraction": 0.90,
        "minimum_ta_routed_gradient_coverage": 0.90,
    },
    "timing": {
        "batch_size": 64,
        "warmup_steps": 2,
        "timed_steps": 5,
        "maximum_ta_enabled_over_frozen_ratio": 3.0,
        "epoch_time_ratio_statistic": (
            "median_enabled_train_seconds_over_median_frozen_train_seconds"
        ),
        "epoch_time_ratio_conditions": ["C2", "C4"],
        "ta_state_source": "events_jsonl_epoch_completed_ta_enabled",
    },
    "schedule": {
        "candidate_epochs": 120,
        "required_best_validation_accuracy": 0.60,
        "require_all_conditions_converged": True,
        "technical_interruption_policy": (
            "same_environment_last_checkpoint_resume_permitted"
        ),
        "cross_environment_resume": "forbidden",
        "retained_failure_artifacts": (
            "allowed_only_after_later_successful_same_run_completion"
        ),
        "fallback_epochs": 160,
        "fallback_action": "fresh_protocol_and_fresh_runs_no_resume",
    },
}
V2R2_PILOT_SEEDS: Tuple[int, ...] = (88,)
V2R2_PILOT_PRIMARY_GROUPS: Tuple[Tuple[str, str, int, int], ...] = (
    ("E1", "cifar100", 20, 6),
)
V2R2_PILOT_ACCEPTANCE: Dict[str, Any] = {
    "identity": "v2r2_seed88_of80_e120",
    "artifact_class": "non_reportable_pilot",
    "seed": 88,
    "dataset": "cifar100",
    "conditions": ["C1", "C2", "C3", "C4"],
    "health_output": "results/pilot/v2r2_seed88_of80_e120_health.json",
    "validation_output": "results/pilot/v2r2_seed88_of80_e120_validation.json",
    "pilot_output_root": "results/pilot/v2r2_seed88_of80_e120",
    "pilot_plan": (
        "environment/unfrozen_pilot_plan_v2r2_seed88_of80_e120.json"
    ),
    "environment": {
        "expected_gpu_substring": "RTX 5090",
        "pytorch_version": "2.9.1+cu128",
        "cuda_runtime": "12.8",
        "precision": "float32",
        "deterministic": True,
    },
    "overfit": {
        "batch_size": 8,
        "steps": 80,
        "minimum_accuracy": 0.50,
        "maximum_loss_fraction": 0.90,
        "minimum_ta_routed_gradient_coverage": 0.90,
    },
    "timing": {
        "batch_size": 64,
        "warmup_steps": 2,
        "timed_steps": 5,
        "maximum_ta_enabled_over_frozen_ratio": 3.0,
        "epoch_time_ratio_statistic": (
            "median_enabled_train_seconds_over_median_frozen_train_seconds"
        ),
        "epoch_time_ratio_conditions": ["C2", "C4"],
        "ta_state_source": "events_jsonl_epoch_completed_ta_enabled",
    },
    "schedule": {
        "candidate_epochs": 120,
        "required_best_validation_accuracy": 0.60,
        "require_all_conditions_converged": True,
        "technical_interruption_policy": (
            "same_environment_last_checkpoint_resume_permitted"
        ),
        "cross_environment_resume": "forbidden",
        "retained_failure_artifacts": (
            "allowed_only_after_later_successful_same_run_completion"
        ),
        "fallback_epochs": 160,
        "fallback_action": "fresh_protocol_and_fresh_runs_no_resume",
    },
}
V2_PILOT_PROFILE_CONTRACTS: Tuple[
    Tuple[Tuple[int, ...], Tuple[Tuple[str, str, int, int], ...], str, Dict[str, Any]],
    ...,
] = (
    (
        V2_PILOT_SEEDS,
        V2_PILOT_PRIMARY_GROUPS,
        "results/formal_v2",
        V2_PILOT_ACCEPTANCE,
    ),
    (
        V2R2_PILOT_SEEDS,
        V2R2_PILOT_PRIMARY_GROUPS,
        "results/formal_v2r2",
        V2R2_PILOT_ACCEPTANCE,
    ),
)
EXPECTED_RUN_COUNT = (
    len(PRESPECIFIED_PRIMARY_GROUPS) * len(SUPPORTED_CONDITIONS) * len(PRESPECIFIED_SEEDS)
)
PRESPECIFIED_THRESHOLDS: Dict[str, float] = {
    "cifar10": 0.70,
    "cifar100": 0.60,
    "cifar10dvs": 0.60,
}
PRESPECIFIED_EFFICIENCY_MARGINS: Dict[str, float] = {
    "latency_b1_percent": 10.0,
    "latency_b128_percent": 10.0,
    "peak_memory_percent": 5.0,
    "modeled_energy_percent": 10.0,
}
PRESPECIFIED_BOOTSTRAP: Dict[str, Any] = {
    "resamples": 10_000,
    "seed": 20_260_719,
    "role": "sensitivity_only",
    "resampling_unit": "complete_seed_block",
    "cellwise_resampling": "forbidden",
    "interval": "two_sided_95_percent_percentile",
}
PRESPECIFIED_PRIMARY_INTERACTION_TEST: Dict[str, str] = {
    "estimand": "seed_level_difference_in_differences_pp",
    "blocking_factor": "seed",
    "test_statistic": "one_sample_t_over_seed_level_did",
    "null_hypothesis": "delta_le_interaction_practical_threshold",
    "alternative_hypothesis": "delta_gt_interaction_practical_threshold",
    "sidedness": "one_sided_greater",
    "estimation_interval": "two_sided_95_percent_t",
}
PRESPECIFIED_EFFICIENCY_COMPARISON: Dict[str, Any] = {
    "contrasts": ["C2_over_C1", "C4_over_C3"],
    "pairing_unit": "same_seed_within_topology_and_time_steps",
    "effect_scale": "natural_log_ratio",
    "summary": "geometric_percent_overhead",
    "confidence_interval": "paired_two_sided_95_percent_t_on_log_ratio",
    "training_environment_rule": "homogeneous_hardware_software_device_and_precision",
    "mixed_training_environment": "analysis_blocked",
    "status_rules": {
        "point_estimate_gt_tolerance": "exceeds_tolerance",
        "point_estimate_le_tolerance_ci_upper_gt_tolerance": (
            "within_tolerance_uncertain"
        ),
        "point_estimate_le_tolerance_ci_upper_le_tolerance": (
            "supported_within_tolerance"
        ),
        "unavailable_or_invalid": "not_assessed",
    },
    "overall_status_precedence": [
        "exceeds_tolerance",
        "within_tolerance_uncertain",
        "not_assessed",
        "supported_within_tolerance",
    ],
}
PRESPECIFIED_RUN_HANDLING: Dict[str, Any] = {
    "pilot_in_confirmatory_analysis": "forbidden",
    "pilot_scope": "one_complete_c1_c4_seed_block_max_four_unique_runs",
    "pilot_plan": "repository_global_immutable_plan",
    "pilot_output_root": "separate_from_formal_results",
    "failed_run_records": "retain",
    "seed_substitution": "forbidden",
    "outlier_exclusion": "forbidden",
    "nonconvergence_excludes_run": False,
    "incomplete_confirmatory_block": "unresolved_no_partial_analysis",
    "protocol_change_after_first_confirmatory_run": "forbidden",
}
PRESPECIFIED_WORDING_GATES: Dict[str, Any] = {
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
PRESPECIFIED_BENCHMARK: Dict[str, Any] = {
    "device_type": "cuda",
    "precision": "float32",
    "batch_sizes": [1, 128],
    "warmup_iterations": 25,
    "timed_iterations": 100,
    "timing_method": "synchronized_cuda_events",
    "input_residency": "preloaded_on_device_before_warmup_and_timing",
}
ENERGY_MODEL_FIELDS: Tuple[str, ...] = (
    "status",
    "constants_path",
    "constants_sha256",
    "source",
)


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "cifar10"
    root: str = "data"
    num_classes: int = 10
    in_channels: int = 3
    val_fraction: float = 0.1
    split_seed: int = 2024
    split_manifest: str = "data/manifests/{dataset}_seed{split_seed}.json"
    frames_path: str | None = None
    labels_path: str | None = None
    test_frames_path: str | None = None
    test_labels_path: str | None = None
    dvs_time_bins: int = 10
    dvs_height: int = 48
    dvs_width: int = 48
    dvs_resize_mode: str = "nearest"
    dvs_test_fraction: float = 0.2
    dvs_split_seed: int = 2024
    normalize: bool = True
    augment: bool = True
    autoaugment: bool = False
    download: bool = True
    num_workers: int = 0
    pin_memory: bool = True


@dataclass(frozen=True)
class ModelConfig:
    condition: str = "C1"
    topology: str = "spiking_resnet"
    neuron: str = "lif"
    depth: int = 20
    time_steps: int = 6
    num_classes: int = 10
    in_channels: int = 3
    base_channels: int = 16
    neuron_cfg: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerConfig:
    epochs: int = 240
    batch_size: int = 256
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    milestones: Tuple[int, ...] = (150, 180, 210)
    gamma: float = 0.1
    ta_start_fraction: float = 0.05
    ta_lr_scale: float = 0.1
    ta_weight_decay: float = 0.0
    grad_clip: float | None = None
    label_smoothing: float = 0.0
    cutmix_alpha: float = 0.0
    cutmix_probability: float = 0.0


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 11
    device: str = "auto"
    output_dir: str = "results"
    run_id: str = "run"
    checkpoint_every: int = 1
    log_every: int = 100
    amp: bool = False
    deterministic: bool = True
    dry_run: bool = False
    limit_batches: int | None = None
    resume: str | None = None


@dataclass(frozen=True)
class RunConfig:
    """Fully resolved configuration for one condition/seed run."""

    protocol_version: int
    experiment: str
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    runtime: RuntimeConfig
    final_test: bool = False
    analysis: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def scientific_dict(self) -> Dict[str, Any]:
        """Return fields that determine the scientific training result.

        Device selection, paths, logging cadence, smoke limits, resume paths,
        and the later test-access flag are execution controls.  Excluding them
        lets a checkpoint resume on another device/path without weakening the
        binding to seed, precision, model, data, optimizer, or analysis rules.
        """

        value = self.as_dict()
        runtime = value["runtime"]
        for key in (
            "device",
            "output_dir",
            "checkpoint_every",
            "log_every",
            "dry_run",
            "limit_batches",
            "resume",
        ):
            runtime.pop(key, None)
        value.pop("final_test", None)
        return value

    @property
    def config_hash(self) -> str:
        payload = canonical_json(self.scientific_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def execution_hash(self) -> str:
        payload = canonical_json(self.as_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> str:
    """Stable JSON representation used in manifests and checkpoint metadata."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _read_yaml(path: str | Path) -> Any:
    if yaml is None:
        raise ConfigError("PyYAML is required to read configuration files")
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration {path}: {exc}") from exc
    return {} if value is None else value


def _check_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _check_keys(value: Mapping[str, Any], allowed: Iterable[str], name: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ConfigError(f"Unknown key(s) in {name}: {', '.join(unknown)}")


def _int(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _float(value: Any, name: str, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{name} must be <= {maximum}")
    return result


def _str(value: Any, name: str, allowed: Iterable[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{name} must be a non-empty string")
    if allowed is not None and value not in allowed:
        raise ConfigError(f"{name} must be one of {tuple(allowed)}, got {value!r}")
    return value


def _require_exact(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        raise ConfigError(f"{name} must be frozen as {expected!r}, got {value!r}")


def _matrix_group_tuple(slot: Mapping[str, Any]) -> Tuple[str, str, int, int]:
    return (
        str(slot["experiment"]),
        str(slot["dataset"]),
        int(slot["depth"]),
        int(slot["time_steps"]),
    )


def validate_protocol(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the top-level protocol YAML.

    The protocol intentionally permits only documented keys.  Individual run
    files use :func:`load_run_config` and have a separate strict schema.
    """

    raw = _check_mapping(raw, "protocol")
    allowed = {
        "protocol_version", "seeds", "output_root", "data", "datasets", "model",
        "optimizer", "runtime", "matrix", "experiments", "conditions", "analysis",
        "protocol_status", "benchmark", "study_stage", "pilot_acceptance",
    }
    _check_keys(raw, allowed, "protocol")
    required = {
        "protocol_version", "seeds", "output_root", "data", "datasets", "model",
        "optimizer", "runtime", "matrix", "experiments", "conditions", "analysis",
        "protocol_status", "benchmark",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ConfigError(f"Protocol is missing required section(s): {', '.join(missing)}")
    version = _int(raw.get("protocol_version", 1), "protocol_version", 1)
    if version not in (1, 2):
        raise ConfigError("protocol_version must be 1 or 2")
    study_stage = raw.get("study_stage")
    if version == 1:
        if study_stage is not None:
            raise ConfigError("protocol_version 1 must not define study_stage")
        if "pilot_acceptance" in raw:
            raise ConfigError("protocol_version 1 must not define pilot_acceptance")
        expected_seeds = PRESPECIFIED_SEEDS
        expected_groups = PRESPECIFIED_PRIMARY_GROUPS
        expected_output_root = None
    else:
        _require_exact(study_stage, "pilot", "study_stage")
        acceptance = _check_mapping(
            raw.get("pilot_acceptance"), "protocol.pilot_acceptance"
        )
        allowed_acceptance_keys = set().union(
            *(contract[3] for contract in V2_PILOT_PROFILE_CONTRACTS)
        )
        _check_keys(acceptance, allowed_acceptance_keys, "protocol.pilot_acceptance")
        matching_profiles = [
            contract
            for contract in V2_PILOT_PROFILE_CONTRACTS
            if dict(acceptance) == contract[3]
        ]
        if len(matching_profiles) != 1:
            raise ConfigError(
                "protocol.pilot_acceptance must exactly match one supported v2 pilot profile"
            )
        expected_seeds, expected_groups, expected_output_root, _ = matching_profiles[0]
    seeds_raw = raw.get("seeds", list(expected_seeds))
    if not isinstance(seeds_raw, Sequence) or isinstance(seeds_raw, (str, bytes)) or not seeds_raw:
        raise ConfigError("seeds must be a non-empty sequence")
    seeds = [_int(v, "seeds[]", 0) for v in seeds_raw]
    if len(set(seeds)) != len(seeds):
        raise ConfigError("seeds must be unique")
    _require_exact(tuple(seeds), expected_seeds, "seeds")
    output_root = str(raw.get("output_root", "results"))
    if not output_root:
        raise ConfigError("output_root cannot be empty")
    if expected_output_root is not None:
        _require_exact(output_root, expected_output_root, "output_root")

    # Normalize the common sections while retaining dataset/matrix declarations.
    normalized: Dict[str, Any] = copy.deepcopy(dict(raw))
    normalized["protocol_version"] = version
    normalized["seeds"] = seeds
    normalized["output_root"] = output_root
    if "optimizer" in raw:
        _validate_optimizer_mapping(_check_mapping(raw["optimizer"], "protocol.optimizer"), "protocol.optimizer")
    if "runtime" in raw:
        _validate_runtime_mapping(_check_mapping(raw["runtime"], "protocol.runtime"), "protocol.runtime")
    if "model" in raw:
        _validate_model_mapping(_check_mapping(raw["model"], "protocol.model"), "protocol.model", allow_condition=False)
    if "data" in raw:
        _validate_data_mapping(_check_mapping(raw["data"], "protocol.data"), "protocol.data")
    if "datasets" in raw:
        datasets = _check_mapping(raw["datasets"], "protocol.datasets")
        for name, section in datasets.items():
            _validate_data_mapping(_check_mapping(section, f"protocol.datasets.{name}"), f"protocol.datasets.{name}")
    if "conditions" in raw:
        conditions = _check_mapping(raw["conditions"], "protocol.conditions")
        unknown = sorted(set(conditions) - set(SUPPORTED_CONDITIONS))
        if unknown:
            raise ConfigError(f"Unknown condition(s): {', '.join(unknown)}")
        missing = sorted(set(SUPPORTED_CONDITIONS) - set(conditions))
        if missing:
            raise ConfigError(f"Protocol must define all C1-C4 conditions; missing: {', '.join(missing)}")
        for condition, section in conditions.items():
            section = _check_mapping(section, f"protocol.conditions.{condition}")
            _check_keys(section, {"topology", "neuron"}, f"protocol.conditions.{condition}")
            expected = CONDITION_SPECS[condition]
            if section.get("topology", expected["topology"]) != expected["topology"] or section.get("neuron", expected["neuron"]) != expected["neuron"]:
                raise ConfigError(f"Condition {condition} does not match its fixed topology/neuron pair")
    if "experiments" in raw:
        experiments = _check_mapping(raw["experiments"], "protocol.experiments")
        _check_keys(experiments, {"E1", "E2", "E3"}, "protocol.experiments")
        if set(experiments) != {"E1", "E2", "E3"}:
            raise ConfigError("protocol.experiments must define exactly E1, E2, and E3")
        for key, value in experiments.items():
            _str(value, f"protocol.experiments.{key}")
    if "matrix" in raw:
        matrix = _check_mapping(raw["matrix"], "protocol.matrix")
        _check_keys(matrix, {"primary"}, "protocol.matrix")
        if set(matrix) != {"primary"}:
            raise ConfigError("protocol.matrix must define exactly the primary group list")
        for group, slots in matrix.items():
            if not isinstance(slots, Sequence) or isinstance(slots, (str, bytes)):
                raise ConfigError(f"protocol.matrix.{group} must be a sequence")
            for index, slot in enumerate(slots):
                slot = _check_mapping(slot, f"protocol.matrix.{group}[{index}]")
                slot_keys = {"experiment", "dataset", "depth", "time_steps"}
                if version == 2:
                    slot_keys.add("seeds")
                _check_keys(slot, slot_keys, f"protocol.matrix.{group}[{index}]")
                for required in ("experiment", "dataset", "depth", "time_steps"):
                    if required not in slot:
                        raise ConfigError(f"protocol.matrix.{group}[{index}] missing {required}")
                _str(slot["experiment"], f"protocol.matrix.{group}[{index}].experiment", ("E1",))
                _str(slot["dataset"], f"protocol.matrix.{group}[{index}].dataset", ("cifar10", "cifar100", "cifar10dvs"))
                _int(slot["depth"], f"protocol.matrix.{group}[{index}].depth", 1)
                _int(slot["time_steps"], f"protocol.matrix.{group}[{index}].time_steps", 1)
                if "seeds" in slot:
                    slot_seeds = slot["seeds"]
                    if (
                        not isinstance(slot_seeds, Sequence)
                        or isinstance(slot_seeds, (str, bytes))
                        or not slot_seeds
                    ):
                        raise ConfigError(
                            f"protocol.matrix.{group}[{index}].seeds must be a non-empty sequence"
                        )
                    parsed_slot_seeds = [
                        _int(value, f"protocol.matrix.{group}[{index}].seeds[]", 0)
                        for value in slot_seeds
                    ]
                    if len(set(parsed_slot_seeds)) != len(parsed_slot_seeds):
                        raise ConfigError(
                            f"protocol.matrix.{group}[{index}].seeds must be unique"
                        )
                    if any(seed not in seeds for seed in parsed_slot_seeds):
                        raise ConfigError(
                            f"protocol.matrix.{group}[{index}].seeds must be drawn from top-level seeds"
                        )
        primary = tuple(_matrix_group_tuple(slot) for slot in matrix.get("primary", ()))
        _require_exact(primary, expected_groups, "protocol.matrix.primary")
        if version == 2:
            slot = matrix["primary"][0]
            _require_exact(
                tuple(slot.get("seeds", ())),
                expected_seeds,
                "protocol.matrix.primary[0].seeds",
            )
    if "analysis" in raw:
        _validate_analysis_mapping(
            _check_mapping(raw["analysis"], "protocol.analysis"), "protocol.analysis"
        )
    if "protocol_status" in raw:
        _validate_protocol_status(
            _check_mapping(raw["protocol_status"], "protocol.protocol_status")
        )
        if version == 2 and raw["protocol_status"].get("frozen") is not False:
            raise ConfigError("The v2 pilot protocol must remain unfrozen")
    if "benchmark" in raw:
        _validate_benchmark_mapping(
            _check_mapping(raw["benchmark"], "protocol.benchmark"), "protocol.benchmark"
        )
    return normalized


def load_protocol(path: str | Path) -> Dict[str, Any]:
    return validate_protocol(_read_yaml(path))


def _validate_data_mapping(data: Mapping[str, Any], name: str) -> None:
    allowed = {
        "dataset", "name", "root", "num_classes", "in_channels", "val_fraction", "split_seed",
        "split_manifest", "frames_path", "labels_path", "dvs_time_bins", "normalize", "augment",
        "autoaugment",
        "dvs_height", "dvs_width", "dvs_resize_mode", "dvs_test_fraction", "dvs_split_seed",
        "test_frames_path", "test_labels_path", "download", "num_workers", "pin_memory", "time_steps",
    }
    _check_keys(data, allowed, name)
    if "dataset" in data:
        _str(data["dataset"], f"{name}.dataset")
    if "name" in data:
        _str(data["name"], f"{name}.name")
    for key in ("num_classes", "in_channels", "dvs_time_bins", "dvs_height", "dvs_width", "time_steps"):
        if key in data:
            _int(data[key], f"{name}.{key}", 1)
    for key in ("split_seed", "dvs_split_seed", "num_workers"):
        if key in data:
            _int(data[key], f"{name}.{key}", 0)
    if "val_fraction" in data:
        value = _float(data["val_fraction"], f"{name}.val_fraction", 0.0, 0.9)
        if value == 0.0:
            raise ConfigError(f"{name}.val_fraction must be > 0")
    if "dvs_test_fraction" in data:
        fraction = _float(data["dvs_test_fraction"], f"{name}.dvs_test_fraction", 0.0, 0.9)
        if fraction == 0.0:
            raise ConfigError(f"{name}.dvs_test_fraction must be > 0")
    if "dvs_resize_mode" in data:
        _str(data["dvs_resize_mode"], f"{name}.dvs_resize_mode", ("nearest",))
    for key in ("root", "split_manifest", "frames_path", "labels_path", "test_frames_path", "test_labels_path"):
        if key in data and data[key] is not None and not isinstance(data[key], str):
            raise ConfigError(f"{name}.{key} must be a string or null")
    for key in ("normalize", "augment", "autoaugment", "download", "pin_memory"):
        if key in data and not isinstance(data[key], bool):
            raise ConfigError(f"{name}.{key} must be boolean")


def _validate_neuron_cfg(value: Mapping[str, Any], name: str) -> None:
    allowed = {
        "tau", "decay", "threshold", "v_threshold", "v_th", "v_rest", "v_reset",
        "width", "surrogate_width", "delta_min",
    }
    _check_keys(value, allowed, name)
    if "tau" in value and "decay" in value and float(value["tau"]) != float(value["decay"]):
        raise ConfigError(f"{name} specifies conflicting tau and decay")
    for key in ("tau", "decay"):
        if key in value:
            tau = _float(value[key], f"{name}.{key}", 0.0, 1.0)
            if tau == 0.0:
                raise ConfigError(f"{name}.{key} must be in (0, 1]")
    for key in ("threshold", "v_threshold", "v_th", "v_rest", "v_reset"):
        if key in value:
            _float(value[key], f"{name}.{key}")
    for key in ("width", "surrogate_width", "delta_min"):
        if key in value:
            parsed = _float(value[key], f"{name}.{key}", 0.0)
            if parsed == 0.0:
                raise ConfigError(f"{name}.{key} must be positive")
    width = value.get("width", value.get("surrogate_width"))
    delta = value.get("delta_min")
    if width is not None and delta is not None and float(width) <= float(delta):
        raise ConfigError(f"{name}.width must exceed delta_min")


def _validate_model_mapping(model: Mapping[str, Any], name: str, allow_condition: bool = True) -> None:
    allowed = {
        "condition", "topology", "neuron", "depth", "time_steps", "timesteps", "num_classes",
        "in_channels", "base_channels", "neuron_cfg",
    }
    _check_keys(model, allowed, name)
    if allow_condition and "condition" in model:
        _str(model["condition"], f"{name}.condition", SUPPORTED_CONDITIONS)
    if "topology" in model:
        _str(model["topology"], f"{name}.topology", ("spiking_resnet", "ms_resnet"))
    if "neuron" in model:
        _str(model["neuron"], f"{name}.neuron", ("lif", "ta_lif", "ta-lif"))
    for key in ("depth", "time_steps", "timesteps", "num_classes", "in_channels", "base_channels"):
        if key in model:
            _int(model[key], f"{name}.{key}", 1)
    if "depth" in model and int(model["depth"]) not in (20, 56):
        raise ConfigError(f"{name}.depth must be 20 or 56 for the frozen CIFAR protocol")
    if "time_steps" in model and "timesteps" in model and model["time_steps"] != model["timesteps"]:
        raise ConfigError(f"{name} specifies conflicting time_steps and timesteps")
    if "neuron_cfg" in model and not isinstance(model["neuron_cfg"], Mapping):
        raise ConfigError(f"{name}.neuron_cfg must be a mapping")
    if "neuron_cfg" in model:
        _validate_neuron_cfg(model["neuron_cfg"], f"{name}.neuron_cfg")


def _validate_optimizer_mapping(opt: Mapping[str, Any], name: str) -> None:
    allowed = {
        "epochs", "batch_size", "lr", "momentum", "weight_decay", "milestones", "gamma",
        "ta_start_fraction", "ta_lr_scale", "ta_weight_decay", "grad_clip",
        "label_smoothing", "cutmix_alpha", "cutmix_probability",
    }
    _check_keys(opt, allowed, name)
    for key in ("epochs", "batch_size"):
        if key in opt:
            _int(opt[key], f"{name}.{key}", 1)
    for key in (
        "lr", "momentum", "weight_decay", "gamma", "ta_start_fraction", "ta_lr_scale",
        "ta_weight_decay", "label_smoothing", "cutmix_alpha", "cutmix_probability",
    ):
        if key in opt:
            _float(opt[key], f"{name}.{key}", 0.0)
    if "momentum" in opt and float(opt["momentum"]) >= 1:
        raise ConfigError(f"{name}.momentum must be < 1")
    if "ta_start_fraction" in opt and float(opt["ta_start_fraction"]) > 1:
        raise ConfigError(f"{name}.ta_start_fraction must be <= 1")
    for key in ("label_smoothing", "cutmix_probability"):
        if key in opt and float(opt[key]) > 1:
            raise ConfigError(f"{name}.{key} must be <= 1")
    if "milestones" in opt:
        values = opt["milestones"]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ConfigError(f"{name}.milestones must be a sequence")
        parsed = [_int(v, f"{name}.milestones[]", 1) for v in values]
        if parsed != sorted(set(parsed)):
            raise ConfigError(f"{name}.milestones must be strictly increasing")
        if "epochs" in opt and any(value >= int(opt["epochs"]) for value in parsed):
            raise ConfigError(f"{name}.milestones must be smaller than epochs")
    if "grad_clip" in opt and opt["grad_clip"] is not None:
        _float(opt["grad_clip"], f"{name}.grad_clip", 0.0)


def _validate_analysis_mapping(analysis: Mapping[str, Any], name: str) -> None:
    allowed = {
        "collect_activity", "accuracy_scale", "validation_accuracy_thresholds",
        "convergence_epoch_rule", "nonconvergence_rule", "failure_rule",
        "alpha", "confidence_level", "primary_interaction_test",
        "multiplicity_correction", "multiplicity_family",
        "interaction_practical_threshold_pp", "efficiency_noninferiority_margins",
        "bootstrap", "efficiency_assessment", "efficiency_confidence_level",
        "efficiency_comparison",
        "run_handling", "wording_gates",
        "jacobian_probes", "jacobian_seed", "jacobian_time_index",
        "diagnostic_batches", "diagnostic_batch_size",
    }
    _check_keys(analysis, allowed, name)
    missing = sorted(allowed - set(analysis))
    if missing:
        raise ConfigError(f"{name} is missing required key(s): {', '.join(missing)}")
    if "collect_activity" in analysis and not isinstance(analysis["collect_activity"], bool):
        raise ConfigError(f"{name}.collect_activity must be boolean")
    _require_exact(analysis.get("collect_activity"), False, f"{name}.collect_activity")
    _require_exact(analysis.get("accuracy_scale"), "proportion", f"{name}.accuracy_scale")
    _require_exact(
        analysis.get("convergence_epoch_rule"),
        "first_validation_epoch_at_or_above_threshold",
        f"{name}.convergence_epoch_rule",
    )
    _require_exact(
        analysis.get("nonconvergence_rule"),
        "not_reached_by_final_epoch",
        f"{name}.nonconvergence_rule",
    )
    for key in ("jacobian_probes", "jacobian_seed", "diagnostic_batches", "diagnostic_batch_size"):
        if key in analysis:
            _int(analysis[key], f"{name}.{key}", 1 if key != "jacobian_seed" else 0)
    _require_exact(analysis.get("diagnostic_batches"), 1, f"{name}.diagnostic_batches")
    _require_exact(analysis.get("diagnostic_batch_size"), 8, f"{name}.diagnostic_batch_size")
    _require_exact(analysis.get("jacobian_probes"), 8, f"{name}.jacobian_probes")
    _require_exact(analysis.get("jacobian_seed"), 20_260_719, f"{name}.jacobian_seed")
    if "jacobian_time_index" in analysis:
        _int(analysis["jacobian_time_index"], f"{name}.jacobian_time_index")
    _require_exact(analysis.get("jacobian_time_index"), -1, f"{name}.jacobian_time_index")
    if "failure_rule" in analysis:
        _str(analysis["failure_rule"], f"{name}.failure_rule")
    _require_exact(analysis.get("alpha"), 0.05, f"{name}.alpha")
    _require_exact(analysis.get("confidence_level"), 0.95, f"{name}.confidence_level")
    interaction_test = _check_mapping(
        analysis.get("primary_interaction_test"), f"{name}.primary_interaction_test"
    )
    _check_keys(
        interaction_test,
        PRESPECIFIED_PRIMARY_INTERACTION_TEST,
        f"{name}.primary_interaction_test",
    )
    _require_exact(
        dict(interaction_test),
        PRESPECIFIED_PRIMARY_INTERACTION_TEST,
        f"{name}.primary_interaction_test",
    )
    _require_exact(
        analysis.get("multiplicity_correction"),
        "holm_two_confirmatory_tests",
        f"{name}.multiplicity_correction",
    )
    _require_exact(
        analysis.get("multiplicity_family"),
        "two_confirmatory_interaction_tests",
        f"{name}.multiplicity_family",
    )
    if "interaction_practical_threshold_pp" in analysis and analysis["interaction_practical_threshold_pp"] is not None:
        _float(
            analysis["interaction_practical_threshold_pp"],
            f"{name}.interaction_practical_threshold_pp",
            0.0,
        )
    _require_exact(
        analysis.get("interaction_practical_threshold_pp"),
        0.50,
        f"{name}.interaction_practical_threshold_pp",
    )
    thresholds = analysis.get("validation_accuracy_thresholds")
    if thresholds is not None:
        thresholds = _check_mapping(thresholds, f"{name}.validation_accuracy_thresholds")
        _check_keys(thresholds, {"cifar10", "cifar100", "cifar10dvs"}, f"{name}.validation_accuracy_thresholds")
        missing_thresholds = sorted(set(PRESPECIFIED_THRESHOLDS) - set(thresholds))
        if missing_thresholds:
            raise ConfigError(
                f"{name}.validation_accuracy_thresholds is missing: {', '.join(missing_thresholds)}"
            )
        for dataset, value in thresholds.items():
            if value is not None:
                _float(value, f"{name}.validation_accuracy_thresholds.{dataset}", 0.0, 1.0)
        _require_exact(
            dict(thresholds),
            PRESPECIFIED_THRESHOLDS,
            f"{name}.validation_accuracy_thresholds",
        )
    bootstrap = _check_mapping(analysis.get("bootstrap"), f"{name}.bootstrap")
    _check_keys(bootstrap, PRESPECIFIED_BOOTSTRAP, f"{name}.bootstrap")
    _require_exact(dict(bootstrap), PRESPECIFIED_BOOTSTRAP, f"{name}.bootstrap")
    _require_exact(
        analysis.get("efficiency_assessment"),
        "descriptive",
        f"{name}.efficiency_assessment",
    )
    _require_exact(
        analysis.get("efficiency_confidence_level"),
        0.95,
        f"{name}.efficiency_confidence_level",
    )
    efficiency_comparison = _check_mapping(
        analysis.get("efficiency_comparison"), f"{name}.efficiency_comparison"
    )
    _check_keys(
        efficiency_comparison,
        PRESPECIFIED_EFFICIENCY_COMPARISON,
        f"{name}.efficiency_comparison",
    )
    _require_exact(
        dict(efficiency_comparison),
        PRESPECIFIED_EFFICIENCY_COMPARISON,
        f"{name}.efficiency_comparison",
    )
    margins = analysis.get("efficiency_noninferiority_margins")
    if margins is not None:
        margins = _check_mapping(margins, f"{name}.efficiency_noninferiority_margins")
        _check_keys(
            margins,
            PRESPECIFIED_EFFICIENCY_MARGINS,
            f"{name}.efficiency_noninferiority_margins",
        )
        for metric, value in margins.items():
            _str(metric, f"{name}.efficiency_noninferiority_margins key")
            if value is not None:
                _float(value, f"{name}.efficiency_noninferiority_margins.{metric}", 0.0)
        _require_exact(
            dict(margins),
            PRESPECIFIED_EFFICIENCY_MARGINS,
            f"{name}.efficiency_noninferiority_margins",
        )
    run_handling = _check_mapping(analysis.get("run_handling"), f"{name}.run_handling")
    _check_keys(run_handling, PRESPECIFIED_RUN_HANDLING, f"{name}.run_handling")
    _require_exact(dict(run_handling), PRESPECIFIED_RUN_HANDLING, f"{name}.run_handling")
    wording_gates = _check_mapping(analysis.get("wording_gates"), f"{name}.wording_gates")
    _check_keys(wording_gates, PRESPECIFIED_WORDING_GATES, f"{name}.wording_gates")
    _require_exact(dict(wording_gates), PRESPECIFIED_WORDING_GATES, f"{name}.wording_gates")


def _validate_benchmark_mapping(benchmark: Mapping[str, Any], name: str) -> None:
    allowed = {*PRESPECIFIED_BENCHMARK, "energy_model"}
    _check_keys(benchmark, allowed, name)
    missing = sorted(allowed - set(benchmark))
    if missing:
        raise ConfigError(f"{name} is missing required key(s): {', '.join(missing)}")
    _str(benchmark["device_type"], f"{name}.device_type", ("cuda",))
    _str(benchmark["precision"], f"{name}.precision", ("float32",))
    sizes = benchmark["batch_sizes"]
    if not isinstance(sizes, Sequence) or isinstance(sizes, (str, bytes)):
        raise ConfigError(f"{name}.batch_sizes must be a sequence")
    for index, size in enumerate(sizes):
        _int(size, f"{name}.batch_sizes[{index}]", 1)
    _int(benchmark["warmup_iterations"], f"{name}.warmup_iterations", 0)
    _int(benchmark["timed_iterations"], f"{name}.timed_iterations", 1)
    _str(benchmark["timing_method"], f"{name}.timing_method")
    _str(benchmark["input_residency"], f"{name}.input_residency")
    core = {key: benchmark[key] for key in PRESPECIFIED_BENCHMARK}
    _require_exact(core, PRESPECIFIED_BENCHMARK, name)
    energy = _check_mapping(benchmark.get("energy_model"), f"{name}.energy_model")
    _check_keys(energy, ENERGY_MODEL_FIELDS, f"{name}.energy_model")
    missing_energy = sorted(set(ENERGY_MODEL_FIELDS) - set(energy))
    if missing_energy:
        raise ConfigError(
            f"{name}.energy_model is missing required key(s): {', '.join(missing_energy)}"
        )
    status = _str(
        energy.get("status"),
        f"{name}.energy_model.status",
        ("not_assessed", "modeled"),
    )
    path = energy.get("constants_path")
    digest = energy.get("constants_sha256")
    source = energy.get("source")
    if status == "not_assessed":
        _require_exact(path, None, f"{name}.energy_model.constants_path")
        _require_exact(digest, "none", f"{name}.energy_model.constants_sha256")
        _require_exact(source, "none", f"{name}.energy_model.source")
        return
    _str(path, f"{name}.energy_model.constants_path")
    if Path(str(path)).is_absolute() or ".." in Path(str(path)).parts:
        raise ConfigError(f"{name}.energy_model.constants_path must be repository-relative")
    _str(digest, f"{name}.energy_model.constants_sha256")
    if len(str(digest)) != 64 or any(
        character not in "0123456789abcdef" for character in str(digest)
    ):
        raise ConfigError(f"{name}.energy_model.constants_sha256 must be lowercase SHA-256")
    _str(source, f"{name}.energy_model.source")
    if str(source).strip().lower() == "none":
        raise ConfigError(f"{name}.energy_model.source must identify the cited model")


def _validate_run_analysis_mapping(
    analysis: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    data: DataConfig,
    model: ModelConfig,
    runtime: RuntimeConfig,
) -> None:
    name = "run.analysis"
    generated = dict(analysis)
    matrix_key = generated.pop("matrix_key", None)
    protocol_hash = generated.pop("protocol_hash", None)
    _validate_analysis_mapping(generated, name)
    if not isinstance(matrix_key, Sequence) or isinstance(matrix_key, (str, bytes)):
        raise ConfigError(f"{name}.matrix_key must be a five-element sequence")
    expected_key = [
        data.dataset,
        model.depth,
        model.time_steps,
        model.condition,
        runtime.seed,
    ]
    _require_exact(list(matrix_key), expected_key, f"{name}.matrix_key")
    if not isinstance(protocol_hash, str) or len(protocol_hash) != 64:
        raise ConfigError(f"{name}.protocol_hash must be a 64-character SHA-256 value")
    try:
        int(protocol_hash, 16)
    except ValueError as exc:
        raise ConfigError(f"{name}.protocol_hash must be hexadecimal") from exc
    if protocol:
        expected_hash = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
        _require_exact(protocol_hash, expected_hash, f"{name}.protocol_hash")


def _validate_protocol_status(status: Mapping[str, Any]) -> None:
    name = "protocol.protocol_status"
    _check_keys(status, {"frozen", "confirmed_by", "confirmed_at", "confirmations", "notes"}, name)
    if "frozen" in status and not isinstance(status["frozen"], bool):
        raise ConfigError(f"{name}.frozen must be boolean")
    for key in ("confirmed_by", "confirmed_at", "notes"):
        if key in status and status[key] is not None and not isinstance(status[key], str):
            raise ConfigError(f"{name}.{key} must be a string or null")
    confirmations = status.get("confirmations", {})
    confirmations = _check_mapping(confirmations, f"{name}.confirmations")
    _check_keys(confirmations, PROTOCOL_CONFIRMATION_FIELDS, f"{name}.confirmations")
    for key, value in confirmations.items():
        if not isinstance(value, bool):
            raise ConfigError(f"{name}.confirmations.{key} must be boolean")


def _validate_runtime_mapping(runtime: Mapping[str, Any], name: str) -> None:
    allowed = {
        "seed", "device", "output_dir", "run_id", "checkpoint_every", "log_every", "amp",
        "deterministic", "dry_run", "limit_batches", "resume", "num_workers", "pin_memory",
    }
    _check_keys(runtime, allowed, name)
    if "seed" in runtime:
        _int(runtime["seed"], f"{name}.seed", 0)
    for key in ("checkpoint_every", "log_every"):
        if key in runtime:
            _int(runtime[key], f"{name}.{key}", 1)
    if "limit_batches" in runtime and runtime["limit_batches"] is not None:
        _int(runtime["limit_batches"], f"{name}.limit_batches", 1)
    for key in ("device", "output_dir", "run_id", "resume"):
        if key in runtime and runtime[key] is not None and not isinstance(runtime[key], str):
            raise ConfigError(f"{name}.{key} must be a string or null")
    for key in ("amp", "deterministic", "dry_run"):
        if key in runtime and not isinstance(runtime[key], bool):
            raise ConfigError(f"{name}.{key} must be boolean")


def _section(raw: Mapping[str, Any], key: str, aliases: Sequence[str] = ()) -> Dict[str, Any]:
    value = raw.get(key)
    if value is None:
        for alias in aliases:
            if alias in raw:
                value = raw[alias]
                break
    if value is None:
        return {}
    return dict(_check_mapping(value, key))


def _resolve_data(raw: Mapping[str, Any], protocol: Mapping[str, Any]) -> DataConfig:
    section = _section(raw, "data", ("dataset",))
    # A run may use a named protocol dataset and override only its seed/path.
    name = section.get("dataset", section.get("name", "cifar10"))
    datasets = protocol.get("datasets", {})
    base = dict(protocol.get("data", {})) if isinstance(protocol.get("data", {}), Mapping) else {}
    if isinstance(datasets, Mapping) and name in datasets:
        base.update(dict(datasets.get(name, {})))
    base.update(section)
    dataset = str(base.get("dataset", base.get("name", name))).lower().replace("-", "")
    aliases = {"cifar10dvs": "cifar10dvs", "cifar10_dvs": "cifar10dvs", "cifar-10-dvs": "cifar10dvs"}
    dataset = aliases.get(dataset, dataset)
    if dataset not in {"cifar10", "cifar100", "cifar10dvs"}:
        raise ConfigError(f"Unsupported dataset {dataset!r}")
    defaults = DataConfig(dataset=dataset)
    values = dataclasses.asdict(defaults)
    values.update({k: v for k, v in base.items() if k in values})
    values["dataset"] = dataset
    if values["split_manifest"]:
        values["split_manifest"] = str(values["split_manifest"]).format(dataset=dataset, split_seed=values["split_seed"])
    return DataConfig(**values)


def _resolve_model(raw: Mapping[str, Any], protocol: Mapping[str, Any], data: DataConfig) -> ModelConfig:
    section = _section(raw, "model")
    base = dict(protocol.get("model", {})) if isinstance(protocol.get("model", {}), Mapping) else {}
    base.update(section)
    condition = str(raw.get("condition", base.get("condition", "C1"))
                    ).upper()
    if condition not in SUPPORTED_CONDITIONS:
        raise ConfigError(f"condition must be one of {SUPPORTED_CONDITIONS}, got {condition!r}")
    expected = CONDITION_SPECS[condition]
    topology = str(base.get("topology", expected["topology"]))
    neuron = str(base.get("neuron", expected["neuron"])).replace("-", "_")
    if topology != expected["topology"] or neuron != expected["neuron"]:
        raise ConfigError(f"{condition} requires topology={expected['topology']} and neuron={expected['neuron']}")
    time_steps = base.get("time_steps", base.get("timesteps", 6))
    values: Dict[str, Any] = {
        "condition": condition,
        "topology": topology,
        "neuron": neuron,
        "depth": int(base.get("depth", 20)),
        "time_steps": int(time_steps),
        "num_classes": int(base.get("num_classes", data.num_classes)),
        "in_channels": int(base.get("in_channels", data.in_channels)),
        "base_channels": int(base.get("base_channels", 16)),
        "neuron_cfg": dict(base.get("neuron_cfg", {})),
    }
    _validate_model_mapping(values, "model")
    return ModelConfig(**values)


def _resolve_optimizer(raw: Mapping[str, Any], protocol: Mapping[str, Any]) -> OptimizerConfig:
    base = dict(protocol.get("optimizer", {})) if isinstance(protocol.get("optimizer", {}), Mapping) else {}
    base.update(_section(raw, "optimizer"))
    values = dataclasses.asdict(OptimizerConfig())
    values.update({k: v for k, v in base.items() if k in values})
    if isinstance(values.get("milestones"), list):
        values["milestones"] = tuple(values["milestones"])
    _validate_optimizer_mapping(values, "optimizer")
    return OptimizerConfig(**values)


def _resolve_runtime(raw: Mapping[str, Any], protocol: Mapping[str, Any], run_id: str) -> RuntimeConfig:
    base = dict(protocol.get("runtime", {})) if isinstance(protocol.get("runtime", {}), Mapping) else {}
    base.update(_section(raw, "runtime"))
    values = dataclasses.asdict(RuntimeConfig(run_id=run_id))
    values.update({k: v for k, v in base.items() if k in values})
    values["run_id"] = str(raw.get("run_id", values.get("run_id", run_id)))
    if isinstance(values.get("limit_batches"), bool):
        raise ConfigError("runtime.limit_batches must be an integer or null")
    _validate_runtime_mapping(values, "runtime")
    return RuntimeConfig(**values)


def validate_run_mapping(raw: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> RunConfig:
    """Resolve one run mapping and reject unknown keys at every section."""

    raw = _check_mapping(raw, "run")
    allowed = {
        "protocol_version", "experiment", "condition", "run_id", "data", "dataset", "model",
        "optimizer", "runtime", "seed", "final_test", "analysis",
    }
    _check_keys(raw, allowed, "run")
    protocol = {} if protocol is None else dict(protocol)
    version = _int(raw.get("protocol_version", protocol.get("protocol_version", 1)), "protocol_version", 1)
    if protocol:
        _require_exact(version, protocol.get("protocol_version"), "protocol_version")
    experiment = _str(raw.get("experiment", "E1"), "experiment")
    data = _resolve_data(raw, protocol)
    # Top-level seed is a deliberate shorthand for generated matrix files.
    raw_for_runtime = dict(raw)
    runtime = _section(raw_for_runtime, "runtime")
    if "seed" in raw and "seed" in runtime:
        raise ConfigError("Specify seed either at run level or runtime.seed, not both")
    if "seed" in raw:
        runtime["seed"] = raw["seed"]
        raw_for_runtime["runtime"] = runtime
    run_id = str(raw.get("run_id", f"{experiment}_{raw.get('condition', 'C1')}_s{raw.get('seed', 11)}"))
    model = _resolve_model(raw, protocol, data)
    optimizer = _resolve_optimizer(raw, protocol)
    runtime_cfg = _resolve_runtime(raw_for_runtime, protocol, run_id)
    final_test = raw.get("final_test", False)
    if not isinstance(final_test, bool):
        raise ConfigError("final_test must be boolean")
    analysis = raw.get("analysis", {})
    if not isinstance(analysis, Mapping):
        raise ConfigError("analysis must be a mapping")
    if protocol or analysis:
        _validate_run_analysis_mapping(
            analysis,
            protocol=protocol,
            data=data,
            model=model,
            runtime=runtime_cfg,
        )
    return RunConfig(
        protocol_version=version,
        experiment=experiment,
        data=data,
        model=model,
        optimizer=optimizer,
        runtime=runtime_cfg,
        final_test=final_test,
        analysis=dict(analysis),
    )


def load_run_config(path: str | Path, protocol_path: str | Path | None = None) -> RunConfig:
    raw = _read_yaml(path)
    protocol = load_protocol(protocol_path) if protocol_path else {}
    return validate_run_mapping(raw, protocol)


def _dataset_specs(protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Resolve declared dataset/depth/time-step slots and their seed blocks."""

    datasets = protocol.get("datasets", {})
    if not isinstance(datasets, Mapping):
        datasets = {}

    def spec(name: str, depth: int, time_steps: int, experiment: str) -> Dict[str, Any]:
        item = dict(datasets.get(name, {})) if isinstance(datasets.get(name, {}), Mapping) else {}
        item.update({"dataset": name, "depth": depth, "time_steps": time_steps, "experiment": experiment})
        return item

    matrix = protocol.get("matrix", {})
    if not isinstance(matrix, Mapping):
        raise ConfigError("protocol.matrix must be a mapping")
    slots: List[Dict[str, Any]] = []
    for raw in matrix.get("primary", []):
        dataset = str(raw["dataset"])
        depth = int(raw["depth"])
        time_steps = int(raw["time_steps"])
        experiment = str(raw["experiment"])
        resolved = spec(dataset, depth, time_steps, experiment)
        resolved["_matrix_seeds"] = list(raw.get("seeds", protocol["seeds"]))
        slots.append(resolved)
    return slots


def generate_run_matrix(protocol: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Generate the versioned C1-C4 matrix without mixing protocol generations."""

    protocol = validate_protocol(protocol)
    protocol_hash = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
    runs: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()
    for slot in _dataset_specs(protocol):
        slot_seeds = list(slot["_matrix_seeds"])
        pairs = (
            ((condition, seed) for condition in SUPPORTED_CONDITIONS for seed in slot_seeds)
            if protocol["protocol_version"] == 1
            else ((condition, seed) for seed in slot_seeds for condition in SUPPORTED_CONDITIONS)
        )
        for condition, seed in pairs:
                key = (slot["dataset"], slot["depth"], slot["time_steps"], condition, seed)
                if key in seen:
                    continue
                seen.add(key)
                run_id = f"{slot['experiment']}_{slot['dataset']}_d{slot['depth']}_t{slot['time_steps']}_{condition}_s{seed}"
                data = {
                    k: v
                    for k, v in slot.items()
                    if k not in {"depth", "time_steps", "experiment", "_matrix_seeds"}
                }
                model = {
                    "condition": condition,
                    "topology": CONDITION_SPECS[condition]["topology"],
                    "neuron": CONDITION_SPECS[condition]["neuron"],
                    "depth": slot["depth"],
                    "time_steps": slot["time_steps"],
                }
                analysis = copy.deepcopy(dict(protocol.get("analysis", {})))
                analysis.update({"matrix_key": list(key), "protocol_hash": protocol_hash})
                runs.append({
                    "protocol_version": protocol["protocol_version"],
                    "experiment": slot["experiment"],
                    "run_id": run_id,
                    "condition": condition,
                    "seed": seed,
                    "data": data,
                    "model": model,
                    "optimizer": dict(protocol.get("optimizer", {})),
                    "runtime": {"output_dir": protocol.get("output_root", "results")},
                    "final_test": False,
                    "analysis": analysis,
                })
    expected_run_count = (
        EXPECTED_RUN_COUNT
        if protocol["protocol_version"] == 1
        else sum(len(slot["_matrix_seeds"]) for slot in _dataset_specs(protocol))
        * len(SUPPORTED_CONDITIONS)
    )
    if len(runs) != expected_run_count:
        raise ConfigError(
            f"Expected {expected_run_count} unique runs, generated {len(runs)}"
        )
    if any(run["experiment"] != "E1" for run in runs):
        raise ConfigError("The confirmatory matrix must contain only E1 runs")
    return runs


def config_from_mapping(raw: Mapping[str, Any], protocol: Mapping[str, Any] | None = None) -> RunConfig:
    """Alias used by scripts and external callers."""

    return validate_run_mapping(raw, protocol)
