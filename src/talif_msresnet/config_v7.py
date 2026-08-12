"""Strict protocol-v7 contracts for the TA-LIF mechanism study.

V7 is an isolated continuation after the consumed, non-reportable V6 health
failure.  It preserves the V6 scientific design while replacing every
execution identity and adding explicit health-batch integrity requirements.
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
    V2_PILOT_SEEDS,
    V2R2_PILOT_SEEDS,
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
from .config_v6 import (
    V6_ACTIVE_CONDITIONS,
    V6_ANALYSIS_CONTRACT,
    V6_BENCHMARK_CONTRACT,
    V6_CIFAR100_CONTRACT,
    V6_CIFAR100_PROVENANCE_CONTRACT,
    V6_CONDITIONAL_EXTENSION,
    V6_CONDITION_SPECS,
    V6_DATA_CONTRACT,
    V6_MODEL_CONTRACT,
    V6_OPERATION_MODE_BY_CONDITION,
    V6_OPTIMIZER_CONTRACT,
    V6_PILOT_ACCEPTANCE,
    V6_ROUTE_MATCHED_CONDITIONS,
    V6_RUNTIME_CONTRACT,
    V6_TA_PARAMETER_CONDITIONS,
    expected_v6_seed_records,
)
from .pathing import artifact_path_reference
from .utils import sha256_file

V7_PROFILE = "cifar100_mechanism_core"
V7_ACTIVE_CONDITIONS: tuple[str, ...] = tuple(V6_ACTIVE_CONDITIONS)
V7_CONDITION_SPECS: dict[str, dict[str, str]] = copy.deepcopy(V6_CONDITION_SPECS)
V7_TA_PARAMETER_CONDITIONS = frozenset(V6_TA_PARAMETER_CONDITIONS)
V7_ROUTE_MATCHED_CONDITIONS = frozenset(V6_ROUTE_MATCHED_CONDITIONS)
V7_OPERATION_MODE_BY_CONDITION = copy.deepcopy(V6_OPERATION_MODE_BY_CONDITION)

V7_SEED_DERIVATION_NAMESPACE = "ta-lif-msresnet/v7-mechanism-study/2026-08-11"
V7_HEALTH_SEED = 1068798027
V7_PILOT_SEED = 1673127435
V7_FORMAL_SEEDS: tuple[int, ...] = (
    404085484,
    937711809,
    959898395,
    895247195,
    373375701,
    1221790045,
    832924126,
    999491910,
)
V7_BOOTSTRAP_SEEDS: dict[str, int] = {
    "M4-M0": 923950155,
    "M4-M2": 656484872,
    "M4-M3": 1561781101,
    "M1-M0": 870432304,
    "PLIF-M0": 1414912849,
    "M4-PLIF": 38145984,
}
V7_BENCHMARK_SEED = 1270794041

V7_CIFAR10_HEALTH_SEED = 443941899
V7_CIFAR10_PILOT_SEED = 1839671317
V7_CIFAR10_FORMAL_SEEDS: tuple[int, ...] = (
    504476786,
    319993704,
    79036247,
    1720934125,
    145388398,
)
V7_CIFAR10_BOOTSTRAP_SEEDS: dict[str, int] = {
    "M4-M0": 1691006615,
    "M4-M3": 1336297218,
}
V7_CIFAR10_BENCHMARK_SEED = 216786900

V7_OUTPUT_ROOT = "results/formal_v7_mechanism"
V7_ARTIFACT_PATHS: dict[str, str] = {
    "protocol": "configs/protocol_v7_mechanism.yaml",
    "signoff": "PREREGISTRATION_SIGNOFF_V7_MECHANISM.md",
    "formal_matrix": "configs/v7_mechanism_generated",
    "pilot_matrix": "configs/v7_mechanism_pilot_generated",
    "freeze_manifest": "FREEZE_MANIFEST_V7_MECHANISM.json",
    "formal_results": V7_OUTPUT_ROOT,
    "pilot_results": "results/pilot/v7_mechanism",
    "analysis_results": "results/analysis/v7_mechanism",
    "benchmark_results": "results/benchmark/v7_mechanism",
}
V7_PROTOCOL_CONFIRMATION_FIELDS: tuple[str, ...] = (
    "v5_evidence_remains_immutable_and_out_of_scope",
    "v6_failure_is_immutable_and_all_v6_seeds_retired",
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

# These contracts are deliberately copied from V6.  V7 changes execution
# identity and health-gate integrity mechanics, not the scientific design.
V7_MODEL_CONTRACT = copy.deepcopy(V6_MODEL_CONTRACT)
V7_OPTIMIZER_CONTRACT = copy.deepcopy(V6_OPTIMIZER_CONTRACT)
V7_RUNTIME_CONTRACT = copy.deepcopy(V6_RUNTIME_CONTRACT)
V7_DATA_CONTRACT = copy.deepcopy(V6_DATA_CONTRACT)
V7_CIFAR100_CONTRACT = copy.deepcopy(V6_CIFAR100_CONTRACT)
V7_CIFAR100_PROVENANCE_CONTRACT = copy.deepcopy(V6_CIFAR100_PROVENANCE_CONTRACT)
V7_CIFAR100_PROVENANCE_CONTRACT.update(
    {
        "archive_path": "data/cifar100/cifar-100-python.tar.gz",
        "archive_bytes": 169001437,
        "archive_sha256": (
            "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7"
        ),
        "train_pickle_path": "data/cifar100/cifar-100-python/train",
        "train_pickle_bytes": 155249918,
        "train_pickle_sha256": (
            "735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b"
        ),
        "meta_pickle_path": "data/cifar100/cifar-100-python/meta",
        "meta_pickle_bytes": 1473,
        "meta_pickle_sha256": (
            "a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9"
        ),
    }
)
V7_MATRIX_CONTRACT: dict[str, Any] = {
    "primary": [
        {
            "experiment": "E9",
            "dataset": "cifar100",
            "depth": 20,
            "time_steps": 6,
            "seeds": list(V7_FORMAL_SEEDS),
        }
    ]
}

V7_BENCHMARK_CONTRACT = copy.deepcopy(V6_BENCHMARK_CONTRACT)
V7_BENCHMARK_CONTRACT["input_seed"] = V7_BENCHMARK_SEED

V7_PILOT_ACCEPTANCE = copy.deepcopy(V6_PILOT_ACCEPTANCE)
V7_PILOT_ACCEPTANCE.update(
    {
        "identity": "v7_cifar100_six_condition_e120_mechanism",
        "health_seed": V7_HEALTH_SEED,
        "pilot_seed": V7_PILOT_SEED,
        "health_output": (
            "results/pilot/v7_mechanism/health_cifar100_s1068798027.json"
        ),
        "attempt_receipt": (
            "results/pilot/v7_mechanism/health_cifar100_s1068798027.attempt.json"
        ),
        "pilot_output_root": (
            "results/pilot/v7_mechanism/cifar100_s1673127435"
        ),
        "pilot_plan": (
            "environment/v7_mechanism_pilot_cifar100_s1673127435.json"
        ),
        "validation_output": "results/pilot/v7_mechanism/validation.json",
    }
)
V7_PILOT_ACCEPTANCE["health"].update(
    {
        "expected_adaptive_update_by_condition": {
            "M0": False,
            "M1": False,
            "M2": True,
            "M3": True,
            "M4": True,
            "PLIF": True,
        },
        "fixed_batch_identity": {
            "seed_source": "same_frozen_health_seed_for_preclaim_and_execution",
            "rng_reset": (
                "isolated_cpu_torch_rng_reset_immediately_before_each_loader_construction"
            ),
            "digest": "sha256_over_fixed_input_and_target_tensors",
            "required_relation": "preclaim_sha256_exactly_equals_execution_sha256",
            "mismatch_action": "block_before_any_condition_probe",
        },
    }
)

V7_ANALYSIS_CONTRACT = copy.deepcopy(V6_ANALYSIS_CONTRACT)
V7_ANALYSIS_CONTRACT["bootstrap"]["seeds"] = dict(V7_BOOTSTRAP_SEEDS)

V7_CONDITIONAL_EXTENSION = copy.deepcopy(V6_CONDITIONAL_EXTENSION)
V7_CONDITIONAL_EXTENSION.update(
    {
        "formal_seeds": list(V7_CIFAR10_FORMAL_SEEDS),
        "health_seed": V7_CIFAR10_HEALTH_SEED,
        "pilot_seed": V7_CIFAR10_PILOT_SEED,
        "bootstrap_seeds": dict(V7_CIFAR10_BOOTSTRAP_SEEDS),
        "benchmark_seed": V7_CIFAR10_BENCHMARK_SEED,
        "results_root": "results/formal_v7_cifar10_boundary",
    }
)

V6_CONSUMED_HEALTH_SEED = 1707261715
V6_RETIRED_UNUSED_SEEDS: tuple[int, ...] = tuple(
    int(record["value"])
    for record in expected_v6_seed_records()
    if int(record["value"]) != V6_CONSUMED_HEALTH_SEED
)
V7_PREDECESSOR_TERMINATION: dict[str, Any] = {
    "protocol_version": 6,
    "status": "FAIL_IMMUTABLE",
    "termination_record": "V6_MECHANISM_TERMINATION.md",
    "canonical_commit": "d83857786e2470bccfd2dac52c8b4e50b0f00b6f",
    "canonical_protocol_sha256": (
        "c13b9cc568c9a26303c39de80447b1ac1048f8c1d46add6c9f747bad7691fecf"
    ),
    "canonical_attempt_sha256": (
        "adbe836d70b9b04c1afc7b2d6e9ca24f7dc426078baf669bbadef7bfcbf71a41"
    ),
    "canonical_report_sha256": (
        "3ce50865947dfa53ed3072ce24c88cbb3d7bbc9383c4a52172ed74b4908da55d"
    ),
    "canonical_log_sha256": (
        "9194eed41e6e7b6da5fc02767bd6446040b77a11fe5642d92de7315220caa66d"
    ),
    "canonical_exit_sha256": (
        "4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865"
    ),
    "consumed_health_seed": V6_CONSUMED_HEALTH_SEED,
    "consumed_health_seed_disposition": (
        "CONSUMED_NONREPORTING_V6_HEALTH_NEVER_RETRY"
    ),
    "retired_unused_seed_disposition": "RESERVED_UNUSED_RETIRED_WITH_V6",
    "retired_unused_seeds": list(V6_RETIRED_UNUSED_SEEDS),
    "evidence_copy_into_v7_roots": "forbidden",
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
    payload = f"{V7_SEED_DERIVATION_NAMESPACE}|{label}|{counter}".encode()
    digest = hashlib.sha256(payload).digest()
    return {
        "label": label,
        "counter": counter,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "value": int.from_bytes(digest[:8], "big") & 0x7FFFFFFF,
    }


def expected_v7_seed_records() -> list[dict[str, Any]]:
    labels = ["health|cifar100|1", "pilot|cifar100|1"]
    labels.extend(f"formal|cifar100|{index}" for index in range(1, 9))
    labels.extend(
        f"bootstrap|cifar100|{name}"
        for name in ("m4-m0", "m4-m2", "m4-m3", "m1-m0", "plif-m0", "m4-plif")
    )
    labels.extend(["benchmark|cifar100|1", "health|cifar10|1", "pilot|cifar10|1"])
    labels.extend(f"formal|cifar10|{index}" for index in range(1, 6))
    labels.extend(f"bootstrap|cifar10|{name}" for name in ("m4-m0", "m4-m3"))
    labels.append("benchmark|cifar10|1")
    return [_record(label) for label in labels]


def all_v1_to_v6_seeds() -> frozenset[int]:
    """Return the complete historical exclusion set used by the V7 ledger."""

    return frozenset(
        {
            *V2_PILOT_SEEDS,
            *V2R2_PILOT_SEEDS,
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
            *(int(record["value"]) for record in expected_v6_seed_records()),
        }
    )


def _validate_seed_ledger(value: Any) -> None:
    ledger = _mapping(value, "protocol.seed_ledger")
    _keys(
        ledger,
        {"namespace", "algorithm", "collision_policy", "historical_seed_policy", "records"},
        "protocol.seed_ledger",
    )
    _exact(ledger["namespace"], V7_SEED_DERIVATION_NAMESPACE, "seed_ledger.namespace")
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
        "all_v1_to_v6_health_pilot_formal_bootstrap_benchmark_and_development_seeds_excluded",
        "seed_ledger.historical_seed_policy",
    )
    records = list(_sequence(ledger["records"], "protocol.seed_ledger.records"))
    _exact(records, expected_v7_seed_records(), "protocol.seed_ledger.records")
    values = [int(record["value"]) for record in records]
    if not all(values) or len(values) != len(set(values)):
        raise ConfigError("V7 seed ledger contains zero or duplicate values")
    overlap = sorted(set(values) & all_v1_to_v6_seeds())
    if overlap:
        raise ConfigError(f"V7 seed ledger reuses historical seeds: {overlap}")


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
    _keys(confirmations, set(V7_PROTOCOL_CONFIRMATION_FIELDS), "protocol.protocol_status.confirmations")
    for field in V7_PROTOCOL_CONFIRMATION_FIELDS:
        _exact(confirmations[field], True, f"protocol_status.confirmations.{field}")


def validate_v7_protocol(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete author-frozen V7 core protocol."""

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
        "predecessor_termination",
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
    _exact(raw["protocol_version"], 7, "protocol_version")
    _exact(raw["study_profile"], V7_PROFILE, "study_profile")
    _exact(raw["study_stage"], "health_pilot_formal", "study_stage")
    _exact(tuple(raw["active_conditions"]), V7_ACTIVE_CONDITIONS, "active_conditions")
    _exact(tuple(raw["seeds"]), V7_FORMAL_SEEDS, "seeds")
    _exact(raw["output_root"], V7_OUTPUT_ROOT, "output_root")
    _exact(dict(raw["artifact_paths"]), V7_ARTIFACT_PATHS, "artifact_paths")
    _validate_status(raw["protocol_status"])
    _exact(dict(raw["predecessor_termination"]), V7_PREDECESSOR_TERMINATION, "predecessor_termination")
    _validate_seed_ledger(raw["seed_ledger"])
    _exact(dict(raw["pilot_acceptance"]), V7_PILOT_ACCEPTANCE, "pilot_acceptance")
    _exact(dict(raw["data"]), V7_DATA_CONTRACT, "data")
    datasets = _mapping(raw["datasets"], "protocol.datasets")
    _keys(datasets, {"cifar100"}, "protocol.datasets")
    _exact(dict(datasets["cifar100"]), V7_CIFAR100_CONTRACT, "datasets.cifar100")
    _exact(dict(_mapping(raw["cifar100_provenance"], "protocol.cifar100_provenance")), V7_CIFAR100_PROVENANCE_CONTRACT, "cifar100_provenance")
    conditions = _mapping(raw["conditions"], "protocol.conditions")
    _keys(conditions, set(V7_ACTIVE_CONDITIONS), "protocol.conditions")
    _exact({name: dict(value) for name, value in conditions.items()}, V7_CONDITION_SPECS, "conditions")
    _exact(dict(raw["model"]), V7_MODEL_CONTRACT, "model")
    _exact(dict(raw["optimizer"]), V7_OPTIMIZER_CONTRACT, "optimizer")
    _exact(dict(raw["runtime"]), V7_RUNTIME_CONTRACT, "runtime")
    _exact(
        dict(raw["experiments"]),
        {
            "E9": "paired six-condition CIFAR-100 mechanism study",
            "E10": "prespecified paired mechanism analysis",
            "E11": "validation-only descriptive resource audit",
        },
        "experiments",
    )
    _exact(dict(raw["analysis"]), V7_ANALYSIS_CONTRACT, "analysis")
    _exact(dict(raw["benchmark"]), V7_BENCHMARK_CONTRACT, "benchmark")
    _exact(dict(raw["matrix"]), V7_MATRIX_CONTRACT, "matrix")
    _exact(dict(raw["conditional_extension"]), V7_CONDITIONAL_EXTENSION, "conditional_extension")
    return copy.deepcopy(dict(raw))


def _v7_provenance_file(project_root: Path, relative_path: str, label: str) -> Path:
    path = (project_root / relative_path).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ConfigError(f"V7 {label} escapes the project root: {path}") from exc
    if not path.is_file():
        raise ConfigError(f"V7 {label} is missing: {path}")
    return path


def validate_v7_cifar100_provenance_files(
    protocol: Mapping[str, Any], *, project_root: str | Path
) -> dict[str, str]:
    """Verify every local CIFAR-100 input used by V7 and its split identity."""

    validated = validate_v7_protocol(protocol)
    root = Path(project_root).resolve()
    contract = validated["cifar100_provenance"]
    provenance_path = _v7_provenance_file(root, str(contract["source_provenance_path"]), "CIFAR-100 source provenance")
    archive_path = _v7_provenance_file(root, str(contract["archive_path"]), "CIFAR-100 archive")
    train_path = _v7_provenance_file(root, str(contract["train_pickle_path"]), "CIFAR-100 train pickle")
    test_path = _v7_provenance_file(root, str(contract["test_pickle_path"]), "CIFAR-100 test pickle")
    meta_path = _v7_provenance_file(root, str(contract["meta_pickle_path"]), "CIFAR-100 meta pickle")
    split_path = _v7_provenance_file(root, str(contract["split_manifest_path"]), "CIFAR-100 split manifest")
    observed = {
        "source_provenance_sha256": sha256_file(provenance_path),
        "archive_sha256": sha256_file(archive_path),
        "train_pickle_sha256": sha256_file(train_path),
        "test_pickle_sha256": sha256_file(test_path),
        "meta_pickle_sha256": sha256_file(meta_path),
        "split_manifest_sha256": sha256_file(split_path),
    }
    for key, value in observed.items():
        if value != contract[key]:
            raise ConfigError(f"V7 CIFAR-100 {key} differs from the frozen protocol: {value} != {contract[key]}")
    for label, path, size_key in (
        ("archive", archive_path, "archive_bytes"),
        ("train pickle", train_path, "train_pickle_bytes"),
        ("test pickle", test_path, "test_pickle_bytes"),
        ("meta pickle", meta_path, "meta_pickle_bytes"),
    ):
        if path.stat().st_size != int(contract[size_key]):
            raise ConfigError(
                f"V7 CIFAR-100 {label} byte count differs from the frozen protocol"
            )
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        split = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Cannot parse frozen V7 CIFAR-100 provenance input: {exc}") from exc
    if not isinstance(provenance, Mapping) or not isinstance(split, Mapping):
        raise ConfigError("Frozen V7 CIFAR-100 provenance and split inputs must be JSON objects")
    if provenance.get("schema") != contract["source_provenance_schema"]:
        raise ConfigError("V7 CIFAR-100 source provenance schema differs from the protocol")
    archive_record = provenance.get("local_archive_binding")
    if not isinstance(archive_record, Mapping) or (
        archive_record.get("path") != contract["archive_path"]
        or archive_record.get("sha256") != contract["archive_sha256"]
        or archive_record.get("bytes") != contract["archive_bytes"]
    ):
        raise ConfigError("V7 CIFAR-100 provenance does not bind the frozen archive")
    extracted = provenance.get("extracted_binding")
    files = extracted.get("files") if isinstance(extracted, Mapping) else None
    train_record = files.get("train") if isinstance(files, Mapping) else None
    test_record = files.get("test") if isinstance(files, Mapping) else None
    meta_record = files.get("meta") if isinstance(files, Mapping) else None
    for label, record, sha_key, size_key in (
        ("train", train_record, "train_pickle_sha256", "train_pickle_bytes"),
        ("test", test_record, "test_pickle_sha256", "test_pickle_bytes"),
        ("meta", meta_record, "meta_pickle_sha256", "meta_pickle_bytes"),
    ):
        if not isinstance(record, Mapping) or (
            record.get("sha256") != contract[sha_key]
            or record.get("bytes") != contract[size_key]
        ):
            raise ConfigError(
                f"V7 CIFAR-100 source provenance does not bind the frozen {label} pickle"
            )
    split_record = provenance.get("development_split_binding")
    if not isinstance(split_record, Mapping) or (
        split_record.get("path") != contract["split_manifest_path"]
        or split_record.get("file_sha256") != contract["split_manifest_sha256"]
    ):
        raise ConfigError("V7 CIFAR-100 source provenance does not bind the frozen split manifest")
    internal_hash = split_record.get("internal_manifest_sha256")
    if not isinstance(internal_hash, str) or split.get("manifest_sha256") != internal_hash:
        raise ConfigError("V7 CIFAR-100 split manifest does not match its provenance record")
    return {
        "cifar100_source_provenance": artifact_path_reference(provenance_path, root),
        "cifar100_source_provenance_sha256": observed["source_provenance_sha256"],
        "cifar100_archive": artifact_path_reference(archive_path, root),
        "cifar100_archive_sha256": observed["archive_sha256"],
        "cifar100_train_pickle": artifact_path_reference(train_path, root),
        "cifar100_train_pickle_sha256": observed["train_pickle_sha256"],
        "cifar100_test_pickle": artifact_path_reference(test_path, root),
        "cifar100_test_pickle_sha256": observed["test_pickle_sha256"],
        "cifar100_meta_pickle": artifact_path_reference(meta_path, root),
        "cifar100_meta_pickle_sha256": observed["meta_pickle_sha256"],
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
        raise ConfigError("V7 core run dataset must be cifar100")
    frozen = dict(V7_DATA_CONTRACT)
    frozen.update(V7_CIFAR100_CONTRACT)
    for key, expected in frozen.items():
        _exact(values[key], expected, f"v7 run data.{key}")
    return DataConfig(**values)


def _merge_model(protocol: Mapping[str, Any], override: Mapping[str, Any], condition: str, data: DataConfig) -> ModelConfig:
    allowed = set(dataclasses.asdict(ModelConfig()))
    unknown = set(override) - allowed
    if unknown:
        raise ConfigError(f"Unknown key(s) in run.model: {', '.join(sorted(unknown))}")
    values = dataclasses.asdict(ModelConfig())
    values.update(protocol["model"])
    values.update(V7_CONDITION_SPECS[condition])
    values.update({"condition": condition, "depth": 20, "time_steps": 6})
    values.update({"num_classes": data.num_classes, "in_channels": data.in_channels})
    values.update(override)
    expected = {
        "condition": condition,
        **V7_CONDITION_SPECS[condition],
        "depth": 20,
        "time_steps": 6,
        "num_classes": 100,
        "in_channels": 3,
        **V7_MODEL_CONTRACT,
    }
    for key, frozen in expected.items():
        _exact(values[key], frozen, f"v7 run model.{key}")
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
    _exact(normalized, V7_OPTIMIZER_CONTRACT, "v7 run optimizer")
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
    frozen = {**V7_RUNTIME_CONTRACT, "run_id": run_id, "seed": seed, "output_dir": expected_output}
    for key, expected in frozen.items():
        _exact(values[key], expected, f"v7 run runtime.{key}")
    return RuntimeConfig(**values)


def validate_v7_run_mapping(raw: Mapping[str, Any], protocol: Mapping[str, Any] | None) -> RunConfig:
    """Resolve one strict V7 formal or pilot run configuration."""

    if protocol is None:
        raise ConfigError("Protocol v7 run validation requires its protocol mapping")
    protocol = validate_v7_protocol(protocol)
    raw = _mapping(raw, "run")
    allowed = {
        "protocol_version", "experiment", "run_id", "condition", "seed", "data",
        "model", "optimizer", "runtime", "final_test", "analysis",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in run: {', '.join(unknown)}")
    _exact(raw.get("protocol_version"), 7, "run.protocol_version")
    experiment = str(raw.get("experiment", ""))
    _exact(experiment, "E9", "run.experiment")
    model_raw = dict(_mapping(raw.get("model", {}), "run.model"))
    runtime_raw = dict(_mapping(raw.get("runtime", {}), "run.runtime"))
    condition = str(raw.get("condition", model_raw.get("condition", "")))
    if condition not in V7_ACTIVE_CONDITIONS:
        raise ConfigError(f"V7 run condition must be one of {V7_ACTIVE_CONDITIONS}")
    seed_value = raw.get("seed", runtime_raw.get("seed"))
    if isinstance(seed_value, bool) or not isinstance(seed_value, int):
        raise ConfigError("V7 run seed must be an integer")
    seed = int(seed_value)
    if seed in V7_FORMAL_SEEDS:
        expected_output = V7_OUTPUT_ROOT
    elif seed == V7_PILOT_SEED:
        expected_output = str(V7_PILOT_ACCEPTANCE["pilot_output_root"])
    else:
        raise ConfigError("V7 run seed is neither a frozen formal nor pilot seed")
    expected_run_id = f"E9_cifar100_d20_t6_{condition}_s{seed}"
    run_id = str(raw.get("run_id", runtime_raw.get("run_id", "")))
    _exact(run_id, expected_run_id, "v7 run run_id")
    data = _merge_data(protocol, _mapping(raw.get("data", {}), "run.data"))
    model = _merge_model(protocol, model_raw, condition, data)
    optimizer = _merge_optimizer(protocol, _mapping(raw.get("optimizer", {}), "run.optimizer"))
    runtime = _merge_runtime(protocol, runtime_raw, run_id=run_id, seed=seed, expected_output=expected_output)
    _exact(raw.get("final_test", False), False, "v7 run final_test")
    analysis = dict(_mapping(raw.get("analysis", {}), "run.analysis"))
    expected_protocol_hash = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
    matrix_key = ["cifar100", 20, 6, condition, seed]
    expected_analysis = copy.deepcopy(V7_ANALYSIS_CONTRACT)
    expected_analysis.update({"matrix_key": matrix_key, "protocol_hash": expected_protocol_hash})
    _exact(analysis, expected_analysis, "v7 run analysis")
    return RunConfig(
        protocol_version=7,
        experiment=experiment,
        data=data,
        model=model,
        optimizer=optimizer,
        runtime=runtime,
        final_test=False,
        analysis=analysis,
    )


def _raw_v7_run(protocol: Mapping[str, Any], *, condition: str, seed: int, output_dir: str) -> dict[str, Any]:
    protocol_hash = hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()
    run_id = f"E9_cifar100_d20_t6_{condition}_s{seed}"
    analysis = copy.deepcopy(V7_ANALYSIS_CONTRACT)
    analysis.update({"matrix_key": ["cifar100", 20, 6, condition, seed], "protocol_hash": protocol_hash})
    return {
        "protocol_version": 7,
        "experiment": "E9",
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "data": dict(V7_CIFAR100_CONTRACT),
        "model": {
            "condition": condition,
            **V7_CONDITION_SPECS[condition],
            "depth": 20,
            "time_steps": 6,
        },
        "optimizer": copy.deepcopy(V7_OPTIMIZER_CONTRACT),
        "runtime": {"output_dir": output_dir},
        "final_test": False,
        "analysis": analysis,
    }


def generate_v7_formal_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = validate_v7_protocol(protocol)
    runs = [
        _raw_v7_run(protocol, condition=condition, seed=seed, output_dir=V7_OUTPUT_ROOT)
        for seed in V7_FORMAL_SEEDS
        for condition in V7_ACTIVE_CONDITIONS
    ]
    if len(runs) != 48 or len({run["run_id"] for run in runs}) != 48:
        raise ConfigError("V7 formal matrix must contain exactly 48 unique runs")
    return runs


def generate_v7_pilot_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    protocol = validate_v7_protocol(protocol)
    output = str(V7_PILOT_ACCEPTANCE["pilot_output_root"])
    runs = [
        _raw_v7_run(protocol, condition=condition, seed=V7_PILOT_SEED, output_dir=output)
        for condition in V7_ACTIVE_CONDITIONS
    ]
    if len(runs) != 6 or len({run["run_id"] for run in runs}) != 6:
        raise ConfigError("V7 pilot matrix must contain exactly six unique runs")
    return runs


def canonicalize_v7_artifact_run_mapping(
    raw: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    project_root: str | Path,
    expected_output_dir: str = V7_OUTPUT_ROOT,
    expected_execution_device: str | None = None,
) -> dict[str, Any]:
    """Normalize narrowly approved V7 runtime artifact execution aliases.

    By default, this accepts only the frozen V7 runtime mapping.  A recovery
    caller may explicitly attest the sole reviewed launch override
    ``cuda:0``; it is then normalized in memory to the frozen logical device
    selector ``auto`` before strict run validation.
    """

    validated = validate_v7_protocol(protocol)
    value = copy.deepcopy(dict(_mapping(raw, "V7 runtime artifact run")))
    _exact(value.get("protocol_version"), 7, "artifact run.protocol_version")
    runtime = value.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ConfigError("V7 runtime artifact run.runtime must be a mapping")
    runtime_value = copy.deepcopy(dict(runtime))
    observed_value = runtime_value.get("output_dir")
    if not isinstance(observed_value, str) or not observed_value.strip():
        raise ConfigError("V7 runtime artifact output_dir must be a non-empty string")
    root = Path(project_root).resolve()
    expected = Path(expected_output_dir)
    if expected.is_absolute():
        raise ConfigError("V7 canonical output_dir must remain repository-relative")
    expected_path = (root / expected).resolve()
    try:
        expected_path.relative_to(root)
    except ValueError as exc:
        raise ConfigError("V7 canonical output_dir escapes the repository") from exc
    observed = Path(observed_value)
    observed_path = observed.resolve() if observed.is_absolute() else (root / observed).resolve()
    if observed_path != expected_path:
        raise ConfigError(
            "V7 runtime artifact output_dir is not path-equivalent to the frozen "
            f"directory: {observed_value!r} != {expected_output_dir!r}"
        )
    runtime_value["output_dir"] = expected_output_dir
    if expected_execution_device is not None:
        frozen_device = V7_RUNTIME_CONTRACT["device"]
        if expected_execution_device != "cuda:0" or frozen_device != "auto":
            raise ConfigError(
                "V7 artifact execution-device normalization only permits "
                "the reviewed 'cuda:0' -> 'auto' mapping"
            )
        observed_device = runtime_value.get("device")
        if observed_device != expected_execution_device:
            raise ConfigError(
                "V7 runtime artifact device is not the reviewed execution "
                f"device: {observed_device!r} != {expected_execution_device!r}"
            )
        runtime_value["device"] = frozen_device
    value["runtime"] = runtime_value
    validate_v7_run_mapping(value, validated)
    return value


def v7_path(project_root: str | Path, key: str) -> Path:
    """Resolve one frozen V7 artifact path inside the repository."""

    root = Path(project_root).resolve()
    if key not in V7_ARTIFACT_PATHS:
        raise ConfigError(f"Unknown V7 artifact path key: {key}")
    resolved = (root / V7_ARTIFACT_PATHS[key]).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConfigError(f"V7 artifact path escapes the repository: {key}") from exc
    return resolved
