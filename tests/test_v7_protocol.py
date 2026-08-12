from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import talif_msresnet.config_v7 as config_v7

from talif_msresnet.config import (
    ConfigError,
    canonical_json,
    generate_run_matrix,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.config_v6 import validate_v6_protocol
from talif_msresnet.config_v7 import (
    V6_CONSUMED_HEALTH_SEED,
    V6_RETIRED_UNUSED_SEEDS,
    V7_ACTIVE_CONDITIONS,
    V7_ARTIFACT_PATHS,
    V7_FORMAL_SEEDS,
    V7_HEALTH_SEED,
    V7_OUTPUT_ROOT,
    V7_PILOT_ACCEPTANCE,
    V7_PILOT_SEED,
    V7_PREDECESSOR_TERMINATION,
    all_v1_to_v6_seeds,
    canonicalize_v7_artifact_run_mapping,
    expected_v7_seed_records,
    generate_v7_pilot_matrix,
    validate_v7_protocol,
    validate_v7_run_mapping,
    validate_v7_cifar100_provenance_files,
)


ROOT = Path(__file__).resolve().parents[1]
V6_PROTOCOL_PATH = ROOT / "configs" / "protocol_v6_mechanism.yaml"
V7_PROTOCOL_PATH = ROOT / "configs" / "protocol_v7_mechanism.yaml"


def _protocol() -> dict[str, object]:
    return load_protocol(V7_PROTOCOL_PATH)


def test_v7_protocol_validates_through_public_dispatch() -> None:
    protocol = _protocol()

    assert validate_v7_protocol(protocol) == protocol
    assert protocol["protocol_version"] == 7
    assert tuple(protocol["active_conditions"]) == V7_ACTIVE_CONDITIONS
    assert tuple(protocol["seeds"]) == V7_FORMAL_SEEDS
    assert protocol["artifact_paths"] == V7_ARTIFACT_PATHS


def test_v7_seed_ledger_is_fresh_unique_and_exact() -> None:
    protocol = _protocol()
    records = expected_v7_seed_records()
    values = [int(record["value"]) for record in records]

    assert protocol["seed_ledger"]["records"] == records
    assert len(records) == 27
    assert len(values) == len(set(values))
    assert all(value > 0 for value in values)
    assert not (set(values) & all_v1_to_v6_seeds())


def test_all_unused_v6_seeds_are_explicitly_retired() -> None:
    predecessor = _protocol()["predecessor_termination"]

    assert predecessor == V7_PREDECESSOR_TERMINATION
    assert predecessor["status"] == "FAIL_IMMUTABLE"
    assert predecessor["consumed_health_seed"] == V6_CONSUMED_HEALTH_SEED
    assert predecessor["retired_unused_seed_disposition"] == (
        "RESERVED_UNUSED_RETIRED_WITH_V6"
    )
    assert tuple(predecessor["retired_unused_seeds"]) == V6_RETIRED_UNUSED_SEEDS
    assert len(predecessor["retired_unused_seeds"]) == 26
    assert V6_CONSUMED_HEALTH_SEED not in predecessor["retired_unused_seeds"]


def test_v6_protocol_and_canonical_failure_identity_remain_unchanged() -> None:
    protocol = load_protocol(V6_PROTOCOL_PATH)

    assert validate_v6_protocol(protocol) == protocol
    assert hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest() == (
        "c13b9cc568c9a26303c39de80447b1ac1048f8c1d46add6c9f747bad7691fecf"
    )
    assert V7_PREDECESSOR_TERMINATION["status"] == "FAIL_IMMUTABLE"


def test_v6_and_v7_execution_identities_are_disjoint() -> None:
    protocol = _protocol()
    v7_values = {int(record["value"]) for record in expected_v7_seed_records()}

    assert V7_HEALTH_SEED in v7_values
    assert V7_PILOT_SEED in v7_values
    assert V6_CONSUMED_HEALTH_SEED not in v7_values
    assert not (set(V6_RETIRED_UNUSED_SEEDS) & v7_values)
    assert all("v6_mechanism" not in path for path in V7_ARTIFACT_PATHS.values())
    assert all("v7" in path.lower() for path in V7_ARTIFACT_PATHS.values())
    assert protocol["predecessor_termination"]["evidence_copy_into_v7_roots"] == (
        "forbidden"
    )


def test_v7_health_integrity_fields_are_hash_bound() -> None:
    health = _protocol()["pilot_acceptance"]["health"]

    assert health["expected_adaptive_update_by_condition"] == {
        "M0": False,
        "M1": False,
        "M2": True,
        "M3": True,
        "M4": True,
        "PLIF": True,
    }
    assert health["fixed_batch_identity"] == {
        "seed_source": "same_frozen_health_seed_for_preclaim_and_execution",
        "rng_reset": (
            "isolated_cpu_torch_rng_reset_immediately_before_each_loader_construction"
        ),
        "digest": "sha256_over_fixed_input_and_target_tensors",
        "required_relation": "preclaim_sha256_exactly_equals_execution_sha256",
        "mismatch_action": "block_before_any_condition_probe",
    }

    changed = copy.deepcopy(_protocol())
    changed["pilot_acceptance"]["health"]["fixed_batch_identity"][
        "required_relation"
    ] = "not_frozen"
    with pytest.raises(ConfigError, match="pilot_acceptance"):
        validate_v7_protocol(changed)


def test_v7_formal_and_pilot_matrices_are_exact_and_resolvable() -> None:
    protocol = _protocol()
    formal = generate_run_matrix(protocol)
    pilot = generate_v7_pilot_matrix(protocol)

    assert len(formal) == 48
    assert len(pilot) == 6
    assert len({run["run_id"] for run in formal}) == 48
    assert len({run["run_id"] for run in pilot}) == 6
    assert {run["condition"] for run in formal} == set(V7_ACTIVE_CONDITIONS)
    assert {run["seed"] for run in formal} == set(V7_FORMAL_SEEDS)
    assert {run["seed"] for run in pilot} == {V7_PILOT_SEED}
    assert all(run["run_id"].startswith("E9_cifar100_d20_t6_") for run in formal + pilot)
    assert all(validate_run_mapping(run, protocol).protocol_version == 7 for run in formal + pilot)


def test_v7_run_contract_requires_repository_relative_output_path() -> None:
    protocol = _protocol()
    raw = generate_run_matrix(protocol)[0]
    absolute = copy.deepcopy(raw)
    absolute["runtime"]["output_dir"] = str((ROOT / V7_OUTPUT_ROOT).resolve())

    with pytest.raises(ConfigError, match="runtime.output_dir"):
        validate_v7_run_mapping(absolute, protocol)

    normalized = canonicalize_v7_artifact_run_mapping(
        absolute,
        protocol,
        project_root=ROOT,
    )
    assert normalized["runtime"]["output_dir"] == V7_OUTPUT_ROOT
    assert validate_v7_run_mapping(normalized, protocol).protocol_version == 7


def test_v7_artifact_path_canonicalizer_rejects_non_equivalent_path(
    tmp_path: Path,
) -> None:
    protocol = _protocol()
    raw = generate_run_matrix(protocol)[0]
    wrong = copy.deepcopy(raw)
    wrong["runtime"]["output_dir"] = str((tmp_path / "wrong").resolve())

    with pytest.raises(ConfigError, match="not path-equivalent"):
        canonicalize_v7_artifact_run_mapping(wrong, protocol, project_root=ROOT)


def test_v7_artifact_device_canonicalizer_accepts_only_reviewed_cuda_zero_override() -> None:
    protocol = _protocol()
    raw = generate_run_matrix(protocol)[0]
    artifact = copy.deepcopy(raw)
    artifact["runtime"]["device"] = "cuda:0"

    with pytest.raises(ConfigError, match="runtime.device"):
        canonicalize_v7_artifact_run_mapping(
            artifact,
            protocol,
            project_root=ROOT,
        )

    normalized = canonicalize_v7_artifact_run_mapping(
        artifact,
        protocol,
        project_root=ROOT,
        expected_execution_device="cuda:0",
    )

    assert artifact["runtime"]["device"] == "cuda:0"
    assert normalized["runtime"]["device"] == "auto"
    assert validate_v7_run_mapping(normalized, protocol).runtime.device == "auto"


@pytest.mark.parametrize("device", ("cuda", "cuda:1", "cpu", "auto"))
def test_v7_artifact_device_canonicalizer_rejects_any_nonreviewed_device(
    device: str,
) -> None:
    protocol = _protocol()
    artifact = copy.deepcopy(generate_run_matrix(protocol)[0])
    artifact["runtime"]["device"] = device

    with pytest.raises(ConfigError, match="reviewed execution device"):
        canonicalize_v7_artifact_run_mapping(
            artifact,
            protocol,
            project_root=ROOT,
            expected_execution_device="cuda:0",
        )


def test_v7_artifact_device_canonicalizer_rejects_unapproved_normalization_request() -> None:
    protocol = _protocol()
    artifact = copy.deepcopy(generate_run_matrix(protocol)[0])
    artifact["runtime"]["device"] = "cuda:1"

    with pytest.raises(ConfigError, match="only permits"):
        canonicalize_v7_artifact_run_mapping(
            artifact,
            protocol,
            project_root=ROOT,
            expected_execution_device="cuda:1",
        )


def test_v7_health_and_pilot_paths_are_isolated_and_seed_bound() -> None:
    acceptance = _protocol()["pilot_acceptance"]

    assert acceptance == V7_PILOT_ACCEPTANCE
    assert str(V7_HEALTH_SEED) in acceptance["health_output"]
    assert str(V7_HEALTH_SEED) in acceptance["attempt_receipt"]
    assert str(V7_PILOT_SEED) in acceptance["pilot_output_root"]
    assert acceptance["health_output"].startswith("results/pilot/v7_mechanism/")
    assert acceptance["pilot_output_root"].startswith("results/pilot/v7_mechanism/")


def test_v7_rejects_mutated_cifar100_train_pickle_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_sha256_file = config_v7.sha256_file
    train_path = (
        ROOT / _protocol()["cifar100_provenance"]["train_pickle_path"]
    ).resolve()

    def mutated_train_hash(path: str | Path) -> str:
        if Path(path).resolve() == train_path:
            return "0" * 64
        return original_sha256_file(path)

    monkeypatch.setattr(config_v7, "sha256_file", mutated_train_hash)

    with pytest.raises(ConfigError, match="train_pickle_sha256"):
        validate_v7_cifar100_provenance_files(_protocol(), project_root=ROOT)
