from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

from talif_msresnet import pilot_v6, preflight
from talif_msresnet.config import ConfigError, load_protocol
from talif_msresnet.config_v6 import (
    validate_v6_cifar100_provenance_files,
    validate_v6_protocol,
)
from talif_msresnet.utils import sha256_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_v6_results as v6_analysis
import archive_v6_evidence as v6_archive
import evaluate_checkpoints as final_eval


def _v6_protocol() -> dict[str, object]:
    return load_protocol(ROOT / "configs" / "protocol_v6_mechanism.yaml")


def _source_record() -> dict[str, str]:
    return {
        "test_source": "data/cifar100/cifar-100-python/test",
        "test_source_sha256": "a" * 64,
        "cifar100_source_provenance": "environment/CIFAR100_SOURCE_PROVENANCE.json",
        "cifar100_source_provenance_sha256": "b" * 64,
        "cifar100_test_pickle": "data/cifar100/cifar-100-python/test",
        "cifar100_test_pickle_sha256": "a" * 64,
        "cifar100_split_manifest": "data/manifests/cifar100_seed2024.json",
        "cifar100_split_manifest_sha256": "c" * 64,
    }


def _write_recovery_fixture(
    tmp_path: Path,
    *,
    source: dict[str, str],
) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint)
    evaluated_at = "2026-08-11T00:00:00+00:00"
    samples = 8
    journal = {
        "run_id": run_dir.name,
        "stage": "results_ready",
        "protocol_hash": "protocol-hash",
        "checkpoint_sha256": checkpoint_sha256,
        "evaluated_at": evaluated_at,
        "test": {"loss": 1.25, "accuracy": 0.75, "samples": samples},
        "test_source": source,
    }
    marker = {
        "status": "complete",
        "run_id": run_dir.name,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluated_at": evaluated_at,
        "test_samples": samples,
        **source,
    }
    manifest = {
        "final_test": {
            "status": "complete",
            "checkpoint": "best.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "evaluated_at": evaluated_at,
            "samples": samples,
            **source,
        }
    }
    (run_dir / final_eval.JOURNAL_NAME).write_text(
        json.dumps(journal), encoding="utf-8"
    )
    (run_dir / final_eval.LOCK_NAME).write_text("stale lock\n", encoding="utf-8")
    (run_dir / "final_test.json").write_text(json.dumps(marker), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, checkpoint, run_dir / "run_manifest.json"


def test_v6_protocol_rejects_torchvision_and_provenance_contract_drift() -> None:
    protocol = _v6_protocol()
    bad_runtime = copy.deepcopy(protocol)
    bad_runtime["pilot_acceptance"]["environment"]["torchvision_version"] = "0.24.1"
    with pytest.raises(ConfigError, match="pilot_acceptance"):
        validate_v6_protocol(bad_runtime)

    bad_provenance = copy.deepcopy(protocol)
    bad_provenance["cifar100_provenance"]["test_pickle_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="cifar100_provenance"):
        validate_v6_protocol(bad_provenance)


def test_v6_frozen_cifar100_files_reject_each_bound_input_drift(tmp_path: Path) -> None:
    protocol = _v6_protocol()
    bindings = protocol["cifar100_provenance"]
    targets: dict[str, Path] = {}
    for key in ("source_provenance_path", "test_pickle_path", "split_manifest_path"):
        source = ROOT / str(bindings[key])
        target = tmp_path / str(bindings[key])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        targets[key] = target

    observed = validate_v6_cifar100_provenance_files(protocol, project_root=tmp_path)
    assert observed["cifar100_test_pickle_sha256"] == bindings["test_pickle_sha256"]

    mutations = (
        ("source_provenance_path", "source_provenance_sha256"),
        ("test_pickle_path", "test_pickle_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
    )
    for path_key, hash_key in mutations:
        target = targets[path_key]
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with pytest.raises(ConfigError, match=hash_key):
            validate_v6_cifar100_provenance_files(protocol, project_root=tmp_path)
        target.write_bytes(original)


def test_source_only_preflight_skips_local_torchvision_runtime_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = ROOT / "configs" / "protocol_v6_mechanism.yaml"
    monkeypatch.setattr(preflight, "_check_dependencies", lambda _names: [])
    monkeypatch.setattr(preflight, "distribution_version", lambda _name: "0.24.1")

    source_only = preflight.check_protocol(
        protocol_path,
        mode="full",
        project_root=ROOT,
        check_dependencies=False,
    )
    assert not any("torchvision version mismatch" in error for error in source_only.errors)

    runtime = preflight.check_protocol(
        protocol_path,
        mode="full",
        project_root=ROOT,
        check_dependencies=True,
    )
    assert any("V6 torchvision version mismatch" in error for error in runtime.errors)


def test_health_runtime_contract_rejects_cpu_torchvision_before_seed_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _v6_protocol()["pilot_acceptance"]["environment"]
    monkeypatch.setattr(pilot_v6, "distribution_version", lambda _name: "0.24.1")

    with pytest.raises(pilot_v6.PilotV6Error, match="torchvision mismatch"):
        pilot_v6.validate_v6_runtime_environment(environment)


def test_v6_final_test_rejects_an_empty_source_binding() -> None:
    with pytest.raises(RuntimeError, match="invalid field set"):
        final_eval._require_v6_test_source_binding({})


def test_v6_recovery_rejects_marker_source_mismatch_without_cleanup(tmp_path: Path) -> None:
    expected = _source_record()
    run_dir, _checkpoint, _manifest_path = _write_recovery_fixture(
        tmp_path, source=expected
    )
    marker_path = run_dir / "final_test.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["cifar100_test_pickle_sha256"] = "d" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match="marker source"):
        final_eval._recover_one(
            run_dir,
            tmp_path,
            "protocol-hash",
            v6_test_source=expected,
        )
    assert (run_dir / final_eval.JOURNAL_NAME).exists()
    assert (run_dir / final_eval.LOCK_NAME).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_sha256", "f" * 64),
        ("evaluated_at", "2026-08-11T00:01:00+00:00"),
        ("samples", 99),
        ("cifar100_source_provenance_sha256", "d" * 64),
    ),
)
def test_v6_recovery_rejects_manifest_final_test_mismatch_without_cleanup(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    expected = _source_record()
    run_dir, _checkpoint, manifest_path = _write_recovery_fixture(tmp_path, source=expected)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_test"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="run manifest"):
        final_eval._recover_one(
            run_dir,
            tmp_path,
            "protocol-hash",
            v6_test_source=expected,
        )
    assert (run_dir / final_eval.JOURNAL_NAME).exists()
    assert (run_dir / final_eval.LOCK_NAME).exists()


def test_v6_recovery_cleans_only_congruent_marker_manifest_and_journal(tmp_path: Path) -> None:
    expected = _source_record()
    run_dir, _checkpoint, _manifest_path = _write_recovery_fixture(
        tmp_path, source=expected
    )

    assert (
        final_eval._recover_one(
            run_dir,
            tmp_path,
            "protocol-hash",
            v6_test_source=expected,
        )
        == "cleaned_committed_transaction"
    )
    assert not (run_dir / final_eval.JOURNAL_NAME).exists()
    assert not (run_dir / final_eval.LOCK_NAME).exists()


def test_v6_analysis_rejects_marker_with_wrong_frozen_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = 123
    condition = "M0"
    run_id = f"E6_cifar100_d20_t6_{condition}_s{seed}"
    source = _source_record()
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint)
    marker = {
        "status": "complete",
        "run_id": run_id,
        "protocol_hash": "protocol-hash",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch_zero_based": 0,
        "selected_epoch": 1,
        "config_hash": "config-hash",
        "evaluated_at": "2026-08-11T00:00:00+00:00",
        "test_accuracy": 0.5,
        "test_loss": 1.0,
        "test_samples": 8,
        **source,
    }
    marker["cifar100_test_pickle_sha256"] = "d" * 64
    metrics = {
        "status": "complete",
        "run_id": run_id,
        "seed": seed,
        "condition": condition,
        "config_hash": "config-hash",
        "test_accuracy": 0.5,
        "test_loss": 1.0,
        "test_samples": 8,
        "test_checkpoint_sha256": checkpoint_sha256,
        "test_evaluated_at": marker["evaluated_at"],
        "training_environment_sha256": "e" * 64,
        "split_manifest_sha256": "f" * 64,
        "shared_weight_sha256": "a" * 64,
    }
    (run_dir / "final_test.json").write_text(json.dumps(marker), encoding="utf-8")
    (run_dir / "seed_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(v6_analysis, "V6_FORMAL_SEEDS", (seed,))
    monkeypatch.setattr(v6_analysis, "V6_ACTIVE_CONDITIONS", (condition,))
    monkeypatch.setattr(
        v6_analysis,
        "load_checkpoint",
        lambda *_args, **_kwargs: {"config": {}, "config_hash": "", "epoch": 0},
    )
    monkeypatch.setattr(
        v6_analysis,
        "validate_v6_cifar100_provenance_files",
        lambda *_args, **_kwargs: {
            key: value for key, value in source.items() if key not in {"test_source", "test_source_sha256"}
        },
    )

    with pytest.raises(v6_analysis.V6AnalysisInputError, match="marker.cifar100_test_pickle_sha256"):
        v6_analysis.collect_final_test_rows(
            tmp_path,
            protocol={"protocol_version": 6},
            expected_protocol_hash="protocol-hash",
        )


def test_v6_archive_state_persists_all_frozen_cifar100_provenance_fields() -> None:
    state = {
        "freeze_manifest_sha256": "f" * 64,
        "benchmark_receipt_sha256": "b" * 64,
        "analysis_sha256": "a" * 64,
        "training_environment_sha256": "e" * 64,
        "cifar100_source_provenance": "environment/CIFAR100_SOURCE_PROVENANCE.json",
        "cifar100_source_provenance_sha256": "1" * 64,
        "cifar100_test_pickle": "data/cifar100/cifar-100-python/test",
        "cifar100_test_pickle_sha256": "2" * 64,
        "cifar100_split_manifest": "data/manifests/cifar100_seed2024.json",
        "cifar100_split_manifest_sha256": "3" * 64,
    }
    payload = v6_archive._archive_state_payload(state, source_commit="a" * 40)

    assert payload["schema"] == v6_archive.ARCHIVE_STATE_SCHEMA
    assert payload["source_commit"] == "a" * 40
    assert {
        key: payload[key] for key in v6_archive._CIFAR100_PROVENANCE_STATE_KEYS
    } == {
        key: state[key] for key in v6_archive._CIFAR100_PROVENANCE_STATE_KEYS
    }
