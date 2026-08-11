from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import analyze_v7_results as v7_analysis
import archive_v7_evidence as v7_archive
import evaluate_checkpoints as final_eval
import pilot_health_gate_v7 as v7_health_gate

from talif_msresnet import config_v7, pilot_v7, preflight
from talif_msresnet.config import ConfigError, load_protocol
from talif_msresnet.config_v7 import (
    validate_v7_cifar100_provenance_files,
    validate_v7_protocol,
)
from talif_msresnet.utils import sha256_file


def _v7_protocol() -> dict[str, object]:
    return load_protocol(ROOT / "configs" / "protocol_v7_mechanism.yaml")


def _source_record() -> dict[str, str]:
    return {
        "test_source": "data/cifar100/cifar-100-python/test",
        "test_source_sha256": "a" * 64,
        "cifar100_source_provenance": "environment/CIFAR100_SOURCE_PROVENANCE.json",
        "cifar100_source_provenance_sha256": "b" * 64,
        "cifar100_archive": "data/cifar100/cifar-100-python.tar.gz",
        "cifar100_archive_sha256": "c" * 64,
        "cifar100_train_pickle": "data/cifar100/cifar-100-python/train",
        "cifar100_train_pickle_sha256": "d" * 64,
        "cifar100_test_pickle": "data/cifar100/cifar-100-python/test",
        "cifar100_test_pickle_sha256": "a" * 64,
        "cifar100_meta_pickle": "data/cifar100/cifar-100-python/meta",
        "cifar100_meta_pickle_sha256": "e" * 64,
        "cifar100_split_manifest": "data/manifests/cifar100_seed2024.json",
        "cifar100_split_manifest_sha256": "f" * 64,
    }


def _benchmark_binding() -> dict[str, str]:
    return {
        "benchmark_receipt": "results/benchmark/v7_mechanism/benchmark_completion.json",
        "benchmark_receipt_sha256": "d" * 64,
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
        **_benchmark_binding(),
    }
    marker = {
        "status": "complete",
        "run_id": run_dir.name,
        "checkpoint_sha256": checkpoint_sha256,
        "evaluated_at": evaluated_at,
        "test_samples": samples,
        **source,
        **_benchmark_binding(),
    }
    manifest = {
        "final_test": {
            "status": "complete",
            "checkpoint": "best.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "evaluated_at": evaluated_at,
            "samples": samples,
            **source,
            **_benchmark_binding(),
        }
    }
    (run_dir / final_eval.JOURNAL_NAME).write_text(
        json.dumps(journal), encoding="utf-8"
    )
    (run_dir / final_eval.LOCK_NAME).write_text("stale lock\n", encoding="utf-8")
    (run_dir / "final_test.json").write_text(json.dumps(marker), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir, checkpoint, run_dir / "run_manifest.json"


def test_v7_protocol_rejects_torchvision_and_provenance_contract_drift() -> None:
    protocol = _v7_protocol()
    bad_runtime = copy.deepcopy(protocol)
    bad_runtime["pilot_acceptance"]["environment"]["torchvision_version"] = "0.24.1"
    with pytest.raises(ConfigError, match="pilot_acceptance"):
        validate_v7_protocol(bad_runtime)

    bad_provenance = copy.deepcopy(protocol)
    bad_provenance["cifar100_provenance"]["test_pickle_sha256"] = "0" * 64
    with pytest.raises(ConfigError, match="cifar100_provenance"):
        validate_v7_protocol(bad_provenance)


def test_v7_frozen_cifar100_files_reject_each_bound_input_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _v7_protocol()
    archive_relative = "data/cifar100/cifar-100-python.tar.gz"
    train_relative = "data/cifar100/cifar-100-python/train"
    test_relative = "data/cifar100/cifar-100-python/test"
    meta_relative = "data/cifar100/cifar-100-python/meta"
    split_relative = "data/manifests/cifar100_seed2024.json"
    provenance_relative = "environment/CIFAR100_SOURCE_PROVENANCE.json"
    archive_path = tmp_path / archive_relative
    train_path = tmp_path / train_relative
    test_path = tmp_path / test_relative
    meta_path = tmp_path / meta_relative
    split_path = tmp_path / split_relative
    provenance_path = tmp_path / provenance_relative
    archive_path.parent.mkdir(parents=True)
    test_path.parent.mkdir(parents=True)
    split_path.parent.mkdir(parents=True)
    provenance_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"frozen CIFAR-100 archive bytes")
    train_path.write_bytes(b"frozen training pickle bytes")
    test_path.write_bytes(b"frozen held-out test bytes")
    meta_path.write_bytes(b"frozen metadata pickle bytes")
    split_path.write_text(
        json.dumps({"manifest_sha256": "internal-split-hash"}) + "\n",
        encoding="utf-8",
    )
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "cifar100-source-provenance-v1",
                "local_archive_binding": {
                    "path": archive_relative,
                    "sha256": sha256_file(archive_path),
                    "bytes": archive_path.stat().st_size,
                },
                "extracted_binding": {
                    "files": {
                        "train": {
                            "sha256": sha256_file(train_path),
                            "bytes": train_path.stat().st_size,
                        },
                        "test": {
                            "sha256": sha256_file(test_path),
                            "bytes": test_path.stat().st_size,
                        },
                        "meta": {
                            "sha256": sha256_file(meta_path),
                            "bytes": meta_path.stat().st_size,
                        },
                    }
                },
                "development_split_binding": {
                    "path": split_relative,
                    "file_sha256": sha256_file(split_path),
                    "internal_manifest_sha256": "internal-split-hash",
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    bindings = {
        "source_provenance_path": provenance_relative,
        "source_provenance_schema": "cifar100-source-provenance-v1",
        "source_provenance_sha256": sha256_file(provenance_path),
        "archive_path": archive_relative,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "train_pickle_path": train_relative,
        "train_pickle_bytes": train_path.stat().st_size,
        "train_pickle_sha256": sha256_file(train_path),
        "test_pickle_path": test_relative,
        "test_pickle_bytes": test_path.stat().st_size,
        "test_pickle_sha256": sha256_file(test_path),
        "meta_pickle_path": meta_relative,
        "meta_pickle_bytes": meta_path.stat().st_size,
        "meta_pickle_sha256": sha256_file(meta_path),
        "split_manifest_path": split_relative,
        "split_manifest_sha256": sha256_file(split_path),
    }
    protocol["cifar100_provenance"] = bindings
    monkeypatch.setattr(config_v7, "V7_CIFAR100_PROVENANCE_CONTRACT", bindings)
    targets = {
        "source_provenance_path": provenance_path,
        "archive_path": archive_path,
        "train_pickle_path": train_path,
        "test_pickle_path": test_path,
        "meta_pickle_path": meta_path,
        "split_manifest_path": split_path,
    }

    observed = validate_v7_cifar100_provenance_files(protocol, project_root=tmp_path)
    assert observed["cifar100_test_pickle_sha256"] == bindings["test_pickle_sha256"]

    mutations = (
        ("source_provenance_path", "source_provenance_sha256"),
        ("archive_path", "archive_sha256"),
        ("train_pickle_path", "train_pickle_sha256"),
        ("test_pickle_path", "test_pickle_sha256"),
        ("meta_pickle_path", "meta_pickle_sha256"),
        ("split_manifest_path", "split_manifest_sha256"),
    )
    for path_key, hash_key in mutations:
        target = targets[path_key]
        original = target.read_bytes()
        target.write_bytes(original + b"\n")
        with pytest.raises(ConfigError, match=hash_key):
            validate_v7_cifar100_provenance_files(protocol, project_root=tmp_path)
        target.write_bytes(original)


def test_source_only_preflight_skips_local_torchvision_runtime_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = ROOT / "configs" / "protocol_v7_mechanism.yaml"
    monkeypatch.setattr(preflight, "_check_dependencies", lambda _names: [])
    monkeypatch.setattr(preflight, "distribution_version", lambda _name: "0.24.1")

    source_only = preflight.check_protocol(
        protocol_path,
        mode="pilot",
        project_root=ROOT,
        check_dependencies=False,
    )
    assert not any("torchvision version mismatch" in error for error in source_only.errors)

    runtime = preflight.check_protocol(
        protocol_path,
        mode="pilot",
        project_root=ROOT,
        check_dependencies=True,
    )
    assert any("V7 torchvision version mismatch" in error for error in runtime.errors)


def test_health_runtime_contract_rejects_cpu_torchvision_before_seed_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = _v7_protocol()["pilot_acceptance"]["environment"]
    monkeypatch.setattr(pilot_v7, "distribution_version", lambda _name: "0.24.1")

    with pytest.raises(pilot_v7.PilotV7Error, match="torchvision mismatch"):
        pilot_v7.validate_v7_runtime_environment(environment)


def test_health_preclaim_blocks_cifar100_provenance_before_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        model=SimpleNamespace(condition="M0"),
        runtime=SimpleNamespace(seed=123, deterministic=True, amp=False),
        data=SimpleNamespace(dataset="cifar100"),
    )
    block = SimpleNamespace(conditions=("M0",), pilot_seed=123)
    protocol = {
        "pilot_acceptance": {
            "environment": {
                "expected_gpu_substring": "RTX 5090",
                "cublas_workspace_config": ":4096:8",
            },
            "health": {"fixed_batch_size": 8},
        }
    }
    git_identity = {"git_commit": "a" * 40, "tracked_clean": True}
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(v7_health_gate, "_configure_deterministic_runtime", lambda: None)
    monkeypatch.setattr(
        v7_health_gate.legacy_gate,
        "require_target_cuda",
        lambda *_args, **_kwargs: {"device": "cuda:0", "name": "RTX 5090"},
    )
    monkeypatch.setattr(
        v7_health_gate.legacy_gate,
        "validate_runtime_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(v7_health_gate, "validate_v7_runtime_environment", lambda _env: None)
    monkeypatch.setattr(
        v7_health_gate.legacy_gate,
        "gpu_idle_precheck",
        lambda *_args, **_kwargs: {"device_uuid": "GPU-test"},
    )
    monkeypatch.setattr(
        v7_health_gate,
        "_training_environment_identity",
        lambda *_args, **_kwargs: ("{}", "b" * 64),
    )
    monkeypatch.setattr(v7_health_gate, "repository_git_identity", lambda _root: git_identity)
    monkeypatch.setattr(v7_health_gate, "_runtime_source_hashes", dict)
    monkeypatch.setattr(
        v7_health_gate,
        "validate_v7_cifar100_provenance_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConfigError("test pickle hash mismatch")
        ),
    )
    monkeypatch.setattr(
        v7_health_gate.v3_gate,
        "load_fixed_real_batch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("loader must not run after failed provenance")
        ),
    )

    with pytest.raises(pilot_v7.PilotV7Error, match="CIFAR-100 provenance"):
        v7_health_gate._preclaim_gate(
            protocol=protocol,
            configs=(config,),
            block=block,
            device_name="cuda:0",
            git_identity=git_identity,
            runtime_sources={},
            source_binding={},
        )


def test_v7_final_test_rejects_an_empty_source_binding() -> None:
    with pytest.raises(RuntimeError, match="invalid field set"):
        final_eval._require_v7_test_source_binding({})


def test_v7_final_test_accepts_only_the_expanded_source_binding() -> None:
    expected = _source_record()
    assert set(expected) == final_eval._V7_FINAL_TEST_SOURCE_KEYS
    assert final_eval._require_v7_test_source_binding(expected) is expected

    incomplete = dict(expected)
    incomplete.pop("cifar100_meta_pickle")
    with pytest.raises(RuntimeError, match="cifar100_meta_pickle"):
        final_eval._require_v7_test_source_binding(incomplete)


def test_v7_analysis_preserves_the_complete_final_test_source_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _source_record()
    monkeypatch.setattr(
        v7_analysis,
        "validate_v7_cifar100_provenance_files",
        lambda *_args, **_kwargs: {
            key: value
            for key, value in expected.items()
            if key not in {"test_source", "test_source_sha256"}
        },
    )

    observed = v7_analysis._v7_final_test_source_binding({"protocol_version": 7})
    assert observed == expected
    assert set(observed) == final_eval._V7_FINAL_TEST_SOURCE_KEYS


def test_v7_recovery_rejects_marker_source_mismatch_without_cleanup(tmp_path: Path) -> None:
    expected = _source_record()
    run_dir, _checkpoint, _manifest_path = _write_recovery_fixture(
        tmp_path, source=expected
    )
    marker_path = run_dir / "final_test.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["cifar100_test_pickle_sha256"] = "d" * 64
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"marker .*source"):
        final_eval._recover_one(
            run_dir,
            tmp_path,
            "protocol-hash",
            v7_test_source=expected,
            v7_receipt_validator=lambda: dict(_benchmark_binding()),
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
def test_v7_recovery_rejects_manifest_final_test_mismatch_without_cleanup(
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
            v7_test_source=expected,
            v7_receipt_validator=lambda: dict(_benchmark_binding()),
        )
    assert (run_dir / final_eval.JOURNAL_NAME).exists()
    assert (run_dir / final_eval.LOCK_NAME).exists()


def test_v7_recovery_cleans_only_congruent_marker_manifest_and_journal(tmp_path: Path) -> None:
    expected = _source_record()
    run_dir, _checkpoint, _manifest_path = _write_recovery_fixture(
        tmp_path, source=expected
    )

    assert (
        final_eval._recover_one(
            run_dir,
            tmp_path,
            "protocol-hash",
            v7_test_source=expected,
            v7_receipt_validator=lambda: dict(_benchmark_binding()),
        )
        == "cleaned_committed_transaction"
    )
    assert not (run_dir / final_eval.JOURNAL_NAME).exists()
    assert not (run_dir / final_eval.LOCK_NAME).exists()


def test_v7_analysis_rejects_marker_with_wrong_frozen_source_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = 123
    condition = "M0"
    run_id = f"E9_cifar100_d20_t6_{condition}_s{seed}"
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
        **_benchmark_binding(),
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
    monkeypatch.setattr(v7_analysis, "V7_FORMAL_SEEDS", (seed,))
    monkeypatch.setattr(v7_analysis, "V7_ACTIVE_CONDITIONS", (condition,))
    monkeypatch.setattr(v7_analysis, "expected_v7_run_ids", lambda: (run_id,))
    monkeypatch.setattr(
        v7_analysis,
        "load_checkpoint",
        lambda *_args, **_kwargs: {"config": {}, "config_hash": "", "epoch": 0},
    )
    monkeypatch.setattr(
        v7_analysis,
        "validate_v7_cifar100_provenance_files",
        lambda *_args, **_kwargs: {
            key: value for key, value in source.items() if key not in {"test_source", "test_source_sha256"}
        },
    )

    with pytest.raises(v7_analysis.V7AnalysisInputError, match="marker.cifar100_test_pickle_sha256"):
        v7_analysis.collect_final_test_rows(
            tmp_path,
            protocol={"protocol_version": 7},
            expected_protocol_hash="protocol-hash",
            expected_benchmark_binding=_benchmark_binding(),
        )


def _write_v7_analysis_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[str, Path, Path, dict[str, str]]:
    seed = 123
    condition = "M0"
    run_id = f"E9_cifar100_d20_t6_{condition}_s{seed}"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    checkpoint = run_dir / "best.pt"
    checkpoint.write_bytes(b"v7-analysis-checkpoint")
    checkpoint_sha256 = sha256_file(checkpoint)
    config_hash = "4" * 64
    environment_hash = "5" * 64
    split_hash = "6" * 64
    shared_hash = "7" * 64
    protocol_hash = "protocol-hash"
    evaluated_at = "2026-08-11T00:00:00+00:00"
    source = _source_record()
    benchmark = _benchmark_binding()
    config = {"analysis": {"protocol_hash": protocol_hash}}
    marker = {
        "status": "complete",
        "run_id": run_id,
        "protocol_hash": protocol_hash,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch_zero_based": 0,
        "selected_epoch": 1,
        "config_hash": config_hash,
        "evaluated_at": evaluated_at,
        "test_accuracy": 0.5,
        "test_loss": 1.0,
        "test_samples": 8,
        **source,
        **benchmark,
    }
    metrics = {
        "status": "complete",
        "run_id": run_id,
        "seed": seed,
        "condition": condition,
        "config_hash": config_hash,
        "test_accuracy": 0.5,
        "test_loss": 1.0,
        "test_samples": 8,
        "test_checkpoint_sha256": checkpoint_sha256,
        "test_evaluated_at": evaluated_at,
        "training_environment_sha256": environment_hash,
        "split_manifest_sha256": split_hash,
        "shared_weight_sha256": shared_hash,
    }
    manifest = {
        "status": "complete",
        "run_id": run_id,
        "config_hash": config_hash,
        "config": config,
        "environment": {"training_environment_sha256": environment_hash},
        "split_manifest_sha256": split_hash,
        "shared_weight_sha256": shared_hash,
        "final_test": {
            "status": "complete",
            "checkpoint": "best.pt",
            "checkpoint_sha256": checkpoint_sha256,
            "evaluated_at": evaluated_at,
            "samples": 8,
            **source,
            **benchmark,
        },
    }
    marker_path = run_dir / "final_test.json"
    manifest_path = run_dir / "run_manifest.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "seed_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    monkeypatch.setattr(v7_analysis, "V7_FORMAL_SEEDS", (seed,))
    monkeypatch.setattr(v7_analysis, "V7_ACTIVE_CONDITIONS", (condition,))
    monkeypatch.setattr(v7_analysis, "expected_v7_run_ids", lambda: (run_id,))
    monkeypatch.setattr(
        v7_analysis,
        "validate_v7_cifar100_provenance_files",
        lambda *_args, **_kwargs: {
            key: value
            for key, value in source.items()
            if key not in {"test_source", "test_source_sha256"}
        },
    )
    monkeypatch.setattr(
        v7_analysis,
        "canonicalize_v7_artifact_run_mapping",
        lambda value, *_args, **_kwargs: value,
    )
    monkeypatch.setattr(
        v7_analysis,
        "validate_run_mapping",
        lambda *_args, **_kwargs: SimpleNamespace(
            runtime=SimpleNamespace(run_id=run_id),
            config_hash=config_hash,
        ),
    )
    monkeypatch.setattr(
        v7_analysis,
        "load_checkpoint",
        lambda *_args, **_kwargs: {
            "config": config,
            "config_hash": config_hash,
            "epoch": 0,
        },
    )
    return run_id, marker_path, manifest_path, benchmark


@pytest.mark.parametrize("target", ("marker", "manifest"))
def test_v7_analysis_rejects_tampered_benchmark_binding_in_both_commit_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    _run_id, marker_path, manifest_path, benchmark = _write_v7_analysis_fixture(
        tmp_path, monkeypatch
    )
    path = marker_path if target == "marker" else manifest_path
    value = json.loads(path.read_text(encoding="utf-8"))
    record = value if target == "marker" else value["final_test"]
    record["benchmark_receipt_sha256"] = "0" * 64
    path.write_text(json.dumps(value), encoding="utf-8")

    expected_failure = (
        "marker benchmark receipt binding"
        if target == "marker"
        else "manifest.final_test benchmark receipt binding"
    )
    with pytest.raises(v7_analysis.V7AnalysisInputError, match=expected_failure):
        v7_analysis.collect_final_test_rows(
            tmp_path,
            protocol={"protocol_version": 7},
            expected_protocol_hash="protocol-hash",
            expected_benchmark_binding=benchmark,
        )


def test_v7_analysis_accepts_only_e9_and_frozen_run_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_id, _marker_path, _manifest_path, benchmark = _write_v7_analysis_fixture(
        tmp_path, monkeypatch
    )
    frame, audit = v7_analysis.collect_final_test_rows(
        tmp_path,
        protocol={"protocol_version": 7},
        expected_protocol_hash="protocol-hash",
        expected_benchmark_binding=benchmark,
    )
    assert tuple(frame["run_id"]) == (run_id,)
    assert tuple(row["run_id"] for row in audit) == (run_id,)

    e6_dir = tmp_path / run_id.replace("E9_", "E6_", 1)
    e6_dir.mkdir()
    with pytest.raises(v7_analysis.V7AnalysisInputError, match="Unexpected run directories"):
        v7_analysis.collect_final_test_rows(
            tmp_path,
            protocol={"protocol_version": 7},
            expected_protocol_hash="protocol-hash",
            expected_benchmark_binding=benchmark,
        )


def test_v7_analysis_revalidates_receipt_without_mutating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "benchmark_completion.json"
    receipt_path.write_text('{"schema":"v7-receipt"}\n', encoding="utf-8")
    receipt_hash = sha256_file(receipt_path)
    run_id = "E9_cifar100_d20_t6_M0_s123"
    rows = [{"run_id": run_id}]
    observed: dict[str, object] = {}
    monkeypatch.setattr(v7_analysis, "expected_v7_run_ids", lambda: (run_id,))

    def validate_receipt(**kwargs: object) -> dict[str, str]:
        observed.update(kwargs)
        return {"schema": "v7-receipt"}

    monkeypatch.setattr(v7_analysis, "validate_v7_benchmark_receipt", validate_receipt)
    receipt = v7_analysis._validate_current_benchmark_receipt(
        protocol={"protocol_version": 7},
        protocol_path=tmp_path / "protocol.yaml",
        protocol_hash="protocol-hash",
        receipt_path=receipt_path,
        receipt_sha256_before=receipt_hash,
        expected_git_commit="a" * 40,
        expected_matrix_manifest_sha256="b" * 64,
        expected_freeze_manifest_sha256="c" * 64,
        expected_formal_rows=rows,
    )
    assert receipt == {"schema": "v7-receipt"}
    assert observed["expected_formal_rows"] is rows
    assert sha256_file(receipt_path) == receipt_hash


def test_v7_analysis_blocks_if_receipt_changes_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt_path = tmp_path / "benchmark_completion.json"
    receipt_path.write_text('{"schema":"v7-receipt"}\n', encoding="utf-8")
    receipt_hash = sha256_file(receipt_path)
    run_id = "E9_cifar100_d20_t6_M0_s123"
    monkeypatch.setattr(v7_analysis, "expected_v7_run_ids", lambda: (run_id,))

    def tamper(**_kwargs: object) -> dict[str, str]:
        receipt_path.write_text('{"schema":"tampered"}\n', encoding="utf-8")
        return {"schema": "tampered"}

    monkeypatch.setattr(v7_analysis, "validate_v7_benchmark_receipt", tamper)
    with pytest.raises(v7_analysis.V7AnalysisInputError, match="changed during"):
        v7_analysis._validate_current_benchmark_receipt(
            protocol={"protocol_version": 7},
            protocol_path=tmp_path / "protocol.yaml",
            protocol_hash="protocol-hash",
            receipt_path=receipt_path,
            receipt_sha256_before=receipt_hash,
            expected_git_commit="a" * 40,
            expected_matrix_manifest_sha256="b" * 64,
            expected_freeze_manifest_sha256="c" * 64,
            expected_formal_rows=[{"run_id": run_id}],
        )


def test_v7_archive_state_persists_all_frozen_cifar100_provenance_fields() -> None:
    state = {
        "freeze_manifest_sha256": "f" * 64,
        "benchmark_receipt_sha256": "b" * 64,
        "analysis_sha256": "a" * 64,
        "training_environment_sha256": "e" * 64,
        "cifar100_source_provenance": "environment/CIFAR100_SOURCE_PROVENANCE.json",
        "cifar100_source_provenance_sha256": "1" * 64,
        "cifar100_archive": "data/cifar100/cifar-100-python.tar.gz",
        "cifar100_archive_sha256": "2" * 64,
        "cifar100_train_pickle": "data/cifar100/cifar-100-python/train",
        "cifar100_train_pickle_sha256": "3" * 64,
        "cifar100_test_pickle": "data/cifar100/cifar-100-python/test",
        "cifar100_test_pickle_sha256": "4" * 64,
        "cifar100_meta_pickle": "data/cifar100/cifar-100-python/meta",
        "cifar100_meta_pickle_sha256": "5" * 64,
        "cifar100_split_manifest": "data/manifests/cifar100_seed2024.json",
        "cifar100_split_manifest_sha256": "6" * 64,
    }
    state["test_source"] = state["cifar100_test_pickle"]
    state["test_source_sha256"] = state["cifar100_test_pickle_sha256"]
    payload = v7_archive._archive_state_payload(state, source_commit="a" * 40)

    assert payload["schema"] == v7_archive.ARCHIVE_STATE_SCHEMA
    assert payload["source_commit"] == "a" * 40
    assert {
        key: payload[key] for key in v7_archive._CIFAR100_PROVENANCE_STATE_KEYS
    } == {
        key: state[key] for key in v7_archive._CIFAR100_PROVENANCE_STATE_KEYS
    }

    incomplete = dict(state)
    incomplete.pop("cifar100_train_pickle_sha256")
    with pytest.raises(
        v7_archive.V7EvidenceArchiveError,
        match="cifar100_train_pickle_sha256",
    ):
        v7_archive._archive_state_payload(incomplete, source_commit="a" * 40)
