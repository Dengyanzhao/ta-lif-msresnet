from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_v5_pilot as validator

from talif_msresnet.config import load_protocol
from talif_msresnet.utils import sha256_file

PROTOCOL_PATH = ROOT / "configs" / "protocol_v5_talif_only.yaml"


def _stubbed_aggregate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_hashes: dict[str, str],
) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL_PATH)
    protocol_path = tmp_path / protocol["artifact_paths"]["protocol"]
    config_dir = tmp_path / protocol["artifact_paths"]["pilot_matrix"]
    protocol_path.parent.mkdir(parents=True)
    config_dir.mkdir(parents=True)
    protocol_path.write_bytes(PROTOCOL_PATH.read_bytes())
    manifest_path = config_dir / "matrix_manifest.json"
    csv_path = config_dir / "run_manifest.csv"
    manifest_path.write_text("{}\n", encoding="utf-8")
    csv_path.write_text("run_id\n", encoding="utf-8")

    rows: list[dict[str, str]] = []
    expected_configs: dict[tuple[str, str], Any] = {}
    blocks: dict[str, Any] = {}
    for dataset in ("cifar100", "cifar10dvs"):
        output_root = tmp_path / "results" / dataset
        output_root.mkdir(parents=True)
        for condition in ("C1", "C2"):
            run_id = f"E1_{dataset}_{condition}"
            (output_root / run_id).mkdir()
            rows.append({"dataset": dataset, "condition": condition})
            expected_configs[(dataset, condition)] = SimpleNamespace(
                runtime=SimpleNamespace(run_id=run_id)
            )
        blocks[dataset] = SimpleNamespace(
            dataset=dataset,
            health_seed=1,
            pilot_seed=2,
            block_hash="b" * 64,
            pilot_output_root=output_root,
        )

    monkeypatch.setattr(validator, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(
        validator, "require_v5_author_freeze", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        validator,
        "resolve_pilot_block",
        lambda _protocol, dataset, **_kwargs: blocks[dataset],
    )
    monkeypatch.setattr(
        validator,
        "_load_generated_matrix",
        lambda **_kwargs: (rows, expected_configs, [], manifest_path, csv_path),
    )
    monkeypatch.setattr(
        validator,
        "_bound_block_evidence",
        lambda **_kwargs: ({"health": {}}, []),
    )

    def fake_validate_run(**kwargs: Any) -> tuple[dict[str, Any], str, str, str]:
        dataset = kwargs["block"].dataset
        dataset_index = ("cifar100", "cifar10dvs").index(dataset)
        return (
            {"integrity_failures": [], "threshold_failures": []},
            environment_hashes[dataset],
            ("c", "d")[dataset_index] * 64,
            ("e", "f")[dataset_index] * 64,
        )

    monkeypatch.setattr(validator, "_validate_run", fake_validate_run)
    return validator.validate_pilot(
        protocol_path=protocol_path,
        config_dir=config_dir,
        repository_root=tmp_path,
    )


def test_v5_aggregate_binds_one_cross_dataset_training_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment_hash = "a" * 64
    report = _stubbed_aggregate_report(
        tmp_path,
        monkeypatch,
        {"cifar100": environment_hash, "cifar10dvs": environment_hash},
    )

    assert report["status"] == "PASS"
    assert report["training_environment_sha256"] == environment_hash
    assert report["datasets"]["cifar100"]["health_seed"] == 1
    assert report["datasets"]["cifar100"]["pilot_seed"] == 2
    assert report["dataset_acceptance_thresholds_independent"] is True
    assert report["cross_dataset_training_environment_hash_equality_required"] is True


def test_v5_aggregate_rejects_cross_dataset_training_environment_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = _stubbed_aggregate_report(
        tmp_path,
        monkeypatch,
        {"cifar100": "a" * 64, "cifar10dvs": "b" * 64},
    )

    assert report["status"] == "INVALID"
    assert report["pass"] is False
    assert report["exit_code"] == 2
    assert report["training_environment_sha256"] is None
    assert all(dataset["status"] == "PASS" for dataset in report["datasets"].values())
    assert any(
        "training environment hashes are not identical across dataset blocks" in failure
        for failure in report["integrity_failures"]
    )


def test_v5_validator_refuses_to_overwrite_canonical_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "results" / "pilot" / "v5_talif_only" / "validation.json"
    output.parent.mkdir(parents=True)
    original = b'{"status":"PASS"}\n'
    output.write_bytes(original)
    monkeypatch.setattr(validator, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        validator,
        "load_protocol",
        lambda _path: {"pilot_acceptance": {"validation_output": str(output.relative_to(tmp_path))}},
    )
    monkeypatch.setattr(
        validator,
        "validate_pilot",
        lambda **_kwargs: pytest.fail("existing output must block before validation"),
    )

    assert validator.main(["--protocol", str(tmp_path / "protocol.yaml")]) == 2
    assert output.read_bytes() == original


def test_v5_recovery_preserves_invalid_and_binds_original_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid_path = tmp_path / "results" / "pilot" / "v5_talif_only" / "validation.json"
    invalid_path.parent.mkdir(parents=True)
    runs = {condition: {"run_id": condition} for condition in ("C1", "C2")}
    invalid = {
        "status": "INVALID",
        "pass": False,
        "decision": validator.INVALID_DECISION,
        "protocol_hash": "a" * 64,
        "acceptance_hash": "b" * 64,
        "threshold_failures": [],
        "datasets": {
            dataset: {"runs": runs, "threshold_failures": []}
            for dataset in ("cifar100", "cifar10dvs")
        },
    }
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    health_commmit = validator.RECOVERY_PILOT_EXECUTION_COMMIT
    recovered_datasets = {}
    for dataset in ("cifar100", "cifar10dvs"):
        health_path = invalid_path.parent / f"health_{dataset}.json"
        health_path.write_text(
            json.dumps({"git_commit": health_commmit}), encoding="utf-8"
        )
        recovered_datasets[dataset] = {
            "runs": runs,
            "threshold_failures": [],
            "health_report_path": health_path.relative_to(tmp_path).as_posix(),
        }
    recovered = {
        "status": "PASS",
        "pass": True,
        "validated_at": "2026-07-30T00:00:00+00:00",
        "protocol_hash": validator.RECOVERY_PROTOCOL_HASH,
        "acceptance_hash": "b" * 64,
        "training_environment_sha256": validator.RECOVERY_TRAINING_ENVIRONMENT_SHA256,
        "datasets": recovered_datasets,
    }
    invalid["protocol_hash"] = validator.RECOVERY_PROTOCOL_HASH
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    monkeypatch.setattr(
        validator,
        "sha256_file",
        lambda path: (
            validator.RECOVERY_ORIGINAL_VALIDATION_SHA256
            if Path(path) == invalid_path
            else sha256_file(path)
        ),
    )
    output = invalid_path.with_name(validator.RECOVERY_OUTPUT_NAME)
    monkeypatch.setattr(
        validator,
        "_recovery_release_binding",
        lambda _root: {
            "base_commit": health_commmit,
            "implementation_commit": "8" * 40,
            "implementation_tree": "9" * 40,
            "recovery_commit": "7" * 40,
            "allowed_changed_paths": list(validator.RECOVERY_ALLOWED_CHANGED_PATHS),
            "observed_changes": [
                f"M\t{path}" for path in validator.RECOVERY_ALLOWED_CHANGED_PATHS
            ],
            "release_record": validator.RECOVERY_RELEASE_RECORD,
            "release_record_sha256": "c" * 64,
            "tracked_clean": True,
        },
    )

    recovery = validator._recovery_report(
        report=recovered,
        legacy_report=invalid,
        invalid_path=invalid_path,
        invalid_report=invalid,
        output=output,
        repository_root=tmp_path,
        pilot_execution_commit=health_commmit,
    )

    assert invalid_path.read_text(encoding="utf-8") == json.dumps(invalid)
    assert recovery["status"] == "PASS"
    assert recovery["pilot_execution_commit"] == health_commmit
    assert recovery["recovery_validator_commit"] == "7" * 40
    assert recovery["original_validation"]["sha256"] == (
        validator.RECOVERY_ORIGINAL_VALIDATION_SHA256
    )


def test_v5_recovery_rejects_changed_observations(tmp_path: Path) -> None:
    invalid_path = tmp_path / "validation.json"
    invalid = {
        "status": "INVALID",
        "pass": False,
        "protocol_hash": "a" * 64,
        "acceptance_hash": "b" * 64,
        "threshold_failures": [],
        "datasets": {
            dataset: {"runs": {"C1": {"best": 0.5}}, "threshold_failures": []}
            for dataset in ("cifar100", "cifar10dvs")
        },
    }
    invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
    recovered = {
        "status": "PASS",
        "pass": True,
        "protocol_hash": "a" * 64,
        "acceptance_hash": "b" * 64,
        "datasets": {
            dataset: {
                "runs": {"C1": {"best": 0.6}},
                "threshold_failures": [],
                "health_report_path": "unused.json",
            }
            for dataset in ("cifar100", "cifar10dvs")
        },
    }

    with pytest.raises(validator.PilotValidationError, match="authorized v5 recovery incident"):
        validator._recovery_report(
            report=recovered,
            legacy_report=invalid,
            invalid_path=invalid_path,
            invalid_report=invalid,
            output=tmp_path / "recovery.json",
            repository_root=tmp_path,
            pilot_execution_commit="6" * 40,
        )


def test_v5_path_normalization_accepts_only_the_bound_absolute_root(
    tmp_path: Path,
) -> None:
    expected = "results/pilot/v5_talif_only/cifar100_s1"
    actual = tmp_path / "results" / "pilot" / "v5_talif_only" / "cifar100_s1"
    raw = {"runtime": {"output_dir": str(actual)}, "marker": {"unchanged": True}}

    normalized = validator.common._canonicalize_execution_output_dir(
        raw,
        expected_output_dir=expected,
        actual_output_root=actual,
    )
    assert normalized["runtime"]["output_dir"] == expected
    assert raw["runtime"]["output_dir"] == str(actual)

    for observed in (
        str(tmp_path / "other"),
        "results/pilot/v5_talif_only/other",
        "../cifar100_s1",
    ):
        candidate = {"runtime": {"output_dir": observed}}
        assert validator.common._canonicalize_execution_output_dir(
            candidate,
            expected_output_dir=expected,
            actual_output_root=actual,
        ) is candidate


def test_v5_path_exception_is_disabled_by_default() -> None:
    assert validator.validate_pilot.__kwdefaults__ is not None
    assert (
        validator.validate_pilot.__kwdefaults__["allow_equivalent_absolute_output_dir"]
        is False
    )
    assert validator.common._validate_run.__kwdefaults__ is not None
    assert (
        validator.common._validate_run.__kwdefaults__[
            "allow_equivalent_absolute_output_dir"
        ]
        is False
    )


def test_v5_checkpoint_embedded_config_uses_the_same_path_normalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_best = {"accuracy": 0.5, "loss": 1.0, "epoch": 0}
    expected_hash = "config-hash"
    expected_environment = "e" * 64
    expected_output = "results/pilot/v5_talif_only/cifar100_s1"
    actual_output = tmp_path / expected_output
    observed: list[dict[str, Any]] = []
    checkpoint = {
        "model_state": {"weight": object()},
        "optimizer_state": {"state": {}, "param_groups": []},
        "scheduler_state": {"last_epoch": 0},
        "config_hash": expected_hash,
        "training_environment_sha256": expected_environment,
        "epoch": 0,
        "best_val": expected_best,
        "train_history": [],
        "config": {"runtime": {"output_dir": str(actual_output)}},
    }
    monkeypatch.setattr(
        validator.common,
        "load_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )

    def validate(raw: dict[str, Any], _protocol: dict[str, Any]) -> Any:
        observed.append(raw)
        return SimpleNamespace(config_hash=expected_hash)

    monkeypatch.setattr(validator.common, "validate_run_mapping", validate)

    failures = validator.common._checkpoint_failures(
        tmp_path / "checkpoint.pt",
        label="checkpoint.pt",
        expected_epoch=0,
        expected_best=expected_best,
        expected_history=[],
        expected_config_hash=expected_hash,
        expected_environment_hash=expected_environment,
        protocol={},
        expected_output_dir=expected_output,
        actual_output_root=actual_output,
    )

    assert failures == []
    assert observed[0]["runtime"]["output_dir"] == expected_output
    assert checkpoint["config"]["runtime"]["output_dir"] == str(actual_output)


def _epoch_events(
    *,
    late_accuracy: float = 0.31,
    post_warmup_zero_fraction: float = 0.0,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    best = {"accuracy": 0.31, "loss": 1.0, "epoch": 0}
    for epoch in range(120):
        accuracy = late_accuracy if epoch >= 110 else 0.31
        events.append(
            {
                "event": "epoch_completed",
                "epoch": epoch,
                "train": {
                    "loss": 1.0 - 0.001 * epoch,
                    "accuracy": 0.1,
                    "seconds": 1.0,
                    "all_zero_block_gradient_batch_fraction": (
                        post_warmup_zero_fraction if epoch >= 5 else 0.0
                    ),
                },
                "val": {"accuracy": accuracy, "loss": 1.0},
                "best_val": best,
            }
        )
    return events


def _summary(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    return validator._v5_threshold_summary(
        events,
        epochs=120,
        formal_convergence_threshold=0.60,
        warmup_epochs=5,
        required_best_accuracy=0.30,
        late_window_epochs=10,
        minimum_late_mean=0.25,
        minimum_late_to_best_ratio=0.80,
        minimum_residual_coverage=0.95,
    )


def test_v5_pilot_threshold_summary_requires_late_window_and_gradients() -> None:
    summary, integrity, thresholds = _summary(_epoch_events())
    assert integrity == []
    assert thresholds == []
    assert summary["late_mean_validation_accuracy"] == pytest.approx(0.31)
    assert summary["minimum_post_warmup_residual_gradient_coverage"] == 1.0

    _summary_value, integrity, thresholds = _summary(_epoch_events(late_accuracy=0.20))
    assert integrity == []
    assert any("final-10-epoch mean" in item for item in thresholds)
    assert any("late-to-best" in item for item in thresholds)

    _summary_value, integrity, thresholds = _summary(
        _epoch_events(post_warmup_zero_fraction=0.051)
    )
    assert integrity == []
    assert any("residual-gradient" in item for item in thresholds)
