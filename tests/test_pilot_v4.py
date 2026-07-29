from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pilot_health_gate_v4 as health_gate  # noqa: E402
import validate_v4_pilot as validator  # noqa: E402
from talif_msresnet.config import load_protocol  # noqa: E402
from talif_msresnet.pilot_v4 import (  # noqa: E402
    LONGITUDINAL_LOSS_REDUCTION_DEFINITION,
    LONGITUDINAL_RESIDUAL_COVERAGE_DEFINITION,
    LONGITUDINAL_SURROGATE_COVERAGE_DEFINITION,
    PilotV4Error,
    _validate_longitudinal_condition,
    expected_pilot_configs,
    require_v4_author_freeze,
    resolve_pilot_block,
    validate_fixed_batch_shape,
)


PROTOCOL_PATH = ROOT / "configs" / "protocol_v4_talif_only.yaml"


@pytest.mark.parametrize(
    "status",
    ("PASS", "FAIL", "INVALID"),
)
def test_v4_health_final_verdict_includes_dataset(status: str) -> None:
    assert health_gate._health_verdict(status, "cifar100") == (
        f"V4_HEALTH_GATE_{status} dataset=cifar100"
    )


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
            seed=1,
            block_hash="b" * 64,
            pilot_output_root=output_root,
        )

    monkeypatch.setattr(validator, "load_protocol", lambda _path: protocol)
    monkeypatch.setattr(validator, "require_v4_author_freeze", lambda *_args, **_kwargs: {})
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

    def fake_validate_run(**kwargs):
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


def test_v4_aggregate_binds_one_cross_dataset_training_environment(
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
    assert report["dataset_acceptance_thresholds_independent"] is True
    assert report["cross_dataset_training_environment_hash_equality_required"] is True
    assert report["cross_dataset_shared_weight_or_split_manifest_hash_equality_required"] is False
    assert (
        report["datasets"]["cifar100"]["shared_weight_sha256"]
        != report["datasets"]["cifar10dvs"]["shared_weight_sha256"]
    )
    assert (
        report["datasets"]["cifar100"]["split_manifest_sha256"]
        != report["datasets"]["cifar10dvs"]["split_manifest_sha256"]
    )


def test_v4_aggregate_rejects_cross_dataset_training_environment_mismatch(
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


def test_v4_freeze_block_and_pilot_configs_are_dataset_specific() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    freeze = require_v4_author_freeze(protocol, repository_root=ROOT)
    assert freeze["responsible_author"] == "Yanzhao Deng"

    expected = {
        "cifar100": (1975342236, 6),
        "cifar10dvs": (1983855948, 10),
    }
    for dataset, (seed, time_steps) in expected.items():
        block = resolve_pilot_block(protocol, dataset, repository_root=ROOT)
        configs = expected_pilot_configs(protocol, dataset)
        assert block.seed == seed
        assert [config.model.condition for config in configs] == ["C1", "C2"]
        assert {config.runtime.seed for config in configs} == {seed}
        assert {config.model.time_steps for config in configs} == {time_steps}
        assert {config.model.terminal_neuron_mode for config in configs} == {"topology_required"}


def test_v4_freeze_rejects_unrecorded_coauthor_claim() -> None:
    protocol = copy.deepcopy(load_protocol(PROTOCOL_PATH))
    protocol["protocol_status"]["confirmed_by"] += ", Peng Yan"
    with pytest.raises(PilotV4Error, match="unrecorded co-author"):
        require_v4_author_freeze(protocol, repository_root=ROOT)


def test_v4_health_shape_contract_covers_static_and_event_inputs() -> None:
    assert validate_fixed_batch_shape(
        "cifar100",
        (64, 3, 32, 32),
        minimum_batch_size=64,
        time_steps=6,
        in_channels=3,
    ) == [64, 3, 32, 32]
    assert validate_fixed_batch_shape(
        "cifar10dvs",
        (64, 10, 2, 48, 48),
        minimum_batch_size=64,
        time_steps=10,
        in_channels=2,
    ) == [64, 10, 2, 48, 48]
    with pytest.raises(PilotV4Error, match="CIFAR10-DVS"):
        validate_fixed_batch_shape(
            "cifar10dvs",
            (64, 2, 48, 48),
            minimum_batch_size=64,
            time_steps=10,
            in_channels=2,
        )


def _longitudinal_pass(protocol: dict[str, Any]) -> dict[str, Any]:
    acceptance = protocol["pilot_acceptance"]["longitudinal_health"]
    return {
        "pass": True,
        "epochs": acceptance["epochs"],
        "train_batches_per_epoch": acceptance["train_batches_per_epoch"],
        "validation_batches_per_epoch": acceptance["validation_batches_per_epoch"],
        "epoch_metrics": [{"epoch": epoch} for epoch in range(acceptance["epochs"])],
        "residual_gradient_batch_coverage_definition": (
            LONGITUDINAL_RESIDUAL_COVERAGE_DEFINITION
        ),
        "minimum_epoch_residual_gradient_batch_coverage": 0.95,
        "surrogate_support_coverage_definition": (
            LONGITUDINAL_SURROGATE_COVERAGE_DEFINITION
        ),
        "minimum_epoch_surrogate_support_coverage": 0.001,
        "loss_reduction_definition": LONGITUDINAL_LOSS_REDUCTION_DEFINITION,
        "initial_validation_loss": 1.0,
        "final_validation_loss": 0.99,
        "loss_reduction_fraction": 0.01,
        "parameter_norm_ratio": 1.0,
        "nonfinite_observations": 0,
    }


def test_v4_longitudinal_health_is_fail_closed_at_every_threshold() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    acceptance = protocol["pilot_acceptance"]["longitudinal_health"]
    passing = _longitudinal_pass(protocol)
    assert _validate_longitudinal_condition(passing, condition="C2", acceptance=acceptance) == []

    mutations = {
        "minimum_epoch_residual_gradient_batch_coverage": 0.949,
        "minimum_epoch_surrogate_support_coverage": 0.0009,
        "loss_reduction_fraction": 0.009,
        "parameter_norm_ratio": 2.001,
        "nonfinite_observations": 1,
    }
    for field, value in mutations.items():
        report = {**passing, field: value}
        failures = _validate_longitudinal_condition(report, condition="C2", acceptance=acceptance)
        assert failures, field


def test_v4_longitudinal_runner_executes_exact_batch_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    base = expected_pilot_configs(protocol, "cifar100")[0]
    acceptance = protocol["pilot_acceptance"]["longitudinal_health"]
    train_loader = [(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))] * 20
    val_loader = [
        (torch.full((2, 1), float(index)), torch.zeros(2, dtype=torch.long))
        for index in range(8)
    ]
    monkeypatch.setattr(
        health_gate,
        "build_loaders",
        lambda *_args, **_kwargs: {
            "train": train_loader,
            "val": val_loader,
            "_manifest": {"manifest_sha256": "a" * 64},
        },
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    class Scheduler:
        def step(self) -> None:
            return None

    monkeypatch.setattr(
        health_gate.legacy_gate,
        "_build_training_objects",
        lambda *_args, **_kwargs: (model, optimizer, Scheduler(), []),
    )
    monkeypatch.setattr(
        health_gate.legacy_gate, "_set_ta_enabled", lambda *_args: None
    )
    monkeypatch.setattr(health_gate.legacy_gate, "_grad_scaler", lambda: object())

    def fake_train(
        current_model,
        loader,
        _optimizer,
        _device,
        epoch,
        _config,
        logger,
        _scaler,
    ):
        batches = list(loader)
        for index, _batch in enumerate(batches, start=1):
            logger.log("train_batch", batch=index)
        with torch.no_grad():
            current_model.weight.mul_(0.999)
        return {
            "loss": 1.0 - 0.02 * epoch,
            "accuracy": 0.1,
            "gradient_mean": 1.0,
            "nonzero_block_gradient_fraction_mean": 1.0,
            "diagnostics": {"surrogate_coverage_min": 0.01},
        }

    validation_batch_values: list[list[float]] = []

    def fake_evaluate(_model, loader, *_args):
        batches = list(loader)
        assert len(batches) == 4
        validation_batch_values.append([float(batch[0][0, 0]) for batch in batches])
        return {"loss": 1.0 - 0.02 * (len(validation_batch_values) - 1), "accuracy": 0.1}

    monkeypatch.setattr(health_gate, "train_one_epoch", fake_train)
    monkeypatch.setattr(health_gate, "evaluate", fake_evaluate)

    report = health_gate.run_longitudinal_condition(
        base,
        condition="C1",
        seed=1975342236,
        device=torch.device("cpu"),
        output_parent=tmp_path,
        acceptance=acceptance,
        expected_split_manifest_sha256="a" * 64,
    )

    assert report["pass"] is True
    assert [row["train_batches"] for row in report["epoch_metrics"]] == [16, 16, 16]
    assert [row["validation_batches"] for row in report["epoch_metrics"]] == [4, 4, 4]
    assert validation_batch_values == [[0.0, 1.0, 2.0, 3.0]] * 3
    assert report["loss_reduction_fraction"] == pytest.approx(0.04)
    assert report["initial_validation_loss"] == pytest.approx(1.0)
    assert report["final_validation_loss"] == pytest.approx(0.96)


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


def _summary(events: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[str]]:
    return validator._v4_threshold_summary(
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


def test_v4_pilot_threshold_summary_requires_late_window_and_gradients() -> None:
    summary, integrity, thresholds = _summary(_epoch_events())
    assert integrity == []
    assert thresholds == []
    assert summary["late_mean_validation_accuracy"] == pytest.approx(0.31)
    assert summary["minimum_post_warmup_residual_gradient_coverage"] == 1.0

    _summary_value, integrity, thresholds = _summary(_epoch_events(late_accuracy=0.20))
    assert integrity == []
    assert any("final-10-epoch mean" in item for item in thresholds)
    assert any("late-to-best" in item for item in thresholds)

    _summary_value, integrity, thresholds = _summary(_epoch_events(post_warmup_zero_fraction=0.051))
    assert integrity == []
    assert any("residual-gradient" in item for item in thresholds)
