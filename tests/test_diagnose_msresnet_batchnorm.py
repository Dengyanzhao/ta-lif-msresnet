"""CPU tests for the non-reporting MS-ResNet BatchNorm diagnostic."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import diagnose_msresnet_batchnorm as diagnosis  # noqa: E402


class _RepeatedBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        value = self.bn(inputs)
        value = self.bn(value + 0.25)
        return self.fc(value.mean(dim=(2, 3)))


def _metric(*, passed: bool, accuracy: float, loss: float) -> dict[str, object]:
    return {
        "passes_existing_overfit_thresholds": passed,
        "accuracy": accuracy,
        "loss": loss,
    }


def _condition(
    *,
    ordinary_pass: bool,
    ordinary_accuracy: float,
    ordinary_loss: float,
    batch_pass: bool,
    batch_accuracy: float,
    batch_loss: float,
    equal: bool = True,
) -> dict[str, object]:
    return {
        "ordinary_eval": _metric(
            passed=ordinary_pass,
            accuracy=ordinary_accuracy,
            loss=ordinary_loss,
        ),
        "bn_only_batch_stats": _metric(
            passed=batch_pass,
            accuracy=batch_accuracy,
            loss=batch_loss,
        ),
        "train_and_bn_only_logits_bitwise_equal": equal,
        "counterfactual_state_unchanged": True,
        "bn_called_once_per_configured_timestep": True,
    }


def test_rejects_known_experiment_seed(tmp_path: Path) -> None:
    args = diagnosis.build_parser().parse_args(
        [
            "--failed-health-report",
            str(tmp_path / "failed.json"),
            "--output",
            str(tmp_path / "diagnosis.json"),
            "--engineering-seed",
            "88",
            "--confirm-nonreporting",
            "NONREPORTING_SINGLE_ATTEMPT",
        ]
    )
    with pytest.raises(diagnosis.DiagnosticError, match="reserved or already consumed"):
        diagnosis.validate_arguments(args)


def test_start_receipt_prevents_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch_root = tmp_path / "engineering_diagnostics"
    monkeypatch_root.mkdir()
    output = monkeypatch_root / "diagnosis_seed314159.json"
    receipt = diagnosis.start_receipt_path(output)
    receipt.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(diagnosis, "ENGINEERING_OUTPUT_ROOT", monkeypatch_root)
    monkeypatch.setattr(diagnosis, "ENGINEERING_OUTPUT", output)
    args = diagnosis.build_parser().parse_args(
        [
            "--failed-health-report",
            str(tmp_path / "failed.json"),
            "--output",
            str(output),
            "--engineering-seed",
            "314159",
            "--confirm-nonreporting",
            "NONREPORTING_SINGLE_ATTEMPT",
        ]
    )
    with pytest.raises(FileExistsError, match="already claimed"):
        diagnosis.validate_arguments(args)


def test_rejects_engineering_seed_shopping(tmp_path: Path) -> None:
    args = diagnosis.build_parser().parse_args(
        [
            "--failed-health-report",
            str(tmp_path / "failed.json"),
            "--output",
            str(tmp_path / "diagnosis_seed271828.json"),
            "--engineering-seed",
            "271828",
            "--confirm-nonreporting",
            "NONREPORTING_SINGLE_ATTEMPT",
        ]
    )

    with pytest.raises(diagnosis.DiagnosticError, match="bound to engineering seed"):
        diagnosis.validate_arguments(args)


def test_output_identity_is_pinned_to_one_path(tmp_path: Path) -> None:
    args = diagnosis.build_parser().parse_args(
        [
            "--failed-health-report",
            str(tmp_path / "failed.json"),
            "--output",
            str(tmp_path / "alternate_seed314159.json"),
            "--engineering-seed",
            "314159",
            "--confirm-nonreporting",
            "NONREPORTING_SINGLE_ATTEMPT",
        ]
    )

    with pytest.raises(diagnosis.DiagnosticError, match="output is pinned"):
        diagnosis.validate_arguments(args)


@pytest.mark.parametrize("flag, message", [("--protocol", "protocol is pinned"), ("--config", "config is pinned")])
def test_reference_protocol_and_config_are_pinned(
    tmp_path: Path,
    flag: str,
    message: str,
) -> None:
    args = diagnosis.build_parser().parse_args(
        [
            flag,
            str(tmp_path / "wrong.yaml"),
            "--failed-health-report",
            str(tmp_path / "failed.json"),
            "--output",
            str(diagnosis.ENGINEERING_OUTPUT),
            "--engineering-seed",
            "314159",
            "--confirm-nonreporting",
            "NONREPORTING_SINGLE_ATTEMPT",
        ]
    )

    with pytest.raises(diagnosis.DiagnosticError, match=message):
        diagnosis.validate_arguments(args)


def test_attempt_receipt_claim_is_exclusive(tmp_path: Path) -> None:
    receipt = tmp_path / "attempt.json"
    diagnosis.claim_attempt_receipt(receipt, {"status": "claimed"})

    with pytest.raises(FileExistsError):
        diagnosis.claim_attempt_receipt(receipt, {"status": "second"})

    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "claimed"


def test_preserved_mode_restores_training_flags_and_batchnorm_buffers() -> None:
    model = _RepeatedBatchNorm()
    model.eval()
    initial_flags = [module.training for module in model.modules()]
    initial_mean = model.bn.running_mean.clone()
    initial_variance = model.bn.running_var.clone()
    initial_count = model.bn.num_batches_tracked.clone()

    with diagnosis.preserved_evaluation_mode(model, "bn_only_batch_stats"):
        assert not model.training
        assert model.bn.training
        model(torch.randn(4, 2, 3, 3))
        assert model.bn.num_batches_tracked > initial_count

    assert [module.training for module in model.modules()] == initial_flags
    assert torch.equal(model.bn.running_mean, initial_mean)
    assert torch.equal(model.bn.running_var, initial_variance)
    assert torch.equal(model.bn.num_batches_tracked, initial_count)


def test_failed_health_report_is_bound_to_hash_and_failure_pattern(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report = {
        "status": "FAIL",
        "pass": False,
        "fatal_error": None,
        "seed": diagnosis.FAILED_HEALTH_SEED,
        "git_commit": diagnosis.FAILED_HEALTH_COMMIT,
        "protocol_hash": diagnosis.FAILED_PROTOCOL_HASH,
        "acceptance_hash": diagnosis.FAILED_ACCEPTANCE_HASH,
        "thresholds": {
            "overfit_steps": diagnosis.OVERFIT_STEPS,
            "minimum_overfit_accuracy": diagnosis.MINIMUM_ACCURACY,
            "maximum_overfit_loss_fraction": diagnosis.MAXIMUM_LOSS_FRACTION,
            "minimum_ta_routed_gradient_coverage": 0.90,
            "maximum_ta_enabled_over_frozen_step_ratio": 3.0,
        },
        "functional_health_by_condition": {
            "C1": {"pass": True},
            "C2": {"pass": True},
            "C3": {"pass": False},
            "C4": {"pass": False},
        },
        "cuda_train_step_timing": {"pass": True},
    }
    path = tmp_path / "failed.json"
    payload = json.dumps(report).encode("utf-8")
    path.write_bytes(payload)
    monkeypatch.setattr(diagnosis, "FAILED_HEALTH_SHA256", hashlib.sha256(payload).hexdigest())

    accepted = diagnosis.validate_failed_health_report(path)

    assert accepted["seed"] == 88
    report["functional_health_by_condition"]["C4"]["pass"] = True
    tampered = json.dumps(report).encode("utf-8")
    path.write_bytes(tampered)
    monkeypatch.setattr(diagnosis, "FAILED_HEALTH_SHA256", hashlib.sha256(tampered).hexdigest())
    with pytest.raises(diagnosis.DiagnosticError, match="does not encode"):
        diagnosis.validate_failed_health_report(path)


def test_canonical_reference_identity_is_exactly_v2r2_seed88() -> None:
    protocol = diagnosis.load_protocol(diagnosis.CANONICAL_PROTOCOL)
    config = diagnosis.load_run_config(
        diagnosis.CANONICAL_CONFIG,
        diagnosis.CANONICAL_PROTOCOL,
    )

    identity = diagnosis.validate_reference_identity(protocol, config)

    assert identity["identity"] == "v2r2_seed88_of80_e120"
    assert identity["protocol_hash"] == diagnosis.FAILED_PROTOCOL_HASH
    assert identity["acceptance_hash"] == diagnosis.FAILED_ACCEPTANCE_HASH


def test_failed_source_binding_rejects_split_drift(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.yaml"
    config = tmp_path / "config.yaml"
    protocol.write_text("protocol", encoding="utf-8")
    config.write_text("config", encoding="utf-8")
    source = {
        "dataset": "cifar100",
        "protocol_sha256": diagnosis.sha256_file(protocol),
        "config_sha256": diagnosis.sha256_file(config),
        "split_manifest_sha256": "a" * 64,
    }

    accepted = diagnosis.validate_failed_source_binding(
        {"source": source},
        protocol_path=protocol,
        config_path=config,
        split_manifest={"manifest_sha256": "a" * 64},
    )

    assert accepted["split_manifest_sha256"] == "a" * 64
    with pytest.raises(diagnosis.DiagnosticError, match="data split differs"):
        diagnosis.validate_failed_source_binding(
            {"source": source},
            protocol_path=protocol,
            config_path=config,
            split_manifest={"manifest_sha256": "b" * 64},
        )


def test_bn_only_matches_train_mode_and_records_each_reuse() -> None:
    torch.manual_seed(7)
    model = _RepeatedBatchNorm()
    model.train()
    for _ in range(3):
        model(torch.randn(4, 2, 3, 3))
    batch = (torch.randn(4, 2, 3, 3), torch.tensor([0, 1, 0, 1]))
    state_before = diagnosis.model_state_sha256(model)

    train_metrics, train_logits = diagnosis.evaluate_fixed_batch(
        model,
        batch,
        mode="train_batch_stats",
    )
    bn_metrics, bn_logits = diagnosis.evaluate_fixed_batch(
        model,
        batch,
        mode="bn_only_batch_stats",
    )
    eval_metrics, _eval_logits = diagnosis.evaluate_fixed_batch(
        model,
        batch,
        mode="eval_running_stats",
        capture_batchnorm=True,
    )

    assert torch.equal(train_logits, bn_logits)
    assert train_metrics["logits_sha256"] == bn_metrics["logits_sha256"]
    observations = eval_metrics["batchnorm_inputs_by_layer_and_timestep"]["bn"]
    assert [row["time_index_zero_based"] for row in observations] == [0, 1]
    assert all(len(row["batch_mean"]) == 2 for row in observations)
    assert all(row["mean_rmse"] >= 0 for row in observations)
    assert diagnosis.model_state_sha256(model) == state_before


def test_root_cause_decision_requires_ms_failure_and_bn_only_recovery() -> None:
    reports = {
        "C1": _condition(
            ordinary_pass=True,
            ordinary_accuracy=1.0,
            ordinary_loss=0.2,
            batch_pass=True,
            batch_accuracy=1.0,
            batch_loss=0.1,
        ),
        "C2": _condition(
            ordinary_pass=True,
            ordinary_accuracy=0.875,
            ordinary_loss=0.3,
            batch_pass=True,
            batch_accuracy=0.875,
            batch_loss=0.2,
        ),
        "C3": _condition(
            ordinary_pass=False,
            ordinary_accuracy=0.125,
            ordinary_loss=8.5,
            batch_pass=True,
            batch_accuracy=0.875,
            batch_loss=0.4,
        ),
        "C4": _condition(
            ordinary_pass=False,
            ordinary_accuracy=0.375,
            ordinary_loss=1.8,
            batch_pass=True,
            batch_accuracy=0.75,
            batch_loss=0.3,
        ),
    }

    supported = diagnosis.root_cause_decision(reports, batch_size=8)
    assert supported["pass"]
    assert supported["status"] == "BN_BATCH_STATS_SUFFICIENT_FOR_FIXED_BATCH_DIVERGENCE"
    assert supported["temporal_batchnorm_status"].startswith("CANDIDATE_NOT_TESTED")

    reports["C4"]["bn_only_batch_stats"] = _metric(
        passed=False,
        accuracy=0.375,
        loss=1.7,
    )
    rejected = diagnosis.root_cause_decision(reports, batch_size=8)
    assert not rejected["pass"]
    assert rejected["status"] == "BN_BATCH_STATS_NOT_SUFFICIENT"
    assert "C4_batch_stats_restores_pass" in rejected["failed_checks"]


def test_root_cause_decision_is_inconclusive_when_failure_is_not_reproduced() -> None:
    reports = {
        condition: _condition(
            ordinary_pass=True,
            ordinary_accuracy=0.75,
            ordinary_loss=0.4,
            batch_pass=True,
            batch_accuracy=0.75,
            batch_loss=0.3,
        )
        for condition in diagnosis.CONDITIONS_IN_ORDER
    }

    decision = diagnosis.root_cause_decision(reports, batch_size=8)

    assert not decision["pass"]
    assert decision["status"] == "INCONCLUSIVE"
    assert not decision["failure_pattern_reproduced"]
