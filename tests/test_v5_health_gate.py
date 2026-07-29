from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from talif_msresnet.config import load_protocol
from talif_msresnet.pilot_v5 import (
    HEALTH_SEED_DISPOSITION,
    LEARNING_METRICS_ROLE,
    PilotV5Error,
    attempt_receipt_payload,
    exclusive_create_json,
    expected_pilot_configs,
    implementation_health_anomalies,
    json_safe,
    resolve_pilot_block,
    validate_attempt_receipt,
    validate_development_probe_report_payload,
)
from talif_msresnet.utils import sha256_file

PROTOCOL_PATH = ROOT / "configs" / "protocol_v5_talif_only.yaml"
SCRIPT_PATH = ROOT / "scripts" / "pilot_health_gate_v5.py"


def _load_gate():
    name = "pilot_health_gate_v5_test_module"
    spec = importlib.util.spec_from_file_location(name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _protocol_and_block(dataset: str = "cifar100"):
    protocol = load_protocol(PROTOCOL_PATH)
    block = resolve_pilot_block(protocol, dataset, repository_root=ROOT)
    return protocol, block


def _development_report(block, dataset: str) -> dict[str, Any]:
    binding = block.development_probe_binding
    item = binding["reports"][dataset]
    return {
        "artifact_class": binding["artifact_class"],
        "reporting_eligibility": binding["reporting_eligibility"],
        "confirmatory_analysis_eligibility": binding[
            "confirmatory_analysis_eligibility"
        ],
        "status": binding["expected_status"],
        "pass": None,
        "thresholds_evaluated": [],
        "dataset": dataset,
        "conditions": ["C1", "C2"],
        "plan": binding["plan"],
        "plan_file_sha256": binding["plan_file_sha256"],
        "plan_hash": binding["plan_hash"],
        "git_commit": binding["expected_git_commit"],
        "tracked_clean": binding["expected_tracked_clean"],
        "integrity_anomalies": [],
        "development_seeds": {
            "fixed_batch": item["fixed_batch_seed"],
            "formal_schedule": item["formal_schedule_seed"],
            "disposition": "DEVELOPMENT_ONLY_EXCLUDE_FROM_V5_HEALTH_PILOT_AND_FORMAL",
        },
        "source": {
            "development_entrypoint_sha256": binding[
                "development_entrypoint_sha256"
            ],
            "training_source_sha256": binding["training_source_sha256"],
            "v4_termination_record_sha256": binding[
                "v4_termination_record_sha256"
            ],
        },
        "v4_failure_evidence": {
            "receipt_sha256": binding["v4_attempt_receipt_sha256"],
            "report_sha256": binding["v4_failure_report_sha256"],
        },
        "fixed_batch_probe_by_condition": {
            "C1": {"integrity_anomalies": []},
            "C2": {"integrity_anomalies": []},
        },
        "formal_schedule_probe_by_condition": {
            "C1": {"integrity_anomalies": []},
            "C2": {"integrity_anomalies": []},
        },
    }


def _checkpoint(condition: str) -> dict[str, Any]:
    value = {
        "pass": True,
        "comparison": "bitwise exact next training step",
        "resume_loader": "talif_msresnet.train._load_resume",
        "checkpoint_name": "last.pt",
        "resume_epoch_match": True,
        "mismatch_count": 0,
        "mismatch_paths": [],
    }
    if condition == "C2":
        value.update(
            {
                "ta_activation_boundary_crossed_on_resume": True,
                "ta_enabled_at_checkpoint": False,
                "ta_enabled_on_next_step": True,
                "checkpoint_epoch_zero_based": 4,
                "expected_start_epoch_zero_based": 5,
                "resumed_start_epoch_zero_based": 5,
                "ta_activation_epoch_zero_based": 5,
            }
        )
    return value


def _fixed_result(condition: str, seed: int) -> dict[str, Any]:
    ta = (
        {
            "pass": True,
            "every_ta_parameter_tensor_has_nonzero_routed_coverage": True,
            "all_ta_gradients_present_and_finite_on_every_probe": True,
            "no_nonzero_gradient_in_unrouted_slots": True,
            "routed_parameter_slots": 4,
            "covered_parameter_slots": 4,
        }
        if condition == "C2"
        else {"pass": True, "status": "not_applicable_lif_condition"}
    )
    return {
        "condition": condition,
        "seed": seed,
        "optimizer_regime": "base_lr_without_warmup_scheduler_steps",
        "learning_metrics_role": LEARNING_METRICS_ROLE,
        "observations": [
            {
                "step": 0,
                "loss": 5.0,
                "accuracy": 0.0,
                "parameter_delta": {
                    "l2": 0.0,
                    "updated_parameter_tensors": 0,
                },
            },
            {
                "step": 80,
                "loss": 6.0,
                "accuracy": 0.0,
                "parameter_delta": {
                    "l2": 1.0,
                    "updated_parameter_tensors": 2,
                },
            },
        ],
        "training_history": [
            {"step": step, "loss": 10.0 + step, "accuracy": 0.0}
            for step in range(1, 81)
        ],
        "shared_conv_and_fc_gradients": {
            "pass": True,
            "required_parameter_count": 3,
            "passed_parameter_count": 3,
        },
        "ta_gradients": ta,
        "checkpoint_boundary": _checkpoint(condition),
        "integrity_anomalies": [],
    }


def _formal_result(condition: str, seed: int, ta_lr_scale: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for epoch in range(6):
        enabled = condition == "C2" and epoch >= 5
        rows.append(
            {
                "epoch": epoch,
                "ta_enabled": enabled,
                "train_batches": 16,
                "validation_batches": 4,
                "absolute_lr_schedule_match": True,
                "expected_base_lr": 0.005,
                "train_loss": 100.0 + epoch,
                "train_accuracy": 0.0,
                "validation_loss": 200.0 + epoch,
                "validation_accuracy": 0.0,
                "gradient_mean": 0.1,
                "residual_gradient_batch_coverage": 0.01,
                "surrogate_support_coverage": 0.0,
                "ta_parameter_delta": {"l2": 1.0 if enabled else 0.0},
                "ta_optimizer_state_count_after": 1 if enabled else 0,
                "ta_gradient_observation": {
                    "parameter_tensor_count": 2 if enabled else 0,
                    "gradient_present_count": 2 if enabled else 0,
                    "finite_gradient_count": 2 if enabled else 0,
                    "nonzero_gradient_count": 1 if enabled else 0,
                },
                "ta_lr_over_base_lr": ta_lr_scale if enabled else None,
            }
        )
    return {
        "condition": condition,
        "seed": seed,
        "epochs": 6,
        "train_batches_per_epoch": 16,
        "validation_batches_per_epoch": 4,
        "ta_activation_epoch_zero_based": 5,
        "learning_metrics_role": LEARNING_METRICS_ROLE,
        "epoch_metrics": rows,
        "integrity_anomalies": [],
    }


def _passing_results(protocol, block):
    reference = expected_pilot_configs(protocol, block.dataset)[0]
    fixed = {
        condition: _fixed_result(condition, block.health_seed)
        for condition in block.conditions
    }
    formal = {
        condition: _formal_result(
            condition, block.health_seed, reference.optimizer.ta_lr_scale
        )
        for condition in block.conditions
    }
    return reference, fixed, formal


def test_v5_health_and_pilot_seeds_are_disjoint() -> None:
    protocol = load_protocol(PROTOCOL_PATH)
    observed: set[int] = set()
    for dataset in ("cifar100", "cifar10dvs"):
        block = resolve_pilot_block(protocol, dataset, repository_root=ROOT)
        configs = expected_pilot_configs(protocol, dataset)
        assert block.health_seed != block.pilot_seed
        assert block.seed == block.pilot_seed
        assert {config.runtime.seed for config in configs} == {block.pilot_seed}
        assert block.health_seed not in {config.runtime.seed for config in configs}
        assert not ({block.health_seed, block.pilot_seed} & observed)
        observed.update((block.health_seed, block.pilot_seed))


def test_v5_attempt_receipt_is_exclusive_and_consumes_only_health_seed(
    tmp_path: Path,
) -> None:
    protocol, original = _protocol_and_block()
    block = dataclasses.replace(
        original,
        attempt_receipt=tmp_path / "health.attempt.json",
    )
    config = expected_pilot_configs(protocol, block.dataset)[0]
    config_path = tmp_path / "reference.yaml"
    config_path.write_text("reference: true\n", encoding="utf-8")
    receipt = attempt_receipt_payload(
        block=block,
        protocol_path=PROTOCOL_PATH,
        config_path=config_path,
        config=config,
        git_identity={"git_commit": "a" * 40, "tracked_clean": True},
        repository_root=ROOT,
        runtime_sources={"scripts/pilot_health_gate_v5.py": "b" * 64},
    )
    exclusive_create_json(block.attempt_receipt, receipt)
    with pytest.raises(FileExistsError):
        exclusive_create_json(block.attempt_receipt, receipt)
    observed = validate_attempt_receipt(
        block.attempt_receipt,
        block=block,
        protocol_path=PROTOCOL_PATH,
        repository_root=ROOT,
        reference_config=config,
    )
    assert observed["seed"] == block.health_seed
    assert observed["health_seed_disposition"] == HEALTH_SEED_DISPOSITION
    assert observed["pilot_seed"] == block.pilot_seed
    assert "UNCONSUMED" in observed["pilot_seed_disposition"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda report: report.update(status="ERROR"), "status mismatch"),
        (lambda report: report.update(git_commit="0" * 40), "git_commit mismatch"),
        (lambda report: report.update(plan_hash="0" * 64), "plan_hash mismatch"),
        (
            lambda report: report["development_seeds"].update(fixed_batch=123),
            "fixed-batch seed mismatch",
        ),
        (
            lambda report: report["fixed_batch_probe_by_condition"]["C2"].update(
                integrity_anomalies=["bad"]
            ),
            "has anomalies",
        ),
    ),
)
def test_v5_development_evidence_semantics_are_bound(mutation, message: str) -> None:
    _protocol, block = _protocol_and_block()
    report = _development_report(block, "cifar100")
    validate_development_probe_report_payload(
        report,
        dataset="cifar100",
        binding=block.development_probe_binding,
    )
    mutation(report)
    with pytest.raises(PilotV5Error, match=message):
        validate_development_probe_report_payload(
            report,
            dataset="cifar100",
            binding=block.development_probe_binding,
        )


def test_v5_health_uses_no_natural_learning_threshold() -> None:
    protocol, block = _protocol_and_block()
    reference, fixed, formal = _passing_results(protocol, block)
    assert (
        implementation_health_anomalies(
            fixed,
            formal,
            block=block,
            protocol=protocol,
            reference_config=reference,
        )
        == []
    )
    # Loss gets worse and accuracy remains zero; both are descriptive and finite.
    fixed["C1"]["observations"][-1]["loss"] = 10_000.0
    formal["C2"]["epoch_metrics"][-1]["validation_accuracy"] = 0.0
    assert (
        implementation_health_anomalies(
            fixed,
            formal,
            block=block,
            protocol=protocol,
            reference_config=reference,
        )
        == []
    )


@pytest.mark.parametrize(
    ("mutation", "fragment"),
    (
        (
            lambda fixed, formal: fixed["C1"]["training_history"][0].update(
                loss=math.nan
            ),
            "nonfinite observation",
        ),
        (
            lambda fixed, formal: fixed["C1"][
                "shared_conv_and_fc_gradients"
            ].update(pass_=False),
            "shared convolution",
        ),
        (
            lambda fixed, formal: formal["C2"]["epoch_metrics"][5].update(
                ta_enabled=False
            ),
            "TA activation state",
        ),
        (
            lambda fixed, formal: formal["C1"]["epoch_metrics"][2].update(
                absolute_lr_schedule_match=False
            ),
            "absolute LR",
        ),
        (
            lambda fixed, formal: fixed["C2"]["checkpoint_boundary"].update(
                ta_activation_boundary_crossed_on_resume=False
            ),
            "exact resume",
        ),
    ),
)
def test_v5_health_aggregates_only_implementation_anomalies(mutation, fragment: str) -> None:
    protocol, block = _protocol_and_block()
    reference, fixed, formal = _passing_results(protocol, block)
    mutation(fixed, formal)
    # The shared-gradient mutation uses an identifier-safe helper key then applies it.
    shared = fixed["C1"]["shared_conv_and_fc_gradients"]
    if "pass_" in shared:
        shared["pass"] = shared.pop("pass_")
    anomalies = implementation_health_anomalies(
        fixed,
        formal,
        block=block,
        protocol=protocol,
        reference_config=reference,
    )
    assert any(fragment in item for item in anomalies)


def test_v5_nonfinite_reports_are_written_as_standard_json(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    safe = json_safe({"status": "FAIL", "value": math.nan, "nested": [math.inf]})
    exclusive_create_json(path, safe)
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    assert json.loads(raw) == {"nested": [None], "status": "FAIL", "value": None}


def test_v5_runtime_source_hashes_bind_pilot_dependency_chain() -> None:
    gate = _load_gate()
    bound = {path.resolve() for path in gate.RUNTIME_SOURCE_PATHS}

    assert {
        (ROOT / "src" / "talif_msresnet" / "pilot_v3.py").resolve(),
        (ROOT / "src" / "talif_msresnet" / "pilot_v4.py").resolve(),
        (ROOT / "src" / "talif_msresnet" / "pilot_v5.py").resolve(),
    } <= bound


def test_v5_fatal_report_has_error_semantics(tmp_path: Path) -> None:
    gate = _load_gate()
    _protocol, original = _protocol_and_block()
    receipt = tmp_path / "health.attempt.json"
    receipt.write_text("{}\n", encoding="utf-8")
    block = dataclasses.replace(original, attempt_receipt=receipt)
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        report = gate._fatal_report(
            block=block,
            git_identity={"git_commit": "a" * 40, "tracked_clean": True},
            runtime_sources={"source.py": "b" * 64},
            exc=exc,
        )
    assert report["status"] == "ERROR"
    assert report["pass"] is False
    assert report["thresholds_evaluated"] == []
    assert report["seed_disposition"] == HEALTH_SEED_DISPOSITION
    assert report["attempt_receipt_sha256"] == sha256_file(receipt)


def test_v5_main_claims_receipt_before_gate_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = _load_gate()
    protocol, original = _protocol_and_block()
    block = dataclasses.replace(
        original,
        attempt_receipt=tmp_path / "attempt.json",
        health_output=tmp_path / "health.json",
    )
    config = expected_pilot_configs(protocol, block.dataset)[0]
    config_path = tmp_path / "reference.yaml"
    config_path.write_text("reference: true\n", encoding="utf-8")
    events: list[str] = []
    real_exclusive = exclusive_create_json

    monkeypatch.setattr(gate, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(gate, "resolve_pilot_block", lambda *args, **kwargs: block)
    monkeypatch.setattr(gate, "require_v5_author_freeze", lambda *args, **kwargs: {"signoff_sha256": "f" * 64})
    monkeypatch.setattr(gate, "_reference_config", lambda *args, **kwargs: (config_path, config))
    monkeypatch.setattr(gate, "validate_development_probe_evidence", lambda *args, **kwargs: {})
    monkeypatch.setattr(gate, "repository_git_identity", lambda root: {"git_commit": "a" * 40, "tracked_clean": True})
    monkeypatch.setattr(gate.calibration, "require_head_bound_sources", lambda paths: [])
    monkeypatch.setattr(gate, "_runtime_source_hashes", lambda: {"source.py": "b" * 64})

    def create(path, value):
        events.append("receipt" if Path(path) == block.attempt_receipt else "report")
        return real_exclusive(path, value)

    def run_gate(**kwargs):
        assert block.attempt_receipt.is_file()
        events.append("gate")
        return {"status": "PASS", "pass": True, "thresholds_evaluated": []}

    monkeypatch.setattr(gate, "exclusive_create_json", create)
    monkeypatch.setattr(gate, "run_gate", run_gate)
    result = gate.main(
        [
            "--protocol",
            str(PROTOCOL_PATH),
            "--config-dir",
            str(tmp_path),
            "--dataset",
            block.dataset,
            "--device",
            "cpu",
        ]
    )
    assert result == 0
    assert events == ["receipt", "gate", "report"]


def test_v5_current_runtime_rebuilds_health_seed_batch_for_pilot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _load_gate()
    protocol, block = _protocol_and_block()
    reference = expected_pilot_configs(protocol, block.dataset)[0]
    hardware = {"name": "RTX 5090", "device": "cpu"}
    environment = gate.environment_manifest()
    environment.update(
        {
            "hardware": hardware,
            "precision": "float32",
            "amp": False,
            "cublas_workspace_config": gate.os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu_idle_precheck": {"pass": True, "device_uuid": "GPU-test"},
        }
    )
    inputs = torch.zeros((10, 3, 32, 32))
    targets = torch.zeros((10,), dtype=torch.long)
    batch_hash = "d" * 64
    seen: dict[str, int] = {}

    def load_batch(config, *, seed, required_batch_size):
        seen["seed"] = seed
        return (
            config,
            inputs,
            targets,
            {"manifest_sha256": "c" * 64},
            {"split_source_fingerprint": "e" * 64},
        )

    monkeypatch.setattr(gate, "repository_git_identity", lambda root: {"git_commit": "a" * 40, "tracked_clean": True})
    monkeypatch.setattr(gate.legacy_gate, "seed_everything", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate.legacy_gate, "require_target_cuda", lambda *args, **kwargs: hardware)
    monkeypatch.setattr(gate.legacy_gate, "validate_runtime_environment", lambda *args, **kwargs: None)
    monkeypatch.setattr(gate.legacy_gate, "gpu_idle_precheck", lambda device: {"pass": True, "device_uuid": "GPU-test"})
    monkeypatch.setattr(gate.legacy_gate, "tensor_batch_sha256", lambda *args: batch_hash)
    monkeypatch.setattr(gate.v3_gate, "load_fixed_real_batch", load_batch)
    monkeypatch.setattr(gate, "_runtime_source_hashes", lambda: {"source.py": "b" * 64})
    report = {
        "status": "PASS",
        "pass": True,
        "git_commit": "a" * 40,
        "environment": environment,
        "source": {
            "split_manifest_sha256": "c" * 64,
            "fixed_batch_sha256": batch_hash,
            "fixed_batch_shape": [8, 3, 32, 32],
            "fixed_batch_size": 8,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "split_source_fingerprint": "e" * 64,
            "runtime_sources_sha256": {"source.py": "b" * 64},
        },
    }
    result = gate.validate_current_runtime_against_health_report(
        report,
        protocol=protocol,
        reference_config=reference,
        block=block,
        device="cpu",
    )
    assert seen["seed"] == block.health_seed
    assert result["health_seed"] == block.health_seed
    assert result["pilot_seed"] == reference.runtime.seed == block.pilot_seed
