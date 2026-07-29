from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PLAN_PATH = ROOT / "configs" / "v5_health_calibration.yaml"
V4_PROTOCOL_PATH = ROOT / "configs" / "protocol_v4_talif_only.yaml"
V4_C1_CONFIG_PATH = (
    ROOT / "configs" / "v4_talif_only_pilot_generated" / "E1_cifar100_d20_t6_C1_s1975342236.yaml"
)


def _load_tool():
    path = ROOT / "scripts" / "calibrate_v5_health.py"
    spec = importlib.util.spec_from_file_location("calibrate_v5_health_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def test_v5_design_probe_plan_is_nonreporting_and_seed_isolated() -> None:
    plan = tool.load_plan(PLAN_PATH)
    excluded = set(plan["seed_derivation"]["excluded_values"])
    selected: list[int] = []

    assert plan["prospective_protocol_version"] == 5
    assert plan["plan_status"] == "development_only_not_a_frozen_protocol"
    assert plan["reporting_eligibility"] == "FORBIDDEN_FROM_MANUSCRIPT_RESULTS"
    assert plan["confirmatory_analysis_eligibility"] is False
    assert plan["fixed_batch_probe"]["learning_metrics_role"] == (
        "descriptive_only_no_pass_fail_threshold"
    )
    assert plan["formal_schedule_probe"]["learning_metrics_role"] == (
        "descriptive_only_no_pass_fail_threshold"
    )
    for dataset in ("cifar100", "cifar10dvs"):
        for role in ("fixed_batch_seed", "formal_schedule_seed"):
            selected.append(plan["datasets"][dataset][role]["value"])
    assert len(selected) == len(set(selected)) == 4
    assert not set(selected) & excluded


@pytest.mark.parametrize(
    ("dataset", "role", "expected_digest", "expected_seed"),
    (
        (
            "cifar100",
            "fixed_batch_seed",
            "d8804c33da52016fda950aa1aee7b3e8fb9d24dbba0cf39e590ff475c3bdd06c",
            1515323759,
        ),
        (
            "cifar100",
            "formal_schedule_seed",
            "bff2a3ec0cc77f794f06102f30f3f3a36d19a7082ec2b288eed3c5bf425a014e",
            214400889,
        ),
        (
            "cifar10dvs",
            "fixed_batch_seed",
            "3b5de0020701f8d06f749e67655deebc296ab9b17c2f4df50d8a6f7f0737bfd6",
            117569744,
        ),
        (
            "cifar10dvs",
            "formal_schedule_seed",
            "69f497dca199653d88b3ac9ef14fb175e5d13fb5bb25da8ad16a2313d701545d",
            563701053,
        ),
    ),
)
def test_v5_development_seed_derivation_is_reproducible(
    dataset: str,
    role: str,
    expected_digest: str,
    expected_seed: int,
) -> None:
    plan = tool.load_plan(PLAN_PATH)
    record = plan["datasets"][dataset][role]
    digest, seed = tool._seed_digest(
        plan["seed_derivation"]["namespace"], record["label"], record["counter"]
    )

    assert digest == expected_digest
    assert seed == expected_seed


def test_v5_design_probe_plan_rejects_seed_overlap() -> None:
    plan = tool.load_plan(PLAN_PATH)
    tampered = copy.deepcopy(plan)
    selected = tampered["datasets"]["cifar100"]["fixed_batch_seed"]["value"]
    tampered["seed_derivation"]["excluded_values"].append(selected)

    with pytest.raises(tool.CalibrationError, match="overlaps an excluded"):
        tool.validate_plan(tampered)


def test_v5_design_probe_rejects_a_different_collision_policy() -> None:
    plan = tool.load_plan(PLAN_PATH)
    tampered = copy.deepcopy(plan)
    tampered["seed_derivation"]["collision_policy"] = "accept_first_value"

    with pytest.raises(tool.CalibrationError, match="collision policy"):
        tool.validate_plan(tampered)


def test_v5_design_probe_requires_the_complete_pre_v5_exclusion_ledger() -> None:
    plan = tool.load_plan(PLAN_PATH)
    tampered = copy.deepcopy(plan)
    tampered["seed_derivation"]["excluded_values"].remove(1983855948)

    with pytest.raises(tool.CalibrationError, match="omits pre-v5 values"):
        tool.validate_plan(tampered)


def test_v5_design_probe_plan_rejects_learning_thresholds() -> None:
    plan = tool.load_plan(PLAN_PATH)
    tampered = copy.deepcopy(plan)
    tampered["fixed_batch_probe"]["minimum_ta_routed_gradient_coverage_observation"] = 0.9

    with pytest.raises(tool.CalibrationError, match="tuned cutoff"):
        tool.validate_plan(tampered)


def test_v5_design_probe_output_cannot_escape_or_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tool.load_plan(PLAN_PATH)
    monkeypatch.setattr(tool, "REPOSITORY_ROOT", tmp_path)
    output_root = tmp_path / "results" / "development" / "v5_health_design_probe"
    plan["output_root"] = str(output_root)
    plan["datasets"]["cifar100"]["default_output"] = str(output_root / "attempt01.json")

    outside = tmp_path / "results" / "formal_v5" / "forbidden.json"
    with pytest.raises(tool.CalibrationError, match="must remain under"):
        tool.resolve_output(plan, "cifar100", outside)

    output_root.mkdir(parents=True)
    existing = output_root / "attempt01.json"
    existing.write_text("{}\n", encoding="utf-8")
    with pytest.raises(tool.CalibrationError, match="already exists"):
        tool.resolve_output(plan, "cifar100", existing)


def test_fixed_batch_probe_uses_base_lr_without_warmup_or_learning_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tool.load_run_config(V4_C1_CONFIG_PATH, V4_PROTOCOL_PATH)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=base.optimizer.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75], gamma=0.1)
    seen: dict[str, Any] = {}

    def fake_build(config, _device, *, ta_enabled):
        seen["warmup_epochs"] = config.optimizer.warmup_epochs
        seen["ta_enabled"] = ta_enabled
        return model, optimizer, scheduler, []

    def fake_step(
        current_model,
        current_optimizer,
        _config,
        batch,
        _device,
        _step,
        _routes,
        _accumulator,
    ):
        inputs, targets = batch
        current_optimizer.zero_grad(set_to_none=True)
        logits = current_model(inputs)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        current_optimizer.step()
        return {
            "loss": float(loss.item()),
            "accuracy": 0.0,
            "gradient_mean": 1.0,
            "nonzero_block_gradient_fraction_mean": 1.0,
        }

    def fake_metrics(current_model, batch):
        inputs, targets = batch
        with torch.no_grad():
            logits = current_model(inputs)
            return {
                "loss": float(torch.nn.functional.cross_entropy(logits, targets).item()),
                "accuracy": 0.0,
            }

    monkeypatch.setattr(tool.legacy_gate, "_build_training_objects", fake_build)
    monkeypatch.setattr(tool.legacy_gate, "_step_with_ta_tracking", fake_step)
    monkeypatch.setattr(tool.legacy_gate, "_fixed_batch_metrics", fake_metrics)
    monkeypatch.setattr(tool.legacy_gate, "_new_ta_gradient_accumulator", lambda _model: {})
    monkeypatch.setattr(
        tool.legacy_gate,
        "inspect_shared_gradients",
        lambda _model: {"pass": True, "required_parameter_count": 1},
    )
    monkeypatch.setattr(
        tool.legacy_gate,
        "checkpoint_next_step_check",
        lambda **_kwargs: {"pass": True, "ta_activation_boundary_crossed_on_resume": False},
    )
    batch = (torch.tensor([[1.0, -1.0], [-1.0, 1.0]]), torch.tensor([0, 1]))
    result = tool.run_fixed_batch_condition(
        base,
        condition="C1",
        seed=1515323759,
        device=torch.device("cpu"),
        output_parent=tmp_path,
        batch=batch,
        settings={
            "observation_steps": [0, 1],
            "optimizer_regime": "base_lr_without_warmup_scheduler_steps",
            "learning_metrics_role": "descriptive_only_no_pass_fail_threshold",
            "minimum_ta_routed_gradient_coverage_observation": 0.0,
        },
        checkpoint_path=tmp_path / "C1" / "last.pt",
    )

    assert seen == {"warmup_epochs": 0, "ta_enabled": False}
    assert result["warmup_epochs_for_fixed_batch"] == 0
    assert result["learning_metrics_role"] == "descriptive_only_no_pass_fail_threshold"
    assert "pass" not in result
    assert result["observations"][-1]["parameter_delta"]["l2"] > 0.0


def test_formal_absolute_lr_schedule_is_checked_epoch_by_epoch() -> None:
    config = tool.load_run_config(V4_C1_CONFIG_PATH, V4_PROTOCOL_PATH)
    observed = [tool._expected_epoch_lrs(config, epoch) for epoch in range(8)]

    assert [base for base, _ta in observed] == pytest.approx(
        [0.005, 0.010, 0.015, 0.020, 0.025, 0.025, 0.025, 0.025]
    )
    assert [ta for _base, ta in observed] == pytest.approx(
        [0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0025, 0.0025, 0.0025]
    )
    groups = [
        {"role": "base_decay", "lr": observed[5][0]},
        {"role": "base_no_decay", "lr": observed[5][0]},
        {"role": "ta", "lr": observed[5][1]},
    ]
    assert tool._absolute_lr_schedule_matches(groups, *observed[5])
    groups[-1]["lr"] = 0.123
    assert not tool._absolute_lr_schedule_matches(groups, *observed[5])


def test_deterministic_flags_are_set_before_cuda_precheck(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def fake_seed(seed, *, deterministic):
        assert seed == 1515323759
        assert deterministic is True
        events.append("seed")

    def fake_cuda(device, expected):
        assert events == ["seed"]
        assert device == torch.device("cuda:0")
        assert expected == "RTX 5090"
        events.append("cuda")
        return {"name": "NVIDIA GeForce RTX 5090"}

    def fake_environment(_binding, _hardware):
        assert events == ["seed", "cuda"]
        events.append("environment")

    monkeypatch.setattr(tool.legacy_gate, "seed_everything", fake_seed)
    monkeypatch.setattr(tool.legacy_gate, "require_target_cuda", fake_cuda)
    monkeypatch.setattr(tool.legacy_gate, "validate_runtime_environment", fake_environment)

    device, hardware = tool._require_probe_device(
        1515323759,
        "cuda:0",
        {"expected_gpu_substring": "RTX 5090"},
    )

    assert device == torch.device("cuda:0")
    assert hardware["name"] == "NVIDIA GeForce RTX 5090"
    assert events == ["seed", "cuda", "environment"]


def test_c2_formal_schedule_probe_crosses_activation_without_learning_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_module = importlib.import_module("talif_msresnet.train")
    base = tool.load_run_config(V4_C1_CONFIG_PATH, V4_PROTOCOL_PATH)

    class TinyWindow(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.center = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
            self.raw_width = torch.nn.Parameter(torch.tensor([0.0, 0.0]))

    class TinyTA(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_weight = torch.nn.Parameter(torch.tensor(0.0))
            self.window = TinyWindow()

    def fake_build(config, device, *, ta_enabled):
        model = TinyTA().to(device)
        optimizer, scheduler, ta_parameters, _names, _activation = (
            train_module.build_optimizer_and_scheduler(model, config)
        )
        tool.legacy_gate._set_ta_enabled(ta_parameters, ta_enabled)
        return model, optimizer, scheduler, ta_parameters

    def fake_train(
        model,
        loader,
        optimizer,
        _device,
        epoch,
        _config,
        logger,
        _scaler,
    ):
        assert len(list(loader)) == 1
        optimizer.zero_grad(set_to_none=True)
        loss = (model.base_weight - 1.0).square()
        if model.window.center.requires_grad:
            loss = loss + (model.window.center - 1.0).square().sum()
            loss = loss + (model.window.raw_width - 1.0).square().sum()
        loss.backward()
        optimizer.step()
        logger.log("train_batch", epoch=epoch, batch=1)
        return {
            "loss": float(loss.detach().item()),
            "accuracy": 0.0,
            "gradient_mean": 1.0,
            "nonzero_block_gradient_fraction_mean": 1.0,
            "diagnostics": {"surrogate_coverage_min": 0.25},
        }

    validation_calls = 0

    def fake_evaluate(_model, loader, _device, _config, _split):
        nonlocal validation_calls
        assert len(list(loader)) == 1
        validation_calls += 1
        return {"loss": float(validation_calls), "accuracy": 0.0}

    monkeypatch.setattr(tool.v3_gate, "_resolved_data_config", lambda config: config)
    monkeypatch.setattr(
        tool,
        "build_loaders",
        lambda *_args, **_kwargs: {
            "train": [(torch.zeros(1), torch.zeros(1, dtype=torch.long))],
            "val": [(torch.zeros(1), torch.zeros(1, dtype=torch.long))],
            "_manifest": {"manifest_sha256": "a" * 64},
        },
    )
    monkeypatch.setattr(tool.legacy_gate, "_build_training_objects", fake_build)
    monkeypatch.setattr(tool, "train_one_epoch", fake_train)
    monkeypatch.setattr(tool, "evaluate", fake_evaluate)
    monkeypatch.setattr(tool.legacy_gate, "seed_everything", lambda *_args, **_kwargs: None)

    report = tool.run_formal_schedule_condition(
        base,
        condition="C2",
        seed=214400889,
        device=torch.device("cpu"),
        output_parent=tmp_path,
        settings={
            "epochs": 7,
            "train_batches_per_epoch": 1,
            "validation_batches_per_epoch": 1,
            "required_activation_epoch_zero_based": 5,
            "learning_metrics_role": "descriptive_only_no_pass_fail_threshold",
        },
        expected_split_manifest_sha256="a" * 64,
    )

    rows = report["epoch_metrics"]
    assert [row["ta_enabled"] for row in rows] == [False] * 5 + [True, True]
    assert all(row["ta_parameter_delta"]["l2"] == 0.0 for row in rows[:5])
    assert all(row["ta_optimizer_state_count_after"] == 0 for row in rows[:5])
    assert rows[5]["ta_gradient_observation"]["nonzero_gradient_count"] == 2
    assert rows[-1]["ta_parameter_delta"]["l2"] > 0.0
    assert all(row["absolute_lr_schedule_match"] for row in rows)
    assert report["descriptive_validation_loss_change_fraction"] < 0.0
    assert report["integrity_anomalies"] == []


def test_runtime_sources_must_be_tracked_and_current(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(arguments, **_kwargs):
        calls.append(list(arguments))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = tool.require_head_bound_sources(
        [ROOT / "scripts" / "calibrate_v5_health.py", PLAN_PATH]
    )

    assert result == ["scripts/calibrate_v5_health.py", "configs/v5_health_calibration.yaml"]
    assert calls[0][1:4] == ["ls-files", "--error-unmatch", "--"]
    assert calls[1][1:5] == ["diff", "--quiet", "HEAD", "--"]


def test_v5_design_probe_entrypoint_prefers_reviewed_source(tmp_path: Path) -> None:
    stale_root = tmp_path / "stale"
    stale_package = stale_root / "talif_msresnet"
    stale_package.mkdir(parents=True)
    (stale_package / "__init__.py").write_text("", encoding="utf-8")
    environment = {**dict(os.environ), "PYTHONPATH": str(stale_root)}

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "calibrate_v5_health.py"), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_v4_termination_record_binds_received_evidence() -> None:
    text = (ROOT / "V4_TALIF_ONLY_TERMINATION.md").read_text(encoding="utf-8")

    assert "TERMINATED AFTER CONSUMED NON-REPORTING CIFAR-100 HEALTH FAILURE" in text
    assert "8115975674e4e1952f0573f691de887dee142018067c7388bd466220125ab6fb" in text
    assert "547ea7b24da9976359605eb67467f89a100b482f3c2728fb2d760564d7f2a4b1" in text
    assert "Never rerun CIFAR-100 v4 health seed `1975342236`" in text


def test_surrogate_support_aggregation_uses_layer_counts_across_batches() -> None:
    train_module = importlib.import_module("talif_msresnet.train")
    accumulator: dict[str, dict[str, float]] = {}

    train_module._accumulate_surrogate_layer_counts(
        accumulator,
        {
            "layers": {
                "early": {"surrogate_active": 1, "surrogate_elements": 2},
                "deep": {"surrogate_active": 1, "surrogate_elements": 4},
            }
        },
    )
    train_module._accumulate_surrogate_layer_counts(
        accumulator,
        {
            "layers": {
                "early": {"surrogate_active": 1, "surrogate_elements": 8},
                "deep": {"surrogate_active": 3, "surrogate_elements": 4},
            }
        },
    )
    summary = train_module._surrogate_layer_summary(accumulator)

    assert summary["early"]["surrogate_coverage"] == pytest.approx(0.2)
    assert summary["deep"]["surrogate_coverage"] == pytest.approx(0.5)
    assert min(row["surrogate_coverage"] for row in summary.values()) == pytest.approx(0.2)


def test_surrogate_support_aggregation_rejects_impossible_counts() -> None:
    train_module = importlib.import_module("talif_msresnet.train")

    with pytest.raises(RuntimeError, match="Invalid surrogate support counts"):
        train_module._accumulate_surrogate_layer_counts(
            {},
            {"layers": {"broken": {"surrogate_active": 2, "surrogate_elements": 1}}},
        )
