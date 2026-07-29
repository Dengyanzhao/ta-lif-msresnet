from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from torch import nn

import talif_msresnet.train as train_module
from talif_msresnet.config import (
    ConfigError,
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    RuntimeConfig,
    generate_run_matrix,
    generate_v4_pilot_matrix,
    load_protocol,
    validate_protocol,
    validate_run_mapping,
)
from talif_msresnet.models import build_model
from talif_msresnet.neurons import BaseNeuron, LIFNeuron, TALIFNeuron
from talif_msresnet.train import (
    _checkpoint_payload,
    _load_resume,
    build_optimizer_and_scheduler,
    optimizer_group_manifest,
    train_one_epoch,
)
from talif_msresnet.utils import save_checkpoint


ROOT = Path(__file__).resolve().parents[1]


def _model_config(condition: str, **overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "condition": condition,
        "depth": 20,
        "time_steps": 1,
        "num_classes": 10,
        "in_channels": 3,
        "base_channels": 1,
        "neuron_cfg": {
            "tau": 0.5,
            "threshold": 1.0,
            "width": 1.0,
            "delta_min": 0.01,
        },
        "init_seed": 19,
    }
    value.update(overrides)
    return value


def _run_config(
    *,
    protocol_version: int = 4,
    condition: str = "C1",
    output_dir: str = "results",
    resume: str | None = None,
    warmup_epochs: int = 5,
    exclude_weight_decay: bool = True,
) -> RunConfig:
    neuron = "ta_lif" if condition == "C2" else "lif"
    return RunConfig(
        protocol_version=protocol_version,
        experiment="E1",
        data=DataConfig(num_classes=10, in_channels=3),
        model=ModelConfig(
            condition=condition,
            topology="spiking_resnet",
            neuron=neuron,
            depth=20,
            time_steps=1,
            num_classes=10,
            in_channels=3,
            base_channels=1,
            terminal_neuron_mode=(
                "topology_required" if protocol_version >= 4 else "always"
            ),
        ),
        optimizer=OptimizerConfig(
            epochs=10,
            batch_size=2,
            lr=0.1,
            milestones=(5, 8),
            gamma=0.1,
            warmup_epochs=warmup_epochs,
            exclude_norm_and_bias_from_weight_decay=exclude_weight_decay,
        ),
        runtime=RuntimeConfig(
            seed=17,
            device="cpu",
            output_dir=output_dir,
            run_id="repair-test",
            resume=resume,
        ),
    )


def test_terminal_neuron_mode_preserves_legacy_counts_and_repairs_only_c1_c2() -> None:
    for condition in ("C1", "C2", "C3", "C4"):
        legacy = build_model(_model_config(condition))
        assert legacy.parameter_report()["neuron_layers"] == 20

    for condition in ("C1", "C2"):
        repaired = build_model(
            _model_config(condition, terminal_neuron_mode="topology_required")
        )
        assert repaired.parameter_report()["neuron_layers"] == 19
        assert isinstance(repaired.terminal_neuron, nn.Identity)

    for condition in ("C3", "C4"):
        repaired = build_model(
            _model_config(condition, terminal_neuron_mode="topology_required")
        )
        assert repaired.parameter_report()["neuron_layers"] == 20
        assert isinstance(repaired.terminal_neuron, BaseNeuron)


@pytest.mark.parametrize(
    "neuron",
    [
        LIFNeuron(threshold=1.0, surrogate_width=1.0),
        TALIFNeuron(steps=1, threshold=1.0, width=1.0, delta_min=0.01),
    ],
)
def test_surrogate_coverage_uses_closed_boundaries_and_resets(
    neuron: BaseNeuron,
) -> None:
    neuron.set_collect_activity(True)
    neuron(torch.tensor([0.49, 0.50, 1.00, 1.50, 1.51]))
    diagnostics = neuron.diagnostics()

    assert diagnostics["surrogate_active"] == 3.0
    assert diagnostics["surrogate_elements"] == 5
    assert diagnostics["surrogate_coverage"] == pytest.approx(3.0 / 5.0)

    neuron.reset_state()
    reset = neuron.diagnostics()
    assert reset["surrogate_active"] == 0.0
    assert reset["surrogate_elements"] == 0
    assert reset["surrogate_coverage"] == 0.0


@pytest.mark.parametrize(
    "neuron",
    [
        LIFNeuron(threshold=1.0, surrogate_width=1.0),
        TALIFNeuron(steps=1, threshold=1.0, width=1.0, delta_min=0.01),
    ],
)
def test_activity_disabled_does_not_pass_a_surrogate_mask(
    neuron: BaseNeuron, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[torch.Tensor | None] = []
    original = neuron._record

    def record(
        spike: torch.Tensor,
        membrane: torch.Tensor,
        surrogate_active: torch.Tensor | None = None,
    ) -> None:
        received.append(surrogate_active)
        original(spike, membrane, surrogate_active)

    monkeypatch.setattr(neuron, "_record", record)
    neuron.set_collect_activity(False)
    neuron(torch.tensor([1.0]))
    assert received == [None]


def test_deepest_zero_coverage_is_not_hidden_by_earlier_layers() -> None:
    model = build_model(
        _model_config("C1", terminal_neuron_mode="topology_required")
    )
    neurons = [module for module in model.modules() if isinstance(module, BaseNeuron)]
    assert len(neurons) == 19
    for neuron in neurons:
        neuron._surrogate_active_sum = torch.tensor(1.0)
        neuron._surrogate_element_count = 1
    neurons[-1]._surrogate_active_sum = torch.tensor(0.0)

    diagnostics = model._activity_diagnostics(timesteps=1)
    assert diagnostics["surrogate_coverage"] == pytest.approx(18.0 / 19.0)
    assert diagnostics["surrogate_coverage_min"] == 0.0
    assert diagnostics["surrogate_coverage_deepest"] == 0.0
    assert diagnostics["surrogate_coverage_deepest_layer"] == "stage3.2.neuron2"
    assert "surrogate_coverage_terminal" not in diagnostics


def test_repaired_terminal_graph_rejects_a_legacy_c2_checkpoint() -> None:
    legacy = build_model(_model_config("C2"))
    repaired = build_model(
        _model_config("C2", terminal_neuron_mode="topology_required")
    )
    with pytest.raises(RuntimeError, match="Unexpected key"):
        repaired.load_state_dict(legacy.state_dict(), strict=True)


@pytest.mark.parametrize("condition", ["C1", "C2"])
def test_repaired_c1_c2_keep_stem_deep_block_and_classifier_gradients(
    condition: str,
) -> None:
    torch.manual_seed(0)
    model = build_model(
        _model_config(
            condition,
            time_steps=6,
            base_channels=2,
            terminal_neuron_mode="topology_required",
        )
    )
    logits = model(torch.randn(2, 3, 8, 8))
    torch.nn.functional.cross_entropy(logits, torch.tensor([1, 7])).backward()

    parameters = dict(model.named_parameters())
    for name in ("stem_conv.weight", "stage3.2.conv2.weight", "fc.weight"):
        gradient = parameters[name].grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.norm().item() > 0.0


def test_legacy_hash_and_serialized_shape_remain_fixed() -> None:
    config = RunConfig(
        protocol_version=1,
        experiment="E1",
        data=DataConfig(),
        model=ModelConfig(),
        optimizer=OptimizerConfig(),
        runtime=RuntimeConfig(),
    )
    assert config.config_hash == (
        "bbcb4a86a1d93add21fbac95c676c9d0a6f8f2172e3ae07fcae434be6731595d"
    )
    serialized = config.as_dict()
    assert "terminal_neuron_mode" not in serialized["model"]
    assert "warmup_epochs" not in serialized["optimizer"]
    assert "exclude_norm_and_bias_from_weight_decay" not in serialized["optimizer"]


@pytest.mark.parametrize(
    "section, field, value",
    [
        ("model", "terminal_neuron_mode", "topology_required"),
        ("optimizer", "warmup_epochs", 5),
        ("optimizer", "exclude_norm_and_bias_from_weight_decay", True),
    ],
)
def test_v3_protocol_rejects_every_v4_repair_field(
    section: str, field: str, value: Any
) -> None:
    protocol = yaml.safe_load(
        (ROOT / "configs" / "protocol_v3_talif_only.yaml").read_text(
            encoding="utf-8"
        )
    )
    protocol[section][field] = value
    with pytest.raises(ConfigError, match="v4"):
        validate_protocol(protocol)


def test_v4_protocol_is_executable_but_direct_run_still_requires_gate_evidence() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v4_talif_only.yaml")
    formal = generate_run_matrix(protocol)
    pilot = generate_v4_pilot_matrix(protocol)

    assert len(formal) == 20
    assert len(pilot) == 4
    config = validate_run_mapping(formal[0], protocol)
    with pytest.raises(ValueError, match="execution evidence"):
        train_module.run(config)


def test_v4_resolved_configs_round_trip_through_the_written_yaml_shape() -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v4_talif_only.yaml")
    raw_runs = generate_run_matrix(protocol) + generate_v4_pilot_matrix(protocol)

    for raw in raw_runs:
        resolved = validate_run_mapping(raw, protocol)
        serialized = resolved.as_dict()

        assert "run_id" not in serialized
        reloaded = validate_run_mapping(serialized, protocol)
        assert reloaded.as_dict() == serialized
        assert reloaded.runtime.run_id == resolved.runtime.run_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw["model"].update(terminal_neuron_mode="always"), "model contract"),
        (lambda raw: raw["optimizer"].update(lr=0.1), "optimizer contract"),
        (
            lambda raw: (
                raw.update(condition="C3"),
                raw["model"].update(
                    condition="C3", topology="ms_resnet", neuron="lif"
                ),
            ),
            "inactive",
        ),
        (lambda raw: raw.update(seed=1230259817), "not assigned"),
        (
            lambda raw: raw["runtime"].update(output_dir="results/unbound"),
            "output_dir",
        ),
    ],
)
def test_v4_run_yaml_cannot_override_the_frozen_matrix_contract(
    mutation: Any, message: str
) -> None:
    protocol = load_protocol(ROOT / "configs" / "protocol_v4_talif_only.yaml")
    raw = copy.deepcopy(generate_run_matrix(protocol)[0])
    mutation(raw)
    with pytest.raises(ConfigError, match=message):
        validate_run_mapping(raw, protocol)


def test_warmup_cannot_extend_past_the_first_milestone() -> None:
    raw = {
        "protocol_version": 3,
        "optimizer": {
            "epochs": 10,
            "milestones": [4, 8],
            "warmup_epochs": 5,
        },
    }
    with pytest.raises(ConfigError, match="first milestone"):
        validate_run_mapping(raw)


def test_optimizer_groups_are_complete_auditable_and_state_is_metadata_free() -> None:
    config = _run_config(condition="C2")
    model = build_model(_model_config("C2", terminal_neuron_mode="topology_required"))
    optimizer, _, _, _, _ = build_optimizer_and_scheduler(model, config)

    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters()
    }

    manifest = optimizer_group_manifest(model, optimizer)
    assert [group["role"] for group in manifest["groups"]] == [
        "base_decay",
        "base_no_decay",
        "ta",
    ]
    assert sum(group["parameter_count"] for group in manifest["groups"]) == len(
        grouped
    )
    assert sum(group["parameter_numel"] for group in manifest["groups"]) == sum(
        parameter.numel() for parameter in model.parameters()
    )
    assert [group["weight_decay"] for group in manifest["groups"]] == [
        config.optimizer.weight_decay,
        0.0,
        config.optimizer.ta_weight_decay,
    ]
    for group in optimizer.state_dict()["param_groups"]:
        assert "parameter_names" not in group
        assert "role" not in group


def test_top_level_linear_bias_is_also_excluded_from_weight_decay() -> None:
    config = _run_config(condition="C1")
    model = nn.Linear(2, 2)
    optimizer, _, _, _, _ = build_optimizer_and_scheduler(model, config)
    manifest = optimizer_group_manifest(model, optimizer)

    assert [group["parameter_names"] for group in manifest["groups"]] == [
        ["weight"],
        ["bias"],
    ]
    assert [group["weight_decay"] for group in manifest["groups"]] == [
        config.optimizer.weight_decay,
        0.0,
    ]


def _advance_schedule(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epochs: int,
) -> list[float]:
    values: list[float] = []
    for _ in range(epochs):
        values.append(float(optimizer.param_groups[0]["lr"]))
        optimizer.step()
        scheduler.step()
    return values


def test_warmup_sequence_and_resume_across_milestone_match_continuous_run(
    tmp_path: Path,
) -> None:
    config = _run_config(condition="C1", exclude_weight_decay=False)
    continuous_model = nn.Linear(2, 2)
    continuous_optimizer, continuous_scheduler, _, _, _ = (
        build_optimizer_and_scheduler(continuous_model, config)
    )
    continuous = _advance_schedule(
        continuous_optimizer, continuous_scheduler, config.optimizer.epochs
    )
    assert continuous[:6] == pytest.approx([0.02, 0.04, 0.06, 0.08, 0.10, 0.01])

    first_model = nn.Linear(2, 2)
    first_optimizer, first_scheduler, _, _, _ = build_optimizer_and_scheduler(
        first_model, config
    )
    first = _advance_schedule(first_optimizer, first_scheduler, 6)
    checkpoint = _checkpoint_payload(
        first_model,
        first_optimizer,
        first_scheduler,
        epoch=5,
        best_val={"accuracy": 0.0, "loss": 1.0, "epoch": 0},
        config=config,
        train_history=[],
    )
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(checkpoint_path, checkpoint)

    resumed_model = nn.Linear(2, 2)
    resumed_optimizer, resumed_scheduler, _, _, _ = build_optimizer_and_scheduler(
        resumed_model, config
    )
    start_epoch, _, _ = _load_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        config,
    )
    assert start_epoch == 6
    resumed = first + _advance_schedule(
        resumed_optimizer,
        resumed_scheduler,
        config.optimizer.epochs - start_epoch,
    )
    assert resumed == pytest.approx(continuous)


def test_tampered_optimizer_group_manifest_is_rejected(tmp_path: Path) -> None:
    config = _run_config(condition="C1", exclude_weight_decay=False)
    model = nn.Linear(2, 2)
    optimizer, scheduler, _, _, _ = build_optimizer_and_scheduler(model, config)
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        epoch=0,
        best_val={},
        config=config,
        train_history=[],
    )
    tampered = copy.deepcopy(payload)
    tampered["optimizer_group_manifest"]["groups"][0]["parameter_names"][0] = (
        "substituted.weight"
    )
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(checkpoint_path, tampered)

    target = nn.Linear(2, 2)
    target_optimizer, target_scheduler, _, _, _ = build_optimizer_and_scheduler(
        target, config
    )
    with pytest.raises(RuntimeError, match="parameter_names"):
        _load_resume(
            checkpoint_path,
            target,
            target_optimizer,
            target_scheduler,
            config,
        )


def test_tampered_optimizer_state_parameter_order_is_rejected(tmp_path: Path) -> None:
    config = _run_config(condition="C1", exclude_weight_decay=False)
    model = nn.Linear(2, 2)
    optimizer, scheduler, _, _, _ = build_optimizer_and_scheduler(model, config)
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        epoch=0,
        best_val={},
        config=config,
        train_history=[],
    )
    payload["optimizer_state"]["param_groups"][0]["params"].reverse()
    checkpoint_path = tmp_path / "last.pt"
    save_checkpoint(checkpoint_path, payload)

    target = nn.Linear(2, 2)
    target_optimizer, target_scheduler, _, _, _ = build_optimizer_and_scheduler(
        target, config
    )
    with pytest.raises(RuntimeError, match="parameter order"):
        _load_resume(
            checkpoint_path,
            target,
            target_optimizer,
            target_scheduler,
            config,
        )


class SpikingBasicBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.scale * 0.0


class ZeroBlockGradientModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = SpikingBasicBlock()
        self.classifier = nn.Linear(4, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.block(value))


class NullLogger:
    def log(self, event: str, **values: Any) -> None:
        del event, values


def test_all_zero_residual_block_gradients_are_retained_in_epoch_metrics() -> None:
    model = ZeroBlockGradientModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = _run_config(
        protocol_version=1,
        warmup_epochs=0,
        exclude_weight_decay=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = train_one_epoch(
        model,
        [(torch.randn(3, 4), torch.tensor([0, 1, 2]))],
        optimizer,
        torch.device("cpu"),
        epoch=0,
        config=config,
        logger=NullLogger(),
        scaler=scaler,
    )
    assert metrics["nonzero_block_gradient_fraction_mean"] == 0.0
    assert metrics["nonzero_block_gradient_fraction_min"] == 0.0
    assert metrics["all_zero_block_gradient_batches"] == 1
    assert metrics["all_zero_block_gradient_batch_fraction"] == 1.0


def _resume_config(tmp_path: Path, checkpoint_path: Path) -> RunConfig:
    return _run_config(
        protocol_version=1,
        output_dir=str(tmp_path),
        resume=str(checkpoint_path),
        warmup_epochs=0,
        exclude_weight_decay=False,
    )


def _patch_resume_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        train_module,
        "_training_environment_identity",
        lambda *args, **kwargs: ("{}", "same-environment"),
    )
    monkeypatch.setattr(
        train_module,
        "build_loaders",
        lambda *args, **kwargs: {"_manifest": {}},
    )
    monkeypatch.setattr(
        train_module,
        "build_model",
        lambda *args, **kwargs: nn.Linear(2, 2),
    )


def _write_previous_run_artifacts(
    tmp_path: Path,
    run_dir: Path,
    previous_manifest: dict[str, Any],
) -> tuple[Path, ...]:
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(previous_manifest, indent=2) + "\n", encoding="utf-8"
    )
    sentinels = {
        run_dir / "resolved_config.json": b'{"old_resolved_config": true}\n',
        run_dir / "seed_metrics.json": b'{"status": "complete"}\n',
        run_dir / "failure.json": b'{"old_failure": true}\n',
        run_dir / "events.jsonl": b'{"event": "old_terminal_event"}\n',
        run_dir / "best.pt": b"old-best-checkpoint\n",
        run_dir / "failed.pt": b"old-failed-checkpoint\n",
        tmp_path / "seed_metrics.csv": b"old,csv\r\nterminal,row\r\n",
    }
    for path, value in sentinels.items():
        path.write_bytes(value)
    return (
        manifest_path,
        run_dir / "resolved_config.json",
        run_dir / "seed_metrics.json",
        run_dir / "failure.json",
        run_dir / "events.jsonl",
        run_dir / "best.pt",
        run_dir / "last.pt",
        run_dir / "failed.pt",
        tmp_path / "seed_metrics.csv",
    )


def _snapshot(paths: tuple[Path, ...]) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in paths}


def test_resume_missing_run_directory_is_not_created(tmp_path: Path) -> None:
    run_dir = tmp_path / "repair-test"
    config = _resume_config(tmp_path, run_dir / "last.pt")

    with pytest.raises(RuntimeError, match="run directory is missing"):
        train_module.run(config)
    assert not run_dir.exists()


def test_rejected_resume_leaves_all_previous_run_artifacts_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "repair-test"
    run_dir.mkdir()
    checkpoint_path = run_dir / "last.pt"
    config = _resume_config(tmp_path, checkpoint_path)
    torch.save({"config_hash": "wrong-config"}, checkpoint_path)
    previous = {
        "run_id": "repair-test",
        "config_hash": config.config_hash,
        "status": "complete",
        "environment": {"training_environment_sha256": "same-environment"},
        "audit_marker": "must-survive",
    }
    protected_paths = _write_previous_run_artifacts(tmp_path, run_dir, previous)
    original_bytes = _snapshot(protected_paths)
    _patch_resume_dependencies(monkeypatch)

    with pytest.raises(RuntimeError, match="Resume checkpoint config_hash"):
        train_module.run(config)
    assert _snapshot(protected_paths) == original_bytes


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (("run_id", "other-run"), ("config_hash", "other-config")),
)
def test_resume_rejects_previous_manifest_identity_mismatch_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    wrong_value: str,
) -> None:
    run_dir = tmp_path / "repair-test"
    run_dir.mkdir()
    checkpoint_path = run_dir / "last.pt"
    config = _resume_config(tmp_path, checkpoint_path)
    torch.save({"config_hash": config.config_hash}, checkpoint_path)
    previous = {
        "run_id": config.runtime.run_id,
        "config_hash": config.config_hash,
        "status": "complete",
        "environment": {"training_environment_sha256": "same-environment"},
    }
    previous[field] = wrong_value
    protected_paths = _write_previous_run_artifacts(tmp_path, run_dir, previous)
    original_bytes = _snapshot(protected_paths)
    _patch_resume_dependencies(monkeypatch)

    with pytest.raises(RuntimeError, match=field):
        train_module.run(config)
    assert _snapshot(protected_paths) == original_bytes


def test_resume_rejects_manifest_checkpoint_optimizer_identity_mismatch_without_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "repair-test"
    run_dir.mkdir()
    checkpoint_path = run_dir / "last.pt"
    config = _resume_config(tmp_path, checkpoint_path)
    source_model = nn.Linear(2, 2)
    source_optimizer, source_scheduler, _, _, _ = build_optimizer_and_scheduler(
        source_model, config
    )
    payload = _checkpoint_payload(
        source_model,
        source_optimizer,
        source_scheduler,
        epoch=0,
        best_val={"accuracy": 0.0, "loss": 1.0, "epoch": 0},
        config=config,
        train_history=[],
        training_environment_sha256="same-environment",
    )
    save_checkpoint(checkpoint_path, payload)
    previous_optimizer_manifest = copy.deepcopy(
        payload["optimizer_group_manifest"]
    )
    previous_optimizer_manifest["groups"][0]["parameter_names"][0] = (
        "substituted.weight"
    )
    previous = {
        "run_id": config.runtime.run_id,
        "config_hash": config.config_hash,
        "status": "complete",
        "environment": {"training_environment_sha256": "same-environment"},
        "optimizer_group_manifest": previous_optimizer_manifest,
    }
    protected_paths = _write_previous_run_artifacts(tmp_path, run_dir, previous)
    original_bytes = _snapshot(protected_paths)
    _patch_resume_dependencies(monkeypatch)

    with pytest.raises(RuntimeError, match="parameter_names"):
        train_module.run(config)
    assert _snapshot(protected_paths) == original_bytes


@pytest.mark.parametrize(
    ("manifest_in_previous", "manifest_in_checkpoint"),
    ((True, False), (False, True)),
)
def test_resume_rejects_one_sided_optimizer_manifest_presence(
    tmp_path: Path,
    manifest_in_previous: bool,
    manifest_in_checkpoint: bool,
) -> None:
    checkpoint_path = tmp_path / "last.pt"
    config = _resume_config(tmp_path, checkpoint_path)
    source_model = nn.Linear(2, 2)
    source_optimizer, source_scheduler, _, _, _ = build_optimizer_and_scheduler(
        source_model, config
    )
    payload = _checkpoint_payload(
        source_model,
        source_optimizer,
        source_scheduler,
        epoch=0,
        best_val={},
        config=config,
        train_history=[],
    )
    saved_manifest = copy.deepcopy(payload["optimizer_group_manifest"])
    if not manifest_in_checkpoint:
        payload.pop("optimizer_group_manifest")
    save_checkpoint(checkpoint_path, payload)
    previous = (
        {"optimizer_group_manifest": saved_manifest}
        if manifest_in_previous
        else {}
    )
    target = nn.Linear(2, 2)
    target_optimizer, target_scheduler, _, _, _ = build_optimizer_and_scheduler(
        target, config
    )

    with pytest.raises(RuntimeError, match="presence"):
        _load_resume(
            checkpoint_path,
            target,
            target_optimizer,
            target_scheduler,
            config,
            expected_run_manifest=previous,
        )


def test_resume_allows_legacy_optimizer_manifest_double_absence(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "last.pt"
    config = _resume_config(tmp_path, checkpoint_path)
    source_model = nn.Linear(2, 2)
    source_optimizer, source_scheduler, _, _, _ = build_optimizer_and_scheduler(
        source_model, config
    )
    payload = _checkpoint_payload(
        source_model,
        source_optimizer,
        source_scheduler,
        epoch=0,
        best_val={},
        config=config,
        train_history=[],
    )
    payload.pop("optimizer_group_manifest")
    save_checkpoint(checkpoint_path, payload)
    target = nn.Linear(2, 2)
    target_optimizer, target_scheduler, _, _, _ = build_optimizer_and_scheduler(
        target, config
    )

    start_epoch, _, _ = _load_resume(
        checkpoint_path,
        target,
        target_optimizer,
        target_scheduler,
        config,
        expected_run_manifest={},
    )
    assert start_epoch == 1


def test_resume_manifest_checkpoint_identity_ignores_current_lr(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "last.pt"
    config = _resume_config(tmp_path, checkpoint_path)
    source_model = nn.Linear(2, 2)
    source_optimizer, source_scheduler, _, _, _ = build_optimizer_and_scheduler(
        source_model, config
    )
    payload = _checkpoint_payload(
        source_model,
        source_optimizer,
        source_scheduler,
        epoch=0,
        best_val={},
        config=config,
        train_history=[],
    )
    save_checkpoint(checkpoint_path, payload)
    previous = {
        "optimizer_group_manifest": copy.deepcopy(
            payload["optimizer_group_manifest"]
        )
    }
    previous["optimizer_group_manifest"]["groups"][0]["current_lr"] = 999.0
    target = nn.Linear(2, 2)
    target_optimizer, target_scheduler, _, _, _ = build_optimizer_and_scheduler(
        target, config
    )

    start_epoch, _, _ = _load_resume(
        checkpoint_path,
        target,
        target_optimizer,
        target_scheduler,
        config,
        expected_run_manifest=previous,
    )
    assert start_epoch == 1
