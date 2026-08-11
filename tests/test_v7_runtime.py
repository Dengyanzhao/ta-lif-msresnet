from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from talif_msresnet.config import (
    ConfigError,
    generate_run_matrix,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.config_v7 import (
    V7_ACTIVE_CONDITIONS,
    canonicalize_v7_artifact_run_mapping,
)
from talif_msresnet.models import build_model
from talif_msresnet.neurons import (
    LIFNeuron,
    PLIFStyleNeuron,
    RouteMatchedLIFNeuron,
    SharedWindowTALIFNeuron,
    TALIFNeuron,
    TimeIndexedTALIFNeuron,
)
from talif_msresnet.train import (
    _shared_weight_sha256,
    build_optimizer_and_scheduler,
    optimizer_group_manifest,
)

EXPECTED_NEURON_CLASSES = {
    "M0": LIFNeuron,
    "M1": RouteMatchedLIFNeuron,
    "M2": SharedWindowTALIFNeuron,
    "M3": TimeIndexedTALIFNeuron,
    "M4": TALIFNeuron,
    "PLIF": PLIFStyleNeuron,
}


@pytest.fixture(scope="module")
def v7_runs():
    protocol = load_protocol("configs/protocol_v7_mechanism.yaml")
    raw_runs = generate_run_matrix(protocol)
    first_seed = raw_runs[0]["seed"]
    return {
        raw["condition"]: validate_run_mapping(raw, protocol)
        for raw in raw_runs
        if raw["seed"] == first_seed
    }


def _model_for(config):
    values = dataclasses.asdict(config.model)
    values["init_seed"] = config.runtime.seed
    return build_model(values)


def test_v7_condition_order_and_neuron_implementation_are_exact(v7_runs) -> None:
    assert tuple(v7_runs) == V7_ACTIVE_CONDITIONS
    for condition, config in v7_runs.items():
        model = _model_for(config)
        assert model.condition == condition
        assert type(model.stem_neuron) is EXPECTED_NEURON_CLASSES[condition]


def test_v7_all_conditions_share_conv_bn_classifier_initialization(v7_runs) -> None:
    hashes = {
        condition: _shared_weight_sha256(_model_for(config))
        for condition, config in v7_runs.items()
    }
    assert len(set(hashes.values())) == 1, hashes


def test_v7_optimizer_groups_only_real_adaptive_parameters(v7_runs) -> None:
    expected_fragments = {
        "M2": {".center", ".raw_width"},
        "M3": {".center", ".raw_width"},
        "M4": {".center", ".raw_width"},
        "PLIF": {".raw_decay"},
    }
    for condition, config in v7_runs.items():
        model = _model_for(config)
        optimizer, _scheduler, adaptive, names, activation_epoch = (
            build_optimizer_and_scheduler(model, config)
        )
        manifest = optimizer_group_manifest(model, optimizer)
        adaptive_groups = [
            group for group in manifest["groups"] if group["role"] == "adaptive"
        ]
        assert activation_epoch == 5
        if condition in expected_fragments:
            assert adaptive
            assert len(adaptive_groups) == 1
            assert adaptive_groups[0]["initial_lr"] == pytest.approx(0.0025)
            assert adaptive_groups[0]["weight_decay"] == 0.0
            assert all(not parameter.requires_grad for parameter in adaptive)
            assert {
                fragment
                for fragment in expected_fragments[condition]
                if any(fragment in name for name in names)
            } == expected_fragments[condition]
        else:
            assert adaptive == []
            assert names == []
            assert adaptive_groups == []


def test_time_indexed_bank_uses_min_t_tminus1_without_modulo() -> None:
    neuron = TimeIndexedTALIFNeuron(steps=3, width=1.0, delta_min=0.05)
    x = torch.zeros(2, 4)
    observed = []
    for _ in range(6):
        neuron(x)
        observed.append(neuron.last_time_index)
    assert observed == [0, 1, 2, 2, 2, 2]
    neuron.reset_state()
    neuron(x)
    assert neuron.last_time_index == 0


def test_plif_style_decay_is_trainable_and_bounded() -> None:
    neuron = PLIFStyleNeuron(tau=0.5)
    before = neuron.decay.detach().clone()
    optimizer = torch.optim.SGD([neuron.raw_decay], lr=0.1)
    x = torch.full((4,), 0.2, requires_grad=True)
    loss = neuron(x).sum() + neuron.membrane.sum()
    loss.backward()
    assert neuron.raw_decay.grad is not None
    assert torch.isfinite(neuron.raw_decay.grad).all()
    optimizer.step()
    after = neuron.decay.detach()
    assert 0.0 < float(after) < 1.0
    assert not torch.equal(before, after)


def test_route_only_ablation_preserves_hard_forward_but_changes_gradient() -> None:
    lif = LIFNeuron(tau=0.5)
    route = RouteMatchedLIFNeuron(tau=0.5)
    x_lif = torch.tensor([0.75], requires_grad=True)
    x_route = x_lif.detach().clone().requires_grad_(True)
    spike_lif = lif(x_lif)
    spike_route = route(x_route)
    assert torch.equal(spike_lif, spike_route)
    assert torch.equal(lif.membrane, route.membrane)
    lif.membrane.sum().backward()
    route.membrane.sum().backward()
    assert x_lif.grad is not None and x_route.grad is not None
    assert not torch.equal(x_lif.grad, x_route.grad)


def test_v7_artifact_output_path_accepts_only_bound_absolute_equivalent(tmp_path: Path) -> None:
    protocol = load_protocol("configs/protocol_v7_mechanism.yaml")
    raw = generate_run_matrix(protocol)[0]
    expected = tmp_path / protocol["output_root"]
    artifact = dict(raw)
    artifact["runtime"] = {
        **raw["runtime"],
        "output_dir": str(expected.resolve()),
    }
    normalized = canonicalize_v7_artifact_run_mapping(
        artifact,
        protocol,
        project_root=tmp_path,
    )
    assert normalized["runtime"]["output_dir"] == protocol["output_root"]
    assert validate_run_mapping(normalized, protocol).runtime.output_dir == protocol["output_root"]


def test_v7_artifact_output_path_rejects_a_different_absolute_directory(tmp_path: Path) -> None:
    protocol = load_protocol("configs/protocol_v7_mechanism.yaml")
    raw = generate_run_matrix(protocol)[0]
    artifact = dict(raw)
    artifact["runtime"] = {
        **raw["runtime"],
        "output_dir": str((tmp_path / "results" / "wrong-v7-root").resolve()),
    }
    with pytest.raises(ConfigError, match="not path-equivalent"):
        canonicalize_v7_artifact_run_mapping(
            artifact,
            protocol,
            project_root=tmp_path,
        )
