"""Small, download-free CPU tests for the four factorial model cells."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from talif_msresnet.models import MSBasicBlock, SpikingBasicBlock, build_model  # noqa: E402


def _config(condition: str, **overrides):
    config = {
        "condition": condition,
        "depth": 20,
        "time_steps": 1,
        "num_classes": 10,
        "in_channels": 3,
        "base_channels": 2,
        "neuron_cfg": {
            "tau": 0.5,
            "threshold": 1.0,
            "width": 1.0,
            "delta_min": 0.01,
        },
        "init_seed": 123,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize(
    ("condition", "topology", "neuron"),
    [
        ("C1", "spiking_resnet", "lif"),
        ("C2", "spiking_resnet", "ta_lif"),
        ("C3", "ms_resnet", "lif"),
        ("C4", "ms_resnet", "ta_lif"),
    ],
)
def test_builds_all_factorial_cells_and_static_forward(
    condition: str, topology: str, neuron: str
) -> None:
    model = build_model(_config(condition))
    model.eval()
    output = model(torch.randn(1, 3, 8, 8))
    assert output.shape == (1, 10)
    assert model.topology == topology
    assert model.neuron_type == neuron


def test_event_frames_and_activity_contract() -> None:
    model = build_model(_config("C4", time_steps=3))
    model.eval()
    logits, diagnostics = model(torch.randn(2, 2, 3, 8, 8), collect_activity=True)
    assert logits.shape == (2, 10)
    assert diagnostics["timesteps"] == 2  # event T, not configured static T
    assert diagnostics["condition"] == "C4"
    assert 0.0 <= diagnostics["spike_rate"] <= 1.0
    assert diagnostics["elements"] > 0
    assert diagnostics["layers"]


def test_resnet20_and_resnet56_have_expected_stage_lengths() -> None:
    model20 = build_model(_config("C1", depth=20))
    model56 = build_model(_config("C1", depth=56))
    assert [len(model20.stage1), len(model20.stage2), len(model20.stage3)] == [3, 3, 3]
    assert [len(model56.stage1), len(model56.stage2), len(model56.stage3)] == [9, 9, 9]


def test_topologies_differ_only_by_post_addition_gate_at_block_level() -> None:
    conventional = build_model(_config("C1"))
    membrane_shortcut = build_model(_config("C3"))
    assert isinstance(conventional.stage1[0], SpikingBasicBlock)
    assert isinstance(membrane_shortcut.stage1[0], MSBasicBlock)
    assert hasattr(conventional.stage1[0], "neuron3")
    assert not hasattr(membrane_shortcut.stage1[0], "neuron3")


def test_convolution_initialization_is_pairable_across_all_conditions() -> None:
    reference = build_model(_config("C1")).convolution_state_dict()
    for condition in ("C2", "C3", "C4"):
        candidate = build_model(_config(condition)).convolution_state_dict()
        assert candidate.keys() == reference.keys()
        for name in reference:
            assert torch.equal(candidate[name], reference[name]), name


def test_explicit_convolution_state_transfer() -> None:
    source = build_model(_config("C1", init_seed=1))
    target = build_model(_config("C4", init_seed=2))
    state = source.convolution_state_dict()
    target.load_convolution_state_dict(state)
    for name, tensor in target.convolution_state_dict().items():
        assert torch.equal(tensor, state[name])


def test_parameter_report_separates_talif_threshold_overhead() -> None:
    lif = build_model(_config("C1", time_steps=2)).parameter_report()
    talif = build_model(_config("C2", time_steps=2)).parameter_report()
    assert lif["threshold_parameters"] == 0
    assert talif["threshold_parameters"] == talif["neuron_layers"] * 2 * 2
    assert talif["total_parameters"] > lif["total_parameters"]
    assert talif["conv_parameters"] == lif["conv_parameters"]
    assert talif["convolution_parameter_names"] == lif["convolution_parameter_names"]


def test_condition_mismatch_and_bad_input_fail_fast() -> None:
    with pytest.raises(ValueError, match="C1 requires"):
        build_model(_config("C1", neuron="ta_lif"))

    model = build_model(_config("C1"))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(3, 8, 8))
    with pytest.raises(ValueError, match="channels"):
        model(torch.randn(1, 1, 8, 8))
