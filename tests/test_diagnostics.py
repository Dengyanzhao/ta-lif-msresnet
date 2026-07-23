from __future__ import annotations

import json

import pandas as pd
import pytest
torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402
import talif_msresnet.diagnostics as diagnostics_module  # noqa: E402

from talif_msresnet.diagnostics import (  # noqa: E402
    DiagnosticProtocol,
    DiagnosticResult,
    diagnose_model,
    representative_batch_sha256,
    write_diagnostic_outputs,
)


class TinyResidualBlock(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FakeNeuronState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_membrane = None
        self.count = None


class TinyDiagnosticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = TinyResidualBlock(1.5)
        self.block2 = TinyResidualBlock(2.0)
        self.fake_neuron = FakeNeuronState()
        self.classifier = nn.Linear(4, 3, bias=False)

    def forward(self, x: torch.Tensor, collect_activity: bool = False):
        x = self.block1(x)
        x = self.block2(x)
        self.fake_neuron.last_membrane = x.detach()
        self.fake_neuron.count = torch.zeros_like(x, dtype=torch.long)
        logits = self.classifier(x)
        if collect_activity:
            return logits, {
                "spike_rate": 0.25,
                "spike_count": 8.0,
                "elements": 32,
                "layers": {
                    "fake_neuron": {
                        "spike_rate": 0.25,
                        "spike_count": 8.0,
                        "elements": 32,
                        "window_v1": [0.5, 0.6],
                        "window_v2": [1.0, 1.1],
                    }
                },
            }
        return logits


def test_cpu_tiny_local_jacobian_moments_and_gradient_long_table() -> None:
    torch.manual_seed(1)
    model = TinyDiagnosticModel()
    inputs = torch.randn(5, 4)
    targets = torch.tensor([0, 1, 2, 0, 1])
    result = diagnose_model(
        model,
        inputs,
        targets,
        protocol=DiagnosticProtocol(probes=4, probe_seed=17, time_index=-1),
        metadata={"diagnostic_id": "tiny"},
    )
    assert result.block_gradients["block"].tolist() == ["block1", "block2"]
    assert result.block_gradients["gradient_rms"].notna().all()
    assert result.summary["gradient_cv_across_blocks"] >= 0

    moments = result.block_jacobians.set_index("block")
    assert moments.loc["block1", "phi_jjt"] == pytest.approx(1.5**2, rel=1e-6)
    assert moments.loc["block2", "phi_jjt"] == pytest.approx(2.0**2, rel=1e-6)
    assert moments.loc["block1", "phi_jjt_squared"] == pytest.approx(1.5**4, rel=1e-6)
    assert moments.loc["block2", "phi_jjt_squared"] == pytest.approx(2.0**4, rel=1e-6)
    assert moments["varphi_jjt"].abs().max() < 1e-5
    assert (moments["probes"] == 4).all()
    assert moments["method"].str.contains("Rademacher").all()


def test_activity_expands_windows_membrane_and_count_occupancy() -> None:
    result = diagnose_model(
        TinyDiagnosticModel(),
        torch.randn(3, 4),
        torch.tensor([0, 1, 2]),
        protocol=DiagnosticProtocol(probes=1),
        metadata={"diagnostic_id": "activity"},
    )
    layer = result.layer_activity.set_index("layer").loc["fake_neuron"]
    assert layer["window_status"] == "reported"
    assert layer["membrane_status"] == "measured_final_time_state"
    assert layer["count_occupancy_status"] == "measured_final_time_state"
    assert json.loads(layer["count_occupancy_json"]) == pytest.approx([1.0])


def test_representative_batch_hash_covers_targets() -> None:
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    targets = torch.tensor([0, 1])
    first = representative_batch_sha256(inputs, targets)
    assert first == representative_batch_sha256(inputs.clone(), targets.clone())
    assert first != representative_batch_sha256(inputs, torch.tensor([1, 0]))


def test_output_writer_is_idempotent_and_protects_other_run(tmp_path) -> None:
    result = DiagnosticResult(
        block_gradients=pd.DataFrame([{"block": "b", "gradient_rms": 1.0}]),
        block_jacobians=pd.DataFrame([{"block": "b", "phi_jjt": 1.0}]),
        layer_activity=pd.DataFrame([{"layer": "n", "spike_rate": 0.1}]),
        summary={"diagnostic_id": "same"},
    )
    paths = write_diagnostic_outputs(result, tmp_path)
    assert all(path.exists() for path in paths.values())
    assert write_diagnostic_outputs(result, tmp_path) == paths

    other = DiagnosticResult(
        result.block_gradients,
        result.block_jacobians,
        result.layer_activity,
        {"diagnostic_id": "other"},
    )
    with pytest.raises(FileExistsError, match="different diagnostic run"):
        write_diagnostic_outputs(other, tmp_path)


def test_one_block_double_backward_failure_aborts_the_whole_diagnostic(monkeypatch) -> None:
    original_grad = diagnostics_module.torch.autograd.grad
    calls = 0

    def fail_on_second_block(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise RuntimeError("synthetic double-backward failure")
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(diagnostics_module.torch.autograd, "grad", fail_on_second_block)
    with pytest.raises(RuntimeError, match="partial-block summaries are forbidden"):
        diagnose_model(
            TinyDiagnosticModel(),
            torch.randn(3, 4),
            torch.tensor([0, 1, 2]),
            protocol=DiagnosticProtocol(probes=1),
            metadata={"diagnostic_id": "must-fail"},
        )
