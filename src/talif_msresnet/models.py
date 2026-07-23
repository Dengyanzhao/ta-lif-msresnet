"""CIFAR Spiking ResNet and membrane-shortcut ResNet factorial models.

The public entry point is :func:`build_model`.  Its canonical configuration
keys are ``condition``, ``topology``, ``neuron``, ``depth``, ``time_steps``,
``num_classes``, ``in_channels``, ``base_channels``, and ``neuron_cfg``.
Defaults are C1, Spiking ResNet, LIF, ResNet-20, T=6, 10 classes, RGB input,
16 base channels, and the neuron defaults documented in ``neurons.py``.

This module is a reference implementation for the planned controlled
experiment, not a claim that every historical implementation detail of the
dissertation source code was recovered.  In particular, it freezes the
composition stated in the manuscript: MS-ResNet keeps a membrane-domain
shortcut, applies neurons inside residual branches, and never inserts a spike
gate into an identity/projection shortcut.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .neurons import BaseNeuron, LIFNeuron, TALIFNeuron

__all__ = [
    "CONDITIONS",
    "SpikingBasicBlock",
    "MSBasicBlock",
    "CIFARSpikingResNet",
    "SpikingResNet",
    "MSResNet",
    "build_model",
]


CONDITIONS: Dict[str, Tuple[str, str]] = {
    "C1": ("spiking_resnet", "lif"),
    "C2": ("spiking_resnet", "ta_lif"),
    "C3": ("ms_resnet", "lif"),
    "C4": ("ms_resnet", "ta_lif"),
}


def _normalise_topology(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "spiking_resnet": "spiking_resnet",
        "spikingresnet": "spiking_resnet",
        "s_resnet": "spiking_resnet",
        "sresnet": "spiking_resnet",
        "resnet": "spiking_resnet",
        "conventional": "spiking_resnet",
        "ms_resnet": "ms_resnet",
        "msresnet": "ms_resnet",
        "membrane_shortcut": "ms_resnet",
    }
    if key not in aliases:
        raise ValueError(f"unsupported topology: {value!r}")
    return aliases[key]


def _normalise_neuron(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "lif": "lif",
        "ta_lif": "ta_lif",
        "talif": "ta_lif",
        "adaptive": "ta_lif",
    }
    if key not in aliases:
        raise ValueError(f"unsupported neuron: {value!r}")
    return aliases[key]


def _depth_value(value: Any) -> int:
    if isinstance(value, str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            raise ValueError(f"cannot parse depth from {value!r}")
        value = int(digits)
    depth = int(value)
    if depth not in (20, 56):
        raise ValueError("this experiment supports CIFAR ResNet depth 20 or 56")
    return depth


def _make_neuron_factory(
    kind: str, time_steps: int, config: Mapping[str, Any]
) -> Callable[[], BaseNeuron]:
    cfg = dict(config)
    tau = float(cfg.get("tau", cfg.get("decay", 0.5)))
    threshold = float(cfg.get("threshold", cfg.get("v_threshold", cfg.get("v_th", 1.0))))
    v_rest = float(cfg.get("v_rest", 0.0))
    v_reset = float(cfg.get("v_reset", 0.0))
    if kind == "lif":
        surrogate_width = float(cfg.get("surrogate_width", cfg.get("width", 1.0)))

        def make_lif() -> BaseNeuron:
            return LIFNeuron(
                tau=tau,
                threshold=threshold,
                surrogate_width=surrogate_width,
                v_rest=v_rest,
                v_reset=v_reset,
            )

        return make_lif

    width = float(cfg.get("width", cfg.get("surrogate_width", 1.0)))
    delta_min = float(cfg.get("delta_min", 1e-3))

    def make_talif() -> BaseNeuron:
        return TALIFNeuron(
            steps=time_steps,
            tau=tau,
            threshold=threshold,
            width=width,
            delta_min=delta_min,
            v_rest=v_rest,
            v_reset=v_reset,
        )

    return make_talif


def _projection(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    if stride == 1 and in_channels == out_channels:
        return nn.Identity()
    # Projection/downsampling is kept linear and has no neuron, as required by
    # the membrane-shortcut intervention definition.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class SpikingBasicBlock(nn.Module):
    """Conventional spiking residual block with post-addition spike gating."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        neuron_factory: Callable[[], BaseNeuron],
    ) -> None:
        super().__init__()
        self.neuron1 = neuron_factory()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.neuron2 = neuron_factory()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = _projection(in_channels, out_channels, stride)
        self.neuron3 = neuron_factory()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = self.shortcut(x)
        branch = self.neuron1(x)
        branch = self.bn1(self.conv1(branch))
        branch = self.neuron2(branch)
        branch = self.bn2(self.conv2(branch))
        return self.neuron3(identity + branch)


class MSBasicBlock(nn.Module):
    """Membrane-shortcut block with no spike gate on the shortcut/addition."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        neuron_factory: Callable[[], BaseNeuron],
    ) -> None:
        super().__init__()
        # Keep these names identical to the conventional block so convolution
        # state_dict keys pair exactly across C1--C4.
        self.neuron1 = neuron_factory()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.neuron2 = neuron_factory()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = _projection(in_channels, out_channels, stride)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = self.shortcut(x)
        branch = self.neuron1(x)
        branch = self.bn1(self.conv1(branch))
        branch = self.neuron2(branch)
        branch = self.bn2(self.conv2(branch))
        # This tensor remains in the membrane domain for the next block.
        return identity + branch


class CIFARSpikingResNet(nn.Module):
    """CIFAR ResNet-20/56 supporting all four factorial conditions."""

    def __init__(
        self,
        *,
        condition: str = "C1",
        topology: str = "spiking_resnet",
        neuron: str = "lif",
        depth: int = 20,
        time_steps: int = 6,
        num_classes: int = 10,
        in_channels: int = 3,
        base_channels: int = 16,
        neuron_cfg: Optional[Mapping[str, Any]] = None,
        init_seed: int = 0,
        shared_conv_weights: Optional[Mapping[str, Tensor]] = None,
        readout: str = "mean",
    ) -> None:
        super().__init__()
        depth = _depth_value(depth)
        if time_steps < 1:
            raise ValueError("time_steps must be >= 1")
        if num_classes < 1 or in_channels < 1 or base_channels < 1:
            raise ValueError("num_classes, in_channels, and base_channels must be positive")
        if readout not in ("mean", "sum", "last"):
            raise ValueError("readout must be one of: mean, sum, last")
        self.condition = condition.upper()
        self.topology = _normalise_topology(topology)
        self.neuron_type = _normalise_neuron(neuron)
        self.depth = depth
        self.time_steps = int(time_steps)
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.readout = readout
        self.init_seed = int(init_seed)
        self.neuron_cfg = dict(neuron_cfg or {})

        blocks_per_stage = (depth - 2) // 6
        neuron_factory = _make_neuron_factory(self.neuron_type, self.time_steps, self.neuron_cfg)
        block_cls = SpikingBasicBlock if self.topology == "spiking_resnet" else MSBasicBlock

        self.stem_conv = nn.Conv2d(
            self.in_channels, self.base_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(self.base_channels)
        # A common stem site is retained identically across all four cells and
        # is therefore not part of the topology intervention.
        self.stem_neuron = neuron_factory()

        channels = self.base_channels
        self.stage1, channels = self._make_stage(
            block_cls, channels, self.base_channels, blocks_per_stage, 1, neuron_factory
        )
        self.stage2, channels = self._make_stage(
            block_cls, channels, self.base_channels * 2, blocks_per_stage, 2, neuron_factory
        )
        self.stage3, channels = self._make_stage(
            block_cls, channels, self.base_channels * 4, blocks_per_stage, 2, neuron_factory
        )
        # The terminal site is specified by the MS-ResNet composition and is
        # mirrored in the conventional topology for a controlled readout.
        self.terminal_neuron = neuron_factory()
        self.fc = nn.Linear(channels, self.num_classes)

        self._initialize_weights(self.init_seed)
        if shared_conv_weights is not None:
            self.load_convolution_state_dict(shared_conv_weights, strict=True)

    @staticmethod
    def _make_stage(
        block_cls: type[nn.Module],
        in_channels: int,
        out_channels: int,
        count: int,
        first_stride: int,
        neuron_factory: Callable[[], BaseNeuron],
    ) -> Tuple[nn.ModuleList, int]:
        blocks: List[nn.Module] = []
        current = in_channels
        for index in range(count):
            stride = first_stride if index == 0 else 1
            blocks.append(block_cls(current, out_channels, stride, neuron_factory))
            current = out_channels
        return nn.ModuleList(blocks), current

    def _initialize_weights(self, seed: int) -> None:
        """Initialise without consuming global RNG state.

        The module traversal order and all convolution names are identical for
        paired conditions.  Therefore a common ``init_seed`` gives bitwise
        identical convolution/BN/classifier initial states in C1--C4.
        """

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    fan_out = module.out_channels * module.kernel_size[0] * module.kernel_size[1]
                    values = torch.randn(
                        module.weight.shape, generator=generator, dtype=torch.float32
                    )
                    module.weight.copy_((values * math.sqrt(2.0 / fan_out)).to(module.weight))
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.fill_(1.0)
                    module.bias.zero_()
                    module.running_mean.zero_()
                    module.running_var.fill_(1.0)
                elif isinstance(module, nn.Linear):
                    bound = 1.0 / math.sqrt(module.in_features)
                    values = torch.rand(
                        module.weight.shape, generator=generator, dtype=torch.float32
                    )
                    module.weight.copy_(((values * 2.0 - 1.0) * bound).to(module.weight))
                    if module.bias is not None:
                        bias = torch.rand(
                            module.bias.shape, generator=generator, dtype=torch.float32
                        )
                        module.bias.copy_(((bias * 2.0 - 1.0) * bound).to(module.bias))

    def reset_state(self) -> None:
        """Clear all neuron state before a new independent input."""

        for module in self.modules():
            if isinstance(module, BaseNeuron):
                module.reset_state(clear_stats=True)

    # Common alias for generic SNN benchmark code.
    reset = reset_state

    def _forward_step(self, frame: Tensor) -> Tensor:
        out = self.stem_neuron(self.stem_bn(self.stem_conv(frame)))
        for block in self.stage1:
            out = block(out)
        for block in self.stage2:
            out = block(out)
        for block in self.stage3:
            out = block(out)
        out = self.terminal_neuron(out)
        out = F.adaptive_avg_pool2d(out, output_size=1).flatten(1)
        return self.fc(out)

    def _activity_diagnostics(self, timesteps: int) -> Dict[str, object]:
        layers: Dict[str, Dict[str, object]] = {}
        spike_count = 0.0
        elements = 0
        for name, module in self.named_modules():
            if isinstance(module, BaseNeuron):
                item = module.diagnostics()
                if isinstance(module, TALIFNeuron):
                    v1, v2 = module.threshold_windows()
                    item.update(
                        {
                            "window_v1": v1.detach().cpu().tolist(),
                            "window_v2": v2.detach().cpu().tolist(),
                        }
                    )
                layers[name] = item
                spike_count += float(item["spike_count"])
                elements += int(item["elements"])
        return {
            "spike_rate": spike_count / elements if elements else 0.0,
            "spike_count": spike_count,
            "elements": elements,
            "timesteps": int(timesteps),
            "condition": self.condition,
            "topology": self.topology,
            "neuron": self.neuron_type,
            "layers": layers,
        }

    def forward(self, x: Tensor, collect_activity: bool = False):  # type: ignore[override]
        """Classify static images or event frames.

        A static tensor ``[B,C,H,W]`` is presented for ``time_steps`` steps.
        Event input ``[B,T,C,H,W]`` uses its supplied frame count; TA-LIF count
        indices clip safely if this exceeds the configured bank length.
        """

        if x.ndim == 4:
            frames = x.unsqueeze(1).expand(-1, self.time_steps, -1, -1, -1)
        elif x.ndim == 5:
            if x.shape[1] < 1:
                raise ValueError("event input must contain at least one frame")
            frames = x
        else:
            raise ValueError("input must have shape [B,C,H,W] or [B,T,C,H,W]")
        if frames.shape[2] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {frames.shape[2]}"
            )
        if not frames.is_floating_point():
            frames = frames.float()

        for module in self.modules():
            if isinstance(module, BaseNeuron):
                module.set_collect_activity(collect_activity)
        self.reset_state()
        logits_by_step = [self._forward_step(frames[:, step]) for step in range(frames.shape[1])]
        temporal = torch.stack(logits_by_step, dim=1)
        if self.readout == "sum":
            logits = temporal.sum(dim=1)
        elif self.readout == "last":
            logits = temporal[:, -1]
        else:
            logits = temporal.mean(dim=1)
        if collect_activity:
            return logits, self._activity_diagnostics(frames.shape[1])
        return logits

    def convolution_state_dict(self) -> Dict[str, Tensor]:
        """Return a CPU copy suitable for explicit paired initialisation."""

        result: Dict[str, Tensor] = {}
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                key = f"{module_name}.weight"
                result[key] = module.weight.detach().cpu().clone()
                if module.bias is not None:
                    result[f"{module_name}.bias"] = module.bias.detach().cpu().clone()
        return result

    def load_convolution_state_dict(
        self, state: Mapping[str, Tensor], *, strict: bool = True
    ) -> None:
        """Load only convolution tensors while preserving neuron parameters."""

        expected: Dict[str, Tensor] = {}
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                expected[f"{module_name}.weight"] = module.weight
                if module.bias is not None:
                    expected[f"{module_name}.bias"] = module.bias
        missing = sorted(set(expected) - set(state))
        extra = sorted(set(state) - set(expected))
        if strict and (missing or extra):
            raise KeyError(f"convolution state mismatch; missing={missing}, extra={extra}")
        with torch.no_grad():
            for name, target in expected.items():
                if name not in state:
                    continue
                source = torch.as_tensor(state[name], device=target.device, dtype=target.dtype)
                if source.shape != target.shape:
                    raise ValueError(
                        f"shape mismatch for {name}: expected {tuple(target.shape)}, "
                        f"got {tuple(source.shape)}"
                    )
                target.copy_(source)

    def parameter_report(self) -> Dict[str, object]:
        """Return counts used by experiment manifests and efficiency tables."""

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        convolution = 0
        normalization = 0
        classifier = sum(parameter.numel() for parameter in self.fc.parameters())
        neuron_parameters = 0
        threshold_parameters = 0
        neuron_layers = 0
        convolution_names: List[str] = []
        for module_name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                convolution += sum(
                    parameter.numel() for parameter in module.parameters(recurse=False)
                )
                convolution_names.extend(
                    f"{module_name}.{name}" for name, _ in module.named_parameters(recurse=False)
                )
            elif isinstance(module, nn.BatchNorm2d):
                normalization += sum(
                    parameter.numel() for parameter in module.parameters(recurse=False)
                )
            elif isinstance(module, BaseNeuron):
                neuron_layers += 1
                count = sum(parameter.numel() for parameter in module.parameters(recurse=False))
                neuron_parameters += count
                if isinstance(module, TALIFNeuron):
                    threshold_parameters += count
        return {
            "model": f"{self.topology}_{self.depth}",
            "condition": self.condition,
            "topology": self.topology,
            "neuron": self.neuron_type,
            "depth": self.depth,
            "time_steps": self.time_steps,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "conv_parameters": convolution,
            "convolution_parameters": convolution,
            "normalization_parameters": normalization,
            "classifier_parameters": classifier,
            "neuron_parameters": neuron_parameters,
            "threshold_parameters": threshold_parameters,
            "neuron_layers": neuron_layers,
            "convolution_parameter_names": convolution_names,
        }


class SpikingResNet(CIFARSpikingResNet):
    """Convenience constructor fixed to the conventional spiking topology."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["topology"] = "spiking_resnet"
        neuron = _normalise_neuron(kwargs.get("neuron", "lif"))
        kwargs.setdefault("condition", "C2" if neuron == "ta_lif" else "C1")
        super().__init__(*args, **kwargs)


class MSResNet(CIFARSpikingResNet):
    """Convenience constructor fixed to the membrane-shortcut topology."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["topology"] = "ms_resnet"
        neuron = _normalise_neuron(kwargs.get("neuron", "lif"))
        kwargs.setdefault("condition", "C4" if neuron == "ta_lif" else "C3")
        super().__init__(*args, **kwargs)


def build_model(model_cfg: Mapping[str, Any]) -> CIFARSpikingResNet:
    """Build one of the C1--C4 controlled factorial models.

    Canonical keys and defaults:

    - ``condition='C1'`` (maps to topology/neuron below)
    - ``topology='spiking_resnet'`` or ``'ms_resnet'``
    - ``neuron='lif'`` or ``'ta_lif'``
    - ``depth=20`` (also 56), ``time_steps=6``
    - ``num_classes=10``, ``in_channels=3``, ``base_channels=16``
    - ``neuron_cfg={}`` with tau/threshold/width/delta_min options

    If ``condition`` is supplied, any explicit topology/neuron must agree with
    it.  This fails fast rather than silently contaminating a factorial cell.
    ``init_seed`` defaults to zero and is deliberately independent of the
    condition, making convolution initialisation pairable across C1--C4.
    """

    if not isinstance(model_cfg, Mapping):
        raise TypeError("model_cfg must be a Mapping")
    cfg = dict(model_cfg)
    # Accept a conventional nested model section while keeping the documented
    # flat schema authoritative for command-line overrides.
    nested = cfg.get("model")
    if isinstance(nested, Mapping):
        combined = dict(nested)
        combined.update({key: value for key, value in cfg.items() if key != "model"})
        cfg = combined

    condition_value = cfg.get("condition")
    if condition_value is None:
        topology = _normalise_topology(cfg.get("topology", "spiking_resnet"))
        neuron = _normalise_neuron(cfg.get("neuron", "lif"))
        reverse = {value: key for key, value in CONDITIONS.items()}
        condition = reverse[(topology, neuron)]
    else:
        condition = str(condition_value).strip().upper()
        if condition not in CONDITIONS:
            raise ValueError(f"condition must be one of {sorted(CONDITIONS)}, got {condition!r}")
        expected_topology, expected_neuron = CONDITIONS[condition]
        topology = _normalise_topology(cfg.get("topology", expected_topology))
        neuron = _normalise_neuron(cfg.get("neuron", expected_neuron))
        if topology != expected_topology or neuron != expected_neuron:
            raise ValueError(
                f"{condition} requires topology={expected_topology!r}, neuron={expected_neuron!r}; "
                f"got topology={topology!r}, neuron={neuron!r}"
            )

    dataset = str(cfg.get("dataset", "")).lower().replace("_", "-")
    default_classes = 100 if dataset == "cifar-100" or dataset == "cifar100" else 10
    return CIFARSpikingResNet(
        condition=condition,
        topology=topology,
        neuron=neuron,
        depth=_depth_value(cfg.get("depth", 20)),
        time_steps=int(cfg.get("time_steps", cfg.get("steps", cfg.get("T", 6)))),
        num_classes=int(cfg.get("num_classes", default_classes)),
        in_channels=int(cfg.get("in_channels", 3)),
        base_channels=int(cfg.get("base_channels", 16)),
        neuron_cfg=cfg.get("neuron_cfg", {}),
        init_seed=int(cfg.get("init_seed", cfg.get("shared_init_seed", 0))),
        shared_conv_weights=cfg.get("shared_conv_weights"),
        readout=str(cfg.get("readout", "mean")).lower(),
    )
