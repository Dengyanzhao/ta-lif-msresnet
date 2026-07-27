from __future__ import annotations

import torch

from talif_msresnet.train import _set_ta_enabled, _ta_enabled_for_epoch


def test_lif_models_never_report_ta_enabled() -> None:
    assert not _ta_enabled_for_epoch([], epoch=119, activation_epoch=6)


def test_ta_models_enable_only_at_the_activation_epoch() -> None:
    parameter = torch.nn.Parameter(torch.tensor(0.5))
    parameters = [parameter]

    assert not _ta_enabled_for_epoch(parameters, epoch=5, activation_epoch=6)
    assert _ta_enabled_for_epoch(parameters, epoch=6, activation_epoch=6)

    _set_ta_enabled(parameters, False)
    assert not parameter.requires_grad
    _set_ta_enabled(parameters, True)
    assert parameter.requires_grad
