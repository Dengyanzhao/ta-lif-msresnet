from __future__ import annotations

import os

import torch

from talif_msresnet import utils


def test_seed_everything_configures_strict_cuda_determinism(monkeypatch) -> None:
    calls: list[tuple[bool, object]] = []

    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)
    monkeypatch.setattr(
        torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, **kwargs: calls.append((enabled, kwargs.get("warn_only"))),
    )

    utils.seed_everything(11, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls == [(True, None)]


def test_seed_everything_preserves_supported_explicit_cublas_workspace(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda enabled: None)

    utils.seed_everything(22, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


def test_seed_everything_rejects_unsupported_cublas_workspace(monkeypatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "invalid")

    try:
        utils.seed_everything(33, deterministic=True)
    except RuntimeError as exc:
        assert "Unsupported CUBLAS_WORKSPACE_CONFIG" in str(exc)
    else:
        raise AssertionError("Unsupported deterministic CuBLAS setting was accepted")
