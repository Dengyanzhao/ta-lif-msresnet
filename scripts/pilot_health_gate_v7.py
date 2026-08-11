#!/usr/bin/env python3
"""Run the one-shot, non-reportable V7 six-condition health gate."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
SCRIPTS_ROOT = REPOSITORY_ROOT / "scripts"
for index, path in enumerate((SRC_ROOT, SCRIPTS_ROOT)):
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(index, value)

import pilot_health_gate as legacy_gate
import pilot_health_gate_v3 as v3_gate

from talif_msresnet.config import (
    RunConfig,
    artifact_paths_for_protocol,
    load_protocol,
    load_run_config,
)
from talif_msresnet.config_v7 import (
    V7_ACTIVE_CONDITIONS,
    validate_v7_cifar100_provenance_files,
)
from talif_msresnet.data import build_loaders
from talif_msresnet.models import build_model
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.pilot_v3 import repository_git_identity
from talif_msresnet.pilot_v7 import (
    HEALTH_ARTIFACT_CLASS,
    HEALTH_SCHEMA_VERSION,
    HEALTH_SEED_DISPOSITION,
    PILOT_SEED_DISPOSITION,
    REPORTING_ELIGIBILITY,
    V7_HEALTH_RUNTIME_SOURCE_PATHS,
    PilotBlock,
    PilotV7Error,
    attempt_receipt_payload,
    exclusive_create_json,
    expected_pilot_configs,
    health_source_binding,
    json_file_payload_sha256,
    require_v7_author_freeze,
    resolve_pilot_block,
    validate_health_recovery_release,
    validate_health_report,
    validate_v7_runtime_environment,
)
from talif_msresnet.train import (
    _checkpoint_payload,
    _forward,
    _load_resume,
    _set_ta_enabled,
    _shared_weight_sha256,
    _training_environment_identity,
    build_optimizer_and_scheduler,
    evaluate,
    optimizer_group_manifest,
    train_one_epoch,
)
from talif_msresnet.utils import (
    environment_manifest,
    save_checkpoint,
    seed_everything,
    sha256_file,
    utc_now,
)

ADAPTIVE_CONDITIONS = frozenset({"M2", "M3", "M4", "PLIF"})
RUNTIME_SOURCE_PATHS = tuple(
    REPOSITORY_ROOT / relative for relative in V7_HEALTH_RUNTIME_SOURCE_PATHS
)


class _NullLogger:
    def log(self, _event: str, **_values: Any) -> None:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "protocol_v7_mechanism.yaml",
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--device", default="cuda:0")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--precheck-only",
        action="store_true",
        help="Run reversible V7 health setup checks without claiming the health seed.",
    )
    mode.add_argument(
        "--pilot-compatibility-precheck-only",
        action="store_true",
        help=(
            "Validate the sealed recovery against the consumed health PASS without "
            "creating pilot artifacts or authorizing training."
        ),
    )
    return parser


def _repository_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def _load_fixed_health_batch(
    base: RunConfig,
    *,
    seed: int,
    required_batch_size: int,
) -> tuple[RunConfig, torch.Tensor, torch.Tensor, Mapping[str, Any], dict[str, Any]]:
    """Load a reproducible augmented batch without changing caller RNG state."""

    with torch.random.fork_rng(devices=[]):
        cpu_generator = torch.Generator(device="cpu").manual_seed(int(seed))
        torch.set_rng_state(cpu_generator.get_state())
        return v3_gate.load_fixed_real_batch(
            base,
            seed=seed,
            required_batch_size=required_batch_size,
        )


def _strict_json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _strict_json_safe(value.detach().cpu().item())
        return _strict_json_safe(value.detach().cpu().tolist())
    if isinstance(value, Mapping):
        return {str(key): _strict_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_strict_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _runtime_source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for path in RUNTIME_SOURCE_PATHS:
        resolved = path.resolve()
        if not resolved.is_file():
            raise PilotV7Error(f"Required V7 runtime source is missing: {resolved}")
        result[artifact_path_reference(resolved, REPOSITORY_ROOT)] = sha256_file(resolved)
    return result


def _require_head_bound_sources(paths: Sequence[Path]) -> None:
    relative: list[str] = []
    root = REPOSITORY_ROOT.resolve()
    for raw in paths:
        path = raw.resolve()
        try:
            relative.append(path.relative_to(root).as_posix())
        except ValueError as exc:
            raise PilotV7Error(f"Runtime source escapes the repository: {path}") from exc
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        detail = tracked.stderr.strip() or tracked.stdout.strip()
        raise PilotV7Error(f"V7 runtime inputs must be committed: {detail}")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *relative],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if unchanged.returncode == 1:
        raise PilotV7Error("V7 runtime inputs differ from the current HEAD")
    if unchanged.returncode != 0:
        detail = unchanged.stderr.strip() or unchanged.stdout.strip()
        raise PilotV7Error(f"Cannot bind V7 runtime inputs to HEAD: {detail}")


def _require_runtime_sources_unchanged(
    expected: Mapping[str, str], expected_git: Mapping[str, Any]
) -> dict[str, Any]:
    current = repository_git_identity(REPOSITORY_ROOT)
    if current.get("git_commit") != expected_git.get("git_commit"):
        raise PilotV7Error("Git commit changed during the one-shot V7 health gate")
    if current.get("tracked_clean") is not True:
        raise PilotV7Error("Tracked files changed during the one-shot V7 health gate")
    if _runtime_source_hashes() != dict(expected):
        raise PilotV7Error("A V7 runtime source changed during the one-shot health gate")
    return current


def _load_exact_configs(
    protocol: Mapping[str, Any], protocol_path: Path, config_dir: Path
) -> tuple[tuple[Path, ...], tuple[RunConfig, ...]]:
    expected = expected_pilot_configs(protocol)
    paths: list[Path] = []
    observed: list[RunConfig] = []
    for config in expected:
        path = (config_dir / f"{config.runtime.run_id}.yaml").resolve()
        if path.parent != config_dir.resolve() or not path.is_file():
            raise PilotV7Error(f"Canonical V7 pilot config is missing: {path}")
        loaded = load_run_config(path, protocol_path)
        if loaded.as_dict() != config.as_dict():
            raise PilotV7Error(
                f"Generated V7 pilot YAML differs from protocol for {config.model.condition}"
            )
        paths.append(path)
        observed.append(loaded)
    if tuple(config.model.condition for config in observed) != V7_ACTIVE_CONDITIONS:
        raise PilotV7Error("V7 health configs are not the ordered six-condition block")
    return tuple(paths), tuple(observed)


def _configure_deterministic_runtime() -> None:
    """Enable deterministic kernels without touching a frozen experiment seed."""

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, TypeError) as exc:  # pragma: no cover - old torch only
        raise PilotV7Error(
            "PyTorch cannot enable deterministic algorithms for the V7 preclaim gate"
        ) from exc
    if not torch.are_deterministic_algorithms_enabled() or (
        torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise PilotV7Error("V7 preclaim deterministic runtime is not strict")


def _expected_adaptive_updates(
    health_contract: Mapping[str, Any],
) -> dict[str, bool]:
    """Read the condition-update expectations from the frozen V7 contract."""

    raw = health_contract.get("expected_adaptive_update_by_condition")
    if not isinstance(raw, Mapping):
        raise PilotV7Error(
            "V7 health contract has no expected_adaptive_update_by_condition mapping"
        )
    if tuple(raw) != V7_ACTIVE_CONDITIONS:
        raise PilotV7Error(
            "V7 expected adaptive-update mapping differs from the frozen condition order"
        )
    if any(not isinstance(raw[condition], bool) for condition in V7_ACTIVE_CONDITIONS):
        raise PilotV7Error("V7 adaptive-update expectations must be boolean")
    return {condition: bool(raw[condition]) for condition in V7_ACTIVE_CONDITIONS}


def _preclaim_gate(
    *,
    protocol: Mapping[str, Any],
    configs: Sequence[RunConfig],
    block: PilotBlock,
    device_name: str,
    git_identity: Mapping[str, Any],
    runtime_sources: Mapping[str, str],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Run all reversible runtime/data checks before claiming the health seed.

    The preclaim gate intentionally does not build a model, execute a training
    step, write a health report, or create the one-shot attempt receipt. It is
    therefore safe to rerun after a hardware or data-setup failure.
    """

    if tuple(config.model.condition for config in configs) != block.conditions:
        raise PilotV7Error("V7 preclaim configs do not cover the frozen condition order")
    if any(config.runtime.seed != block.pilot_seed for config in configs):
        raise PilotV7Error("V7 preclaim reference configs must retain the pilot seed")
    if any(not config.runtime.deterministic or config.runtime.amp for config in configs):
        raise PilotV7Error("V7 preclaim requires deterministic float32 configs")
    if not isinstance(device_name, str) or not device_name.strip():
        raise PilotV7Error("V7 health gate requires an explicit CUDA device")
    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise PilotV7Error(f"Invalid V7 CUDA device {device_name!r}: {exc}") from exc
    if device.type != "cuda" or device.index is None:
        raise PilotV7Error("V7 health preclaim requires an explicit CUDA device such as cuda:0")

    environment_contract = protocol["pilot_acceptance"]["environment"]
    _configure_deterministic_runtime()
    hardware = legacy_gate.require_target_cuda(
        device, str(environment_contract["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_contract, hardware)
    validate_v7_runtime_environment(environment_contract)
    expected_cublas = str(environment_contract["cublas_workspace_config"])
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != expected_cublas:
        raise PilotV7Error("CUBLAS_WORKSPACE_CONFIG differs from the V7 environment contract")
    idle = legacy_gate.gpu_idle_precheck(device)
    environment_text, environment_sha256 = _training_environment_identity(
        device, amp=False, deterministic=True
    )
    environment_identity = json.loads(environment_text)
    if not _finite_tree(environment_identity):
        raise PilotV7Error("V7 preclaim environment identity contains non-finite values")
    if repository_git_identity(REPOSITORY_ROOT) != dict(git_identity):
        raise PilotV7Error("Git identity changed before the V7 health claim")
    if _runtime_source_hashes() != dict(runtime_sources):
        raise PilotV7Error("V7 runtime source hashes changed before the health claim")

    if configs[0].data.dataset != "cifar100":
        raise PilotV7Error("V7 preclaim loader must be bound to CIFAR-100")
    try:
        cifar100_provenance = validate_v7_cifar100_provenance_files(
            protocol,
            project_root=REPOSITORY_ROOT,
        )
    except ValueError as exc:
        raise PilotV7Error(
            f"V7 preclaim CIFAR-100 provenance differs from the frozen protocol: {exc}"
        ) from exc
    health_contract = protocol["pilot_acceptance"]["health"]
    required_batch = int(health_contract["fixed_batch_size"])
    _resolved, inputs, targets, split_manifest, source_identity = _load_fixed_health_batch(
        configs[0], seed=block.health_seed, required_batch_size=required_batch
    )
    (
        _reconstructed,
        reconstructed_inputs,
        reconstructed_targets,
        reconstructed_manifest,
        reconstructed_source_identity,
    ) = _load_fixed_health_batch(
        configs[0], seed=block.health_seed, required_batch_size=required_batch
    )
    if int(inputs.shape[0]) != required_batch or int(targets.shape[0]) != required_batch:
        raise PilotV7Error("V7 preclaim fixed batch does not have the frozen size")
    if not torch.isfinite(inputs).all().item() or targets.dtype != torch.long:
        raise PilotV7Error("V7 preclaim fixed batch is non-finite or has invalid labels")
    if not isinstance(split_manifest, Mapping) or not split_manifest.get("manifest_sha256"):
        raise PilotV7Error("V7 preclaim loader did not return a split manifest")
    if not isinstance(reconstructed_manifest, Mapping) or not isinstance(
        reconstructed_source_identity, Mapping
    ):
        raise PilotV7Error("V7 preclaim reconstruction returned invalid source evidence")
    fixed_batch_sha256 = legacy_gate.tensor_batch_sha256(inputs, targets)
    reconstructed_batch_sha256 = legacy_gate.tensor_batch_sha256(
        reconstructed_inputs, reconstructed_targets
    )
    if reconstructed_batch_sha256 != fixed_batch_sha256:
        raise PilotV7Error(
            "V7 preclaim fixed batch cannot be reconstructed exactly before seed claim"
        )
    if dict(reconstructed_manifest) != dict(split_manifest):
        raise PilotV7Error(
            "V7 preclaim split manifest changed between fixed-batch reconstructions"
        )
    if dict(reconstructed_source_identity) != dict(source_identity):
        raise PilotV7Error(
            "V7 preclaim source identity changed between fixed-batch reconstructions"
        )

    # Explicitly touch the validation loader without opening the held-out test split.
    resolved_data = v3_gate._resolved_data_config(configs[0])
    resolved_data = dataclasses.replace(
        resolved_data,
        optimizer=dataclasses.replace(resolved_data.optimizer, batch_size=required_batch),
        runtime=dataclasses.replace(resolved_data.runtime, seed=block.health_seed),
        final_test=False,
    )
    loaders = build_loaders(resolved_data, final_test=False, seed=block.health_seed)
    if "test" in loaders or "val" not in loaders:
        raise PilotV7Error("V7 preclaim unexpectedly touched or omitted validation")
    validation_loader = loaders["val"]
    if len(validation_loader) < 1:
        raise PilotV7Error("V7 preclaim validation loader is empty")
    validation_batch = next(iter(validation_loader))
    if (
        not isinstance(validation_batch, (tuple, list))
        or len(validation_batch) != 2
        or int(validation_batch[0].shape[0]) < 1
        or not torch.isfinite(torch.as_tensor(validation_batch[0])).all().item()
    ):
        raise PilotV7Error("V7 preclaim validation loader yielded an invalid batch")
    validation_manifest = loaders.get("_manifest")
    if not isinstance(validation_manifest, Mapping) or (
        validation_manifest.get("manifest_sha256") != split_manifest.get("manifest_sha256")
    ):
        raise PilotV7Error("V7 preclaim validation split differs from fixed batch")
    return {
        "pass": True,
        "device": str(device),
        "hardware": hardware,
        "gpu_idle_precheck": idle,
        "training_environment_identity": environment_identity,
        "training_environment_sha256": environment_sha256,
        "split_manifest_sha256": split_manifest["manifest_sha256"],
        "fixed_batch_sha256": fixed_batch_sha256,
        "fixed_batch_reconstruction_sha256": reconstructed_batch_sha256,
        "fixed_batch_reconstruction_count": 2,
        "fixed_batch_shape": list(inputs.shape),
        "fixed_batch_size": required_batch,
        "validation_batch_shape": list(validation_batch[0].shape),
        "validation_batch_size": int(validation_batch[0].shape[0]),
        "source": dict(source_binding),
        "cifar100_provenance": dict(cifar100_provenance),
        "source_identity": dict(source_identity),
        "runtime_sources_sha256": dict(runtime_sources),
    }


def _diagnostic_config(
    config: RunConfig,
    *,
    health_seed: int,
    device: torch.device,
    output_parent: Path,
) -> RunConfig:
    runtime = dataclasses.replace(
        config.runtime,
        seed=health_seed,
        device=str(device),
        output_dir=str(output_parent),
        run_id=f"NONREPORTING_v7_health_{config.model.condition}_s{health_seed}",
        amp=False,
        deterministic=True,
        dry_run=False,
        limit_batches=None,
        resume=None,
    )
    return dataclasses.replace(config, runtime=runtime, final_test=False)


def _build_objects(
    config: RunConfig, device: torch.device
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    Any,
    list[torch.nn.Parameter],
    list[str],
    int,
]:
    model_config = dataclasses.asdict(config.model)
    model_config["init_seed"] = config.runtime.seed
    model = build_model(model_config).to(device)
    optimizer, scheduler, adaptive, names, activation = build_optimizer_and_scheduler(model, config)
    return model, optimizer, scheduler, list(adaptive), list(names), int(activation)


def _fixed_batch_metrics(
    model: torch.nn.Module, batch: tuple[torch.Tensor, torch.Tensor]
) -> dict[str, float]:
    inputs, targets = batch
    model.eval()
    with torch.no_grad():
        logits, _ = _forward(model, inputs, collect_activity=False)
        loss = float(torch.nn.functional.cross_entropy(logits, targets).item())
        accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise PilotV7Error("V7 fixed-batch evaluation produced non-finite metrics")
    return {"loss": loss, "accuracy": accuracy}


def _named_gradient_accumulator(
    model: torch.nn.Module, names: Sequence[str]
) -> dict[str, dict[str, Any]]:
    parameters = dict(model.named_parameters())
    missing = sorted(set(names) - set(parameters))
    if missing:
        raise PilotV7Error(f"Gradient audit names are absent from the model: {missing}")
    return {
        name: {
            "probe_count": 0,
            "present_on_every_probe": True,
            "finite_on_every_probe": True,
            "ever_nonzero": False,
            "maximum_l2_norm": 0.0,
        }
        for name in names
    }


def _update_gradient_accumulator(
    model: torch.nn.Module, accumulator: MutableMapping[str, dict[str, Any]]
) -> None:
    parameters = dict(model.named_parameters())
    for name, row in accumulator.items():
        row["probe_count"] += 1
        gradient = parameters[name].grad
        if gradient is None:
            row["present_on_every_probe"] = False
            row["finite_on_every_probe"] = False
            continue
        finite = bool(torch.isfinite(gradient).all().item())
        row["finite_on_every_probe"] = bool(row["finite_on_every_probe"] and finite)
        if not finite:
            continue
        norm = float(torch.linalg.vector_norm(gradient.detach().double()).item())
        row["maximum_l2_norm"] = max(float(row["maximum_l2_norm"]), norm)
        row["ever_nonzero"] = bool(row["ever_nonzero"] or norm > 0.0)


def _gradient_summary(accumulator: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    passed = bool(accumulator) and all(
        bool(row.get("present_on_every_probe"))
        and bool(row.get("finite_on_every_probe"))
        and bool(row.get("ever_nonzero"))
        for row in accumulator.values()
    )
    return {
        "pass": passed,
        "required_parameter_count": len(accumulator),
        "passed_parameter_count": sum(
            bool(row.get("present_on_every_probe"))
            and bool(row.get("finite_on_every_probe"))
            and bool(row.get("ever_nonzero"))
            for row in accumulator.values()
        ),
        "parameters": {name: dict(row) for name, row in accumulator.items()},
    }


def _optimizer_state_report(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    state = optimizer.state_dict()
    finite = _finite_tree(state)
    parameter_states = state.get("state")
    populated = isinstance(parameter_states, Mapping) and bool(parameter_states)
    groups = state.get("param_groups")
    return {
        "pass": bool(finite and populated and isinstance(groups, list) and groups),
        "finite": finite,
        "populated_parameter_state_count": (
            len(parameter_states) if isinstance(parameter_states, Mapping) else 0
        ),
        "parameter_group_count": len(groups) if isinstance(groups, list) else 0,
    }


def _adaptive_update_report(
    model: torch.nn.Module,
    names: Sequence[str],
    initial: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    rows: dict[str, Any] = {}
    for name in names:
        current = parameters[name].detach()
        before = initial[name].to(device=current.device, dtype=current.dtype)
        changed = current != before
        rows[name] = {
            "changed_elements": int(torch.count_nonzero(changed).item()),
            "numel": int(current.numel()),
            "finite": bool(torch.isfinite(current).all().item()),
            "maximum_absolute_change": float((current - before).abs().max().item()),
        }
    nonzero = bool(rows) and all(row["changed_elements"] > 0 for row in rows.values())
    return {
        "pass": bool(nonzero and all(row["finite"] for row in rows.values())),
        "nonzero": nonzero,
        "parameters": rows,
    }


def _one_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    step: int,
) -> dict[str, Any]:
    return train_one_epoch(
        model,
        [batch],
        optimizer,
        device,
        step,
        config,
        _NullLogger(),  # type: ignore[arg-type]
        legacy_gate._grad_scaler(),
    )


def _fixed_probe(
    config: RunConfig,
    *,
    device: torch.device,
    batch: tuple[torch.Tensor, torch.Tensor],
    steps: int,
) -> dict[str, Any]:
    seed_everything(config.runtime.seed, deterministic=True)
    model, optimizer, _scheduler, adaptive, adaptive_names, activation = _build_objects(
        config, device
    )
    expects_adaptive = config.model.condition in ADAPTIVE_CONDITIONS
    if expects_adaptive != bool(adaptive_names):
        raise PilotV7Error(
            f"{config.model.condition}: adaptive parameter ownership differs from protocol"
        )
    _set_ta_enabled(adaptive, expects_adaptive)
    initial_adaptive = {
        name: dict(model.named_parameters())[name].detach().cpu().clone() for name in adaptive_names
    }
    shared_names = legacy_gate._shared_gradient_parameter_names(model)
    shared_accumulator = _named_gradient_accumulator(model, shared_names)
    adaptive_accumulator = _named_gradient_accumulator(model, adaptive_names)
    initial_metrics = _fixed_batch_metrics(model, batch)
    losses: list[float] = []
    finite = True
    for step in range(steps):
        metrics = _one_step(model, optimizer, config, batch, device, step)
        losses.append(float(metrics["loss"]))
        _update_gradient_accumulator(model, shared_accumulator)
        if adaptive_accumulator:
            _update_gradient_accumulator(model, adaptive_accumulator)
        finite = bool(
            finite
            and _finite_tree(metrics)
            and all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        )
    final_metrics = _fixed_batch_metrics(model, batch)
    shared_gradients = _gradient_summary(shared_accumulator)
    adaptive_gradients = (
        _gradient_summary(adaptive_accumulator)
        if expects_adaptive
        else {"pass": True, "status": "NOT_APPLICABLE", "parameters": {}}
    )
    adaptive_update = (
        _adaptive_update_report(model, adaptive_names, initial_adaptive)
        if expects_adaptive
        else {"pass": True, "nonzero": False, "parameters": {}}
    )
    optimizer_state = _optimizer_state_report(optimizer)
    manifest = optimizer_group_manifest(model, optimizer)
    adaptive_groups = [group for group in manifest["groups"] if group.get("role") == "adaptive"]
    if expects_adaptive:
        group_pass = bool(
            len(adaptive_groups) == 1
            and adaptive_groups[0].get("parameter_names") == adaptive_names
            and math.isclose(
                float(adaptive_groups[0].get("initial_lr", math.nan)),
                config.optimizer.lr * config.optimizer.ta_lr_scale,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            and float(adaptive_groups[0].get("weight_decay", math.nan))
            == config.optimizer.ta_weight_decay
        )
    else:
        group_pass = not adaptive_groups and adaptive_names == []
    passed = bool(
        finite
        and shared_gradients["pass"]
        and adaptive_gradients["pass"]
        and adaptive_update["pass"]
        and optimizer_state["pass"]
        and group_pass
    )
    return {
        "pass": passed,
        "steps": steps,
        "finite": finite,
        "activation_epoch_zero_based": activation,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "loss_history": losses,
        "shared_gradient_audit": shared_gradients,
        "adaptive_gradient_audit": adaptive_gradients,
        "adaptive_update": adaptive_update,
        "optimizer_state": optimizer_state,
        "optimizer_group_manifest": manifest,
        "optimizer_group_contract_pass": group_pass,
        "adaptive_parameter_names": adaptive_names,
    }


def _continuation_snapshot(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return legacy_gate._continuation_snapshot(model, optimizer, scheduler, metrics)


def _checkpoint_resume_check(
    config: RunConfig,
    *,
    device: torch.device,
    batch: tuple[torch.Tensor, torch.Tensor],
    checkpoint_path: Path,
    training_environment_sha256: str,
    boundary_epoch: int,
) -> dict[str, Any]:
    seed_everything(config.runtime.seed, deterministic=True)
    if checkpoint_path.name != "last.pt":
        raise PilotV7Error("V7 checkpoint health evidence must use last.pt")
    model, optimizer, scheduler, adaptive, adaptive_names, activation = _build_objects(
        config, device
    )
    if activation != boundary_epoch + 1:
        raise PilotV7Error(
            f"{config.model.condition}: activation epoch {activation} does not cross "
            f"checkpoint epoch {boundary_epoch}"
        )
    history: list[dict[str, Any]] = []
    before: dict[str, Any] | None = None
    for epoch in range(boundary_epoch + 1):
        _set_ta_enabled(adaptive, False)
        before = _one_step(model, optimizer, config, batch, device, epoch)
        history.append({"epoch": epoch, **before})
        scheduler.step()
    assert before is not None
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        boundary_epoch,
        {
            "accuracy": float(before["accuracy"]),
            "loss": float(before["loss"]),
            "epoch": boundary_epoch,
        },
        config,
        history,
        training_environment_sha256,
    )
    save_checkpoint(checkpoint_path, payload)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    expected_start = boundary_epoch + 1
    expected_enabled = bool(adaptive_names)
    _set_ta_enabled(adaptive, expected_enabled)
    expected_metrics = _one_step(model, optimizer, config, batch, device, expected_start)
    scheduler.step()
    expected = _continuation_snapshot(model, optimizer, scheduler, expected_metrics)

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_adaptive, resumed_names, _ = (
        _build_objects(config, device)
    )
    start_epoch, _best, resumed_history = _load_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        config,
        expected_training_environment_sha256=training_environment_sha256,
    )
    _set_ta_enabled(resumed_adaptive, bool(resumed_names and start_epoch >= activation))
    observed_metrics = _one_step(
        resumed_model,
        resumed_optimizer,
        config,
        batch,
        device,
        start_epoch,
    )
    resumed_scheduler.step()
    observed = _continuation_snapshot(
        resumed_model, resumed_optimizer, resumed_scheduler, observed_metrics
    )
    mismatches = legacy_gate._comparison_mismatches(expected, observed)
    exact = bool(start_epoch == expected_start and not mismatches)
    return {
        "pass": exact,
        "resume_exact": exact,
        "comparison": "bitwise_exact_next_training_step",
        "resume_loader": "talif_msresnet.train._load_resume",
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_epoch_zero_based": boundary_epoch,
        "expected_start_epoch_zero_based": expected_start,
        "resumed_start_epoch_zero_based": start_epoch,
        "activation_epoch_zero_based": activation,
        "adaptive_enabled_at_checkpoint": False,
        "adaptive_enabled_on_next_step": expected_enabled,
        "activation_boundary_crossed": bool(expected_enabled),
        "adaptive_parameter_names": adaptive_names,
        "resumed_history_rows": len(resumed_history),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_retained": False,
        "mismatch_count": len(mismatches),
        "mismatch_paths": mismatches[:50],
    }


def _schedule_probe(
    config: RunConfig,
    *,
    device: torch.device,
    batch: tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    train_batches: int,
    validation_batches: int,
) -> dict[str, Any]:
    seed_everything(config.runtime.seed, deterministic=True)
    model, optimizer, scheduler, adaptive, adaptive_names, activation = _build_objects(
        config, device
    )
    initial_adaptive = {
        name: dict(model.named_parameters())[name].detach().cpu().clone() for name in adaptive_names
    }
    states: list[bool] = []
    epoch_rows: list[dict[str, Any]] = []
    finite = True
    for epoch in range(epochs):
        enabled = bool(adaptive and epoch >= activation)
        states.append(enabled)
        _set_ta_enabled(adaptive, enabled)
        train_metrics = train_one_epoch(
            model,
            [batch] * train_batches,
            optimizer,
            device,
            epoch,
            config,
            _NullLogger(),  # type: ignore[arg-type]
            legacy_gate._grad_scaler(),
        )
        val_metrics = evaluate(
            model,
            [batch] * validation_batches,
            device,
            config,
            "validation_health",
        )
        finite = bool(
            finite
            and _finite_tree(train_metrics)
            and _finite_tree(val_metrics)
            and all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "adaptive_enabled": enabled,
                "train": train_metrics,
                "validation": val_metrics,
            }
        )
        scheduler.step()
    expected_states = [False] * (epochs - 1) + [bool(adaptive_names)]
    update = (
        _adaptive_update_report(model, adaptive_names, initial_adaptive)
        if adaptive_names
        else {"pass": True, "nonzero": False, "parameters": {}}
    )
    optimizer_state = _optimizer_state_report(optimizer)
    passed = bool(
        finite
        and states == expected_states
        and update["pass"]
        and optimizer_state["pass"]
        and activation == epochs - 1
    )
    return {
        "pass": passed,
        "finite": finite,
        "epochs": epochs,
        "train_batches_per_epoch": train_batches,
        "validation_batches_per_epoch": validation_batches,
        "activation_epoch_zero_based": activation,
        "adaptive_enabled_by_epoch": states,
        "adaptive_parameter_update": update,
        "optimizer_state": optimizer_state,
        "epochs_evidence": epoch_rows,
    }


def _initial_forward_evidence(
    configs: Sequence[RunConfig],
    *,
    device: torch.device,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, Any]:
    outputs: dict[str, torch.Tensor] = {}
    shared_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    for config in configs:
        seed_everything(config.runtime.seed, deterministic=True)
        model, _optimizer, _scheduler, _adaptive, _names, _activation = _build_objects(
            config, device
        )
        model.eval()
        with torch.no_grad():
            logits, _ = _forward(model, batch[0], collect_activity=False)
        condition = config.model.condition
        cpu_logits = logits.detach().cpu().contiguous()
        outputs[condition] = cpu_logits
        shared_hashes[condition] = _shared_weight_sha256(model)
        digest = hashlib.sha256()
        digest.update(str(tuple(cpu_logits.shape)).encode("ascii"))
        digest.update(str(cpu_logits.dtype).encode("ascii"))
        digest.update(cpu_logits.numpy().tobytes())
        output_hashes[condition] = digest.hexdigest()
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    reference = outputs[V7_ACTIVE_CONDITIONS[0]]
    equivalent = all(
        torch.equal(reference, outputs[condition]) for condition in V7_ACTIVE_CONDITIONS[1:]
    )
    shared_equal = len(set(shared_hashes.values())) == 1
    return {
        "pass": bool(equivalent and shared_equal),
        "initial_forward_equivalent": equivalent,
        "shared_initialization_equal": shared_equal,
        "output_sha256_by_condition": output_hashes,
        "shared_weight_sha256_by_condition": shared_hashes,
    }


def _condition_health_summary(
    condition: str,
    *,
    fixed: Mapping[str, Any],
    schedule: Mapping[str, Any],
    resume: Mapping[str, Any],
    expected_adaptive_update_by_condition: Mapping[str, bool],
) -> dict[str, Any]:
    adaptive_names = list(fixed["adaptive_parameter_names"])
    if condition not in expected_adaptive_update_by_condition:
        raise PilotV7Error(f"No frozen adaptive-update expectation for {condition}")
    expects_adaptive = expected_adaptive_update_by_condition[condition]
    if not isinstance(expects_adaptive, bool):
        raise PilotV7Error(f"Adaptive-update expectation for {condition} is not boolean")
    expected_states = [False] * 5 + [expects_adaptive]
    finite = bool(fixed["finite"] and schedule["finite"])
    adaptive_update = bool(fixed["adaptive_update"]["nonzero"])
    passed = bool(
        fixed["pass"]
        and schedule["pass"]
        and resume["pass"]
        and schedule["adaptive_enabled_by_epoch"] == expected_states
        and adaptive_update == expects_adaptive
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "finite": finite,
        "resume_exact": bool(resume["resume_exact"]),
        "adaptive_enabled_by_epoch": schedule["adaptive_enabled_by_epoch"],
        "adaptive_parameter_names": adaptive_names,
        "expected_adaptive_parameter_update_nonzero": expects_adaptive,
        "adaptive_parameter_update_nonzero": adaptive_update,
        "fixed_batch": fixed,
        "formal_schedule_boundary": schedule,
        "checkpoint_resume": resume,
    }


def run_gate(
    *,
    protocol: Mapping[str, Any],
    protocol_path: Path,
    config_paths: Sequence[Path],
    configs: Sequence[RunConfig],
    block: PilotBlock,
    device_name: str,
    git_identity: Mapping[str, Any],
    runtime_sources: Mapping[str, str],
    source_binding: Mapping[str, Any],
    attempt_receipt_sha256: str,
    expected_fixed_batch_sha256: str,
) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.perf_counter()
    health_contract = protocol["pilot_acceptance"]["health"]
    expected_adaptive_update_by_condition = _expected_adaptive_updates(health_contract)
    environment_contract = protocol["pilot_acceptance"]["environment"]
    if tuple(config.model.condition for config in configs) != block.conditions:
        raise PilotV7Error("V7 health run does not cover the frozen condition order")
    if any(config.runtime.seed != block.pilot_seed for config in configs):
        raise PilotV7Error("V7 reference configs must retain the disjoint pilot seed")
    if any(not config.runtime.deterministic or config.runtime.amp for config in configs):
        raise PilotV7Error("V7 health requires deterministic float32 reference configs")

    seed_everything(block.health_seed, deterministic=True)
    device = torch.device(device_name)
    hardware = legacy_gate.require_target_cuda(
        device, str(environment_contract["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_contract, hardware)
    validate_v7_runtime_environment(environment_contract)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != environment_contract["cublas_workspace_config"]:
        raise PilotV7Error("CUBLAS_WORKSPACE_CONFIG differs from the V7 environment contract")
    idle = legacy_gate.gpu_idle_precheck(device)
    environment_text, environment_sha256 = _training_environment_identity(
        device, amp=False, deterministic=True
    )
    environment_identity = json.loads(environment_text)

    required_batch = int(health_contract["fixed_batch_size"])
    _resolved, inputs, targets, split_manifest, source_identity = _load_fixed_health_batch(
        configs[0], seed=block.health_seed, required_batch_size=required_batch
    )
    inputs = inputs[:required_batch]
    targets = targets[:required_batch]
    if int(inputs.shape[0]) != required_batch or int(targets.shape[0]) != required_batch:
        raise PilotV7Error("V7 health loader did not yield the exact frozen batch size")
    fixed_batch_sha256 = legacy_gate.tensor_batch_sha256(inputs, targets)
    if fixed_batch_sha256 != expected_fixed_batch_sha256:
        raise PilotV7Error("V7 execution fixed batch differs from preclaim")
    try:
        cifar100_provenance = validate_v7_cifar100_provenance_files(
            protocol,
            project_root=REPOSITORY_ROOT,
        )
    except ValueError as exc:
        raise PilotV7Error(
            f"V7 execution CIFAR-100 provenance differs from the frozen protocol: {exc}"
        ) from exc
    batch = (
        inputs.to(device, non_blocking=False),
        targets.to(device, non_blocking=False),
    )
    diagnostic_configs = tuple(
        _diagnostic_config(
            config,
            health_seed=block.health_seed,
            device=device,
            output_parent=block.health_output.parent,
        )
        for config in configs
    )
    initial = _initial_forward_evidence(diagnostic_configs, device=device, batch=batch)

    by_condition: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="v7-health-", dir=str(block.health_output.parent)
    ) as temporary:
        temporary_root = Path(temporary)
        for config in diagnostic_configs:
            condition = config.model.condition
            fixed = _fixed_probe(
                config,
                device=device,
                batch=batch,
                steps=int(health_contract["fixed_batch_steps"]),
            )
            gc.collect()
            torch.cuda.empty_cache()
            schedule = _schedule_probe(
                config,
                device=device,
                batch=batch,
                epochs=int(health_contract["schedule_epochs"]),
                train_batches=int(health_contract["train_batches_per_epoch"]),
                validation_batches=int(health_contract["validation_batches_per_epoch"]),
            )
            gc.collect()
            torch.cuda.empty_cache()
            resume = _checkpoint_resume_check(
                config,
                device=device,
                batch=batch,
                checkpoint_path=temporary_root / condition / "last.pt",
                training_environment_sha256=environment_sha256,
                boundary_epoch=int(health_contract["required_ta_activation_epoch_zero_based"]) - 1,
            )
            by_condition[condition] = _condition_health_summary(
                condition,
                fixed=fixed,
                schedule=schedule,
                resume=resume,
                expected_adaptive_update_by_condition=(
                    expected_adaptive_update_by_condition
                ),
            )
            gc.collect()
            torch.cuda.empty_cache()

    anomalies: list[str] = []
    if not initial["pass"]:
        anomalies.append("six-condition initial forward/shared initialization equivalence failed")
    for condition in block.conditions:
        if by_condition[condition]["status"] != "PASS":
            anomalies.append(f"{condition}: implementation health checks failed")
    _require_runtime_sources_unchanged(runtime_sources, git_identity)
    status = "PASS" if not anomalies else "FAIL"
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v7-pilot six-condition implementation health validation",
        "status": status,
        "pass": status == "PASS",
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity["git_commit"],
        "tracked_clean": git_identity["tracked_clean"],
        "attempt_receipt": artifact_path_reference(block.attempt_receipt, REPOSITORY_ROOT),
        "attempt_receipt_sha256": attempt_receipt_sha256,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
        "source": {
            **dict(source_binding),
            "reference_config_hash": configs[0].config_hash,
            "reference_config_seed": configs[0].runtime.seed,
            "health_seed": block.health_seed,
            "pilot_seed": block.pilot_seed,
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "fixed_batch_sha256": fixed_batch_sha256,
            "fixed_batch_shape": list(inputs.shape),
            "fixed_batch_size": required_batch,
            "cifar100_provenance": dict(cifar100_provenance),
            **source_identity,
        },
        "environment": {
            **environment_manifest(),
            "contract": dict(environment_contract),
            "hardware": hardware,
            "training_environment_identity": environment_identity,
            "training_environment_sha256": environment_sha256,
            "precision": "float32",
            "amp": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu_idle_precheck": idle,
        },
        "checks": {
            "contract": dict(health_contract),
            "expected_adaptive_update_by_condition": (
                expected_adaptive_update_by_condition
            ),
            "performance_thresholds_evaluated": [],
            "initial_forward_equivalent": initial["initial_forward_equivalent"],
            "shared_initialization_equal": initial["shared_initialization_equal"],
            "initial_evidence": initial,
            "by_condition": by_condition,
        },
        "integrity_anomalies": anomalies,
        "failures": anomalies,
    }


def _fatal_report(
    *,
    block: PilotBlock,
    git_identity: Mapping[str, Any],
    runtime_sources: Mapping[str, str],
    source_binding: Mapping[str, Any],
    attempt_receipt_sha256: str,
    preclaim: Mapping[str, Any],
    exc: BaseException,
) -> dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    return {
        "schema_version": HEALTH_SCHEMA_VERSION,
        "artifact_class": HEALTH_ARTIFACT_CLASS,
        "reporting_eligibility": REPORTING_ELIGIBILITY,
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v7-pilot six-condition implementation health validation",
        "status": "ERROR",
        "pass": False,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "conditions": list(block.conditions),
        "protocol_hash": block.protocol_hash,
        "acceptance_hash": block.acceptance_hash,
        "block_hash": block.block_hash,
        "git_commit": git_identity.get("git_commit"),
        "tracked_clean": git_identity.get("tracked_clean"),
        "attempt_receipt": artifact_path_reference(block.attempt_receipt, REPOSITORY_ROOT),
        "attempt_receipt_sha256": attempt_receipt_sha256,
        "finished_at": utc_now(),
        "fatal_error": message,
        "traceback": traceback.format_exc(),
        "source": dict(source_binding),
        "preclaim": _strict_json_safe(preclaim),
        "integrity_anomalies": ["fatal execution error"],
        "failures": [message],
        "health_seed_disposition": HEALTH_SEED_DISPOSITION,
        "pilot_seed_disposition": PILOT_SEED_DISPOSITION,
    }


def validate_current_runtime_against_health_report(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
    block: PilotBlock,
    device: str,
) -> dict[str, Any]:
    """Rebuild V7 health context while authorizing only the pilot seed."""

    if report.get("status") != "PASS" or report.get("pass") is not True:
        raise PilotV7Error("Only a canonical V7 health PASS can authorize the pilot")
    if reference_config.data.dataset != block.dataset:
        raise PilotV7Error("V7 pilot reference config dataset differs from the health block")
    if reference_config.runtime.seed != block.pilot_seed:
        raise PilotV7Error("V7 pilot launch config must retain the pilot seed")
    if not device or device == "auto":
        raise PilotV7Error("V7 pilot launch requires an explicit CUDA device")
    identity = repository_git_identity(REPOSITORY_ROOT)
    recovery = report.get("compatibility_recovery")
    verified_recovery: dict[str, Any] | None = None
    expected_commit = report.get("git_commit")
    if recovery is not None:
        if not isinstance(recovery, Mapping):
            raise PilotV7Error("V7 health compatibility recovery binding is malformed")
        verified_recovery = validate_health_recovery_release(
            REPOSITORY_ROOT,
            health_path=block.health_output,
            attempt_receipt_path=block.attempt_receipt,
        )
        if dict(recovery) != verified_recovery:
            raise PilotV7Error("V7 health compatibility recovery binding changed")
        expected_commit = verified_recovery["recovery_commit"]
    if identity.get("tracked_clean") is not True or identity.get(
        "git_commit"
    ) != expected_commit:
        raise PilotV7Error("Current Git identity differs from the V7 authorized release")
    environment_contract = protocol["pilot_acceptance"]["environment"]
    fixed_contract = protocol["pilot_acceptance"]["health"]
    selected_device = torch.device(device)
    seed_everything(block.health_seed, deterministic=True)
    hardware = legacy_gate.require_target_cuda(
        selected_device, str(environment_contract["expected_gpu_substring"])
    )
    legacy_gate.validate_runtime_environment(environment_contract, hardware)
    validate_v7_runtime_environment(environment_contract)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != environment_contract["cublas_workspace_config"]:
        raise PilotV7Error("Current CUBLAS workspace setting differs from V7 health")
    environment_text, environment_hash = _training_environment_identity(
        selected_device, amp=False, deterministic=True
    )
    report_environment = report.get("environment")
    if (
        not isinstance(report_environment, Mapping)
        or report_environment.get("training_environment_sha256") != environment_hash
    ):
        raise PilotV7Error("Current training environment differs from the V7 health report")
    if report_environment.get("training_environment_identity") != json.loads(environment_text):
        raise PilotV7Error("Current training environment identity payload changed")
    idle = legacy_gate.gpu_idle_precheck(selected_device)
    prior_idle = report_environment.get("gpu_idle_precheck")
    if not isinstance(prior_idle, Mapping) or prior_idle.get("device_uuid") != idle.get(
        "device_uuid"
    ):
        raise PilotV7Error("Current CUDA device UUID differs from the V7 health report")
    required = int(fixed_contract["fixed_batch_size"])
    _resolved, inputs, targets, manifest, source_identity = _load_fixed_health_batch(
        reference_config, seed=block.health_seed, required_batch_size=required
    )
    inputs = inputs[:required]
    targets = targets[:required]
    try:
        cifar100_provenance = validate_v7_cifar100_provenance_files(
            protocol,
            project_root=REPOSITORY_ROOT,
        )
    except ValueError as exc:
        raise PilotV7Error(
            f"Current V7 CIFAR-100 provenance differs from health: {exc}"
        ) from exc
    source = report.get("source")
    if not isinstance(source, Mapping):
        raise PilotV7Error("V7 health report source evidence is missing")
    observed = {
        "split_manifest_sha256": manifest["manifest_sha256"],
        "fixed_batch_sha256": legacy_gate.tensor_batch_sha256(inputs, targets),
        "fixed_batch_shape": list(inputs.shape),
        "fixed_batch_size": required,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        **source_identity,
    }
    mismatches = [key for key, value in observed.items() if source.get(key) != value]
    if source.get("cifar100_provenance") != cifar100_provenance:
        mismatches.append("cifar100_provenance")
    preclaim = report.get("preclaim")
    if not isinstance(preclaim, Mapping) or preclaim.get(
        "cifar100_provenance"
    ) != cifar100_provenance:
        mismatches.append("preclaim.cifar100_provenance")
    runtime_hashes = source.get("runtime_sources_sha256")
    expected_runtime_hashes = dict(runtime_hashes) if isinstance(runtime_hashes, Mapping) else {}
    if verified_recovery is not None:
        implementation_hashes = verified_recovery.get("implementation_file_sha256")
        if not isinstance(implementation_hashes, Mapping):
            raise PilotV7Error("V7 health compatibility runtime binding is malformed")
        expected_runtime_hashes.update(
            {
                relative: str(digest)
                for relative, digest in implementation_hashes.items()
                if relative in V7_HEALTH_RUNTIME_SOURCE_PATHS
            }
        )
    if not isinstance(runtime_hashes, Mapping) or (
        _runtime_source_hashes() != expected_runtime_hashes
    ):
        mismatches.append("runtime_sources_sha256")
    if mismatches:
        raise PilotV7Error(
            "Current V7 dataset/runtime identity differs from health at: "
            + ", ".join(dict.fromkeys(mismatches))
        )
    return {
        "pass": True,
        "protocol_version": 7,
        "dataset": block.dataset,
        "health_seed": block.health_seed,
        "pilot_seed": block.pilot_seed,
        "block_hash": block.block_hash,
        "device": hardware["device"],
        "device_uuid": idle["device_uuid"],
        "hardware": hardware,
        "training_environment_sha256": environment_hash,
        "cifar100_provenance": cifar100_provenance,
        **(
            {"health_compatibility_recovery": verified_recovery}
            if verified_recovery is not None
            else {}
        ),
        **observed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        protocol_path = args.protocol.resolve()
        protocol = load_protocol(protocol_path)
        if protocol.get("protocol_version") != 7:
            raise PilotV7Error("This health gate accepts only protocol_version 7")
        block = resolve_pilot_block(protocol, repository_root=REPOSITORY_ROOT)
        author = require_v7_author_freeze(protocol, repository_root=REPOSITORY_ROOT)
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else _repository_path(artifact_paths_for_protocol(protocol)["pilot_matrix"])
        )
        expected_config_dir = _repository_path(
            artifact_paths_for_protocol(protocol)["pilot_matrix"]
        )
        if config_dir != expected_config_dir:
            raise PilotV7Error("V7 health config directory differs from artifact binding")
        config_paths, configs = _load_exact_configs(protocol, protocol_path, config_dir)
        pilot_validation = _repository_path(
            protocol["pilot_acceptance"]["validation_output"]
        )
        pilot_artifacts = (
            ("pilot plan", block.pilot_plan),
            ("pilot output root", block.pilot_output_root),
            ("pilot validation", pilot_validation),
        )
        if args.pilot_compatibility_precheck_only:
            if not block.health_output.is_file() or not block.attempt_receipt.is_file():
                raise PilotV7Error(
                    "V7 compatibility precheck requires the existing canonical health PASS "
                    "and attempt receipt"
                )
            for label, path in pilot_artifacts:
                if path.exists():
                    raise PilotV7Error(
                        f"{label} exists before the zero-seed compatibility precheck: {path}"
                    )
            health_sha256_before = sha256_file(block.health_output)
            receipt_sha256_before = sha256_file(block.attempt_receipt)
            bound_paths = (
                *RUNTIME_SOURCE_PATHS,
                protocol_path,
                *config_paths,
                REPOSITORY_ROOT / artifact_paths_for_protocol(protocol)["signoff"],
            )
            _require_head_bound_sources(bound_paths)
            report = validate_health_report(
                block.health_output,
                protocol_path,
                repository_root=REPOSITORY_ROOT,
                require_current_tracked_clean=True,
            )
            context = validate_current_runtime_against_health_report(
                report,
                protocol=protocol,
                reference_config=configs[0],
                block=block,
                device=args.device,
            )
            if (
                sha256_file(block.health_output) != health_sha256_before
                or sha256_file(block.attempt_receipt) != receipt_sha256_before
            ):
                raise PilotV7Error(
                    "V7 compatibility precheck changed immutable health evidence"
                )
            for label, path in pilot_artifacts:
                if path.exists():
                    raise PilotV7Error(
                        f"{label} appeared during the zero-seed compatibility precheck: {path}"
                    )
            recovery = context.get("health_compatibility_recovery")
            if not isinstance(recovery, Mapping):
                raise PilotV7Error(
                    "V7 compatibility precheck did not return the sealed recovery binding"
                )
            print(
                "V7_PILOT_COMPATIBILITY_PRECHECK_PASS "
                f"dataset={block.dataset} recovery_commit={recovery['recovery_commit']}"
            )
            print(f"HEALTH_SHA256_UNCHANGED={health_sha256_before}")
            print(f"ATTEMPT_SHA256_UNCHANGED={receipt_sha256_before}")
            print("NO_HEALTH_OR_PILOT_SEED_WAS_CLAIMED")
            return 0
        if block.health_output.exists():
            raise PilotV7Error("Canonical V7 health output exists; no retry is allowed")
        if block.attempt_receipt.exists():
            raise PilotV7Error("The V7 health seed is already consumed; no retry is allowed")
        for label, path in pilot_artifacts:
            if path.exists():
                raise PilotV7Error(f"{label} exists before the one-shot health gate: {path}")
        git_identity = repository_git_identity(REPOSITORY_ROOT)
        if git_identity.get("tracked_clean") is not True:
            raise PilotV7Error("V7 health evidence requires a clean tracked worktree")
        bound_paths = (
            *RUNTIME_SOURCE_PATHS,
            protocol_path,
            *config_paths,
            REPOSITORY_ROOT / artifact_paths_for_protocol(protocol)["signoff"],
        )
        _require_head_bound_sources(bound_paths)
        runtime_sources = _runtime_source_hashes()
        source_binding = health_source_binding(
            block=block,
            protocol_path=protocol_path,
            config_paths=config_paths,
            configs=configs,
            repository_root=REPOSITORY_ROOT,
            runtime_sources=runtime_sources,
        )
        preclaim = _preclaim_gate(
            protocol=protocol,
            configs=configs,
            block=block,
            device_name=args.device,
            git_identity=git_identity,
            runtime_sources=runtime_sources,
            source_binding=source_binding,
        )
        print(
            "V7_HEALTH_PRECLAIM_PASS "
            f"dataset={block.dataset} device={preclaim['device']} "
            f"fixed_batch_sha256={preclaim['fixed_batch_sha256']}"
        )
        if args.precheck_only:
            if block.attempt_receipt.exists() or block.health_output.exists():
                raise PilotV7Error(
                    "V7 precheck-only must not create or observe health-seed artifacts"
                )
            print("V7_HEALTH_PRECHECK_ONLY_PASS")
            print("NO_HEALTH_SEED_WAS_CLAIMED")
            return 0
        receipt = attempt_receipt_payload(
            block=block,
            protocol_path=protocol_path,
            config_paths=config_paths,
            configs=configs,
            git_identity=git_identity,
            repository_root=REPOSITORY_ROOT,
            runtime_sources=runtime_sources,
        )
        attempt_receipt_sha256 = json_file_payload_sha256(receipt)
    except Exception as exc:  # noqa: BLE001 - normalize all pre-claim failures.
        print(f"V7_HEALTH_GATE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        exclusive_create_json(block.attempt_receipt, receipt)
    except Exception as exc:  # noqa: BLE001 - no receipt means no claimed seed.
        print(f"V7_HEALTH_GATE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        print(f"V7_AUTHOR_FREEZE_PASS signoff_sha256={author['signoff_sha256']}")
        print(
            "V7_HEALTH_ATTEMPT_CLAIMED "
            f"dataset={block.dataset} health_seed={block.health_seed} "
            f"pilot_seed={block.pilot_seed}"
        )
        print("HEALTH_SEED_EXECUTION_BEGINS")
        report = run_gate(
            protocol=protocol,
            protocol_path=protocol_path,
            config_paths=config_paths,
            configs=configs,
            block=block,
            device_name=args.device,
            git_identity=git_identity,
            runtime_sources=runtime_sources,
            source_binding=source_binding,
            attempt_receipt_sha256=attempt_receipt_sha256,
            expected_fixed_batch_sha256=str(preclaim["fixed_batch_sha256"]),
        )
    except BaseException as exc:  # noqa: BLE001 - preserve consumed-seed evidence.
        report = _fatal_report(
            block=block,
            git_identity=git_identity,
            runtime_sources=runtime_sources,
            source_binding=source_binding,
            attempt_receipt_sha256=attempt_receipt_sha256,
            preclaim=preclaim,
            exc=exc,
        )
    report["preclaim"] = _strict_json_safe(preclaim)
    if block.health_output.exists():
        print("V7_HEALTH_GATE_BLOCKED: output appeared during execution", file=sys.stderr)
        return 2
    try:
        exclusive_create_json(block.health_output, _strict_json_safe(report))
    except FileExistsError:
        print("V7_HEALTH_GATE_BLOCKED: output appeared during execution", file=sys.stderr)
        return 2
    print(f"V7_HEALTH_GATE_{report['status']} dataset={block.dataset}")
    print(f"NON_REPORTING_OUTPUT={block.health_output}")
    print(f"HEALTH_SEED_{block.health_seed}_CONSUMED_DO_NOT_RETRY")
    print(f"PILOT_SEED_{block.pilot_seed}_REMAINS_UNCONSUMED")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
