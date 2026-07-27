#!/usr/bin/env python3
"""Run the non-reporting implementation health gate before a v2 pilot.

This command is deliberately separate from the formal matrix runner.  Its
output is implementation-diagnostic evidence only and must never enter a
confirmatory analysis or ``results/runs``.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import (  # noqa: E402
    RunConfig,
    generate_run_matrix,
    load_protocol,
    load_run_config,
    validate_run_mapping,
)
from talif_msresnet.data import build_loaders  # noqa: E402
from talif_msresnet.models import CONDITIONS, build_model  # noqa: E402
from talif_msresnet.neurons import TALIFNeuron  # noqa: E402
from talif_msresnet.train import (  # noqa: E402
    _checkpoint_payload,
    _forward,
    _load_resume,
    _set_ta_enabled,
    build_optimizer_and_scheduler,
    train_one_epoch,
)
from talif_msresnet.utils import (  # noqa: E402
    atomic_write_json,
    capture_rng_state,
    environment_manifest,
    save_checkpoint,
    seed_everything,
    sha256_file,
    stable_hash,
    utc_now,
)


ARTIFACT_CLASS = "NON_REPORTING_IMPLEMENTATION_HEALTH_GATE"
CONDITIONS_IN_ORDER = ("C1", "C2", "C3", "C4")
MAX_PRECHECK_GPU_UTILIZATION_PERCENT = 10
GPU_IDLE_SETTLE_SECONDS = 1.0


class HealthGateError(RuntimeError):
    """A fail-closed pilot health-gate violation."""


class _NullLogger:
    def log(self, event: str, **values: Any) -> None:
        del event, values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs"
        / "v2_pilot_generated"
        / "E1_cifar100_d20_t6_C1_s77.yaml",
        help="Resolved CIFAR-100 reference config used only to build the health check.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "protocol_v2_pilot.yaml",
        help="Protocol used to validate and resolve --config.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON path outside the formal results root (no overwrite).",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--expected-gpu-substring",
        default="RTX 5090",
        help="Case-insensitive hardware binding; use an empty value only for a documented rerun.",
    )
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--overfit-batch-size", type=int, default=8)
    parser.add_argument(
        "--timing-batch-size",
        type=int,
        help="Defaults to the reference config's formal batch size.",
    )
    parser.add_argument("--overfit-steps", type=int, default=40)
    parser.add_argument("--min-overfit-accuracy", type=float, default=0.50)
    parser.add_argument("--max-overfit-loss-fraction", type=float, default=0.90)
    parser.add_argument(
        "--min-ta-routed-gradient-coverage",
        type=float,
        default=0.90,
        help="Minimum aggregate routed-slot coverage across TA parameter tensors.",
    )
    parser.add_argument("--timing-warmup", type=int, default=2)
    parser.add_argument("--timing-iterations", type=int, default=5)
    parser.add_argument("--max-ta-step-ratio", type=float, default=3.0)
    return parser


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def validate_output_path(
    output: Path,
    *,
    formal_roots: Iterable[Path],
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    """Reject overwrite and any destination inside a formal result tree."""

    resolved = output.resolve()
    roots = {repository_root.resolve() / "results" / "runs"}
    roots.update(Path(root).resolve() for root in formal_roots)
    for root in roots:
        if _is_within(resolved, root):
            raise HealthGateError(
                f"Non-reporting health output cannot be written inside formal root: {root}"
            )
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite existing health-gate output: {resolved}")
    return resolved


def validate_gate_arguments(args: argparse.Namespace) -> None:
    positive_ints = {
        "--overfit-batch-size": args.overfit_batch_size,
        "--overfit-steps": args.overfit_steps,
        "--timing-iterations": args.timing_iterations,
    }
    if args.timing_batch_size is not None:
        positive_ints["--timing-batch-size"] = args.timing_batch_size
    for name, value in positive_ints.items():
        if int(value) < 1:
            raise HealthGateError(f"{name} must be >= 1")
    if args.overfit_steps < 2:
        raise HealthGateError("--overfit-steps must be >= 2 for checkpoint continuation")
    if args.timing_warmup < 0:
        raise HealthGateError("--timing-warmup must be >= 0")
    bounded = {
        "--min-overfit-accuracy": args.min_overfit_accuracy,
        "--min-ta-routed-gradient-coverage": args.min_ta_routed_gradient_coverage,
    }
    for name, value in bounded.items():
        if not 0.0 <= float(value) <= 1.0:
            raise HealthGateError(f"{name} must be in [0, 1]")
    if not 0.0 < float(args.max_overfit_loss_fraction) < 1.0:
        raise HealthGateError("--max-overfit-loss-fraction must be in (0, 1)")
    if not math.isfinite(float(args.max_ta_step_ratio)) or args.max_ta_step_ratio <= 0:
        raise HealthGateError("--max-ta-step-ratio must be finite and positive")


def validate_protocol_health_binding(
    args: argparse.Namespace,
    config: RunConfig,
    protocol: Mapping[str, Any],
    output: Path,
) -> None:
    """Prevent a CLI override from weakening the predeclared pilot gate."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise HealthGateError("Protocol has no pilot_acceptance health-gate binding")
    overfit = acceptance.get("overfit")
    timing = acceptance.get("timing")
    environment = acceptance.get("environment")
    if not all(isinstance(item, Mapping) for item in (overfit, timing, environment)):
        raise HealthGateError(
            "Protocol pilot_acceptance is missing environment/overfit/timing settings"
        )
    expected_runs = [
        run
        for run in generate_run_matrix(protocol)
        if run.get("condition") == "C1"
        and run.get("seed") == acceptance.get("seed")
        and run.get("data", {}).get("dataset") == acceptance.get("dataset")
    ]
    if len(expected_runs) != 1:
        raise HealthGateError("Protocol does not generate exactly one reference C1 pilot cell")
    expected_config = validate_run_mapping(expected_runs[0], protocol)
    if config.as_dict() != expected_config.as_dict():
        raise HealthGateError("Reference config differs from the protocol-generated C1 pilot cell")
    timing_batch_size = (
        int(args.timing_batch_size)
        if args.timing_batch_size is not None
        else int(config.optimizer.batch_size)
    )
    observed = {
        "artifact_class": "non_reportable_pilot",
        "seed": int(args.seed),
        "dataset": config.data.dataset,
        "conditions": list(CONDITIONS_IN_ORDER),
        "overfit": {
            "batch_size": int(args.overfit_batch_size),
            "steps": int(args.overfit_steps),
            "minimum_accuracy": float(args.min_overfit_accuracy),
            "maximum_loss_fraction": float(args.max_overfit_loss_fraction),
            "minimum_ta_routed_gradient_coverage": float(args.min_ta_routed_gradient_coverage),
        },
        "timing": {
            "batch_size": timing_batch_size,
            "warmup_steps": int(args.timing_warmup),
            "timed_steps": int(args.timing_iterations),
            "maximum_ta_enabled_over_frozen_ratio": float(args.max_ta_step_ratio),
            "epoch_time_ratio_statistic": timing.get("epoch_time_ratio_statistic"),
            "epoch_time_ratio_conditions": list(timing.get("epoch_time_ratio_conditions", [])),
            "ta_state_source": timing.get("ta_state_source"),
        },
    }
    expected = {
        "artifact_class": acceptance.get("artifact_class"),
        "seed": acceptance.get("seed"),
        "dataset": acceptance.get("dataset"),
        "conditions": list(acceptance.get("conditions", [])),
        "overfit": dict(overfit),
        "timing": dict(timing),
    }
    if observed != expected:
        raise HealthGateError(
            "CLI/config health settings differ from protocol pilot_acceptance: "
            f"observed={observed} expected={expected}"
        )
    if args.expected_gpu_substring != environment.get("expected_gpu_substring"):
        raise HealthGateError("--expected-gpu-substring differs from pilot_acceptance")
    if environment.get("precision") != "float32" or config.runtime.amp:
        raise HealthGateError("Pilot health gate is bound to float32 with amp=false")
    if environment.get("deterministic") is not True or not config.runtime.deterministic:
        raise HealthGateError("Pilot health gate is bound to strict deterministic execution")
    protocol_hash = stable_hash(protocol)
    if config.analysis.get("protocol_hash") != protocol_hash:
        raise HealthGateError("Reference config protocol_hash differs from the pilot protocol")
    if config.runtime.seed != args.seed or config.model.condition != "C1":
        raise HealthGateError(
            "Reference config must be the C1 pilot matrix cell for "
            f"pilot_acceptance seed {acceptance.get('seed')}"
        )
    expected_output = _absolute_repository_path(str(acceptance.get("health_output", "")))
    if output.resolve() != expected_output:
        raise HealthGateError(
            f"--output must match protocol pilot_acceptance.health_output: {expected_output}"
        )


def require_target_cuda(device: torch.device, expected_gpu_substring: str) -> dict[str, Any]:
    """Bind the performance gate to one real CUDA device."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise HealthGateError("The pilot health gate requires a real CUDA device")
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    torch.cuda.set_device(index)
    properties = torch.cuda.get_device_properties(index)
    expected = expected_gpu_substring.strip()
    if expected and expected.lower() not in properties.name.lower():
        raise HealthGateError(
            f"GPU mismatch: expected name containing {expected!r}, got {properties.name!r}"
        )
    if not torch.are_deterministic_algorithms_enabled():
        raise HealthGateError("Strict PyTorch deterministic algorithms are not enabled")
    if torch.is_deterministic_algorithms_warn_only_enabled():
        raise HealthGateError("Deterministic algorithms are in warn-only mode")
    if torch.backends.cudnn.benchmark or not torch.backends.cudnn.deterministic:
        raise HealthGateError("cuDNN deterministic settings are not active")
    return {
        "device": f"cuda:{index}",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
    }


def validate_runtime_environment(
    binding: Mapping[str, Any],
    hardware: Mapping[str, Any],
) -> None:
    expected_gpu = str(binding.get("expected_gpu_substring", ""))
    if not expected_gpu or expected_gpu.lower() not in str(hardware.get("name", "")).lower():
        raise HealthGateError("Runtime GPU does not satisfy pilot_acceptance.environment")
    if torch.__version__ != binding.get("pytorch_version"):
        raise HealthGateError(
            f"PyTorch mismatch: {torch.__version__!r} != {binding.get('pytorch_version')!r}"
        )
    if str(torch.version.cuda) != str(binding.get("cuda_runtime")):
        raise HealthGateError(
            f"CUDA runtime mismatch: {torch.version.cuda!r} != {binding.get('cuda_runtime')!r}"
        )
    if binding.get("precision") != "float32":
        raise HealthGateError("Pilot acceptance precision must be float32")
    if binding.get("deterministic") is not True:
        raise HealthGateError("Pilot acceptance must require deterministic execution")


def gpu_idle_precheck(
    device: torch.device,
    *,
    samples: int = 3,
    maximum_utilization_percent: int = MAX_PRECHECK_GPU_UTILIZATION_PERCENT,
) -> dict[str, Any]:
    """Reject a busy target before timing; AB/BA ordering handles residual drift."""

    if samples < 1 or not 0 <= maximum_utilization_percent <= 100:
        raise ValueError("Invalid GPU idle-precheck settings")
    visible_index = torch.cuda.current_device() if device.index is None else int(device.index)
    try:
        raw_device_uuid = str(torch.cuda.get_device_properties(visible_index).uuid).strip()
    except Exception as exc:
        raise HealthGateError(f"Cannot identify target CUDA device UUID: {exc}") from exc
    if not raw_device_uuid:
        raise HealthGateError("Target CUDA device has no UUID for nvidia-smi binding")
    device_uuid = (
        raw_device_uuid
        if raw_device_uuid.upper().startswith(("GPU-", "MIG-"))
        else f"GPU-{raw_device_uuid}"
    )
    observed: list[int] = []
    try:
        torch.cuda.synchronize(device)
        time.sleep(GPU_IDLE_SETTLE_SECONDS)
        for index in range(samples):
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device_uuid}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise HealthGateError(f"nvidia-smi utilization query failed: {detail}")
            rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
            if len(rows) != 1:
                raise HealthGateError(
                    f"nvidia-smi returned {len(rows)} utilization rows for one device"
                )
            utilization = int(rows[0])
            if not 0 <= utilization <= 100:
                raise HealthGateError(
                    f"nvidia-smi returned invalid GPU utilization: {utilization}"
                )
            observed.append(utilization)
            if index + 1 < samples:
                time.sleep(0.25)
    except HealthGateError:
        raise
    except Exception as exc:
        raise HealthGateError(f"Cannot verify target GPU utilization: {exc}") from exc
    passed = max(observed) <= maximum_utilization_percent
    report = {
        "pass": passed,
        "source": "nvidia-smi",
        "device_selector": device_uuid,
        "device_uuid": device_uuid,
        "cuda_visible_index": visible_index,
        "measurement_point": "immediately_before_cuda_timing",
        "settle_seconds": GPU_IDLE_SETTLE_SECONDS,
        "samples_percent": observed,
        "maximum_observed_percent": max(observed),
        "maximum_allowed_percent": maximum_utilization_percent,
    }
    if not passed:
        raise HealthGateError(f"Target GPU is not idle enough for timing: {report}")
    return report


def _absolute_repository_path(
    value: str | Path,
    repository_root: Path = REPOSITORY_ROOT,
) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def repository_git_identity(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Return the current commit and tracked-only cleanliness, or fail closed."""

    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise HealthGateError(f"Git identity command failed: {detail}")
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise HealthGateError(f"Git HEAD is not a full lowercase commit hash: {commit!r}")
    tracked_status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "tracked_clean": not bool(tracked_status)}


def validate_pilot_health_report(
    report_path: str | Path,
    protocol_path: str | Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    expected_git_commit: str | None = None,
    require_current_tracked_clean: bool = True,
) -> dict[str, Any]:
    """Read and validate the canonical PASS report without mutating repository state."""

    protocol_file = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_file)
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise HealthGateError("Protocol has no pilot_acceptance health-gate binding")
    expected_path = _absolute_repository_path(
        str(acceptance.get("health_output", "")), repository_root
    )
    path = Path(report_path).resolve()
    if path != expected_path:
        raise HealthGateError(
            f"Health report path differs from protocol: {path} != {expected_path}"
        )
    if not path.is_file():
        raise HealthGateError(f"Canonical pilot health report is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HealthGateError(f"Cannot read pilot health report: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise HealthGateError("Pilot health report must be a JSON object")
    report = dict(raw)

    current_identity: Mapping[str, Any] | None = None
    if expected_git_commit is None or require_current_tracked_clean:
        current_identity = repository_git_identity(repository_root)
    if expected_git_commit is None:
        assert current_identity is not None
        expected_git_commit = str(current_identity["git_commit"])
    failures: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    def lower_sha256(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    require(report.get("schema_version") == 2, "schema_version must be 2")
    require(report.get("artifact_class") == ARTIFACT_CLASS, "artifact_class is invalid")
    require(report.get("status") == "PASS", "status must be PASS")
    require(report.get("pass") is True, "pass must be true")
    require(
        report.get("reporting_eligibility") == "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "reporting eligibility marker is invalid",
    )
    require(
        report.get("confirmatory_analysis_eligibility") is False,
        "confirmatory_analysis_eligibility must be false",
    )
    require(report.get("seed") == acceptance.get("seed"), "pilot seed mismatch")
    require(
        report.get("conditions") == list(acceptance.get("conditions", [])),
        "top-level condition order mismatch",
    )
    require(report.get("protocol_hash") == stable_hash(protocol), "protocol_hash mismatch")
    require(
        report.get("acceptance_hash") == stable_hash(dict(acceptance)),
        "acceptance_hash mismatch",
    )
    report_commit = report.get("git_commit")
    valid_report_commit = isinstance(report_commit, str) and len(report_commit) == 40
    valid_report_commit = bool(
        valid_report_commit and all(character in "0123456789abcdef" for character in report_commit)
    )
    require(valid_report_commit, "git_commit must be a full lowercase hash")
    require(report_commit == expected_git_commit, "git_commit mismatch")
    require(report.get("tracked_clean") is True, "report was not produced from a clean tree")
    if require_current_tracked_clean:
        assert current_identity is not None
        require(
            current_identity.get("tracked_clean") is True,
            "current tracked worktree is not clean",
        )

    overfit = acceptance.get("overfit")
    timing_acceptance = acceptance.get("timing")
    require(isinstance(overfit, Mapping), "protocol overfit acceptance is missing")
    require(isinstance(timing_acceptance, Mapping), "protocol timing acceptance is missing")
    if isinstance(overfit, Mapping) and isinstance(timing_acceptance, Mapping):
        expected_thresholds = {
            "overfit_steps": overfit.get("steps"),
            "minimum_overfit_accuracy": overfit.get("minimum_accuracy"),
            "maximum_overfit_loss_fraction": overfit.get("maximum_loss_fraction"),
            "minimum_ta_routed_gradient_coverage": overfit.get(
                "minimum_ta_routed_gradient_coverage"
            ),
            "maximum_ta_enabled_over_frozen_step_ratio": timing_acceptance.get(
                "maximum_ta_enabled_over_frozen_ratio"
            ),
        }
        require(report.get("thresholds") == expected_thresholds, "threshold binding mismatch")

    source = report.get("source")
    require(isinstance(source, Mapping), "source metadata is missing")
    if isinstance(source, Mapping):
        require(source.get("dataset") == acceptance.get("dataset"), "dataset mismatch")
        require(
            source.get("overfit_batch_size")
            == (overfit.get("batch_size") if isinstance(overfit, Mapping) else None),
            "overfit batch-size mismatch",
        )
        require(
            source.get("timing_batch_size")
            == (
                timing_acceptance.get("batch_size")
                if isinstance(timing_acceptance, Mapping)
                else None
            ),
            "timing batch-size mismatch",
        )
        require(
            source.get("protocol_sha256") == sha256_file(protocol_file),
            "protocol file SHA-256 mismatch",
        )
        reported_protocol = Path(str(source.get("protocol", ""))).resolve()
        require(reported_protocol == protocol_file, "reported protocol path mismatch")
        config_file = Path(str(source.get("config", ""))).resolve()
        require(
            _is_within(config_file, repository_root),
            "reported reference config is outside the repository",
        )
        require(config_file.is_file(), "reported reference config is missing")
        if config_file.is_file():
            require(
                source.get("config_sha256") == sha256_file(config_file),
                "reference config SHA-256 mismatch",
            )
            try:
                reported_config = load_run_config(config_file, protocol_file)
                expected_cells = [
                    run
                    for run in generate_run_matrix(protocol)
                    if run.get("condition") == "C1"
                    and run.get("seed") == acceptance.get("seed")
                    and run.get("data", {}).get("dataset") == acceptance.get("dataset")
                ]
                expected_config = (
                    validate_run_mapping(expected_cells[0], protocol)
                    if len(expected_cells) == 1
                    else None
                )
                require(
                    expected_config is not None
                    and reported_config.as_dict() == expected_config.as_dict(),
                    "reported reference config differs from the generated pilot cell",
                )
            except Exception as exc:
                failures.append(f"cannot validate reported reference config: {exc}")
        require(
            lower_sha256(source.get("split_manifest_sha256")),
            "split-manifest SHA-256 is invalid",
        )
        require(
            lower_sha256(source.get("fixed_batch_sha256")),
            "fixed-batch SHA-256 is invalid",
        )
        shape = source.get("fixed_batch_shape")
        require(
            isinstance(shape, list)
            and len(shape) == 4
            and all(isinstance(value, int) and value > 0 for value in shape)
            and shape[0] >= int(timing_acceptance.get("batch_size", 0)),
            "fixed real CIFAR-100 batch shape is invalid",
        )

    environment_binding = acceptance.get("environment")
    environment = report.get("environment")
    require(isinstance(environment_binding, Mapping), "environment binding is missing")
    require(isinstance(environment, Mapping), "environment evidence is missing")
    if isinstance(environment_binding, Mapping) and isinstance(environment, Mapping):
        hardware = environment.get("hardware")
        expected_gpu = str(environment_binding.get("expected_gpu_substring", ""))
        require(isinstance(hardware, Mapping), "hardware evidence is missing")
        if isinstance(hardware, Mapping):
            observed_gpu = str(hardware.get("name", ""))
            require(
                bool(expected_gpu and expected_gpu.lower() in observed_gpu.lower()),
                "GPU hardware binding mismatch",
            )
        require(
            environment.get("pytorch") == environment_binding.get("pytorch_version"),
            "PyTorch version mismatch",
        )
        require(
            str(environment.get("cuda_version")) == str(environment_binding.get("cuda_runtime")),
            "CUDA runtime mismatch",
        )
        require(
            environment.get("precision") == environment_binding.get("precision"),
            "precision mismatch",
        )
        require(environment.get("amp") is False, "amp must be false")
        require(environment.get("torch_deterministic") is True, "determinism is disabled")
        require(
            environment.get("torch_deterministic_warn_only") is False,
            "determinism is warn-only",
        )
        require(environment.get("cudnn_benchmark") is False, "cuDNN benchmark is enabled")
        require(
            environment.get("cudnn_deterministic") is True,
            "cuDNN deterministic mode is disabled",
        )
        idle = environment.get("gpu_idle_precheck")
        require(isinstance(idle, Mapping), "GPU idle precheck is missing")
        if isinstance(idle, Mapping):
            require(idle.get("pass") is True, "GPU idle precheck did not pass")
            require(idle.get("source") == "nvidia-smi", "GPU idle source is invalid")
            require(
                isinstance(idle.get("device_selector"), str)
                and bool(idle["device_selector"]),
                "GPU idle device selector is invalid",
            )
            require(
                isinstance(idle.get("device_uuid"), str)
                and idle.get("device_uuid") == idle.get("device_selector"),
                "GPU idle UUID binding is invalid",
            )
            require(
                isinstance(idle.get("cuda_visible_index"), int)
                and idle["cuda_visible_index"] >= 0,
                "GPU idle CUDA visible index is invalid",
            )
            require(
                idle.get("measurement_point") == "immediately_before_cuda_timing",
                "GPU idle measurement point is invalid",
            )
            idle_samples = idle.get("samples_percent")
            valid_idle_samples = bool(
                isinstance(idle_samples, list)
                and len(idle_samples) >= 3
                and all(
                    isinstance(value, int) and 0 <= value <= 100
                    for value in idle_samples
                )
            )
            require(
                valid_idle_samples,
                "GPU idle precheck samples are invalid",
            )
            require(
                idle.get("maximum_allowed_percent") == MAX_PRECHECK_GPU_UTILIZATION_PERCENT,
                "GPU idle precheck threshold mismatch",
            )
            if valid_idle_samples:
                observed_max = max(idle_samples)
                require(
                    idle.get("maximum_observed_percent") == observed_max,
                    "GPU idle maximum does not match samples",
                )
                require(
                    observed_max <= MAX_PRECHECK_GPU_UTILIZATION_PERCENT,
                    "GPU idle samples exceed the permitted threshold",
                )

    def finite_number(value: Any, *, positive: bool = False) -> bool:
        valid = (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
        return bool(valid and (not positive or float(value) > 0))

    conditions = report.get("functional_health_by_condition")
    require(isinstance(conditions, Mapping), "per-condition health results are missing")
    if isinstance(conditions, Mapping) and isinstance(overfit, Mapping):
        require(set(conditions) == set(CONDITIONS_IN_ORDER), "condition set mismatch")
        for condition in CONDITIONS_IN_ORDER:
            item = conditions.get(condition)
            require(
                isinstance(item, Mapping) and item.get("pass") is True,
                f"{condition} functional health did not pass",
            )
            if not isinstance(item, Mapping):
                continue
            overfit_result = item.get("overfit")
            require(isinstance(overfit_result, Mapping), f"{condition} overfit result missing")
            if isinstance(overfit_result, Mapping):
                loss_fraction = overfit_result.get("loss_fraction")
                final_accuracy = overfit_result.get("final_accuracy")
                history = overfit_result.get("train_loss_history")
                require(overfit_result.get("pass") is True, f"{condition} overfit did not pass")
                require(
                    overfit_result.get("steps") == overfit.get("steps"),
                    f"{condition} overfit step count mismatch",
                )
                require(
                    finite_number(loss_fraction)
                    and float(loss_fraction) <= float(overfit["maximum_loss_fraction"]),
                    f"{condition} overfit loss threshold failed",
                )
                require(
                    finite_number(final_accuracy)
                    and float(final_accuracy) >= float(overfit["minimum_accuracy"]),
                    f"{condition} overfit accuracy threshold failed",
                )
                require(
                    isinstance(history, list)
                    and len(history) == int(overfit["steps"])
                    and all(finite_number(value) for value in history),
                    f"{condition} overfit loss history is incomplete",
                )
            shared = item.get("shared_conv_and_fc_gradients")
            require(isinstance(shared, Mapping), f"{condition} shared gradients missing")
            if isinstance(shared, Mapping):
                required_count = shared.get("required_parameter_count")
                passed_count = shared.get("passed_parameter_count")
                parameter_rows = shared.get("parameters")
                require(shared.get("pass") is True, f"{condition} shared gradients failed")
                require(
                    isinstance(required_count, int)
                    and required_count > 0
                    and passed_count == required_count,
                    f"{condition} shared gradient count mismatch",
                )
                require(
                    isinstance(parameter_rows, Mapping)
                    and len(parameter_rows) == required_count
                    and all(
                        isinstance(row, Mapping)
                        and row.get("pass") is True
                        and row.get("finite") is True
                        and isinstance(row.get("nonzero_count"), int)
                        and row["nonzero_count"] > 0
                        for row in parameter_rows.values()
                    ),
                    f"{condition} shared gradient parameter evidence is incomplete",
                )
            ta_result = item.get("ta_gradients")
            require(isinstance(ta_result, Mapping), f"{condition} TA gradient result missing")
            if isinstance(ta_result, Mapping):
                require(ta_result.get("pass") is True, f"{condition} TA gradient gate failed")
                if condition in {"C2", "C4"}:
                    coverage = ta_result.get("aggregate_routed_gradient_coverage")
                    require(
                        finite_number(coverage)
                        and float(coverage)
                        >= float(overfit["minimum_ta_routed_gradient_coverage"]),
                        f"{condition} TA routed-gradient coverage failed",
                    )
                    require(
                        isinstance(ta_result.get("covered_parameter_slots"), int)
                        and ta_result["covered_parameter_slots"] > 0,
                        f"{condition} has no covered TA parameter slot",
                    )
                    require(
                        ta_result.get("every_ta_parameter_tensor_has_nonzero_routed_coverage")
                        is True,
                        f"{condition} has a TA parameter tensor without coverage",
                    )
                    require(
                        ta_result.get("all_ta_gradients_present_and_finite_on_every_probe") is True,
                        f"{condition} has missing/non-finite TA gradients",
                    )
                    require(
                        ta_result.get("no_nonzero_gradient_in_unrouted_slots") is True,
                        f"{condition} has gradient in an unrouted TA slot",
                    )
                else:
                    require(
                        ta_result.get("status") == "not_applicable_lif_condition",
                        f"{condition} TA status is invalid for LIF",
                    )
            checkpoint = item.get("checkpoint_next_step")
            require(isinstance(checkpoint, Mapping), f"{condition} checkpoint result missing")
            if isinstance(checkpoint, Mapping):
                require(checkpoint.get("pass") is True, f"{condition} checkpoint gate failed")
                require(
                    checkpoint.get("resume_loader") == "talif_msresnet.train._load_resume",
                    f"{condition} did not use the real resume loader",
                )
                require(
                    checkpoint.get("checkpoint_name") == "last.pt",
                    f"{condition} checkpoint is not last.pt",
                )
                require(
                    checkpoint.get("resume_epoch_match") is True
                    and checkpoint.get("scheduler_stepped_before_last_checkpoint") is True
                    and checkpoint.get("mismatch_count") == 0,
                    f"{condition} checkpoint continuation evidence is invalid",
                )
                checkpoint_epoch = checkpoint.get("checkpoint_epoch_zero_based")
                expected_start = checkpoint.get("expected_start_epoch_zero_based")
                activation_epoch = checkpoint.get("ta_activation_epoch_zero_based")
                history_rows = checkpoint.get("precheckpoint_epoch_count")
                require(
                    isinstance(checkpoint_epoch, int)
                    and isinstance(expected_start, int)
                    and expected_start == checkpoint_epoch + 1
                    and isinstance(history_rows, int)
                    and history_rows == checkpoint_epoch + 1,
                    f"{condition} checkpoint epoch history is invalid",
                )
                if condition in {"C2", "C4"}:
                    require(
                        isinstance(activation_epoch, int)
                        and checkpoint_epoch == activation_epoch - 1
                        and expected_start == activation_epoch
                        and checkpoint.get("ta_enabled_at_checkpoint") is False
                        and checkpoint.get("ta_enabled_on_next_step") is True
                        and checkpoint.get("ta_activation_boundary_crossed_on_resume") is True,
                        f"{condition} did not verify the frozen-to-enabled TA resume boundary",
                    )

    timing = report.get("cuda_train_step_timing")
    require(isinstance(timing, Mapping), "CUDA timing results are missing")
    if isinstance(timing, Mapping):
        require(timing.get("pass") is True, "CUDA timing gate did not pass")
        timing_conditions = timing.get("conditions")
        require(isinstance(timing_conditions, Mapping), "per-condition timing is missing")
        if isinstance(timing_conditions, Mapping):
            require(set(timing_conditions) == set(CONDITIONS_IN_ORDER), "timing set mismatch")
            for condition in ("C1", "C3"):
                item = timing_conditions.get(condition)
                standard = item.get("lif_standard") if isinstance(item, Mapping) else None
                require(
                    isinstance(standard, Mapping)
                    and finite_number(standard.get("cuda_step_median_ms"), positive=True)
                    and finite_number(standard.get("peak_allocated_bytes"), positive=True),
                    f"{condition} LIF timing evidence is invalid",
                )
            for condition in ("C2", "C4"):
                item = timing_conditions.get(condition)
                rounds = item.get("rounds") if isinstance(item, Mapping) else None
                aggregate = item.get("aggregate") if isinstance(item, Mapping) else None
                require(
                    isinstance(rounds, list)
                    and len(rounds) == 2
                    and [row.get("order") for row in rounds if isinstance(row, Mapping)]
                    == [["ta_frozen", "ta_enabled"], ["ta_enabled", "ta_frozen"]],
                    f"{condition} timing is not reversed AB/BA",
                )
                require(
                    isinstance(aggregate, Mapping)
                    and finite_number(
                        aggregate.get("ta_frozen_peak_allocated_bytes"), positive=True
                    )
                    and finite_number(
                        aggregate.get("ta_enabled_peak_allocated_bytes"), positive=True
                    ),
                    f"{condition} peak-memory evidence is invalid",
                )
        ratios = timing.get("ta_enabled_over_frozen_ratios")
        require(isinstance(ratios, Mapping), "TA timing ratios are missing")
        if isinstance(ratios, Mapping) and isinstance(timing_acceptance, Mapping):
            require(set(ratios) == {"C2", "C4"}, "TA timing ratio condition set mismatch")
            limit = float(timing_acceptance["maximum_ta_enabled_over_frozen_ratio"])
            for condition in ("C2", "C4"):
                item = ratios.get(condition)
                value = item.get("ta_enabled_over_frozen") if isinstance(item, Mapping) else None
                round_values = item.get("round_ratios") if isinstance(item, Mapping) else None
                valid_rounds = bool(
                    isinstance(round_values, list)
                    and len(round_values) == 2
                    and all(
                        finite_number(round_value) and float(round_value) <= limit
                        for round_value in round_values
                    )
                )
                require(
                    bool(
                        isinstance(item, Mapping)
                        and item.get("pass") is True
                        and finite_number(value)
                        and float(value) <= limit
                        and valid_rounds
                    ),
                    f"{condition} TA timing ratio is invalid",
                )
    require(report.get("failures") == [], "health report contains failure records")
    if failures:
        raise HealthGateError("Pilot health report rejected:\n- " + "\n- ".join(failures))
    return report


def validate_current_runtime_against_health_report(
    report: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    reference_config: RunConfig,
    device: str,
) -> dict[str, Any]:
    """Reproduce the health report's stable runtime and data identity before launch."""

    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise HealthGateError("Protocol has no pilot_acceptance runtime binding")
    environment_binding = acceptance.get("environment")
    overfit = acceptance.get("overfit")
    timing = acceptance.get("timing")
    if not all(isinstance(item, Mapping) for item in (environment_binding, overfit, timing)):
        raise HealthGateError("Pilot runtime/data acceptance mappings are incomplete")
    if not device or device == "auto":
        raise HealthGateError("The v2 pilot launch requires an explicit CUDA device")

    seed = int(acceptance["seed"])
    seed_everything(seed, deterministic=True)
    selected_device = torch.device(device)
    hardware = require_target_cuda(
        selected_device, str(environment_binding["expected_gpu_substring"])
    )
    validate_runtime_environment(environment_binding, hardware)
    report_environment = report.get("environment")
    if not isinstance(report_environment, Mapping):
        raise HealthGateError("Health report has no environment evidence")
    if hardware != report_environment.get("hardware"):
        raise HealthGateError(
            "Current CUDA hardware differs from the health-gate hardware; rerun the health gate"
        )

    current_environment = {
        **environment_manifest(),
        "precision": "float32",
        "amp": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    stable_environment_keys = (
        "hostname",
        "platform",
        "python",
        "pytorch",
        "numpy",
        "cuda_available",
        "cuda_version",
        "cuda_device",
        "precision",
        "amp",
        "cublas_workspace_config",
        "torch_deterministic",
        "torch_deterministic_warn_only",
        "cudnn_benchmark",
        "cudnn_deterministic",
    )
    mismatches = [
        key
        for key in stable_environment_keys
        if current_environment.get(key) != report_environment.get(key)
    ]
    if mismatches:
        raise HealthGateError(
            "Current software/host environment differs from the health report at: "
            + ", ".join(mismatches)
        )

    idle = gpu_idle_precheck(selected_device)
    report_idle = report_environment.get("gpu_idle_precheck")
    if not isinstance(report_idle, Mapping) or (
        idle.get("device_uuid") != report_idle.get("device_uuid")
    ):
        raise HealthGateError(
            "Current CUDA device UUID differs from the health gate; rerun the health gate"
        )

    required_batch_size = max(int(overfit["batch_size"]), int(timing["batch_size"]))
    _resolved, inputs, targets, split_manifest = load_fixed_real_batch(
        reference_config,
        seed=seed,
        required_batch_size=required_batch_size,
    )
    source = report.get("source")
    if not isinstance(source, Mapping):
        raise HealthGateError("Health report has no fixed-data source evidence")
    split_hash = str(split_manifest.get("manifest_sha256", ""))
    batch_hash = tensor_batch_sha256(inputs, targets)
    if split_hash != source.get("split_manifest_sha256"):
        raise HealthGateError(
            "Current CIFAR-100 split manifest differs from the health gate"
        )
    if batch_hash != source.get("fixed_batch_sha256"):
        raise HealthGateError("Current fixed CIFAR-100 batch differs from the health gate")
    if list(inputs.shape) != source.get("fixed_batch_shape"):
        raise HealthGateError("Current fixed CIFAR-100 batch shape differs from the health gate")
    return {
        "pass": True,
        "device": hardware["device"],
        "device_uuid": idle["device_uuid"],
        "hardware": hardware,
        "stable_environment": {
            key: current_environment.get(key) for key in stable_environment_keys
        },
        "split_manifest_sha256": split_hash,
        "fixed_batch_sha256": batch_hash,
        "fixed_batch_shape": list(inputs.shape),
    }


def diagnostic_config(
    base: RunConfig,
    *,
    condition: str,
    seed: int,
    device: torch.device,
    output_parent: Path,
) -> RunConfig:
    topology, neuron = CONDITIONS[condition]
    model = dataclasses.replace(
        base.model,
        condition=condition,
        topology=topology,
        neuron=neuron,
    )
    runtime = dataclasses.replace(
        base.runtime,
        seed=int(seed),
        device=str(device),
        output_dir=str(output_parent),
        run_id=f"NONREPORTING_pilot_health_{condition}_s{seed}",
        amp=False,
        deterministic=True,
        dry_run=False,
        limit_batches=1,
        resume=None,
    )
    return dataclasses.replace(base, model=model, runtime=runtime, final_test=False)


def load_fixed_real_batch(
    base: RunConfig,
    *,
    seed: int,
    required_batch_size: int,
) -> tuple[RunConfig, torch.Tensor, torch.Tensor, Mapping[str, Any]]:
    """Load one transformed training batch once, then hold its tensors fixed."""

    if base.data.dataset != "cifar100":
        raise HealthGateError(f"Health gate requires CIFAR-100, got {base.data.dataset!r}")
    root = _absolute_repository_path(base.data.root)
    manifest_path = _absolute_repository_path(base.data.split_manifest)
    if not root.exists():
        raise HealthGateError(f"CIFAR-100 root is missing: {root}")
    if not manifest_path.is_file():
        raise HealthGateError(f"Existing CIFAR-100 split manifest is required: {manifest_path}")
    data = dataclasses.replace(
        base.data,
        root=str(root),
        split_manifest=str(manifest_path),
        download=False,
        num_workers=0,
    )
    optimizer = dataclasses.replace(base.optimizer, batch_size=int(required_batch_size))
    runtime = dataclasses.replace(base.runtime, seed=int(seed))
    resolved = dataclasses.replace(base, data=data, optimizer=optimizer, runtime=runtime)
    loaders = build_loaders(resolved, final_test=False, seed=seed)
    try:
        batch = next(iter(loaders["train"]))
    except StopIteration as exc:
        raise HealthGateError("CIFAR-100 training loader produced no batch") from exc
    inputs = torch.as_tensor(batch[0]).detach().cpu().contiguous()
    targets = torch.as_tensor(batch[1]).detach().cpu().long().contiguous()
    if inputs.shape[0] < required_batch_size or targets.shape[0] < required_batch_size:
        raise HealthGateError(
            f"Fixed batch has {inputs.shape[0]} samples; {required_batch_size} are required"
        )
    if not torch.isfinite(inputs).all().item():
        raise HealthGateError("Fixed CIFAR-100 batch contains non-finite input values")
    manifest = loaders.get("_manifest")
    if not isinstance(manifest, Mapping) or not manifest.get("manifest_sha256"):
        raise HealthGateError("CIFAR-100 loader did not return a bound split manifest")
    return resolved, inputs, targets, manifest


def tensor_batch_sha256(inputs: torch.Tensor, targets: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in (inputs, targets):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _grad_scaler() -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=False)


def _build_training_objects(
    config: RunConfig,
    device: torch.device,
    *,
    ta_enabled: bool,
) -> tuple[nn.Module, torch.optim.Optimizer, Any, list[nn.Parameter]]:
    model_cfg = dataclasses.asdict(config.model)
    model_cfg["init_seed"] = config.runtime.seed
    model = build_model(model_cfg).to(device)
    optimizer, scheduler, ta_parameters, _ta_names, _activation = build_optimizer_and_scheduler(
        model, config
    )
    _set_ta_enabled(ta_parameters, ta_enabled)
    return model, optimizer, scheduler, ta_parameters


def _one_step(
    model: nn.Module,
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
        _grad_scaler(),
    )


@torch.no_grad()
def _fixed_batch_metrics(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    inputs, targets = batch
    model.eval()
    logits, _diagnostics = _forward(model, inputs, collect_activity=False)
    loss = float(F.cross_entropy(logits, targets).item())
    accuracy = float((logits.argmax(dim=1) == targets).float().mean().item())
    if not math.isfinite(loss) or not math.isfinite(accuracy):
        raise HealthGateError("Fixed-batch evaluation produced non-finite metrics")
    return {"loss": loss, "accuracy": accuracy}


def _shared_gradient_parameter_names(model: nn.Module) -> list[str]:
    names: list[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            names.extend(
                f"{module_name}.{name}" if module_name else name
                for name, _parameter in module.named_parameters(recurse=False)
            )
    classifier = getattr(model, "fc", None)
    if not isinstance(classifier, nn.Module):
        raise HealthGateError("Model exposes no fc classifier for the gradient gate")
    names.extend(f"fc.{name}" for name, _parameter in classifier.named_parameters(recurse=True))
    return sorted(set(names))


def inspect_shared_gradients(model: nn.Module) -> dict[str, Any]:
    parameters = dict(model.named_parameters())
    rows: dict[str, Any] = {}
    for name in _shared_gradient_parameter_names(model):
        parameter = parameters.get(name)
        gradient = None if parameter is None else parameter.grad
        finite = gradient is not None and bool(torch.isfinite(gradient).all().item())
        nonzero_count = (
            int(torch.count_nonzero(gradient).item()) if gradient is not None and finite else 0
        )
        norm = (
            float(torch.linalg.vector_norm(gradient.detach().double()).item())
            if gradient is not None and finite
            else float("nan")
        )
        rows[name] = {
            "gradient_present": gradient is not None,
            "finite": finite,
            "nonzero_count": nonzero_count,
            "numel": int(parameter.numel()) if parameter is not None else 0,
            "l2_norm": norm,
            "pass": bool(finite and nonzero_count > 0 and math.isfinite(norm) and norm > 0),
        }
    passed = bool(rows) and all(bool(row["pass"]) for row in rows.values())
    return {
        "pass": passed,
        "required_parameter_count": len(rows),
        "passed_parameter_count": sum(bool(row["pass"]) for row in rows.values()),
        "parameters": rows,
    }


def _capture_ta_routes(model: nn.Module, route_union: MutableMapping[str, set[int]]) -> list[Any]:
    handles: list[Any] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, TALIFNeuron):
            continue
        route_union.setdefault(module_name, set())

        def hook(
            current: TALIFNeuron,
            _inputs: tuple[Any, ...],
            _output: Any,
            *,
            name: str = module_name,
        ) -> None:
            if current.last_c_pre is None:
                return
            route_union[name].update(
                int(value) for value in torch.unique(current.last_c_pre).detach().cpu().tolist()
            )

        handles.append(module.register_forward_hook(hook))
    return handles


def _new_ta_gradient_accumulator(model: nn.Module) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".center") or name.endswith(".raw_width"):
            result[name] = {
                "bank_size": int(parameter.numel()),
                "finite_on_every_probe": True,
                "gradient_present_on_every_probe": True,
                "nonzero_slots": set(),
                "probe_count": 0,
            }
    return result


def _update_ta_gradient_accumulator(
    model: nn.Module,
    accumulator: MutableMapping[str, dict[str, Any]],
) -> None:
    parameters = dict(model.named_parameters())
    for name, state in accumulator.items():
        gradient = parameters[name].grad
        state["probe_count"] += 1
        if gradient is None:
            state["gradient_present_on_every_probe"] = False
            state["finite_on_every_probe"] = False
            continue
        finite = bool(torch.isfinite(gradient).all().item())
        state["finite_on_every_probe"] = bool(state["finite_on_every_probe"] and finite)
        if finite:
            flat = gradient.detach().reshape(-1)
            state["nonzero_slots"].update(
                int(index)
                for index in torch.nonzero(flat != 0, as_tuple=False).flatten().cpu().tolist()
            )


def summarize_ta_gradients(
    route_union: Mapping[str, set[int]],
    accumulator: Mapping[str, Mapping[str, Any]],
    *,
    minimum_coverage: float,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    total_routed = 0
    total_covered = 0
    every_parameter_has_coverage = True
    all_finite = True
    all_present = True
    no_unexpected_nonzero = True
    for name, state in accumulator.items():
        module_name = name.rsplit(".", 1)[0]
        bank_size = int(state["bank_size"])
        routed = sorted(index for index in route_union.get(module_name, set()) if index < bank_size)
        nonzero = set(int(index) for index in state["nonzero_slots"])
        covered = sorted(set(routed) & nonzero)
        zero_routed = sorted(set(routed) - nonzero)
        unused = sorted(set(range(bank_size)) - set(routed))
        unexpected = sorted(nonzero - set(routed))
        coverage = len(covered) / len(routed) if routed else 0.0
        present = bool(state["gradient_present_on_every_probe"])
        finite = bool(state["finite_on_every_probe"])
        parameter_has_coverage = bool(routed and covered)
        every_parameter_has_coverage = every_parameter_has_coverage and parameter_has_coverage
        all_finite = all_finite and finite
        all_present = all_present and present
        no_unexpected_nonzero = no_unexpected_nonzero and not unexpected
        total_routed += len(routed)
        total_covered += len(covered)
        rows[name] = {
            "bank_size": bank_size,
            "routed_slots": routed,
            "nonzero_routed_gradient_slots": covered,
            "zero_gradient_routed_slots": zero_routed,
            "unrouted_slots_exempt": unused,
            "unexpected_nonzero_unrouted_slots": unexpected,
            "routed_gradient_coverage": coverage,
            "gradient_present_on_every_probe": present,
            "finite_on_every_probe": finite,
            "probe_count": int(state["probe_count"]),
        }
    aggregate = total_covered / total_routed if total_routed else 0.0
    passed = bool(
        rows
        and total_routed > 0
        and total_covered > 0
        and all_present
        and all_finite
        and every_parameter_has_coverage
        and no_unexpected_nonzero
        and aggregate >= minimum_coverage
    )
    return {
        "pass": passed,
        "unused_history_slot_policy": "unrouted bank slots are reported and exempt",
        "minimum_aggregate_routed_gradient_coverage": minimum_coverage,
        "aggregate_routed_gradient_coverage": aggregate,
        "routed_parameter_slots": total_routed,
        "covered_parameter_slots": total_covered,
        "every_ta_parameter_tensor_has_nonzero_routed_coverage": (every_parameter_has_coverage),
        "all_ta_gradients_present_and_finite_on_every_probe": all_present and all_finite,
        "no_nonzero_gradient_in_unrouted_slots": no_unexpected_nonzero,
        "parameters": rows,
    }


def _step_with_ta_tracking(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    step: int,
    route_union: MutableMapping[str, set[int]],
    accumulator: MutableMapping[str, dict[str, Any]],
) -> dict[str, Any]:
    handles = _capture_ta_routes(model, route_union)
    try:
        metrics = _one_step(model, optimizer, config, batch, device, step)
    finally:
        for handle in handles:
            handle.remove()
    if accumulator:
        _update_ta_gradient_accumulator(model, accumulator)
    return metrics


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value


def _comparison_mismatches(expected: Any, observed: Any, path: str = "root") -> list[str]:
    if isinstance(expected, torch.Tensor) and isinstance(observed, torch.Tensor):
        return [] if torch.equal(expected.cpu(), observed.cpu()) else [path]
    if isinstance(expected, np.ndarray) and isinstance(observed, np.ndarray):
        return [] if np.array_equal(expected, observed) else [path]
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            return [f"{path}.keys"]
        mismatches: list[str] = []
        for key in expected:
            mismatches.extend(_comparison_mismatches(expected[key], observed[key], f"{path}.{key}"))
        return mismatches
    if isinstance(expected, (tuple, list)) and isinstance(observed, type(expected)):
        if len(expected) != len(observed):
            return [f"{path}.length"]
        mismatches = []
        for index, (left, right) in enumerate(zip(expected, observed)):
            mismatches.extend(_comparison_mismatches(left, right, f"{path}[{index}]"))
        return mismatches
    if isinstance(expected, float) and isinstance(observed, float):
        if math.isnan(expected) and math.isnan(observed):
            return []
    return [] if expected == observed else [path]


def _continuation_snapshot(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    deterministic_metrics = {
        key: metrics[key]
        for key in (
            "loss",
            "accuracy",
            "samples",
            "gradient_mean",
            "gradient_cv",
            "gradient_cv_method",
            "diagnostics",
        )
    }
    return _cpu_clone(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": deterministic_metrics,
            "rng_state": capture_rng_state(),
        }
    )


def checkpoint_next_step_check(
    *,
    config: RunConfig,
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Compare the exact next step through the trainer's real resume path."""

    seed_everything(config.runtime.seed, deterministic=True)
    if checkpoint_path.name != "last.pt":
        raise HealthGateError("Checkpoint continuation gate must use a file named last.pt")
    if config.runtime.dry_run:
        raise HealthGateError("Checkpoint continuation gate cannot use runtime.dry_run=true")
    is_ta = config.model.neuron == "ta_lif"
    activation_epoch = int(math.ceil(config.optimizer.epochs * config.optimizer.ta_start_fraction))
    if is_ta and activation_epoch < 1:
        raise HealthGateError(
            "TA checkpoint continuation requires at least one frozen epoch before activation"
        )
    checkpoint_epoch = activation_epoch - 1 if is_ta else 0
    model, optimizer, scheduler, ta_parameters = _build_training_objects(
        config,
        device,
        ta_enabled=False,
    )
    train_history: list[dict[str, Any]] = []
    before_metrics: dict[str, Any] | None = None
    for epoch in range(checkpoint_epoch + 1):
        _set_ta_enabled(
            ta_parameters,
            bool(is_ta and epoch >= activation_epoch),
        )
        before_metrics = _one_step(model, optimizer, config, batch, device, epoch)
        train_history.append({"epoch": epoch, **before_metrics})
        # Formal last.pt is saved after the epoch scheduler step.
        scheduler.step()
    assert before_metrics is not None
    ta_enabled_at_checkpoint = bool(is_ta and checkpoint_epoch >= activation_epoch)
    payload = _checkpoint_payload(
        model,
        optimizer,
        scheduler,
        epoch=checkpoint_epoch,
        best_val={
            "accuracy": float(before_metrics["accuracy"]),
            "loss": float(before_metrics["loss"]),
            "epoch": checkpoint_epoch,
        },
        config=config,
        train_history=train_history,
    )
    save_checkpoint(checkpoint_path, payload)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    expected_start_epoch = checkpoint_epoch + 1
    ta_enabled_on_next_step = bool(is_ta and expected_start_epoch >= activation_epoch)
    _set_ta_enabled(ta_parameters, ta_enabled_on_next_step)
    expected_metrics = _one_step(model, optimizer, config, batch, device, expected_start_epoch)
    scheduler.step()
    expected = _continuation_snapshot(model, optimizer, scheduler, expected_metrics)

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_ta = _build_training_objects(
        config,
        device,
        ta_enabled=False,
    )
    resumed_start_epoch, _best_val, resumed_history = _load_resume(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_scheduler,
        config,
    )
    _set_ta_enabled(
        resumed_ta,
        bool(is_ta and resumed_start_epoch >= activation_epoch),
    )
    observed_metrics = _one_step(
        resumed_model,
        resumed_optimizer,
        config,
        batch,
        device,
        resumed_start_epoch,
    )
    resumed_scheduler.step()
    observed = _continuation_snapshot(
        resumed_model, resumed_optimizer, resumed_scheduler, observed_metrics
    )
    mismatches = _comparison_mismatches(expected, observed)
    epoch_match = resumed_start_epoch == expected_start_epoch
    report = {
        "pass": bool(epoch_match and not mismatches),
        "comparison": "bitwise exact next training step",
        "resume_loader": "talif_msresnet.train._load_resume",
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_epoch_zero_based": checkpoint_epoch,
        "expected_start_epoch_zero_based": expected_start_epoch,
        "resumed_start_epoch_zero_based": resumed_start_epoch,
        "resume_epoch_match": epoch_match,
        "ta_activation_epoch_zero_based": activation_epoch,
        "ta_enabled_at_checkpoint": ta_enabled_at_checkpoint,
        "ta_enabled_on_next_step": ta_enabled_on_next_step,
        "ta_activation_boundary_crossed_on_resume": bool(
            is_ta
            and not ta_enabled_at_checkpoint
            and ta_enabled_on_next_step
            and expected_start_epoch == activation_epoch
        ),
        "scheduler_stepped_before_last_checkpoint": True,
        "precheckpoint_epoch_count": len(train_history),
        "resumed_history_rows": len(resumed_history),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_retained": False,
        "mismatch_count": len(mismatches),
        "mismatch_paths": mismatches[:50],
    }
    return report


def run_condition_health(
    base: RunConfig,
    *,
    condition: str,
    seed: int,
    device: torch.device,
    output_parent: Path,
    batch: tuple[torch.Tensor, torch.Tensor],
    overfit_steps: int,
    minimum_accuracy: float,
    maximum_loss_fraction: float,
    minimum_ta_coverage: float,
    checkpoint_path: Path,
) -> dict[str, Any]:
    seed_everything(seed, deterministic=True)
    config = diagnostic_config(
        base,
        condition=condition,
        seed=seed,
        device=device,
        output_parent=output_parent,
    )
    model, optimizer, _scheduler, _ta_parameters = _build_training_objects(
        config,
        device,
        ta_enabled=config.model.neuron == "ta_lif",
    )
    route_union: dict[str, set[int]] = {}
    ta_accumulator = _new_ta_gradient_accumulator(model)
    initial = _fixed_batch_metrics(model, batch)
    losses: list[float] = []

    first = _step_with_ta_tracking(
        model,
        optimizer,
        config,
        batch,
        device,
        0,
        route_union,
        ta_accumulator,
    )
    losses.append(float(first["loss"]))
    shared_gradient_report = inspect_shared_gradients(model)

    for step in range(1, overfit_steps):
        metrics = _step_with_ta_tracking(
            model,
            optimizer,
            config,
            batch,
            device,
            step,
            route_union,
            ta_accumulator,
        )
        losses.append(float(metrics["loss"]))

    final = _fixed_batch_metrics(model, batch)
    loss_fraction = final["loss"] / initial["loss"] if initial["loss"] > 0 else math.inf
    overfit_pass = bool(
        math.isfinite(loss_fraction)
        and loss_fraction <= maximum_loss_fraction
        and final["accuracy"] >= minimum_accuracy
    )
    if config.model.neuron == "ta_lif":
        ta_report = summarize_ta_gradients(
            route_union,
            ta_accumulator,
            minimum_coverage=minimum_ta_coverage,
        )
    else:
        ta_report = {"pass": True, "status": "not_applicable_lif_condition"}
    checkpoint_report = checkpoint_next_step_check(
        config=config,
        batch=batch,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    passed = bool(
        shared_gradient_report["pass"]
        and ta_report["pass"]
        and checkpoint_report["pass"]
        and overfit_pass
    )
    return {
        "pass": passed,
        "condition": condition,
        "topology": config.model.topology,
        "neuron": config.model.neuron,
        "overfit": {
            "pass": overfit_pass,
            "steps": overfit_steps,
            "initial_loss": initial["loss"],
            "final_loss": final["loss"],
            "loss_fraction": loss_fraction,
            "maximum_loss_fraction": maximum_loss_fraction,
            "initial_accuracy": initial["accuracy"],
            "final_accuracy": final["accuracy"],
            "minimum_accuracy": minimum_accuracy,
            "train_loss_history": losses,
        },
        "shared_conv_and_fc_gradients": shared_gradient_report,
        "ta_gradients": ta_report,
        "checkpoint_next_step": checkpoint_report,
    }


def _timed_mode(
    config: RunConfig,
    *,
    device: torch.device,
    batch: tuple[torch.Tensor, torch.Tensor],
    ta_enabled: bool,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    seed_everything(config.runtime.seed, deterministic=True)
    model, optimizer, _scheduler, _ta_parameters = _build_training_objects(
        config,
        device,
        ta_enabled=ta_enabled,
    )
    for step in range(warmup):
        _one_step(model, optimizer, config, batch, device, step)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = int(torch.cuda.memory_allocated(device))
    cuda_ms: list[float] = []
    wall_ms: list[float] = []
    losses: list[float] = []
    for step in range(iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()
        metrics = _one_step(model, optimizer, config, batch, device, step + warmup)
        end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter() - wall_start) * 1000.0)
        cuda_ms.append(float(start_event.elapsed_time(end_event)))
        losses.append(float(metrics["loss"]))
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if not all(math.isfinite(value) and value > 0 for value in cuda_ms):
        raise HealthGateError("CUDA timing produced a non-finite or non-positive duration")
    if not all(math.isfinite(value) for value in losses):
        raise HealthGateError("Timed training produced non-finite loss")
    return {
        "ta_parameters_enabled": ta_enabled,
        "warmup_steps": warmup,
        "timed_steps": iterations,
        "cuda_step_ms": cuda_ms,
        "cuda_step_median_ms": float(statistics.median(cuda_ms)),
        "cuda_step_mean_ms": float(statistics.mean(cuda_ms)),
        "wall_step_ms": wall_ms,
        "wall_step_median_ms": float(statistics.median(wall_ms)),
        "baseline_allocated_bytes": baseline_allocated,
        "peak_allocated_bytes": peak_allocated,
        "incremental_peak_allocated_bytes": max(0, peak_allocated - baseline_allocated),
        "peak_reserved_bytes": peak_reserved,
    }


def run_cuda_timing_gate(
    base: RunConfig,
    *,
    seed: int,
    device: torch.device,
    output_parent: Path,
    batch: tuple[torch.Tensor, torch.Tensor],
    warmup: int,
    iterations: int,
    maximum_ratio: float,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    ratios: dict[str, Any] = {}
    for condition in CONDITIONS_IN_ORDER:
        config = diagnostic_config(
            base,
            condition=condition,
            seed=seed,
            device=device,
            output_parent=output_parent,
        )
        if config.model.neuron != "ta_lif":
            rows[condition] = {
                "lif_standard": _timed_mode(
                    config,
                    device=device,
                    batch=batch,
                    ta_enabled=False,
                    warmup=warmup,
                    iterations=iterations,
                )
            }
            gc.collect()
            torch.cuda.empty_cache()
            continue

        round_orders = ((False, True), (True, False))
        rounds: list[dict[str, Any]] = []
        samples: dict[str, list[float]] = {"ta_frozen": [], "ta_enabled": []}
        peaks: dict[str, list[int]] = {"ta_frozen": [], "ta_enabled": []}
        round_ratios: list[float] = []
        for round_index, order in enumerate(round_orders, start=1):
            round_row: dict[str, Any] = {
                "round": round_index,
                "order": ["ta_enabled" if enabled else "ta_frozen" for enabled in order],
            }
            for enabled in order:
                label = "ta_enabled" if enabled else "ta_frozen"
                result = _timed_mode(
                    config,
                    device=device,
                    batch=batch,
                    ta_enabled=enabled,
                    warmup=warmup,
                    iterations=iterations,
                )
                round_row[label] = result
                samples[label].extend(float(value) for value in result["cuda_step_ms"])
                peaks[label].append(int(result["peak_allocated_bytes"]))
                gc.collect()
                torch.cuda.empty_cache()
            frozen_median = float(round_row["ta_frozen"]["cuda_step_median_ms"])
            enabled_median = float(round_row["ta_enabled"]["cuda_step_median_ms"])
            ratio = enabled_median / frozen_median if frozen_median > 0 else math.inf
            round_row["ta_enabled_over_frozen"] = ratio
            round_row["pass"] = bool(math.isfinite(ratio) and ratio <= maximum_ratio)
            round_ratios.append(ratio)
            rounds.append(round_row)

        frozen_combined = float(statistics.median(samples["ta_frozen"]))
        enabled_combined = float(statistics.median(samples["ta_enabled"]))
        combined_ratio = enabled_combined / frozen_combined if frozen_combined > 0 else math.inf
        conservative_ratio = max(round_ratios)
        ratio_pass = bool(
            all(math.isfinite(value) and value <= maximum_ratio for value in round_ratios)
            and math.isfinite(combined_ratio)
            and combined_ratio <= maximum_ratio
        )
        rows[condition] = {
            "design": "two fresh-model rounds with reversed AB/BA order",
            "rounds": rounds,
            "aggregate": {
                "ta_frozen_cuda_step_median_ms": frozen_combined,
                "ta_enabled_cuda_step_median_ms": enabled_combined,
                "ta_frozen_peak_allocated_bytes": max(peaks["ta_frozen"]),
                "ta_enabled_peak_allocated_bytes": max(peaks["ta_enabled"]),
                "combined_ta_enabled_over_frozen": combined_ratio,
                "conservative_ta_enabled_over_frozen": conservative_ratio,
            },
        }
        ratios[condition] = {
            "ta_enabled_over_frozen": conservative_ratio,
            "combined_ta_enabled_over_frozen": combined_ratio,
            "round_ratios": round_ratios,
            "maximum_allowed": maximum_ratio,
            "pass": ratio_pass,
        }
    passed = bool(ratios) and all(bool(item["pass"]) for item in ratios.values())
    return {
        "pass": passed,
        "method": (
            "synchronized CUDA events around real train_one_epoch single-batch steps; "
            "TA comparisons use two fresh-model rounds in reversed AB/BA order"
        ),
        "conditions": rows,
        "ta_enabled_over_frozen_ratios": ratios,
    }


def _formal_root_from_config(config: RunConfig) -> Path:
    return _absolute_repository_path(config.runtime.output_dir)


def run_gate(args: argparse.Namespace, config: RunConfig, output: Path) -> dict[str, Any]:
    started_at = utc_now()
    started_clock = time.perf_counter()
    if not config.runtime.deterministic:
        raise HealthGateError("Reference config must request deterministic execution")
    if config.runtime.amp:
        raise HealthGateError("This exact checkpoint/timing gate requires reference amp=false")
    seed_everything(args.seed, deterministic=True)
    device = torch.device(args.device)
    hardware = require_target_cuda(device, args.expected_gpu_substring)
    git_identity = repository_git_identity(REPOSITORY_ROOT)
    if not git_identity["tracked_clean"]:
        raise HealthGateError("Pilot health evidence requires a clean tracked worktree")
    protocol = load_protocol(args.protocol)
    acceptance = protocol.get("pilot_acceptance")
    if not isinstance(acceptance, Mapping):
        raise HealthGateError("Protocol has no pilot_acceptance health-gate binding")
    environment_binding = acceptance.get("environment")
    if not isinstance(environment_binding, Mapping):
        raise HealthGateError("Protocol has no pilot_acceptance.environment binding")
    validate_runtime_environment(environment_binding, hardware)

    timing_batch_size = (
        int(args.timing_batch_size)
        if args.timing_batch_size is not None
        else int(config.optimizer.batch_size)
    )
    required_batch_size = max(int(args.overfit_batch_size), timing_batch_size)
    config, all_inputs, all_targets, split_manifest = load_fixed_real_batch(
        config,
        seed=args.seed,
        required_batch_size=required_batch_size,
    )
    overfit_batch = (
        all_inputs[: args.overfit_batch_size].to(device, non_blocking=False),
        all_targets[: args.overfit_batch_size].to(device, non_blocking=False),
    )
    timing_batch = (
        all_inputs[:timing_batch_size].to(device, non_blocking=False),
        all_targets[:timing_batch_size].to(device, non_blocking=False),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    condition_reports: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="pilot-health-checkpoints-", dir=str(output.parent)
    ) as temporary:
        temporary_root = Path(temporary)
        for condition in CONDITIONS_IN_ORDER:
            condition_reports[condition] = run_condition_health(
                config,
                condition=condition,
                seed=args.seed,
                device=device,
                output_parent=output.parent,
                batch=overfit_batch,
                overfit_steps=args.overfit_steps,
                minimum_accuracy=args.min_overfit_accuracy,
                maximum_loss_fraction=args.max_overfit_loss_fraction,
                minimum_ta_coverage=args.min_ta_routed_gradient_coverage,
                checkpoint_path=temporary_root / condition / "last.pt",
            )
            gc.collect()
            torch.cuda.empty_cache()

    idle_precheck = gpu_idle_precheck(device)
    timing_report = run_cuda_timing_gate(
        config,
        seed=args.seed,
        device=device,
        output_parent=output.parent,
        batch=timing_batch,
        warmup=args.timing_warmup,
        iterations=args.timing_iterations,
        maximum_ratio=args.max_ta_step_ratio,
    )
    condition_pass = all(bool(item["pass"]) for item in condition_reports.values())
    passed = bool(condition_pass and timing_report["pass"])
    failures: list[str] = []
    for condition, item in condition_reports.items():
        if not item["pass"]:
            failures.append(f"{condition}: functional/gradient/checkpoint/overfit gate failed")
    for condition, item in timing_report["ta_enabled_over_frozen_ratios"].items():
        if not item["pass"]:
            failures.append(
                f"{condition}: TA step ratio {item['ta_enabled_over_frozen']:.4f} "
                f"exceeds {item['maximum_allowed']:.4f}"
            )
    return {
        "schema_version": 2,
        "artifact_class": ARTIFACT_CLASS,
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v2-pilot implementation health validation",
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "protocol_hash": stable_hash(protocol),
        "acceptance_hash": stable_hash(dict(acceptance)),
        "git_commit": git_identity["git_commit"],
        "tracked_clean": git_identity["tracked_clean"],
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": time.perf_counter() - started_clock,
        "seed": args.seed,
        "conditions": list(CONDITIONS_IN_ORDER),
        "source": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "protocol": str(args.protocol.resolve()),
            "protocol_sha256": sha256_file(args.protocol),
            "dataset": "cifar100",
            "split_manifest_sha256": split_manifest["manifest_sha256"],
            "fixed_batch_sha256": tensor_batch_sha256(all_inputs, all_targets),
            "fixed_batch_shape": list(all_inputs.shape),
            "overfit_batch_size": args.overfit_batch_size,
            "timing_batch_size": timing_batch_size,
        },
        "environment": {
            **environment_manifest(),
            "hardware": hardware,
            "python_implementation": platform.python_implementation(),
            "precision": "float32",
            "amp": False,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "torch_deterministic_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu_idle_precheck": idle_precheck,
        },
        "thresholds": {
            "overfit_steps": args.overfit_steps,
            "minimum_overfit_accuracy": args.min_overfit_accuracy,
            "maximum_overfit_loss_fraction": args.max_overfit_loss_fraction,
            "minimum_ta_routed_gradient_coverage": (args.min_ta_routed_gradient_coverage),
            "maximum_ta_enabled_over_frozen_step_ratio": args.max_ta_step_ratio,
        },
        "functional_health_by_condition": condition_reports,
        "cuda_train_step_timing": timing_report,
        "failures": failures,
    }


def _fatal_report(args: argparse.Namespace, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_class": ARTIFACT_CLASS,
        "reporting_eligibility": "FORBIDDEN_FROM_MANUSCRIPT_RESULTS",
        "confirmatory_analysis_eligibility": False,
        "purpose": "pre-v2-pilot implementation health validation",
        "status": "FAIL",
        "pass": False,
        "finished_at": utc_now(),
        "seed": getattr(args, "seed", None),
        "fatal_error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_gate_arguments(args)
        initial_output = validate_output_path(
            args.output,
            formal_roots=[],
            repository_root=REPOSITORY_ROOT,
        )
        config = load_run_config(args.config, args.protocol)
        output = validate_output_path(
            initial_output,
            formal_roots=[_formal_root_from_config(config)],
            repository_root=REPOSITORY_ROOT,
        )
        protocol = load_protocol(args.protocol)
        validate_protocol_health_binding(args, config, protocol, output)
    except Exception as exc:
        print(f"PILOT_HEALTH_GATE_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        report = run_gate(args, config, output)
    except Exception as exc:
        report = _fatal_report(args, exc)
    atomic_write_json(output, report)
    print(f"PILOT_HEALTH_GATE_{report['status']}")
    print(f"NON_REPORTING_OUTPUT={output}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
