#!/usr/bin/env python3
"""Create the one fixed validation batch used by every V6 resource audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.benchmark_v6 import BATCH_FILENAME
from talif_msresnet.config import load_protocol, load_run_config
from talif_msresnet.config_v6 import (
    V6_ACTIVE_CONDITIONS,
    V6_ARTIFACT_PATHS,
    V6_BENCHMARK_SEED,
    V6_FORMAL_SEEDS,
    validate_v6_protocol,
)
from talif_msresnet.data import build_loaders
from talif_msresnet.diagnostics import representative_batch_sha256
from talif_msresnet.freeze import verify_formal_freeze
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.preflight import check_protocol
from talif_msresnet.utils import (
    atomic_torch_save,
    sha256_file,
    utc_now,
)


class V6BatchExportError(RuntimeError):
    """Raised when the fixed validation input cannot be bound to V6."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / V6_ARTIFACT_PATHS["protocol"],
    )
    parser.add_argument("--config-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def _collect_batch(loader: Any, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise V6BatchExportError("V6 validation loader returned an invalid batch")
        batch_inputs, batch_targets = batch[0], batch[1]
        if not isinstance(batch_inputs, torch.Tensor) or not isinstance(
            batch_targets, torch.Tensor
        ):
            raise V6BatchExportError("V6 validation loader returned non-tensor data")
        take = min(batch_size - collected, int(batch_inputs.shape[0]))
        inputs.append(batch_inputs[:take].cpu())
        targets.append(batch_targets[:take].long().cpu())
        collected += take
        if collected == batch_size:
            break
    if collected != batch_size:
        raise V6BatchExportError(
            f"V6 validation loader provided {collected} samples; expected {batch_size}"
        )
    return torch.cat(inputs, dim=0).contiguous(), torch.cat(targets, dim=0).contiguous()


def _existing_is_identical(
    output: Path,
    *,
    expected_metadata: dict[str, Any],
    expected_batch_hash: str,
) -> bool:
    try:
        payload = torch.load(output, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(output, map_location="cpu")
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    inputs = payload.get("inputs")
    targets = payload.get("targets")
    if not isinstance(metadata, dict) or not isinstance(inputs, torch.Tensor) or not isinstance(
        targets, torch.Tensor
    ):
        return False
    observed = dict(metadata)
    expected = dict(expected_metadata)
    observed.pop("created_at", None)
    expected.pop("created_at", None)
    return observed == expected and representative_batch_sha256(inputs, targets) == expected_batch_hash


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size != 128:
            raise V6BatchExportError("V6 benchmark batch size is frozen at 128")
        protocol_path = args.protocol.resolve()
        validate_v6_protocol(load_protocol(protocol_path))
        expected_protocol = (PROJECT_ROOT / V6_ARTIFACT_PATHS["protocol"]).resolve()
        if protocol_path != expected_protocol:
            raise V6BatchExportError(f"V6 requires the canonical protocol path: {expected_protocol}")
        report = check_protocol(
            protocol_path,
            mode="final-test",
            project_root=PROJECT_ROOT,
            check_dependencies=False,
        )
        if not report.ok:
            raise V6BatchExportError("V6 final-test preflight failed: " + "; ".join(report.errors))
        verify_formal_freeze(
            project_root=PROJECT_ROOT,
            protocol_path=protocol_path,
            matrix_dir=PROJECT_ROOT / V6_ARTIFACT_PATHS["formal_matrix"],
        )
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else (PROJECT_ROOT / V6_ARTIFACT_PATHS["formal_matrix"]).resolve()
        )
        if config_dir != (PROJECT_ROOT / V6_ARTIFACT_PATHS["formal_matrix"]).resolve():
            raise V6BatchExportError("V6 config directory is not the frozen formal matrix")
        config_path = config_dir / (
            f"E6_cifar100_d20_t6_{V6_ACTIVE_CONDITIONS[0]}_s{V6_FORMAL_SEEDS[0]}.yaml"
        )
        if not config_path.is_file():
            raise V6BatchExportError(f"Frozen reference config is missing: {config_path}")
        config = load_run_config(config_path, protocol_path)
        if config.model.condition != V6_ACTIVE_CONDITIONS[0] or config.runtime.seed != V6_FORMAL_SEEDS[0]:
            raise V6BatchExportError("Frozen reference config is not the first V6 matrix row")
        loaders = build_loaders(config, final_test=False, seed=V6_BENCHMARK_SEED)
        inputs, targets = _collect_batch(loaders["val"], args.batch_size)
        manifest = loaders.get("_manifest", {})
        if not isinstance(manifest, dict) or not isinstance(manifest.get("manifest_sha256"), str):
            raise V6BatchExportError("Validation loader did not expose a split-manifest hash")
        batch_hash = representative_batch_sha256(inputs, targets)
        metadata: dict[str, Any] = {
            "schema": "ta-lif-msresnet-v6-validation-batch-v1",
            "created_at": utc_now(),
            "source_split": "validation",
            "test_data_accessed": False,
            "config": artifact_path_reference(config_path, PROJECT_ROOT),
            "config_file_sha256": sha256_file(config_path),
            "config_hash": config.config_hash,
            "protocol": artifact_path_reference(protocol_path, PROJECT_ROOT),
            "protocol_file_sha256": sha256_file(protocol_path),
            "protocol_hash": report.protocol_hash,
            "dataset": config.data.dataset,
            "split_manifest_sha256": manifest["manifest_sha256"],
            "input_seed": V6_BENCHMARK_SEED,
            "in_channels": config.model.in_channels,
            "time_steps": config.model.time_steps,
            "num_classes": config.model.num_classes,
            "normalize": config.data.normalize,
            "batch_size": args.batch_size,
            "input_shape": list(inputs.shape),
            "target_shape": list(targets.shape),
            "representative_batch_sha256": batch_hash,
        }
        output = (
            args.output.resolve()
            if args.output is not None
            else (PROJECT_ROOT / V6_ARTIFACT_PATHS["benchmark_results"] / BATCH_FILENAME).resolve()
        )
        expected_output = (PROJECT_ROOT / V6_ARTIFACT_PATHS["benchmark_results"] / BATCH_FILENAME).resolve()
        if output != expected_output:
            raise V6BatchExportError(f"V6 output must be canonical: {expected_output}")
        if output.exists():
            if _existing_is_identical(
                output, expected_metadata=metadata, expected_batch_hash=batch_hash
            ):
                print(f"V6_VALIDATION_BATCH_REUSED={output}")
                print(f"V6_VALIDATION_BATCH_SHA256={sha256_file(output)}")
                return 0
            raise V6BatchExportError("Existing V6 validation batch differs; refusing overwrite")
        atomic_torch_save(output, {"inputs": inputs, "targets": targets, "metadata": metadata})
        print(f"V6_VALIDATION_BATCH_CREATED={output}")
        print(f"V6_VALIDATION_BATCH_SHA256={sha256_file(output)}")
        print(f"V6_VALIDATION_BATCH_CONTENT_SHA256={batch_hash}")
        return 0
    except (V6BatchExportError, OSError, ValueError, RuntimeError) as exc:
        print(f"V6_VALIDATION_BATCH_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
