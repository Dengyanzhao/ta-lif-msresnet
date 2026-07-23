#!/usr/bin/env python3
"""Export a fixed validation batch for E3 diagnostics and activity benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import load_run_config  # noqa: E402
from talif_msresnet.data import build_loaders  # noqa: E402
from talif_msresnet.diagnostics import representative_batch_sha256  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.utils import atomic_torch_save, stable_hash, utc_now  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite fixed representative batch: {args.output}")
    config = load_run_config(args.config)
    loaders = build_loaders(config, final_test=False, seed=config.runtime.seed)
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    collected = 0
    for batch in loaders["val"]:
        take = min(args.batch_size - collected, int(batch[0].shape[0]))
        inputs.append(batch[0][:take].cpu())
        targets.append(batch[1][:take].cpu())
        collected += take
        if collected == args.batch_size:
            break
    if collected != args.batch_size:
        raise RuntimeError(f"Validation loader provided only {collected} samples")
    input_tensor = torch.cat(inputs, dim=0).contiguous()
    target_tensor = torch.cat(targets, dim=0).long().contiguous()
    manifest = loaders.get("_manifest", {})
    manifest_hash = manifest.get("manifest_sha256", stable_hash(manifest)) if isinstance(manifest, dict) else ""
    metadata: dict[str, Any] = {
        "created_at": utc_now(),
        "source_split": "validation",
        "test_data_accessed": False,
        "config": artifact_path_reference(args.config, PROJECT_ROOT),
        "config_hash": config.config_hash,
        "protocol_hash": config.analysis.get("protocol_hash", ""),
        "dataset": config.data.dataset,
        "split_manifest_sha256": manifest_hash,
        "in_channels": config.model.in_channels,
        "time_steps": config.model.time_steps,
        "num_classes": config.model.num_classes,
        "normalize": config.data.normalize,
        "batch_size": args.batch_size,
        "input_shape": list(input_tensor.shape),
        "target_shape": list(target_tensor.shape),
        "representative_batch_sha256": representative_batch_sha256(input_tensor, target_tensor),
    }
    atomic_torch_save(
        args.output,
        {"inputs": input_tensor, "targets": target_tensor, "metadata": metadata},
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
