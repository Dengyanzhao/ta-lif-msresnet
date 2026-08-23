#!/usr/bin/env python3
"""Create the single frozen validation-only batch for every V7 benchmark."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.benchmark_v7 import (  # noqa: E402
    BATCH_FILENAME,
    canonical_bound_paths,
    repository_git_identity,
)
from talif_msresnet.config import load_protocol, load_run_config  # noqa: E402
from talif_msresnet.config_v7 import (  # noqa: E402
    V7_ACTIVE_CONDITIONS,
    V7_ARTIFACT_PATHS,
    V7_BENCHMARK_SEED,
    V7_FORMAL_SEEDS,
    validate_v7_cifar100_provenance_files,
    validate_v7_protocol,
)
from talif_msresnet.data import build_loaders  # noqa: E402
from talif_msresnet.diagnostics import representative_batch_sha256  # noqa: E402
from talif_msresnet.freeze import verify_formal_freeze  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.utils import (  # noqa: E402
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
    stable_hash,
    utc_now,
)


class V7BatchExportError(RuntimeError):
    """Raised before any V7 benchmark input can be accepted."""


# The first benchmark attempt was made before the evaluator-only recovery
# commit.  The tensors are immutable; only these two provenance fields may be
# rebound after the new formal freeze is created.
V7_BATCH_RECOVERY_BASE_COMMIT = "b780b38322b491f849592d71814b7eb44de16948"
V7_BATCH_RECOVERY_BASE_FREEZE_SHA256 = (
    "c2c596e14a4f4144bec1d5f12cc0bcf2020f60eca463712559dc3e1658b752e0"
)
V7_BATCH_RECOVERY_BASE_FILE_SHA256 = (
    "cf1557a2192d337d9f01f18c85c190fd4ad98b8eb750bde0b2fe80150597906f"
)
V7_BATCH_RECOVERY_CONTENT_SHA256 = (
    "c71537580ee9f8d17f7d66d85027f73691f12a533fd87379bf268d28aa9f3869"
)
V7_BATCH_RECOVERY_SIDECAR = "validation_batch_compatibility_recovery.json"
V7_BATCH_RECOVERY_ALLOWED_METADATA_PATHS = (
    "$.git_commit",
    "$.freeze_manifest_sha256",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / V7_ARTIFACT_PATHS["protocol"],
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
            raise V7BatchExportError("V7 validation loader returned an invalid batch")
        batch_inputs, batch_targets = batch[0], batch[1]
        if not isinstance(batch_inputs, torch.Tensor) or not isinstance(
            batch_targets, torch.Tensor
        ):
            raise V7BatchExportError("V7 validation loader returned non-tensor data")
        take = min(batch_size - collected, int(batch_inputs.shape[0]))
        inputs.append(batch_inputs[:take].cpu())
        targets.append(batch_targets[:take].long().cpu())
        collected += take
        if collected == batch_size:
            break
    if collected != batch_size:
        raise V7BatchExportError(
            f"V7 validation loader provided {collected} samples; expected {batch_size}"
        )
    return torch.cat(inputs).contiguous(), torch.cat(targets).contiguous()


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
    observed.pop("compatibility_recovery", None)
    expected.pop("created_at", None)
    return (
        observed == expected
        and representative_batch_sha256(inputs, targets) == expected_batch_hash
    )


def _rebind_existing_batch(
    output: Path,
    *,
    expected_metadata: dict[str, Any],
    expected_batch_hash: str,
    sidecar_path: Path,
) -> bool:
    """Rebind only evaluator provenance on the sealed first-attempt batch.

    The function fails closed unless the original file hash, content hash,
    source commit, freeze hash, and every non-allowed metadata field match the
    known first benchmark attempt.  Tensor values are loaded and saved
    unchanged; no dataset loader is used for this recovery write.
    """

    try:
        payload = torch.load(output, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(output, map_location="cpu")
    if not isinstance(payload, Mapping):
        return False
    metadata = payload.get("metadata")
    inputs = payload.get("inputs")
    targets = payload.get("targets")
    if not isinstance(metadata, Mapping) or not isinstance(inputs, torch.Tensor) or not isinstance(
        targets, torch.Tensor
    ):
        return False
    observed = dict(metadata)
    if sha256_file(output) != V7_BATCH_RECOVERY_BASE_FILE_SHA256:
        return False
    content_hash = representative_batch_sha256(inputs, targets)
    if content_hash != V7_BATCH_RECOVERY_CONTENT_SHA256 or content_hash != expected_batch_hash:
        return False
    if observed.get("git_commit") != V7_BATCH_RECOVERY_BASE_COMMIT:
        return False
    if observed.get("freeze_manifest_sha256") != V7_BATCH_RECOVERY_BASE_FREEZE_SHA256:
        return False

    comparable_observed = dict(observed)
    comparable_expected = dict(expected_metadata)
    for value in (comparable_observed, comparable_expected):
        value.pop("created_at", None)
        value.pop("compatibility_recovery", None)
    differences = {
        key
        for key in set(comparable_observed) | set(comparable_expected)
        if comparable_observed.get(key) != comparable_expected.get(key)
    }
    allowed = {path[2:] for path in V7_BATCH_RECOVERY_ALLOWED_METADATA_PATHS}
    if differences != allowed:
        return False

    if sidecar_path.exists():
        raise V7BatchExportError(
            "V7 batch compatibility sidecar exists while the batch still has the "
            "pre-recovery file identity"
        )

    rebound = dict(observed)
    rebound["git_commit"] = expected_metadata["git_commit"]
    rebound["freeze_manifest_sha256"] = expected_metadata["freeze_manifest_sha256"]
    atomic_torch_save(output, {"inputs": inputs, "targets": targets, "metadata": rebound})
    new_file_sha256 = sha256_file(output)
    record = {
        "schema": "ta-lif-msresnet-v7-validation-batch-compatibility-recovery-v1",
        "artifact_class": "NON_REPORTABLE_V7_VALIDATION_BATCH_METADATA_REBIND",
        "status": "PASS",
        "pass": True,
        "test_data_accessed": False,
        "base_commit": V7_BATCH_RECOVERY_BASE_COMMIT,
        "recovery_commit": expected_metadata["git_commit"],
        "base_freeze_manifest_sha256": V7_BATCH_RECOVERY_BASE_FREEZE_SHA256,
        "recovery_freeze_manifest_sha256": expected_metadata["freeze_manifest_sha256"],
        "base_batch_file_sha256": V7_BATCH_RECOVERY_BASE_FILE_SHA256,
        "rebound_batch_file_sha256": new_file_sha256,
        "batch_content_sha256": content_hash,
        "allowed_metadata_paths": list(V7_BATCH_RECOVERY_ALLOWED_METADATA_PATHS),
        "old_metadata": {
            "git_commit": V7_BATCH_RECOVERY_BASE_COMMIT,
            "freeze_manifest_sha256": V7_BATCH_RECOVERY_BASE_FREEZE_SHA256,
        },
        "new_metadata": {
            "git_commit": expected_metadata["git_commit"],
            "freeze_manifest_sha256": expected_metadata["freeze_manifest_sha256"],
        },
    }
    record["record_sha256"] = stable_hash(record)
    atomic_write_json(sidecar_path, record)
    print(f"V7_VALIDATION_BATCH_COMPATIBILITY_REBOUND={output}")
    print(f"V7_VALIDATION_BATCH_BASE_SHA256={V7_BATCH_RECOVERY_BASE_FILE_SHA256}")
    print(f"V7_VALIDATION_BATCH_REBOUND_SHA256={new_file_sha256}")
    print(f"V7_VALIDATION_BATCH_CONTENT_SHA256={content_hash}")
    print(f"V7_VALIDATION_BATCH_RECOVERY_SIDECAR={sidecar_path}")
    return True


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.batch_size != 128:
            raise V7BatchExportError("V7 benchmark batch size is frozen at 128")
        protocol_path = args.protocol.resolve()
        expected_protocol = (PROJECT_ROOT / V7_ARTIFACT_PATHS["protocol"]).resolve()
        if protocol_path != expected_protocol:
            raise V7BatchExportError(f"V7 requires the canonical protocol: {expected_protocol}")
        protocol = validate_v7_protocol(load_protocol(protocol_path))
        report = check_protocol(
            protocol_path,
            mode="final-test",
            project_root=PROJECT_ROOT,
            check_dependencies=False,
        )
        if not report.ok:
            raise V7BatchExportError("V7 final-test preflight failed: " + "; ".join(report.errors))
        verify_formal_freeze(
            project_root=PROJECT_ROOT,
            protocol_path=protocol_path,
            matrix_dir=PROJECT_ROOT / V7_ARTIFACT_PATHS["formal_matrix"],
        )
        validate_v7_cifar100_provenance_files(protocol, project_root=PROJECT_ROOT)
        commit, clean = repository_git_identity(PROJECT_ROOT)
        if not clean:
            raise V7BatchExportError("V7 validation batch requires a clean tracked release commit")
        config_dir = (
            args.config_dir.resolve()
            if args.config_dir is not None
            else (PROJECT_ROOT / V7_ARTIFACT_PATHS["formal_matrix"]).resolve()
        )
        expected_config_dir = (PROJECT_ROOT / V7_ARTIFACT_PATHS["formal_matrix"]).resolve()
        if config_dir != expected_config_dir:
            raise V7BatchExportError("V7 config directory is not the frozen formal matrix")
        config_path = config_dir / (
            f"E9_cifar100_d20_t6_{V7_ACTIVE_CONDITIONS[0]}_s{V7_FORMAL_SEEDS[0]}.yaml"
        )
        if not config_path.is_file():
            raise V7BatchExportError(f"Frozen V7 reference config is missing: {config_path}")
        config = load_run_config(config_path, protocol_path)
        if (
            config.model.condition != V7_ACTIVE_CONDITIONS[0]
            or config.runtime.seed != V7_FORMAL_SEEDS[0]
        ):
            raise V7BatchExportError("Frozen reference config is not the first V7 matrix row")
        loaders = build_loaders(config, final_test=False, seed=V7_BENCHMARK_SEED)
        inputs, targets = _collect_batch(loaders["val"], args.batch_size)
        manifest = loaders.get("_manifest", {})
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("manifest_sha256"), str
        ):
            raise V7BatchExportError("Validation loader did not expose a split-manifest hash")
        batch_hash = representative_batch_sha256(inputs, targets)
        bound = canonical_bound_paths(protocol, PROJECT_ROOT)
        metadata: dict[str, Any] = {
            "schema": "ta-lif-msresnet-v7-validation-batch-v1",
            "created_at": utc_now(),
            "source_split": "validation",
            "test_data_accessed": False,
            "git_commit": commit,
            "tracked_clean": True,
            "config": artifact_path_reference(config_path, PROJECT_ROOT),
            "config_file_sha256": sha256_file(config_path),
            "config_hash": config.config_hash,
            "protocol": artifact_path_reference(protocol_path, PROJECT_ROOT),
            "protocol_file_sha256": sha256_file(bound["protocol"]),
            "matrix_manifest": artifact_path_reference(bound["matrix_manifest"], PROJECT_ROOT),
            "matrix_manifest_sha256": sha256_file(bound["matrix_manifest"]),
            "freeze_manifest": artifact_path_reference(bound["freeze_manifest"], PROJECT_ROOT),
            "freeze_manifest_sha256": sha256_file(bound["freeze_manifest"]),
            "protocol_hash": report.protocol_hash,
            "dataset": config.data.dataset,
            "split_manifest_sha256": manifest["manifest_sha256"],
            "input_seed": V7_BENCHMARK_SEED,
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
            else (PROJECT_ROOT / V7_ARTIFACT_PATHS["benchmark_results"] / BATCH_FILENAME).resolve()
        )
        expected_output = (
            PROJECT_ROOT / V7_ARTIFACT_PATHS["benchmark_results"] / BATCH_FILENAME
        ).resolve()
        if output != expected_output:
            raise V7BatchExportError(f"V7 output must be canonical: {expected_output}")
        if output.exists():
            if _existing_is_identical(
                output, expected_metadata=metadata, expected_batch_hash=batch_hash
            ):
                print(f"V7_VALIDATION_BATCH_REUSED={output}")
                print(f"V7_VALIDATION_BATCH_SHA256={sha256_file(output)}")
                return 0
            if _rebind_existing_batch(
                output,
                expected_metadata=metadata,
                expected_batch_hash=batch_hash,
                sidecar_path=output.parent / V7_BATCH_RECOVERY_SIDECAR,
            ):
                return 0
            raise V7BatchExportError("Existing V7 validation batch differs; refusing overwrite")
        atomic_torch_save(output, {"inputs": inputs, "targets": targets, "metadata": metadata})
        print(f"V7_VALIDATION_BATCH_CREATED={output}")
        print(f"V7_VALIDATION_BATCH_SHA256={sha256_file(output)}")
        print(f"V7_VALIDATION_BATCH_CONTENT_SHA256={batch_hash}")
        return 0
    except (V7BatchExportError, OSError, ValueError, RuntimeError) as exc:
        print(f"V7_VALIDATION_BATCH_BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
