#!/usr/bin/env python3
"""Benchmark a selected checkpoint matrix slice with one fixed validation tensor."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.config import load_protocol, validate_run_mapping  # noqa: E402
from talif_msresnet.data import (  # noqa: E402
    load_representative_batch_artifact,
    validate_representative_batch_artifact,
)
from talif_msresnet.utils import load_checkpoint, sha256_file  # noqa: E402
from benchmark_checkpoint import (  # noqa: E402
    benchmark_device_identity,
    build_benchmark_identity,
    checkpoint_split_manifest_sha256,
    resolve_frozen_energy_constants,
    validate_frozen_benchmark_request,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with (path / "run_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "benchmark_id" not in (reader.fieldnames or []):
            raise ValueError(
                f"Existing benchmark file predates benchmark_id; archive or migrate it: {path}"
            )
        values = [row.get("benchmark_id", "") for row in reader if row.get("benchmark_id")]
        if len(values) != len(set(values)):
            raise ValueError(f"Existing benchmark file contains duplicate benchmark_id values: {path}")
        return set(values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(PROJECT_ROOT / "configs" / "generated"))
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results" / "runs"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "results" / "benchmark_results.csv"))
    parser.add_argument("--input-tensor", required=True, type=Path)
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", choices=("E1",))
    parser.add_argument("--depth", type=int)
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--condition", choices=("C1", "C2", "C3", "C4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument(
        "--energy-constants",
        type=Path,
        help="Optional repetition of the exact constants path already frozen in the protocol",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("--warmup must be >= 0 and --iterations must be >= 1")
    report = check_protocol(args.protocol, mode="full", project_root=PROJECT_ROOT)
    if not report.ok:
        raise SystemExit("Benchmark preflight blocked:\n" + "\n".join(f"- {e}" for e in report.errors))
    protocol = load_protocol(args.protocol)
    benchmark_protocol = protocol["benchmark"]
    try:
        validate_frozen_benchmark_request(
            benchmark_protocol,
            device=args.device,
            precision=args.precision,
            warmup_iterations=args.warmup,
            timed_iterations=args.iterations,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    config_dir = Path(args.config_dir).resolve()
    results_root = Path(args.results_root).resolve()
    output = Path(args.output).resolve()
    input_path = args.input_tensor.resolve()
    try:
        energy_path, energy_constants_sha256 = resolve_frozen_energy_constants(
            benchmark_protocol,
            protocol_path=args.protocol,
            requested_path=args.energy_constants,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rows = [row for row in _rows(config_dir) if row.get("dataset") == args.dataset]
    for key, value in (
        ("experiment", args.experiment),
        ("condition", args.condition),
    ):
        if value is not None:
            rows = [row for row in rows if row.get(key) == value]
    if args.depth is not None:
        rows = [row for row in rows if int(row["depth"]) == args.depth]
    if args.time_steps is not None:
        rows = [row for row in rows if int(row["time_steps"]) == args.time_steps]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No checkpoints match the requested benchmark slice")

    completed_ids = _completed_ids(output)
    representative_inputs, representative_targets, representative_metadata = (
        load_representative_batch_artifact(input_path)
    )
    input_file_sha256 = sha256_file(input_path)
    device_identity = benchmark_device_identity(args.device)
    failures = 0
    for index, row in enumerate(rows, start=1):
        run_id = row["run_id"]
        checkpoint = results_root / run_id / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        payload = load_checkpoint(checkpoint, map_location="cpu")
        config = payload.get("config", {})
        analysis = config.get("analysis", {}) if isinstance(config, dict) else {}
        if not isinstance(analysis, dict) or analysis.get("protocol_hash") != report.protocol_hash:
            raise RuntimeError(f"{run_id}: checkpoint protocol hash differs from frozen protocol")
        resolved_config = validate_run_mapping(config)
        if payload.get("config_hash") != resolved_config.config_hash:
            raise RuntimeError(f"{run_id}: checkpoint scientific config hash is invalid")
        input_batch_sha256 = validate_representative_batch_artifact(
            representative_inputs,
            representative_targets,
            representative_metadata,
            expected_dataset=resolved_config.data.dataset,
            expected_protocol_hash=report.protocol_hash,
            expected_split_manifest_sha256=checkpoint_split_manifest_sha256(
                checkpoint, str(payload.get("config_hash", ""))
            ),
            expected_in_channels=resolved_config.model.in_channels,
            expected_time_steps=resolved_config.model.time_steps,
            expected_num_classes=resolved_config.model.num_classes,
        )
        identity = build_benchmark_identity(
            protocol_hash=report.protocol_hash,
            checkpoint_sha256=sha256_file(checkpoint),
            input_file_sha256=input_file_sha256,
            input_batch_sha256=input_batch_sha256,
            input_kind="representative_validation_batch",
            device_identity=device_identity,
            precision=args.precision,
            warmup_iterations=args.warmup,
            timed_iterations=args.iterations,
            energy_constants_sha256=energy_constants_sha256,
        )
        benchmark_id = str(identity["benchmark_id"])
        if benchmark_id in completed_ids:
            print(f"[{index}/{len(rows)}] {run_id}: already_complete")
            continue
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "benchmark_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--input-tensor",
            str(input_path),
            "--device",
            str(device_identity["device"]),
            "--warmup",
            str(args.warmup),
            "--iterations",
            str(args.iterations),
            "--precision",
            args.precision,
            "--expected-benchmark-id",
            benchmark_id,
            "--protocol",
            str(Path(args.protocol).resolve()),
        ]
        if energy_path is not None:
            command.extend(["--energy-constants", str(energy_path)])
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(SRC_ROOT), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
        print(f"[{index}/{len(rows)}] {run_id}: returncode={completed.returncode}")
        if completed.returncode:
            failures += 1
            if not args.continue_on_error:
                break
        else:
            completed_ids.add(benchmark_id)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
