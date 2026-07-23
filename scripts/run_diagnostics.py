#!/usr/bin/env python3
"""Run idempotent E3 checkpoint diagnostics over a selected matrix slice."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import load_protocol  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.utils import load_checkpoint  # noqa: E402
from diagnose_checkpoint import validate_frozen_diagnostic_settings  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _rows(path: Path) -> list[dict[str, str]]:
    with (path / "run_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_diagnostic_protocol(path: Path, protocol: dict) -> None:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Diagnostic protocol JSON must be a mapping")
    allowed = {"device", "batch_size", "probes", "probe_seed", "time_index"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown diagnostic protocol keys: {unknown}")
    validate_frozen_diagnostic_settings(values, protocol["analysis"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default=str(PROJECT_ROOT / "configs" / "generated"))
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results" / "runs"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "diagnostics"))
    parser.add_argument("--input-batch", required=True, type=Path)
    parser.add_argument("--diagnostic-protocol", default=str(PROJECT_ROOT / "examples" / "diagnostic_protocol.json"))
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--experiment", choices=("E1",), default="E1")
    parser.add_argument("--depth", type=int)
    parser.add_argument("--time-steps", type=int)
    parser.add_argument("--condition", choices=("C1", "C2", "C3", "C4"))
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_dir = _project_path(args.config_dir)
    results_root = _project_path(args.results_root)
    output_root = _project_path(args.output_root)
    input_batch = _project_path(args.input_batch)
    diagnostic_protocol = _project_path(args.diagnostic_protocol)
    protocol_path = _project_path(args.protocol)
    report = check_protocol(protocol_path, mode="full", project_root=PROJECT_ROOT)
    if not report.ok:
        raise SystemExit("Diagnostic preflight blocked:\n" + "\n".join(f"- {e}" for e in report.errors))
    protocol = load_protocol(protocol_path)
    _validate_diagnostic_protocol(diagnostic_protocol, protocol)
    json_settings = json.loads(diagnostic_protocol.read_text(encoding="utf-8"))
    if args.device is not None and json_settings.get("device") not in (None, args.device):
        raise SystemExit(
            "--device cannot override the device recorded in the diagnostic protocol JSON"
        )
    rows = [
        row
        for row in _rows(config_dir)
        if row.get("dataset") == args.dataset and row.get("experiment") == args.experiment
    ]
    if args.depth is not None:
        rows = [row for row in rows if int(row["depth"]) == args.depth]
    if args.time_steps is not None:
        rows = [row for row in rows if int(row["time_steps"]) == args.time_steps]
    if args.condition:
        rows = [row for row in rows if row.get("condition") == args.condition]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No checkpoints match the requested diagnostic slice")

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
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "diagnose_checkpoint.py"),
            "--checkpoint",
            str(checkpoint),
            "--input-batch",
            str(input_batch),
            "--output",
            str(output_root / run_id),
            "--protocol",
            str(diagnostic_protocol),
            "--study-protocol",
            str(protocol_path),
        ]
        if args.device:
            command.extend(["--device", args.device])
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
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
