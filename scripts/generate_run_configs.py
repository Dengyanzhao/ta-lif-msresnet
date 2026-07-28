#!/usr/bin/env python
"""Generate the protocol-bound run configuration matrix."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import (  # noqa: E402
    RunConfig,
    active_conditions_for_protocol,
    artifact_paths_for_protocol,
    generate_run_matrix,
    generate_v3_pilot_matrix,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.utils import atomic_write_json, sha256_file, stable_hash  # noqa: E402


def _write_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(dict(value), handle, sort_keys=False, allow_unicode=False)


def _protocol_reference(protocol_path: str | Path) -> str:
    path = Path(protocol_path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Protocol path must stay inside the project root") from exc


def _manifest_rows(runs: Sequence[RunConfig], output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        model = run.model
        data = run.data
        config_file = f"{run.runtime.run_id}.yaml"
        rows.append({
            "run_id": run.runtime.run_id,
            "experiment": run.experiment,
            "dataset": data.dataset,
            "depth": model.depth,
            "time_steps": model.time_steps,
            "condition": model.condition,
            "topology": model.topology,
            "neuron": model.neuron,
            "seed": run.runtime.seed,
            "config_file": config_file,
            "config_hash": run.config_hash,
            "config_file_sha256": sha256_file(output_dir / config_file),
            "matrix_key": json.dumps(run.analysis["matrix_key"], separators=(",", ":")),
            "protocol_hash": run.analysis["protocol_hash"],
        })
    return rows


def generate(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    stage: str = "formal",
) -> list[dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    if stage not in {"formal", "pilot"}:
        raise ValueError("stage must be formal or pilot")
    if stage == "pilot" and protocol["protocol_version"] != 3:
        raise ValueError("Pilot matrix generation through this mode requires protocol v3")
    runs = (
        generate_v3_pilot_matrix(protocol)
        if stage == "pilot"
        else generate_run_matrix(protocol)
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_yaml_names = {f"{run['run_id']}.yaml" for run in runs}
    for path in output_dir.glob("*.yaml"):
        if path.name not in expected_yaml_names:
            path.unlink()
    resolved_runs: list[RunConfig] = []
    for run in runs:
        # Validate before writing so a partially generated matrix cannot hide a typo.
        resolved = validate_run_mapping(run, protocol)
        _write_yaml(output_dir / f"{run['run_id']}.yaml", resolved.as_dict())
        resolved_runs.append(resolved)
    rows = _manifest_rows(resolved_runs, output_dir)
    fields = list(rows[0])
    manifest_csv = output_dir / "run_manifest.csv"
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(output_dir / "matrix_manifest.json", {
        "protocol": _protocol_reference(protocol_path),
        "protocol_hash": stable_hash(protocol),
        "run_count": len(runs),
        "conditions": list(active_conditions_for_protocol(protocol)),
        "seeds": list(dict.fromkeys(run.runtime.seed for run in resolved_runs)),
        "matrix_hash": stable_hash(runs),
        "runs": rows,
    })
    return runs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument(
        "--output",
        help=(
            "Generated matrix directory. Defaults to configs/generated for legacy "
            "protocols and to artifact_paths.formal_matrix for protocol v3."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("formal", "pilot"),
        default="formal",
        help="Generate the formal matrix or the protocol-v3 non-reportable pilot matrix",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate every protocol-defined run without writing files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(args.protocol)
    matrix_path_key = "pilot_matrix" if args.stage == "pilot" else "formal_matrix"
    if args.output is None:
        output = (
            PROJECT_ROOT / artifact_paths_for_protocol(protocol)[matrix_path_key]
            if protocol["protocol_version"] == 3
            else PROJECT_ROOT / "configs" / "generated"
        )
    else:
        output = Path(args.output)
    if protocol["protocol_version"] == 3:
        expected_output = (
            PROJECT_ROOT / artifact_paths_for_protocol(protocol)[matrix_path_key]
        ).resolve()
        if output.resolve() != expected_output:
            raise SystemExit(
                f"Protocol v3 {args.stage} generation must use its isolated matrix path: "
                f"{expected_output}"
            )
    if args.dry_run:
        runs = (
            generate_v3_pilot_matrix(protocol)
            if args.stage == "pilot"
            else generate_run_matrix(protocol)
        )
        for run in runs:
            validate_run_mapping(run, protocol)
        print(f"Validated {len(runs)} unique configurations; no files written")
        return 0
    runs = generate(args.protocol, output, stage=args.stage)
    print(f"Generated {len(runs)} unique configurations in {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
