#!/usr/bin/env python
"""Generate the protocol-bound run configuration matrix."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tempfile
import uuid
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
    generate_v4_pilot_matrix,
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


def _matrix_runs(
    protocol: Mapping[str, Any],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    if stage == "formal":
        return generate_run_matrix(protocol)
    version = int(protocol["protocol_version"])
    if version == 3:
        return generate_v3_pilot_matrix(protocol)
    if version == 4:
        return generate_v4_pilot_matrix(protocol)
    raise ValueError(
        "Pilot matrix generation through this mode requires protocol v3 or v4"
    )


def _expected_matrix_dir(
    protocol: Mapping[str, Any],
    *,
    stage: str,
) -> Path:
    matrix_path_key = "pilot_matrix" if stage == "pilot" else "formal_matrix"
    return (PROJECT_ROOT / artifact_paths_for_protocol(protocol)[matrix_path_key]).resolve()


def _write_matrix_artifacts(
    protocol_path: str | Path,
    protocol: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    resolved_runs: Sequence[RunConfig],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_yaml_names = {f"{run['run_id']}.yaml" for run in runs}
    for path in output_dir.glob("*.yaml"):
        if path.name not in expected_yaml_names:
            path.unlink()
    for resolved in resolved_runs:
        _write_yaml(output_dir / f"{resolved.runtime.run_id}.yaml", resolved.as_dict())
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


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_staged_directory(staging_dir: Path, output_dir: Path) -> None:
    backup_dir: Path | None = None
    if output_dir.exists() or output_dir.is_symlink():
        backup_dir = output_dir.with_name(
            f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        )
        output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except BaseException:
        if backup_dir is not None:
            backup_dir.rename(output_dir)
        raise
    if backup_dir is not None:
        _remove_path(backup_dir)


def _generate_v4_transactionally(
    protocol_path: str | Path,
    protocol: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    resolved_runs: Sequence[RunConfig],
    output_dir: Path,
) -> None:
    if not output_dir.parent.is_dir():
        raise ValueError(
            f"The isolated v4 matrix parent directory does not exist: {output_dir.parent}"
        )
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=str(output_dir.parent),
        )
    )
    try:
        _write_matrix_artifacts(
            protocol_path,
            protocol,
            runs,
            resolved_runs,
            staging_dir,
        )
        _publish_staged_directory(staging_dir, output_dir)
    finally:
        if staging_dir.exists() or staging_dir.is_symlink():
            _remove_path(staging_dir)


def generate(
    protocol_path: str | Path,
    output_dir: str | Path,
    *,
    stage: str = "formal",
) -> list[dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    if stage not in {"formal", "pilot"}:
        raise ValueError("stage must be formal or pilot")
    output_dir = Path(output_dir)
    if protocol["protocol_version"] == 4:
        expected_output = _expected_matrix_dir(protocol, stage=stage)
        if output_dir.resolve() != expected_output:
            raise ValueError(
                f"Protocol v4 {stage} generation must use its isolated matrix path: "
                f"{expected_output}"
            )
    runs = _matrix_runs(protocol, stage=stage)
    resolved_runs = [validate_run_mapping(run, protocol) for run in runs]
    if protocol["protocol_version"] == 4:
        expected_count = 4 if stage == "pilot" else 20
        if len(resolved_runs) != expected_count:
            raise ValueError(
                f"Protocol v4 {stage} matrix must contain exactly {expected_count} runs"
            )
        _generate_v4_transactionally(
            protocol_path,
            protocol,
            runs,
            resolved_runs,
            output_dir,
        )
        return runs

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_yaml_names = {f"{run['run_id']}.yaml" for run in runs}
    for path in output_dir.glob("*.yaml"):
        if path.name not in expected_yaml_names:
            path.unlink()
    for resolved in resolved_runs:
        _write_yaml(output_dir / f"{resolved.runtime.run_id}.yaml", resolved.as_dict())
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
            "protocols and to the isolated artifact_paths matrix for protocols v3/v4."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("formal", "pilot"),
        default="formal",
        help="Generate the formal matrix or a protocol-v3/v4 non-reportable pilot matrix",
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
            if protocol["protocol_version"] in (3, 4)
            else PROJECT_ROOT / "configs" / "generated"
        )
    else:
        output = Path(args.output)
    if protocol["protocol_version"] in (3, 4):
        expected_output = (
            PROJECT_ROOT / artifact_paths_for_protocol(protocol)[matrix_path_key]
        ).resolve()
        if output.resolve() != expected_output:
            raise SystemExit(
                f"Protocol v{protocol['protocol_version']} {args.stage} generation "
                "must use its isolated matrix path: "
                f"{expected_output}"
            )
    if args.dry_run:
        runs = _matrix_runs(protocol, stage=args.stage)
        for run in runs:
            validate_run_mapping(run, protocol)
        print(f"Validated {len(runs)} unique configurations; no files written")
        return 0
    runs = generate(args.protocol, output, stage=args.stage)
    print(f"Generated {len(runs)} unique configurations in {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
