#!/usr/bin/env python3
"""Generate the pre-specified KBS Tables 4--6 from seed-level outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.aggregate import (  # noqa: E402
    load_benchmark_results,
    load_run_manifest,
    load_seed_metrics,
    write_analysis_tables,
)
from talif_msresnet.config import (  # noqa: E402
    PRESPECIFIED_BOOTSTRAP,
    PRESPECIFIED_EFFICIENCY_COMPARISON,
    PRESPECIFIED_PRIMARY_INTERACTION_TEST,
    load_protocol,
)
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.statistics import (  # noqa: E402
    CONFIRMATORY_ALPHA,
    CONFIRMATORY_CI_LEVEL,
    CONFIRMATORY_SEEDS,
    INTERACTION_SESOI_PP,
    PRIMARY_ACCURACY_GROUPS,
    SENSITIVITY_BOOTSTRAP_RESAMPLES,
    SENSITIVITY_BOOTSTRAP_SEED,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze complete paired seed blocks and write Tables 4, 5, and 6."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="A consolidated seed_metrics.csv or a run-results directory.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory for table4_accuracy.csv, table5_diagnostics.csv, and table6_efficiency.csv.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional run_manifest.csv/JSON path; defaults to discovery under --input.",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=None,
        help="Optional benchmark_results.csv/path; defaults to discovery under --input.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "protocol.yaml",
        help="Frozen protocol containing the practical interaction threshold.",
    )
    return parser


def _optional_auxiliary_source(explicit: Path | None, input_path: Path) -> Path | None:
    """Discover manifest/benchmark files only when --input is a directory."""

    if explicit is not None:
        return explicit
    return input_path if input_path.is_dir() else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    preflight = check_protocol(
        args.protocol, mode="final-test", project_root=REPOSITORY_ROOT, check_dependencies=False
    )
    if not preflight.ok:
        raise SystemExit(
            "Analysis requires the frozen protocol:\n"
            + "\n".join(f"- {item}" for item in preflight.errors)
        )
    protocol = load_protocol(args.protocol)
    analysis_protocol = protocol["analysis"]
    practical_threshold = analysis_protocol["interaction_practical_threshold_pp"]
    efficiency_margins = analysis_protocol["efficiency_noninferiority_margins"]
    energy_model = protocol["benchmark"]["energy_model"]
    bootstrap_protocol = analysis_protocol["bootstrap"]
    bootstrap_resamples = int(bootstrap_protocol["resamples"])
    bootstrap_seed = int(bootstrap_protocol["seed"])
    frozen_contract = {
        "alpha": (float(analysis_protocol["alpha"]), CONFIRMATORY_ALPHA),
        "confidence_level": (
            float(analysis_protocol["confidence_level"]), CONFIRMATORY_CI_LEVEL
        ),
        "interaction_practical_threshold_pp": (
            float(practical_threshold), INTERACTION_SESOI_PP
        ),
        "bootstrap.resamples": (
            bootstrap_resamples, SENSITIVITY_BOOTSTRAP_RESAMPLES
        ),
        "bootstrap.seed": (bootstrap_seed, SENSITIVITY_BOOTSTRAP_SEED),
    }
    mismatches = [
        f"{name}: protocol={observed!r}, code={expected!r}"
        for name, (observed, expected) in frozen_contract.items()
        if observed != expected
    ]
    if analysis_protocol.get("primary_interaction_test") != PRESPECIFIED_PRIMARY_INTERACTION_TEST:
        mismatches.append(
            "primary_interaction_test differs from the seed-level one-sided SESOI contract"
        )
    if analysis_protocol.get("multiplicity_correction") != "holm_two_confirmatory_tests":
        mismatches.append("multiplicity_correction must be holm_two_confirmatory_tests")
    if analysis_protocol.get("multiplicity_family") != "two_confirmatory_interaction_tests":
        mismatches.append("multiplicity_family must be two_confirmatory_interaction_tests")
    if bootstrap_protocol != PRESPECIFIED_BOOTSTRAP:
        mismatches.append("bootstrap differs from the complete-seed-block sensitivity contract")
    if analysis_protocol.get("efficiency_comparison") != PRESPECIFIED_EFFICIENCY_COMPARISON:
        mismatches.append("efficiency_comparison differs from the paired log-ratio contract")
    if mismatches:
        raise SystemExit(
            "Frozen protocol and confirmatory analysis code disagree:\n- "
            + "\n- ".join(mismatches)
        )
    metrics = load_seed_metrics(args.input)
    if "protocol_hash" not in metrics.columns:
        raise SystemExit("Seed metrics do not contain protocol_hash")
    result_protocol_hashes = {
        str(value).strip() for value in metrics["protocol_hash"].dropna() if str(value).strip()
    }
    if result_protocol_hashes != {preflight.protocol_hash}:
        raise SystemExit(
            "Seed metrics are not uniquely bound to the current frozen protocol: "
            f"{sorted(result_protocol_hashes)}"
        )
    group_columns = {"experiment", "dataset", "depth", "time_steps"}
    if not group_columns.issubset(metrics.columns):
        raise SystemExit(
            "Seed metrics are missing confirmatory-group columns: "
            f"{sorted(group_columns - set(metrics.columns))}"
        )
    primary = metrics.loc[metrics["experiment"].astype(str).str.upper() == "E1"]
    observed_groups = {
        (
            str(row.experiment).upper(),
            str(row.dataset).lower().replace("-", "").replace("_", ""),
            int(row.depth),
            int(row.time_steps),
        )
        for row in primary.itertuples(index=False)
    }
    expected_groups = {
        (
            str(slot["experiment"]),
            str(slot["dataset"]),
            int(slot["depth"]),
            int(slot["time_steps"]),
        )
        for slot in protocol["matrix"]["primary"]
    }
    if observed_groups != expected_groups:
        raise SystemExit(
            "Seed metrics do not contain exactly the two frozen confirmatory groups: "
            f"observed={sorted(observed_groups)} expected={sorted(expected_groups)}"
        )
    manifest_source = _optional_auxiliary_source(args.manifest, args.input)
    benchmark_source = _optional_auxiliary_source(args.benchmark, args.input)
    manifest = (
        load_run_manifest(manifest_source)
        if manifest_source is not None
        else metrics.iloc[0:0].copy()
    )
    benchmark = (
        load_benchmark_results(benchmark_source)
        if benchmark_source is not None
        else metrics.iloc[0:0].copy()
    )
    paths = write_analysis_tables(
        metrics,
        args.output,
        manifest=manifest if not manifest.empty else None,
        benchmark=benchmark if not benchmark.empty else None,
        efficiency_margins=efficiency_margins,
        expected_energy_constants_sha256=energy_model["constants_sha256"],
    )
    metadata = {
        "input": artifact_path_reference(args.input, REPOSITORY_ROOT),
        "manifest": (
            artifact_path_reference(manifest_source, REPOSITORY_ROOT)
            if manifest_source is not None
            else None
        ),
        "benchmark": (
            artifact_path_reference(benchmark_source, REPOSITORY_ROOT)
            if benchmark_source is not None
            else None
        ),
        "n_seed_metric_rows": int(len(metrics)),
        "n_manifest_rows": int(len(manifest)),
        "n_benchmark_rows": int(len(benchmark)),
        "alpha": analysis_protocol["alpha"],
        "confidence_level": analysis_protocol["confidence_level"],
        "primary_interaction_test": analysis_protocol["primary_interaction_test"],
        "multiplicity_correction": analysis_protocol["multiplicity_correction"],
        "multiplicity_family": analysis_protocol["multiplicity_family"],
        "confirmatory_groups": protocol["matrix"]["primary"],
        "confirmatory_seeds": list(CONFIRMATORY_SEEDS),
        "code_primary_accuracy_groups": [list(group) for group in PRIMARY_ACCURACY_GROUPS],
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_role": bootstrap_protocol["role"],
        "bootstrap_resampling_unit": bootstrap_protocol["resampling_unit"],
        "bootstrap_interval": bootstrap_protocol["interval"],
        "protocol": artifact_path_reference(args.protocol, REPOSITORY_ROOT),
        "protocol_hash": preflight.protocol_hash,
        "interaction_practical_threshold_pp": practical_threshold,
        "efficiency_assessment": analysis_protocol["efficiency_assessment"],
        "efficiency_confidence_level": analysis_protocol["efficiency_confidence_level"],
        "efficiency_comparison": analysis_protocol["efficiency_comparison"],
        "wording_gates": analysis_protocol["wording_gates"],
        "efficiency_noninferiority_margins": efficiency_margins,
        "energy_model": energy_model,
        "tables": {
            name: artifact_path_reference(path, REPOSITORY_ROOT)
            for name, path in paths.items()
        },
    }
    metadata_path = args.output / "analysis_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
