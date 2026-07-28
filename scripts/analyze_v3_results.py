#!/usr/bin/env python3
"""Analyze only the frozen TA-LIF-only v3 C1/C2 formal results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.aggregate import load_seed_metrics  # noqa: E402
from talif_msresnet.config import load_protocol  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.statistics_v3 import (  # noqa: E402
    V3_ALPHA,
    V3_BOOTSTRAP_RESAMPLES,
    V3_BOOTSTRAP_SEEDS,
    V3_CONDITIONS,
    V3_CONFIDENCE_LEVEL,
    V3_DATASET_GROUPS,
    V3_FORMAL_SEEDS,
    V3_PRIMARY_DATASET,
    V3_REPLICATION_DATASET,
    V3_SIGN_FLIP_ASSIGNMENTS,
    analyze_v3_paired_accuracy,
)
from talif_msresnet.utils import atomic_write_json, sha256_file, stable_hash  # noqa: E402


V3_PROTOCOL_REFERENCE = "configs/protocol_v3_talif_only.yaml"
V3_FORMAL_RESULTS_REFERENCE = "results/formal_v3_talif_only"
V3_ANALYSIS_RESULTS_REFERENCE = "results/analysis/v3_talif_only"
V3_ANALYSIS_JSON = "v3_paired_accuracy_analysis.json"
V3_RAW_DIFFERENCES_CSV = "v3_paired_seed_differences.csv"

_EXPECTED_ARTIFACT_PATHS: Mapping[str, str] = {
    "protocol": V3_PROTOCOL_REFERENCE,
    "signoff": "PREREGISTRATION_SIGNOFF_V3_TALIF_ONLY.md",
    "formal_matrix": "configs/v3_talif_only_generated",
    "pilot_matrix": "configs/v3_talif_only_pilot_generated",
    "freeze_manifest": "FREEZE_MANIFEST_V3_TALIF_ONLY.json",
    "formal_results": V3_FORMAL_RESULTS_REFERENCE,
    "pilot_results": "results/pilot/v3_talif_only",
    "analysis_results": V3_ANALYSIS_RESULTS_REFERENCE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated v3 paired C2-C1 accuracy analysis."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / V3_PROTOCOL_REFERENCE,
        help="Frozen TA-LIF-only v3 protocol; no legacy protocol is accepted.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Exact v3 formal-results root; defaults to the protocol-bound path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact v3 analysis-results root; defaults to the protocol-bound path.",
    )
    return parser


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def validate_v3_protocol_contract(protocol: Mapping[str, Any]) -> None:
    """Fail closed when executable analysis and frozen protocol disagree."""

    failures: list[str] = []

    def require(label: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            failures.append(f"{label}: protocol={observed!r}, code={expected!r}")

    require("protocol_version", protocol.get("protocol_version"), 3)
    require("active_conditions", protocol.get("active_conditions"), list(V3_CONDITIONS))
    require("seeds", protocol.get("seeds"), list(V3_FORMAL_SEEDS))
    require("output_root", protocol.get("output_root"), V3_FORMAL_RESULTS_REFERENCE)
    require("artifact_paths", protocol.get("artifact_paths"), dict(_EXPECTED_ARTIFACT_PATHS))

    expected_matrix = [
        {
            "experiment": "E1",
            "dataset": dataset,
            "depth": depth,
            "time_steps": time_steps,
        }
        for dataset, (depth, time_steps) in V3_DATASET_GROUPS.items()
    ]
    matrix = _mapping(protocol.get("matrix"), "protocol.matrix")
    require("matrix.primary", matrix.get("primary"), expected_matrix)

    analysis = _mapping(protocol.get("analysis"), "protocol.analysis")
    require("analysis.accuracy_scale", analysis.get("accuracy_scale"), "proportion")
    require("analysis.alpha", analysis.get("alpha"), V3_ALPHA)
    require(
        "analysis.confidence_level",
        analysis.get("confidence_level"),
        V3_CONFIDENCE_LEVEL,
    )
    require("analysis.primary_dataset", analysis.get("primary_dataset"), V3_PRIMARY_DATASET)
    require(
        "analysis.replication_dataset",
        analysis.get("replication_dataset"),
        V3_REPLICATION_DATASET,
    )

    primary = _mapping(analysis.get("primary_accuracy_test"), "analysis.primary_accuracy_test")
    primary_contract = {
        "estimand": "mean_seed_paired_c2_minus_c1_test_accuracy_pp",
        "blocking_factor": "seed",
        "test_statistic": "one_sample_t_over_seed_level_differences",
        "null_hypothesis": "mean_delta_le_0",
        "alternative_hypothesis": "mean_delta_gt_0",
        "sidedness": "one_sided_greater",
        "estimation_interval": "two_sided_95_percent_t",
        "degrees_of_freedom": 4,
        "decision_rule": "p_lt_0_05_and_mean_delta_gt_0",
        "normality_pretest_switch": "forbidden",
        "zero_variance_rule": "not_estimable_primary_inconclusive",
    }
    require("analysis.primary_accuracy_test", primary, primary_contract)

    replication = _mapping(
        analysis.get("replication_analysis"), "analysis.replication_analysis"
    )
    require("analysis.replication_analysis.cannot_rescue_primary", replication.get(
        "cannot_rescue_primary"
    ), True)
    require(
        "analysis.replication_analysis.cross_dataset_multiplicity",
        replication.get("cross_dataset_multiplicity"),
        "none_single_confirmatory_primary",
    )

    sign_flip = _mapping(analysis.get("sign_flip"), "analysis.sign_flip")
    require("analysis.sign_flip.assignments", sign_flip.get("assignments"), V3_SIGN_FLIP_ASSIGNMENTS)
    require(
        "analysis.sign_flip.enumeration",
        sign_flip.get("enumeration"),
        "all_2_power_5_seed_level_sign_flips",
    )
    require("analysis.sign_flip.sidedness", sign_flip.get("sidedness"), "one_sided_greater")
    require(
        "analysis.sign_flip.role",
        sign_flip.get("role"),
        "sensitivity_only_cannot_rescue_primary",
    )

    bootstrap = _mapping(analysis.get("bootstrap"), "analysis.bootstrap")
    bootstrap_contract = {
        "resamples": V3_BOOTSTRAP_RESAMPLES,
        "resampling_unit": "complete_seed_pair",
        "cellwise_resampling": "forbidden",
        "rng": "numpy_generator_pcg64",
        "index_draw": "integers_0_5_size_10000_by_5_endpoint_false",
        "statistic": "mean_seed_paired_c2_minus_c1_test_accuracy_pp",
        "interval": "two_sided_95_percent_percentile",
        "quantile_method": "linear",
        "seeds": dict(V3_BOOTSTRAP_SEEDS),
        "role": "sensitivity_only_cannot_rescue_primary",
    }
    require("analysis.bootstrap", bootstrap, bootstrap_contract)

    run_handling = _mapping(analysis.get("run_handling"), "analysis.run_handling")
    require(
        "analysis.run_handling.incomplete_seed_pair",
        run_handling.get("incomplete_seed_pair"),
        "unresolved_no_partial_analysis",
    )
    require(
        "analysis.run_handling.seed_substitution",
        run_handling.get("seed_substitution"),
        "forbidden",
    )
    require(
        "analysis.run_handling.pilot_in_reportable_analysis",
        run_handling.get("pilot_in_reportable_analysis"),
        "forbidden",
    )

    wording = _mapping(analysis.get("wording_gates"), "analysis.wording_gates")
    wording_contract = {
        "cifar100_improved": "primary_p_lt_0_05_and_mean_delta_gt_0",
        "primary_not_passed": "inconclusive_no_equivalence_or_no_effect_claim",
        "dvs_cannot_rescue_primary": True,
        "dvs_directional_support": "mean_delta_gt_0",
        "dvs_replication_claim": "one_sided_p_lt_0_05_and_mean_delta_gt_0",
        "cross_domain_improvement": "both_dataset_tests_pass_and_both_means_gt_0",
    }
    require("analysis.wording_gates", wording, wording_contract)

    if failures:
        raise ValueError(
            "Frozen v3 protocol and paired-analysis code disagree:\n- "
            + "\n- ".join(failures)
        )


def isolated_analysis_paths(
    protocol_path: Path,
    protocol: Mapping[str, Any],
    input_path: Path | None,
    output_path: Path | None,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> tuple[Path, Path]:
    """Resolve and enforce the dedicated v3 protocol/input/output locations."""

    root = repository_root.resolve()
    expected_protocol = (root / V3_PROTOCOL_REFERENCE).resolve()
    if protocol_path.resolve() != expected_protocol:
        raise ValueError(
            f"v3 analysis requires protocol {expected_protocol}; got {protocol_path.resolve()}"
        )
    paths = _mapping(protocol.get("artifact_paths"), "protocol.artifact_paths")
    expected_input = (root / str(paths["formal_results"])).resolve()
    expected_output = (root / str(paths["analysis_results"])).resolve()
    actual_input = expected_input if input_path is None else input_path.resolve()
    actual_output = expected_output if output_path is None else output_path.resolve()
    if actual_input != expected_input:
        raise ValueError(f"v3 input must be the isolated formal-results root {expected_input}")
    if actual_output != expected_output:
        raise ValueError(f"v3 output must be the isolated analysis-results root {expected_output}")
    if actual_input == actual_output:
        raise ValueError("v3 formal results and analysis output must be isolated directories")
    return actual_input, actual_output


def _same_number(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def audit_v3_final_test_artifacts(
    metrics: Any,
    input_root: Path,
    *,
    expected_protocol_hash: str,
) -> list[dict[str, str]]:
    """Bind every analyzed row to one committed final-test transaction."""

    expected_run_ids = set(metrics["run_id"].astype(str))
    observed_run_ids = {
        path.name
        for path in input_root.iterdir()
        if path.is_dir() and path.name.startswith("E1_")
    }
    if observed_run_ids != expected_run_ids:
        raise ValueError(
            "The isolated v3 result root contains a different E1 run-directory set: "
            f"missing={sorted(expected_run_ids - observed_run_ids)}, "
            f"extra={sorted(observed_run_ids - expected_run_ids)}"
        )

    evidence: list[dict[str, str]] = []
    for row in metrics.sort_values(["dataset", "seed", "condition"]).itertuples(index=False):
        run_id = str(row.run_id)
        run_dir = input_root / run_id
        required = {
            "seed_metrics": run_dir / "seed_metrics.json",
            "run_manifest": run_dir / "run_manifest.json",
            "checkpoint": run_dir / "best.pt",
            "final_test": run_dir / "final_test.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise ValueError(f"{run_id}: missing final-analysis artifacts {missing}")
        transaction_files = [
            path.name
            for path in (run_dir / "final_test.in_progress.json", run_dir / "final_test.lock")
            if path.exists()
        ]
        if transaction_files:
            raise ValueError(
                f"{run_id}: unresolved final-test transaction files {transaction_files}"
            )

        per_run = json.loads(required["seed_metrics"].read_text(encoding="utf-8"))
        manifest = json.loads(required["run_manifest"].read_text(encoding="utf-8"))
        marker = json.loads(required["final_test"].read_text(encoding="utf-8"))
        if not all(isinstance(value, Mapping) for value in (per_run, manifest, marker)):
            raise ValueError(f"{run_id}: final-analysis JSON artifact is not a mapping")

        checkpoint_hash = sha256_file(required["checkpoint"])
        expected_text = {
            "run_id": run_id,
            "config_hash": str(row.config_hash),
            "protocol_hash": expected_protocol_hash,
            "checkpoint_sha256": str(row.test_checkpoint_sha256),
            "evaluated_at": str(row.test_evaluated_at),
        }
        marker_text = {
            "run_id": marker.get("run_id"),
            "config_hash": marker.get("config_hash"),
            "protocol_hash": marker.get("protocol_hash"),
            "checkpoint_sha256": marker.get("checkpoint_sha256"),
            "evaluated_at": marker.get("evaluated_at"),
        }
        if marker.get("status") != "complete" or marker_text != expected_text:
            raise ValueError(f"{run_id}: final_test.json differs from consolidated metrics")
        if checkpoint_hash != expected_text["checkpoint_sha256"]:
            raise ValueError(f"{run_id}: best.pt SHA-256 differs from final-test evidence")
        if not _same_number(marker.get("test_accuracy"), row.test_accuracy):
            raise ValueError(f"{run_id}: final-test accuracy differs from consolidated metrics")

        per_run_checks = {
            "run_id": per_run.get("run_id") == run_id,
            "config_hash": per_run.get("config_hash") == str(row.config_hash),
            "protocol_hash": per_run.get("protocol_hash") == expected_protocol_hash,
            "test_checkpoint_sha256": (
                per_run.get("test_checkpoint_sha256") == checkpoint_hash
            ),
            "test_evaluated_at": per_run.get("test_evaluated_at") == str(row.test_evaluated_at),
            "test_accuracy": _same_number(per_run.get("test_accuracy"), row.test_accuracy),
        }
        failed_per_run = [name for name, passed in per_run_checks.items() if not passed]
        if failed_per_run:
            raise ValueError(
                f"{run_id}: per-run seed_metrics.json differs in {failed_per_run}"
            )

        final_test = manifest.get("final_test")
        if not isinstance(final_test, Mapping):
            raise ValueError(f"{run_id}: run_manifest.json lacks final_test evidence")
        manifest_checks = {
            "run_id": manifest.get("run_id") == run_id,
            "config_hash": manifest.get("config_hash") == str(row.config_hash),
            "status": final_test.get("status") == "complete",
            "checkpoint_sha256": final_test.get("checkpoint_sha256") == checkpoint_hash,
            "evaluated_at": final_test.get("evaluated_at") == str(row.test_evaluated_at),
        }
        failed_manifest = [name for name, passed in manifest_checks.items() if not passed]
        if failed_manifest:
            raise ValueError(
                f"{run_id}: run_manifest.json differs in {failed_manifest}"
            )

        evidence.append(
            {
                "run_id": run_id,
                "seed_metrics_sha256": sha256_file(required["seed_metrics"]),
                "run_manifest_sha256": sha256_file(required["run_manifest"]),
                "best_checkpoint_sha256": checkpoint_hash,
                "final_test_sha256": sha256_file(required["final_test"]),
            }
        )
    return evidence


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = (
        "dataset",
        "dataset_role",
        "seed",
        "c1_accuracy_pp",
        "c2_accuracy_pp",
        "c2_minus_c1_difference_pp",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    validate_v3_protocol_contract(protocol)
    input_path, output_path = isolated_analysis_paths(
        protocol_path,
        protocol,
        args.input,
        args.output,
    )
    report_path = output_path / V3_ANALYSIS_JSON
    raw_path = output_path / V3_RAW_DIFFERENCES_CSV
    existing = [path for path in (report_path, raw_path) if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing v3 analysis artifacts:\n- "
            + "\n- ".join(str(path) for path in existing)
        )

    preflight = check_protocol(
        protocol_path,
        mode="final-test",
        project_root=REPOSITORY_ROOT,
        check_dependencies=False,
    )
    if not preflight.ok:
        raise SystemExit(
            "v3 analysis requires a frozen final-test protocol:\n- "
            + "\n- ".join(preflight.errors)
        )
    expected_hash = stable_hash(protocol)
    if preflight.protocol_hash != expected_hash:
        raise SystemExit("Preflight protocol hash differs from the loaded v3 protocol")

    metrics = load_seed_metrics(input_path)
    result = analyze_v3_paired_accuracy(
        metrics,
        expected_protocol_hash=expected_hash,
    )
    artifact_audit = audit_v3_final_test_artifacts(
        metrics,
        input_path,
        expected_protocol_hash=expected_hash,
    )
    raw_rows: list[dict[str, Any]] = []
    for dataset in (V3_PRIMARY_DATASET, V3_REPLICATION_DATASET):
        dataset_result = result.datasets[dataset]
        raw_rows.extend(
            {
                "dataset": dataset,
                "dataset_role": dataset_result.role,
                "seed": item.seed,
                "c1_accuracy_pp": item.c1_accuracy_pp,
                "c2_accuracy_pp": item.c2_accuracy_pp,
                "c2_minus_c1_difference_pp": item.difference_pp,
            }
            for item in dataset_result.raw_differences
        )

    source_artifacts = (
        sorted(set(metrics["source_file"].fillna("").astype(str)))
        if "source_file" in metrics.columns
        else []
    )
    payload = {
        "schema": "ta-lif-only-v3-paired-analysis-v1",
        "protocol_version": 3,
        "protocol": artifact_path_reference(protocol_path, REPOSITORY_ROOT),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_hash": expected_hash,
        "input_root": artifact_path_reference(input_path, REPOSITORY_ROOT),
        "input_source_artifacts": source_artifacts,
        "n_seed_metric_rows": int(len(metrics)),
        "input_artifact_audit": artifact_audit,
        "analysis": result.to_dict(),
        "raw_differences_csv": artifact_path_reference(raw_path, REPOSITORY_ROOT),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(raw_path, raw_rows)
    try:
        atomic_write_json(report_path, payload)
    except BaseException:
        raw_path.unlink(missing_ok=True)
        raise
    print(f"analysis: {report_path}")
    print(f"raw_differences: {raw_path}")
    print(f"primary_result: {result.claim_gates.primary_result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
