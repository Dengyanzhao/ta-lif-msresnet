#!/usr/bin/env python3
"""Analyze only the frozen TA-LIF-only v4 C1/C2 formal results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analyze_v3_results import (  # noqa: E402
    _atomic_write_csv,
    audit_v3_final_test_artifacts,
)
from talif_msresnet.aggregate import load_seed_metrics  # noqa: E402
from talif_msresnet.config import load_protocol  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.statistics_v4 import (  # noqa: E402
    V4_ALL_FORMAL_SEEDS,
    V4_ALPHA,
    V4_BOOTSTRAP_RESAMPLES,
    V4_BOOTSTRAP_SEEDS,
    V4_CONDITIONS,
    V4_CONFIDENCE_LEVEL,
    V4_DATASET_GROUPS,
    V4_FORMAL_SEEDS_BY_DATASET,
    V4_PRIMARY_DATASET,
    V4_REPLICATION_DATASET,
    V4_SIGN_FLIP_ASSIGNMENTS,
    analyze_v4_paired_accuracy,
)
from talif_msresnet.utils import atomic_write_json, sha256_file, stable_hash  # noqa: E402


V4_PROTOCOL_REFERENCE = "configs/protocol_v4_talif_only.yaml"
V4_FORMAL_RESULTS_REFERENCE = "results/formal_v4_talif_only"
V4_ANALYSIS_RESULTS_REFERENCE = "results/analysis/v4_talif_only"
V4_ANALYSIS_JSON = "v4_paired_accuracy_analysis.json"
V4_RAW_DIFFERENCES_CSV = "v4_paired_seed_differences.csv"

_EXPECTED_ARTIFACT_PATHS: Mapping[str, str] = {
    "protocol": V4_PROTOCOL_REFERENCE,
    "signoff": "PREREGISTRATION_SIGNOFF_V4_TALIF_ONLY.md",
    "formal_matrix": "configs/v4_talif_only_generated",
    "pilot_matrix": "configs/v4_talif_only_pilot_generated",
    "freeze_manifest": "FREEZE_MANIFEST_V4_TALIF_ONLY.json",
    "formal_results": V4_FORMAL_RESULTS_REFERENCE,
    "pilot_results": "results/pilot/v4_talif_only",
    "analysis_results": V4_ANALYSIS_RESULTS_REFERENCE,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated v4 paired C2-C1 accuracy analysis."
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / V4_PROTOCOL_REFERENCE,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Exact v4 formal-results root; defaults to the protocol-bound path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exact v4 analysis-results root; defaults to the protocol-bound path.",
    )
    return parser


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def validate_v4_protocol_contract(protocol: Mapping[str, Any]) -> None:
    """Fail closed when executable analysis and the frozen v4 design disagree."""

    failures: list[str] = []

    def require(label: str, observed: Any, expected: Any) -> None:
        if observed != expected:
            failures.append(f"{label}: protocol={observed!r}, code={expected!r}")

    require("protocol_version", protocol.get("protocol_version"), 4)
    require("active_conditions", protocol.get("active_conditions"), list(V4_CONDITIONS))
    require("seeds", protocol.get("seeds"), list(V4_ALL_FORMAL_SEEDS))
    require("output_root", protocol.get("output_root"), V4_FORMAL_RESULTS_REFERENCE)
    require("artifact_paths", protocol.get("artifact_paths"), dict(_EXPECTED_ARTIFACT_PATHS))

    expected_matrix = [
        {
            "experiment": "E1",
            "dataset": dataset,
            "depth": depth,
            "time_steps": time_steps,
            "seeds": list(V4_FORMAL_SEEDS_BY_DATASET[dataset]),
        }
        for dataset, (depth, time_steps) in V4_DATASET_GROUPS.items()
    ]
    matrix = _mapping(protocol.get("matrix"), "protocol.matrix")
    require("matrix.primary", matrix.get("primary"), expected_matrix)

    analysis = _mapping(protocol.get("analysis"), "protocol.analysis")
    require("analysis.collect_activity", analysis.get("collect_activity"), False)
    require("analysis.accuracy_scale", analysis.get("accuracy_scale"), "proportion")
    require(
        "analysis.validation_accuracy_thresholds",
        analysis.get("validation_accuracy_thresholds"),
        {"cifar100": 0.60, "cifar10dvs": 0.60},
    )
    require(
        "analysis.convergence_epoch_rule",
        analysis.get("convergence_epoch_rule"),
        "first_validation_epoch_at_or_above_threshold",
    )
    require(
        "analysis.nonconvergence_rule",
        analysis.get("nonconvergence_rule"),
        "not_reached_by_final_epoch_no_exclusion",
    )
    require("analysis.alpha", analysis.get("alpha"), V4_ALPHA)
    require(
        "analysis.confidence_level",
        analysis.get("confidence_level"),
        V4_CONFIDENCE_LEVEL,
    )
    require("analysis.primary_dataset", analysis.get("primary_dataset"), V4_PRIMARY_DATASET)
    require(
        "analysis.replication_dataset",
        analysis.get("replication_dataset"),
        V4_REPLICATION_DATASET,
    )

    primary = _mapping(analysis.get("primary_accuracy_test"), "analysis.primary_accuracy_test")
    require(
        "analysis.primary_accuracy_test",
        primary,
        {
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
        },
    )

    replication = _mapping(
        analysis.get("replication_analysis"), "analysis.replication_analysis"
    )
    require(
        "analysis.replication_analysis",
        replication,
        {
            "method": "same_paired_t_estimate_interval_and_raw_differences",
            "role": "prespecified_replication_external_validity_not_confirmatory",
            "cross_dataset_multiplicity": "none_single_confirmatory_primary",
            "cannot_rescue_primary": True,
        },
    )

    sign_flip = _mapping(analysis.get("sign_flip"), "analysis.sign_flip")
    require(
        "analysis.sign_flip",
        sign_flip,
        {
            "assignments": V4_SIGN_FLIP_ASSIGNMENTS,
            "enumeration": "all_2_power_5_seed_level_sign_flips",
            "sidedness": "one_sided_greater",
            "role": "sensitivity_only_cannot_rescue_primary",
        },
    )

    bootstrap = _mapping(analysis.get("bootstrap"), "analysis.bootstrap")
    require(
        "analysis.bootstrap",
        bootstrap,
        {
            "resamples": V4_BOOTSTRAP_RESAMPLES,
            "resampling_unit": "complete_seed_pair",
            "cellwise_resampling": "forbidden",
            "rng": "numpy_generator_pcg64",
            "index_draw": "integers_0_5_size_10000_by_5_endpoint_false",
            "statistic": "mean_seed_paired_c2_minus_c1_test_accuracy_pp",
            "interval": "two_sided_95_percent_percentile",
            "quantile_method": "linear",
            "seeds": dict(V4_BOOTSTRAP_SEEDS),
            "role": "sensitivity_only_cannot_rescue_primary",
        },
    )

    model_selection = _mapping(
        analysis.get("model_selection"), "analysis.model_selection"
    )
    require(
        "analysis.model_selection",
        model_selection,
        {
            "checkpoint": "best.pt",
            "primary_order": "maximum_validation_accuracy",
            "first_tie_breaker": "minimum_validation_loss",
            "exact_tie_breaker": "earliest_epoch",
            "test_based_selection": "forbidden",
        },
    )

    test_access = _mapping(analysis.get("test_access"), "analysis.test_access")
    require(
        "analysis.test_access",
        test_access,
        {
            "training_config_final_test": False,
            "when": "after_all_20_runs_and_best_checkpoints_pass_frozen_audit",
            "access_count_per_checkpoint": 1,
            "reselection_or_repeated_evaluation": "forbidden",
            "ambiguous_interruption": "stop_document_and_obtain_author_decision",
        },
    )

    run_handling = _mapping(analysis.get("run_handling"), "analysis.run_handling")
    require(
        "analysis.run_handling",
        run_handling,
        {
            "pilot_in_reportable_analysis": "forbidden",
            "failed_run_records": "retain",
            "seed_substitution": "forbidden",
            "outlier_exclusion": "forbidden",
            "nonconvergence_excludes_run": False,
            "incomplete_seed_pair": "unresolved_no_partial_analysis",
            "cross_environment_checkpoint_resume": "forbidden",
            "protocol_change_after_first_reportable_run": "forbidden",
        },
    )

    wording = _mapping(analysis.get("wording_gates"), "analysis.wording_gates")
    require(
        "analysis.wording_gates",
        wording,
        {
            "cifar100_improved": "primary_p_lt_0_05_and_mean_delta_gt_0",
            "primary_not_passed": "inconclusive_no_equivalence_or_no_effect_claim",
            "dvs_cannot_rescue_primary": True,
            "dvs_directional_support": "mean_delta_gt_0",
            "dvs_replication_claim": "one_sided_p_lt_0_05_and_mean_delta_gt_0",
            "cross_domain_improvement": "both_dataset_tests_pass_and_both_means_gt_0",
        },
    )

    if failures:
        raise ValueError(
            "Frozen v4 protocol and paired-analysis code disagree:\n- "
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
    root = repository_root.resolve()
    expected_protocol = (root / V4_PROTOCOL_REFERENCE).resolve()
    if protocol_path.resolve() != expected_protocol:
        raise ValueError(
            f"v4 analysis requires protocol {expected_protocol}; got {protocol_path.resolve()}"
        )
    paths = _mapping(protocol.get("artifact_paths"), "protocol.artifact_paths")
    expected_input = (root / str(paths["formal_results"])).resolve()
    expected_output = (root / str(paths["analysis_results"])).resolve()
    actual_input = expected_input if input_path is None else input_path.resolve()
    actual_output = expected_output if output_path is None else output_path.resolve()
    if actual_input != expected_input:
        raise ValueError(f"v4 input must be the isolated formal-results root {expected_input}")
    if actual_output != expected_output:
        raise ValueError(f"v4 output must be the isolated analysis-results root {expected_output}")
    if actual_input == actual_output:
        raise ValueError("v4 formal results and analysis output must be isolated directories")
    return actual_input, actual_output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    validate_v4_protocol_contract(protocol)
    input_path, output_path = isolated_analysis_paths(
        protocol_path,
        protocol,
        args.input,
        args.output,
    )
    report_path = output_path / V4_ANALYSIS_JSON
    raw_path = output_path / V4_RAW_DIFFERENCES_CSV
    existing = [path for path in (report_path, raw_path) if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing v4 analysis artifacts:\n- "
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
            "v4 analysis requires a frozen final-test protocol:\n- "
            + "\n- ".join(preflight.errors)
        )
    expected_hash = stable_hash(protocol)
    if preflight.protocol_hash != expected_hash:
        raise SystemExit("Preflight protocol hash differs from the loaded v4 protocol")

    metrics = load_seed_metrics(input_path)
    result = analyze_v4_paired_accuracy(
        metrics,
        expected_protocol_hash=expected_hash,
    )
    # The v3-named helper audits only version-neutral final-test transactions.
    artifact_audit = audit_v3_final_test_artifacts(
        metrics,
        input_path,
        expected_protocol_hash=expected_hash,
    )
    raw_rows: list[dict[str, Any]] = []
    for dataset in (V4_PRIMARY_DATASET, V4_REPLICATION_DATASET):
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
        "schema": "ta-lif-only-v4-paired-analysis-v1",
        "protocol_version": 4,
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
