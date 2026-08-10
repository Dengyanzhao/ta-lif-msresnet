#!/usr/bin/env python3
"""Analyze only the frozen V6 six-condition CIFAR-100 formal results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPOSITORY_ROOT / "src"
SRC_PATH = str(SRC_ROOT)
if SRC_PATH in sys.path:
    sys.path.remove(SRC_PATH)
sys.path.insert(0, SRC_PATH)

from talif_msresnet.config import (
    artifact_paths_for_protocol,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.config_v6 import (
    V6_ACTIVE_CONDITIONS,
    V6_ANALYSIS_CONTRACT,
    V6_ARTIFACT_PATHS,
    V6_FORMAL_SEEDS,
    canonicalize_v6_artifact_run_mapping,
    validate_v6_cifar100_provenance_files,
    validate_v6_protocol,
)
from talif_msresnet.pathing import artifact_path_reference
from talif_msresnet.preflight import check_protocol
from talif_msresnet.utils import (
    atomic_write_json,
    load_checkpoint,
    sha256_file,
    stable_hash,
)
from talif_msresnet.v6_statistics import (
    V6_HOLM_CONTRASTS,
    V6_REPLICATION_CONTRAST,
    V6_SECONDARY_CONTRASTS,
    analyze_v6_mechanism_accuracy,
    mdes_grid,
)

ANALYSIS_JSON = "v6_mechanism_accuracy_analysis.json"
RAW_ACCURACIES_CSV = "v6_seed_condition_test_accuracies.csv"
RAW_DIFFERENCES_CSV = "v6_paired_seed_differences.csv"
ANALYSIS_SCHEMA = "ta-lif-msresnet-v6-mechanism-analysis-v1"


class V6AnalysisInputError(RuntimeError):
    """Raised when formal artifacts are incomplete or not frozen V6 evidence."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPOSITORY_ROOT / V6_ARTIFACT_PATHS["protocol"],
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V6AnalysisInputError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V6AnalysisInputError(f"{label} must contain a JSON object: {path}")
    return value


def _canonical_roots(
    protocol: Mapping[str, Any],
    input_path: Path | None,
    output_path: Path | None,
) -> tuple[Path, Path]:
    artifacts = artifact_paths_for_protocol(protocol)
    expected_input = (REPOSITORY_ROOT / artifacts["formal_results"]).resolve()
    expected_output = (REPOSITORY_ROOT / artifacts["analysis_results"]).resolve()
    actual_input = expected_input if input_path is None else input_path.resolve()
    actual_output = expected_output if output_path is None else output_path.resolve()
    if actual_input != expected_input:
        raise V6AnalysisInputError(
            f"V6 input must be the isolated formal-results root {expected_input}"
        )
    if actual_output != expected_output:
        raise V6AnalysisInputError(
            f"V6 output must be the isolated analysis-results root {expected_output}"
        )
    if actual_input == actual_output:
        raise V6AnalysisInputError("Formal results and analysis output must be isolated")
    return actual_input, actual_output


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_complete_matrix_integrity(audit: Sequence[Mapping[str, Any]]) -> None:
    """Enforce V6's global environment/split and paired-init contracts."""

    expected_count = len(V6_FORMAL_SEEDS) * len(V6_ACTIVE_CONDITIONS)
    if len(audit) != expected_count:
        raise V6AnalysisInputError(
            f"Expected {expected_count} V6 artifact-audit rows, got {len(audit)}"
        )
    environments = {row.get("training_environment_sha256") for row in audit}
    splits = {row.get("split_manifest_sha256") for row in audit}
    test_pickles = {row.get("cifar100_test_pickle_sha256") for row in audit}
    source_provenances = {
        row.get("cifar100_source_provenance_sha256") for row in audit
    }
    if len(environments) != 1 or not _is_sha256(next(iter(environments), None)):
        raise V6AnalysisInputError(
            "V6 formal runs do not share one valid training_environment_sha256"
        )
    if len(splits) != 1 or not _is_sha256(next(iter(splits), None)):
        raise V6AnalysisInputError(
            "V6 formal runs do not share one valid split_manifest_sha256"
        )
    if len(test_pickles) != 1 or not _is_sha256(next(iter(test_pickles), None)):
        raise V6AnalysisInputError(
            "V6 final-test markers do not share one valid CIFAR-100 test-pickle SHA-256"
        )
    if len(source_provenances) != 1 or not _is_sha256(
        next(iter(source_provenances), None)
    ):
        raise V6AnalysisInputError(
            "V6 final-test markers do not share one valid CIFAR-100 source-provenance SHA-256"
        )
    for seed in V6_FORMAL_SEEDS:
        block = [row for row in audit if row.get("seed") == seed]
        block.sort(key=lambda row: V6_ACTIVE_CONDITIONS.index(str(row.get("condition"))))
        conditions = tuple(row.get("condition") for row in block)
        shared = {row.get("shared_weight_sha256") for row in block}
        if conditions != V6_ACTIVE_CONDITIONS:
            raise V6AnalysisInputError(
                f"Seed {seed}: condition block is not the frozen ordered six-condition set"
            )
        if len(shared) != 1 or not _is_sha256(next(iter(shared), None)):
            raise V6AnalysisInputError(
                f"Seed {seed}: conditions do not share one valid initialization hash"
            )


def collect_final_test_rows(
    results_root: Path,
    *,
    protocol: Mapping[str, Any],
    expected_protocol_hash: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Load exactly 48 committed one-time final-test markers and their bindings."""

    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    expected_ids = {
        f"E6_cifar100_d20_t6_{condition}_s{seed}"
        for seed in V6_FORMAL_SEEDS
        for condition in V6_ACTIVE_CONDITIONS
    }
    if not results_root.is_dir():
        raise V6AnalysisInputError(f"V6 formal-results root is missing: {results_root}")
    try:
        source_binding = validate_v6_cifar100_provenance_files(
            protocol,
            project_root=REPOSITORY_ROOT,
        )
    except ValueError as exc:
        raise V6AnalysisInputError(
            f"V6 CIFAR-100 source provenance validation failed: {exc}"
        ) from exc
    expected_test_source = {
        "test_source": source_binding["cifar100_test_pickle"],
        "test_source_sha256": source_binding["cifar100_test_pickle_sha256"],
        **source_binding,
    }
    observed_dirs = {path.name for path in results_root.iterdir() if path.is_dir()}
    unexpected = sorted(
        name
        for name in observed_dirs - expected_ids
        if not name.startswith(".") and name not in {"logs"}
    )
    if unexpected:
        raise V6AnalysisInputError(
            "Unexpected run directories in the isolated V6 root: " + ", ".join(unexpected)
        )

    for seed in V6_FORMAL_SEEDS:
        for condition in V6_ACTIVE_CONDITIONS:
            run_id = f"E6_cifar100_d20_t6_{condition}_s{seed}"
            run_dir = results_root / run_id
            marker_path = run_dir / "final_test.json"
            checkpoint_path = run_dir / "best.pt"
            manifest_path = run_dir / "run_manifest.json"
            metrics_path = run_dir / "seed_metrics.json"
            for path, label in (
                (marker_path, "final-test marker"),
                (checkpoint_path, "best checkpoint"),
                (manifest_path, "run manifest"),
                (metrics_path, "seed metrics"),
            ):
                if not path.is_file():
                    raise V6AnalysisInputError(f"{run_id}: missing {label}: {path}")
            marker = _read_json(marker_path, f"{run_id} final-test marker")
            manifest = _read_json(manifest_path, f"{run_id} run manifest")
            metrics = _read_json(metrics_path, f"{run_id} seed metrics")
            config = manifest.get("config")
            analysis = config.get("analysis") if isinstance(config, Mapping) else None
            failures: list[str] = []
            expected_marker = {
                "status": "complete",
                "run_id": run_id,
                "protocol_hash": expected_protocol_hash,
            }
            for key, expected in expected_marker.items():
                if marker.get(key) != expected:
                    failures.append(f"marker.{key}")
            if not isinstance(analysis, Mapping) or analysis.get(
                "protocol_hash"
            ) != expected_protocol_hash:
                failures.append("manifest.config.analysis.protocol_hash")
            if manifest.get("status") != "complete" or metrics.get("status") != "complete":
                failures.append("training completion status")
            if metrics.get("run_id") != run_id or manifest.get("run_id") != run_id:
                failures.append("run identity")
            if metrics.get("seed") != seed:
                failures.append("seed")
            if metrics.get("condition") != condition:
                failures.append("condition")
            checkpoint_sha256 = sha256_file(checkpoint_path)
            if marker.get("checkpoint_sha256") != checkpoint_sha256:
                failures.append("checkpoint_sha256")
            config_hashes = {
                str(value)
                for value in (
                    marker.get("config_hash"),
                    metrics.get("config_hash"),
                    manifest.get("config_hash"),
                )
            }
            if len(config_hashes) != 1 or "" in config_hashes:
                failures.append("config_hash consistency")
            try:
                normalized_config = canonicalize_v6_artifact_run_mapping(
                    config,
                    protocol,
                    project_root=REPOSITORY_ROOT,
                )
                resolved_config = validate_run_mapping(normalized_config, protocol)
                if resolved_config.runtime.run_id != run_id:
                    failures.append("manifest config run_id")
                if resolved_config.config_hash != str(metrics.get("config_hash")):
                    failures.append("manifest config hash")
            except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
                failures.append("manifest config contract")
            try:
                checkpoint_payload = load_checkpoint(checkpoint_path, map_location="cpu")
                checkpoint_config = checkpoint_payload.get("config")
                checkpoint_hash = checkpoint_payload.get("config_hash")
                if not isinstance(checkpoint_config, Mapping):
                    failures.append("checkpoint config")
                else:
                    normalized_checkpoint = canonicalize_v6_artifact_run_mapping(
                        checkpoint_config,
                        protocol,
                        project_root=REPOSITORY_ROOT,
                    )
                    resolved_checkpoint = validate_run_mapping(
                        normalized_checkpoint, protocol
                    )
                    if resolved_checkpoint.runtime.run_id != run_id:
                        failures.append("checkpoint config run_id")
                    if str(checkpoint_hash) != resolved_checkpoint.config_hash:
                        failures.append("checkpoint config hash")
                    if str(checkpoint_hash) != str(metrics.get("config_hash")):
                        failures.append("checkpoint/metric config hash")
                    checkpoint_epoch = int(checkpoint_payload.get("epoch"))
                    if int(marker.get("checkpoint_epoch_zero_based")) != checkpoint_epoch:
                        failures.append("checkpoint epoch")
                    if int(marker.get("selected_epoch")) != checkpoint_epoch + 1:
                        failures.append("selected epoch")
            except (
                AttributeError,
                EOFError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                failures.append("checkpoint payload contract")
            if not isinstance(marker.get("test_source"), str) or not str(
                marker.get("test_source")
            ).strip():
                failures.append("test_source")
            if not isinstance(marker.get("test_source_sha256"), str) or not str(
                marker.get("test_source_sha256")
            ).strip():
                failures.append("test_source_sha256")
            for key, expected in expected_test_source.items():
                if marker.get(key) != expected:
                    failures.append(f"marker.{key}")
            accuracy = marker.get("test_accuracy")
            if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
                failures.append("test_accuracy")
            elif not 0.0 <= float(accuracy) <= 1.0:
                failures.append("test_accuracy range")
            metric_accuracy = metrics.get("test_accuracy")
            if isinstance(metric_accuracy, bool) or not isinstance(
                metric_accuracy, (int, float)
            ):
                failures.append("metrics.test_accuracy")
            elif not math.isclose(
                float(metric_accuracy), float(accuracy), rel_tol=0.0, abs_tol=1e-15
            ):
                failures.append("marker/metrics test_accuracy")
            for field in ("test_loss", "test_samples", "test_checkpoint_sha256", "test_evaluated_at"):
                if field == "test_checkpoint_sha256":
                    if metrics.get(field) != checkpoint_sha256:
                        failures.append("metrics.test_checkpoint_sha256")
                elif field == "test_evaluated_at":
                    if metrics.get(field) != marker.get("evaluated_at"):
                        failures.append("metrics.test_evaluated_at")
                elif field == "test_samples":
                    if metrics.get(field) != marker.get(field):
                        failures.append("metrics.test_samples")
                else:
                    if not math.isclose(
                        float(metrics.get(field)),
                        float(marker.get(field)),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ):
                        failures.append("marker/metrics test_loss")
            final_record = manifest.get("final_test")
            if not isinstance(final_record, Mapping):
                failures.append("manifest.final_test")
            elif (
                final_record.get("status") != "complete"
                or final_record.get("checkpoint_sha256") != checkpoint_sha256
                or final_record.get("evaluated_at") != marker.get("evaluated_at")
                or final_record.get("samples") != marker.get("test_samples")
            ):
                failures.append("manifest.final_test consistency")
            elif any(
                final_record.get(key) != expected
                for key, expected in expected_test_source.items()
            ):
                failures.append("manifest.final_test source provenance")
            environment = manifest.get("environment")
            manifest_environment_hash = (
                environment.get("training_environment_sha256")
                if isinstance(environment, Mapping)
                else None
            )
            if manifest_environment_hash != metrics.get("training_environment_sha256"):
                failures.append("training environment consistency")
            if manifest.get("split_manifest_sha256") != metrics.get(
                "split_manifest_sha256"
            ):
                failures.append("split manifest consistency")
            if manifest.get("shared_weight_sha256") != metrics.get(
                "shared_weight_sha256"
            ):
                failures.append("shared initialization consistency")
            if failures:
                raise V6AnalysisInputError(
                    f"{run_id}: invalid final-test evidence at " + ", ".join(failures)
                )
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "test_accuracy_pp": 100.0 * float(accuracy),
                    "run_id": run_id,
                    "marker_path": artifact_path_reference(marker_path, REPOSITORY_ROOT),
                    "marker_sha256": sha256_file(marker_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                }
            )
            audit.append(
                {
                    "run_id": run_id,
                    "marker_sha256": sha256_file(marker_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "config_hash": metrics.get("config_hash"),
                    "test_accuracy": float(accuracy),
                    "test_loss": float(marker.get("test_loss")),
                    "test_samples": int(marker.get("test_samples")),
                    "selected_epoch": int(marker.get("selected_epoch")),
                    "seed": seed,
                    "condition": condition,
                    "training_environment_sha256": metrics.get(
                        "training_environment_sha256"
                    ),
                    "split_manifest_sha256": metrics.get("split_manifest_sha256"),
                    "shared_weight_sha256": metrics.get("shared_weight_sha256"),
                    "cifar100_test_pickle_sha256": marker.get(
                        "cifar100_test_pickle_sha256"
                    ),
                    "cifar100_source_provenance_sha256": marker.get(
                        "cifar100_source_provenance_sha256"
                    ),
                }
            )
    _validate_complete_matrix_integrity(audit)
    return pd.DataFrame(rows), audit


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _difference_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    results = [analysis["replication"], *analysis["holm_family"], *analysis["secondary_estimates"]]
    rows: list[dict[str, Any]] = []
    for result in results:
        for seed, difference in result["seed_differences_pp"]:
            rows.append(
                {
                    "contrast": result["name"],
                    "treatment": result["treatment"],
                    "reference": result["reference"],
                    "seed": seed,
                    "difference_pp": difference,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = args.protocol.resolve()
    expected_protocol = (REPOSITORY_ROOT / V6_ARTIFACT_PATHS["protocol"]).resolve()
    if protocol_path != expected_protocol:
        raise SystemExit(f"V6 analysis requires protocol {expected_protocol}")
    protocol = validate_v6_protocol(load_protocol(protocol_path))
    input_root, output_root = _canonical_roots(protocol, args.input, args.output)
    report_path = output_root / ANALYSIS_JSON
    accuracies_path = output_root / RAW_ACCURACIES_CSV
    differences_path = output_root / RAW_DIFFERENCES_CSV
    existing = [path for path in (report_path, accuracies_path, differences_path) if path.exists()]
    if existing:
        raise SystemExit(
            "Refusing to overwrite existing V6 analysis artifacts:\n- "
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
            "V6 analysis preflight blocked:\n- " + "\n- ".join(preflight.errors)
        )
    expected_hash = stable_hash(protocol)
    if preflight.protocol_hash != expected_hash:
        raise SystemExit("V6 preflight protocol hash differs from the loaded protocol")

    frame, artifact_audit = collect_final_test_rows(
        input_root,
        protocol=protocol,
        expected_protocol_hash=expected_hash,
    )
    result = analyze_v6_mechanism_accuracy(frame)
    result_dict = result.to_dict()
    accuracy_rows = frame.to_dict("records")
    difference_rows = _difference_rows(result_dict)
    payload = {
        "schema": ANALYSIS_SCHEMA,
        "protocol_version": 6,
        "protocol": artifact_path_reference(protocol_path, REPOSITORY_ROOT),
        "protocol_file_sha256": sha256_file(protocol_path),
        "protocol_hash": expected_hash,
        "analysis_contract": V6_ANALYSIS_CONTRACT,
        "input_root": artifact_path_reference(input_root, REPOSITORY_ROOT),
        "n_final_test_markers": len(artifact_audit),
        "input_artifact_audit": artifact_audit,
        "cifar100_final_test_source_binding": validate_v6_cifar100_provenance_files(
            protocol,
            project_root=REPOSITORY_ROOT,
        ),
        "registered_contrasts": {
            "replication": V6_REPLICATION_CONTRAST.name,
            "holm_family": [item.name for item in V6_HOLM_CONTRASTS],
            "secondary": [item.name for item in V6_SECONDARY_CONTRASTS],
        },
        "analysis": result_dict,
        "prospective_mdes": [item.__dict__ for item in mdes_grid()],
        "raw_accuracies_csv": artifact_path_reference(
            accuracies_path, REPOSITORY_ROOT
        ),
        "raw_differences_csv": artifact_path_reference(
            differences_path, REPOSITORY_ROOT
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_write_csv(accuracies_path, accuracy_rows)
    try:
        _atomic_write_csv(differences_path, difference_rows)
        atomic_write_json(report_path, payload)
    except BaseException:
        accuracies_path.unlink(missing_ok=True)
        differences_path.unlink(missing_ok=True)
        raise
    print(f"analysis: {report_path}")
    print(f"raw_accuracies: {accuracies_path}")
    print(f"raw_differences: {differences_path}")
    print(f"study_decision: {result.study_conclusion.decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
