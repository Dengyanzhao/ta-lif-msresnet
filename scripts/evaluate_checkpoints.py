#!/usr/bin/env python3
"""Evaluate every frozen best checkpoint exactly once on its independent test set."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.config import (  # noqa: E402
    EXPECTED_RUN_COUNT,
    artifact_paths_for_protocol,
    expected_run_count_for_protocol,
    load_protocol,
    validate_run_mapping,
)
from talif_msresnet.data import build_test_loader  # noqa: E402
from talif_msresnet.freeze import verify_formal_freeze  # noqa: E402
from talif_msresnet.models import build_model  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.preflight import check_protocol  # noqa: E402
from talif_msresnet.train import (  # noqa: E402
    SEED_METRIC_FIELDS,
    _training_environment_identity,
    _v4_formal_execution_evidence,
    evaluate,
)
from talif_msresnet.utils import (  # noqa: E402
    JSONLLogger,
    atomic_write_json,
    load_checkpoint,
    resolve_device,
    seed_everything,
    sha256_file,
    upsert_csv_row,
    utc_now,
)


JOURNAL_NAME = "final_test.in_progress.json"
LOCK_NAME = "final_test.lock"


def _recovery_instruction(run_id: str) -> str:
    return (
        f"Inspect {run_id}/{JOURNAL_NAME}. If stage='results_ready', run this script with "
        f"--recover-run {run_id} to commit without touching test data again. For any earlier stage, "
        "test access is ambiguous: do not delete the journal or rerun automatically; document the "
        "incident and obtain an explicit author decision."
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _read_expected(
    config_dir: Path,
    expected_count: int = EXPECTED_RUN_COUNT,
) -> list[dict[str, str]]:
    manifest = config_dir / "run_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}; generate the frozen matrix first")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "run_id", "experiment", "dataset", "depth", "time_steps", "condition", "topology",
            "neuron", "seed", "config_hash", "protocol_hash", "config_file", "config_file_sha256",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"Frozen run_manifest.csv is missing columns: {sorted(missing)}")
        rows = list(reader)
    if (
        len(rows) != expected_count
        or len({row["run_id"] for row in rows}) != expected_count
    ):
        raise RuntimeError(
            f"Final test requires the complete {expected_count}-run matrix"
        )
    for row in rows:
        config_path = config_dir / row["config_file"]
        if not config_path.exists() or sha256_file(config_path) != row["config_file_sha256"]:
            raise RuntimeError(f"Frozen generated config file hash mismatch: {config_path}")
    return rows


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "seed_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Malformed {path}")
    return value


def _audit_consolidated_metrics(
    rows: list[dict[str, str]],
    results_root: Path,
    recovery_run: str | None = None,
) -> dict[str, dict[str, str]]:
    """Bind the consolidated CSV to every planned run ID and hash."""

    path = results_root / "seed_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing consolidated seed metrics: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if list(reader.fieldnames or []) != list(SEED_METRIC_FIELDS):
            raise RuntimeError(f"Consolidated seed_metrics.csv schema differs from the trainer schema: {path}")
        metrics_rows = list(reader)
    planned = {row["run_id"]: row for row in rows}
    ids = [row.get("run_id", "") for row in metrics_rows]
    expected_count = len(rows)
    if (
        len(metrics_rows) != expected_count
        or len(set(ids)) != expected_count
        or set(ids) != set(planned)
    ):
        missing = sorted(set(planned) - set(ids))
        extra = sorted(set(ids) - set(planned))
        raise RuntimeError(
            f"Consolidated seed_metrics.csv must contain exactly the {expected_count} "
            "unique planned runs; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    by_id = {row["run_id"]: row for row in metrics_rows}
    identity_fields = (
        "experiment", "dataset", "depth", "time_steps", "condition", "topology", "neuron", "seed",
    )
    test_fields = (
        "test_loss", "test_accuracy", "test_samples", "test_checkpoint_sha256", "test_evaluated_at",
    )
    errors: list[str] = []
    for run_id, plan in planned.items():
        metric = by_id[run_id]
        if metric.get("config_hash") != plan.get("config_hash"):
            errors.append(f"{run_id}: consolidated/planned config hashes differ")
        for field in identity_fields:
            if str(metric.get(field, "")) != str(plan.get(field, "")):
                errors.append(f"{run_id}: consolidated {field} differs from the plan")
        per_run = _load_metrics(results_root / run_id)
        manifest_path = results_root / run_id / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{run_id}: cannot read run_manifest.json: {exc}")
            continue
        manifest_config = manifest.get("config", {})
        manifest_analysis = manifest_config.get("analysis", {}) if isinstance(manifest_config, Mapping) else {}
        comparisons = {
            "config_hash": (
                metric.get("config_hash", ""), per_run.get("config_hash", ""),
                manifest.get("config_hash", ""), plan.get("config_hash", ""),
            ),
            "protocol_hash": (
                metric.get("protocol_hash", ""), per_run.get("protocol_hash", ""),
                manifest_analysis.get("protocol_hash", "") if isinstance(manifest_analysis, Mapping) else "",
                plan.get("protocol_hash", ""),
            ),
            "split_manifest_sha256": (
                metric.get("split_manifest_sha256", ""), per_run.get("split_manifest_sha256", ""),
                manifest.get("split_manifest_sha256", ""),
            ),
            "shared_weight_sha256": (
                metric.get("shared_weight_sha256", ""), per_run.get("shared_weight_sha256", ""),
                manifest.get("shared_weight_sha256", ""),
            ),
            "status": (metric.get("status", ""), per_run.get("status", "")),
        }
        for field, values in comparisons.items():
            normalized = {str(value) for value in values}
            if len(normalized) != 1 or "" in normalized:
                errors.append(f"{run_id}: {field} differs across plan/consolidated/per-run/manifest")

        marker_exists = (results_root / run_id / "final_test.json").exists()
        marker_test_values: dict[str, Any] = {}
        if marker_exists:
            try:
                marker = json.loads((results_root / run_id / "final_test.json").read_text(encoding="utf-8"))
                marker_test_values = {
                    "test_loss": marker.get("test_loss"),
                    "test_accuracy": marker.get("test_accuracy"),
                    "test_samples": marker.get("test_samples"),
                    "test_checkpoint_sha256": marker.get("checkpoint_sha256"),
                    "test_evaluated_at": marker.get("evaluated_at"),
                }
            except Exception as exc:
                errors.append(f"{run_id}: cannot read committed final-test marker: {exc}")
        journal_path = results_root / run_id / JOURNAL_NAME
        recoverable = False
        recovery_expected: dict[str, Any] = {}
        if run_id == recovery_run and journal_path.exists():
            try:
                recovery_journal = json.loads(journal_path.read_text(encoding="utf-8"))
                recoverable = recovery_journal.get("stage") == "results_ready"
                result = recovery_journal.get("test", {})
                recovery_expected = {
                    "test_loss": result.get("loss"),
                    "test_accuracy": result.get("accuracy"),
                    "test_samples": result.get("samples"),
                    "test_checkpoint_sha256": recovery_journal.get("checkpoint_sha256"),
                    "test_evaluated_at": recovery_journal.get("evaluated_at"),
                }
            except Exception:
                recoverable = False
        for field in test_fields:
            csv_value = metric.get(field, "")
            per_value = per_run.get(field, "")
            if recoverable:
                expected = str(recovery_expected.get(field, ""))
                if any(str(value) not in ("", expected) for value in (csv_value, per_value)):
                    errors.append(f"{run_id}: stored {field} differs from the recoverable journal")
            elif str(csv_value) != str(per_value):
                errors.append(f"{run_id}: consolidated/per-run {field} differs")
            if marker_exists and str(csv_value) != str(marker_test_values.get(field, "")):
                errors.append(f"{run_id}: consolidated/marker {field} differs")
        any_test_value = any(
            value not in (None, "")
            for field in test_fields
            for value in (metric.get(field), per_run.get(field))
        )
        if any_test_value and not marker_exists and not recoverable:
            errors.append(f"{run_id}: test fields are populated without a committed marker")
    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:30])
        raise RuntimeError(f"Consolidated seed-metric audit failed:\n{preview}")
    return by_id


def _audit_all_frozen(
    rows: list[dict[str, str]],
    results_root: Path,
    protocol_hash: str,
) -> None:
    errors: list[str] = []
    for row in rows:
        run_id = row["run_id"]
        run_dir = results_root / run_id
        checkpoint = run_dir / "best.pt"
        journal = run_dir / JOURNAL_NAME
        lock = run_dir / LOCK_NAME
        marker = run_dir / "final_test.json"
        if journal.exists() or lock.exists():
            if marker.exists():
                errors.append(
                    f"{run_id}: committed final test has stale transaction files; "
                    f"run --recover-run {run_id} to verify and clean them"
                )
            else:
                errors.append(f"{run_id}: interrupted final-test transaction. {_recovery_instruction(run_id)}")
        try:
            metrics = _load_metrics(run_dir)
        except Exception as exc:
            errors.append(f"{run_id}: {exc}")
            continue
        if metrics.get("status") != "complete":
            errors.append(f"{run_id}: status={metrics.get('status')!r}, expected 'complete'")
        if row.get("config_hash") and str(metrics.get("config_hash")) != row["config_hash"]:
            errors.append(f"{run_id}: seed metric config hash differs from generated manifest")
        if metrics.get("test_accuracy") not in (None, ""):
            if not marker.exists():
                errors.append(f"{run_id}: test metric exists without final_test.json audit marker")
        if not checkpoint.exists():
            errors.append(f"{run_id}: missing best.pt")
            continue
        if marker.exists():
            try:
                marker_value = json.loads(marker.read_text(encoding="utf-8"))
                if marker_value.get("checkpoint_sha256") != sha256_file(checkpoint):
                    errors.append(f"{run_id}: final-test marker checkpoint hash differs from best.pt")
                if metrics.get("test_accuracy") in (None, ""):
                    errors.append(f"{run_id}: final-test marker exists but seed metric is empty")
            except Exception as exc:
                errors.append(f"{run_id}: malformed final_test.json: {exc}")
        payload = load_checkpoint(checkpoint, map_location="cpu")
        config = _mapping(payload.get("config"), f"{run_id} checkpoint config")
        analysis = config.get("analysis", {})
        analysis = analysis if isinstance(analysis, Mapping) else {}
        if analysis.get("protocol_hash") != protocol_hash:
            errors.append(f"{run_id}: checkpoint protocol hash differs from frozen protocol")
        if str(payload.get("config_hash")) != str(metrics.get("config_hash")):
            errors.append(f"{run_id}: checkpoint and seed metric config hashes differ")
    if errors:
        preview = "\n".join(f"- {item}" for item in errors[:30])
        suffix = f"\n... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise RuntimeError(f"Final-test freeze audit failed:\n{preview}{suffix}")


def _test_source_record(config: Any) -> dict[str, Any]:
    if config.data.dataset != "cifar10dvs":
        return {
            "test_source": f"torchvision official {config.data.dataset} test partition",
            "test_source_sha256": "dataset implementation does not expose one aggregate file",
        }
    source = Path(str(config.data.test_frames_path))
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    index = source / "index.csv" if source.is_dir() else source
    source_hash = sha256_file(index)
    return {
        "test_source": artifact_path_reference(
            source,
            PROJECT_ROOT,
            external_identifier=f"index-sha256:{source_hash}",
        ),
        "test_source_sha256": source_hash,
    }


def _transaction_paths(run_dir: Path) -> tuple[Path, Path]:
    return run_dir / JOURNAL_NAME, run_dir / LOCK_NAME


def _begin_transaction(run_dir: Path, journal: Mapping[str, Any]) -> tuple[Path, Path]:
    journal_path, lock_path = _transaction_paths(run_dir)
    if journal_path.exists() or lock_path.exists():
        raise RuntimeError(f"{run_dir.name}: interrupted/concurrent final test. {_recovery_instruction(run_dir.name)}")
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"{run_dir.name}: final-test lock already exists. {_recovery_instruction(run_dir.name)}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started_at={utc_now()}\n".encode("ascii"))
    finally:
        os.close(descriptor)
    try:
        atomic_write_json(journal_path, journal)
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return journal_path, lock_path


def _cleanup_transaction(run_dir: Path) -> None:
    journal_path, lock_path = _transaction_paths(run_dir)
    journal_path.unlink(missing_ok=True)
    lock_path.unlink(missing_ok=True)


def _commit_final_test(run_dir: Path, results_root: Path, journal: Mapping[str, Any]) -> None:
    if journal.get("stage") != "results_ready":
        raise RuntimeError(f"{run_dir.name}: transaction has no recoverable final-test result")
    result = _mapping(journal.get("test"), "transaction test result")
    checkpoint_hash = str(journal["checkpoint_sha256"])
    evaluated_at = str(journal["evaluated_at"])
    metrics = _load_metrics(run_dir)
    existing_accuracy = metrics.get("test_accuracy")
    if existing_accuracy not in (None, "") and not math.isclose(
        float(existing_accuracy), float(result["accuracy"]), rel_tol=0.0, abs_tol=1e-15,
    ):
        raise RuntimeError(f"{run_dir.name}: existing test metric differs from the transaction journal")
    metrics.update({
        "test_loss": float(result["loss"]),
        "test_accuracy": float(result["accuracy"]),
        "test_samples": int(result["samples"]),
        "test_checkpoint_sha256": checkpoint_hash,
        "test_evaluated_at": evaluated_at,
    })
    upsert_csv_row(
        results_root / "seed_metrics.csv", metrics, SEED_METRIC_FIELDS,
        key_field="run_id", require_existing=True,
    )
    atomic_write_json(run_dir / "seed_metrics.json", metrics)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_test"] = {
        "status": "complete",
        "checkpoint": "best.pt",
        "checkpoint_sha256": checkpoint_hash,
        "evaluated_at": evaluated_at,
        "samples": int(result["samples"]),
    }
    atomic_write_json(manifest_path, manifest)

    marker = {
        "format_version": 1,
        "status": "complete",
        "run_id": run_dir.name,
        "config_hash": journal["config_hash"],
        "protocol_hash": journal["protocol_hash"],
        "checkpoint": artifact_path_reference(
            run_dir / "best.pt",
            PROJECT_ROOT,
            external_identifier=f"checkpoint-sha256:{checkpoint_hash}",
        ),
        "checkpoint_sha256": checkpoint_hash,
        "selected_epoch": int(journal["checkpoint_epoch_zero_based"]) + 1,
        "checkpoint_epoch_zero_based": int(journal["checkpoint_epoch_zero_based"]),
        "model_selection_rule": (
            "highest validation accuracy; ties resolved by lower validation loss; "
            "exact ties resolved by earliest epoch"
        ),
        "evaluated_at": evaluated_at,
        "device": journal["device"],
        "test_loss": float(result["loss"]),
        "test_accuracy": float(result["accuracy"]),
        "test_samples": int(result["samples"]),
        **dict(_mapping(journal.get("test_source"), "transaction test source")),
    }
    # This marker is the atomic commit record and is deliberately written last.
    atomic_write_json(run_dir / "final_test.json", marker)
    JSONLLogger(run_dir / "events.jsonl").log("final_test_completed", **marker)


def _recover_one(run_dir: Path, results_root: Path, protocol_hash: str) -> str:
    journal_path, lock_path = _transaction_paths(run_dir)
    marker_path = run_dir / "final_test.json"
    if not journal_path.exists():
        if lock_path.exists():
            raise RuntimeError(
                f"{run_dir.name}: lock exists without a journal; test access cannot be established. "
                f"{_recovery_instruction(run_dir.name)}"
            )
        raise FileNotFoundError(f"No interrupted final-test journal for {run_dir.name}")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    checkpoint_hash = sha256_file(run_dir / "best.pt")
    if journal.get("run_id") != run_dir.name or journal.get("checkpoint_sha256") != checkpoint_hash:
        raise RuntimeError(f"{run_dir.name}: journal does not match the current best.pt")
    if journal.get("protocol_hash") != protocol_hash:
        raise RuntimeError(f"{run_dir.name}: journal protocol hash differs from the frozen protocol")
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("status") == "complete" and marker.get("checkpoint_sha256") == checkpoint_hash:
            _cleanup_transaction(run_dir)
            return "cleaned_committed_transaction"
        raise RuntimeError(f"{run_dir.name}: marker and transaction journal disagree")
    if journal.get("stage") != "results_ready":
        raise RuntimeError(f"{run_dir.name}: test access is ambiguous. {_recovery_instruction(run_dir.name)}")
    _commit_final_test(run_dir, results_root, journal)
    _cleanup_transaction(run_dir)
    return "recovered_without_test_access"


def _evaluate_one(run_dir: Path, device_name: str, results_root: Path) -> str:
    checkpoint_path = run_dir / "best.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    marker_path = run_dir / "final_test.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("checkpoint_sha256") == checkpoint_hash and marker.get("status") == "complete":
            return "already_complete"
        raise FileExistsError(f"{marker_path} belongs to a different checkpoint; refusing to overwrite")
    journal_path, lock_path = _transaction_paths(run_dir)
    if journal_path.exists() or lock_path.exists():
        raise RuntimeError(f"{run_dir.name}: interrupted final test. {_recovery_instruction(run_dir.name)}")

    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = validate_run_mapping(_mapping(payload.get("config"), "checkpoint config"))
    if config.runtime.dry_run:
        raise RuntimeError(f"{run_dir.name}: dry-run checkpoints cannot be final-tested")
    if payload.get("config_hash") != config.config_hash:
        raise RuntimeError(f"{run_dir.name}: checkpoint scientific config hash is invalid")
    metrics = _load_metrics(run_dir)
    if metrics.get("test_accuracy") not in (None, ""):
        raise RuntimeError(f"{run_dir.name}: test_accuracy is already populated without a marker")
    runtime = dataclasses.replace(
        config.runtime, device=device_name, dry_run=False, limit_batches=None, resume=None,
    )
    config = dataclasses.replace(config, runtime=runtime, final_test=False)
    device = resolve_device(device_name)
    model_config = dataclasses.asdict(config.model)
    model_config["init_seed"] = config.runtime.seed
    model = build_model(model_config).to(device)
    model.load_state_dict(payload["model_state"], strict=True)

    journal: dict[str, Any] = {
        "format_version": 1,
        "status": "in_progress",
        "stage": "prepared",
        "run_id": run_dir.name,
        "config_hash": config.config_hash,
        "protocol_hash": config.analysis.get("protocol_hash"),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch_zero_based": int(payload["epoch"]),
        "started_at": utc_now(),
        "device": str(device),
        "test_source": _test_source_record(config),
    }
    journal_path, _ = _begin_transaction(run_dir, journal)
    test_access_started = False
    try:
        journal["stage"] = "test_access_started"
        atomic_write_json(journal_path, journal)
        test_access_started = True
        loader = build_test_loader(config, seed=config.runtime.seed)
        test = evaluate(model, loader, device, config, "test")
        if not math.isfinite(float(test["accuracy"])) or not math.isfinite(float(test["loss"])):
            raise RuntimeError(f"{run_dir.name}: non-finite final-test metric")
        journal.update({
            "stage": "results_ready",
            "evaluated_at": utc_now(),
            "test": {
                "loss": float(test["loss"]),
                "accuracy": float(test["accuracy"]),
                "samples": int(test["samples"]),
            },
        })
        atomic_write_json(journal_path, journal)
        _commit_final_test(run_dir, results_root, journal)
        _cleanup_transaction(run_dir)
        return "evaluated"
    except BaseException:
        if not test_access_started:
            _cleanup_transaction(run_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root")
    parser.add_argument("--config-dir")
    parser.add_argument("--protocol", default=str(PROJECT_ROOT / "configs" / "protocol.yaml"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, help="Evaluate the first N pending checkpoints after full freeze audit")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--recover-run",
        metavar="RUN_ID",
        help="Commit a results_ready transaction without constructing or reading the test loader again",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.chdir(PROJECT_ROOT)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.recover_run and args.limit is not None:
        raise SystemExit("--recover-run cannot be combined with --limit")
    report = check_protocol(
        args.protocol,
        mode="final-test",
        project_root=PROJECT_ROOT,
        check_dependencies=True,
    )
    if not report.ok:
        details = "\n".join(f"- {item}" for item in report.errors)
        raise SystemExit(f"Final-test preflight blocked:\n{details}")
    protocol = load_protocol(args.protocol)
    artifacts = artifact_paths_for_protocol(protocol)
    default_results = PROJECT_ROOT / artifacts["formal_results"]
    default_configs = PROJECT_ROOT / artifacts["formal_matrix"]
    results_root = Path(args.results_root or default_results).resolve()
    config_dir = Path(args.config_dir or default_configs).resolve()
    protocol_version = int(protocol.get("protocol_version", 1))
    if protocol_version in (3, 4):
        if results_root != default_results.resolve():
            raise SystemExit(
                f"Protocol v{protocol_version} final test must use "
                "artifact_paths.formal_results"
            )
        if config_dir != default_configs.resolve():
            raise SystemExit(
                f"Protocol v{protocol_version} final test must use "
                "artifact_paths.formal_matrix"
            )
    if protocol_version == 4:
        freeze_manifest = verify_formal_freeze(
            project_root=PROJECT_ROOT,
            protocol_path=args.protocol,
            matrix_dir=config_dir,
        )
        execution_evidence = _v4_formal_execution_evidence(
            protocol,
            freeze_manifest,
            protocol_path=args.protocol,
        )
        seed_everything(0, deterministic=True)
        selected_device = resolve_device(args.device)
        _identity, environment_sha256 = _training_environment_identity(
            selected_device,
            amp=False,
            deterministic=True,
        )
        if environment_sha256 != execution_evidence["training_environment_sha256"]:
            raise SystemExit(
                "Protocol v4 final-test environment differs from the pilot/formal freeze"
            )
    rows = _read_expected(
        config_dir,
        expected_count=expected_run_count_for_protocol(protocol),
    )
    _audit_consolidated_metrics(rows, results_root, recovery_run=args.recover_run)
    if args.recover_run:
        planned_ids = {row["run_id"] for row in rows}
        if args.recover_run not in planned_ids:
            raise SystemExit(f"Unknown planned run_id: {args.recover_run}")
        status = _recover_one(results_root / args.recover_run, results_root, report.protocol_hash)
        print(f"{args.recover_run}: {status}")
        return 0
    _audit_all_frozen(rows, results_root, report.protocol_hash)
    pending = [row for row in rows if not (results_root / row["run_id"] / "final_test.json").exists()]
    if args.limit is not None:
        pending = pending[: args.limit]
    failures = 0
    for index, row in enumerate(pending, start=1):
        run_dir = results_root / row["run_id"]
        try:
            status = _evaluate_one(run_dir, args.device, results_root)
            print(f"[{index}/{len(pending)}] {row['run_id']}: {status}")
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(pending)}] {row['run_id']}: FAILED: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 1
    print(f"Final-test pass complete: {len(pending) - failures} evaluated, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
