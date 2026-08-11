"""Blocking checks performed before smoke, pilot, full, or final-test execution."""

from __future__ import annotations

import importlib
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    confirmation_fields_for_protocol,
    load_protocol,
)
from .freeze import FreezeGateError, verify_formal_freeze


@dataclass(frozen=True)
class PreflightReport:
    mode: str
    protocol_path: str
    protocol_hash: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "protocol_path": self.protocol_path,
            "protocol_hash": self.protocol_hash,
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _check_dependencies(names: Sequence[str]) -> list[str]:
    missing: list[str] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"Python dependency {name!r} is unavailable: {type(exc).__name__}: {exc}")
    return missing


def _check_mechanism_torchvision_version(
    protocol: Mapping[str, Any], *, protocol_version: int
) -> list[str]:
    acceptance = protocol.get("pilot_acceptance")
    environment = acceptance.get("environment") if isinstance(acceptance, Mapping) else None
    expected = environment.get("torchvision_version") if isinstance(environment, Mapping) else None
    if not isinstance(expected, str) or not expected:
        return [
            f"V{protocol_version} pilot_acceptance.environment.torchvision_version "
            "must be frozen"
        ]
    try:
        observed = distribution_version("torchvision")
    except PackageNotFoundError:
        return [f"V{protocol_version} requires an installed torchvision distribution"]
    if observed != expected:
        return [
            f"V{protocol_version} torchvision version mismatch: "
            f"{observed!r} != {expected!r}"
        ]
    return []


def _resolve(project_root: Path, value: Any) -> Path | None:
    if not _nonempty(value):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def _class_counts(labels: Sequence[int], indices: Sequence[int] | None = None) -> dict[str, int]:
    selected = labels if indices is None else [labels[index] for index in indices]
    return {str(label): int(count) for label, count in sorted(Counter(selected).items())}


def _check_manifest_integrity(path: Path, label: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if not path.exists():
        return None, [f"CIFAR10-DVS {label} is missing: {path}"]
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, [f"CIFAR10-DVS {label} is malformed: {type(exc).__name__}: {exc}"]
    if not isinstance(value, dict):
        return None, [f"CIFAR10-DVS {label} must be a mapping"]
    recorded = value.get("manifest_sha256")
    unhashed = dict(value)
    unhashed.pop("manifest_sha256", None)
    if not isinstance(recorded, str) or recorded != _stable_hash(unhashed):
        errors.append(f"CIFAR10-DVS {label} failed its internal SHA-256 check")
    return value, errors


def _check_dvs_provenance(
    project_root: Path,
    trainval: Path,
    test: Path,
    common_data: Mapping[str, Any],
    dvs: Mapping[str, Any],
) -> list[str]:
    """Bind the indexed tensors, conversion record, and all three partitions."""

    from .data import IndexedFrameDataset, label_fingerprint, stratified_split_indices
    from .utils import sha256_file

    errors: list[str] = []
    if trainval.resolve() != test.resolve():
        return [
            "CIFAR10-DVS folder protocol requires trainval/test to reference the same indexed root"
        ]
    root = trainval if trainval.is_dir() else trainval.parent
    index_path = root / "index.csv"
    conversion_path = root / "conversion_manifest.json"
    split_path = root / "split_manifest.json"
    if not index_path.exists():
        return [f"CIFAR10-DVS index.csv is missing: {index_path}"]
    if not conversion_path.exists():
        return [f"CIFAR10-DVS conversion_manifest.json is missing: {conversion_path}"]
    try:
        conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"CIFAR10-DVS conversion manifest is malformed: {type(exc).__name__}: {exc}"]
    if not isinstance(conversion, Mapping):
        return ["CIFAR10-DVS conversion manifest must be a mapping"]
    conversion_version = conversion.get("format_version")
    if conversion_version not in {1, 2}:
        errors.append("CIFAR10-DVS conversion manifest format_version is unsupported")
    if conversion_version == 2:
        source = conversion.get("source")
        if not isinstance(source, Mapping):
            errors.append("CIFAR10-DVS version 2 conversion has no source metadata")
        else:
            expected_source = {
                "kind": "preextracted_aedat4_third_party_conversion",
                "format": "AEDAT4",
                "loader": "tonic.io.read_aedat4",
                "source_manifest": "source_manifest.json",
                "official_archive_byte_identity": "not_verified",
            }
            for key, expected in expected_source.items():
                if source.get(key) != expected:
                    errors.append(f"CIFAR10-DVS conversion source {key} is invalid")
            source_manifest_path = root / "source_manifest.json"
            if not source_manifest_path.is_file():
                errors.append(f"CIFAR10-DVS source manifest is missing: {source_manifest_path}")
            else:
                if source.get("source_manifest_sha256") != sha256_file(source_manifest_path):
                    errors.append(
                        "CIFAR10-DVS current source manifest hash differs from conversion manifest"
                    )
                source_manifest, source_errors = _check_manifest_integrity(
                    source_manifest_path, "AEDAT4 source manifest"
                )
                errors.extend(source_errors)
                if isinstance(source_manifest, Mapping):
                    if source_manifest.get("schema") != "cifar10dvs-aedat4-source-v1":
                        errors.append("CIFAR10-DVS AEDAT4 source manifest schema is invalid")
                    if source_manifest.get("official_archive_byte_identity") != "not_verified":
                        errors.append(
                            "CIFAR10-DVS AEDAT4 source must disclose unverified official archive identity"
                        )
    elif conversion.get("source") is not None:
        errors.append("CIFAR10-DVS version 1 conversion contains version 2 source metadata")
    if conversion.get("index_sha256") != sha256_file(index_path):
        errors.append("CIFAR10-DVS current index.csv hash differs from conversion manifest")
    if not split_path.exists():
        errors.append(f"CIFAR10-DVS split_manifest.json is missing: {split_path}")
        split_manifest = None
    else:
        if conversion.get("split_manifest_sha256") != sha256_file(split_path):
            errors.append("CIFAR10-DVS current split manifest hash differs from conversion manifest")
        split_manifest, split_errors = _check_manifest_integrity(
            split_path, "trainval/test split manifest"
        )
        errors.extend(split_errors)

    expected_t = int(dvs.get("dvs_time_bins", 0))
    expected_c = int(dvs.get("in_channels", 0))
    conversion_spec = conversion.get("conversion", {})
    conversion_spec = conversion_spec if isinstance(conversion_spec, Mapping) else {}
    if conversion_spec.get("time_bins") != expected_t:
        errors.append("CIFAR10-DVS conversion time_bins differs from protocol")
    for key in ("height", "width", "resize_mode"):
        if conversion_spec.get(key) != dvs.get(f"dvs_{key}"):
            errors.append(f"CIFAR10-DVS conversion {key} differs from protocol")
    partition = conversion.get("partition", {})
    partition = partition if isinstance(partition, Mapping) else {}
    if partition.get("test_fraction") != dvs.get("dvs_test_fraction"):
        errors.append("CIFAR10-DVS test_fraction differs from protocol")
    if partition.get("split_seed") != dvs.get("dvs_split_seed"):
        errors.append("CIFAR10-DVS split_seed differs from protocol")
    if dvs.get("normalize") is not False or conversion.get("normalization") != "none_raw_event_counts":
        errors.append("CIFAR10-DVS normalization must be frozen as none_raw_event_counts")

    train_set = IndexedFrameDataset(
        root,
        split="trainval",
        expected_time_steps=expected_t,
        expected_channels=expected_c,
        verify_hashes=False,
    )
    test_set = IndexedFrameDataset(
        root,
        split="test",
        expected_time_steps=expected_t,
        expected_channels=expected_c,
        verify_hashes=False,
    )
    train_ids = {row["sample_id"] for row in train_set.rows}
    test_ids = {row["sample_id"] for row in test_set.rows}
    if train_ids & test_ids:
        errors.append("CIFAR10-DVS trainval/test sample IDs overlap")

    rows = [*train_set.rows, *test_set.rows]
    if conversion.get("sample_count") != len(rows):
        errors.append("CIFAR10-DVS conversion sample_count differs from index.csv")
    try:
        indexed = sorted((int(row["source_index"]), row) for row in rows)
    except (KeyError, TypeError, ValueError) as exc:
        return [*errors, f"CIFAR10-DVS source_index values are invalid: {exc}"]
    source_indices = [index for index, _ in indexed]
    if len(set(source_indices)) != len(source_indices) or set(source_indices) != set(range(len(rows))):
        errors.append("CIFAR10-DVS source indices are not a unique complete range")
        return errors
    labels = [int(row["label"]) for _, row in indexed]
    trainval_source = {int(row["source_index"]) for row in train_set.rows}
    test_source = {int(row["source_index"]) for row in test_set.rows}
    if trainval_source & test_source or trainval_source | test_source != set(source_indices):
        errors.append("CIFAR10-DVS trainval/test rows are not a disjoint complete partition")

    receipt_path = root / "verification_receipt.json"
    if not receipt_path.is_file():
        errors.append(
            "CIFAR10-DVS full verification receipt is missing; run scripts/verify_cifar10dvs.py"
        )
    else:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            receipt = None
            errors.append(
                f"CIFAR10-DVS full verification receipt is malformed: {type(exc).__name__}: {exc}"
            )
        if isinstance(receipt, Mapping):
            recorded_receipt_hash = receipt.get("receipt_sha256")
            unhashed_receipt = dict(receipt)
            unhashed_receipt.pop("receipt_sha256", None)
            if not isinstance(recorded_receipt_hash, str) or recorded_receipt_hash != _stable_hash(
                unhashed_receipt
            ):
                errors.append("CIFAR10-DVS full verification receipt failed its SHA-256 check")
            expected_receipt = {
                "schema": "cifar10dvs-full-verification-v1",
                "status": "pass",
                "verifier": {"name": "scripts/verify_cifar10dvs.py", "version": 1},
                "sample_count": len(rows),
                "trainval_count": len(train_set),
                "test_count": len(test_set),
                "expected_shape": [
                    expected_t,
                    expected_c,
                    int(dvs.get("dvs_height", 0)),
                    int(dvs.get("dvs_width", 0)),
                ],
                "serialized_sample_bytes": sum(
                    int(row["sample_path"].stat().st_size) for row in rows
                ),
                "conversion_manifest_sha256": sha256_file(conversion_path),
                "index_sha256": sha256_file(index_path),
                "split_manifest_sha256": sha256_file(split_path) if split_path.is_file() else None,
            }
            for key, expected in expected_receipt.items():
                if receipt.get(key) != expected:
                    errors.append(f"CIFAR10-DVS verification receipt {key} differs from current data")

    row_errors: set[str] = set()
    for _, row in indexed:
        if str(row.get("split", "")).strip().lower() not in {"trainval", "test"}:
            row_errors.add("CIFAR10-DVS index rows contain invalid split labels")
        if not 0 <= int(row["label"]) < int(dvs.get("num_classes", 0)):
            row_errors.add("CIFAR10-DVS index rows contain out-of-range class labels")
        recorded_sample_hash = str(row.get("sha256", "")).strip().lower()
        if len(recorded_sample_hash) != 64 or any(
            character not in "0123456789abcdef" for character in recorded_sample_hash
        ):
            row_errors.add("CIFAR10-DVS index rows contain invalid sample SHA-256 values")
        expected_row_values = {
            "time_steps": expected_t,
            "channels": expected_c,
            "height": int(dvs.get("dvs_height", 0)),
            "width": int(dvs.get("dvs_width", 0)),
            "split_seed": int(dvs.get("dvs_split_seed", 0)),
        }
        for key, expected in expected_row_values.items():
            try:
                observed = int(row.get(key, ""))
            except (TypeError, ValueError):
                observed = None
            if observed != expected:
                row_errors.add(f"CIFAR10-DVS index rows contain invalid {key}")
                break
        try:
            row_fraction = float(row.get("test_fraction", "nan"))
        except (TypeError, ValueError):
            row_fraction = float("nan")
        if row_fraction != float(dvs.get("dvs_test_fraction", -1)):
            row_errors.add("CIFAR10-DVS index rows contain invalid test_fraction")
        if row.get("resize_mode") != dvs.get("dvs_resize_mode"):
            row_errors.add("CIFAR10-DVS index rows contain invalid resize_mode")
        if row.get("normalization") != "none_raw_event_counts":
            row_errors.add("CIFAR10-DVS index rows contain invalid normalization")
    errors.extend(sorted(row_errors))

    if split_manifest is not None:
        expected_counts = {
            "all": _class_counts(labels),
            "trainval": _class_counts(labels, sorted(trainval_source)),
            "test": _class_counts(labels, sorted(test_source)),
        }
        checks = {
            "dataset": "cifar10dvs",
            "split_strategy": "stratified_by_label",
            "source_length": len(rows),
            "test_fraction": dvs.get("dvs_test_fraction"),
            "split_seed": dvs.get("dvs_split_seed"),
            "label_fingerprint_sha256": label_fingerprint(labels),
            "trainval_indices": sorted(trainval_source),
            "test_indices": sorted(test_source),
            "class_counts": expected_counts,
        }
        for key, expected in checks.items():
            if split_manifest.get(key) != expected:
                errors.append(f"CIFAR10-DVS split manifest {key} differs from index.csv")

    trainval_labels = [int(row["label"]) for row in train_set.rows]
    val_fraction = float(dvs.get("val_fraction", common_data.get("val_fraction", 0.0)))
    split_seed = int(dvs.get("split_seed", common_data.get("split_seed", 0)))
    train_relative, val_relative = stratified_split_indices(
        trainval_labels, val_fraction, split_seed
    )
    train_source = {int(train_set.rows[index]["source_index"]) for index in train_relative}
    val_source = {int(train_set.rows[index]["source_index"]) for index in val_relative}
    if (
        train_source & val_source
        or train_source & test_source
        or val_source & test_source
        or train_source | val_source | test_source != set(source_indices)
    ):
        errors.append("CIFAR10-DVS train/validation/test are not a disjoint complete partition")

    manifest_template = dvs.get("split_manifest", common_data.get("split_manifest"))
    if isinstance(manifest_template, str) and manifest_template:
        train_val_path = _resolve(
            project_root,
            manifest_template.format(dataset="cifar10dvs", split_seed=split_seed),
        )
        if train_val_path is not None and train_val_path.exists():
            train_val_manifest, manifest_errors = _check_manifest_integrity(
                train_val_path, "train/validation split manifest"
            )
            errors.extend(manifest_errors)
            if train_val_manifest is not None:
                expected_train_val = {
                    "dataset": "cifar10dvs",
                    "source_length": len(train_set),
                    "val_fraction": val_fraction,
                    "seed": split_seed,
                    "split_strategy": "stratified_by_label",
                    "source_fingerprint": sha256_file(index_path),
                    "label_fingerprint_sha256": label_fingerprint(trainval_labels),
                    "train_indices": train_relative,
                    "val_indices": val_relative,
                    "class_counts": {
                        "all": _class_counts(trainval_labels),
                        "train": _class_counts(trainval_labels, train_relative),
                        "val": _class_counts(trainval_labels, val_relative),
                    },
                }
                for key, expected in expected_train_val.items():
                    if train_val_manifest.get(key) != expected:
                        errors.append(
                            f"CIFAR10-DVS train/validation manifest {key} differs from protocol/index"
                        )
    return errors


def check_protocol(
    protocol_path: str | Path,
    *,
    mode: str = "full",
    project_root: str | Path | None = None,
    check_dependencies: bool = True,
) -> PreflightReport:
    """Validate the protocol and return all actionable blockers at once."""

    if mode not in {"smoke", "pilot", "full", "final-test"}:
        raise ValueError("mode must be smoke, pilot, full, or final-test")
    path = Path(protocol_path).resolve()
    root = Path(project_root).resolve() if project_root is not None else path.parents[1]
    protocol = load_protocol(path)
    protocol_version = int(protocol.get("protocol_version", 1))
    confirmation_fields = confirmation_fields_for_protocol(protocol)
    errors: list[str] = []
    warnings: list[str] = []

    if protocol_version == 6:
        from .config_v6 import validate_v6_protocol

        try:
            protocol = validate_v6_protocol(protocol)
        except ValueError as exc:
            errors.append(f"V6 protocol contract failed: {exc}")
    elif protocol_version == 7:
        from .config_v7 import validate_v7_protocol

        try:
            protocol = validate_v7_protocol(protocol)
        except ValueError as exc:
            errors.append(f"V7 protocol contract failed: {exc}")

    if check_dependencies:
        required = ("yaml", "numpy", "torch", "torchvision")
        if mode in {"pilot", "full", "final-test"}:
            required += ("pandas", "scipy", "statsmodels")
        errors.extend(_check_dependencies(required))
        if protocol_version in (6, 7) and mode in {"pilot", "full", "final-test"}:
            errors.extend(
                _check_mechanism_torchvision_version(
                    protocol, protocol_version=protocol_version
                )
            )

    if protocol_version == 6 and mode in {"full", "final-test"}:
        from .config_v6 import validate_v6_cifar100_provenance_files

        try:
            validate_v6_cifar100_provenance_files(protocol, project_root=root)
        except ValueError as exc:
            errors.append(f"V6 CIFAR-100 provenance gate failed: {exc}")
    elif protocol_version == 7 and mode in {"full", "final-test"}:
        from .config_v7 import validate_v7_cifar100_provenance_files

        try:
            validate_v7_cifar100_provenance_files(protocol, project_root=root)
        except ValueError as exc:
            errors.append(f"V7 CIFAR-100 provenance gate failed: {exc}")

    status = protocol.get("protocol_status", {})
    if not isinstance(status, Mapping):
        errors.append("protocol_status must be a mapping")
        status = {}
    if status.get("frozen") is not True:
        warnings.append(
            "Protocol contains agreed draft values but is unsigned and unfrozen; "
            "formal training remains blocked"
        )
    # v4/v5 Phase A is an executable author freeze for health/pilot work. Older
    # pilot protocols intentionally retain their historical unfrozen behavior.
    author_freeze_required = mode in {"full", "final-test"} or (
        protocol_version in (4, 5, 6, 7) and mode == "pilot"
    )
    if author_freeze_required:
        if status.get("frozen") is not True:
            errors.append(
                "protocol_status.frozen must be true before this execution stage"
            )
        if not _nonempty(status.get("confirmed_by")):
            errors.append("protocol_status.confirmed_by is required")
        if not _nonempty(status.get("confirmed_at")):
            errors.append("protocol_status.confirmed_at is required (ISO-8601 recommended)")
        confirmations = status.get("confirmations", {})
        if not isinstance(confirmations, Mapping):
            confirmations = {}
        for field in confirmation_fields:
            if confirmations.get(field) is not True:
                errors.append(f"protocol_status.confirmations.{field} must be true")

    if mode in {"pilot", "full", "final-test"}:
        matrix = protocol.get("matrix", {})
        matrix_slots = matrix.get("primary", ()) if isinstance(matrix, Mapping) else ()
        matrix_datasets = {
            str(slot.get("dataset"))
            for slot in matrix_slots
            if isinstance(slot, Mapping) and slot.get("dataset")
        }
        analysis = protocol.get("analysis", {})
        if not isinstance(analysis, Mapping):
            analysis = {}
        thresholds = analysis.get("validation_accuracy_thresholds", {})
        if not isinstance(thresholds, Mapping):
            thresholds = {}
        for dataset in sorted(matrix_datasets):
            if thresholds.get(dataset) is None:
                errors.append(f"analysis.validation_accuracy_thresholds.{dataset} must be frozen")
        if protocol_version in (3, 4, 5):
            primary_test = analysis.get("primary_accuracy_test", {})
            if not isinstance(primary_test, Mapping):
                errors.append(
                    "analysis.primary_accuracy_test must be defined for the "
                    f"TA-LIF-only v{protocol_version} protocol"
                )
            elif primary_test.get("decision_rule") != "p_lt_0_05_and_mean_delta_gt_0":
                errors.append(
                    "analysis.primary_accuracy_test decision rule differs from the "
                    f"TA-LIF-only v{protocol_version} contract"
                )
            bootstrap = analysis.get("bootstrap", {})
            if not isinstance(bootstrap, Mapping) or bootstrap.get("resamples") != 10_000:
                errors.append(
                    "analysis.bootstrap must define the fixed 10000-resample "
                    f"TA-LIF-only v{protocol_version} contract"
                )
        elif protocol_version not in (6, 7):
            if analysis.get("interaction_practical_threshold_pp") is None:
                errors.append("analysis.interaction_practical_threshold_pp must be frozen")
            margins = analysis.get("efficiency_noninferiority_margins", {})
            if not isinstance(margins, Mapping) or not margins:
                errors.append("analysis.efficiency_noninferiority_margins must be defined")
            else:
                for metric, value in margins.items():
                    if value is None:
                        errors.append(
                            f"analysis.efficiency_noninferiority_margins.{metric} must be frozen"
                        )

        benchmark = protocol.get("benchmark", {})
        energy = benchmark.get("energy_model", {}) if isinstance(benchmark, Mapping) else {}
        if isinstance(energy, Mapping) and energy.get("status") == "modeled":
            constants_path = _resolve(root, energy.get("constants_path"))
            if constants_path is None or not constants_path.is_file():
                errors.append(f"Frozen energy constants are missing: {constants_path}")
            else:
                from .utils import sha256_file

                if sha256_file(constants_path) != energy.get("constants_sha256"):
                    errors.append("Frozen energy constants SHA-256 differs from protocol")

        dvs = protocol.get("datasets", {}).get("cifar10dvs", {})
        if "cifar10dvs" not in matrix_datasets:
            pass
        elif isinstance(dvs, Mapping):
            trainval = _resolve(root, dvs.get("frames_path"))
            test = _resolve(root, dvs.get("test_frames_path"))
            if trainval is None or not trainval.exists():
                errors.append(f"CIFAR10-DVS trainval frames are missing: {trainval}")
            if test is None or not test.exists():
                errors.append(f"CIFAR10-DVS test frames are missing: {test}")
            if trainval is not None and test is not None and trainval.exists() and test.exists():
                try:
                    common_data = protocol.get("data", {})
                    common_data = common_data if isinstance(common_data, Mapping) else {}
                    errors.extend(_check_dvs_provenance(root, trainval, test, common_data, dvs))
                except Exception as exc:
                    errors.append(f"CIFAR10-DVS index validation failed: {type(exc).__name__}: {exc}")
        else:
            errors.append("datasets.cifar10dvs must be configured")

    if protocol.get("data", {}).get("augment") is not True:
        warnings.append("Basic static-image crop/flip augmentation is disabled")
    if protocol.get("data", {}).get("autoaugment") is not True:
        warnings.append("AutoAugment is explicitly disabled in the current protocol")
    optimizer = protocol.get("optimizer", {})
    if not optimizer.get("cutmix_alpha"):
        warnings.append("CutMix is explicitly disabled in the current protocol")
    if not optimizer.get("label_smoothing"):
        warnings.append("Label smoothing is explicitly disabled in the current protocol")
    analysis = protocol.get("analysis", {})
    efficiency = analysis.get("efficiency", {}) if isinstance(analysis, Mapping) else {}
    if (
        isinstance(analysis, Mapping)
        and analysis.get("efficiency_assessment") == "descriptive"
    ) or (
        isinstance(efficiency, Mapping)
        and efficiency.get("role") == "descriptive_only"
    ):
        warnings.append(
            "Efficiency margins are descriptive reference bands, not confirmatory hypothesis tests"
        )

    confirmations = status.get("confirmations", {})
    freeze_fields_complete = (
        status.get("frozen") is True
        and _nonempty(status.get("confirmed_by"))
        and _nonempty(status.get("confirmed_at"))
        and isinstance(confirmations, Mapping)
        and all(confirmations.get(field) is True for field in confirmation_fields)
    )
    if mode in {"full", "final-test"} and freeze_fields_complete:
        try:
            verify_formal_freeze(project_root=root, protocol_path=path)
        except FreezeGateError as exc:
            errors.append(f"Phase B freeze manifest verification failed: {exc}")

    return PreflightReport(
        mode=mode,
        protocol_path=str(path),
        protocol_hash=_stable_hash(protocol),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
