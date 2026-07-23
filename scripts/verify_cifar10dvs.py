#!/usr/bin/env python
"""Fail-closed full verification of a prepared CIFAR10-DVS folder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.data import label_fingerprint  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.utils import atomic_write_json, sha256_file, stable_hash, utc_now  # noqa: E402


REQUIRED_COLUMNS = {
    "sample_id", "relative_path", "label", "split", "source_index", "sha256",
    "time_steps", "channels", "height", "width", "resize_mode", "split_seed",
    "test_fraction", "normalization", "tonic_dataset",
}
EXTERNAL_PATH_REFERENCE_RE = re.compile(r"external:path-sha256:[0-9a-f]{64}")
VERIFICATION_RECEIPT_NAME = "verification_receipt.json"
VERIFICATION_RECEIPT_SCHEMA = "cifar10dvs-full-verification-v1"
SOURCE_MANIFEST_SCHEMA = "cifar10dvs-aedat4-source-v1"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SOURCE_KIND = "preextracted_aedat4_third_party_conversion"
OFFICIAL_ARCHIVE_MD5 = "ce3a4a0682dc0943703bd8f749a7701c"
CIFAR10DVS_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
EXPECTED_SAMPLES_PER_CLASS = 1000


class VerificationError(RuntimeError):
    pass


def _mapping_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VerificationError(f"Malformed {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"{label} must be a JSON object: {path}")
    return value


def _internal_manifest_hash(value: Mapping[str, Any]) -> str:
    unhashed = dict(value)
    unhashed.pop("manifest_sha256", None)
    payload = json.dumps(unhashed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_portable_path_reference(value: Any) -> bool:
    text = str(value or "").strip()
    if Path(text).is_absolute() or EXTERNAL_PATH_REFERENCE_RE.fullmatch(text) is not None:
        return True
    if not text or text.startswith("external:"):
        return False
    return all(part not in {"", ".", ".."} for part in Path(text).parts)


def _verify_extracted_source(root: Path, conversion: Mapping[str, Any]) -> None:
    """Verify the self-contained inventory for a third-party AEDAT4 conversion."""

    source = conversion.get("source")
    if not isinstance(source, Mapping):
        raise VerificationError("Version 2 conversion manifest has no source metadata")
    expected_source = {
        "kind": SOURCE_KIND,
        "format": "AEDAT4",
        "loader": "tonic.io.read_aedat4",
        "source_manifest": SOURCE_MANIFEST_NAME,
        "official_archive_byte_identity": "not_verified",
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise VerificationError(f"Conversion source {key} is invalid")
    decoder = source.get("decoder")
    if (
        not isinstance(decoder, Mapping)
        or decoder.get("package") != "aedat"
        or not str(decoder.get("version", "")).strip()
        or decoder.get("version") == "unknown"
    ):
        raise VerificationError("Conversion source AEDAT4 decoder metadata is invalid")

    source_path = root / SOURCE_MANIFEST_NAME
    if not source_path.is_file():
        raise VerificationError(f"Missing AEDAT4 source manifest: {source_path}")
    if source.get("source_manifest_sha256") != sha256_file(source_path):
        raise VerificationError("AEDAT4 source manifest SHA-256 differs from conversion manifest")
    manifest = _mapping_json(source_path, "AEDAT4 source manifest")
    if manifest.get("manifest_sha256") != _internal_manifest_hash(manifest):
        raise VerificationError("AEDAT4 source manifest failed its internal SHA-256 check")
    expected_manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "source_kind": SOURCE_KIND,
        "format": "AEDAT4",
        "official_archive_filename": "CIFAR10DVS.zip",
        "official_archive_expected_md5": OFFICIAL_ARCHIVE_MD5,
        "official_archive_byte_identity": "not_verified",
        "class_order": list(CIFAR10DVS_CLASSES),
        "samples_per_class": EXPECTED_SAMPLES_PER_CLASS,
        "file_count": len(CIFAR10DVS_CLASSES) * EXPECTED_SAMPLES_PER_CLASS,
        "ordering": "fixed_class_order_then_numeric_sample_number",
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise VerificationError(f"AEDAT4 source manifest {key} is invalid")
    if not str(manifest.get("created_at", "")).strip():
        raise VerificationError("AEDAT4 source manifest has no created_at timestamp")
    if not _is_portable_path_reference(manifest.get("source_root")):
        raise VerificationError("AEDAT4 source manifest source_root is invalid")
    if "not verified" not in str(manifest.get("provenance_note", "")).lower():
        raise VerificationError("AEDAT4 source manifest provenance disclosure is missing")

    readme = manifest.get("readme")
    if (
        not isinstance(readme, Mapping)
        or readme.get("relative_path") != "README.txt"
        or not isinstance(readme.get("size_bytes"), int)
        or readme.get("size_bytes", 0) <= 0
        or not _is_sha256(readme.get("sha256"))
    ):
        raise VerificationError("AEDAT4 source manifest README record is invalid")
    files = manifest.get("files")
    expected_count = len(CIFAR10DVS_CLASSES) * EXPECTED_SAMPLES_PER_CLASS
    if not isinstance(files, list) or len(files) != expected_count:
        raise VerificationError("AEDAT4 source manifest file inventory count is invalid")
    total_bytes = 0
    for source_index, record in enumerate(files):
        if not isinstance(record, Mapping):
            raise VerificationError(f"AEDAT4 source record {source_index} is not a mapping")
        label, sample_number = divmod(source_index, EXPECTED_SAMPLES_PER_CLASS)
        class_name = CIFAR10DVS_CLASSES[label]
        expected_record = {
            "source_index": source_index,
            "class": class_name,
            "label": label,
            "sample_number": sample_number,
            "relative_path": f"{class_name}/cifar10_{class_name}_{sample_number}.aedat4",
        }
        for key, expected in expected_record.items():
            if record.get(key) != expected:
                raise VerificationError(
                    f"AEDAT4 source record {source_index} has invalid {key}"
                )
        size = record.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise VerificationError(
                f"AEDAT4 source record {source_index} has invalid size_bytes"
            )
        if not _is_sha256(record.get("sha256")):
            raise VerificationError(f"AEDAT4 source record {source_index} has invalid SHA-256")
        total_bytes += size
    if manifest.get("total_bytes") != total_bytes:
        raise VerificationError("AEDAT4 source manifest total_bytes differs from inventory")
    tree_fingerprint = stable_hash({
        "format": "AEDAT4",
        "class_order": list(CIFAR10DVS_CLASSES),
        "readme": dict(readme),
        "files": files,
    })
    if manifest.get("tree_fingerprint_sha256") != tree_fingerprint:
        raise VerificationError("AEDAT4 source manifest tree fingerprint is invalid")
    if source.get("tree_fingerprint_sha256") != tree_fingerprint:
        raise VerificationError("Conversion source tree fingerprint differs from source manifest")


def _integer(row: Mapping[str, str], key: str, row_number: int) -> int:
    try:
        return int(row.get(key, ""))
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"Invalid {key} at index.csv row {row_number}") from exc


def _number(row: Mapping[str, str], key: str, row_number: int) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError) as exc:
        raise VerificationError(f"Invalid {key} at index.csv row {row_number}") from exc


def _class_counts(labels: Sequence[int], indices: Sequence[int] | None = None) -> dict[str, int]:
    selected = labels if indices is None else [labels[index] for index in indices]
    return {str(label): int(count) for label, count in sorted(Counter(selected).items())}


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only
        value = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise VerificationError(f"Cannot load sample tensor: {path}: {exc}") from exc
    if not isinstance(value, torch.Tensor):
        raise VerificationError(f"Sample is not a tensor: {path}")
    return value


def verify_dataset(root: str | Path, *, progress_every: int = 100) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise VerificationError(f"Prepared CIFAR10-DVS root is missing: {root}")

    index_path = root / "index.csv"
    conversion_path = root / "conversion_manifest.json"
    split_path = root / "split_manifest.json"
    conversion = _mapping_json(conversion_path, "conversion manifest")
    split_manifest = _mapping_json(split_path, "split manifest")
    if not index_path.is_file():
        raise VerificationError(f"Missing index.csv: {index_path}")
    conversion_version = conversion.get("format_version")
    if conversion_version not in {1, 2} or split_manifest.get("format_version") != 1:
        raise VerificationError("Unsupported conversion or split manifest format_version")
    if conversion.get("dataset") != "tonic.datasets.CIFAR10DVS":
        raise VerificationError("Conversion manifest dataset is not tonic.datasets.CIFAR10DVS")
    if conversion.get("normalization") != "none_raw_event_counts":
        raise VerificationError("Conversion manifest normalization is not none_raw_event_counts")
    if not str(conversion.get("tonic_version", "")).strip():
        raise VerificationError("Conversion manifest has no tonic_version")
    if not str(conversion.get("created_at", "")).strip():
        raise VerificationError("Conversion manifest has no created_at timestamp")
    data_root = str(conversion.get("data_root", "")).strip()
    if not _is_portable_path_reference(data_root):
        raise VerificationError(
            "Conversion manifest data_root must be a repository-relative path, legacy "
            "absolute path, or external:path-sha256 reference"
        )
    if conversion_version == 2:
        _verify_extracted_source(root, conversion)
    elif conversion.get("source") is not None:
        raise VerificationError("Version 1 conversion manifest must not contain version 2 source metadata")
    if conversion.get("index_sha256") != sha256_file(index_path):
        raise VerificationError("index.csv SHA-256 differs from conversion manifest")
    if conversion.get("split_manifest_sha256") != sha256_file(split_path):
        raise VerificationError("split_manifest.json SHA-256 differs from conversion manifest")
    if split_manifest.get("manifest_sha256") != _internal_manifest_hash(split_manifest):
        raise VerificationError("split_manifest.json failed its internal SHA-256 check")

    conversion_spec = conversion.get("conversion")
    partition = conversion.get("partition")
    if not isinstance(conversion_spec, Mapping) or not isinstance(partition, Mapping):
        raise VerificationError("Conversion manifest conversion/partition metadata is missing")
    try:
        expected_shape = (
            int(conversion_spec["time_bins"]),
            2,
            int(conversion_spec["height"]),
            int(conversion_spec["width"]),
        )
        expected_split_seed = int(partition["split_seed"])
        expected_test_fraction = float(partition["test_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise VerificationError("Conversion manifest contains invalid shape/partition metadata") from exc
    if min(expected_shape) < 1 or not 0 < expected_test_fraction < 1:
        raise VerificationError("Conversion manifest contains out-of-range shape/partition metadata")
    if conversion_spec.get("resize_mode") != "nearest":
        raise VerificationError("Conversion manifest resize_mode is not nearest")
    sensor_size = conversion_spec.get("sensor_size")
    if (
        not isinstance(sensor_size, list)
        or len(sensor_size) != 3
        or any(not isinstance(value, int) or value < 1 for value in sensor_size)
        or sensor_size[-1] != 2
    ):
        raise VerificationError("Conversion manifest sensor_size is invalid")

    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise VerificationError(f"index.csv is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise VerificationError("index.csv contains no samples")
    if conversion.get("sample_count") != len(rows):
        raise VerificationError("Conversion sample_count differs from index.csv")

    by_source: dict[int, tuple[int, str]] = {}
    sample_ids: set[str] = set()
    indexed_paths: set[Path] = set()
    serialized_sample_bytes = 0
    expected_row_shape = {
        "time_steps": expected_shape[0],
        "channels": expected_shape[1],
        "height": expected_shape[2],
        "width": expected_shape[3],
    }
    for row_number, row in enumerate(rows, start=2):
        sample_id = row.get("sample_id", "").strip()
        if not sample_id or sample_id in sample_ids:
            raise VerificationError(f"Duplicate or empty sample_id at index.csv row {row_number}")
        relative_text = row.get("relative_path", "").strip()
        relative = Path(relative_text)
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "samples":
            raise VerificationError(f"Invalid sample relative_path at index.csv row {row_number}")
        sample_path = (root / relative).resolve()
        try:
            sample_path.relative_to(root)
        except ValueError as exc:
            raise VerificationError(f"Sample path escapes dataset root at row {row_number}") from exc
        if sample_path.suffix.lower() != ".pt" or sample_path in indexed_paths:
            raise VerificationError(f"Duplicate or non-.pt sample path at row {row_number}")
        if not sample_path.is_file():
            raise VerificationError(f"Missing sample file at index.csv row {row_number}: {sample_path}")

        source_index = _integer(row, "source_index", row_number)
        label = _integer(row, "label", row_number)
        split = row.get("split", "").strip().lower()
        if source_index in by_source or source_index < 0:
            raise VerificationError(f"Duplicate or invalid source_index at row {row_number}")
        if not 0 <= label < 10:
            raise VerificationError(f"Out-of-range CIFAR10-DVS label at row {row_number}")
        if split not in {"trainval", "test"}:
            raise VerificationError(f"Invalid split at index.csv row {row_number}")
        if sample_id != f"cifar10dvs_{source_index:05d}":
            raise VerificationError(f"sample_id/source_index mismatch at index.csv row {row_number}")
        for key, expected in expected_row_shape.items():
            if _integer(row, key, row_number) != expected:
                raise VerificationError(f"{key} metadata mismatch at index.csv row {row_number}")
        if _integer(row, "split_seed", row_number) != expected_split_seed:
            raise VerificationError(f"split_seed metadata mismatch at index.csv row {row_number}")
        if _number(row, "test_fraction", row_number) != expected_test_fraction:
            raise VerificationError(f"test_fraction metadata mismatch at index.csv row {row_number}")
        if row.get("resize_mode") != conversion_spec.get("resize_mode"):
            raise VerificationError(f"resize_mode metadata mismatch at index.csv row {row_number}")
        if row.get("normalization") != conversion.get("normalization"):
            raise VerificationError(f"normalization metadata mismatch at index.csv row {row_number}")
        if row.get("tonic_dataset") != "CIFAR10DVS":
            raise VerificationError(f"tonic_dataset metadata mismatch at index.csv row {row_number}")

        recorded_hash = row.get("sha256", "").strip().lower()
        if len(recorded_hash) != 64 or any(ch not in "0123456789abcdef" for ch in recorded_hash):
            raise VerificationError(f"Invalid sample SHA-256 at index.csv row {row_number}")
        observed_hash = sha256_file(sample_path)
        if observed_hash != recorded_hash:
            raise VerificationError(f"Sample SHA-256 mismatch: {sample_path}")
        tensor = _load_tensor(sample_path)
        if tuple(tensor.shape) != expected_shape:
            raise VerificationError(
                f"Sample tensor shape mismatch: {sample_path}: {tuple(tensor.shape)} != {expected_shape}"
            )
        if tensor.dtype != torch.float32:
            raise VerificationError(f"Sample tensor dtype is not float32: {sample_path}: {tensor.dtype}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise VerificationError(f"Sample tensor contains non-finite values: {sample_path}")
        if bool((tensor < 0).any().item()):
            raise VerificationError(f"Sample tensor contains negative event counts: {sample_path}")

        sample_ids.add(sample_id)
        indexed_paths.add(sample_path)
        by_source[source_index] = (label, split)
        serialized_sample_bytes += sample_path.stat().st_size
        verified = row_number - 1
        if progress_every > 0 and (verified % progress_every == 0 or verified == len(rows)):
            print(f"Verified {verified}/{len(rows)} samples")

    if set(by_source) != set(range(len(rows))):
        raise VerificationError("source_index values are not a unique complete range")
    actual_sample_paths = {path.resolve() for path in (root / "samples").rglob("*.pt")}
    if actual_sample_paths != indexed_paths:
        raise VerificationError("samples directory contains unindexed or missing .pt files")

    labels = [by_source[index][0] for index in range(len(rows))]
    trainval_indices = [index for index in range(len(rows)) if by_source[index][1] == "trainval"]
    test_indices = [index for index in range(len(rows)) if by_source[index][1] == "test"]
    expected_split = {
        "dataset": "cifar10dvs",
        "source_length": len(rows),
        "split_strategy": "stratified_by_label",
        "test_fraction": expected_test_fraction,
        "split_seed": expected_split_seed,
        "label_fingerprint_sha256": label_fingerprint(labels),
        "trainval_indices": trainval_indices,
        "test_indices": test_indices,
        "class_counts": {
            "all": _class_counts(labels),
            "trainval": _class_counts(labels, trainval_indices),
            "test": _class_counts(labels, test_indices),
        },
    }
    for key, expected in expected_split.items():
        if split_manifest.get(key) != expected:
            raise VerificationError(f"Split manifest {key} differs from fully loaded samples/index")

    storage = conversion.get("storage_preflight")
    if not isinstance(storage, Mapping):
        raise VerificationError("Conversion manifest has no storage_preflight record")
    expected_payload_bytes = len(rows) * expected_shape[0] * expected_shape[1]
    expected_payload_bytes *= expected_shape[2] * expected_shape[3] * 4
    if storage.get("estimated_tensor_payload_bytes") != expected_payload_bytes:
        raise VerificationError("Estimated tensor payload differs from conversion manifest")
    required_free = storage.get("required_free_bytes")
    available_free = storage.get("available_free_bytes_at_start")
    if (
        not isinstance(required_free, int)
        or not isinstance(available_free, int)
        or required_free < expected_payload_bytes
        or available_free < required_free
    ):
        raise VerificationError("Conversion storage preflight record is invalid")
    if storage.get("serialized_sample_bytes") != serialized_sample_bytes:
        raise VerificationError("Serialized sample byte count differs from conversion manifest")
    return {
        "root": str(root),
        "sample_count": len(rows),
        "trainval_count": len(trainval_indices),
        "test_count": len(test_indices),
        "expected_shape": list(expected_shape),
        "serialized_sample_bytes": serialized_sample_bytes,
        "conversion_manifest_sha256": sha256_file(conversion_path),
        "index_sha256": sha256_file(index_path),
        "split_manifest_sha256": sha256_file(split_path),
        "status": "verified",
    }


def write_verification_receipt(root: str | Path, summary: Mapping[str, Any]) -> Path:
    """Persist a small, hash-bound proof that ``verify_dataset`` completed."""

    root_path = Path(root).resolve()
    required = (
        "sample_count", "trainval_count", "test_count", "expected_shape",
        "serialized_sample_bytes", "conversion_manifest_sha256", "index_sha256",
        "split_manifest_sha256",
    )
    missing = [key for key in required if key not in summary]
    if missing:
        raise VerificationError(f"Cannot create verification receipt; summary is missing {missing}")
    receipt: dict[str, Any] = {
        "schema": VERIFICATION_RECEIPT_SCHEMA,
        "status": "pass",
        "verified_at": utc_now(),
        "verifier": {"name": "scripts/verify_cifar10dvs.py", "version": 1},
        "root": artifact_path_reference(root_path, PROJECT_ROOT),
        **{key: summary[key] for key in required},
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    path = root_path / VERIFICATION_RECEIPT_NAME
    atomic_write_json(path, receipt)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(PROJECT_ROOT / "data" / "cifar10dvs" / "processed"),
        help="Prepared folder containing index.csv, manifests, and samples/",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--json", action="store_true", help="Print only the JSON summary")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be non-negative")
    try:
        summary = verify_dataset(
            args.root,
            progress_every=0 if args.json else args.progress_every,
        )
        receipt_path = write_verification_receipt(args.root, summary)
    except Exception as exc:
        print(f"CIFAR10-DVS verification failed: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(
            f"PASS: fully verified {summary['sample_count']} CIFAR10-DVS samples "
            f"under {summary['root']}"
        )
        print(f"Wrote verification receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
