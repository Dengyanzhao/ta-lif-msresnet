#!/usr/bin/env python
"""Convert tonic CIFAR10-DVS to an auditable lazy tensor dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from talif_msresnet.data import label_fingerprint, stratified_split_indices  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402
from talif_msresnet.utils import atomic_write_json, sha256_file, stable_hash, utc_now  # noqa: E402


INDEX_FIELDS = (
    "sample_id", "relative_path", "label", "split", "source_index", "sha256",
    "time_steps", "channels", "height", "width", "resize_mode", "split_seed",
    "test_fraction", "normalization", "tonic_dataset",
)

FLOAT32_BYTES = 4
MIN_DISK_HEADROOM_BYTES = 512 * 1024 * 1024
DISK_HEADROOM_FRACTION = 0.10
AEDAT4_HEADER = b"#!AER-DAT4.0\r\n"
SOURCE_MANIFEST_NAME = "source_manifest.json"
SOURCE_MANIFEST_SCHEMA = "cifar10dvs-aedat4-source-v1"
SOURCE_KIND = "preextracted_aedat4_third_party_conversion"
OFFICIAL_ARCHIVE_MD5 = "ce3a4a0682dc0943703bd8f749a7701c"
EXPECTED_SAMPLES_PER_CLASS = 1000
CIFAR10DVS_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


@dataclass(frozen=True)
class _AEDAT4SourceFile:
    source_index: int
    class_name: str
    label: int
    sample_number: int
    path: Path
    relative_path: str


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _discover_extracted_aedat4(
    root: str | Path,
    *,
    expected_per_class: int | None = None,
) -> list[_AEDAT4SourceFile]:
    """Validate and deterministically enumerate a pre-extracted AEDAT4 tree."""

    root = Path(root).resolve()
    expected = EXPECTED_SAMPLES_PER_CLASS if expected_per_class is None else expected_per_class
    if expected < 1:
        raise ValueError("expected_per_class must be positive")
    if not root.is_dir():
        raise RuntimeError(f"Extracted AEDAT4 root is missing: {root}")
    expected_root_names = {*CIFAR10DVS_CLASSES, "README.txt"}
    entries = list(root.iterdir())
    actual_root_names = {entry.name for entry in entries}
    if actual_root_names != expected_root_names:
        missing = sorted(expected_root_names - actual_root_names)
        extra = sorted(actual_root_names - expected_root_names)
        raise RuntimeError(
            "Extracted AEDAT4 root must contain exactly README.txt and the 10 standard "
            f"class directories; missing={missing}, extra={extra}"
        )
    readme = root / "README.txt"
    if not readme.is_file() or readme.is_symlink():
        raise RuntimeError("Extracted AEDAT4 README.txt must be a regular file")

    discovered: list[_AEDAT4SourceFile] = []
    for label, class_name in enumerate(CIFAR10DVS_CLASSES):
        class_dir = root / class_name
        if not class_dir.is_dir() or class_dir.is_symlink():
            raise RuntimeError(f"CIFAR10-DVS class path must be a regular directory: {class_dir}")
        children = list(class_dir.iterdir())
        pattern = re.compile(
            rf"cifar10_{re.escape(class_name)}_(0|[1-9][0-9]{{0,2}})\.aedat4"
        )
        by_number: dict[int, Path] = {}
        invalid: list[str] = []
        for child in children:
            match = pattern.fullmatch(child.name)
            if child.is_symlink() or not child.is_file() or match is None:
                invalid.append(child.name)
                continue
            number = int(match.group(1))
            if number in by_number:
                invalid.append(child.name)
                continue
            by_number[number] = child
        expected_numbers = set(range(expected))
        actual_numbers = set(by_number)
        if invalid or actual_numbers != expected_numbers:
            missing = sorted(expected_numbers - actual_numbers)
            extra = sorted(actual_numbers - expected_numbers)
            raise RuntimeError(
                f"Invalid extracted AEDAT4 class {class_name!r}: expected numbered files "
                f"0..{expected - 1}; missing={missing[:10]}, extra={extra[:10]}, "
                f"invalid={sorted(invalid)[:10]}"
            )
        for sample_number in range(expected):
            source_index = label * expected + sample_number
            path = by_number[sample_number]
            discovered.append(
                _AEDAT4SourceFile(
                    source_index=source_index,
                    class_name=class_name,
                    label=label,
                    sample_number=sample_number,
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                )
            )
    return discovered


def _validate_aedat4_events(value: Any, path: Path) -> np.ndarray:
    events = np.asarray(value)
    if events.ndim != 1 or len(events) == 0 or events.dtype.names is None:
        raise RuntimeError(f"AEDAT4 event stream must be a non-empty structured vector: {path}")
    names = tuple(events.dtype.names)
    if names == ("t", "x", "y", "on"):
        events.dtype.names = ("t", "x", "y", "p")
        names = tuple(events.dtype.names or ())
    if names != ("t", "x", "y", "p"):
        raise RuntimeError(f"AEDAT4 event fields must be exactly t,x,y,p: {path}: {names}")
    for field in ("t", "x", "y"):
        if not np.issubdtype(events[field].dtype, np.integer):
            raise RuntimeError(f"AEDAT4 {field} values must be integers: {path}")
    timestamps = events["t"]
    if bool(np.any(timestamps[1:] < timestamps[:-1])):
        raise RuntimeError(f"AEDAT4 timestamps are not monotonically non-decreasing: {path}")
    for field in ("x", "y"):
        values = events[field]
        if int(values.min()) < 0 or int(values.max()) >= 128:
            raise RuntimeError(f"AEDAT4 {field} coordinates are outside 0..127: {path}")
    polarities = events["p"]
    if bool(np.any((polarities != 0) & (polarities != 1))):
        raise RuntimeError(f"AEDAT4 polarity values are outside {{0,1}}: {path}")
    return events


class _ExtractedAEDAT4Dataset:
    sensor_size = (128, 128, 2)

    def __init__(self, files: Sequence[_AEDAT4SourceFile], reader: Any) -> None:
        self.files = list(files)
        self.targets = [entry.label for entry in self.files]
        self._reader = reader

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        entry = self.files[index]
        try:
            events = self._reader(str(entry.path))
        except ModuleNotFoundError as exc:
            if exc.name == "aedat":
                raise RuntimeError(
                    "AEDAT4 decoding requires the aedat package; install the project [event] extra"
                ) from exc
            raise
        return _validate_aedat4_events(events, entry.path), entry.label


def _build_source_manifest(
    root: Path,
    files: Sequence[_AEDAT4SourceFile],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for offset, entry in enumerate(files, start=1):
        size = entry.path.stat().st_size
        if size <= len(AEDAT4_HEADER):
            raise RuntimeError(f"AEDAT4 source file is empty or truncated: {entry.path}")
        with entry.path.open("rb") as handle:
            if handle.read(len(AEDAT4_HEADER)) != AEDAT4_HEADER:
                raise RuntimeError(f"AEDAT4 source header is invalid: {entry.path}")
        total_bytes += size
        records.append({
            "source_index": entry.source_index,
            "class": entry.class_name,
            "label": entry.label,
            "sample_number": entry.sample_number,
            "relative_path": entry.relative_path,
            "size_bytes": size,
            "sha256": sha256_file(entry.path),
        })
        if offset % 500 == 0 or offset == len(files):
            print(f"Hashed {offset}/{len(files)} AEDAT4 source files")

    readme_path = root / "README.txt"
    readme = {
        "relative_path": "README.txt",
        "size_bytes": readme_path.stat().st_size,
        "sha256": sha256_file(readme_path),
    }
    tree_fingerprint = stable_hash({
        "format": "AEDAT4",
        "class_order": list(CIFAR10DVS_CLASSES),
        "readme": readme,
        "files": records,
    })
    manifest: dict[str, Any] = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "source_kind": SOURCE_KIND,
        "format": "AEDAT4",
        "source_root": artifact_path_reference(root, PROJECT_ROOT),
        "official_archive_filename": "CIFAR10DVS.zip",
        "official_archive_expected_md5": OFFICIAL_ARCHIVE_MD5,
        "official_archive_byte_identity": "not_verified",
        "provenance_note": (
            "Third-party AEDAT 2.0 to AEDAT4 conversion; byte identity with the official "
            "Figshare CIFAR10DVS.zip was not verified."
        ),
        "class_order": list(CIFAR10DVS_CLASSES),
        "samples_per_class": EXPECTED_SAMPLES_PER_CLASS,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "ordering": "fixed_class_order_then_numeric_sample_number",
        "readme": readme,
        "files": records,
        "tree_fingerprint_sha256": tree_fingerprint,
    }
    manifest["manifest_sha256"] = stable_hash(manifest)
    return manifest


def _normalize_tonic_frames(value: Any, time_bins: int) -> torch.Tensor:
    frames = torch.as_tensor(value)
    if frames.ndim == 3:  # [T,H,W]
        frames = frames.unsqueeze(1)
    elif frames.ndim == 4:
        if frames.shape[0] == time_bins and frames.shape[1] in (1, 2):
            pass
        elif frames.shape[0] == time_bins and frames.shape[-1] in (1, 2):
            frames = frames.permute(0, 3, 1, 2)
        elif frames.shape[0] in (1, 2) and frames.shape[1] == time_bins:
            frames = frames.permute(1, 0, 2, 3)
        else:
            raise RuntimeError(f"Cannot interpret tonic frame shape {tuple(frames.shape)}")
    else:
        raise RuntimeError(f"Expected tonic frames with 3 or 4 dimensions, got {tuple(frames.shape)}")
    if int(frames.shape[0]) != int(time_bins):
        raise RuntimeError(f"ToFrame returned T={frames.shape[0]}, expected {time_bins}")
    return frames.float().contiguous()


def _dataset_labels(dataset: Any) -> list[int]:
    for attribute in ("targets", "labels"):
        values = getattr(dataset, attribute, None)
        if values is not None and len(values) == len(dataset):
            return [int(value) for value in values]
    labels: list[int] = []
    print("Dataset exposes no target vector; scanning labels once before conversion.")
    for index in range(len(dataset)):
        item = dataset[index]
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise RuntimeError(f"Unexpected CIFAR10DVS item at source index {index}")
        labels.append(int(item[1]))
    return labels


def _counts(labels: Sequence[int], indices: Sequence[int]) -> dict[str, int]:
    return {str(label): int(count) for label, count in sorted(Counter(labels[index] for index in indices).items())}


def _storage_preflight(
    output_parent: Path,
    *,
    sample_count: int,
    time_bins: int,
    channels: int,
    height: int,
    width: int,
) -> dict[str, int]:
    """Require room for float32 tensors plus serialization/filesystem headroom."""

    if sample_count < 1:
        raise RuntimeError("CIFAR10-DVS contains no samples")
    tensor_payload_bytes = (
        int(sample_count)
        * int(time_bins)
        * int(channels)
        * int(height)
        * int(width)
        * FLOAT32_BYTES
    )
    headroom_bytes = max(
        MIN_DISK_HEADROOM_BYTES,
        int(tensor_payload_bytes * DISK_HEADROOM_FRACTION),
    )
    required_free_bytes = tensor_payload_bytes + headroom_bytes
    available_free_bytes = int(shutil.disk_usage(output_parent).free)
    if available_free_bytes < required_free_bytes:
        raise RuntimeError(
            "Insufficient disk space for CIFAR10-DVS conversion: "
            f"need at least {required_free_bytes} bytes, found {available_free_bytes} bytes "
            f"on {output_parent}"
        )
    return {
        "estimated_tensor_payload_bytes": tensor_payload_bytes,
        "required_free_bytes": required_free_bytes,
        "available_free_bytes_at_start": available_free_bytes,
    }


def _new_staging_path(output: Path) -> Path:
    """Return a unique same-directory path so final rename stays on one volume."""

    token = f"{os.getpid()}-{uuid.uuid4().hex}"
    return output.parent / f".{output.name}.staging-{token}"


def _remove_staging(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def prepare(args: argparse.Namespace) -> Path:
    try:
        import tonic
    except ImportError as exc:
        raise RuntimeError("tonic is required; install the project with the [event] extra") from exc

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    aedat4_root_value = getattr(args, "aedat4_root", None)
    source_manifest: dict[str, Any] | None = None
    if aedat4_root_value:
        aedat4_root = Path(aedat4_root_value).resolve()
        source_files = _discover_extracted_aedat4(aedat4_root)
        reader = getattr(getattr(tonic, "io", None), "read_aedat4", None)
        if reader is None:
            raise RuntimeError("tonic.io.read_aedat4 is unavailable")
        dataset = _ExtractedAEDAT4Dataset(source_files, reader)
        source_manifest = _build_source_manifest(aedat4_root, source_files)
        source_root = aedat4_root
    else:
        data_root_value = getattr(args, "data_root", None)
        if not data_root_value:
            raise RuntimeError("Exactly one of data_root or aedat4_root is required")
        source_root = Path(data_root_value).resolve()
        dataset = tonic.datasets.CIFAR10DVS(save_to=str(source_root))

    labels = _dataset_labels(dataset)
    trainval_indices, test_indices = stratified_split_indices(labels, args.test_fraction, args.split_seed)
    trainval_set, test_set = set(trainval_indices), set(test_indices)
    if trainval_set & test_set or trainval_set | test_set != set(range(len(dataset))):
        raise RuntimeError("Internal split error: trainval/test sets are not a disjoint partition")

    sensor_size = getattr(dataset, "sensor_size", (128, 128, 2))
    transform = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=args.time_bins)
    storage = _storage_preflight(
        output.parent,
        sample_count=len(dataset),
        time_bins=args.time_bins,
        channels=2,
        height=args.height,
        width=args.width,
    )
    staging = _new_staging_path(output)
    staging_created = False
    try:
        staging.mkdir(exist_ok=False)
        staging_created = True
        if source_manifest is not None:
            atomic_write_json(staging / SOURCE_MANIFEST_NAME, source_manifest)
        samples_dir = staging / "samples"
        samples_dir.mkdir()
        rows: list[dict[str, Any]] = []
        serialized_sample_bytes = 0
        for source_index in range(len(dataset)):
            events, observed_label = dataset[source_index][:2]
            label = int(observed_label)
            if label != labels[source_index]:
                raise RuntimeError(
                    f"Label changed between scan and conversion at source index {source_index}"
                )
            frames = _normalize_tonic_frames(transform(events), args.time_bins)
            if int(frames.shape[1]) != 2:
                raise RuntimeError(
                    f"CIFAR10-DVS must retain two polarity channels, got C={frames.shape[1]}"
                )
            if tuple(frames.shape[-2:]) != (args.height, args.width):
                frames = F.interpolate(frames, size=(args.height, args.width), mode="nearest")
            sample_id = f"cifar10dvs_{source_index:05d}"
            relative_path = Path("samples") / f"{sample_id}.pt"
            sample_path = staging / relative_path
            torch.save(frames.contiguous(), sample_path)
            serialized_sample_bytes += sample_path.stat().st_size
            split = "test" if source_index in test_set else "trainval"
            rows.append({
                "sample_id": sample_id,
                "relative_path": relative_path.as_posix(),
                "label": label,
                "split": split,
                "source_index": source_index,
                "sha256": sha256_file(sample_path),
                "time_steps": int(frames.shape[0]),
                "channels": int(frames.shape[1]),
                "height": int(frames.shape[2]),
                "width": int(frames.shape[3]),
                "resize_mode": "nearest",
                "split_seed": args.split_seed,
                "test_fraction": args.test_fraction,
                "normalization": "none_raw_event_counts",
                "tonic_dataset": "CIFAR10DVS",
            })
            if (source_index + 1) % 100 == 0 or source_index + 1 == len(dataset):
                print(f"Converted {source_index + 1}/{len(dataset)} samples")

        index_path = staging / "index.csv"
        with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)

        split_manifest = {
            "format_version": 1,
            "dataset": "cifar10dvs",
            "source_length": len(dataset),
            "split_strategy": "stratified_by_label",
            "test_fraction": args.test_fraction,
            "split_seed": args.split_seed,
            "label_fingerprint_sha256": label_fingerprint(labels),
            "trainval_indices": trainval_indices,
            "test_indices": test_indices,
            "class_counts": {
                "all": _counts(labels, range(len(labels))),
                "trainval": _counts(labels, trainval_indices),
                "test": _counts(labels, test_indices),
            },
        }
        split_payload = json.dumps(
            split_manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        split_manifest["manifest_sha256"] = hashlib.sha256(
            split_payload.encode("utf-8")
        ).hexdigest()
        atomic_write_json(staging / "split_manifest.json", split_manifest)
        conversion_manifest: dict[str, Any] = {
            "format_version": 2 if source_manifest is not None else 1,
            "created_at": utc_now(),
            "dataset": "tonic.datasets.CIFAR10DVS",
            "tonic_version": getattr(tonic, "__version__", "unknown"),
            "data_root": artifact_path_reference(source_root, PROJECT_ROOT),
            "sample_count": len(rows),
            "index_sha256": sha256_file(index_path),
            "split_manifest_sha256": sha256_file(staging / "split_manifest.json"),
            "conversion": {
                "time_bins": args.time_bins,
                "height": args.height,
                "width": args.width,
                "resize_mode": "nearest",
                "sensor_size": list(sensor_size),
            },
            "partition": {"test_fraction": args.test_fraction, "split_seed": args.split_seed},
            "normalization": "none_raw_event_counts",
            "storage_preflight": {
                **storage,
                "serialized_sample_bytes": serialized_sample_bytes,
            },
        }
        if source_manifest is not None:
            source_manifest_path = staging / SOURCE_MANIFEST_NAME
            conversion_manifest["source"] = {
                "kind": SOURCE_KIND,
                "format": "AEDAT4",
                "loader": "tonic.io.read_aedat4",
                "decoder": {"package": "aedat", "version": _package_version("aedat")},
                "source_manifest": SOURCE_MANIFEST_NAME,
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "tree_fingerprint_sha256": source_manifest["tree_fingerprint_sha256"],
                "official_archive_byte_identity": "not_verified",
            }
        atomic_write_json(staging / "conversion_manifest.json", conversion_manifest)
        if output.exists():
            raise FileExistsError(f"Output appeared during conversion; refusing to replace it: {output}")
        os.replace(staging, output)
    except BaseException as exc:
        try:
            if staging_created:
                _remove_staging(staging)
        except OSError as cleanup_exc:
            raise RuntimeError(
                f"Conversion failed and staging cleanup also failed: {staging}: {cleanup_exc}"
            ) from exc
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-root", help="tonic official archive download/cache directory")
    source.add_argument(
        "--aedat4-root",
        help="Strict pre-extracted root containing README.txt and 10 AEDAT4 class directories",
    )
    parser.add_argument("--output", required=True, help="New output directory; existing paths are rejected")
    parser.add_argument("--time-bins", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--test-fraction", required=True, type=float)
    parser.add_argument("--split-seed", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.time_bins < 1 or args.height < 1 or args.width < 1:
        raise SystemExit("--time-bins, --height, and --width must be positive")
    if not 0 < args.test_fraction < 1:
        raise SystemExit("--test-fraction must be strictly between 0 and 1")
    if args.split_seed < 0:
        raise SystemExit("--split-seed must be non-negative")
    try:
        output = prepare(args)
    except Exception as exc:
        print(f"CIFAR10-DVS preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Prepared lazy CIFAR10-DVS dataset at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
