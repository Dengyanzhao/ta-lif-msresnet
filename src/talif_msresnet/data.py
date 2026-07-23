"""Reproducible dataset and DataLoader construction.

Static CIFAR data uses one persisted train/validation manifest per dataset and
split seed.  The test split is not returned unless ``final_test=True``.  DVS
loaders accept the preprocessed tensor/NPZ formats commonly produced by event
conversion scripts and normalize them to ``[T, C, H, W]`` per sample.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

from .utils import atomic_write_json, sha256_file, worker_seed_fn


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class DataError(RuntimeError):
    pass


def load_representative_batch_artifact(
    path: str | Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Load the auditable validation-batch artifact used by E3/benchmarks."""

    source = Path(path)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise DataError(
            "Representative batch must be the mapping written by "
            "scripts/export_representative_batch.py"
        )
    inputs = payload.get("inputs")
    targets = payload.get("targets")
    metadata = payload.get("metadata")
    if not isinstance(inputs, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise DataError("Representative batch must contain inputs and targets tensors")
    if not isinstance(metadata, Mapping):
        raise DataError("Representative batch must contain an auditable metadata mapping")
    if inputs.shape[0] != targets.shape[0] or inputs.shape[0] < 1:
        raise DataError("Representative inputs/targets have an invalid batch dimension")
    return inputs.contiguous(), targets.long().contiguous(), dict(metadata)


def validate_representative_batch_artifact(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    metadata: Mapping[str, Any],
    *,
    expected_dataset: str,
    expected_protocol_hash: str,
    expected_split_manifest_sha256: str,
    expected_in_channels: int,
    expected_time_steps: int,
    expected_num_classes: int,
) -> str:
    """Validate provenance and shape, returning the full batch content hash."""

    from .diagnostics import representative_batch_sha256

    expected_dataset = str(expected_dataset).lower().replace("-", "").replace("_", "")
    observed_dataset = str(metadata.get("dataset", "")).lower().replace("-", "").replace("_", "")
    if observed_dataset != expected_dataset:
        raise DataError(
            f"Representative batch dataset mismatch: {observed_dataset!r} != {expected_dataset!r}"
        )
    if metadata.get("source_split") != "validation" or metadata.get("test_data_accessed") is not False:
        raise DataError("Representative batch must come exclusively from the validation split")
    if not expected_protocol_hash or metadata.get("protocol_hash") != expected_protocol_hash:
        raise DataError("Representative batch protocol hash differs from the checkpoint protocol")
    if (
        not expected_split_manifest_sha256
        or metadata.get("split_manifest_sha256") != expected_split_manifest_sha256
    ):
        raise DataError("Representative batch split-manifest hash differs from the training run")

    if expected_dataset == "cifar10dvs":
        if inputs.ndim != 5:
            raise DataError("CIFAR10-DVS representative inputs must be [N,T,C,H,W]")
        observed_time_steps, observed_channels = int(inputs.shape[1]), int(inputs.shape[2])
        if observed_time_steps != int(expected_time_steps):
            raise DataError(
                f"Representative batch time-step mismatch: {observed_time_steps} != {expected_time_steps}"
            )
    else:
        if inputs.ndim != 4:
            raise DataError("Static CIFAR representative inputs must be [N,C,H,W]")
        observed_channels = int(inputs.shape[1])
    if observed_channels != int(expected_in_channels):
        raise DataError(
            f"Representative batch channel mismatch: {observed_channels} != {expected_in_channels}"
        )
    if metadata.get("in_channels") != observed_channels:
        raise DataError("Representative batch channel metadata differs from its tensor")
    if expected_dataset == "cifar10dvs" and metadata.get("time_steps") != observed_time_steps:
        raise DataError("Representative batch time-step metadata differs from its tensor")
    if metadata.get("num_classes") != int(expected_num_classes):
        raise DataError("Representative batch class-count metadata differs from the checkpoint")
    if targets.ndim != 1:
        raise DataError("Representative targets must be a one-dimensional class-index tensor")
    if targets.numel() and (
        int(targets.min().item()) < 0 or int(targets.max().item()) >= int(expected_num_classes)
    ):
        raise DataError("Representative targets fall outside the checkpoint class range")

    content_hash = representative_batch_sha256(inputs, targets)
    if metadata.get("representative_batch_sha256") != content_hash:
        raise DataError("Representative batch content hash does not match its metadata")
    recorded_shape = metadata.get("input_shape")
    if recorded_shape is not None and list(recorded_shape) != list(inputs.shape):
        raise DataError("Representative batch input shape differs from its metadata")
    return content_hash


def _manifest_fingerprint(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def label_fingerprint(labels: Sequence[int] | np.ndarray | torch.Tensor) -> str:
    """Hash labels in source order using a platform-independent int64 encoding."""

    values = _normalize_labels(labels)
    array = np.asarray(values, dtype="<i8")
    digest = hashlib.sha256()
    digest.update(str(len(values)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _normalize_labels(labels: Sequence[int] | np.ndarray | torch.Tensor) -> List[int]:
    if isinstance(labels, torch.Tensor):
        value = labels.detach().cpu()
        if value.ndim > 1:
            value = value.argmax(dim=-1)
        return [int(item) for item in value.reshape(-1).tolist()]
    array = np.asarray(labels)
    if array.ndim > 1:
        array = array.argmax(axis=-1)
    return [int(item) for item in array.reshape(-1).tolist()]


def _class_counts(labels: Sequence[int], indices: Sequence[int] | None = None) -> Dict[str, int]:
    selected = labels if indices is None else [labels[index] for index in indices]
    return {str(label): int(count) for label, count in sorted(Counter(selected).items())}


def stratified_split_indices(
    labels: Sequence[int] | np.ndarray | torch.Tensor,
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Create a deterministic split with per-class train and validation support.

    Classes with at least two samples contribute to both subsets. Singleton
    classes remain in training because placing one in validation would remove
    that class from model fitting.
    """

    values = _normalize_labels(labels)
    if len(values) < 2:
        raise DataError("A stratified split requires at least two labels")
    if not 0 < float(val_fraction) < 1:
        raise DataError("val_fraction must be strictly between 0 and 1")
    groups: Dict[int, List[int]] = defaultdict(list)
    for index, label in enumerate(values):
        groups[label].append(index)
    rng = random.Random(int(seed))
    for label in sorted(groups):
        rng.shuffle(groups[label])

    minima = {label: 1 if len(indices) >= 2 else 0 for label, indices in groups.items()}
    maxima = {label: len(indices) - 1 if len(indices) >= 2 else 0 for label, indices in groups.items()}
    capacity = sum(maxima.values())
    if capacity < 1:
        raise DataError("Stratification cannot create a non-empty validation set from singleton classes")
    desired = max(sum(minima.values()), int(round(len(values) * float(val_fraction))))
    desired = min(capacity, desired)
    quotas = {
        label: min(maxima[label], max(minima[label], int(np.floor(len(indices) * float(val_fraction)))))
        for label, indices in groups.items()
    }
    while sum(quotas.values()) < desired:
        candidates = [label for label in sorted(groups) if quotas[label] < maxima[label]]
        label = max(candidates, key=lambda item: (len(groups[item]) * val_fraction - quotas[item], -item))
        quotas[label] += 1
    while sum(quotas.values()) > desired:
        candidates = [label for label in sorted(groups) if quotas[label] > minima[label]]
        label = min(candidates, key=lambda item: (len(groups[item]) * val_fraction - quotas[item], item))
        quotas[label] -= 1

    val_indices = sorted(index for label, indices in groups.items() for index in indices[:quotas[label]])
    val_set = set(val_indices)
    train_indices = [index for index in range(len(values)) if index not in val_set]
    return train_indices, val_indices


def create_split_manifest(
    length: int,
    val_fraction: float,
    seed: int,
    path: str | Path,
    dataset: str,
    source_fingerprint: str | None = None,
    labels: Sequence[int] | np.ndarray | torch.Tensor | None = None,
) -> Dict[str, Any]:
    """Create or validate a deterministic, preferably stratified manifest."""

    if length < 2:
        raise DataError("A train/validation split requires at least two samples")
    if not 0 < float(val_fraction) < 1:
        raise DataError("val_fraction must be strictly between 0 and 1")
    normalized_labels = _normalize_labels(labels) if labels is not None else None
    if normalized_labels is not None and len(normalized_labels) != length:
        raise DataError(f"Expected {length} labels, got {len(normalized_labels)}")
    strategy = "stratified_by_label" if normalized_labels is not None else "random"
    labels_sha256 = label_fingerprint(normalized_labels) if normalized_labels is not None else None
    path = Path(path)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"Cannot read split manifest {path}: {exc}") from exc
        expected = {
            "dataset": dataset,
            "source_length": length,
            "val_fraction": float(val_fraction),
            "seed": int(seed),
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise DataError(
                    f"Split manifest {path} does not match current data ({key}: "
                    f"{manifest.get(key)!r} != {value!r}); create a new manifest"
                )
        if source_fingerprint is not None and manifest.get("source_fingerprint") != source_fingerprint:
            raise DataError(f"Split manifest {path} source fingerprint differs")
        if manifest.get("split_strategy") != strategy:
            raise DataError(
                f"Split manifest {path} uses {manifest.get('split_strategy')!r}, expected {strategy!r}"
            )
        _validate_indices(manifest.get("train_indices"), manifest.get("val_indices"), length)
        if normalized_labels is not None:
            if manifest.get("label_fingerprint_sha256") != labels_sha256:
                raise DataError(f"Split manifest {path} label fingerprint differs")
            expected_counts = {
                "all": _class_counts(normalized_labels),
                "train": _class_counts(normalized_labels, manifest.get("train_indices", [])),
                "val": _class_counts(normalized_labels, manifest.get("val_indices", [])),
            }
            if manifest.get("class_counts") != expected_counts:
                raise DataError(f"Split manifest {path} class counts differ from current labels")
        recorded_hash = manifest.get("manifest_sha256")
        if not isinstance(recorded_hash, str) or not recorded_hash:
            raise DataError(f"Split manifest {path} has no SHA-256 integrity field")
        unhashed = dict(manifest)
        unhashed.pop("manifest_sha256", None)
        if recorded_hash != _manifest_fingerprint(unhashed):
            raise DataError(f"Split manifest {path} failed its SHA-256 integrity check")
        return manifest

    if normalized_labels is not None:
        train_indices, val_indices = stratified_split_indices(normalized_labels, val_fraction, seed)
    else:
        indices = list(range(length))
        random.Random(int(seed)).shuffle(indices)
        val_count = max(1, min(length - 1, int(round(length * float(val_fraction)))))
        val_indices = sorted(indices[:val_count])
        train_indices = sorted(indices[val_count:])
    manifest: Dict[str, Any] = {
        "format_version": 2,
        "dataset": dataset,
        "source_length": int(length),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "split_strategy": strategy,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }
    if normalized_labels is not None:
        manifest["label_fingerprint_sha256"] = labels_sha256
        manifest["class_counts"] = {
            "all": _class_counts(normalized_labels),
            "train": _class_counts(normalized_labels, train_indices),
            "val": _class_counts(normalized_labels, val_indices),
        }
    if source_fingerprint is not None:
        manifest["source_fingerprint"] = source_fingerprint
    manifest["manifest_sha256"] = _manifest_fingerprint(manifest)
    atomic_write_json(path, manifest)
    return manifest


def _validate_indices(train_indices: Any, val_indices: Any, length: int) -> None:
    if not isinstance(train_indices, list) or not isinstance(val_indices, list):
        raise DataError("Split manifest indices must be lists")
    train = [int(x) for x in train_indices]
    val = [int(x) for x in val_indices]
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise DataError("Split manifest contains duplicate indices")
    if set(train) & set(val) or set(train) | set(val) != set(range(length)):
        raise DataError("Split manifest indices do not form a partition")


def _resolve_manifest_path(config: Any, dataset: str) -> Path:
    value = getattr(config, "split_manifest", None)
    if value is None:
        value = f"data/manifests/{dataset}_seed{getattr(config, 'split_seed', 2024)}.json"
    return Path(str(value).format(dataset=dataset, split_seed=getattr(config, "split_seed", 2024)))


class TransformDataset(Dataset):
    def __init__(self, base: Dataset, indices: Sequence[int], transform: Any = None) -> None:
        self.base = base
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        item = self.base[self.indices[index]]
        if isinstance(item, tuple):
            value, label = item[0], item[1]
            if self.transform is not None:
                value = self.transform(value)
            return value, int(label)
        if self.transform is not None:
            item = self.transform(item)
        return item


def _normalize_frame_sample(
    frames: Any,
    expected_time_steps: int | None = None,
    expected_channels: int | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(frames)
    if tensor.ndim == 3:  # [T,H,W]
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise DataError(f"DVS sample must be [T,C,H,W] or [T,H,W], got {tuple(tensor.shape)}")
    if expected_time_steps is not None and int(tensor.shape[0]) != int(expected_time_steps):
        raise DataError(
            f"DVS time-bin mismatch: data has T={tensor.shape[0]}, run requires T={expected_time_steps}"
        )
    if expected_channels is not None and int(tensor.shape[1]) != int(expected_channels):
        raise DataError(
            f"DVS channel mismatch: data has C={tensor.shape[1]}, run requires C={expected_channels}"
        )
    return tensor.float().contiguous()


class TensorFrameDataset(Dataset):
    """Dataset for preprocessed DVS frames with per-sample temporal tensors."""

    def __init__(
        self,
        frames: torch.Tensor,
        labels: torch.Tensor,
        expected_time_steps: int | None = None,
        expected_channels: int | None = None,
    ) -> None:
        if frames.ndim not in (4, 5):
            raise DataError(f"DVS frames must have 4 or 5 dimensions, got {tuple(frames.shape)}")
        if frames.ndim == 4:  # [N, T, H, W] -> one event channel
            frames = frames.unsqueeze(2)
        if labels.ndim > 1:
            labels = labels.argmax(dim=-1)
        if frames.shape[0] != labels.shape[0]:
            raise DataError("DVS frames and labels have different sample counts")
        if expected_time_steps is not None and int(frames.shape[1]) != int(expected_time_steps):
            raise DataError(
                f"DVS time-bin mismatch: data has T={frames.shape[1]}, run requires T={expected_time_steps}"
            )
        if expected_channels is not None and int(frames.shape[2]) != int(expected_channels):
            raise DataError(
                f"DVS channel mismatch: data has C={frames.shape[2]}, run requires C={expected_channels}"
            )
        self.frames = frames.float().contiguous()
        self.labels = labels.long().contiguous()

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.frames[index], int(self.labels[index])


class IndexedFrameDataset(Dataset):
    """Lazy DVS dataset backed by ``index.csv`` and one tensor file per sample."""

    REQUIRED_COLUMNS = {"sample_id", "label"}

    def __init__(
        self,
        path: str | Path,
        split: str | None = None,
        expected_time_steps: int | None = None,
        expected_channels: int | None = None,
        verify_hashes: bool = True,
    ) -> None:
        source = Path(path)
        self.index_path = source / "index.csv" if source.is_dir() else source
        if self.index_path.name.lower() != "index.csv" or not self.index_path.is_file():
            raise DataError(f"Folder-format DVS data requires index.csv: {source}")
        self.root = self.index_path.parent.resolve()
        with self.index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            missing = self.REQUIRED_COLUMNS - columns
            if missing or not ({"relative_path", "path", "file"} & columns):
                raise DataError(f"index.csv missing required columns: {sorted(missing or {'relative_path/path/file'})}")
            all_rows = [dict(row) for row in reader]
        declared_splits = {row.get("split", "").strip().lower() for row in all_rows if row.get("split", "").strip()}
        if split == "trainval":
            rows = [row for row in all_rows if row.get("split", "").strip().lower() != "test"]
        elif split == "test" and declared_splits:
            rows = [row for row in all_rows if row.get("split", "").strip().lower() == "test"]
        else:
            rows = all_rows
        if not rows:
            raise DataError(f"index.csv contains no samples for split={split!r}")

        sample_ids: set[str] = set()
        paths: set[Path] = set()
        normalized: List[Dict[str, Any]] = []
        for row_number, row in enumerate(rows, start=2):
            sample_id = row.get("sample_id", "").strip()
            if not sample_id or sample_id in sample_ids:
                raise DataError(f"Duplicate or empty sample_id at index.csv row {row_number}")
            relative = next((row.get(key, "").strip() for key in ("relative_path", "path", "file") if row.get(key, "").strip()), "")
            sample_path = (self.root / relative).resolve()
            try:
                sample_path.relative_to(self.root)
            except ValueError as exc:
                raise DataError(f"Sample path escapes dataset root at row {row_number}: {relative}") from exc
            if sample_path in paths or not sample_path.is_file():
                raise DataError(f"Duplicate or missing sample file at row {row_number}: {relative}")
            try:
                label = int(row["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DataError(f"Invalid label at index.csv row {row_number}") from exc
            for column, expected in (("time_steps", expected_time_steps), ("channels", expected_channels)):
                if expected is not None and row.get(column, "").strip() and int(row[column]) != int(expected):
                    raise DataError(
                        f"DVS {column} mismatch in index.csv row {row_number}: {row[column]} != {expected}"
                    )
            normalized.append({**row, "sample_id": sample_id, "sample_path": sample_path, "label": label})
            sample_ids.add(sample_id)
            paths.add(sample_path)
        self.rows = normalized
        self.labels = torch.tensor([row["label"] for row in normalized], dtype=torch.long)
        self.expected_time_steps = expected_time_steps
        self.expected_channels = expected_channels
        self.verify_hashes = bool(verify_hashes)
        self._verified_paths: set[Path] = set()
        self.source_fingerprint = sha256_file(self.index_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = row["sample_path"]
        expected_hash = row.get("sha256", "").strip().lower()
        if self.verify_hashes and expected_hash and path not in self._verified_paths:
            if sha256_file(path).lower() != expected_hash:
                raise DataError(f"Sample SHA-256 mismatch: {path}")
            self._verified_paths.add(path)
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(path, map_location="cpu")
        if isinstance(value, Mapping):
            frame_value = next((value[key] for key in ("frames", "data", "x", "inputs") if key in value), None)
            if frame_value is None:
                raise DataError(f"Sample mapping has no frames/data/x/inputs: {path}")
            if "label" in value and int(value["label"]) != row["label"]:
                raise DataError(f"Sample label differs from index.csv: {path}")
            value = frame_value
        frames = _normalize_frame_sample(value, self.expected_time_steps, self.expected_channels)
        return frames, int(row["label"])


def _extract_arrays(value: Any) -> Tuple[Any, Any]:
    if isinstance(value, Mapping):
        frames = next((value[k] for k in ("frames", "data", "x", "inputs") if k in value), None)
        labels = next((value[k] for k in ("labels", "targets", "y") if k in value), None)
        if frames is None:
            raise DataError("DVS mapping must contain frames/data/x/inputs")
        if labels is None:
            raise DataError("DVS mapping must contain labels/targets/y")
        return frames, labels
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return value[0], value[1]
    raise DataError("DVS tensor file must contain a (frames, labels) pair or mapping")


def load_dvs_arrays(frames_path: str | Path, labels_path: str | Path | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load NPZ/PT/Tensor files and return normalized frame and label tensors."""

    frames_path = Path(frames_path)
    suffix = frames_path.suffix.lower()
    if suffix == ".npz":
        with np.load(frames_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            frame_key = next((k for k in ("frames", "data", "x", "inputs") if k in keys), None)
            label_key = next((k for k in ("labels", "targets", "y") if k in keys), None)
            if frame_key is None or (label_key is None and labels_path is None):
                raise DataError(f"NPZ {frames_path} must contain frames/data/x and labels/targets/y")
            frames = archive[frame_key]
            labels = archive[label_key] if label_key is not None else np.load(labels_path, allow_pickle=False)
    elif suffix in {".pt", ".pth", ".tensor"}:
        try:
            value = torch.load(frames_path, map_location="cpu", weights_only=False)
        except TypeError:
            value = torch.load(frames_path, map_location="cpu")
        frames, labels = _extract_arrays(value) if labels_path is None else (value, torch.load(labels_path, map_location="cpu"))
    else:
        raise DataError(f"Unsupported DVS file extension: {frames_path.suffix}")
    frames_tensor = torch.as_tensor(frames)
    labels_tensor = torch.as_tensor(labels)
    # Convert [N,T,H,W] to [N,T,1,H,W].  For [N,T,C,H,W], retain channels.
    if frames_tensor.ndim == 4:
        frames_tensor = frames_tensor.unsqueeze(2)
    elif frames_tensor.ndim == 5:
        pass
    else:
        raise DataError(f"DVS frames must be [N,T,H,W] or [N,T,C,H,W], got {tuple(frames_tensor.shape)}")
    return frames_tensor.float(), labels_tensor.long()


def _find_dvs_file(config: Any) -> Path:
    requested = getattr(config, "frames_path", None)
    if requested:
        path = Path(requested)
        if path.is_file():
            return path
        if path.is_dir():
            if (path / "index.csv").is_file():
                return path
            candidates = sorted([*path.glob("*.npz"), *path.glob("*.pt"), *path.glob("*.pth"), *path.glob("*.tensor")])
            if candidates:
                return candidates[0]
        raise DataError(f"DVS frames_path does not exist or contains no supported file: {path}")
    root = Path(getattr(config, "root", "data"))
    if (root / "index.csv").is_file():
        return root
    indices = sorted(root.rglob("index.csv"))
    if indices:
        return indices[0].parent
    candidates = sorted([*root.rglob("*.npz"), *root.rglob("*.pt"), *root.rglob("*.pth"), *root.rglob("*.tensor")])
    if not candidates:
        raise DataError(f"No preprocessed DVS tensor/NPZ file found under {root}")
    return candidates[0]


def _torchvision() -> Any:
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise DataError("torchvision is required for CIFAR loaders") from exc
    return datasets, transforms


def _cifar_transform(config: Any, train: bool, dataset_name: str) -> Any:
    datasets, transforms = _torchvision()
    normalize = bool(getattr(config, "normalize", True))
    augment = bool(getattr(config, "augment", False)) and train
    if dataset_name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    transform_ops: List[Any] = []
    if augment:
        transform_ops.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        if bool(getattr(config, "autoaugment", False)):
            transform_ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
    transform_ops.append(transforms.ToTensor())
    if normalize:
        transform_ops.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform_ops)


def _cifar_dataset(config: Any, train: bool, dataset_name: str) -> Tuple[Dataset, Any]:
    datasets, _ = _torchvision()
    root = Path(getattr(config, "root", "data"))
    cls = datasets.CIFAR100 if dataset_name == "cifar100" else datasets.CIFAR10
    transform = _cifar_transform(config, train, dataset_name)
    dataset = cls(
        root=str(root), train=train, transform=None,
        download=bool(getattr(config, "download", True)),
    )
    return dataset, transform


def _loader_config(config: Any) -> Any:
    """Expose merged data/optimizer/runtime fields to loader helpers."""

    if not hasattr(config, "data"):
        return config
    from types import SimpleNamespace

    data = config.data
    values = dict(vars(data)) if hasattr(data, "__dict__") else {}
    for source_name in ("model", "optimizer", "runtime"):
        source = getattr(config, source_name, None)
        if source is not None and hasattr(source, "__dict__"):
            values.update(vars(source))
    values.setdefault("seed", getattr(config.runtime, "seed", 0))
    values.setdefault("final_test", getattr(config, "final_test", False))
    return SimpleNamespace(**values)


class EpochShuffleSampler(Sampler[int]):
    """Epoch-addressable sampler so resumed runs do not repeat epoch-zero order."""

    def __init__(self, dataset: Dataset, seed: int, worker_generator: torch.Generator) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0
        self.worker_generator = worker_generator

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.worker_generator.manual_seed(self.seed + self.epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        yield from torch.randperm(len(self.dataset), generator=generator).tolist()

    def __len__(self) -> int:
        return len(self.dataset)


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, config: Any) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = EpochShuffleSampler(dataset, seed, generator) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=bool(getattr(config, "pin_memory", True)),
        worker_init_fn=worker_seed_fn,
        generator=generator,
        persistent_workers=False,
    )


def build_cifar_loaders(config: Any, final_test: bool = False, seed: int | None = None) -> Dict[str, DataLoader]:
    config = _loader_config(config)
    dataset_name = str(getattr(config, "dataset", "cifar10")).lower().replace("-", "")
    if dataset_name not in {"cifar10", "cifar100"}:
        raise DataError(f"Unsupported static dataset: {dataset_name}")
    base, train_transform = _cifar_dataset(config, True, dataset_name)
    # Build the validation transform without constructing the official test
    # dataset; final_test is the only path allowed to touch that split.
    eval_transform = _cifar_transform(config, False, dataset_name)
    manifest_path = _resolve_manifest_path(config, dataset_name)
    source_fingerprint = f"{dataset_name}:{len(base)}"
    targets = getattr(base, "targets", None)
    if targets is None:
        raise DataError(f"{dataset_name} does not expose targets required for stratification")
    manifest = create_split_manifest(
        len(base), float(getattr(config, "val_fraction", 0.1)), int(getattr(config, "split_seed", 2024)),
        manifest_path, dataset_name, source_fingerprint, labels=targets,
    )
    train_set = TransformDataset(base, manifest["train_indices"], train_transform)
    val_set = TransformDataset(base, manifest["val_indices"], eval_transform)
    run_seed = int(getattr(config, "seed", 0) if seed is None else seed)
    batch_size = int(getattr(config, "batch_size", 256))
    loaders: Dict[str, DataLoader] = {
        "train": _loader(train_set, batch_size, True, run_seed, config),
        "val": _loader(val_set, batch_size, False, run_seed + 1, config),
    }
    if final_test:
        test_base, test_transform = _cifar_dataset(config, False, dataset_name)
        test_set = TransformDataset(test_base, range(len(test_base)), test_transform)
        loaders["test"] = _loader(test_set, batch_size, False, run_seed + 2, config)
    loaders["_manifest"] = manifest  # type: ignore[assignment]
    return loaders


def build_dvs_loaders(config: Any, final_test: bool = False, seed: int | None = None) -> Dict[str, DataLoader]:
    config = _loader_config(config)
    frames_path = _find_dvs_file(config)
    labels_path = getattr(config, "labels_path", None)
    expected_time_steps = int(getattr(config, "time_steps", getattr(config, "dvs_time_bins", 0)))
    configured_bins = int(getattr(config, "dvs_time_bins", expected_time_steps))
    if expected_time_steps < 1 or configured_bins != expected_time_steps:
        raise DataError(
            f"DVS configuration mismatch: model time_steps={expected_time_steps}, data dvs_time_bins={configured_bins}"
        )
    expected_channels = int(getattr(config, "in_channels", 0))
    if expected_channels < 1:
        raise DataError("DVS in_channels must be positive")
    if frames_path.is_dir() or frames_path.name.lower() == "index.csv":
        dataset: Dataset = IndexedFrameDataset(
            frames_path, split="trainval", expected_time_steps=expected_time_steps,
            expected_channels=expected_channels,
        )
        labels = dataset.labels  # type: ignore[attr-defined]
        source_fingerprint = dataset.source_fingerprint  # type: ignore[attr-defined]
    else:
        frames, labels = load_dvs_arrays(frames_path, labels_path)
        dataset = TensorFrameDataset(
            frames, labels, expected_time_steps=expected_time_steps,
            expected_channels=expected_channels,
        )
        source_fingerprint = sha256_file(frames_path)
    dataset_name = "cifar10dvs"
    manifest_path = _resolve_manifest_path(config, dataset_name)
    manifest = create_split_manifest(
        len(dataset), float(getattr(config, "val_fraction", 0.1)), int(getattr(config, "split_seed", 2024)),
        manifest_path, dataset_name, source_fingerprint, labels=labels,
    )
    run_seed = int(getattr(config, "seed", 0) if seed is None else seed)
    batch_size = int(getattr(config, "batch_size", 256))
    loaders: Dict[str, DataLoader] = {
        "train": _loader(Subset(dataset, manifest["train_indices"]), batch_size, True, run_seed, config),
        "val": _loader(Subset(dataset, manifest["val_indices"]), batch_size, False, run_seed + 1, config),
    }
    # A separate test tensor can be supplied; never treat the held-out val set as test.
    test_path = getattr(config, "test_frames_path", None)
    if final_test and test_path:
        test_source = Path(test_path)
        if test_source.is_dir() or test_source.name.lower() == "index.csv":
            test_dataset = IndexedFrameDataset(
                test_source, split="test", expected_time_steps=expected_time_steps,
                expected_channels=expected_channels,
            )
        else:
            test_frames, test_labels = load_dvs_arrays(test_source, getattr(config, "test_labels_path", None))
            test_dataset = TensorFrameDataset(
                test_frames, test_labels, expected_time_steps=expected_time_steps,
                expected_channels=expected_channels,
            )
        loaders["test"] = _loader(test_dataset, batch_size, False, run_seed + 2, config)
    loaders["_manifest"] = manifest  # type: ignore[assignment]
    return loaders


def build_loaders(config: Any, final_test: bool = False, seed: int | None = None) -> Dict[str, DataLoader]:
    config = _loader_config(config)
    dataset = str(getattr(config, "dataset", "cifar10")).lower().replace("-", "")
    if dataset in {"cifar10", "cifar100"}:
        return build_cifar_loaders(config, final_test=final_test, seed=seed)
    if dataset == "cifar10dvs":
        return build_dvs_loaders(config, final_test=final_test, seed=seed)
    raise DataError(f"Unknown dataset {dataset!r}")


def build_test_loader(config: Any, seed: int | None = None) -> DataLoader:
    """Construct only the independent test loader after checkpoint freeze."""

    config = _loader_config(config)
    dataset = str(getattr(config, "dataset", "cifar10")).lower().replace("-", "")
    run_seed = int(getattr(config, "seed", 0) if seed is None else seed)
    batch_size = int(getattr(config, "batch_size", 256))
    if dataset in {"cifar10", "cifar100"}:
        test_base, transform = _cifar_dataset(config, False, dataset)
        test_set = TransformDataset(test_base, range(len(test_base)), transform)
        return _loader(test_set, batch_size, False, run_seed + 2, config)
    if dataset == "cifar10dvs":
        test_path = getattr(config, "test_frames_path", None)
        if not test_path:
            raise DataError("CIFAR10-DVS final test requires test_frames_path")
        source = Path(test_path)
        expected_time_steps = int(getattr(config, "time_steps", 0))
        expected_channels = int(getattr(config, "in_channels", 0))
        if source.is_dir() or source.name.lower() == "index.csv":
            test_set = IndexedFrameDataset(
                source,
                split="test",
                expected_time_steps=expected_time_steps,
                expected_channels=expected_channels,
            )
        else:
            frames, labels = load_dvs_arrays(source, getattr(config, "test_labels_path", None))
            test_set = TensorFrameDataset(
                frames,
                labels,
                expected_time_steps=expected_time_steps,
                expected_channels=expected_channels,
            )
        return _loader(test_set, batch_size, False, run_seed + 2, config)
    raise DataError(f"Unknown dataset {dataset!r}")
