from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from talif_msresnet.data import (  # noqa: E402
    DataError,
    IndexedFrameDataset,
    TensorFrameDataset,
    build_dvs_loaders,
    create_split_manifest,
    EpochShuffleSampler,
    load_representative_batch_artifact,
    validate_representative_batch_artifact,
)
from talif_msresnet.diagnostics import representative_batch_sha256  # noqa: E402
from talif_msresnet.utils import sha256_file  # noqa: E402


def test_stratified_manifest_records_and_validates_labels(tmp_path: Path) -> None:
    labels = [0] * 5 + [1] * 5 + [2] * 5
    path = tmp_path / "split.json"
    manifest = create_split_manifest(
        len(labels), 0.2, 17, path, "tiny", source_fingerprint="tiny:v1", labels=labels,
    )
    assert manifest["split_strategy"] == "stratified_by_label"
    assert manifest["class_counts"]["all"] == {"0": 5, "1": 5, "2": 5}
    assert manifest["class_counts"]["val"] == {"0": 1, "1": 1, "2": 1}
    assert set(manifest["train_indices"]).isdisjoint(manifest["val_indices"])
    assert create_split_manifest(
        len(labels), 0.2, 17, path, "tiny", source_fingerprint="tiny:v1", labels=labels,
    ) == manifest

    reordered = labels.copy()
    reordered[0], reordered[5] = reordered[5], reordered[0]
    with pytest.raises(DataError, match="label fingerprint"):
        create_split_manifest(
            len(reordered), 0.2, 17, path, "tiny", source_fingerprint="tiny:v1", labels=reordered,
        )


def _write_folder_dataset(root: Path) -> None:
    samples = root / "samples"
    samples.mkdir(parents=True)
    fields = (
        "sample_id", "relative_path", "label", "split", "source_index", "sha256",
        "time_steps", "channels", "height", "width",
    )
    rows = []
    for index in range(8):
        sample_id = f"tiny_{index:03d}"
        relative = Path("samples") / f"{sample_id}.pt"
        tensor = torch.full((4, 2, 6, 6), float(index))
        torch.save(tensor, root / relative)
        rows.append({
            "sample_id": sample_id,
            "relative_path": relative.as_posix(),
            "label": index % 2,
            "split": "trainval" if index < 6 else "test",
            "source_index": index,
            "sha256": sha256_file(root / relative),
            "time_steps": 4,
            "channels": 2,
            "height": 6,
            "width": 6,
        })
    with (root / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_lazy_folder_dataset_and_dvs_loader_are_download_free(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    _write_folder_dataset(root)
    dataset = IndexedFrameDataset(root, split="trainval", expected_time_steps=4, expected_channels=2)
    assert len(dataset) == 6
    frames, label = dataset[1]
    assert frames.shape == (4, 2, 6, 6)
    assert label == 1

    config = SimpleNamespace(
        dataset="cifar10dvs",
        root=str(root),
        frames_path=str(root),
        labels_path=None,
        test_frames_path=str(root),
        test_labels_path=None,
        split_manifest=str(tmp_path / "loader_split.json"),
        split_seed=23,
        val_fraction=1 / 3,
        dvs_time_bins=4,
        time_steps=4,
        in_channels=2,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        seed=7,
    )
    loaders = build_dvs_loaders(config, final_test=True, seed=7)
    assert len(loaders["train"].dataset) == 4
    assert len(loaders["val"].dataset) == 2
    assert len(loaders["test"].dataset) == 2
    assert loaders["_manifest"]["split_strategy"] == "stratified_by_label"


def test_dvs_shape_mismatches_fail_before_training(tmp_path: Path) -> None:
    frames = torch.zeros(4, 3, 2, 5, 5)
    labels = torch.tensor([0, 0, 1, 1])
    with pytest.raises(DataError, match="time-bin mismatch"):
        TensorFrameDataset(frames, labels, expected_time_steps=4, expected_channels=2)
    with pytest.raises(DataError, match="channel mismatch"):
        TensorFrameDataset(frames, labels, expected_time_steps=3, expected_channels=1)

    root = tmp_path / "frames"
    _write_folder_dataset(root)
    with pytest.raises(DataError, match="time_steps mismatch"):
        IndexedFrameDataset(root, split="trainval", expected_time_steps=5, expected_channels=2)


def test_lazy_dataset_verifies_sample_hash_on_access(tmp_path: Path) -> None:
    root = tmp_path / "frames"
    _write_folder_dataset(root)
    dataset = IndexedFrameDataset(root, split="trainval", expected_time_steps=4, expected_channels=2)
    sample_path = root / "samples" / "tiny_000.pt"
    torch.save(torch.ones(4, 2, 6, 6), sample_path)
    with pytest.raises(DataError, match="SHA-256 mismatch"):
        dataset[0]


def test_epoch_sampler_is_resume_addressable() -> None:
    dataset = list(range(12))
    first_generator = torch.Generator()
    first = EpochShuffleSampler(dataset, 41, first_generator)
    first.set_epoch(3)
    expected = list(first)

    resumed_generator = torch.Generator()
    resumed = EpochShuffleSampler(dataset, 41, resumed_generator)
    resumed.set_epoch(3)
    assert list(resumed) == expected
    resumed.set_epoch(4)
    assert list(resumed) != expected


def test_representative_batch_provenance_and_shape_are_enforced(tmp_path: Path) -> None:
    inputs = torch.randn(6, 3, 32, 32)
    targets = torch.tensor([0, 1, 2, 3, 4, 5])
    metadata = {
        "dataset": "cifar10",
        "source_split": "validation",
        "test_data_accessed": False,
        "protocol_hash": "protocol-v1",
        "split_manifest_sha256": "split-v1",
        "in_channels": 3,
        "time_steps": 6,
        "num_classes": 10,
        "input_shape": list(inputs.shape),
        "representative_batch_sha256": representative_batch_sha256(inputs, targets),
    }
    path = tmp_path / "representative.pt"
    torch.save({"inputs": inputs, "targets": targets, "metadata": metadata}, path)
    loaded_inputs, loaded_targets, loaded_metadata = load_representative_batch_artifact(path)
    assert validate_representative_batch_artifact(
        loaded_inputs,
        loaded_targets,
        loaded_metadata,
        expected_dataset="cifar10",
        expected_protocol_hash="protocol-v1",
        expected_split_manifest_sha256="split-v1",
        expected_in_channels=3,
        expected_time_steps=6,
        expected_num_classes=10,
    ) == metadata["representative_batch_sha256"]

    with pytest.raises(DataError, match="dataset mismatch"):
        mismatched_metadata = {**loaded_metadata, "num_classes": 100}
        validate_representative_batch_artifact(
            loaded_inputs,
            loaded_targets,
            mismatched_metadata,
            expected_dataset="cifar100",
            expected_protocol_hash="protocol-v1",
            expected_split_manifest_sha256="split-v1",
            expected_in_channels=3,
            expected_time_steps=6,
            expected_num_classes=100,
        )
