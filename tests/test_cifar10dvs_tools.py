from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_cifar10dvs as prepare_tool  # noqa: E402
import verify_cifar10dvs as verify_tool  # noqa: E402
from talif_msresnet.utils import sha256_file  # noqa: E402


class _TinyCIFAR10DVS:
    sensor_size = (6, 6, 2)
    targets = [0, 0, 1, 1]

    def __init__(self, save_to: str) -> None:
        self.save_to = save_to

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        frames = torch.full((4, 2, 6, 6), float(index), dtype=torch.float32)
        return frames, self.targets[index]


def _fake_tonic(*, fail_index: int | None = None) -> SimpleNamespace:
    class ToFrame:
        def __init__(self, sensor_size, n_time_bins: int) -> None:
            assert sensor_size == (6, 6, 2)
            assert n_time_bins == 4

        def __call__(self, events):
            if fail_index is not None and int(events.flatten()[0].item()) == fail_index:
                raise RuntimeError("synthetic conversion failure")
            return events

    return SimpleNamespace(
        __version__="test-tonic",
        datasets=SimpleNamespace(CIFAR10DVS=_TinyCIFAR10DVS),
        transforms=SimpleNamespace(ToFrame=ToFrame),
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_root=str(tmp_path / "raw"),
        output=str(tmp_path / "processed"),
        time_bins=4,
        height=6,
        width=6,
        test_fraction=0.5,
        split_seed=7,
    )


def _make_extracted_tree(root: Path, *, samples_per_class: int = 2) -> Path:
    root.mkdir(parents=True)
    (root / "README.txt").write_text("converted CIFAR10-DVS fixture\n", encoding="utf-8")
    for class_name in prepare_tool.CIFAR10DVS_CLASSES:
        class_dir = root / class_name
        class_dir.mkdir()
        for number in reversed(range(samples_per_class)):
            path = class_dir / f"cifar10_{class_name}_{number}.aedat4"
            path.write_bytes(prepare_tool.AEDAT4_HEADER + f"fixture-{number}".encode())
    return root


def _direct_args(tmp_path: Path, source: Path) -> argparse.Namespace:
    args = _args(tmp_path)
    args.data_root = None
    args.aedat4_root = str(source)
    return args


def _fake_direct_tonic() -> SimpleNamespace:
    event_dtype = np.dtype([
        ("t", np.uint64), ("x", np.uint16), ("y", np.uint16), ("p", bool),
    ])

    def read_aedat4(_path: str):
        events = np.zeros(4, dtype=event_dtype)
        events["t"] = [0, 1, 2, 3]
        events["x"] = [0, 1, 2, 127]
        events["y"] = [127, 2, 1, 0]
        events["p"] = [False, True, False, True]
        return events

    class ToFrame:
        def __init__(self, sensor_size, n_time_bins: int) -> None:
            assert sensor_size == (128, 128, 2)
            assert n_time_bins == 4

        def __call__(self, _events):
            return torch.ones((4, 2, 6, 6), dtype=torch.float32)

    class MustNotInstantiate:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("direct AEDAT4 mode must not instantiate the tonic dataset")

    return SimpleNamespace(
        __version__="test-tonic",
        datasets=SimpleNamespace(CIFAR10DVS=MustNotInstantiate),
        transforms=SimpleNamespace(ToFrame=ToFrame),
        io=SimpleNamespace(read_aedat4=read_aedat4),
    )


def test_prepare_commits_same_directory_staging_and_verifier_loads_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic())
    observed_renames: list[tuple[Path, Path]] = []
    original_replace = prepare_tool.os.replace

    def replace(source, destination) -> None:
        if Path(source).is_dir():
            observed_renames.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(prepare_tool.os, "replace", replace)
    output = prepare_tool.prepare(_args(tmp_path))

    assert output == (tmp_path / "processed").resolve()
    assert output.is_dir()
    assert len(observed_renames) == 1
    assert observed_renames[0][0].parent == observed_renames[0][1].parent
    assert not list(tmp_path.glob(".processed.staging-*"))
    conversion = json.loads(
        (output / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    assert conversion["data_root"].startswith("external:path-sha256:")
    assert str(tmp_path) not in conversion["data_root"]
    summary = verify_tool.verify_dataset(output, progress_every=0)
    assert summary["status"] == "verified"
    assert summary["sample_count"] == 4
    assert summary["expected_shape"] == [4, 2, 6, 6]
    receipt = verify_tool.write_verification_receipt(output, summary)
    assert receipt.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "pass"


def test_failed_full_verification_does_not_replace_previous_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic())
    output = prepare_tool.prepare(_args(tmp_path))
    summary = verify_tool.verify_dataset(output, progress_every=0)
    receipt = verify_tool.write_verification_receipt(output, summary)
    before = receipt.read_bytes()

    sample = next((output / "samples").glob("*.pt"))
    sample.write_bytes(sample.read_bytes() + b"tamper")
    with pytest.raises(verify_tool.VerificationError):
        verify_tool.verify_dataset(output, progress_every=0)
    assert receipt.read_bytes() == before


def test_verifier_rejects_malformed_external_data_root_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic())
    output = prepare_tool.prepare(_args(tmp_path))
    conversion_path = output / "conversion_manifest.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    conversion["data_root"] = "external:path-sha256:not-a-digest"
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")

    with pytest.raises(verify_tool.VerificationError, match="data_root must be"):
        verify_tool.verify_dataset(output, progress_every=0)


def test_verifier_accepts_repository_relative_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic())
    output = prepare_tool.prepare(_args(tmp_path))
    conversion_path = output / "conversion_manifest.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    conversion["data_root"] = "data/cifar10dvs/raw"
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")
    assert verify_tool.verify_dataset(output, progress_every=0)["status"] == "verified"


def test_prepare_cleans_failed_staging_and_is_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic(fail_index=1))
    with pytest.raises(RuntimeError, match="synthetic conversion failure"):
        prepare_tool.prepare(args)
    assert not (tmp_path / "processed").exists()
    assert not list(tmp_path.glob(".processed.staging-*"))

    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic())
    assert prepare_tool.prepare(args).is_dir()


def test_prepare_blocks_before_staging_when_disk_space_is_insufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic())
    monkeypatch.setattr(
        prepare_tool.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=1),
    )
    with pytest.raises(RuntimeError, match="Insufficient disk space"):
        prepare_tool.prepare(_args(tmp_path))
    assert not (tmp_path / "processed").exists()
    assert not list(tmp_path.glob(".processed.staging-*"))


def test_extracted_aedat4_discovery_is_numeric_and_rejects_bad_structure(
    tmp_path: Path,
) -> None:
    source = _make_extracted_tree(tmp_path / "source", samples_per_class=2)
    files = prepare_tool._discover_extracted_aedat4(source, expected_per_class=2)
    assert [(entry.label, entry.sample_number) for entry in files] == [
        (label, number)
        for label in range(len(prepare_tool.CIFAR10DVS_CLASSES))
        for number in range(2)
    ]
    assert [entry.source_index for entry in files] == list(range(20))

    bad = source / "airplane" / "cifar10_airplane_2.aedat4"
    bad.write_bytes(prepare_tool.AEDAT4_HEADER + b"extra")
    with pytest.raises(RuntimeError, match="Invalid extracted AEDAT4 class"):
        prepare_tool._discover_extracted_aedat4(source, expected_per_class=2)


def test_direct_aedat4_prepare_avoids_download_and_binds_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_extracted_tree(tmp_path / "source", samples_per_class=2)
    monkeypatch.setitem(sys.modules, "tonic", _fake_direct_tonic())
    monkeypatch.setattr(prepare_tool, "EXPECTED_SAMPLES_PER_CLASS", 2)
    monkeypatch.setattr(verify_tool, "EXPECTED_SAMPLES_PER_CLASS", 2)

    output = prepare_tool.prepare(_direct_args(tmp_path, source))
    conversion = json.loads(
        (output / "conversion_manifest.json").read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        (output / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert conversion["format_version"] == 2
    assert conversion["source"]["kind"] == prepare_tool.SOURCE_KIND
    assert conversion["source"]["official_archive_byte_identity"] == "not_verified"
    assert source_manifest["file_count"] == 20
    assert source_manifest["official_archive_byte_identity"] == "not_verified"
    assert [record["source_index"] for record in source_manifest["files"]] == list(range(20))
    assert verify_tool.verify_dataset(output, progress_every=0)["sample_count"] == 20


def test_verifier_rejects_tampered_extracted_source_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _make_extracted_tree(tmp_path / "source", samples_per_class=2)
    monkeypatch.setitem(sys.modules, "tonic", _fake_direct_tonic())
    monkeypatch.setattr(prepare_tool, "EXPECTED_SAMPLES_PER_CLASS", 2)
    monkeypatch.setattr(verify_tool, "EXPECTED_SAMPLES_PER_CLASS", 2)
    output = prepare_tool.prepare(_direct_args(tmp_path, source))

    source_manifest_path = output / "source_manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_manifest["official_archive_byte_identity"] = "verified"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    with pytest.raises(verify_tool.VerificationError, match="source manifest SHA-256"):
        verify_tool.verify_dataset(output, progress_every=0)


def test_verifier_rejects_tensor_shape_even_when_hash_manifests_are_rebound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tonic", _fake_tonic())
    output = prepare_tool.prepare(_args(tmp_path))
    index_path = output / "index.csv"
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    sample_path = output / rows[0]["relative_path"]
    torch.save(torch.zeros(3, 2, 6, 6), sample_path)
    rows[0]["sha256"] = sha256_file(sample_path)
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    conversion_path = output / "conversion_manifest.json"
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    conversion["index_sha256"] = sha256_file(index_path)
    conversion["storage_preflight"]["serialized_sample_bytes"] = sum(
        path.stat().st_size for path in (output / "samples").glob("*.pt")
    )
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")

    with pytest.raises(verify_tool.VerificationError, match="tensor shape mismatch"):
        verify_tool.verify_dataset(output, progress_every=0)
