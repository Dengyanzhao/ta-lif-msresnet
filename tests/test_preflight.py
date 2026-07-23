from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from talif_msresnet.data import label_fingerprint, stratified_split_indices  # noqa: E402
from talif_msresnet.preflight import (  # noqa: E402
    FreezeGateError,
    _check_dvs_provenance,
    _stable_hash,
    check_protocol,
)
import talif_msresnet.preflight as preflight_module  # noqa: E402
from talif_msresnet.utils import sha256_file  # noqa: E402


def test_repository_protocol_is_author_frozen_and_blocked_until_phase_b(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    def reject_manifest(**_kwargs):
        raise FreezeGateError("fixture: Phase B absent")

    monkeypatch.setattr(preflight_module, "verify_formal_freeze", reject_manifest)
    report = check_protocol(
        root / "configs" / "protocol.yaml",
        mode="full",
        project_root=root,
        check_dependencies=False,
    )
    assert not report.ok
    assert not any(item.startswith("protocol_status.") for item in report.errors)
    assert not any("interaction_practical_threshold_pp" in item for item in report.errors)
    assert not any("unsigned and unfrozen" in item for item in report.warnings)
    assert any(
        "Phase B freeze manifest verification failed: fixture: Phase B absent" in item
        for item in report.errors
    )


def test_smoke_preflight_allows_unfrozen_protocol(
    unfrozen_protocol_path: Path,
) -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    report = check_protocol(
        unfrozen_protocol_path,
        mode="smoke",
        project_root=root,
        check_dependencies=False,
    )
    assert report.ok
    assert any("unsigned and unfrozen" in item for item in report.warnings)
    assert not any(item.startswith("protocol_status.") for item in report.errors)


def _counts(labels, indices):
    return {str(key): value for key, value in sorted(Counter(labels[i] for i in indices).items())}


def test_dvs_preflight_binds_index_and_split_manifest_hashes(tmp_path: Path) -> None:
    root = tmp_path / "processed"
    samples = root / "samples"
    samples.mkdir(parents=True)
    labels = [index % 2 for index in range(20)]
    trainval, test = stratified_split_indices(labels, 0.2, 7)
    test_set = set(test)
    fields = (
        "sample_id", "relative_path", "label", "split", "source_index", "sha256",
        "time_steps", "channels", "height", "width", "resize_mode", "split_seed",
        "test_fraction", "normalization",
    )
    rows = []
    for index, label in enumerate(labels):
        path = samples / f"sample_{index:03d}.pt"
        torch.save(torch.zeros(4, 2, 6, 6), path)
        rows.append(
            {
                "sample_id": f"sample_{index:03d}",
                "relative_path": path.relative_to(root).as_posix(),
                "label": label,
                "split": "test" if index in test_set else "trainval",
                "source_index": index,
                "sha256": sha256_file(path),
                "time_steps": 4,
                "channels": 2,
                "height": 6,
                "width": 6,
                "resize_mode": "nearest",
                "split_seed": 7,
                "test_fraction": 0.2,
                "normalization": "none_raw_event_counts",
            }
        )
    index_path = root / "index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    split = {
        "format_version": 1,
        "dataset": "cifar10dvs",
        "source_length": len(labels),
        "split_strategy": "stratified_by_label",
        "test_fraction": 0.2,
        "split_seed": 7,
        "label_fingerprint_sha256": label_fingerprint(labels),
        "trainval_indices": trainval,
        "test_indices": test,
        "class_counts": {
            "all": _counts(labels, range(len(labels))),
            "trainval": _counts(labels, trainval),
            "test": _counts(labels, test),
        },
    }
    canonical = json.dumps(split, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    split["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    split_path = root / "split_manifest.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    conversion = {
        "format_version": 1,
        "sample_count": len(labels),
        "index_sha256": sha256_file(index_path),
        "split_manifest_sha256": sha256_file(split_path),
        "conversion": {"time_bins": 4, "height": 6, "width": 6, "resize_mode": "nearest"},
        "partition": {"test_fraction": 0.2, "split_seed": 7},
        "normalization": "none_raw_event_counts",
    }
    (root / "conversion_manifest.json").write_text(
        json.dumps(conversion, indent=2), encoding="utf-8"
    )
    receipt = {
        "schema": "cifar10dvs-full-verification-v1",
        "status": "pass",
        "verified_at": "2026-07-22T00:00:00+00:00",
        "verifier": {"name": "scripts/verify_cifar10dvs.py", "version": 1},
        "root": "external:path-sha256:" + "0" * 64,
        "sample_count": len(labels),
        "trainval_count": len(trainval),
        "test_count": len(test),
        "expected_shape": [4, 2, 6, 6],
        "serialized_sample_bytes": sum(path.stat().st_size for path in samples.glob("*.pt")),
        "conversion_manifest_sha256": sha256_file(root / "conversion_manifest.json"),
        "index_sha256": sha256_file(index_path),
        "split_manifest_sha256": sha256_file(split_path),
    }
    receipt["receipt_sha256"] = _stable_hash(receipt)
    (root / "verification_receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    common = {
        "val_fraction": 0.25,
        "split_seed": 9,
        "split_manifest": "manifests/{dataset}_seed{split_seed}.json",
    }
    dvs = {
        "frames_path": str(root),
        "test_frames_path": str(root),
        "in_channels": 2,
        "num_classes": 2,
        "dvs_time_bins": 4,
        "dvs_height": 6,
        "dvs_width": 6,
        "dvs_resize_mode": "nearest",
        "dvs_test_fraction": 0.2,
        "dvs_split_seed": 7,
        "val_fraction": 0.25,
        "normalize": False,
    }
    assert _check_dvs_provenance(tmp_path, root, root, common, dvs) == []

    (root / "verification_receipt.json").unlink()
    errors = _check_dvs_provenance(tmp_path, root, root, common, dvs)
    assert any("full verification receipt is missing" in error for error in errors)

    with index_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    errors = _check_dvs_provenance(tmp_path, root, root, common, dvs)
    assert any("index.csv hash differs" in error for error in errors)
