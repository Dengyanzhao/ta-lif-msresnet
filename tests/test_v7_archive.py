from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import archive_v7_evidence as archive_v7

from talif_msresnet.benchmark_v7 import V7BenchmarkError
from talif_msresnet.utils import sha256_file, stable_hash


def _install_reportable_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], Path, Path]:
    protocol = {"protocol_version": 7}
    artifacts = {
        "freeze_manifest": "FREEZE_MANIFEST_V7_MECHANISM.json",
        "formal_matrix": "configs/v7_mechanism_generated",
        "formal_results": "results/formal_v7_mechanism",
        "benchmark_results": "results/benchmark/v7_mechanism",
        "analysis_results": "results/analysis/v7_mechanism",
    }
    freeze_path = tmp_path / artifacts["freeze_manifest"]
    matrix_path = tmp_path / artifacts["formal_matrix"] / "matrix_manifest.json"
    receipt_path = (
        tmp_path / artifacts["benchmark_results"] / "benchmark_completion.json"
    )
    run_id = "E9_cifar100_d20_t6_M0_s404085484"
    run_dir = tmp_path / artifacts["formal_results"] / run_id
    analysis_root = tmp_path / artifacts["analysis_results"]
    for path in (freeze_path, matrix_path, receipt_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{path.name}\n", encoding="utf-8")
    run_dir.mkdir(parents=True)
    (run_dir / "final_test.json").write_text(
        json.dumps({"status": "complete", "run_id": run_id}) + "\n",
        encoding="utf-8",
    )
    analysis_root.mkdir(parents=True)
    (analysis_root / archive_v7.RAW_ACCURACIES_CSV).write_text(
        "seed,condition,test_accuracy_pp\n", encoding="utf-8"
    )
    (analysis_root / archive_v7.RAW_DIFFERENCES_CSV).write_text(
        "contrast,seed,difference_pp\n", encoding="utf-8"
    )
    audit = [
        {
            "run_id": run_id,
            "checkpoint_sha256": "1" * 64,
            "config_hash": "2" * 64,
            "shared_weight_sha256": "3" * 64,
            "training_environment_sha256": "4" * 64,
            "split_manifest_sha256": "5" * 64,
        }
    ]
    commit = "a" * 40
    receipt_hash = sha256_file(receipt_path)
    analysis = {
        "protocol_version": 7,
        "protocol_hash": stable_hash(protocol),
        "n_final_test_markers": 48,
        "source_git_commit": commit,
        "tracked_clean": True,
        "formal_matrix_manifest_sha256": sha256_file(matrix_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "benchmark_receipt": receipt_path.relative_to(tmp_path).as_posix(),
        "benchmark_receipt_sha256": receipt_hash,
        "benchmark_receipt_schema": "receipt-v1",
        "input_artifact_audit": audit,
    }
    analysis_path = analysis_root / archive_v7.ANALYSIS_JSON
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")
    provenance = {
        "cifar100_source_provenance": "environment/CIFAR100_SOURCE_PROVENANCE.json",
        "cifar100_source_provenance_sha256": "6" * 64,
        "cifar100_archive": "data/cifar100/cifar-100-python.tar.gz",
        "cifar100_archive_sha256": "7" * 64,
        "cifar100_train_pickle": "data/cifar100/cifar-100-python/train",
        "cifar100_train_pickle_sha256": "8" * 64,
        "cifar100_test_pickle": "data/cifar100/cifar-100-python/test",
        "cifar100_test_pickle_sha256": "9" * 64,
        "cifar100_meta_pickle": "data/cifar100/cifar-100-python/meta",
        "cifar100_meta_pickle_sha256": "a" * 64,
        "cifar100_split_manifest": "data/manifests/cifar100_seed2024.json",
        "cifar100_split_manifest_sha256": "b" * 64,
    }
    analysis["cifar100_final_test_source_binding"] = {
        "test_source": provenance["cifar100_test_pickle"],
        "test_source_sha256": provenance["cifar100_test_pickle_sha256"],
        **provenance,
    }
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")

    monkeypatch.setattr(archive_v7, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        archive_v7, "artifact_paths_for_protocol", lambda _protocol: artifacts
    )
    monkeypatch.setattr(
        archive_v7,
        "validate_v7_cifar100_provenance_files",
        lambda *_args, **_kwargs: provenance,
    )
    monkeypatch.setattr(
        archive_v7,
        "canonical_bound_paths",
        lambda *_args, **_kwargs: {
            "protocol": tmp_path / "configs/protocol_v7_mechanism.yaml",
            "matrix_manifest": matrix_path,
            "freeze_manifest": freeze_path,
        },
    )
    monkeypatch.setattr(archive_v7, "verify_formal_freeze", lambda **_kwargs: {})
    monkeypatch.setattr(
        archive_v7,
        "repository_git_identity",
        lambda _root: (commit, True),
    )
    monkeypatch.setattr(
        archive_v7, "expected_receipt_path", lambda *_args: receipt_path
    )
    monkeypatch.setattr(archive_v7, "expected_run_ids", lambda: (run_id,))
    monkeypatch.setattr(
        archive_v7,
        "collect_final_test_rows",
        lambda *_args, **_kwargs: (None, audit),
    )
    return protocol, receipt_path, analysis_path


def test_v7_archive_rejects_benchmark_receipt_mutation_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, receipt_path, _analysis_path = _install_reportable_fixture(
        tmp_path, monkeypatch
    )

    def mutate_receipt(**_kwargs: Any) -> dict[str, str]:
        receipt_path.write_text("mutated\n", encoding="utf-8")
        return {
            "schema": "receipt-v1",
            "training_environment_sha256": "4" * 64,
        }

    monkeypatch.setattr(archive_v7, "validate_receipt", mutate_receipt)

    with pytest.raises(
        archive_v7.V7EvidenceArchiveError, match="changed while archive inputs"
    ):
        archive_v7._validate_reportable_state(
            tmp_path / "configs/protocol_v7_mechanism.yaml", protocol
        )


def test_v7_archive_rejects_analysis_benchmark_receipt_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, _receipt_path, analysis_path = _install_reportable_fixture(
        tmp_path, monkeypatch
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["benchmark_receipt_sha256"] = "0" * 64
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        archive_v7,
        "validate_receipt",
        lambda **_kwargs: {
            "schema": "receipt-v1",
            "training_environment_sha256": "4" * 64,
        },
    )

    with pytest.raises(
        archive_v7.V7EvidenceArchiveError, match="analysis is not bound"
    ):
        archive_v7._validate_reportable_state(
            tmp_path / "configs/protocol_v7_mechanism.yaml", protocol
        )


def test_v7_archive_rejects_analysis_final_test_source_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, _receipt_path, analysis_path = _install_reportable_fixture(
        tmp_path, monkeypatch
    )
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["cifar100_final_test_source_binding"][
        "cifar100_meta_pickle_sha256"
    ] = "0" * 64
    analysis_path.write_text(json.dumps(analysis) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        archive_v7,
        "validate_receipt",
        lambda **_kwargs: {
            "schema": "receipt-v1",
            "training_environment_sha256": "4" * 64,
        },
    )

    with pytest.raises(
        archive_v7.V7EvidenceArchiveError, match="analysis is not bound"
    ):
        archive_v7._validate_reportable_state(
            tmp_path / "configs/protocol_v7_mechanism.yaml", protocol
        )


def test_v7_archive_wraps_benchmark_receipt_validation_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, _receipt_path, _analysis_path = _install_reportable_fixture(
        tmp_path, monkeypatch
    )

    def reject_receipt(**_kwargs: Any) -> dict[str, Any]:
        raise V7BenchmarkError("formal checkpoint hash mismatch")

    monkeypatch.setattr(archive_v7, "validate_receipt", reject_receipt)

    with pytest.raises(
        archive_v7.V7EvidenceArchiveError, match="checkpoint hash mismatch"
    ):
        archive_v7._validate_reportable_state(
            tmp_path / "configs/protocol_v7_mechanism.yaml", protocol
        )
