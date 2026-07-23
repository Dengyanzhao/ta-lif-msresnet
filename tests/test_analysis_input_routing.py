from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from analyze_results import _optional_auxiliary_source  # noqa: E402
from talif_msresnet.aggregate import load_seed_metrics  # noqa: E402
from talif_msresnet.pathing import artifact_path_reference  # noqa: E402


def test_single_seed_metrics_file_is_not_reused_as_manifest_or_benchmark(tmp_path) -> None:
    metrics = tmp_path / "seed_metrics.csv"
    metrics.write_text("run_id,accuracy\nrun,1\n", encoding="utf-8")
    assert _optional_auxiliary_source(None, metrics) is None


def test_directory_input_remains_eligible_for_auxiliary_discovery(tmp_path) -> None:
    assert _optional_auxiliary_source(None, tmp_path) == tmp_path


def test_explicit_auxiliary_path_wins_for_file_input(tmp_path) -> None:
    metrics = tmp_path / "seed_metrics.csv"
    benchmark = tmp_path / "benchmark_results.csv"
    metrics.touch()
    benchmark.touch()
    assert _optional_auxiliary_source(benchmark, metrics) == benchmark


def test_aggregate_source_file_is_portable_for_external_input(tmp_path) -> None:
    metrics = tmp_path / "seed_metrics.csv"
    metrics.write_text("run_id,accuracy\nrun,1\n", encoding="utf-8")

    loaded = load_seed_metrics(metrics)

    expected = artifact_path_reference(metrics, ROOT)
    assert loaded.loc[0, "source_file"] == expected
    assert expected.startswith("external:file-sha256:")
    assert str(tmp_path) not in expected
