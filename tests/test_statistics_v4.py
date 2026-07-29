from __future__ import annotations

import pandas as pd
import pytest

from talif_msresnet.statistics_v4 import (
    V4_BOOTSTRAP_SEEDS,
    V4_FORMAL_SEEDS_BY_DATASET,
    V4PairedAnalysisError,
    analyze_v4_paired_accuracy,
    complete_pair_bootstrap,
    validate_v4_paired_blocks,
)


PROTOCOL_HASH = "v4-protocol-hash"


def v4_metrics(
    *,
    cifar100_differences: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    dvs_differences: tuple[float, ...] = (-1.0, 0.0, 1.0, 2.0, 3.0),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specifications = (
        ("cifar100", 20, 6, 0.70, cifar100_differences),
        ("cifar10dvs", 20, 10, 0.65, dvs_differences),
    )
    for dataset, depth, time_steps, base_accuracy, differences in specifications:
        seeds = V4_FORMAL_SEEDS_BY_DATASET[dataset]
        for seed, difference in zip(seeds, differences, strict=True):
            shared_hash = f"shared-{dataset}-{seed}"
            for condition in ("C1", "C2"):
                accuracy = base_accuracy + (difference / 100.0 if condition == "C2" else 0.0)
                rows.append(
                    {
                        "run_id": (
                            f"E1_{dataset}_d{depth}_t{time_steps}_{condition}_s{seed}"
                        ),
                        "experiment": "E1",
                        "dataset": dataset,
                        "depth": depth,
                        "time_steps": time_steps,
                        "condition": condition,
                        "neuron": "lif" if condition == "C1" else "ta_lif",
                        "topology": "spiking_resnet",
                        "seed": seed,
                        "status": "complete",
                        "failed": 0,
                        "test_accuracy": accuracy,
                        "protocol_hash": PROTOCOL_HASH,
                        "config_hash": f"config-{dataset}-{condition}-{seed}",
                        "split_manifest_sha256": f"split-{dataset}",
                        "shared_weight_sha256": shared_hash,
                        "test_checkpoint_sha256": (
                            f"checkpoint-{dataset}-{condition}-{seed}"
                        ),
                        "test_evaluated_at": "2026-08-01T00:00:00+00:00",
                    }
                )
    return pd.DataFrame(rows)


def test_v4_analysis_preserves_dataset_specific_seed_order() -> None:
    result = analyze_v4_paired_accuracy(
        v4_metrics(),
        expected_protocol_hash=PROTOCOL_HASH,
    )

    primary = result.datasets["cifar100"]
    replication = result.datasets["cifar10dvs"]
    assert tuple(item.seed for item in primary.raw_differences) == (
        V4_FORMAL_SEEDS_BY_DATASET["cifar100"]
    )
    assert tuple(item.seed for item in replication.raw_differences) == (
        V4_FORMAL_SEEDS_BY_DATASET["cifar10dvs"]
    )
    assert [item.difference_pp for item in primary.raw_differences] == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )
    assert result.claim_gates.primary_improvement_allowed


def test_v4_bootstrap_is_deterministic_and_dataset_bound() -> None:
    differences = [1.0, 2.0, 3.0, 4.0, 5.0]
    first = complete_pair_bootstrap(differences, dataset="cifar100")
    second = complete_pair_bootstrap(differences, dataset="cifar100")
    dvs = complete_pair_bootstrap(differences, dataset="cifar10dvs")

    assert first == second
    assert first.random_seed == V4_BOOTSTRAP_SEEDS["cifar100"]
    assert dvs.random_seed == V4_BOOTSTRAP_SEEDS["cifar10dvs"]
    assert first.n_resamples == 10_000


def test_v4_validation_rejects_cross_dataset_seed_substitution() -> None:
    frame = v4_metrics()
    cifar_seed = V4_FORMAL_SEEDS_BY_DATASET["cifar100"][0]
    dvs_seed = V4_FORMAL_SEEDS_BY_DATASET["cifar10dvs"][0]
    frame.loc[
        (frame["dataset"] == "cifar100") & (frame["seed"] == cifar_seed),
        "seed",
    ] = dvs_seed

    with pytest.raises(V4PairedAnalysisError, match="formal seeds differ"):
        validate_v4_paired_blocks(frame, expected_protocol_hash=PROTOCOL_HASH)


def test_v4_validation_rejects_partial_pair_and_c3() -> None:
    frame = v4_metrics().iloc[:-1].copy()
    with pytest.raises(V4PairedAnalysisError, match="exactly 20 rows"):
        validate_v4_paired_blocks(frame, expected_protocol_hash=PROTOCOL_HASH)

    frame = v4_metrics()
    frame.loc[0, "condition"] = "C3"
    with pytest.raises(V4PairedAnalysisError, match="conditions must be exactly C1/C2"):
        validate_v4_paired_blocks(frame, expected_protocol_hash=PROTOCOL_HASH)


def test_v4_replication_cannot_rescue_failed_primary() -> None:
    result = analyze_v4_paired_accuracy(
        v4_metrics(
            cifar100_differences=(-5.0, -4.0, -3.0, -2.0, -1.0),
            dvs_differences=(1.0, 2.0, 3.0, 4.0, 5.0),
        ),
        expected_protocol_hash=PROTOCOL_HASH,
    )

    assert not result.claim_gates.primary_improvement_allowed
    assert result.claim_gates.dvs_replication_improvement_allowed
    assert not result.claim_gates.cross_domain_improvement_allowed
    assert result.claim_gates.dvs_cannot_rescue_primary
