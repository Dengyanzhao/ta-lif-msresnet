from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from talif_msresnet.aggregate import (
    _overall_efficiency_status,
    _paired_log_ratio_summary,
    _validate_training_environment,
    build_table4,
)
from talif_msresnet.statistics import (
    CONFIRMATORY_SEEDS,
    CONDITION_LABELS,
    IncompleteSeedBlockError,
    PRIMARY_ACCURACY_GROUPS,
)


def _complete_confirmatory_metrics() -> pd.DataFrame:
    rows = []
    did = dict(zip(CONFIRMATORY_SEEDS, (0.70, 0.80, 0.90, 1.00, 1.10)))
    for dataset, depth, time_steps in PRIMARY_ACCURACY_GROUPS:
        for seed in CONFIRMATORY_SEEDS:
            base = 70.0 + seed / 1000.0
            values = {
                "C1": base,
                "C2": base + 0.20,
                "C3": base + 0.30,
                "C4": base + 0.50 + did[seed],
            }
            for condition, accuracy in values.items():
                neuron, topology = CONDITION_LABELS[condition]
                rows.append(
                    {
                        "experiment": "E1",
                        "dataset": dataset,
                        "depth": depth,
                        "time_steps": time_steps,
                        "condition": condition,
                        "seed": seed,
                        "accuracy": accuracy,
                        "status": "complete",
                        "neuron": neuron,
                        "topology": topology,
                        "protocol_hash": "a" * 64,
                        "split_manifest_sha256": f"split-{dataset}",
                        "shared_weight_sha256": (
                            f"shared-{dataset}-{depth}-{time_steps}-{seed}"
                        ),
                        "test_checkpoint_sha256": (
                            f"checkpoint-{dataset}-{depth}-{time_steps}-{condition}-{seed}"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_confirmatory_table_executes_five_seed_did_holm_and_bootstrap() -> None:
    table = build_table4(_complete_confirmatory_metrics())
    assert len(table) == 2
    assert table["interaction_estimate"].tolist() == pytest.approx([0.9] * 2)
    assert table["interaction_reject_holm_0_05"].all()
    assert table["synergy_supported"].all()
    assert table["bootstrap_resamples"].eq(10_000).all()
    assert table["bootstrap_seed"].eq(20_260_719).all()


def _paired_frame(ratios: list[float]) -> pd.DataFrame:
    rows = []
    for seed, ratio in zip(CONFIRMATORY_SEEDS, ratios):
        rows.append({"condition": "C1", "seed": seed, "time_steps": 6, "latency": 1.0})
        rows.append({"condition": "C2", "seed": seed, "time_steps": 6, "latency": ratio})
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("ratios", "expected"),
    [
        ([1.05] * 5, "supported_within_tolerance"),
        ([1.12] * 5, "exceeds_tolerance"),
        ([1.01, 1.03, 1.08, 1.12, 1.16], "within_tolerance_uncertain"),
    ],
)
def test_paired_log_ratio_reaches_all_three_tolerance_states(
    ratios: list[float], expected: str
) -> None:
    result = _paired_log_ratio_summary(
        _paired_frame(ratios),
        treatment="C2",
        baseline="C1",
        time_steps=6,
        metric="latency",
        margin_percent=10.0,
    )
    assert result["status"] == expected


def test_known_exceedance_precedes_a_missing_metric_in_overall_status() -> None:
    assert _overall_efficiency_status(
        ["not_assessed_missing_metric", "exceeds_tolerance"]
    ) == "exceeds_tolerance"


def _environment(name: str) -> tuple[str, str]:
    encoded = json.dumps(
        {
            "device": "cuda:0",
            "hardware": {"name": name},
            "software": {"pytorch": "test"},
            "precision": "float32",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_training_environment_audit_rejects_mixed_hardware() -> None:
    first = _environment("gpu-a")
    second = _environment("gpu-b")
    frame = pd.DataFrame(
        {
            "training_environment_identity": [first[0], second[0]],
            "training_environment_sha256": [first[1], second[1]],
        }
    )
    with pytest.raises(IncompleteSeedBlockError, match="mix hardware"):
        _validate_training_environment(frame)


def test_training_environment_audit_rejects_hash_mismatch() -> None:
    identity, _ = _environment("gpu-a")
    frame = pd.DataFrame(
        {
            "training_environment_identity": [identity],
            "training_environment_sha256": ["0" * 64],
        }
    )
    with pytest.raises(IncompleteSeedBlockError, match="identity/hash mismatch"):
        _validate_training_environment(frame)
