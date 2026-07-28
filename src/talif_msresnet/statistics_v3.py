"""Frozen paired-accuracy statistics for the isolated TA-LIF-only v3 study.

This module intentionally does not reuse the legacy four-cell factorial
statistics.  It accepts only the complete 20-row v3 formal result matrix and
keeps the confirmatory CIFAR-100 decision separate from the CIFAR10-DVS
replication and both sensitivity analyses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


V3_CONDITIONS: tuple[str, str] = ("C1", "C2")
V3_FORMAL_SEEDS: tuple[int, ...] = (
    1882214332,
    836017246,
    126468528,
    2075015328,
    419442269,
)
V3_DATASET_GROUPS: Mapping[str, tuple[int, int]] = {
    "cifar100": (20, 6),
    "cifar10dvs": (20, 10),
}
V3_PRIMARY_DATASET = "cifar100"
V3_REPLICATION_DATASET = "cifar10dvs"
V3_ALPHA = 0.05
V3_CONFIDENCE_LEVEL = 0.95
V3_SIGN_FLIP_ASSIGNMENTS = 32
V3_BOOTSTRAP_RESAMPLES = 10_000
V3_BOOTSTRAP_SEEDS: Mapping[str, int] = {
    "cifar100": 1033863572,
    "cifar10dvs": 1367073951,
}

_AUDIT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "experiment",
    "dataset",
    "depth",
    "time_steps",
    "condition",
    "neuron",
    "topology",
    "seed",
    "status",
    "protocol_hash",
    "config_hash",
    "split_manifest_sha256",
    "shared_weight_sha256",
    "test_checkpoint_sha256",
    "test_evaluated_at",
)


class V3PairedAnalysisError(ValueError):
    """Raised when data violate the frozen v3 paired-analysis contract."""


@dataclass(frozen=True)
class PairedDifference:
    dataset: str
    seed: int
    c1_accuracy_pp: float
    c2_accuracy_pp: float
    difference_pp: float


@dataclass(frozen=True)
class PairedTResult:
    estimable: bool
    n_pairs: int
    degrees_of_freedom: int
    mean_difference_pp: float
    standard_deviation_pp: float
    standard_error_pp: float
    t_statistic: float | None
    p_value_one_sided_greater: float | None
    ci95_low_pp: float | None
    ci95_high_pp: float | None
    alpha: float = V3_ALPHA
    confidence_level: float = V3_CONFIDENCE_LEVEL
    null_hypothesis: str = "mean_delta_le_0"
    alternative_hypothesis: str = "mean_delta_gt_0"


@dataclass(frozen=True)
class SignFlipSensitivity:
    observed_mean_pp: float
    assignments: int
    extreme_assignments: int
    p_value_one_sided_greater: float
    role: str = "sensitivity_only_cannot_rescue_primary"


@dataclass(frozen=True)
class BootstrapSensitivity:
    observed_mean_pp: float
    ci95_low_pp: float
    ci95_high_pp: float
    n_resamples: int
    random_seed: int
    n_complete_pairs: int
    rng: str = "numpy.random.Generator(numpy.random.PCG64)"
    interval: str = "two_sided_95_percent_linear_percentile"
    role: str = "sensitivity_only_cannot_rescue_primary"


@dataclass(frozen=True)
class DatasetPairedAnalysis:
    dataset: str
    role: str
    raw_differences: tuple[PairedDifference, ...]
    paired_t: PairedTResult
    sign_flip: SignFlipSensitivity
    bootstrap: BootstrapSensitivity


@dataclass(frozen=True)
class V3ClaimGates:
    primary_improvement_allowed: bool
    primary_result: str
    dvs_directional_support: bool
    dvs_replication_improvement_allowed: bool
    cross_domain_improvement_allowed: bool
    dvs_cannot_rescue_primary: bool = True


@dataclass(frozen=True)
class V3PairedAnalysis:
    protocol_hash: str
    datasets: Mapping[str, DatasetPairedAnalysis]
    claim_gates: V3ClaimGates

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without converting missing inference to NaN."""

        return asdict(self)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise V3PairedAnalysisError(f"Missing required v3 columns: {', '.join(missing)}")


def _nonempty_text(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame[list(columns)].fillna("").astype(str).apply(lambda value: value.str.strip().eq(""))


def _accuracy_in_percentage_points(frame: pd.DataFrame) -> pd.Series:
    """Extract final test accuracy and normalize the frozen proportion scale to pp."""

    if "test_accuracy" in frame.columns:
        values = pd.to_numeric(frame["test_accuracy"], errors="coerce")
        scale = 100.0
        label = "test_accuracy"
        valid_range = values.between(0.0, 1.0)
    elif "accuracy_pp" in frame.columns:
        values = pd.to_numeric(frame["accuracy_pp"], errors="coerce")
        scale = 1.0
        label = "accuracy_pp"
        valid_range = values.between(0.0, 100.0)
    elif "accuracy" in frame.columns:
        values = pd.to_numeric(frame["accuracy"], errors="coerce")
        # ``load_seed_metrics`` exposes percentage points as ``accuracy``.  A
        # direct protocol-scale frame in [0, 1] is also accepted and normalized.
        scale = 100.0 if values.notna().all() and values.between(0.0, 1.0).all() else 1.0
        label = "accuracy"
        valid_range = values.between(0.0, 1.0 if scale == 100.0 else 100.0)
    else:
        raise V3PairedAnalysisError(
            "Missing final test accuracy: expected test_accuracy, accuracy_pp, or accuracy"
        )
    finite = np.isfinite(values.to_numpy(dtype=float))
    if not finite.all() or not valid_range.all():
        rows = frame.index[~(pd.Series(finite, index=frame.index) & valid_range)].tolist()[:10]
        raise V3PairedAnalysisError(f"Invalid {label} values at rows {rows}")
    return values.astype(float) * scale


def _expected_run_id(dataset: str, condition: str, seed: int) -> str:
    depth, time_steps = V3_DATASET_GROUPS[dataset]
    return f"E1_{dataset}_d{depth}_t{time_steps}_{condition}_s{seed}"


def validate_v3_paired_blocks(
    frame: pd.DataFrame,
    *,
    expected_protocol_hash: str,
) -> pd.DataFrame:
    """Validate and normalize the exact 2 x 2 x 5 v3 formal result matrix.

    Validation is deliberately whole-frame.  The function never filters extra
    rows, so C3/C4 records, pilot seeds, partial pairs, duplicate records, and
    reused test checkpoints all fail before any statistic is calculated.
    """

    if not isinstance(expected_protocol_hash, str) or not expected_protocol_hash.strip():
        raise ValueError("expected_protocol_hash must be a non-empty string")
    _require_columns(frame, _AUDIT_COLUMNS)
    if len(frame) != 20:
        raise V3PairedAnalysisError(
            f"The v3 formal analysis requires exactly 20 rows, found {len(frame)}"
        )

    normalized = frame.copy()
    normalized["experiment"] = normalized["experiment"].astype(str).str.strip().str.upper()
    normalized["dataset"] = normalized["dataset"].astype(str).str.strip().str.lower()
    normalized["condition"] = normalized["condition"].astype(str).str.strip().str.upper()
    normalized["neuron"] = (
        normalized["neuron"].astype(str).str.strip().str.lower().str.replace("-", "_", regex=False)
    )
    normalized["topology"] = (
        normalized["topology"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_", regex=False)
    )
    for column in ("depth", "time_steps", "seed"):
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        integral = finite & (numeric.to_numpy(dtype=float) == np.floor(numeric.to_numpy(dtype=float)))
        if not integral.all():
            rows = normalized.index[~integral].tolist()[:10]
            raise V3PairedAnalysisError(f"Invalid integer {column} values at rows {rows}")
        normalized[column] = numeric.astype(int)
    normalized["accuracy_pp"] = _accuracy_in_percentage_points(normalized)

    empty = _nonempty_text(
        normalized,
        (
            "run_id",
            "protocol_hash",
            "config_hash",
            "split_manifest_sha256",
            "shared_weight_sha256",
            "test_checkpoint_sha256",
            "test_evaluated_at",
        ),
    )
    if empty.any(axis=None):
        rows = normalized.index[empty.any(axis=1)].tolist()[:10]
        raise V3PairedAnalysisError(f"Missing v3 audit or final-test values at rows {rows}")

    failures: list[str] = []
    if set(normalized["experiment"]) != {"E1"}:
        failures.append(f"experiments must be exactly ['E1'], found {sorted(set(normalized['experiment']))}")
    observed_datasets = set(normalized["dataset"])
    if observed_datasets != set(V3_DATASET_GROUPS):
        failures.append(
            "datasets must be exactly cifar100 and cifar10dvs; "
            f"found {sorted(observed_datasets)}"
        )
    observed_conditions = set(normalized["condition"])
    if observed_conditions != set(V3_CONDITIONS):
        failures.append(
            f"conditions must be exactly C1/C2; found {sorted(observed_conditions)}"
        )
    protocol_hashes = set(normalized["protocol_hash"].astype(str).str.strip())
    if protocol_hashes != {expected_protocol_hash}:
        failures.append(
            "result protocol hashes differ from the isolated v3 protocol: "
            f"{sorted(protocol_hashes)}"
        )

    duplicate_keys = normalized.duplicated(
        ["experiment", "dataset", "depth", "time_steps", "condition", "seed"], keep=False
    )
    if duplicate_keys.any():
        examples = normalized.loc[
            duplicate_keys, ["dataset", "condition", "seed", "run_id"]
        ].to_dict("records")
        failures.append(f"duplicate v3 result rows: {examples[:10]}")
    if normalized["run_id"].duplicated().any():
        failures.append("run_id is reused by more than one result row")
    if normalized["config_hash"].duplicated().any():
        failures.append("config_hash is reused by more than one result row")
    if normalized["test_checkpoint_sha256"].duplicated().any():
        failures.append("test checkpoint SHA-256 is reused by more than one result row")

    statuses = normalized["status"].fillna("").astype(str).str.strip().str.lower()
    if not statuses.eq("complete").all():
        failures.append("all v3 formal rows must have status='complete'")
    if "failed" in normalized.columns:
        failed = pd.to_numeric(normalized["failed"], errors="coerce")
        if failed.isna().any() or not failed.eq(0).all():
            failures.append("all v3 formal rows must record failed=0")

    expected_seed_set = set(V3_FORMAL_SEEDS)
    split_hash_by_dataset: dict[str, str] = {}
    shared_hash_owners: dict[str, tuple[str, int]] = {}
    for dataset, (expected_depth, expected_time_steps) in V3_DATASET_GROUPS.items():
        dataset_rows = normalized.loc[normalized["dataset"] == dataset]
        if len(dataset_rows) != 10:
            failures.append(f"dataset={dataset}: expected 10 C1/C2 rows, found {len(dataset_rows)}")
            continue
        if set(dataset_rows["depth"]) != {expected_depth}:
            failures.append(f"dataset={dataset}: depth must be {expected_depth}")
        if set(dataset_rows["time_steps"]) != {expected_time_steps}:
            failures.append(f"dataset={dataset}: time_steps must be {expected_time_steps}")
        observed_seeds = set(int(value) for value in dataset_rows["seed"])
        if observed_seeds != expected_seed_set:
            failures.append(
                f"dataset={dataset}: formal seeds differ; "
                f"missing={sorted(expected_seed_set - observed_seeds)}, "
                f"extra={sorted(observed_seeds - expected_seed_set)}"
            )
        split_hashes = set(dataset_rows["split_manifest_sha256"].astype(str).str.strip())
        if len(split_hashes) != 1:
            failures.append(f"dataset={dataset}: multiple split-manifest SHA-256 values")
        else:
            split_hash_by_dataset[dataset] = next(iter(split_hashes))

        for seed in V3_FORMAL_SEEDS:
            pair = dataset_rows.loc[dataset_rows["seed"] == seed]
            if len(pair) != 2 or set(pair["condition"]) != set(V3_CONDITIONS):
                failures.append(f"dataset={dataset}, seed={seed}: incomplete C1/C2 pair")
                continue
            mappings = {
                str(row.condition): (str(row.neuron), str(row.topology))
                for row in pair.itertuples(index=False)
            }
            if mappings != {
                "C1": ("lif", "spiking_resnet"),
                "C2": ("ta_lif", "spiking_resnet"),
            }:
                failures.append(
                    f"dataset={dataset}, seed={seed}: C1/C2 neuron-topology mapping differs"
                )
            expected_ids = {
                _expected_run_id(dataset, condition, seed) for condition in V3_CONDITIONS
            }
            if set(pair["run_id"].astype(str)) != expected_ids:
                failures.append(f"dataset={dataset}, seed={seed}: run_id binding differs")
            shared_hashes = set(pair["shared_weight_sha256"].astype(str).str.strip())
            if len(shared_hashes) != 1:
                failures.append(
                    f"dataset={dataset}, seed={seed}: C1/C2 shared initialization hashes differ"
                )
            else:
                shared_hash = next(iter(shared_hashes))
                owner = shared_hash_owners.setdefault(shared_hash, (dataset, seed))
                if owner != (dataset, seed):
                    failures.append(
                        "shared initialization SHA-256 is reused across distinct seed blocks"
                    )

    if len(split_hash_by_dataset) == 2 and len(set(split_hash_by_dataset.values())) != 2:
        failures.append("the two datasets reuse one split-manifest SHA-256")
    if failures:
        raise V3PairedAnalysisError("Invalid v3 paired design: " + "; ".join(failures))

    return normalized


def paired_differences_pp(
    normalized: pd.DataFrame,
    dataset: str,
) -> tuple[PairedDifference, ...]:
    """Return C2-C1 test-accuracy differences in the frozen seed order."""

    if dataset not in V3_DATASET_GROUPS:
        raise ValueError(f"Unknown v3 dataset: {dataset}")
    selected = normalized.loc[normalized["dataset"] == dataset]
    pivot = selected.pivot(index="seed", columns="condition", values="accuracy_pp")
    pivot = pivot.reindex(V3_FORMAL_SEEDS)
    if tuple(pivot.columns) != V3_CONDITIONS or pivot.isna().any(axis=None):
        raise V3PairedAnalysisError(f"dataset={dataset}: cannot construct complete C1/C2 pairs")
    return tuple(
        PairedDifference(
            dataset=dataset,
            seed=int(seed),
            c1_accuracy_pp=float(row.C1),
            c2_accuracy_pp=float(row.C2),
            difference_pp=float(row.C2 - row.C1),
        )
        for seed, row in pivot.iterrows()
    )


def paired_t_test(differences_pp: Sequence[float]) -> PairedTResult:
    """Apply the frozen one-sided t test and two-sided 95% t interval."""

    values = np.asarray(differences_pp, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise V3PairedAnalysisError("The paired t test requires exactly five finite differences")
    n_pairs = len(values)
    degrees_of_freedom = n_pairs - 1
    estimate = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = float(standard_deviation / np.sqrt(n_pairs))
    if np.ptp(values) == 0.0:
        return PairedTResult(
            estimable=False,
            n_pairs=n_pairs,
            degrees_of_freedom=degrees_of_freedom,
            mean_difference_pp=estimate,
            standard_deviation_pp=standard_deviation,
            standard_error_pp=standard_error,
            t_statistic=None,
            p_value_one_sided_greater=None,
            ci95_low_pp=None,
            ci95_high_pp=None,
        )

    t_statistic = float(estimate / standard_error)
    p_value = float(stats.t.sf(t_statistic, degrees_of_freedom))
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    ci_low = float(estimate - critical * standard_error)
    ci_high = float(estimate + critical * standard_error)
    if not np.isfinite((t_statistic, p_value, ci_low, ci_high)).all():
        raise V3PairedAnalysisError("Paired t inference produced non-finite output")
    return PairedTResult(
        estimable=True,
        n_pairs=n_pairs,
        degrees_of_freedom=degrees_of_freedom,
        mean_difference_pp=estimate,
        standard_deviation_pp=standard_deviation,
        standard_error_pp=standard_error,
        t_statistic=t_statistic,
        p_value_one_sided_greater=p_value,
        ci95_low_pp=ci_low,
        ci95_high_pp=ci_high,
    )


def exhaustive_sign_flip(differences_pp: Sequence[float]) -> SignFlipSensitivity:
    """Enumerate all 2^5 seed-level sign assignments for a one-sided sensitivity p."""

    values = np.asarray(differences_pp, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise V3PairedAnalysisError("Sign-flip sensitivity requires exactly five finite differences")
    signs = np.asarray(tuple(product((-1.0, 1.0), repeat=5)), dtype=float)
    statistics = (signs * values).mean(axis=1)
    observed = float(values.mean())
    extreme = int(np.count_nonzero(statistics >= observed))
    if len(statistics) != V3_SIGN_FLIP_ASSIGNMENTS:
        raise AssertionError("The frozen five-pair sign-flip space must contain 32 assignments")
    return SignFlipSensitivity(
        observed_mean_pp=observed,
        assignments=len(statistics),
        extreme_assignments=extreme,
        p_value_one_sided_greater=float(extreme / len(statistics)),
    )


def complete_pair_bootstrap(
    differences_pp: Sequence[float],
    *,
    dataset: str,
) -> BootstrapSensitivity:
    """Run the frozen 10,000-draw PCG64 complete-pair percentile bootstrap."""

    values = np.asarray(differences_pp, dtype=float)
    if values.shape != (5,) or not np.isfinite(values).all():
        raise V3PairedAnalysisError("Bootstrap sensitivity requires exactly five finite differences")
    if dataset not in V3_BOOTSTRAP_SEEDS:
        raise ValueError(f"Unknown v3 bootstrap dataset: {dataset}")
    random_seed = V3_BOOTSTRAP_SEEDS[dataset]
    rng = np.random.Generator(np.random.PCG64(random_seed))
    indices = rng.integers(
        0,
        5,
        size=(V3_BOOTSTRAP_RESAMPLES, 5),
        endpoint=False,
    )
    bootstrap_means = values[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975],
        method="linear",
    )
    return BootstrapSensitivity(
        observed_mean_pp=float(values.mean()),
        ci95_low_pp=float(ci_low),
        ci95_high_pp=float(ci_high),
        n_resamples=V3_BOOTSTRAP_RESAMPLES,
        random_seed=random_seed,
        n_complete_pairs=len(values),
    )


def _dataset_analysis(
    normalized: pd.DataFrame,
    dataset: str,
) -> DatasetPairedAnalysis:
    raw = paired_differences_pp(normalized, dataset)
    values = [item.difference_pp for item in raw]
    return DatasetPairedAnalysis(
        dataset=dataset,
        role=(
            "sole_confirmatory_primary"
            if dataset == V3_PRIMARY_DATASET
            else "prespecified_replication_external_validity"
        ),
        raw_differences=raw,
        paired_t=paired_t_test(values),
        sign_flip=exhaustive_sign_flip(values),
        bootstrap=complete_pair_bootstrap(values, dataset=dataset),
    )


def _passes_improvement_gate(result: PairedTResult) -> bool:
    return bool(
        result.estimable
        and result.mean_difference_pp > 0.0
        and result.p_value_one_sided_greater is not None
        and result.p_value_one_sided_greater < V3_ALPHA
    )


def analyze_v3_paired_accuracy(
    frame: pd.DataFrame,
    *,
    expected_protocol_hash: str,
) -> V3PairedAnalysis:
    """Analyze the isolated v3 C1/C2 formal matrix and apply wording gates."""

    normalized = validate_v3_paired_blocks(
        frame,
        expected_protocol_hash=expected_protocol_hash,
    )
    datasets = {
        dataset: _dataset_analysis(normalized, dataset)
        for dataset in (V3_PRIMARY_DATASET, V3_REPLICATION_DATASET)
    }
    primary = datasets[V3_PRIMARY_DATASET].paired_t
    replication = datasets[V3_REPLICATION_DATASET].paired_t
    primary_pass = _passes_improvement_gate(primary)
    replication_pass = _passes_improvement_gate(replication)
    claim_gates = V3ClaimGates(
        primary_improvement_allowed=primary_pass,
        primary_result=(
            "TA-LIF improved accuracy on CIFAR-100" if primary_pass else "inconclusive"
        ),
        dvs_directional_support=replication.mean_difference_pp > 0.0,
        dvs_replication_improvement_allowed=replication_pass,
        cross_domain_improvement_allowed=primary_pass and replication_pass,
    )
    return V3PairedAnalysis(
        protocol_hash=expected_protocol_hash,
        datasets=datasets,
        claim_gates=claim_gates,
    )
