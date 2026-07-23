"""Pre-specified statistics for the TA-LIF x MS-ResNet factorial study.

The four conditions form paired seed blocks.  This module deliberately refuses
partial blocks: silently dropping a failed run would break the paired design and
can bias both the main effects and the interaction estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf


CONDITIONS: tuple[str, ...] = ("C1", "C2", "C3", "C4")
CONDITION_FACTORS: Mapping[str, tuple[float, float]] = {
    # +/- 0.5 coding makes the coefficients directly interpretable as the
    # average main effects and the interaction as the difference-in-differences.
    "C1": (-0.5, -0.5),  # LIF, conventional spiking ResNet
    "C2": (+0.5, -0.5),  # TA-LIF, conventional spiking ResNet
    "C3": (-0.5, +0.5),  # LIF, MS-ResNet
    "C4": (+0.5, +0.5),  # TA-LIF, MS-ResNet
}
CONDITION_LABELS: Mapping[str, tuple[str, str]] = {
    "C1": ("lif", "spiking_resnet"),
    "C2": ("ta_lif", "spiking_resnet"),
    "C3": ("lif", "ms_resnet"),
    "C4": ("ta_lif", "ms_resnet"),
}
DEFAULT_GROUP_COLUMNS: tuple[str, ...] = ("dataset", "depth", "time_steps")
MODEL_FORMULA = "accuracy ~ neuron * topology + C(seed)"
SUCCESS_STATUSES = frozenset({"complete", "completed", "success", "succeeded", "finished", "ok"})
CONFIRMATORY_SEEDS: tuple[int, ...] = (11, 22, 33, 44, 55)
PRIMARY_ACCURACY_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("cifar100", 20, 6),
    ("cifar10dvs", 20, 10),
)
CONFIRMATORY_ALPHA = 0.05
CONFIRMATORY_CI_LEVEL = 0.95
INTERACTION_SESOI_PP = 0.50
ACCURACY_COMPOSITION_MARGIN_PP = 0.50
SENSITIVITY_BOOTSTRAP_RESAMPLES = 10_000
SENSITIVITY_BOOTSTRAP_SEED = 20_260_719


class IncompleteSeedBlockError(ValueError):
    """Raised when the four-condition paired-seed contract is violated."""


@dataclass(frozen=True)
class EffectEstimate:
    name: str
    estimate: float
    standard_error: float
    ci_low: float
    ci_high: float
    p_value: float


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    p_value_two_sided: float
    n_resamples: int
    random_seed: int
    n_seed_blocks: int


@dataclass
class FactorialAccuracyResult:
    group: dict[str, Any]
    n_seed_blocks: int
    n_observations: int
    cell_summary: pd.DataFrame
    neuron_effect: EffectEstimate
    topology_effect: EffectEstimate
    interaction_effect: EffectEstimate
    interaction_difference_in_differences: float
    bootstrap_interaction: BootstrapEstimate
    residual_shapiro_w: float
    residual_shapiro_p: float
    residual_degrees_of_freedom: float
    formula: str = MODEL_FORMULA


@dataclass(frozen=True)
class SeedInteractionResult:
    """Confirmatory interaction estimate from the five seed-level DID values."""

    group: dict[str, Any]
    seed_contrasts: tuple[tuple[int, float], ...]
    estimate: float
    standard_deviation: float
    standard_error: float
    degrees_of_freedom: int
    ci_low: float
    ci_high: float
    ci90_low: float
    ci90_high: float
    sesoi_pp: float
    t_statistic_sesoi: float
    p_value_one_sided_sesoi: float
    bootstrap_sensitivity: BootstrapEstimate


@dataclass(frozen=True)
class SeedContrastResult:
    """Paired seed-level accuracy contrast used by frozen wording gates."""

    name: str
    seed_contrasts: tuple[tuple[int, float], ...]
    estimate: float
    standard_deviation: float
    standard_error: float
    degrees_of_freedom: int
    ci95_low: float
    ci95_high: float
    ci90_low: float
    ci90_high: float
    noninferiority_margin_pp: float
    noninferiority_lower_bound_95: float


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _group_label(key: Any, columns: Sequence[str]) -> str:
    values = key if isinstance(key, tuple) else (key,)
    return ", ".join(f"{column}={value}" for column, value in zip(columns, values))


def validate_complete_seed_blocks(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    condition_column: str = "condition",
    seed_column: str = "seed",
    outcome_column: str | None = "accuracy",
    status_column: str | None = "status",
    expected_conditions: Sequence[str] = CONDITIONS,
    min_seed_blocks: int = 5,
    expected_seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Validate and return a normalized copy of complete paired seed blocks.

    Each ``group_columns + seed`` block must contain exactly one row for every
    condition.  When present, status must indicate success and the requested
    outcome must be finite.  The returned frame has upper-case condition labels.
    """

    audit_columns = (
        "neuron", "topology", "protocol_hash", "split_manifest_sha256",
        "shared_weight_sha256", "test_checkpoint_sha256",
    )
    required = [*group_columns, condition_column, seed_column, *audit_columns]
    if outcome_column is not None:
        required.append(outcome_column)
    _require_columns(frame, required)
    if frame.empty:
        raise IncompleteSeedBlockError("No seed-level rows were supplied")

    normalized = frame.copy()
    normalized[condition_column] = normalized[condition_column].astype(str).str.strip().str.upper()
    if expected_seeds is not None:
        numeric_seeds = pd.to_numeric(normalized[seed_column], errors="coerce")
        invalid_seed = ~np.isfinite(numeric_seeds.to_numpy(dtype=float)) | (
            numeric_seeds.to_numpy(dtype=float) != np.floor(numeric_seeds.to_numpy(dtype=float))
        )
        if invalid_seed.any():
            rows = normalized.index[invalid_seed].tolist()[:10]
            raise IncompleteSeedBlockError(f"Invalid integer seed values at rows {rows}")
        normalized[seed_column] = numeric_seeds.astype(int)
    key_columns = [*group_columns, condition_column, seed_column]

    null_key = normalized[[*group_columns, condition_column, seed_column]].isna().any(axis=1)
    if null_key.any():
        rows = normalized.index[null_key].tolist()[:10]
        raise IncompleteSeedBlockError(f"Null group/condition/seed values at rows {rows}")

    expected = {str(value).upper() for value in expected_conditions}
    observed = set(normalized[condition_column].unique())
    unexpected = sorted(observed - expected)
    if unexpected:
        raise IncompleteSeedBlockError(f"Unexpected condition labels: {unexpected}")

    empty_audit = normalized[list(audit_columns)].fillna("").astype(str).apply(
        lambda column: column.str.strip().eq("")
    )
    if empty_audit.any(axis=None):
        rows = normalized.index[empty_audit.any(axis=1)].tolist()[:10]
        raise IncompleteSeedBlockError(f"Missing pairing/audit values at rows {rows}")
    for index, row in normalized.iterrows():
        expected_neuron, expected_topology = CONDITION_LABELS[row[condition_column]]
        neuron = str(row["neuron"]).strip().lower().replace("-", "_")
        topology = str(row["topology"]).strip().lower().replace("-", "_")
        if neuron != expected_neuron or topology != expected_topology:
            raise IncompleteSeedBlockError(
                f"Condition mapping mismatch at row {index}: {row[condition_column]} "
                f"has neuron={neuron}, topology={topology}"
            )

    duplicates = normalized.duplicated(key_columns, keep=False)
    if duplicates.any():
        examples = normalized.loc[duplicates, key_columns].head(10).to_dict("records")
        raise IncompleteSeedBlockError(f"Duplicate condition rows within seed blocks: {examples}")

    failures: list[str] = []
    if normalized["protocol_hash"].nunique(dropna=False) != 1:
        failures.append("Seed rows are bound to multiple protocol hashes")
    for dataset, dataset_frame in normalized.groupby(group_columns[0], sort=True, dropna=False):
        if dataset_frame["split_manifest_sha256"].nunique(dropna=False) != 1:
            failures.append(f"dataset={dataset}: multiple train/validation split manifests")
    grouper: str | list[str] = list(group_columns)
    if len(group_columns) == 1:
        grouper = group_columns[0]
    frozen_seed_set = set(int(seed) for seed in expected_seeds) if expected_seeds is not None else None
    if frozen_seed_set is not None and len(frozen_seed_set) != len(tuple(expected_seeds or ())):
        raise ValueError("expected_seeds must not contain duplicates")
    for group_key, group in normalized.groupby(grouper, sort=True, dropna=False):
        label = _group_label(group_key, group_columns)
        seeds = list(pd.unique(group[seed_column]))
        if frozen_seed_set is not None and set(int(seed) for seed in seeds) != frozen_seed_set:
            missing_seeds = sorted(frozen_seed_set - set(int(seed) for seed in seeds))
            extra_seeds = sorted(set(int(seed) for seed in seeds) - frozen_seed_set)
            failures.append(
                f"{label}: confirmatory seeds differ; missing={missing_seeds or 'none'}, "
                f"extra={extra_seeds or 'none'}"
            )
        elif frozen_seed_set is None and len(seeds) < min_seed_blocks:
            failures.append(
                f"{label}: found {len(seeds)} seed blocks; at least {min_seed_blocks} are required"
            )
        for seed, seed_frame in group.groupby(seed_column, sort=True, dropna=False):
            condition_set = set(seed_frame[condition_column])
            if condition_set != expected:
                missing = sorted(expected - condition_set)
                extra = sorted(condition_set - expected)
                failures.append(
                    f"{label}, seed={seed}: missing={missing or 'none'}, extra={extra or 'none'}"
                )
            if seed_frame["shared_weight_sha256"].nunique(dropna=False) != 1:
                failures.append(f"{label}, seed={seed}: C1-C4 shared initialization hashes differ")

    duplicated_checkpoints = normalized["test_checkpoint_sha256"].duplicated(keep=False)
    if duplicated_checkpoints.any():
        failures.append("A test checkpoint SHA-256 is reused by more than one seed-condition row")

    if outcome_column is not None:
        numeric_outcome = pd.to_numeric(normalized[outcome_column], errors="coerce")
        invalid = ~np.isfinite(numeric_outcome.to_numpy(dtype=float))
        if invalid.any():
            examples = normalized.loc[invalid, key_columns].head(10).to_dict("records")
            failures.append(f"non-finite {outcome_column} values: {examples}")
        normalized[outcome_column] = numeric_outcome

    if status_column and status_column in normalized.columns:
        status = normalized[status_column].fillna("").astype(str).str.strip().str.lower()
        bad_status = ~status.isin(SUCCESS_STATUSES)
        if bad_status.any():
            examples = normalized.loc[bad_status, [*key_columns, status_column]].head(10)
            failures.append(f"non-success run statuses: {examples.to_dict('records')}")

    if failures:
        preview = "; ".join(failures[:20])
        remainder = len(failures) - 20
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise IncompleteSeedBlockError(f"Incomplete paired-seed design: {preview}{suffix}")

    return normalized


def condition_effect_codes(conditions: pd.Series) -> pd.DataFrame:
    """Return +/-0.5 neuron and topology codes for C1--C4."""

    labels = conditions.astype(str).str.strip().str.upper()
    unknown = sorted(set(labels) - set(CONDITION_FACTORS))
    if unknown:
        raise ValueError(f"Cannot effect-code unknown conditions: {unknown}")
    codes = labels.map(CONDITION_FACTORS)
    return pd.DataFrame(
        {
            "neuron": codes.map(lambda pair: pair[0]).astype(float),
            "topology": codes.map(lambda pair: pair[1]).astype(float),
        },
        index=conditions.index,
    )


def difference_in_differences(
    frame: pd.DataFrame,
    *,
    outcome_column: str = "accuracy",
    condition_column: str = "condition",
) -> float:
    """Compute (C4 - C3) - (C2 - C1) from condition means."""

    _require_columns(frame, [outcome_column, condition_column])
    labels = frame[condition_column].astype(str).str.strip().str.upper()
    values = pd.to_numeric(frame[outcome_column], errors="coerce")
    means = values.groupby(labels).mean()
    missing = [condition for condition in CONDITIONS if condition not in means]
    if missing:
        raise IncompleteSeedBlockError(f"Cannot compute interaction; missing conditions {missing}")
    return float((means["C4"] - means["C3"]) - (means["C2"] - means["C1"]))


def grouped_seed_bootstrap(
    frame: pd.DataFrame,
    *,
    outcome_column: str = "accuracy",
    condition_column: str = "condition",
    seed_column: str = "seed",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    n_resamples: int = 10_000,
    random_seed: int = 20_260_719,
    alpha: float = 0.05,
    min_seed_blocks: int = 5,
    expected_seeds: Sequence[int] | None = None,
) -> BootstrapEstimate:
    """Bootstrap the interaction while resampling complete seed blocks.

    Resampling the four observations independently would destroy the pairing.
    Here each seed contributes one difference-in-differences contrast and those
    complete-block contrasts are resampled as a unit.
    """

    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    normalized = validate_complete_seed_blocks(
        frame,
        group_columns=group_columns,
        condition_column=condition_column,
        seed_column=seed_column,
        outcome_column=outcome_column,
        min_seed_blocks=min_seed_blocks,
        expected_seeds=expected_seeds,
    )
    if normalized.groupby(list(group_columns), dropna=False).ngroups != 1:
        raise ValueError("grouped_seed_bootstrap expects exactly one dataset/depth/time-step group")

    pivot = normalized.pivot(index=seed_column, columns=condition_column, values=outcome_column)
    contrasts = (
        pivot["C4"].to_numpy(dtype=float)
        - pivot["C3"].to_numpy(dtype=float)
        - pivot["C2"].to_numpy(dtype=float)
        + pivot["C1"].to_numpy(dtype=float)
    )
    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, len(contrasts), size=(n_resamples, len(contrasts)))
    draws = contrasts[indices].mean(axis=1)
    ci_low, ci_high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])

    # The +1 correction prevents a reported p-value of exactly zero.
    lower_tail = (np.count_nonzero(draws <= 0.0) + 1) / (n_resamples + 1)
    upper_tail = (np.count_nonzero(draws >= 0.0) + 1) / (n_resamples + 1)
    p_value = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return BootstrapEstimate(
        estimate=float(contrasts.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value_two_sided=float(p_value),
        n_resamples=n_resamples,
        random_seed=random_seed,
        n_seed_blocks=len(contrasts),
    )


def seed_level_interaction_contrasts(
    frame: pd.DataFrame,
    *,
    outcome_column: str = "accuracy",
    condition_column: str = "condition",
    seed_column: str = "seed",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    expected_seeds: Sequence[int] = CONFIRMATORY_SEEDS,
) -> pd.Series:
    """Return one ``(C4-C3)-(C2-C1)`` accuracy contrast per frozen seed."""

    normalized = validate_complete_seed_blocks(
        frame,
        group_columns=group_columns,
        condition_column=condition_column,
        seed_column=seed_column,
        outcome_column=outcome_column,
        expected_seeds=expected_seeds,
    )
    if normalized.groupby(list(group_columns), dropna=False).ngroups != 1:
        raise ValueError("seed_level_interaction_contrasts expects exactly one analysis group")
    pivot = normalized.pivot(index=seed_column, columns=condition_column, values=outcome_column)
    pivot = pivot.reindex([int(seed) for seed in expected_seeds])
    contrasts = pivot["C4"] - pivot["C3"] - pivot["C2"] + pivot["C1"]
    values = contrasts.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise IncompleteSeedBlockError("Confirmatory seed-level interaction contains non-finite values")
    contrasts.name = "interaction_did_pp"
    return contrasts


def analyze_seed_level_condition_contrast(
    frame: pd.DataFrame,
    treatment: str,
    comparator: str,
    *,
    outcome_column: str = "accuracy",
    condition_column: str = "condition",
    seed_column: str = "seed",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> SeedContrastResult:
    """Estimate one paired condition contrast across the five frozen seeds."""

    treatment = str(treatment).strip().upper()
    comparator = str(comparator).strip().upper()
    if treatment not in CONDITIONS or comparator not in CONDITIONS or treatment == comparator:
        raise ValueError("treatment and comparator must be distinct C1--C4 conditions")
    normalized = validate_complete_seed_blocks(
        frame,
        group_columns=group_columns,
        condition_column=condition_column,
        seed_column=seed_column,
        outcome_column=outcome_column,
        expected_seeds=CONFIRMATORY_SEEDS,
    )
    if normalized.groupby(list(group_columns), dropna=False).ngroups != 1:
        raise ValueError("analyze_seed_level_condition_contrast expects one analysis group")
    pivot = normalized.pivot(index=seed_column, columns=condition_column, values=outcome_column)
    pivot = pivot.reindex(CONFIRMATORY_SEEDS)
    contrasts = pivot[treatment] - pivot[comparator]
    values = contrasts.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise IncompleteSeedBlockError(
            f"Seed-level {treatment}-{comparator} contrast contains non-finite values"
        )
    n = len(values)
    degrees_of_freedom = n - 1
    estimate = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = float(standard_deviation / np.sqrt(n))
    if standard_error <= np.finfo(float).eps:
        raise IncompleteSeedBlockError(
            f"Seed-level {treatment}-{comparator} contrast has zero estimable variance"
        )
    critical95 = float(stats.t.ppf(0.975, degrees_of_freedom))
    critical90 = float(stats.t.ppf(0.95, degrees_of_freedom))
    ci95_low = float(estimate - critical95 * standard_error)
    ci95_high = float(estimate + critical95 * standard_error)
    ci90_low = float(estimate - critical90 * standard_error)
    ci90_high = float(estimate + critical90 * standard_error)
    values_to_check = (
        estimate, standard_deviation, standard_error, ci95_low, ci95_high, ci90_low, ci90_high,
    )
    if not np.isfinite(values_to_check).all():
        raise IncompleteSeedBlockError(
            f"Seed-level {treatment}-{comparator} inference produced non-finite output"
        )
    return SeedContrastResult(
        name=f"{treatment}-{comparator}",
        seed_contrasts=tuple((int(seed), float(value)) for seed, value in contrasts.items()),
        estimate=estimate,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        ci90_low=ci90_low,
        ci90_high=ci90_high,
        noninferiority_margin_pp=ACCURACY_COMPOSITION_MARGIN_PP,
        noninferiority_lower_bound_95=ci90_low,
    )


def analyze_seed_level_interaction(
    frame: pd.DataFrame,
    *,
    outcome_column: str = "accuracy",
    condition_column: str = "condition",
    seed_column: str = "seed",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> SeedInteractionResult:
    """Run the frozen one-sided SESOI test and two-sided 95% CI.

    The scientific replication unit is a complete C1--C4 seed block.  The
    confirmatory null is ``mean(DID) <= 0.50`` percentage points.  Bootstrap
    output is generated unconditionally as sensitivity analysis only.
    """

    contrasts = seed_level_interaction_contrasts(
        frame,
        outcome_column=outcome_column,
        condition_column=condition_column,
        seed_column=seed_column,
        group_columns=group_columns,
        expected_seeds=CONFIRMATORY_SEEDS,
    )
    values = contrasts.to_numpy(dtype=float)
    n = len(values)
    degrees_of_freedom = n - 1
    estimate = float(values.mean())
    standard_deviation = float(values.std(ddof=1))
    standard_error = float(standard_deviation / np.sqrt(n))
    critical = float(stats.t.ppf(1.0 - CONFIRMATORY_ALPHA / 2.0, degrees_of_freedom))
    critical90 = float(stats.t.ppf(0.95, degrees_of_freedom))
    if standard_error <= np.finfo(float).eps:
        raise IncompleteSeedBlockError(
            "Confirmatory interaction has zero estimable between-seed variance"
        )
    ci_low = float(estimate - critical * standard_error)
    ci_high = float(estimate + critical * standard_error)
    ci90_low = float(estimate - critical90 * standard_error)
    ci90_high = float(estimate + critical90 * standard_error)
    t_statistic = float((estimate - INTERACTION_SESOI_PP) / standard_error)
    p_value = float(stats.t.sf(t_statistic, degrees_of_freedom))
    if not np.isfinite(
        [
            estimate, standard_deviation, standard_error, ci_low, ci_high,
            ci90_low, ci90_high, t_statistic, p_value,
        ]
    ).all():
        raise IncompleteSeedBlockError("Confirmatory interaction inference produced non-finite output")

    normalized = validate_complete_seed_blocks(
        frame,
        group_columns=group_columns,
        condition_column=condition_column,
        seed_column=seed_column,
        outcome_column=outcome_column,
        expected_seeds=CONFIRMATORY_SEEDS,
    )
    grouped = normalized.groupby(list(group_columns), sort=True, dropna=False)
    group_key, _ = next(iter(grouped))
    key_values = group_key if isinstance(group_key, tuple) else (group_key,)
    bootstrap = grouped_seed_bootstrap(
        normalized,
        outcome_column=outcome_column,
        condition_column=condition_column,
        seed_column=seed_column,
        group_columns=group_columns,
        n_resamples=SENSITIVITY_BOOTSTRAP_RESAMPLES,
        random_seed=SENSITIVITY_BOOTSTRAP_SEED,
        alpha=CONFIRMATORY_ALPHA,
        expected_seeds=CONFIRMATORY_SEEDS,
    )
    return SeedInteractionResult(
        group=dict(zip(group_columns, key_values)),
        seed_contrasts=tuple((int(seed), float(value)) for seed, value in contrasts.items()),
        estimate=estimate,
        standard_deviation=standard_deviation,
        standard_error=standard_error,
        degrees_of_freedom=degrees_of_freedom,
        ci_low=ci_low,
        ci_high=ci_high,
        ci90_low=ci90_low,
        ci90_high=ci90_high,
        sesoi_pp=INTERACTION_SESOI_PP,
        t_statistic_sesoi=t_statistic,
        p_value_one_sided_sesoi=p_value,
        bootstrap_sensitivity=bootstrap,
    )


def _effect_from_fit(fit: Any, coefficient: str, name: str, alpha: float) -> EffectEstimate:
    interval = fit.conf_int(alpha=alpha).loc[coefficient]
    return EffectEstimate(
        name=name,
        estimate=float(fit.params[coefficient]),
        standard_error=float(fit.bse[coefficient]),
        ci_low=float(interval.iloc[0]),
        ci_high=float(interval.iloc[1]),
        p_value=float(fit.pvalues[coefficient]),
    )


def fit_seed_blocked_accuracy(
    frame: pd.DataFrame,
    *,
    outcome_column: str = "accuracy",
    condition_column: str = "condition",
    seed_column: str = "seed",
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
    alpha: float = 0.05,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20_260_719,
    min_seed_blocks: int = 5,
) -> FactorialAccuracyResult:
    """Fit ``accuracy ~ neuron * topology + C(seed)`` with effect coding."""

    normalized = validate_complete_seed_blocks(
        frame,
        group_columns=group_columns,
        condition_column=condition_column,
        seed_column=seed_column,
        outcome_column=outcome_column,
        min_seed_blocks=min_seed_blocks,
    )
    grouped = normalized.groupby(list(group_columns), sort=True, dropna=False)
    if grouped.ngroups != 1:
        raise ValueError("fit_seed_blocked_accuracy expects exactly one analysis group")

    group_key, single_group = next(iter(grouped))
    key_values = group_key if isinstance(group_key, tuple) else (group_key,)
    group_metadata = dict(zip(group_columns, key_values))

    model_frame = single_group.rename(
        columns={outcome_column: "accuracy", seed_column: "seed"}
    ).copy()
    codes = condition_effect_codes(model_frame[condition_column])
    model_frame[["neuron", "topology"]] = codes[["neuron", "topology"]]
    model_frame["accuracy"] = pd.to_numeric(model_frame["accuracy"], errors="raise")

    fit = smf.ols(MODEL_FORMULA, data=model_frame).fit()
    neuron_effect = _effect_from_fit(fit, "neuron", "TA-LIF main effect", alpha)
    topology_effect = _effect_from_fit(fit, "topology", "MS-ResNet main effect", alpha)
    interaction_effect = _effect_from_fit(
        fit, "neuron:topology", "neuron-by-topology interaction", alpha
    )

    direct_interaction = difference_in_differences(
        single_group,
        outcome_column=outcome_column,
        condition_column=condition_column,
    )
    if not np.isclose(interaction_effect.estimate, direct_interaction, atol=1e-10, rtol=1e-10):
        raise RuntimeError(
            "Effect-coded interaction does not equal the direct difference-in-differences"
        )

    cell_summary = (
        single_group.assign(
            **{condition_column: single_group[condition_column].astype(str).str.upper()}
        )
        .groupby(condition_column, sort=True)[outcome_column]
        .agg([("mean", "mean"), ("sd", "std"), ("n", "count")])
        .reindex(CONDITIONS)
        .reset_index()
    )

    residuals = np.asarray(fit.resid, dtype=float)
    if 3 <= len(residuals) <= 5000 and np.ptp(residuals) > np.finfo(float).eps:
        shapiro = stats.shapiro(residuals)
        shapiro_w, shapiro_p = float(shapiro.statistic), float(shapiro.pvalue)
    else:
        shapiro_w, shapiro_p = float("nan"), float("nan")

    bootstrap = grouped_seed_bootstrap(
        single_group,
        outcome_column=outcome_column,
        condition_column=condition_column,
        seed_column=seed_column,
        group_columns=group_columns,
        n_resamples=bootstrap_resamples,
        random_seed=bootstrap_seed,
        alpha=alpha,
        min_seed_blocks=min_seed_blocks,
    )
    return FactorialAccuracyResult(
        group=group_metadata,
        n_seed_blocks=int(single_group[seed_column].nunique()),
        n_observations=int(fit.nobs),
        cell_summary=cell_summary,
        neuron_effect=neuron_effect,
        topology_effect=topology_effect,
        interaction_effect=interaction_effect,
        interaction_difference_in_differences=direct_interaction,
        bootstrap_interaction=bootstrap,
        residual_shapiro_w=shapiro_w,
        residual_shapiro_p=shapiro_p,
        residual_degrees_of_freedom=float(fit.df_resid),
    )


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm-adjusted p-values, failing closed on an incomplete family."""

    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError("Holm adjustment requires a non-empty one-dimensional family")
    if not np.isfinite(values).all():
        raise ValueError("Holm family contains a missing or non-finite p-value")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Holm family p-values must be between zero and one")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    count = len(sorted_values)
    sorted_adjusted = np.maximum.accumulate(
        np.asarray([(count - index) * value for index, value in enumerate(sorted_values)])
    )
    sorted_adjusted = np.minimum(sorted_adjusted, 1.0)
    restored = np.empty_like(sorted_adjusted)
    restored[order] = sorted_adjusted
    return restored
