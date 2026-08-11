"""Prospective paired-seed statistics for the frozen V7 mechanism contract.

The module is isolated from V5.  It implements the V7 replication test, the
two-member Holm mechanism family, the pre-specified M1 route branch, secondary
estimates, and prospective MDES sensitivity without changing training code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .config_v7 import (
    V7_ACTIVE_CONDITIONS,
    V7_ANALYSIS_CONTRACT,
    V7_BOOTSTRAP_SEEDS,
    V7_FORMAL_SEEDS,
)
from .statistics import holm_adjust

V7_ALPHA = float(V7_ANALYSIS_CONTRACT["alpha"])
V7_SESOI_PP = float(V7_ANALYSIS_CONTRACT["sesoi_pp"])
V7_CONDITIONS: tuple[str, ...] = V7_ACTIVE_CONDITIONS
V7_PLANNED_SEED_BLOCKS = len(V7_FORMAL_SEEDS)
V7_SIGN_FLIP_ASSIGNMENTS = int(V7_ANALYSIS_CONTRACT["sign_flip"]["assignments"])
V7_SIGN_FLIP_ROLE = str(V7_ANALYSIS_CONTRACT["sign_flip"]["role"])
V7_BOOTSTRAP_RESAMPLES = int(V7_ANALYSIS_CONTRACT["bootstrap"]["resamples"])
V7_BOOTSTRAP_ROLE = str(V7_ANALYSIS_CONTRACT["bootstrap"]["role"])
V7_BOOTSTRAP_RNG = str(V7_ANALYSIS_CONTRACT["bootstrap"]["rng"])
V7_BOOTSTRAP_INTERVAL = str(V7_ANALYSIS_CONTRACT["bootstrap"]["interval"])
V7_BOOTSTRAP_QUANTILE_METHOD = str(
    V7_ANALYSIS_CONTRACT["bootstrap"]["quantile_method"]
)


class V7StatisticsError(ValueError):
    """Raised when input or analysis violates the frozen V7 contract."""


@dataclass(frozen=True)
class ContrastSpec:
    """One direct paired contrast, expressed as treatment minus reference."""

    name: str
    treatment: str
    reference: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("contrast name must be non-empty")
        if self.treatment == self.reference:
            raise ValueError("treatment and reference must differ")
        unknown = {self.treatment, self.reference} - set(V7_CONDITIONS)
        if unknown:
            raise ValueError(f"unknown V7 conditions in contrast: {sorted(unknown)}")


V7_REPLICATION_CONTRAST = ContrastSpec("M4-M0", "M4", "M0")
V7_HOLM_CONTRASTS: tuple[ContrastSpec, ...] = (
    ContrastSpec("M4-M2", "M4", "M2"),
    ContrastSpec("M4-M3", "M4", "M3"),
)
V7_SECONDARY_CONTRASTS: tuple[ContrastSpec, ...] = (
    ContrastSpec("M1-M0", "M1", "M0"),
    ContrastSpec("PLIF-M0", "PLIF", "M0"),
    ContrastSpec("M4-PLIF", "M4", "PLIF"),
)


@dataclass(frozen=True)
class PairedContrastResult:
    name: str
    treatment: str
    reference: str
    n_pairs: int
    seed_differences_pp: tuple[tuple[int, float], ...]
    estimate_pp: float
    standard_deviation_pp: float
    standard_error_pp: float
    degrees_of_freedom: int
    ci95_low_pp: float
    ci95_high_pp: float
    superiority_t: float
    superiority_p_raw: float
    superiority_p_holm: float | None = None


@dataclass(frozen=True)
class TOSTResult:
    margin_pp: float
    ci90_low_pp: float
    ci90_high_pp: float
    p_lower: float
    p_upper: float
    p_value: float
    equivalent: bool


@dataclass(frozen=True)
class RouteConclusion:
    decision: str
    claim: str
    contrast: PairedContrastResult
    tost: TOSTResult


@dataclass(frozen=True)
class SignFlipSensitivity:
    observed_mean_pp: float
    assignments: int
    extreme_assignments: int
    p_value_one_sided_greater: float
    role: str = V7_SIGN_FLIP_ROLE


@dataclass(frozen=True)
class BootstrapSensitivity:
    observed_mean_pp: float
    ci95_low_pp: float
    ci95_high_pp: float
    n_resamples: int
    random_seed: int
    n_complete_blocks: int
    rng: str = V7_BOOTSTRAP_RNG
    interval: str = V7_BOOTSTRAP_INTERVAL
    quantile_method: str = V7_BOOTSTRAP_QUANTILE_METHOD
    role: str = V7_BOOTSTRAP_ROLE


@dataclass(frozen=True)
class ContrastSensitivity:
    name: str
    sign_flip: SignFlipSensitivity
    bootstrap: BootstrapSensitivity


@dataclass(frozen=True)
class StudyConclusion:
    decision: str
    claim: str
    replication_passed: bool
    holm_rejections: tuple[str, ...]
    route_decision: str


@dataclass(frozen=True)
class V7MechanismAnalysis:
    alpha: float
    sesoi_pp: float
    n_seed_blocks: int
    conditions: tuple[str, ...]
    replication: PairedContrastResult
    replication_passed: bool
    holm_family: tuple[PairedContrastResult, ...]
    secondary_estimates: tuple[PairedContrastResult, ...]
    route_conclusion: RouteConclusion
    sensitivity_analyses: tuple[ContrastSensitivity, ...]
    study_conclusion: StudyConclusion

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MDESSensitivity:
    n_pairs: int
    paired_sd_pp: float
    alpha: float
    power: float
    family_size: int
    conservative_local_alpha: float
    standardized_mdes: float
    detectable_difference_pp: float
    one_sided_sign_flip_min_p: float
    conservative_holm_sign_flip_floor: float


def _validate_alpha(alpha: float) -> float:
    value = float(alpha)
    if not 0.0 < value < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    return value


def _validate_frozen_alpha(alpha: float) -> float:
    value = _validate_alpha(alpha)
    if value != V7_ALPHA:
        raise ValueError(f"V7 alpha must remain frozen at {V7_ALPHA:g}")
    return value


def _require_contrast_identity(
    result: PairedContrastResult,
    spec: ContrastSpec,
) -> None:
    observed = (result.name, result.treatment, result.reference)
    expected = (spec.name, spec.treatment, spec.reference)
    if observed != expected:
        raise V7StatisticsError(
            f"Expected frozen contrast identity {expected}, found {observed}"
        )


def validate_v7_accuracy_blocks(
    frame: pd.DataFrame,
    *,
    expected_conditions: Sequence[str] = V7_CONDITIONS,
    expected_seeds: Sequence[int] = V7_FORMAL_SEEDS,
) -> pd.DataFrame:
    """Require the exact eight complete seed blocks and explicit pp input."""

    required = {"seed", "condition", "test_accuracy_pp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise V7StatisticsError(f"Missing V7 columns: {', '.join(missing)}")
    if frame.empty:
        raise V7StatisticsError("No V7 accuracy rows were supplied")

    normalized = frame.copy()
    normalized["condition"] = normalized["condition"].astype(str).str.strip().str.upper()
    numeric_seed = pd.to_numeric(normalized["seed"], errors="coerce")
    seed_values = numeric_seed.to_numpy(dtype=float)
    valid_seed = np.isfinite(seed_values) & (seed_values == np.floor(seed_values))
    if not valid_seed.all():
        raise V7StatisticsError(
            f"Invalid integer seed values at rows {normalized.index[~valid_seed].tolist()[:10]}"
        )
    normalized["seed"] = numeric_seed.astype(int)

    accuracy = pd.to_numeric(normalized["test_accuracy_pp"], errors="coerce")
    accuracy_values = accuracy.to_numpy(dtype=float)
    valid_accuracy = np.isfinite(accuracy_values) & (accuracy_values >= 0.0) & (
        accuracy_values <= 100.0
    )
    if not valid_accuracy.all():
        raise V7StatisticsError(
            "test_accuracy_pp must contain finite percentage points in [0, 100]"
        )
    normalized["test_accuracy_pp"] = accuracy.astype(float)

    conditions = tuple(str(value).strip().upper() for value in expected_conditions)
    if conditions != V7_CONDITIONS:
        raise ValueError(f"expected_conditions must equal the frozen V7 order {V7_CONDITIONS}")
    observed_conditions = set(normalized["condition"])
    if observed_conditions != set(conditions):
        raise V7StatisticsError(
            "V7 conditions differ from the frozen set: "
            f"missing={sorted(set(conditions) - observed_conditions)}, "
            f"extra={sorted(observed_conditions - set(conditions))}"
        )

    duplicate = normalized.duplicated(["seed", "condition"], keep=False)
    if duplicate.any():
        examples = normalized.loc[duplicate, ["seed", "condition"]].to_dict("records")
        raise V7StatisticsError(f"Duplicate V7 seed-condition rows: {examples[:10]}")

    frozen_seeds = tuple(int(value) for value in expected_seeds)
    if frozen_seeds != V7_FORMAL_SEEDS:
        raise ValueError("expected_seeds must equal the exact frozen V7 formal seed order")
    observed_seeds = {int(value) for value in normalized["seed"].unique()}
    if observed_seeds != set(frozen_seeds):
        raise V7StatisticsError(
            "V7 seeds differ from the frozen set: "
            f"missing={sorted(set(frozen_seeds) - observed_seeds)}, "
            f"extra={sorted(observed_seeds - set(frozen_seeds))}"
        )

    counts = normalized.groupby("seed", sort=True)["condition"].agg(
        lambda values: tuple(sorted(values))
    )
    expected_block = tuple(sorted(conditions))
    incomplete = counts[counts != expected_block]
    if not incomplete.empty:
        raise V7StatisticsError(
            "Incomplete V7 seed blocks: "
            + "; ".join(f"seed={seed}: {value}" for seed, value in incomplete.items())
        )
    return normalized


def _paired_values(frame: pd.DataFrame, spec: ContrastSpec) -> tuple[np.ndarray, np.ndarray]:
    pivot = frame.pivot(index="seed", columns="condition", values="test_accuracy_pp").reindex(
        V7_FORMAL_SEEDS
    )
    differences = (
        pivot[spec.treatment].to_numpy(dtype=float)
        - pivot[spec.reference].to_numpy(dtype=float)
    )
    return pivot.index.to_numpy(dtype=int), differences


def paired_contrast(frame: pd.DataFrame, spec: ContrastSpec) -> PairedContrastResult:
    """Calculate one unshifted, one-sided paired t contrast and a 95% CI."""

    seeds, differences = _paired_values(frame, spec)
    n_pairs = int(differences.size)
    if n_pairs != V7_PLANNED_SEED_BLOCKS:
        raise V7StatisticsError(
            f"Contrast {spec.name} requires {V7_PLANNED_SEED_BLOCKS} pairs, found {n_pairs}"
        )
    estimate = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        raise V7StatisticsError(
            f"Contrast {spec.name} has zero paired variance; t inference is not estimable"
        )
    se = sd / sqrt(n_pairs)
    df = n_pairs - 1
    critical = float(stats.t.ppf(0.975, df))
    statistic = estimate / se
    return PairedContrastResult(
        name=spec.name,
        treatment=spec.treatment,
        reference=spec.reference,
        n_pairs=n_pairs,
        seed_differences_pp=tuple(
            (int(seed), float(value)) for seed, value in zip(seeds, differences)
        ),
        estimate_pp=estimate,
        standard_deviation_pp=sd,
        standard_error_pp=se,
        degrees_of_freedom=df,
        ci95_low_pp=estimate - critical * se,
        ci95_high_pp=estimate + critical * se,
        superiority_t=statistic,
        superiority_p_raw=float(stats.t.sf(statistic, df)),
    )


def paired_tost(
    result: PairedContrastResult,
    *,
    margin_pp: float = V7_SESOI_PP,
    alpha: float = V7_ALPHA,
) -> TOSTResult:
    """Run the frozen paired TOST on an already validated seed contrast."""

    alpha = _validate_frozen_alpha(alpha)
    _require_contrast_identity(result, V7_SECONDARY_CONTRASTS[0])
    if float(margin_pp) != V7_SESOI_PP:
        raise ValueError(f"V7 SESOI must remain frozen at {V7_SESOI_PP:g} pp")
    if margin_pp <= 0.0:
        raise ValueError("TOST margin must be positive")
    estimate = result.estimate_pp
    se = result.standard_error_pp
    df = result.degrees_of_freedom
    lower_t = (estimate + margin_pp) / se
    upper_t = (estimate - margin_pp) / se
    p_lower = float(stats.t.sf(lower_t, df))
    p_upper = float(stats.t.cdf(upper_t, df))
    p_value = max(p_lower, p_upper)
    critical = float(stats.t.ppf(1.0 - alpha, df))
    return TOSTResult(
        margin_pp=float(margin_pp),
        ci90_low_pp=estimate - critical * se,
        ci90_high_pp=estimate + critical * se,
        p_lower=p_lower,
        p_upper=p_upper,
        p_value=p_value,
        equivalent=p_value < alpha,
    )


def classify_route_contribution(
    result: PairedContrastResult,
    *,
    alpha: float = V7_ALPHA,
    sesoi_pp: float = V7_SESOI_PP,
) -> RouteConclusion:
    """Apply the pre-specified M1-M0 positive/equivalent/inconclusive branch."""

    alpha = _validate_alpha(alpha)
    tost = paired_tost(result, margin_pp=sesoi_pp, alpha=alpha)
    positive = result.ci95_low_pp > 0.0 and result.estimate_pp >= sesoi_pp
    if positive:
        decision = "route_positive_evidence"
        claim = (
            "M1-M0 had a two-sided 95% CI above zero and a mean difference "
            f"of at least {sesoi_pp:g} pp."
        )
    elif tost.equivalent:
        decision = "route_practical_equivalence"
        claim = f"M1-M0 was equivalent within +/-{sesoi_pp:g} pp by paired TOST."
    else:
        decision = "route_contribution_inconclusive"
        claim = (
            "M1-M0 established neither the pre-specified positive-evidence gate "
            "nor practical equivalence; non-significance is not no effect."
        )
    return RouteConclusion(decision, claim, result, tost)


def exhaustive_sign_flip(result: PairedContrastResult) -> SignFlipSensitivity:
    """Enumerate all 2^8 seed-level sign assignments for sensitivity only."""

    values = np.asarray([value for _, value in result.seed_differences_pp], dtype=float)
    if values.shape != (V7_PLANNED_SEED_BLOCKS,) or not np.isfinite(values).all():
        raise V7StatisticsError(
            f"Sign-flip sensitivity for {result.name} requires exactly "
            f"{V7_PLANNED_SEED_BLOCKS} finite differences"
        )
    signs = np.asarray(
        tuple(product((-1.0, 1.0), repeat=V7_PLANNED_SEED_BLOCKS)), dtype=float
    )
    statistics = (signs * values).mean(axis=1)
    observed = float(values.mean())
    extreme = int(np.count_nonzero(statistics >= observed))
    if len(statistics) != V7_SIGN_FLIP_ASSIGNMENTS:
        raise AssertionError(
            f"The frozen V7 sign-flip space must contain {V7_SIGN_FLIP_ASSIGNMENTS} assignments"
        )
    return SignFlipSensitivity(
        observed_mean_pp=observed,
        assignments=len(statistics),
        extreme_assignments=extreme,
        p_value_one_sided_greater=float(extreme / len(statistics)),
    )


def complete_block_bootstrap(result: PairedContrastResult) -> BootstrapSensitivity:
    """Run the frozen 10,000-draw PCG64 complete-block percentile bootstrap."""

    values = np.asarray([value for _, value in result.seed_differences_pp], dtype=float)
    if values.shape != (V7_PLANNED_SEED_BLOCKS,) or not np.isfinite(values).all():
        raise V7StatisticsError(
            f"Bootstrap sensitivity for {result.name} requires exactly "
            f"{V7_PLANNED_SEED_BLOCKS} finite differences"
        )
    specs = (
        V7_REPLICATION_CONTRAST,
        *V7_HOLM_CONTRASTS,
        *V7_SECONDARY_CONTRASTS,
    )
    expected_by_name = {spec.name: spec for spec in specs}
    if result.name not in expected_by_name or result.name not in V7_BOOTSTRAP_SEEDS:
        raise ValueError(f"Unknown V7 bootstrap contrast: {result.name}")
    _require_contrast_identity(result, expected_by_name[result.name])
    random_seed = int(V7_BOOTSTRAP_SEEDS[result.name])
    rng = np.random.Generator(np.random.PCG64(random_seed))
    indices = rng.integers(
        0,
        V7_PLANNED_SEED_BLOCKS,
        size=(V7_BOOTSTRAP_RESAMPLES, V7_PLANNED_SEED_BLOCKS),
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
        n_resamples=V7_BOOTSTRAP_RESAMPLES,
        random_seed=random_seed,
        n_complete_blocks=len(values),
    )


def contrast_sensitivity(result: PairedContrastResult) -> ContrastSensitivity:
    """Bind both frozen sensitivity analyses to one named direct contrast."""

    return ContrastSensitivity(
        name=result.name,
        sign_flip=exhaustive_sign_flip(result),
        bootstrap=complete_block_bootstrap(result),
    )


def classify_study(
    *,
    replication_passed: bool,
    holm_family: Sequence[PairedContrastResult],
    route_decision: str,
    alpha: float = V7_ALPHA,
) -> StudyConclusion:
    """Apply V7 wording gates; secondary evidence can never rescue the main family."""

    alpha = _validate_frozen_alpha(alpha)
    if route_decision not in {
        "route_positive_evidence",
        "route_practical_equivalence",
        "route_contribution_inconclusive",
    }:
        raise V7StatisticsError(f"Unknown V7 route decision: {route_decision}")
    if len(holm_family) != 2 or any(item.superiority_p_holm is None for item in holm_family):
        raise V7StatisticsError("The complete two-member Holm family is required")
    for result, spec in zip(holm_family, V7_HOLM_CONTRASTS):
        _require_contrast_identity(result, spec)
        adjusted = float(result.superiority_p_holm)
        if not np.isfinite(adjusted) or not 0.0 <= adjusted <= 1.0:
            raise V7StatisticsError(f"Invalid Holm-adjusted p-value for {result.name}")
    rejected = tuple(
        item.name
        for item in holm_family
        if item.superiority_p_holm is not None and item.superiority_p_holm < alpha
    )
    if not replication_passed:
        decision = "replication_not_confirmed_mechanism_claim_blocked"
        claim = "The independent M4-M0 replication did not pass; mechanism wording is blocked."
    elif len(rejected) == 2:
        decision = "mechanism_supported"
        claim = "Both direct Holm contrasts and the independent M4-M0 replication passed."
    elif len(rejected) == 1:
        decision = "partial_mechanism_support"
        claim = "Exactly one direct Holm contrast and the independent M4-M0 replication passed."
    else:
        decision = "mechanism_inconclusive"
        claim = "Neither direct Holm contrast passed; secondary results cannot rescue the family."
    return StudyConclusion(decision, claim, replication_passed, rejected, route_decision)


def analyze_v7_mechanism_accuracy(
    frame: pd.DataFrame,
    *,
    alpha: float = V7_ALPHA,
    expected_seeds: Sequence[int] = V7_FORMAL_SEEDS,
) -> V7MechanismAnalysis:
    """Run the exact frozen V7 replication, Holm, secondary, and route branches."""

    alpha = _validate_frozen_alpha(alpha)
    normalized = validate_v7_accuracy_blocks(frame, expected_seeds=expected_seeds)
    replication = paired_contrast(normalized, V7_REPLICATION_CONTRAST)
    replication_passed = replication.superiority_p_raw < alpha and replication.estimate_pp > 0.0

    holm_raw = tuple(paired_contrast(normalized, spec) for spec in V7_HOLM_CONTRASTS)
    adjusted_p = holm_adjust([item.superiority_p_raw for item in holm_raw])
    holm_family = tuple(
        replace(item, superiority_p_holm=float(value))
        for item, value in zip(holm_raw, adjusted_p)
    )
    secondary = tuple(paired_contrast(normalized, spec) for spec in V7_SECONDARY_CONTRASTS)
    route = classify_route_contribution(secondary[0], alpha=alpha, sesoi_pp=V7_SESOI_PP)
    sensitivities = tuple(
        contrast_sensitivity(result)
        for result in (replication, *holm_family, *secondary)
    )
    conclusion = classify_study(
        replication_passed=replication_passed,
        holm_family=holm_family,
        route_decision=route.decision,
        alpha=alpha,
    )
    return V7MechanismAnalysis(
        alpha=alpha,
        sesoi_pp=V7_SESOI_PP,
        n_seed_blocks=int(normalized["seed"].nunique()),
        conditions=V7_CONDITIONS,
        replication=replication,
        replication_passed=replication_passed,
        holm_family=holm_family,
        secondary_estimates=secondary,
        route_conclusion=route,
        sensitivity_analyses=sensitivities,
        study_conclusion=conclusion,
    )


def paired_t_standardized_mdes(
    *,
    n_pairs: int,
    alpha: float,
    power: float,
) -> float:
    """Return standardized MDES for a one-sided paired t test by exact NCT power."""

    alpha = _validate_alpha(alpha)
    if int(n_pairs) != n_pairs or n_pairs < 3:
        raise ValueError("n_pairs must be an integer of at least three")
    if not 0.0 < float(power) < 1.0:
        raise ValueError("power must be in (0, 1)")
    df = int(n_pairs) - 1
    critical = float(stats.t.ppf(1.0 - alpha, df))

    def achieved(standardized_effect: float) -> float:
        noncentrality = standardized_effect * sqrt(n_pairs)
        return float(stats.nct.sf(critical, df, noncentrality))

    upper = 1.0
    while achieved(upper) < power:
        upper *= 2.0
        if upper > 1_000.0:
            raise RuntimeError("Could not bracket paired-t MDES")
    return float(optimize.brentq(lambda value: achieved(value) - power, 0.0, upper))


def mdes_sensitivity(
    *,
    paired_sd_pp: float,
    n_pairs: int = V7_PLANNED_SEED_BLOCKS,
    family_size: int = len(V7_HOLM_CONTRASTS),
    alpha: float = V7_ALPHA,
    power: float = 0.80,
) -> MDESSensitivity:
    """Conservative prospective sensitivity at Holm's first local threshold."""

    alpha = _validate_alpha(alpha)
    if int(family_size) != family_size or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    if paired_sd_pp <= 0.0 or not np.isfinite(paired_sd_pp):
        raise ValueError("paired_sd_pp must be finite and positive")
    local_alpha = alpha / family_size
    standardized = paired_t_standardized_mdes(
        n_pairs=n_pairs,
        alpha=local_alpha,
        power=power,
    )
    sign_flip_min = 2.0 ** (-int(n_pairs))
    return MDESSensitivity(
        n_pairs=int(n_pairs),
        paired_sd_pp=float(paired_sd_pp),
        alpha=alpha,
        power=float(power),
        family_size=int(family_size),
        conservative_local_alpha=local_alpha,
        standardized_mdes=standardized,
        detectable_difference_pp=standardized * paired_sd_pp,
        one_sided_sign_flip_min_p=sign_flip_min,
        conservative_holm_sign_flip_floor=min(1.0, family_size * sign_flip_min),
    )


def mdes_grid(
    paired_sds_pp: Sequence[float] = (0.3844867, 0.50, 0.75, 1.00),
) -> tuple[MDESSensitivity, ...]:
    """Return the frozen V7 reference and variance-sensitivity MDES values."""

    return tuple(mdes_sensitivity(paired_sd_pp=float(value)) for value in paired_sds_pp)
