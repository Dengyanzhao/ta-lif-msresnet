"""Load seed-level outputs and generate machine-readable Tables 4--6."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .pathing import artifact_path_reference
from .statistics import (
    ACCURACY_COMPOSITION_MARGIN_PP,
    CONFIRMATORY_ALPHA,
    CONFIRMATORY_SEEDS,
    CONDITIONS,
    DEFAULT_GROUP_COLUMNS,
    INTERACTION_SESOI_PP,
    IncompleteSeedBlockError,
    PRIMARY_ACCURACY_GROUPS,
    SENSITIVITY_BOOTSTRAP_RESAMPLES,
    SENSITIVITY_BOOTSTRAP_SEED,
    SUCCESS_STATUSES,
    analyze_seed_level_interaction,
    analyze_seed_level_condition_contrast,
    fit_seed_blocked_accuracy,
    holm_adjust,
    validate_complete_seed_blocks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "accuracy": ("accuracy", "test_accuracy", "test_top1", "test_acc", "top1"),
    "selected_epoch": ("selected_epoch", "best_epoch"),
    "loss_auc": ("loss_auc", "train_loss_auc", "training_loss_auc"),
    "gradient_cv": ("gradient_cv", "grad_cv", "block_gradient_cv"),
    "jacobian_phi": ("jacobian_phi", "jacobian_moment1", "jacobian_mean"),
    "jacobian_varphi": (
        "jacobian_varphi",
        "jacobian_moment2",
        "jacobian_variance",
    ),
    "total_parameters": ("total_parameters", "parameters", "parameter_count"),
    "threshold_parameters": (
        "threshold_parameters",
        "talif_parameters",
        "ta_lif_parameters",
    ),
    "peak_training_memory_bytes": (
        "peak_training_memory_bytes",
        "training_peak_allocated_bytes",
    ),
    "train_epoch_seconds": ("train_epoch_seconds", "training_time_per_epoch_s"),
    "latency_b1_ms": ("latency_b1_ms", "latency_b1_mean_ms"),
    "latency_b128_ms": ("latency_b128_ms", "latency_b128_mean_ms"),
    "peak_allocated_b1_bytes": (
        "peak_allocated_b1_bytes",
        "peak_memory_b1_bytes",
    ),
    "peak_allocated_b128_bytes": (
        "peak_allocated_b128_bytes",
        "peak_memory_b128_bytes",
    ),
    "firing_rate": ("firing_rate", "spike_rate"),
    "syops_per_sample": ("syops_per_sample", "synaptic_additions_per_sample"),
    "macs_per_sample": ("macs_per_sample", "multiply_accumulates_per_sample"),
    "threshold_accesses_per_sample": (
        "threshold_accesses_per_sample",
        "threshold_bank_accesses_per_sample",
    ),
    "count_updates_per_sample": ("count_updates_per_sample", "spike_count_updates_per_sample"),
    "energy_j_per_sample": ("energy_j_per_sample", "estimated_energy_j_per_sample"),
}

TABLE5_METRICS = ("loss_auc", "gradient_cv", "jacobian_phi", "jacobian_varphi")
TABLE6_METRICS = (
    "total_parameters",
    "threshold_parameters",
    "peak_training_memory_bytes",
    "train_epoch_seconds",
    "latency_b1_ms",
    "latency_b128_ms",
    "peak_allocated_b1_bytes",
    "peak_allocated_b128_bytes",
    "firing_rate",
    "syops_per_sample",
    "macs_per_sample",
    "threshold_accesses_per_sample",
    "count_updates_per_sample",
    "energy_j_per_sample",
)
TABLE6_TRAINING_METRICS = ("peak_training_memory_bytes", "train_epoch_seconds")
TABLE6_BENCHMARK_METRICS = tuple(
    metric for metric in TABLE6_METRICS if metric not in TABLE6_TRAINING_METRICS
)
EXPECTED_SEED_BLOCKS = 5
TABLE5_DIAGNOSTIC_COLUMNS = (
    "diagnostic_id",
    "diagnostic_batch_sha256",
    "diagnostic_protocol_hash",
    "gradient_cv_method",
    "jacobian_method",
    "jacobian_probes",
    "jacobian_probe_seed",
    "jacobian_time_index",
)
EFFICIENCY_MARGIN_METRICS: Mapping[str, str] = {
    "latency_b1_percent": "latency_b1_ms",
    "latency_b128_percent": "latency_b128_ms",
    "peak_memory_percent": "peak_training_memory_bytes",
    "modeled_energy_percent": "energy_j_per_sample",
}
TRAINING_ENVIRONMENT_COLUMNS: tuple[str, ...] = (
    "training_environment_identity",
    "training_environment_sha256",
)
EFFICIENCY_TRAINING_METRICS = frozenset({"peak_training_memory_bytes"})
BENCHMARK_HOMOGENEITY_COLUMNS: tuple[str, ...] = (
    "protocol_hash",
    "hardware",
    "benchmark_device_identity",
    "precision",
    "input_file_sha256",
    "input_batch_sha256",
    "warmup_iterations",
    "timed_iterations",
    "software",
    "energy_constants_sha256",
)


def _read_structured_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return pd.json_normalize(payload)
        return pd.json_normalize([payload])
    raise ValueError(f"Unsupported structured file: {path}")


def _discover_files(path: str | Path, preferred_names: Sequence[str]) -> list[Path]:
    location = Path(path)
    if location.is_file():
        return [location]
    if not location.exists():
        raise FileNotFoundError(location)
    for name in preferred_names:
        direct = location / name
        if direct.is_file():
            return [direct]
    discovered: list[Path] = []
    for name in preferred_names:
        discovered.extend(location.rglob(name))
    return sorted(set(discovered))


def _load_many(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_structured_file(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["source_file"] = artifact_path_reference(path, PROJECT_ROOT)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_seed_metrics(path: str | Path) -> pd.DataFrame:
    """Load a consolidated seed file or recursively discover per-run files."""

    files = _discover_files(
        path,
        ("seed_metrics.csv", "seed_metrics.jsonl", "seed_metrics.json", "metrics.csv"),
    )
    if not files:
        raise FileNotFoundError(f"No seed metrics found under {Path(path)}")
    return standardize_columns(_load_many(files))


def load_run_manifest(path: str | Path) -> pd.DataFrame:
    """Load a consolidated manifest or recursively discover run manifests."""

    files = _discover_files(
        path,
        ("run_manifest.csv", "run_manifest.jsonl", "run_manifest.json"),
    )
    if not files:
        return pd.DataFrame()
    return standardize_columns(_load_many(files))


def load_benchmark_results(path: str | Path) -> pd.DataFrame:
    files = _discover_files(
        path,
        ("benchmark_results.csv", "benchmarks.csv", "benchmark.jsonl", "benchmark.json"),
    )
    if not files:
        return pd.DataFrame()
    return standardize_columns(_load_many(files))


def _coalesce_aliases(frame: pd.DataFrame, canonical: str, aliases: Sequence[str]) -> None:
    available = [column for column in aliases if column in frame.columns]
    if not available:
        return
    if canonical not in frame.columns:
        frame[canonical] = frame[available[0]]
        available = available[1:]
    for alias in available:
        if alias == canonical:
            continue
        frame[canonical] = frame[canonical].where(frame[canonical].notna(), frame[alias])


def standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize accepted aliases without manufacturing missing measurements."""

    normalized = frame.copy()
    canonical_accuracy_present = "accuracy" in normalized.columns
    for canonical, aliases in COLUMN_ALIASES.items():
        _coalesce_aliases(normalized, canonical, aliases)

    nested_aliases = {
        "dataset": ("config.data.dataset",),
        "depth": ("config.model.depth",),
        "time_steps": ("config.model.time_steps", "config.model.timesteps"),
        "condition": ("config.model.condition",),
        "topology": ("config.model.topology",),
        "neuron": ("config.model.neuron",),
        "seed": ("config.runtime.seed",),
        "experiment": ("config.experiment",),
    }
    for canonical, aliases in nested_aliases.items():
        _coalesce_aliases(normalized, canonical, (canonical, *aliases))

    # The bundled trainer records test_accuracy as a [0, 1] proportion.  The
    # manuscript model and tables use percentage points.  A user-supplied
    # canonical `accuracy` column is left untouched and must already be in %.
    if (
        not canonical_accuracy_present
        and "test_accuracy" in normalized.columns
        and "accuracy" in normalized.columns
    ):
        finite = pd.to_numeric(normalized["accuracy"], errors="coerce").dropna()
        if not finite.empty and finite.between(0.0, 1.0).all():
            normalized["accuracy"] = pd.to_numeric(normalized["accuracy"], errors="coerce") * 100.0

    if "condition" in normalized.columns:
        normalized["condition"] = normalized["condition"].astype(str).str.strip().str.upper()
    for column in (
        "depth",
        "time_steps",
        "seed",
        "selected_epoch",
        "jacobian_probes",
        "jacobian_probe_seed",
        "jacobian_time_index",
    ):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if "converged" in normalized.columns:
        normalized["converged"] = pd.to_numeric(normalized["converged"], errors="coerce")
    numeric_columns = {"accuracy", *TABLE5_METRICS, *TABLE6_METRICS}
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _truthy(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna("").astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "primary"}
    )


def select_primary_accuracy_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the two frozen E1 dataset/depth/time-step groups."""

    metrics = standardize_columns(frame)
    required = ["dataset", "depth", "time_steps"]
    missing = [column for column in required if column not in metrics.columns]
    if missing:
        raise ValueError(f"Cannot select primary comparisons; missing columns {missing}")
    candidates = metrics.copy()
    if "experiment" in candidates.columns:
        experiment = candidates["experiment"].fillna("").astype(str).str.strip().str.upper()
        candidates = candidates.loc[experiment == "E1"].copy()
    canonical_group = pd.Series(
        list(
            zip(
                candidates["dataset"].map(_dataset_key),
                pd.to_numeric(candidates["depth"], errors="coerce"),
                pd.to_numeric(candidates["time_steps"], errors="coerce"),
            )
        ),
        index=candidates.index,
    )
    expected = set(PRIMARY_ACCURACY_GROUPS)
    all_observed = {
        (_dataset_key(row.dataset), int(row.depth), int(row.time_steps))
        for row in candidates[["dataset", "depth", "time_steps"]].itertuples(index=False)
    }
    if all_observed != expected:
        missing_groups = sorted(expected - all_observed)
        extra_groups = sorted(all_observed - expected)
        raise IncompleteSeedBlockError(
            "E1 groups do not match the two frozen primary groups: "
            f"missing={missing_groups or 'none'}, extra={extra_groups or 'none'}"
        )
    selected = candidates.loc[canonical_group.isin(expected)].copy()
    observed = {
        (_dataset_key(row.dataset), int(row.depth), int(row.time_steps))
        for row in selected[["dataset", "depth", "time_steps"]].itertuples(index=False)
    }
    if observed != expected:
        missing_groups = sorted(expected - observed)
        extra_groups = sorted(observed - expected)
        raise IncompleteSeedBlockError(
            "Confirmatory family does not match the two frozen primary groups: "
            f"missing={missing_groups or 'none'}, extra={extra_groups or 'none'}"
        )
    return selected


def _effect_record(prefix: str, effect: Any) -> dict[str, float]:
    return {
        f"{prefix}_estimate": effect.estimate,
        f"{prefix}_se": effect.standard_error,
        f"{prefix}_ci_low": effect.ci_low,
        f"{prefix}_ci_high": effect.ci_high,
        f"{prefix}_p_raw": effect.p_value,
    }


def build_table4(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Generate the frozen two-test confirmatory interaction family."""

    primary = select_primary_accuracy_rows(metrics)
    primary = validate_complete_seed_blocks(
        primary,
        expected_seeds=CONFIRMATORY_SEEDS,
    )

    records: list[dict[str, Any]] = []
    for dataset_key, depth, time_steps in PRIMARY_ACCURACY_GROUPS:
        group = primary.loc[
            (primary["dataset"].map(_dataset_key) == dataset_key)
            & (primary["depth"] == depth)
            & (primary["time_steps"] == time_steps)
        ].copy()
        interaction = analyze_seed_level_interaction(group)
        c4_minus_c3 = analyze_seed_level_condition_contrast(group, "C4", "C3")
        c4_minus_c2 = analyze_seed_level_condition_contrast(group, "C4", "C2")
        descriptive = fit_seed_blocked_accuracy(
            group,
            bootstrap_resamples=SENSITIVITY_BOOTSTRAP_RESAMPLES,
            bootstrap_seed=SENSITIVITY_BOOTSTRAP_SEED,
            min_seed_blocks=len(CONFIRMATORY_SEEDS),
        )
        record: dict[str, Any] = {
            **interaction.group,
            "metric": "test top-1 accuracy (%)",
            "n_seed_blocks": len(interaction.seed_contrasts),
            "confirmatory_unit": "seed-level DID: (C4-C3)-(C2-C1)",
            "confirmatory_test": "one-sample t; H0 mean DID <= 0.50 pp",
            "confirmatory_alpha_familywise": CONFIRMATORY_ALPHA,
            "confirmatory_ci": "two-sided 95% t interval on seed-level DID",
            "frozen_seeds": ";".join(str(seed) for seed in CONFIRMATORY_SEEDS),
            "seed_interaction_did_pp": json.dumps(
                {str(seed): value for seed, value in interaction.seed_contrasts},
                sort_keys=True,
                separators=(",", ":"),
            ),
            "legacy_main_effect_model_formula": descriptive.formula,
            "effect_coding": "LIF/SpikingResNet=-0.5; TA-LIF/MS-ResNet=+0.5",
            "main_effects_inference_role": "secondary_unadjusted",
        }
        for cell in descriptive.cell_summary.to_dict("records"):
            condition = str(cell["condition"]).lower()
            record[f"{condition}_mean"] = cell["mean"]
            record[f"{condition}_sd"] = cell["sd"]
            record[f"{condition}_n"] = int(cell["n"])
            record[f"{condition}_mean_sd"] = f"{cell['mean']:.3f} +/- {cell['sd']:.3f}"
        record.update(_effect_record("neuron_main", descriptive.neuron_effect))
        record.update(_effect_record("topology_main", descriptive.topology_effect))
        record.update(
            {
                "interaction_estimate": interaction.estimate,
                "interaction_sd": interaction.standard_deviation,
                "interaction_se": interaction.standard_error,
                "interaction_df": interaction.degrees_of_freedom,
                "interaction_ci_low": interaction.ci_low,
                "interaction_ci_high": interaction.ci_high,
                "interaction_ci90_low": interaction.ci90_low,
                "interaction_ci90_high": interaction.ci90_high,
                "interaction_sesoi_pp": interaction.sesoi_pp,
                "interaction_t_sesoi": interaction.t_statistic_sesoi,
                "interaction_p_sesoi_raw": interaction.p_value_one_sided_sesoi,
                "interaction_p_raw": interaction.p_value_one_sided_sesoi,
                "interaction_bootstrap_estimate": interaction.bootstrap_sensitivity.estimate,
                "interaction_bootstrap_ci_low": interaction.bootstrap_sensitivity.ci_low,
                "interaction_bootstrap_ci_high": interaction.bootstrap_sensitivity.ci_high,
                "interaction_bootstrap_p_descriptive": (
                    interaction.bootstrap_sensitivity.p_value_two_sided
                ),
                "bootstrap_resamples": interaction.bootstrap_sensitivity.n_resamples,
                "bootstrap_seed": interaction.bootstrap_sensitivity.random_seed,
                "bootstrap_role": "sensitivity_only_not_confirmatory",
                "legacy_ols_interaction_estimate": descriptive.interaction_effect.estimate,
                "legacy_ols_residual_shapiro_w": descriptive.residual_shapiro_w,
                "legacy_ols_residual_shapiro_p": descriptive.residual_shapiro_p,
            }
        )
        for prefix, contrast in (
            ("c4_minus_c3", c4_minus_c3),
            ("c4_minus_c2", c4_minus_c2),
        ):
            record.update(
                {
                    f"{prefix}_seed_pp": json.dumps(
                        {str(seed): value for seed, value in contrast.seed_contrasts},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    f"{prefix}_estimate": contrast.estimate,
                    f"{prefix}_sd": contrast.standard_deviation,
                    f"{prefix}_se": contrast.standard_error,
                    f"{prefix}_df": contrast.degrees_of_freedom,
                    f"{prefix}_ci95_low": contrast.ci95_low,
                    f"{prefix}_ci95_high": contrast.ci95_high,
                    f"{prefix}_ci90_low": contrast.ci90_low,
                    f"{prefix}_ci90_high": contrast.ci90_high,
                    f"{prefix}_noninferiority_margin_pp": (
                        contrast.noninferiority_margin_pp
                    ),
                    f"{prefix}_noninferiority_lower_bound_95": (
                        contrast.noninferiority_lower_bound_95
                    ),
                }
            )
        records.append(record)

    table = pd.DataFrame.from_records(records)
    if len(table) != len(PRIMARY_ACCURACY_GROUPS):
        raise IncompleteSeedBlockError("Confirmatory interaction family is incomplete")
    table["interaction_p_sesoi_holm"] = holm_adjust(
        table["interaction_p_sesoi_raw"].to_numpy(dtype=float)
    )
    table["interaction_p_holm"] = table["interaction_p_sesoi_holm"]
    table["interaction_reject_holm_0_05"] = (
        table["interaction_p_sesoi_holm"] < CONFIRMATORY_ALPHA
    )
    table["interaction_practical_threshold_pp"] = INTERACTION_SESOI_PP
    table["interaction_exceeds_practical_threshold"] = (
        table["interaction_estimate"] > INTERACTION_SESOI_PP
    )
    table["interaction_ci_exceeds_practical_threshold"] = (
        table["interaction_ci_low"] > INTERACTION_SESOI_PP
    )
    table["synergy_supported"] = table["interaction_reject_holm_0_05"]
    table["c4_minus_c3_noninferior_0_50pp"] = (
        table["c4_minus_c3_noninferiority_lower_bound_95"]
        > -ACCURACY_COMPOSITION_MARGIN_PP
    )
    table["c4_minus_c2_noninferior_0_50pp"] = (
        table["c4_minus_c2_noninferiority_lower_bound_95"]
        > -ACCURACY_COMPOSITION_MARGIN_PP
    )
    table["interaction_equivalent_within_0_50pp"] = (
        (table["interaction_ci90_low"] > -ACCURACY_COMPOSITION_MARGIN_PP)
        & (table["interaction_ci90_high"] < ACCURACY_COMPOSITION_MARGIN_PP)
    )
    table["complementary_supported"] = (
        ~table["synergy_supported"]
        & (table["c4_minus_c3_ci95_low"] > 0.0)
        & (table["c4_minus_c2_ci95_low"] > 0.0)
    )
    table["composable_additive_supported"] = (
        table["interaction_equivalent_within_0_50pp"]
        & table["c4_minus_c3_noninferior_0_50pp"]
        & table["c4_minus_c2_noninferior_0_50pp"]
    )
    table["inconclusive"] = ~(
        table["synergy_supported"]
        | table["complementary_supported"]
        | table["composable_additive_supported"]
    )
    table["interaction_interpretation"] = np.where(
        table["synergy_supported"],
        "synergy_supported_for_this_dataset_depth_only",
        np.where(
            table["complementary_supported"] & table["composable_additive_supported"],
            "complementary_and_composable_additive_supported",
            np.where(
                table["complementary_supported"],
                "complementary_supported_without_synergy",
                np.where(
                    table["composable_additive_supported"],
                    "composable_additive_supported_without_synergy",
                    "inconclusive_about_composition",
                ),
            ),
        ),
    )
    return table


def _dataset_key(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def _primary_group_series(frame: pd.DataFrame) -> pd.Series:
    required = {"dataset", "depth", "time_steps"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise IncompleteSeedBlockError(
            f"Primary-group selection is missing columns: {missing}"
        )
    return pd.Series(
        list(
            zip(
                frame["dataset"].map(_dataset_key),
                pd.to_numeric(frame["depth"], errors="coerce"),
                pd.to_numeric(frame["time_steps"], errors="coerce"),
            )
        ),
        index=frame.index,
    )


def _select_primary_group_subset(
    frame: pd.DataFrame,
    *,
    label: str,
    require_all: bool,
) -> pd.DataFrame:
    """Select only retained E1 groups, rejecting stale or foreign group rows."""

    selected = standardize_columns(frame)
    if selected.empty:
        return selected
    if "experiment" in selected.columns:
        experiment = selected["experiment"].fillna("").astype(str).str.strip().str.upper()
        invalid = ~experiment.isin(("", "E1"))
        if invalid.any():
            values = sorted(set(experiment.loc[invalid]))
            raise IncompleteSeedBlockError(f"{label} contains non-E1 rows: {values}")
    group_keys = _primary_group_series(selected)
    expected = set(PRIMARY_ACCURACY_GROUPS)
    observed = set(group_keys)
    extra = sorted(observed - expected)
    missing = sorted(expected - observed)
    if extra or (require_all and missing):
        raise IncompleteSeedBlockError(
            f"{label} groups differ from the frozen primary groups: "
            f"missing={missing or 'none'}, extra={extra or 'none'}"
        )
    return selected.loc[group_keys.isin(expected)].copy()


def _select_table5_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    return _select_primary_group_subset(
        metrics,
        label="Table 5 diagnostics",
        require_all=True,
    )


def _failure_counts(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [*DEFAULT_GROUP_COLUMNS, "condition"]
    if frame.empty or any(column not in frame.columns for column in keys):
        return pd.DataFrame(columns=[*keys, "failed_runs"])
    failed = pd.Series(False, index=frame.index)
    if "status" in frame.columns:
        status = frame["status"].fillna("").astype(str).str.strip().str.lower()
        failed |= ~status.isin(SUCCESS_STATUSES)
    if "failed" in frame.columns:
        failed |= _truthy(frame["failed"])
    relevant = frame.loc[failed].copy()
    if relevant.empty:
        groups = frame[keys].drop_duplicates()
        groups["failed_runs"] = 0
        return groups
    count_column = "run_id" if "run_id" in relevant.columns else "condition"
    counts = (
        relevant.groupby(keys, dropna=False)[count_column]
        .nunique()
        .rename("failed_runs")
        .reset_index()
    )
    return counts


def _single_nonempty_value(frame: pd.DataFrame, column: str) -> Any:
    if column not in frame.columns:
        raise IncompleteSeedBlockError(f"Table 5 diagnostics are missing column {column!r}")
    values = frame[column]
    nonempty = values.notna() & values.astype(str).str.strip().ne("")
    if not nonempty.all():
        rows = frame.index[~nonempty].tolist()[:10]
        raise IncompleteSeedBlockError(
            f"Table 5 diagnostics contain empty {column!r} values at rows {rows}"
        )
    unique = pd.unique(values)
    if len(unique) != 1:
        raise IncompleteSeedBlockError(
            f"Table 5 diagnostics mix {column!r} values: {list(unique)[:10]}"
        )
    return unique[0]


def _validate_table5_diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        raise IncompleteSeedBlockError("No retained primary rows are available for Table 5")
    normalized = validate_complete_seed_blocks(
        frame,
        expected_seeds=CONFIRMATORY_SEEDS,
    )
    expected_rows = (
        len(PRIMARY_ACCURACY_GROUPS) * len(CONDITIONS) * len(CONFIRMATORY_SEEDS)
    )
    if len(normalized) != expected_rows:
        raise IncompleteSeedBlockError(
            "Table 5 requires both retained groups with exactly C1-C4 x 5 seed rows"
        )

    required_numeric = (*TABLE5_METRICS, "converged")
    for column in required_numeric:
        if column not in normalized.columns:
            raise IncompleteSeedBlockError(f"Table 5 is missing required metric {column!r}")
        values = pd.to_numeric(normalized[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            rows = normalized.index[~np.isfinite(values.to_numpy(dtype=float))].tolist()[:10]
            raise IncompleteSeedBlockError(
                f"Table 5 has non-finite {column!r} values at rows {rows}"
            )
        normalized[column] = values
    if not normalized["converged"].isin((0, 1)).all():
        raise IncompleteSeedBlockError("Table 5 converged values must be coded as 0 or 1")

    diagnostic_ids = normalized.get("diagnostic_id", pd.Series(index=normalized.index, dtype=object))
    if diagnostic_ids.isna().any() or diagnostic_ids.astype(str).str.strip().eq("").any():
        raise IncompleteSeedBlockError("Table 5 requires a diagnostic_id for every seed-condition row")
    if diagnostic_ids.duplicated().any():
        raise IncompleteSeedBlockError("Table 5 diagnostic_id values must be unique per checkpoint")

    audit = {
        column: _single_nonempty_value(normalized, column)
        for column in TABLE5_DIAGNOSTIC_COLUMNS
        if column not in {"diagnostic_id", "diagnostic_batch_sha256"}
    }
    for column in ("jacobian_probes", "jacobian_probe_seed", "jacobian_time_index"):
        parsed = pd.to_numeric(pd.Series([audit[column]]), errors="coerce").iloc[0]
        if not np.isfinite(parsed) or float(parsed) != int(parsed):
            raise IncompleteSeedBlockError(f"Table 5 has invalid {column!r}")
        audit[column] = int(parsed)
    if audit["jacobian_probes"] < 1 or audit["jacobian_probe_seed"] < 0:
        raise IncompleteSeedBlockError("Table 5 Jacobian probes/seed are outside valid bounds")
    return normalized, audit


def build_table5(metrics: pd.DataFrame, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Generate complete diagnostics for both retained confirmatory groups."""

    selected, diagnostic_audit = _validate_table5_diagnostics(
        _select_table5_rows(standardize_columns(metrics))
    )
    columns = [
        "dataset",
        "depth",
        "time_steps",
        "condition",
        "n_seed_rows",
        *(f"{metric}_{suffix}" for metric in TABLE5_METRICS for suffix in ("mean", "sd", "n")),
        "failed_runs",
        "nonconverged_runs",
        "diagnostic_batch_sha256",
        "diagnostic_protocol_hash",
        "gradient_cv_method",
        "jacobian_method",
        "jacobian_probes",
        "jacobian_probe_seed",
        "jacobian_time_index",
        "missing_optional_metrics",
    ]
    keys = [*DEFAULT_GROUP_COLUMNS, "condition"]
    records: list[dict[str, Any]] = []
    for key, group in selected.groupby(keys, sort=True, dropna=False):
        record = {
            **dict(zip(keys, key)),
            **diagnostic_audit,
            "diagnostic_batch_sha256": _single_nonempty_value(
                group, "diagnostic_batch_sha256"
            ),
        }
        record["n_seed_rows"] = int(len(group))
        for metric in TABLE5_METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
            record[f"{metric}_n"] = int(len(values))
        convergence = pd.to_numeric(group["converged"], errors="raise")
        record["nonconverged_runs"] = int((convergence == 0).sum())
        record["missing_optional_metrics"] = ""
        records.append(record)
    table = pd.DataFrame.from_records(records)

    failure_source = (
        _select_primary_group_subset(
            manifest,
            label="Table 5 manifest",
            require_all=True,
        )
        if manifest is not None and not manifest.empty
        else selected
    )
    failures = _failure_counts(failure_source)
    table = table.merge(failures, on=keys, how="left")
    table["failed_runs"] = table["failed_runs"].fillna(0).astype(int)
    return table.reindex(columns=columns)


def _join_unique(series: pd.Series) -> str:
    values = sorted({str(value).strip() for value in series.dropna() if str(value).strip()})
    return ";".join(values)


def _summary_values(group: pd.DataFrame, metric: str) -> tuple[float, float, int]:
    values = (
        pd.to_numeric(group[metric], errors="coerce").dropna()
        if metric in group.columns
        else pd.Series(dtype=float)
    )
    return (
        float(values.mean()) if len(values) else np.nan,
        float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        int(len(values)),
    )


def _tolerance_columns() -> list[str]:
    columns = ["tolerance_baseline_condition"]
    for margin_name in EFFICIENCY_MARGIN_METRICS:
        prefix = margin_name.removesuffix("_percent")
        columns.extend(
            [
                f"{prefix}_tolerance_margin_percent",
                f"{prefix}_geometric_overhead_percent",
                f"{prefix}_paired_ci_low_percent",
                f"{prefix}_paired_ci_high_percent",
                f"{prefix}_paired_n",
                f"{prefix}_tolerance_status",
            ]
        )
    columns.append("efficiency_tolerance_status")
    return columns


def _validate_benchmark_contract(frame: pd.DataFrame) -> dict[str, Any]:
    """Fail closed if one primary-group benchmark mixes measurement settings."""

    if frame.empty:
        return {column: "" for column in BENCHMARK_HOMOGENEITY_COLUMNS}
    required = {
        "benchmark_id",
        "dataset",
        "depth",
        "condition",
        "time_steps",
        "seed",
        *BENCHMARK_HOMOGENEITY_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise IncompleteSeedBlockError(
            f"Benchmark results are missing frozen identity columns: {missing}"
        )
    benchmark_ids = frame["benchmark_id"].fillna("").astype(str).str.strip()
    if benchmark_ids.eq("").any() or benchmark_ids.duplicated().any():
        raise IncompleteSeedBlockError("Benchmark IDs must be non-empty and unique")
    duplicated_seed = frame.duplicated(
        ["dataset", "depth", "condition", "time_steps", "seed"], keep=False
    )
    if duplicated_seed.any():
        examples = frame.loc[
            duplicated_seed, ["dataset", "depth", "condition", "time_steps", "seed"]
        ].head(10)
        raise IncompleteSeedBlockError(
            "Timing repetitions must remain inside one benchmark row; duplicate seed-level "
            f"benchmarks were supplied: {examples.to_dict('records')}"
        )
    numeric_seeds = pd.to_numeric(frame["seed"], errors="coerce")
    if not np.isfinite(numeric_seeds.to_numpy(dtype=float)).all():
        raise IncompleteSeedBlockError("Benchmark seed values must be finite integers")
    observed_seeds = set(int(value) for value in numeric_seeds)
    unexpected = sorted(observed_seeds - set(CONFIRMATORY_SEEDS))
    if unexpected:
        raise IncompleteSeedBlockError(f"Benchmark rows contain non-frozen seeds: {unexpected}")

    audit: dict[str, Any] = {}
    for column in BENCHMARK_HOMOGENEITY_COLUMNS:
        values = frame[column]
        nonempty = values.notna() & values.astype(str).str.strip().ne("")
        if not nonempty.all():
            rows = frame.index[~nonempty].tolist()[:10]
            raise IncompleteSeedBlockError(
                f"Benchmark identity column {column!r} is empty at rows {rows}"
            )
        unique = pd.unique(values.astype(str).str.strip())
        if len(unique) != 1:
            raise IncompleteSeedBlockError(
                f"Benchmark results mix {column!r}: {list(unique)[:10]}"
            )
        audit[column] = values.iloc[0]
    if int(float(audit["timed_iterations"])) < 1 or int(float(audit["warmup_iterations"])) < 0:
        raise IncompleteSeedBlockError("Benchmark warm-up/timed iteration counts are invalid")
    energy = (
        pd.to_numeric(frame["energy_j_per_sample"], errors="coerce")
        if "energy_j_per_sample" in frame.columns
        else pd.Series(dtype=float)
    )
    if energy.notna().any() and str(audit["energy_constants_sha256"]) == "none":
        raise IncompleteSeedBlockError(
            "Modeled energy values are present without a frozen energy-constant hash"
        )
    return audit


def _validate_benchmark_collection(
    frame: pd.DataFrame,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    """Validate group-specific inputs and study-wide benchmark settings."""

    if frame.empty:
        return {}
    if "benchmark_id" not in frame.columns:
        raise IncompleteSeedBlockError("Benchmark results are missing benchmark_id")
    benchmark_ids = frame["benchmark_id"].fillna("").astype(str).str.strip()
    if benchmark_ids.eq("").any() or benchmark_ids.duplicated().any():
        raise IncompleteSeedBlockError(
            "Benchmark IDs must be non-empty and unique across both primary groups"
        )

    group_keys = _primary_group_series(frame)
    audits: dict[tuple[str, int, int], dict[str, Any]] = {}
    for group_key in PRIMARY_ACCURACY_GROUPS:
        group = frame.loc[group_keys == group_key].copy()
        if not group.empty:
            audits[group_key] = _validate_benchmark_contract(group)

    group_specific = {"input_file_sha256", "input_batch_sha256"}
    for column in BENCHMARK_HOMOGENEITY_COLUMNS:
        if column not in group_specific:
            _single_nonempty_value(frame, column)
    return audits


def _validate_training_environment(frame: pd.DataFrame) -> dict[str, str]:
    """Require one verifiable training environment for peak-memory comparisons."""

    missing = sorted(set(TRAINING_ENVIRONMENT_COLUMNS) - set(frame.columns))
    if missing:
        raise IncompleteSeedBlockError(
            f"Training rows are missing environment identity columns: {missing}"
        )
    identities = frame["training_environment_identity"].fillna("").astype(str).str.strip()
    digests = frame["training_environment_sha256"].fillna("").astype(str).str.strip()
    if identities.eq("").any() or digests.eq("").any():
        raise IncompleteSeedBlockError("Training environment identity/hash must be non-empty")
    for index, (encoded, recorded) in enumerate(zip(identities, digests)):
        try:
            parsed = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise IncompleteSeedBlockError(
                f"Training environment identity is invalid JSON at row {frame.index[index]}"
            ) from exc
        canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        observed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if canonical != encoded or observed != recorded:
            raise IncompleteSeedBlockError(
                f"Training environment identity/hash mismatch at row {frame.index[index]}"
            )
    unique_identities = pd.unique(identities)
    unique_digests = pd.unique(digests)
    if len(unique_identities) != 1 or len(unique_digests) != 1:
        raise IncompleteSeedBlockError("Training rows mix hardware, software, device, or precision")
    return {
        "training_environment_identity": str(unique_identities[0]),
        "training_environment_sha256": str(unique_digests[0]),
    }


def _paired_log_ratio_summary(
    frame: pd.DataFrame,
    *,
    treatment: str,
    baseline: str,
    time_steps: int,
    metric: str,
    margin_percent: float | None,
) -> dict[str, Any]:
    """Summarize same-seed treatment/baseline log ratios with a t interval."""

    result: dict[str, Any] = {
        "margin": margin_percent if margin_percent is not None else np.nan,
        "overhead": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "n": 0,
        "status": "not_assessed_missing_margin" if margin_percent is None else "not_assessed",
    }
    if margin_percent is None:
        return result
    if frame.empty or metric not in frame.columns or "seed" not in frame.columns:
        result["status"] = "not_assessed_missing_metric"
        return result
    selected = frame.loc[
        frame["condition"].isin((treatment, baseline))
        & (frame["time_steps"] == time_steps),
        ["condition", "seed", metric],
    ].copy()
    if selected.duplicated(["condition", "seed"]).any():
        raise IncompleteSeedBlockError(
            f"Duplicate seed-level {metric} measurements for {treatment}/{baseline}, T={time_steps}"
        )
    selected["seed"] = pd.to_numeric(selected["seed"], errors="coerce")
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    pivot = selected.pivot(index="seed", columns="condition", values=metric)
    expected = set(CONFIRMATORY_SEEDS)
    observed_treatment = set(
        int(value) for value in selected.loc[selected["condition"] == treatment, "seed"].dropna()
    )
    observed_baseline = set(
        int(value) for value in selected.loc[selected["condition"] == baseline, "seed"].dropna()
    )
    if observed_treatment != expected or observed_baseline != expected:
        result["status"] = "not_assessed_incomplete_seed_pairs"
        return result
    pivot = pivot.reindex(CONFIRMATORY_SEEDS)
    treatment_values = pd.to_numeric(pivot[treatment], errors="coerce").to_numpy(dtype=float)
    baseline_values = pd.to_numeric(pivot[baseline], errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(treatment_values).all()
        or not np.isfinite(baseline_values).all()
        or (treatment_values <= 0.0).any()
        or (baseline_values <= 0.0).any()
    ):
        result["status"] = "not_assessed_nonpositive_or_nonfinite_metric"
        return result
    log_ratios = np.log(treatment_values / baseline_values)
    n = len(log_ratios)
    mean_log = float(log_ratios.mean())
    sd_log = float(log_ratios.std(ddof=1))
    se_log = float(sd_log / np.sqrt(n))
    critical = float(stats.t.ppf(0.975, n - 1))
    ci_low_log = mean_log - critical * se_log
    ci_high_log = mean_log + critical * se_log
    overhead = float(np.expm1(mean_log) * 100.0)
    ci_low = float(np.expm1(ci_low_log) * 100.0)
    ci_high = float(np.expm1(ci_high_log) * 100.0)
    if not np.isfinite([overhead, ci_low, ci_high]).all():
        raise IncompleteSeedBlockError(f"Paired log-ratio inference failed for {metric}")
    if overhead > margin_percent:
        status = "exceeds_tolerance"
    elif ci_high <= margin_percent:
        status = "supported_within_tolerance"
    else:
        status = "within_tolerance_uncertain"
    return {
        "margin": float(margin_percent),
        "overhead": overhead,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "status": status,
    }


def _overall_efficiency_status(statuses: Sequence[str]) -> str:
    """Summarize metric states without hiding a known exceedance behind missing data."""

    if "exceeds_tolerance" in statuses:
        return "exceeds_tolerance"
    if "within_tolerance_uncertain" in statuses:
        return "within_tolerance_uncertain"
    if any(status.startswith("not_assessed") for status in statuses):
        return "not_assessed"
    if statuses and all(status == "supported_within_tolerance" for status in statuses):
        return "supported_within_tolerance"
    return "not_assessed"


def _apply_efficiency_tolerance(
    table: pd.DataFrame,
    training: pd.DataFrame,
    benchmark: pd.DataFrame,
    margins: Mapping[str, float] | None,
) -> pd.DataFrame:
    result = table.copy()
    if margins is not None:
        missing = sorted(set(EFFICIENCY_MARGIN_METRICS) - set(margins))
        if missing:
            raise ValueError(f"Efficiency noninferiority margins are missing: {missing}")
        parsed_margins = {name: float(margins[name]) for name in EFFICIENCY_MARGIN_METRICS}
        if any(not np.isfinite(value) or value < 0 for value in parsed_margins.values()):
            raise ValueError("Efficiency noninferiority margins must be finite and non-negative")
    else:
        parsed_margins = {}

    baseline_for = {"C1": "C1", "C2": "C1", "C3": "C3", "C4": "C3"}
    records: list[dict[str, Any]] = []
    for _, row in result.iterrows():
        record = row.to_dict()
        condition = str(row["condition"])
        dataset_key = _dataset_key(row["dataset"])
        depth = int(row["depth"])
        time_steps = int(row["time_steps"])
        training_group = training.loc[
            (training["dataset"].map(_dataset_key) == dataset_key)
            & (training["depth"] == depth)
            & (training["time_steps"] == time_steps)
        ]
        benchmark_group = (
            benchmark.loc[
                (benchmark["dataset"].map(_dataset_key) == dataset_key)
                & (benchmark["depth"] == depth)
                & (benchmark["time_steps"] == time_steps)
            ]
            if not benchmark.empty
            else benchmark
        )
        baseline_condition = baseline_for[condition]
        record["tolerance_baseline_condition"] = baseline_condition
        statuses: list[str] = []
        for margin_name, metric in EFFICIENCY_MARGIN_METRICS.items():
            prefix = margin_name.removesuffix("_percent")
            margin = parsed_margins.get(margin_name)
            if condition == baseline_condition:
                summary = {
                    "margin": margin if margin is not None else np.nan,
                    "overhead": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.0,
                    "n": len(CONFIRMATORY_SEEDS),
                    "status": "reference",
                }
            else:
                source = (
                    training_group
                    if metric in EFFICIENCY_TRAINING_METRICS
                    else benchmark_group
                )
                summary = _paired_log_ratio_summary(
                    source,
                    treatment=condition,
                    baseline=baseline_condition,
                    time_steps=time_steps,
                    metric=metric,
                    margin_percent=margin,
                )
            record[f"{prefix}_tolerance_margin_percent"] = summary["margin"]
            record[f"{prefix}_geometric_overhead_percent"] = summary["overhead"]
            record[f"{prefix}_paired_ci_low_percent"] = summary["ci_low"]
            record[f"{prefix}_paired_ci_high_percent"] = summary["ci_high"]
            record[f"{prefix}_paired_n"] = summary["n"]
            record[f"{prefix}_tolerance_status"] = summary["status"]
            if summary["status"] != "reference":
                statuses.append(str(summary["status"]))

        if condition == baseline_condition:
            record["efficiency_tolerance_status"] = "reference"
        else:
            record["efficiency_tolerance_status"] = _overall_efficiency_status(statuses)
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_table6(
    metrics: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    *,
    efficiency_margins: Mapping[str, float] | None = None,
    expected_energy_constants_sha256: str | None = None,
) -> pd.DataFrame:
    """Generate efficiency summaries for both retained confirmatory groups."""

    training_selected = _select_primary_group_subset(
        metrics,
        label="Table 6 training metrics",
        require_all=True,
    )
    training_selected = training_selected.loc[
        training_selected["condition"].isin(CONDITIONS)
    ].copy()
    training_selected = validate_complete_seed_blocks(
        training_selected,
        expected_seeds=CONFIRMATORY_SEEDS,
    )
    training_environment_audit = _validate_training_environment(training_selected)

    benchmark_rows = (
        standardize_columns(benchmark)
        if benchmark is not None and not benchmark.empty
        else pd.DataFrame()
    )
    if not benchmark_rows.empty:
        benchmark_rows = _select_primary_group_subset(
            benchmark_rows,
            label="Table 6 benchmarks",
            require_all=False,
        )
        benchmark_rows = benchmark_rows.loc[
            benchmark_rows["condition"].isin(CONDITIONS)
        ].copy()
    benchmark_audits = _validate_benchmark_collection(benchmark_rows)
    if (
        not benchmark_rows.empty
        and expected_energy_constants_sha256 is not None
        and _single_nonempty_value(benchmark_rows, "energy_constants_sha256")
        != expected_energy_constants_sha256
    ):
        raise IncompleteSeedBlockError(
            "Benchmark energy constants SHA-256 differs from the frozen protocol"
        )
    training_protocols = {
        str(value).strip()
        for value in training_selected["protocol_hash"].dropna()
        if str(value).strip()
    }
    if benchmark_rows.empty:
        pass
    elif (
        len(training_protocols) != 1
        or _single_nonempty_value(benchmark_rows, "protocol_hash")
        not in training_protocols
    ):
        raise IncompleteSeedBlockError(
            "Benchmark protocol_hash does not match the unique seed-metrics protocol_hash"
        )

    keys = ["dataset", "depth", "condition", "time_steps"]
    records: list[dict[str, Any]] = []
    training_group_keys = _primary_group_series(training_selected)
    benchmark_group_keys = (
        _primary_group_series(benchmark_rows)
        if not benchmark_rows.empty
        else pd.Series(dtype=object)
    )
    for group_key in PRIMARY_ACCURACY_GROUPS:
        dataset_key, depth, time_steps = group_key
        training_primary_group = training_selected.loc[
            training_group_keys == group_key
        ].copy()
        dataset_values = pd.unique(training_primary_group["dataset"])
        if len(dataset_values) != 1:
            raise IncompleteSeedBlockError(
                f"Table 6 group {group_key} has inconsistent dataset labels"
            )
        dataset = dataset_values[0]
        benchmark_primary_group = (
            benchmark_rows.loc[benchmark_group_keys == group_key].copy()
            if not benchmark_rows.empty
            else pd.DataFrame()
        )
        benchmark_audit = benchmark_audits.get(
            group_key,
            {column: "" for column in BENCHMARK_HOMOGENEITY_COLUMNS},
        )
        benchmark_identity_status = (
            "homogeneous_complete"
            if len(benchmark_primary_group)
            == len(CONDITIONS) * len(CONFIRMATORY_SEEDS)
            else (
                "homogeneous_incomplete"
                if not benchmark_primary_group.empty
                else "not_assessed_no_benchmark_rows"
            )
        )
        for condition in CONDITIONS:
            training_group = training_selected.loc[
                (training_group_keys == group_key)
                & (training_selected["condition"] == condition)
            ]
            benchmark_group = (
                benchmark_primary_group.loc[
                    benchmark_primary_group["condition"] == condition
                ]
                if not benchmark_primary_group.empty
                else pd.DataFrame()
            )
            record: dict[str, Any] = {
                "dataset": dataset,
                "depth": depth,
                "condition": condition,
                "time_steps": time_steps,
                "n_benchmarks": int(len(benchmark_group)),
                "benchmark_replication_unit": "checkpoint_seed",
                "timed_iterations_role": "within_checkpoint_technical_repeats_not_n",
                "benchmark_identity_audit_status": benchmark_identity_status,
                **training_environment_audit,
                **benchmark_audit,
            }
            for metric in TABLE6_BENCHMARK_METRICS:
                mean, sd, count = _summary_values(benchmark_group, metric)
                record[f"{metric}_mean"] = mean
                record[f"{metric}_sd"] = sd
                record[f"{metric}_n"] = count
            for metric in TABLE6_TRAINING_METRICS:
                mean, sd, count = _summary_values(training_group, metric)
                record[f"{metric}_mean"] = mean
                record[f"{metric}_sd"] = sd
                record[f"{metric}_n"] = count
            record["energy_status"] = (
                _join_unique(benchmark_group["energy_status"])
                if "energy_status" in benchmark_group.columns and not benchmark_group.empty
                else "missing_benchmark"
            )
            record["energy_model_source"] = (
                _join_unique(benchmark_group["energy_model_source"])
                if "energy_model_source" in benchmark_group.columns and not benchmark_group.empty
                else ""
            )
            missing_metrics = [
                metric
                for metric in TABLE6_METRICS
                if int(record[f"{metric}_n"]) < EXPECTED_SEED_BLOCKS
            ]
            record["missing_optional_metrics"] = ";".join(missing_metrics)
            records.append(record)

    table = _apply_efficiency_tolerance(
        pd.DataFrame.from_records(records),
        training_selected,
        benchmark_rows,
        efficiency_margins,
    )
    output_columns = [
        *keys,
        "n_benchmarks",
        "benchmark_replication_unit",
        "timed_iterations_role",
        "benchmark_identity_audit_status",
        *(f"{metric}_{suffix}" for metric in TABLE6_METRICS for suffix in ("mean", "sd", "n")),
        "energy_status",
        "energy_model_source",
        *TRAINING_ENVIRONMENT_COLUMNS,
        *BENCHMARK_HOMOGENEITY_COLUMNS,
        "missing_optional_metrics",
        *_tolerance_columns(),
    ]
    return table.sort_values(
        ["dataset", "depth", "time_steps", "condition"]
    ).reindex(columns=output_columns)


def write_analysis_tables(
    metrics: pd.DataFrame,
    output_directory: str | Path,
    *,
    manifest: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    efficiency_margins: Mapping[str, float] | None = None,
    expected_energy_constants_sha256: str | None = None,
) -> dict[str, Path]:
    """Write Tables 4--6 as CSV and return their paths."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "table4": build_table4(metrics),
        "table5": build_table5(metrics, manifest=manifest),
        "table6": build_table6(
            metrics,
            benchmark=benchmark,
            efficiency_margins=efficiency_margins,
            expected_energy_constants_sha256=expected_energy_constants_sha256,
        ),
    }
    paths = {
        "table4": output / "table4_accuracy.csv",
        "table5": output / "table5_diagnostics.csv",
        "table6": output / "table6_efficiency.csv",
    }
    for name, table in tables.items():
        table.to_csv(paths[name], index=False, lineterminator="\n")
    return paths
