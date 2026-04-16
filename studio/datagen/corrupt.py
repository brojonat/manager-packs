"""Corruption module — layer realistic data quality problems onto any clean dataset.

Each corruption is a pure function: (df, rng, **params) -> (df, manifest_entry).
Corruptions are composable, deterministic given seed, and fully recorded in the
sidecar JSON so downstream models can validate against known corruptions.
"""

from typing import Any

import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    """Return columns with numeric dtype (int or float)."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _choose_columns(
    df: pd.DataFrame,
    columns: list[str] | None,
    exclude: list[str] | None = None,
    numeric_only: bool = False,
) -> list[str]:
    """Resolve target columns for a corruption."""
    if columns:
        return columns
    candidates = _numeric_cols(df) if numeric_only else list(df.columns)
    if exclude:
        candidates = [c for c in candidates if c not in exclude]
    return candidates


# ── MCAR missing ──────────────────────────────────────────────────


def mcar_missing(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float = 0.1,
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Set values to NaN uniformly at random (Missing Completely At Random)."""
    df = df.copy()
    targets = _choose_columns(df, columns, exclude)
    n_injected = 0
    affected = {}

    for col in targets:
        mask = rng.random(len(df)) < rate
        n_col = int(mask.sum())
        if n_col > 0:
            df.loc[mask, col] = np.nan
            n_injected += n_col
            affected[col] = n_col

    manifest = {
        "type": "mcar_missing",
        "rate": rate,
        "columns": targets,
        "n_injected": n_injected,
        "per_column": affected,
    }
    return df, manifest


# ── MAR missing ───────────────────────────────────────────────────


def mar_missing(
    df: pd.DataFrame,
    rng: np.random.Generator,
    driver_column: str,
    target_column: str,
    rate: float = 0.3,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Missingness in target_column depends on driver_column value.

    Higher values of driver_column → higher probability of target_column
    being NaN. Models the pattern: "income is missing more often for
    high earners."
    """
    df = df.copy()
    driver = df[driver_column]
    # Probability of missing scales linearly from 0 to 2*rate across the
    # range of the driver, so the average missing rate ≈ rate
    driver_norm = (driver - driver.min()) / (driver.max() - driver.min() + 1e-10)
    probs = 2 * rate * driver_norm
    mask = rng.random(len(df)) < probs.to_numpy()
    n_injected = int(mask.sum())
    df.loc[mask, target_column] = np.nan

    manifest = {
        "type": "mar_missing",
        "driver_column": driver_column,
        "target_column": target_column,
        "rate": rate,
        "n_injected": n_injected,
        "mechanism": f"P(missing) proportional to {driver_column}",
    }
    return df, manifest


# ── MNAR missing ──────────────────────────────────────────────────


def mnar_missing(
    df: pd.DataFrame,
    rng: np.random.Generator,
    column: str,
    quantile: float = 0.9,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Censor values above a quantile (Missing Not At Random).

    Models the pattern: "measurements above the sensor's max range are
    recorded as NaN."
    """
    df = df.copy()
    threshold = float(df[column].quantile(quantile))
    mask = df[column] > threshold
    n_injected = int(mask.sum())
    df.loc[mask, column] = np.nan

    manifest = {
        "type": "mnar_missing",
        "column": column,
        "quantile": quantile,
        "threshold": threshold,
        "n_censored": n_injected,
        "mechanism": f"values above {quantile:.0%} quantile ({threshold:.4f}) set to NaN",
    }
    return df, manifest


# ── Label noise ───────────────────────────────────────────────────


def label_noise(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_column: str = "target",
    rate: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Flip target labels with probability `rate`.

    Works for integer/categorical targets. For binary: flips 0↔1.
    For multiclass: replaces with a random different label.
    """
    df = df.copy()
    labels = df[target_column].to_numpy().copy()
    unique_labels = np.unique(labels[~pd.isna(labels)])
    flip_mask = rng.random(len(df)) < rate
    n_flipped = int(flip_mask.sum())

    if len(unique_labels) == 2:
        # Binary: simple flip
        label_map = {unique_labels[0]: unique_labels[1], unique_labels[1]: unique_labels[0]}
        for idx in np.where(flip_mask)[0]:
            labels[idx] = label_map.get(labels[idx], labels[idx])
    else:
        # Multiclass: pick a random different label
        for idx in np.where(flip_mask)[0]:
            candidates = unique_labels[unique_labels != labels[idx]]
            if len(candidates) > 0:
                labels[idx] = rng.choice(candidates)

    df[target_column] = labels

    manifest = {
        "type": "label_noise",
        "target_column": target_column,
        "rate": rate,
        "n_flipped": n_flipped,
        "n_unique_labels": len(unique_labels),
    }
    return df, manifest


# ── Target leakage ────────────────────────────────────────────────


def target_leakage(
    df: pd.DataFrame,
    rng: np.random.Generator,
    target_column: str = "target",
    correlation: float = 0.95,
    leakage_name: str = "leaked_feature",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a synthetic feature highly correlated with the target.

    Simulates a post-hoc feature that wouldn't exist at prediction time
    (e.g., "account_balance_after_transaction" leaking into a fraud model).
    """
    df = df.copy()
    target = df[target_column].to_numpy().astype(float)
    noise = rng.normal(0, 1, size=len(df))
    # Blend target signal with noise to achieve approximate correlation
    leaked = correlation * (target - target.mean()) / (target.std() + 1e-10) + (1 - correlation) * noise
    df[leakage_name] = leaked

    actual_corr = float(np.corrcoef(target, leaked)[0, 1])

    manifest = {
        "type": "target_leakage",
        "target_column": target_column,
        "leakage_column": leakage_name,
        "intended_correlation": correlation,
        "actual_correlation": round(actual_corr, 4),
        "mechanism": f"{leakage_name} is ~{correlation:.0%} correlated with {target_column}",
    }
    return df, manifest


# ── Outlier injection ─────────────────────────────────────────────


def outlier_injection(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float = 0.02,
    scale: float = 10.0,
    columns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inject extreme values into numeric columns.

    For each selected column, `rate` fraction of values are replaced with
    values drawn from N(column_mean, scale * column_std).
    """
    df = df.copy()
    targets = _choose_columns(df, columns, exclude, numeric_only=True)
    n_injected = 0
    affected = {}

    for col in targets:
        col_data = df[col].to_numpy().astype(float)
        col_mean = float(np.nanmean(col_data))
        col_std = float(np.nanstd(col_data))
        if col_std == 0:
            continue

        mask = rng.random(len(df)) < rate
        n_col = int(mask.sum())
        if n_col > 0:
            # Cast to float first so outlier floats don't conflict with int dtypes
            df[col] = df[col].astype(float)
            outlier_values = rng.normal(col_mean, scale * col_std, size=n_col)
            df.loc[mask, col] = outlier_values
            n_injected += n_col
            affected[col] = n_col

    manifest = {
        "type": "outlier_injection",
        "rate": rate,
        "scale": scale,
        "columns": targets,
        "n_injected": n_injected,
        "per_column": affected,
    }
    return df, manifest


# ── Duplicate rows ────────────────────────────────────────────────


def duplicate_rows(
    df: pd.DataFrame,
    rng: np.random.Generator,
    rate: float = 0.05,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add exact duplicate rows sampled uniformly from the dataset."""
    n_dupes = max(1, int(len(df) * rate))
    dupe_idx = rng.choice(len(df), size=n_dupes, replace=True)
    dupes = df.iloc[dupe_idx].copy()
    df_out = pd.concat([df, dupes], ignore_index=True)

    manifest = {
        "type": "duplicate_rows",
        "rate": rate,
        "n_duplicates_added": n_dupes,
        "original_rows": len(df),
        "final_rows": len(df_out),
    }
    return df_out, manifest


# ── Orchestrator ──────────────────────────────────────────────────

# Map of corruption name → function for the CLI to dispatch to
CORRUPTIONS = {
    "mcar_missing": mcar_missing,
    "mar_missing": mar_missing,
    "mnar_missing": mnar_missing,
    "label_noise": label_noise,
    "target_leakage": target_leakage,
    "outlier_injection": outlier_injection,
    "duplicate_rows": duplicate_rows,
}
