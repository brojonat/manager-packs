"""CLI subcommand: corrupt — layer data quality problems onto any clean dataset."""

import json
from pathlib import Path

import click
import numpy as np
import pandas as pd

from datagen.corrupt import (
    duplicate_rows,
    label_noise,
    mar_missing,
    mcar_missing,
    mnar_missing,
    outlier_injection,
    target_leakage,
)


@click.command("corrupt")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output parquet path.")
@click.option("--seed", default=42, show_default=True)
# MCAR
@click.option("--missing-rate", default=0.0, show_default=True, help="MCAR missing rate (0 = off).")
@click.option("--missing-columns", default=None, help="Comma-separated columns for MCAR (default: all numeric).")
@click.option("--missing-exclude", default=None, help="Comma-separated columns to exclude from MCAR.")
# MAR
@click.option("--mar-driver", default=None, help="Column whose values drive MAR missingness.")
@click.option("--mar-target", default=None, help="Column to inject MAR missing values into.")
@click.option("--mar-rate", default=0.3, show_default=True, help="Average MAR missing rate.")
# MNAR
@click.option("--mnar-column", default=None, help="Column to censor (MNAR).")
@click.option("--mnar-quantile", default=0.9, show_default=True, help="Quantile above which values are censored.")
# Label noise
@click.option("--label-noise", "label_noise_rate", default=0.0, show_default=True, help="Label flip rate (0 = off).")
@click.option("--target-column", default="target", show_default=True, help="Name of the target/label column.")
# Leakage
@click.option("--leakage/--no-leakage", default=False, help="Add a leaked feature correlated with target.")
@click.option("--leakage-corr", default=0.95, show_default=True, help="Correlation of leaked feature with target.")
# Outliers
@click.option("--outlier-rate", default=0.0, show_default=True, help="Outlier injection rate (0 = off).")
@click.option("--outlier-scale", default=10.0, show_default=True, help="Outlier magnitude (multiples of column std).")
# Duplicates
@click.option("--duplicate-rate", default=0.0, show_default=True, help="Duplicate row rate (0 = off).")
def corrupt(
    input_path: str,
    output: str,
    seed: int,
    # MCAR
    missing_rate: float,
    missing_columns: str | None,
    missing_exclude: str | None,
    # MAR
    mar_driver: str | None,
    mar_target: str | None,
    mar_rate: float,
    # MNAR
    mnar_column: str | None,
    mnar_quantile: float,
    # Label noise
    label_noise_rate: float,
    target_column: str,
    # Leakage
    leakage: bool,
    leakage_corr: float,
    # Outliers
    outlier_rate: float,
    outlier_scale: float,
    # Duplicates
    duplicate_rate: float,
):
    """Apply data quality corruptions to an existing parquet dataset.

    Reads INPUT_PATH (parquet), applies selected corruptions, writes a
    new parquet + sidecar JSON with a corruption manifest recording
    exactly what was done. Composable with any datagen generator output.

    Example:

        datagen corrupt studio/data/binary-classification.parquet \\
            --missing-rate 0.1 --label-noise 0.05 --leakage \\
            -o studio/data/binary-classification-corrupted.parquet
    """
    rng = np.random.default_rng(seed)
    input_p = Path(input_path)
    df = pd.read_parquet(input_p)
    original_rows = len(df)
    original_cols = len(df.columns)

    # Load existing sidecar if present
    sidecar_in = input_p.with_suffix(".json")
    if sidecar_in.exists():
        existing_sidecar = json.loads(sidecar_in.read_text())
    else:
        existing_sidecar = {}

    corruptions_applied = []

    # ── Apply corruptions in a fixed, sensible order ──────────────
    # (leakage first so the leaked column can be corrupted by later steps;
    # duplicates last since they should duplicate the corrupted data)

    if leakage and target_column in df.columns:
        df, manifest = target_leakage(df, rng, target_column=target_column, correlation=leakage_corr)
        corruptions_applied.append(manifest)
        click.echo(f"  leakage: added '{manifest['leakage_column']}' (r={manifest['actual_correlation']:.3f})")

    if missing_rate > 0:
        cols = missing_columns.split(",") if missing_columns else None
        excl = missing_exclude.split(",") if missing_exclude else None
        df, manifest = mcar_missing(df, rng, rate=missing_rate, columns=cols, exclude=excl)
        corruptions_applied.append(manifest)
        click.echo(f"  MCAR missing: {manifest['n_injected']} values across {len(manifest['per_column'])} columns")

    if mar_driver and mar_target:
        df, manifest = mar_missing(df, rng, driver_column=mar_driver, target_column=mar_target, rate=mar_rate)
        corruptions_applied.append(manifest)
        click.echo(f"  MAR missing: {manifest['n_injected']} values in '{mar_target}' driven by '{mar_driver}'")

    if mnar_column:
        df, manifest = mnar_missing(df, rng, column=mnar_column, quantile=mnar_quantile)
        corruptions_applied.append(manifest)
        click.echo(f"  MNAR censoring: {manifest['n_censored']} values above {mnar_quantile:.0%} in '{mnar_column}'")

    if label_noise_rate > 0 and target_column in df.columns:
        df, manifest = label_noise(df, rng, target_column=target_column, rate=label_noise_rate)
        corruptions_applied.append(manifest)
        click.echo(f"  label noise: {manifest['n_flipped']} labels flipped ({label_noise_rate:.0%})")

    if outlier_rate > 0:
        df, manifest = outlier_injection(df, rng, rate=outlier_rate, scale=outlier_scale)
        corruptions_applied.append(manifest)
        click.echo(f"  outliers: {manifest['n_injected']} extreme values across {len(manifest['per_column'])} columns")

    if duplicate_rate > 0:
        df, manifest = duplicate_rows(df, rng, rate=duplicate_rate)
        corruptions_applied.append(manifest)
        click.echo(f"  duplicates: {manifest['n_duplicates_added']} rows added ({original_rows} -> {manifest['final_rows']})")

    if not corruptions_applied:
        click.echo("  no corruptions selected — output is identical to input")

    # ── Write output ──────────────────────────────────────────────
    out_p = Path(output)
    if out_p.suffix != ".parquet":
        out_p = out_p.with_suffix(".parquet")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)

    sidecar_out = {
        **existing_sidecar,
        "corrupted": True,
        "corruption_seed": seed,
        "corruption_source": str(input_p),
        "n_rows_before": original_rows,
        "n_rows_after": len(df),
        "n_cols_before": original_cols,
        "n_cols_after": len(df.columns),
        "corruptions": corruptions_applied,
    }
    sidecar_out_path = out_p.with_suffix(".json")
    sidecar_out_path.write_text(json.dumps(sidecar_out, indent=2, default=str) + "\n")

    click.echo(f"  wrote {out_p}  ({len(df)} rows, {len(df.columns)} cols)")
    click.echo(f"  wrote {sidecar_out_path}  ({len(corruptions_applied)} corruptions)")
