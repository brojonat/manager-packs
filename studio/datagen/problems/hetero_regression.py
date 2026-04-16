"""Heteroscedastic regression — linear model with non-constant variance and outliers."""

import click
import numpy as np
import pandas as pd

from datagen.output import write_dataset


@click.command("hetero-regression")
@click.option("--n", default=500, show_default=True, help="Number of samples.")
@click.option("--intercept", default=2.0, show_default=True, help="True intercept.")
@click.option("--slope", default=1.5, show_default=True, help="True slope.")
@click.option("--base-sigma", default=0.5, show_default=True, help="Baseline noise at x=0.")
@click.option("--hetero-strength", default=1.0, show_default=True, help="How much sigma grows with x (0=constant).")
@click.option("--outlier-frac", default=0.05, show_default=True, help="Fraction of outlier points.")
@click.option("--outlier-scale", default=8.0, show_default=True, help="Scale of outlier displacement.")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def hetero_regression(
    n: int,
    intercept: float,
    slope: float,
    base_sigma: float,
    hetero_strength: float,
    outlier_frac: float,
    outlier_scale: float,
    seed: int,
    output: str | None,
):
    """Generate linear regression data with heteroscedastic noise and outliers."""
    rng = np.random.default_rng(seed)

    x = np.sort(rng.uniform(0, 10, size=n))

    # Heteroscedastic noise: sigma grows linearly with x
    sigma_x = base_sigma + hetero_strength * 0.1 * x
    noise = rng.normal(0, sigma_x)
    y = intercept + slope * x + noise

    # Inject outliers
    n_outliers = int(n * outlier_frac)
    is_outlier = np.zeros(n, dtype=bool)
    if n_outliers > 0:
        outlier_idx = rng.choice(n, size=n_outliers, replace=False)
        y[outlier_idx] += rng.normal(0, outlier_scale, size=n_outliers)
        is_outlier[outlier_idx] = True

    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "sigma_x": sigma_x,
            "is_outlier": is_outlier,
        }
    )

    ground_truth = {
        "n": n,
        "true_intercept": intercept,
        "true_slope": slope,
        "base_sigma": base_sigma,
        "hetero_strength": hetero_strength,
        "sigma_range": [float(sigma_x.min()), float(sigma_x.max())],
        "outlier_frac": outlier_frac,
        "outlier_scale": outlier_scale,
        "n_outliers": n_outliers,
        "seed": seed,
    }
    write_dataset("hetero-regression", df, ground_truth, output)
