"""Regression — sklearn make_regression with known coefficients."""

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from datagen.output import write_dataset


@click.command("regression")
@click.option("--n", default=500, show_default=True, help="Number of samples.")
@click.option("--features", default=8, show_default=True, help="Feature count.")
@click.option("--informative", default=5, show_default=True, help="Informative features.")
@click.option("--noise", default=0.5, show_default=True, help="Gaussian noise std.")
@click.option("--bias", default=0.0, show_default=True, help="Bias term.")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def regression(
    n: int,
    features: int,
    informative: int,
    noise: float,
    bias: float,
    seed: int,
    output: str | None,
):
    """Generate a regression dataset with known true coefficients."""
    X, y, coef = make_regression(
        n_samples=n,
        n_features=features,
        n_informative=informative,
        noise=noise,
        bias=bias,
        coef=True,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y.astype(np.float64)

    ground_truth = {
        "n": n,
        "n_features": features,
        "n_informative": informative,
        "noise": noise,
        "bias": bias,
        "seed": seed,
        "true_coefficients": {feature_names[i]: float(coef[i]) for i in range(features)},
    }
    write_dataset("regression", df, ground_truth, output)
