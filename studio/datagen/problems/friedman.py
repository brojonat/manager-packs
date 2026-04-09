"""Friedman1 — non-linear regression benchmark from sklearn.

The Friedman #1 problem:

    y = 10 * sin(π * x₀ * x₁) + 20 * (x₂ - 0.5)² + 10 * x₃ + 5 * x₄ + noise

Features `x₀..x₄` are informative; `x₅..x₉` (added by --noise-features)
are pure noise. The function has clear non-linear interactions (sin of
a product, a quadratic) that linear regression cannot capture, which is
the whole point of using it for an XGBoost regression demo.
"""

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_friedman1

from datagen.output import write_dataset


@click.command("friedman")
@click.option("--n", default=2000, show_default=True, help="Number of samples.")
@click.option(
    "--noise-features",
    default=5,
    show_default=True,
    help="Total non-informative features added on top of the 5 informative ones.",
)
@click.option("--noise", default=1.0, show_default=True, help="Std of additive Gaussian noise.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--output", default=None, help="Output path.")
def friedman(n: int, noise_features: int, noise: float, seed: int, output: str | None):
    """Generate the Friedman #1 non-linear regression dataset."""
    n_features = 5 + noise_features  # make_friedman1 always uses 5 informative + extras
    X, y = make_friedman1(n_samples=n, n_features=n_features, noise=noise, random_state=seed)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names).assign(target=y.astype(np.float64))

    ground_truth = {
        "n": n,
        "n_features": n_features,
        "n_informative": 5,  # features 0..4 are informative; rest are noise
        "informative_features": [f"feature_{i}" for i in range(5)],
        "noise_features": [f"feature_{i}" for i in range(5, n_features)],
        "noise_std": noise,
        "irreducible_rmse": noise,  # best possible RMSE = noise std
        "seed": seed,
        "function_form": "y = 10*sin(pi*x0*x1) + 20*(x2-0.5)^2 + 10*x3 + 5*x4 + N(0,noise)",
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
    }
    write_dataset("friedman", df, ground_truth, output)
