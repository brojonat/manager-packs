"""Binary classification — sklearn make_classification with 2 classes."""

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from datagen.output import write_dataset


@click.command("binary-classification")
@click.option("--n", default=1000, show_default=True, help="Number of samples.")
@click.option("--features", default=10, show_default=True, help="Total feature count.")
@click.option("--informative", default=5, show_default=True, help="Informative features.")
@click.option("--redundant", default=2, show_default=True, help="Redundant features.")
@click.option("--class-sep", default=1.0, show_default=True, help="Class separation factor.")
@click.option(
    "--imbalance",
    default=0.5,
    show_default=True,
    help="Fraction of class 1 (0.5 = balanced).",
)
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--output", default=None, help="Output path.")
def binary_classification(
    n: int,
    features: int,
    informative: int,
    redundant: int,
    class_sep: float,
    imbalance: float,
    seed: int,
    output: str | None,
):
    """Generate a binary classification dataset with known structure."""
    X, y = make_classification(
        n_samples=n,
        n_features=features,
        n_informative=informative,
        n_redundant=redundant,
        n_classes=2,
        weights=[1 - imbalance, imbalance],
        class_sep=class_sep,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y.astype(np.int8)

    ground_truth = {
        "n": n,
        "n_features": features,
        "n_informative": informative,
        "n_redundant": redundant,
        "class_sep": class_sep,
        "imbalance": imbalance,
        "seed": seed,
        "class_balance": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
    }
    write_dataset("binary-classification", df, ground_truth, output)
