"""Multiclass classification — sklearn make_classification with N classes."""

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from datagen.output import write_dataset


@click.command("multiclass-classification")
@click.option("--n", default=1000, show_default=True, help="Number of samples.")
@click.option("--features", default=10, show_default=True, help="Total feature count.")
@click.option("--informative", default=6, show_default=True, help="Informative features.")
@click.option("--classes", default=4, show_default=True, help="Number of classes.")
@click.option("--clusters-per-class", default=1, show_default=True)
@click.option("--class-sep", default=1.0, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def multiclass_classification(
    n: int,
    features: int,
    informative: int,
    classes: int,
    clusters_per_class: int,
    class_sep: float,
    seed: int,
    output: str | None,
):
    """Generate a multiclass classification dataset."""
    X, y = make_classification(
        n_samples=n,
        n_features=features,
        n_informative=informative,
        n_classes=classes,
        n_clusters_per_class=clusters_per_class,
        class_sep=class_sep,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y.astype(np.int16)

    ground_truth = {
        "n": n,
        "n_features": features,
        "n_informative": informative,
        "n_classes": classes,
        "clusters_per_class": clusters_per_class,
        "class_sep": class_sep,
        "seed": seed,
        "class_counts": {str(c): int((y == c).sum()) for c in range(classes)},
    }
    write_dataset("multiclass-classification", df, ground_truth, output)
