"""Multilabel classification — sklearn make_multilabel_classification."""

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_multilabel_classification

from datagen.output import write_dataset


@click.command("multilabel-classification")
@click.option("--n", default=1000, show_default=True, help="Number of samples.")
@click.option("--features", default=20, show_default=True, help="Feature count.")
@click.option("--labels", default=5, show_default=True, help="Number of labels.")
@click.option("--length", default=2, show_default=True, help="Avg labels per sample.")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def multilabel_classification(
    n: int,
    features: int,
    labels: int,
    length: int,
    seed: int,
    output: str | None,
):
    """Generate a multilabel classification dataset (samples can have multiple labels)."""
    X, Y = make_multilabel_classification(
        n_samples=n,
        n_features=features,
        n_classes=labels,
        n_labels=length,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(features)]
    df = pd.DataFrame(X, columns=feature_names)
    for i in range(labels):
        df[f"label_{i}"] = Y[:, i].astype(np.int8)

    label_card = float(Y.sum(axis=1).mean())  # avg labels per sample
    label_density = float(label_card / labels)

    ground_truth = {
        "n": n,
        "n_features": features,
        "n_labels": labels,
        "avg_labels_per_sample_target": length,
        "seed": seed,
        "label_cardinality": label_card,
        "label_density": label_density,
        "label_counts": {f"label_{i}": int(Y[:, i].sum()) for i in range(labels)},
    }
    write_dataset("multilabel-classification", df, ground_truth, output)
