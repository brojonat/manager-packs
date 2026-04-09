"""Blobs — sklearn make_blobs (clustering / unsupervised)."""

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from datagen.output import write_dataset


@click.command("blobs")
@click.option("--n", default=600, show_default=True, help="Number of samples.")
@click.option("--features", default=2, show_default=True, help="Feature count.")
@click.option("--centers", default=3, show_default=True, help="Number of cluster centers.")
@click.option("--cluster-std", default=1.0, show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def blobs(
    n: int,
    features: int,
    centers: int,
    cluster_std: float,
    seed: int,
    output: str | None,
):
    """Generate isotropic Gaussian blobs for clustering experiments."""
    X, y, centers_arr = make_blobs(
        n_samples=n,
        n_features=features,
        centers=centers,
        cluster_std=cluster_std,
        return_centers=True,
        random_state=seed,
    )

    feature_names = [f"feature_{i}" for i in range(features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["cluster"] = y.astype(np.int16)  # the true cluster id (held back at fit time)

    ground_truth = {
        "n": n,
        "n_features": features,
        "n_centers": centers,
        "cluster_std": cluster_std,
        "seed": seed,
        "true_centers": [[float(v) for v in row] for row in centers_arr],
        "cluster_counts": {str(c): int((y == c).sum()) for c in range(centers)},
    }
    write_dataset("blobs", df, ground_truth, output)
