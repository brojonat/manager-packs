"""High-dim tabular — low intrinsic dimensionality embedded in high-dim noise."""

import click
import numpy as np
import pandas as pd

from datagen.output import write_dataset


@click.command("high-dim")
@click.option("--n", default=500, show_default=True, help="Number of samples.")
@click.option("--intrinsic-dim", default=3, show_default=True, help="True dimensionality of the signal.")
@click.option("--ambient-dim", default=50, show_default=True, help="Total feature count (signal + noise).")
@click.option("--noise-scale", default=0.3, show_default=True, help="Scale of ambient noise dimensions.")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def high_dim(
    n: int,
    intrinsic_dim: int,
    ambient_dim: int,
    noise_scale: float,
    seed: int,
    output: str | None,
):
    """Generate data on a low-dim manifold embedded in high-dim space.

    Signal lives in `intrinsic_dim` dimensions; the remaining features
    are pure noise. A random rotation mixes signal into all ambient
    dimensions so PCA/UMAP must discover the structure.
    """
    rng = np.random.default_rng(seed)

    if intrinsic_dim >= ambient_dim:
        raise click.BadParameter(
            f"intrinsic-dim ({intrinsic_dim}) must be < ambient-dim ({ambient_dim})"
        )

    # Generate the low-dim signal
    Z = rng.standard_normal((n, intrinsic_dim))

    # Embed into ambient space via a random linear map
    # A is (intrinsic_dim, ambient_dim) so X = Z @ A lives in ambient_dim
    A = rng.standard_normal((intrinsic_dim, ambient_dim))
    X_signal = Z @ A

    # Add isotropic noise in the ambient space
    X_noise = rng.normal(0, noise_scale, size=(n, ambient_dim))
    X = X_signal + X_noise

    # Optional: add a target that depends only on the latent signal
    # y = nonlinear function of the first 2 latent dims
    y = np.sin(Z[:, 0]) + Z[:, 1] ** 2 + 0.3 * rng.standard_normal(n)

    feature_names = [f"feature_{i}" for i in range(ambient_dim)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y.astype(np.float64)

    # Compute explained variance by the signal subspace
    total_var = float(np.var(X, axis=0).sum())
    signal_var = float(np.var(X_signal, axis=0).sum())

    ground_truth = {
        "n": n,
        "intrinsic_dim": intrinsic_dim,
        "ambient_dim": ambient_dim,
        "noise_scale": noise_scale,
        "seed": seed,
        "signal_variance_fraction": signal_var / total_var if total_var > 0 else 0.0,
        "target_function": "sin(z0) + z1^2 + noise",
        "embedding_matrix_shape": list(A.shape),
    }
    write_dataset("high-dim", df, ground_truth, output)
