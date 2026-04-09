"""Coin flip — Bernoulli sequence with optional drift."""

import click
import numpy as np
import pandas as pd

from datagen.output import write_dataset


@click.command("coin-flip")
@click.option("--n", default=200, show_default=True, help="Number of flips.")
@click.option("--p", default=0.5, show_default=True, help="True heads probability.")
@click.option(
    "--drift",
    default=0.0,
    show_default=True,
    help="Linear drift in p over the sequence (delta added linearly from 0 to drift).",
)
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--output", default=None, help="Output path (default: studio/data/coin-flip.parquet).")
def coin_flip(n: int, p: float, drift: float, seed: int, output: str | None):
    """Generate a sequence of coin flips with known bias."""
    rng = np.random.default_rng(seed)

    if drift == 0.0:
        probs = np.full(n, p)
    else:
        probs = np.clip(p + np.linspace(0.0, drift, n), 0.0, 1.0)

    outcomes = rng.binomial(1, probs).astype(np.int8)
    df = pd.DataFrame(
        {
            "flip_index": np.arange(n, dtype=np.int32),
            "outcome": outcomes,
        }
    )

    ground_truth = {
        "n": n,
        "true_p": p,
        "drift": drift,
        "seed": seed,
        "is_stationary": drift == 0.0,
        "empirical_p": float(outcomes.mean()),
    }
    write_dataset("coin-flip", df, ground_truth, output)
