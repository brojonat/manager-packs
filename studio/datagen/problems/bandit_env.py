"""Bandit environment — Bernoulli arms with optional context features."""

import click
import numpy as np
import pandas as pd

from datagen.output import write_dataset


@click.command("bandit-env")
@click.option("--n-rounds", default=5000, show_default=True, help="Total rounds.")
@click.option("--n-arms", default=5, show_default=True, help="Number of arms.")
@click.option("--n-contexts", default=0, show_default=True, help="Discrete context levels (0 = no context).")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def bandit_env(
    n_rounds: int,
    n_arms: int,
    n_contexts: int,
    seed: int,
    output: str | None,
):
    """Generate a multi-armed bandit environment with known reward probabilities."""
    rng = np.random.default_rng(seed)

    # Generate true arm probabilities
    if n_contexts == 0:
        # Non-contextual: one probability per arm
        true_probs = np.sort(rng.uniform(0.05, 0.5, size=n_arms))[::-1]
        # Ensure a clear best arm
        true_probs[0] = min(true_probs[0] + 0.1, 0.85)

        # Uniform random arm pulls (the "data collection policy")
        arms = rng.integers(0, n_arms, size=n_rounds)
        rewards = rng.binomial(1, true_probs[arms]).astype(np.int8)

        df = pd.DataFrame(
            {
                "round": np.arange(n_rounds, dtype=np.int32),
                "arm": arms.astype(np.int16),
                "reward": rewards,
            }
        )

        ground_truth = {
            "n_rounds": n_rounds,
            "n_arms": n_arms,
            "contextual": False,
            "seed": seed,
            "true_probs": [float(p) for p in true_probs],
            "best_arm": int(np.argmax(true_probs)),
            "best_prob": float(true_probs.max()),
            "gap": float(true_probs[0] - true_probs[1]),
        }
    else:
        # Contextual: probability depends on (context, arm)
        true_probs_matrix = rng.uniform(0.05, 0.6, size=(n_contexts, n_arms))
        # Ensure each context has a different best arm
        for c in range(min(n_contexts, n_arms)):
            true_probs_matrix[c, c] = min(
                true_probs_matrix[c, c] + 0.2, 0.85,
            )

        contexts = rng.integers(0, n_contexts, size=n_rounds)
        arms = rng.integers(0, n_arms, size=n_rounds)
        rewards = rng.binomial(
            1, true_probs_matrix[contexts, arms],
        ).astype(np.int8)

        df = pd.DataFrame(
            {
                "round": np.arange(n_rounds, dtype=np.int32),
                "context": contexts.astype(np.int16),
                "arm": arms.astype(np.int16),
                "reward": rewards,
            }
        )

        best_per_context = {
            str(c): {
                "best_arm": int(np.argmax(true_probs_matrix[c])),
                "best_prob": float(true_probs_matrix[c].max()),
            }
            for c in range(n_contexts)
        }

        ground_truth = {
            "n_rounds": n_rounds,
            "n_arms": n_arms,
            "n_contexts": n_contexts,
            "contextual": True,
            "seed": seed,
            "true_probs_matrix": [
                [float(p) for p in row] for row in true_probs_matrix
            ],
            "best_per_context": best_per_context,
        }

    write_dataset("bandit-env", df, ground_truth, output)
