"""A/B test stream — paired Bernoulli arms with known lift."""

import click
import numpy as np
import pandas as pd

from datagen.output import write_dataset


@click.command("ab-test-stream")
@click.option("--n", default=4000, show_default=True, help="Total visitors (split across arms).")
@click.option("--p-control", default=0.05, show_default=True, help="Control conversion rate.")
@click.option("--lift", default=0.015, show_default=True, help="Absolute lift (treatment = control + lift).")
@click.option("--allocation", default=0.5, show_default=True, help="Fraction of traffic to treatment.")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def ab_test_stream(
    n: int,
    p_control: float,
    lift: float,
    allocation: float,
    seed: int,
    output: str | None,
):
    """Generate a sequential A/B test event stream with known lift."""
    rng = np.random.default_rng(seed)
    p_treatment = p_control + lift

    # Assign arms
    arms = rng.choice(
        ["control", "treatment"],
        size=n,
        p=[1 - allocation, allocation],
    )
    n_control = int((arms == "control").sum())
    n_treatment = int((arms == "treatment").sum())

    # Generate outcomes
    outcomes = np.where(
        arms == "control",
        rng.binomial(1, p_control, size=n),
        rng.binomial(1, p_treatment, size=n),
    ).astype(np.int8)

    df = pd.DataFrame(
        {
            "visit_id": np.arange(n, dtype=np.int32),
            "arm": pd.Categorical(arms, categories=["control", "treatment"]),
            "converted": outcomes,
        }
    )

    conv_control = int(outcomes[arms == "control"].sum())
    conv_treatment = int(outcomes[arms == "treatment"].sum())

    ground_truth = {
        "n": n,
        "p_control": p_control,
        "p_treatment": p_treatment,
        "true_lift": lift,
        "true_relative_lift": lift / p_control if p_control > 0 else None,
        "allocation": allocation,
        "seed": seed,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "conversions_control": conv_control,
        "conversions_treatment": conv_treatment,
        "empirical_p_control": conv_control / n_control if n_control > 0 else 0.0,
        "empirical_p_treatment": conv_treatment / n_treatment if n_treatment > 0 else 0.0,
    }
    write_dataset("ab-test-stream", df, ground_truth, output)
