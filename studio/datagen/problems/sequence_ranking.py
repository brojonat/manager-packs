"""Sequence/ranking data — user visits with position bias and sequence bonuses."""

import click
import numpy as np
import pandas as pd

from datagen.output import write_dataset


@click.command("sequence-ranking")
@click.option("--n-visits", default=2000, show_default=True, help="Number of user visits.")
@click.option("--n-items", default=8, show_default=True, help="Number of items in each ordering.")
@click.option("--position-decay", default=0.15, show_default=True, help="Exponential decay rate for position bias.")
@click.option("--n-pair-bonuses", default=3, show_default=True, help="Number of planted adjacent-pair bonuses.")
@click.option("--seed", default=42, show_default=True)
@click.option("--output", default=None)
def sequence_ranking(
    n_visits: int,
    n_items: int,
    position_decay: float,
    n_pair_bonuses: int,
    seed: int,
    output: str | None,
):
    """Generate user visit data with latent item values, position bias, and sequence bonuses.

    Each visit shows all items in a random order. Click probability for
    each item depends on: (1) item's latent value, (2) position in the
    list (exponential decay), and (3) bonuses when certain pairs appear
    adjacent. Revenue per visit = sum of clicked item values.
    """
    rng = np.random.default_rng(seed)

    # Latent item values (higher = more intrinsically interesting)
    item_values = rng.uniform(0.5, 3.0, size=n_items)

    # Position bias: probability multiplier decays with position
    # position_bias[k] = exp(-decay * k) for position k=0,1,...
    positions = np.arange(n_items)
    position_bias = np.exp(-position_decay * positions)

    # Pair bonuses: specific adjacent pairs get a click probability boost
    pair_bonuses = {}
    all_pairs = [
        (i, j)
        for i in range(n_items)
        for j in range(n_items)
        if i != j
    ]
    bonus_pairs = rng.choice(
        len(all_pairs),
        size=min(n_pair_bonuses, len(all_pairs)),
        replace=False,
    )
    for idx in bonus_pairs:
        pair = all_pairs[idx]
        pair_bonuses[pair] = float(rng.uniform(0.5, 1.5))

    # Generate visits
    rows = []
    for visit_id in range(n_visits):
        # Random ordering of items
        ordering = rng.permutation(n_items)

        # Compute click probabilities
        click_probs = np.zeros(n_items)
        for pos_k in range(n_items):
            item_k = ordering[pos_k]
            base_prob = item_values[item_k] * position_bias[pos_k]

            # Check adjacent pair bonus
            bonus = 0.0
            if pos_k > 0:
                prev_item = ordering[pos_k - 1]
                if (prev_item, item_k) in pair_bonuses:
                    bonus += pair_bonuses[(prev_item, item_k)]
            if pos_k < n_items - 1:
                next_item = ordering[pos_k + 1]
                if (item_k, next_item) in pair_bonuses:
                    bonus += pair_bonuses[(item_k, next_item)]

            click_probs[pos_k] = np.clip(
                0.05 * (base_prob + bonus), 0.0, 0.95,
            )

        # Generate clicks
        clicks = rng.binomial(1, click_probs)
        total_revenue = float(
            sum(item_values[ordering[k]] for k in range(n_items) if clicks[k])
        )

        rows.append(
            {
                "visit_id": visit_id,
                "ordering": ",".join(str(int(x)) for x in ordering),
                "clicks": ",".join(str(int(x)) for x in clicks),
                "total_revenue": round(total_revenue, 2),
                "n_clicks": int(clicks.sum()),
            }
        )

    df = pd.DataFrame(rows)
    df["visit_id"] = df["visit_id"].astype(np.int32)
    df["n_clicks"] = df["n_clicks"].astype(np.int16)

    ground_truth = {
        "n_visits": n_visits,
        "n_items": n_items,
        "position_decay": position_decay,
        "seed": seed,
        "item_values": [float(v) for v in item_values],
        "position_bias": [float(b) for b in position_bias],
        "pair_bonuses": {
            f"{pair[0]}->{pair[1]}": bonus
            for pair, bonus in pair_bonuses.items()
        },
        "best_item": int(np.argmax(item_values)),
        "mean_revenue_per_visit": float(df["total_revenue"].mean()),
        "mean_clicks_per_visit": float(df["n_clicks"].mean()),
    }
    write_dataset("sequence-ranking", df, ground_truth, output)
