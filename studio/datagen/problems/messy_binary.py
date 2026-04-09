"""Messy binary classification — a synthetic dataset with planted issues
that good EDA should catch.

Used by the tabular-eda bundle. Each "problem" is a known issue in the
data that an EDA workflow should surface. The sidecar JSON documents
the planted issues so the EDA pipeline can be validated against them.

Planted issues:
  1. **Target leakage**: a feature perfectly correlated with the target
     (modeling on this would yield 100% test accuracy and fail in prod)
  2. **High-cardinality categorical**: an ID-like column with hundreds
     of unique values (would explode a OneHotEncoder)
  3. **Near-constant feature**: one value covers >99% of rows (no signal)
  4. **MCAR missing data**: 30% of one numeric column is NaN
  5. **Heavily skewed feature**: log-normal distribution with a long tail
  6. **Outliers**: 2% of one column is sampled from a wide uniform
  7. **Redundant feature**: nearly-perfect linear copy of another feature
"""

import click
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from datagen.output import write_dataset


@click.command("messy-binary")
@click.option("--n", default=2000, show_default=True, help="Number of samples.")
@click.option("--seed", default=42, show_default=True, help="Random seed.")
@click.option("--output", default=None, help="Output path.")
def messy_binary(n: int, seed: int, output: str | None):
    """Generate a binary classification dataset with planted EDA issues."""
    rng = np.random.default_rng(seed)

    # Clean baseline from make_classification
    X_clean, y = make_classification(
        n_samples=n,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=2,
        weights=[0.85, 0.15],  # mild imbalance
        class_sep=1.0,
        random_state=seed,
    )
    df = (
        pd.DataFrame(X_clean, columns=[f"feature_{i}" for i in range(6)])
        .assign(target=y.astype(np.int8))
    )

    # 1. Target leakage: a feature perfectly correlated with target
    df["account_balance_post_action"] = df["target"] * 100.0 + rng.normal(0, 1, n)

    # 2. High-cardinality categorical (1 unique value per ~3 rows)
    df["user_id"] = [f"user_{i % (n // 3)}" for i in range(n)]

    # 3. Near-constant feature: one value 99% of the time
    df["fraud_blocklist"] = rng.choice([0, 1], size=n, p=[0.99, 0.01]).astype(np.int8)

    # 4. MCAR missing: 30% of feature_0 is NaN
    missing_mask = rng.random(n) < 0.30
    df.loc[missing_mask, "feature_0"] = np.nan

    # 5. Heavily skewed feature (log-normal)
    df["transaction_amount"] = rng.lognormal(mean=5.0, sigma=2.0, size=n)

    # 6. Outliers in latency_ms (2% extreme values)
    base_latency = rng.normal(100, 20, n)
    outlier_mask = rng.random(n) < 0.02
    base_latency[outlier_mask] = rng.uniform(1500, 5000, outlier_mask.sum())
    df["latency_ms"] = base_latency

    # 7. Redundant feature: linear copy of feature_1 + tiny noise
    df["feature_1_copy"] = df["feature_1"] + rng.normal(0, 0.01, n)

    # 8. A useful low-cardinality categorical (legitimately useful)
    df["region"] = rng.choice(["north", "south", "east", "west"], size=n)

    ground_truth = {
        "n": n,
        "seed": seed,
        "target": "target",
        "target_type": "binary",
        "class_balance": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "planted_issues": {
            "target_leakage": {
                "feature": "account_balance_post_action",
                "expected_pearson_with_target": ">0.99",
                "lesson": "leakage features look too good to be true",
            },
            "high_cardinality_categorical": {
                "feature": "user_id",
                "expected_unique_count": n // 3,
                "lesson": "OneHotEncoder would explode",
            },
            "near_constant": {
                "feature": "fraud_blocklist",
                "expected_top_value_freq": ">0.98",
                "lesson": "no signal, drop or ignore",
            },
            "missing_data_mcar": {
                "feature": "feature_0",
                "expected_missing_pct": 0.30,
                "lesson": "30% MCAR — impute or drop rows",
            },
            "skewed_distribution": {
                "feature": "transaction_amount",
                "lesson": "long right tail, consider log transform",
            },
            "outliers": {
                "feature": "latency_ms",
                "expected_outlier_pct": 0.02,
                "lesson": "use robust scaler or winsorize",
            },
            "redundant_feature": {
                "feature_pair": ["feature_1", "feature_1_copy"],
                "expected_pearson": ">0.99",
                "lesson": "drop one — multicollinearity",
            },
        },
        "useful_categorical": "region",
    }
    write_dataset("messy-binary", df, ground_truth, output)
