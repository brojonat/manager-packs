"""Load the multiclass classifier from MLflow and predict on a sample.

Shows the multiclass-specific reload pattern: predict_proba returns
shape (n_samples, n_classes) instead of (n_samples, 2). Top-K
prediction is built on top of that.
"""

import argparse
from pathlib import Path

import ibis
import mlflow
import mlflow.sklearn
import numpy as np

DEFAULT_TRACKING = Path(__file__).resolve().parent.parent / "mlruns"
DEFAULT_DATA = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "multiclass-classification.parquet"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n-rows", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")

    table = ibis.duckdb.connect().read_parquet(str(args.data))
    feature_cols = [c for c in table.columns if c.startswith("feature_")]
    sample = (
        table
        .select(*feature_cols, "target")
        .limit(args.n_rows)
        .execute()
    )
    X = sample[feature_cols]
    y_true = sample["target"]

    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)

    print(f"Predictions (top-{args.top_k} shown)")
    print()
    for i in range(len(pred)):
        top_indices = np.argsort(proba[i])[::-1][: args.top_k]
        top_pairs = ", ".join(f"c{int(c)}: {proba[i, c]:.3f}" for c in top_indices)
        marker = "✓" if pred[i] == int(y_true.iloc[i]) else "✗"
        print(f"  row {i}: pred=c{pred[i]}  true=c{int(y_true.iloc[i])}  {marker}")
        print(f"          top-{args.top_k}: {top_pairs}")


if __name__ == "__main__":
    main()
