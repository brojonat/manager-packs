"""Load the multilabel classifier from MLflow and predict on a sample.

Multilabel-specific reload pattern: predict() returns a (n_samples,
n_labels) matrix of 0/1 values. predict_proba() returns a list of
length n_labels, each entry shape (n_samples, 2).
"""

import argparse
from pathlib import Path

import ibis
import mlflow
import mlflow.sklearn
import numpy as np

DEFAULT_TRACKING = Path(__file__).resolve().parent.parent / "mlruns"
DEFAULT_DATA = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "multilabel-classification.parquet"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n-rows", type=int, default=5)
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")

    table = ibis.duckdb.connect().read_parquet(str(args.data))
    feature_cols = [c for c in table.columns if c.startswith("feature_")]
    label_cols = [c for c in table.columns if c.startswith("label_")]
    sample = (
        table
        .select(*feature_cols, *label_cols)
        .limit(args.n_rows)
        .execute()
    )
    X = sample[feature_cols]
    Y_true = sample[label_cols].to_numpy().astype(int)

    Y_pred = model.predict(X)
    # predict_proba on MultiOutputClassifier returns a list (one (n, 2) per label)
    proba_list = model.predict_proba(X)
    pos_proba = np.column_stack([p[:, 1] for p in proba_list])

    print(f"Multilabel predictions ({args.n_rows} rows)")
    print()
    print(f"  {'row':<4}  {'true':<{len(label_cols) * 2 + 1}}  {'pred':<{len(label_cols) * 2 + 1}}  {'P(pos) per label'}")
    print("  " + "-" * 78)
    for i in range(len(Y_pred)):
        true_str = " ".join(str(int(v)) for v in Y_true[i])
        pred_str = " ".join(str(int(v)) for v in Y_pred[i])
        proba_str = " ".join(f"{p:.2f}" for p in pos_proba[i])
        match = "✓" if (Y_true[i] == Y_pred[i]).all() else "✗"
        print(f"  {i:<4}  {true_str:<{len(label_cols) * 2 + 1}}  {pred_str:<{len(label_cols) * 2 + 1}}  {proba_str}  {match}")


if __name__ == "__main__":
    main()
