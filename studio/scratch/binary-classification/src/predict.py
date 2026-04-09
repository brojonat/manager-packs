"""Load a trained binary classifier from MLflow and predict on a new sample.

Demonstrates the reload + apply-threshold path. The model comes back as
a single Pipeline (preprocessing + XGBClassifier), so the prediction
code is the same regardless of feature scaling.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

DEFAULT_TRACKING = Path(__file__).resolve().parent.parent / "mlruns"
DEFAULT_DATA = Path(__file__).resolve().parent.parent.parent.parent / "data" / "binary-classification.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (use the best_f1_threshold from training)")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Sample data to predict on (defaults to first 5 rows of training data)")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")

    df = pd.read_parquet(args.data).head(5)
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols]
    y_true = df["target"].astype(int)

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= args.threshold).astype(int)

    print(f"Predictions @ threshold={args.threshold:.2f}")
    print()
    for i, (p, yhat, ytrue) in enumerate(zip(proba, pred, y_true)):
        marker = "✓" if yhat == ytrue else "✗"
        print(f"  row {i}: P(positive)={p:.4f} → pred={yhat}  true={ytrue}  {marker}")


if __name__ == "__main__":
    main()
