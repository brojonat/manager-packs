"""Load a trained model from MLflow and predict on a new flip index.

Demonstrates the reload path: model + preprocessing come back as one
Pipeline because we logged via `mlflow.sklearn.log_model`. No need for
the original training code.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

DEFAULT_TRACKING = Path(__file__).resolve().parent.parent / "mlruns"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID from train.py output.")
    parser.add_argument("--flip-index", type=int, default=0)
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model")

    X = pd.DataFrame({"flip_index": [args.flip_index]})
    proba = model.predict_proba(X)[0]
    print(f"flip_index={args.flip_index}")
    print(f"  P(heads) = {proba[1]:.4f}")
    print(f"  P(tails) = {proba[0]:.4f}")


if __name__ == "__main__":
    main()
