"""Load the three regression models from MLflow and predict point + interval.

Demonstrates the "three model" reload pattern: a point estimator plus
two quantile estimators give you a prediction interval per row.
"""

import argparse
from pathlib import Path

import ibis
import mlflow
import mlflow.sklearn

DEFAULT_TRACKING = Path(__file__).resolve().parent.parent / "mlruns"
DEFAULT_DATA = Path(__file__).resolve().parent.parent.parent.parent / "data" / "friedman.parquet"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--n-rows", type=int, default=5)
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")

    point_model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model_point")
    lower_model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model_lower")
    upper_model = mlflow.sklearn.load_model(f"runs:/{args.run_id}/model_upper")

    # Pull the conformal calibration from the run's metrics
    run_info = mlflow.tracking.MlflowClient().get_run(args.run_id)
    conformal_q = float(run_info.data.metrics.get("conformal_q", 0.0))

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

    y_point = point_model.predict(X)
    # Conformalized: expand raw quantile bounds by ±conformal_q
    y_low = lower_model.predict(X) - conformal_q
    y_high = upper_model.predict(X) + conformal_q

    print(f"Conformal calibration: ±{conformal_q:.4f}")
    print()
    print(f"{'row':>4}  {'point':>10}  {'interval (conformal)':>24}  {'true':>10}  {'inside?':>8}")
    print("-" * 72)
    for i in range(len(y_point)):
        inside = "✓" if y_low[i] <= y_true.iloc[i] <= y_high[i] else "✗"
        interval = f"[{y_low[i]:.2f}, {y_high[i]:.2f}]"
        print(f"{i:>4}  {y_point[i]:>10.4f}  {interval:>24}  {y_true.iloc[i]:>10.4f}  {inside:>8}")


if __name__ == "__main__":
    main()
