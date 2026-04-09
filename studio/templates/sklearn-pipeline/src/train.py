"""Train logistic regression on the coin-flip dataset, log to MLflow.

This is the reference template every tabular bundle copies. The model
is trivial (logistic regression on a single index feature) but the
plumbing — Pipeline + ColumnTransformer + cross-val + MLflow logging
+ artifact persistence — is the actual point.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from plots import calibration_plot, coefficient_plot, empirical_vs_predicted

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "coin-flip.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"


def data_hash(df: pd.DataFrame) -> str:
    """Stable short hash of the data so MLflow runs are linked to inputs."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def build_pipeline() -> Pipeline:
    """The whole modeling pipeline.

    Preprocessing lives inside the Pipeline so it travels with the model
    on save/load. This is the convention we want every bundle to follow.
    """
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("idx", StandardScaler(), ["flip_index"]),
                    ],
                    remainder="drop",
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--experiment", default="coin-flip-sklearn")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Data not found at {args.data}. Run `datagen coin-flip` first."
        )

    df = pd.read_parquet(args.data)
    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    truth = sidecar.get("ground_truth", {})
    true_p = truth.get("true_p")

    X = df[["flip_index"]]
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        # --- params ---
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "n_rows": len(df),
                "seed": args.seed,
                "test_size": args.test_size,
                "cv_folds": args.cv_folds,
                "model": "LogisticRegression",
                "max_iter": 1000,
            }
        )

        # --- tags ---
        mlflow.set_tag("data_hash", data_hash(df))
        if true_p is not None:
            mlflow.set_tag("true_p", str(true_p))
        if "drift" in truth:
            mlflow.set_tag("drift", str(truth["drift"]))

        # --- fit ---
        pipeline = build_pipeline()

        cv_scores = -cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=args.cv_folds,
            scoring="neg_log_loss",
        )
        mlflow.log_metric("cv_log_loss_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_log_loss_std", float(cv_scores.std()))

        pipeline.fit(X_train, y_train)

        # --- evaluate ---
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        mlflow.log_metric("test_log_loss", float(log_loss(y_test, pipeline.predict_proba(X_test))))

        # --- interpret coefficients ---
        clf: LogisticRegression = pipeline.named_steps["clf"]
        intercept = float(clf.intercept_[0])
        coef_std = float(clf.coef_[0][0])  # on standardized scale
        # The intercept maps to P(heads) at the mean index because the
        # scaler centers the feature on 0. So:
        p_at_mean_index = float(expit(intercept))

        mlflow.log_metric("intercept_logit", intercept)
        mlflow.log_metric("coef_flip_index_standardized", coef_std)
        mlflow.log_metric("p_at_mean_index", p_at_mean_index)
        if true_p is not None:
            mlflow.log_metric("p_recovery_error", float(abs(p_at_mean_index - true_p)))

        # --- log model (this is the sanctioned serialization path) ---
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(5),
        )

        # --- artifacts: plots ---
        all_proba = pipeline.predict_proba(X)[:, 1]
        plots_dir = TEMPLATE_DIR / "_tmp_plots"
        plots_dir.mkdir(exist_ok=True)

        f1 = empirical_vs_predicted(df, all_proba)
        f1_path = plots_dir / "empirical_vs_predicted.png"
        f1.savefig(f1_path, dpi=120)
        mlflow.log_artifact(str(f1_path), artifact_path="plots")

        f2 = calibration_plot(y_test.to_numpy(), test_proba)
        f2_path = plots_dir / "calibration.png"
        f2.savefig(f2_path, dpi=120)
        mlflow.log_artifact(str(f2_path), artifact_path="plots")

        f3 = coefficient_plot(intercept, coef_std, "flip_index")
        f3_path = plots_dir / "coefficients.png"
        f3.savefig(f3_path, dpi=120)
        mlflow.log_artifact(str(f3_path), artifact_path="plots")

        # --- artifact: sidecar (so reload code knows the ground truth) ---
        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        print(f"run_id:           {run.info.run_id}")
        print(f"experiment:       {args.experiment}")
        print(f"cv log-loss:      {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"intercept (logit):{intercept:+.4f}")
        print(f"coef (std flip):  {coef_std:+.4f}")
        print(f"P(heads) @ mean:  {p_at_mean_index:.4f}")
        if true_p is not None:
            print(f"true p:           {true_p}")
            print(f"|recovery error|: {abs(p_at_mean_index - true_p):.4f}")


if __name__ == "__main__":
    main()
