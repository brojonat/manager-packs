"""Train an XGBoost binary classifier on the binary-classification dataset.

Specializes the sklearn-pipeline template with the things that make
binary classification actually work in practice:

- scale_pos_weight for class imbalance (XGBoost-native, no resampling)
- Threshold tuning to maximize F1 (0.5 is rarely the right cutoff)
- Calibration verification (Brier score + reliability diagram)
- SHAP feature importance (interpretability XGBoost gives you for free)

The plumbing (Pipeline / ColumnTransformer / MLflow / artifact persistence)
is identical to the Phase 0 template.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ibis
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "binary-classification.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"

sys.path.insert(0, str(THIS_DIR))
from plots import (  # noqa: E402
    calibration_plot,
    confusion_matrix_plot,
    roc_pr_curves,
    shap_summary,
    threshold_sweep,
)


def data_hash(df: pd.DataFrame) -> str:
    """Stable short hash of the materialized dataframe (post-ibis-execute)."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def build_pipeline(scale_pos_weight: float, seed: int) -> Pipeline:
    """XGBClassifier inside a sklearn Pipeline.

    Why no early stopping in this Pipeline:
    early-stopping needs an `eval_set` of *preprocessed* validation data,
    which Pipeline can't easily provide in a single fit() call. The clean
    fix is a two-stage fit (preprocess separately, then fit with eval_set),
    but that breaks the "preprocessing travels with the model" rule.
    Instead we set a moderate `n_estimators` with a low learning rate and
    let the regularization (L2 + max_depth + subsample) do the work. The
    SKILL.md has a section on doing early stopping the manual way if you
    want it.
    """
    feature_cols = [c for c in build_pipeline.feature_cols]  # set in main
    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[("num", StandardScaler(), feature_cols)],
                    remainder="drop",
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--experiment", default="binary-classification-xgb")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data not found at {args.data}. Run `datagen binary-classification` first.")

    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    truth = sidecar.get("ground_truth", {})

    # --- ibis: load + summarize at the source, materialize once for sklearn ---
    table = ibis.duckdb.connect().read_parquet(str(args.data))
    feature_cols = [c for c in table.columns if c.startswith("feature_")]

    # Class balance via an ibis aggregation (pushed down to DuckDB)
    class_stats = (
        table
        .aggregate(
            n_pos=table.target.sum().cast("int64"),
            n_total=table.count(),
        )
        .execute()
        .iloc[0]
    )
    n_pos = int(class_stats["n_pos"])
    n_neg = int(class_stats["n_total"]) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Materialize features + target for sklearn (the ibis → pandas boundary)
    data = (
        table
        .select(*feature_cols, "target")
        .execute()
    )
    X = data[feature_cols]
    y = data["target"].astype(int)

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
                "n_rows": len(data),
                "n_features": len(feature_cols),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "scale_pos_weight": round(scale_pos_weight, 4),
                "seed": args.seed,
                "test_size": args.test_size,
                "model": "XGBClassifier",
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
            }
        )

        # --- tags ---
        mlflow.set_tag("data_hash", data_hash(data))
        mlflow.set_tag("imbalance_ratio", f"{n_pos / len(y):.4f}")
        for k, v in truth.items():
            if k != "class_balance":
                mlflow.set_tag(f"truth.{k}", str(v))

        # --- fit ---
        build_pipeline.feature_cols = feature_cols
        pipeline = build_pipeline(scale_pos_weight, args.seed)
        pipeline.fit(X_train, y_train)

        # --- predict on test ---
        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_pred_default = (test_proba >= 0.5).astype(int)

        # --- metrics ---
        roc_auc = float(roc_auc_score(y_test, test_proba))
        pr_auc = float(average_precision_score(y_test, test_proba))
        ll = float(log_loss(y_test, test_proba))
        brier = float(brier_score_loss(y_test, test_proba))
        f1_default = float(f1_score(y_test, test_pred_default))

        mlflow.log_metric("test_roc_auc", roc_auc)
        mlflow.log_metric("test_pr_auc", pr_auc)
        mlflow.log_metric("test_log_loss", ll)
        mlflow.log_metric("test_brier_score", brier)
        mlflow.log_metric("test_f1_at_0.5", f1_default)

        # --- threshold tuning on test (in production: use a held-out val) ---
        plots_dir = TEMPLATE_DIR / "_tmp_plots"
        plots_dir.mkdir(exist_ok=True)

        f_thresh, best = threshold_sweep(y_test.to_numpy(), test_proba)
        mlflow.log_metric("best_f1_threshold", best["best_f1_threshold"])
        mlflow.log_metric("best_f1", best["best_f1"])
        mlflow.log_metric("precision_at_best_f1", best["precision_at_best_f1"])
        mlflow.log_metric("recall_at_best_f1", best["recall_at_best_f1"])

        test_pred_tuned = (test_proba >= best["best_f1_threshold"]).astype(int)

        # --- log model ---
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(5),
        )

        # --- artifacts: plots ---
        f_roc = roc_pr_curves(y_test.to_numpy(), test_proba)
        f_roc.savefig(plots_dir / "roc_pr.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "roc_pr.png"), artifact_path="plots")

        f_cal = calibration_plot(y_test.to_numpy(), test_proba)
        f_cal.savefig(plots_dir / "calibration.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "calibration.png"), artifact_path="plots")

        f_thresh.savefig(plots_dir / "threshold_sweep.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "threshold_sweep.png"), artifact_path="plots")

        f_cm = confusion_matrix_plot(y_test.to_numpy(), test_pred_tuned, best["best_f1_threshold"])
        f_cm.savefig(plots_dir / "confusion_matrix.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "confusion_matrix.png"), artifact_path="plots")

        # --- SHAP (use the underlying booster, on transformed features) ---
        preprocessor = pipeline.named_steps["preprocess"]
        clf = pipeline.named_steps["clf"]
        X_test_t = preprocessor.transform(X_test.iloc[:200])  # subsample for speed
        f_shap = shap_summary(clf, X_test_t, feature_cols)
        f_shap.savefig(plots_dir / "shap_summary.png", dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(plots_dir / "shap_summary.png"), artifact_path="plots")

        # --- sidecar artifact ---
        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        print(f"run_id:           {run.info.run_id}")
        print(f"experiment:       {args.experiment}")
        print(f"class balance:    {n_pos}/{len(y)} positive ({n_pos / len(y):.2%})")
        print(f"scale_pos_weight: {scale_pos_weight:.3f}")
        print(f"test ROC-AUC:     {roc_auc:.4f}")
        print(f"test PR-AUC:      {pr_auc:.4f}  (baseline = {n_pos / len(y):.4f})")
        print(f"test log-loss:    {ll:.4f}")
        print(f"test Brier score: {brier:.4f}  (lower = better calibrated)")
        print(f"F1 @ 0.5:         {f1_default:.4f}")
        print(f"F1 @ best ({best['best_f1_threshold']:.2f}): {best['best_f1']:.4f}  ← tuned threshold")
        print(f"  precision={best['precision_at_best_f1']:.3f}  recall={best['recall_at_best_f1']:.3f}")


if __name__ == "__main__":
    main()
