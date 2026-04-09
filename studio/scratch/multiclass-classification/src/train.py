"""Train an XGBoost multiclass classifier on the multiclass dataset.

Specializes the binary-classification template with the things that
matter for multiclass:

- objective="multi:softprob" + eval_metric="mlogloss"
- Per-class metrics (precision, recall, F1, support) — never just accuracy
- macro vs micro vs weighted averaging — log all three
- Confusion matrix as a primary diagnostic
- Top-K accuracy for many-class problems
- sample_weight per row for class imbalance (XGBoost has no
  scale_pos_weight equivalent for multiclass)
- One-vs-rest ROC + per-class SHAP for interpretability
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
    accuracy_score,
    f1_score,
    log_loss,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "multiclass-classification.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"

sys.path.insert(0, str(THIS_DIR))
from plots import (  # noqa: E402
    class_balance_plot,
    confusion_matrix_plot,
    per_class_metrics_plot,
    roc_ovr_plot,
    shap_summary,
)


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def build_pipeline(feature_cols: list[str], n_classes: int, seed: int) -> Pipeline:
    """XGBClassifier multiclass inside a sklearn Pipeline."""
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
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--use-sample-weights",
        action="store_true",
        help="Pass class-balanced sample_weight to handle imbalance.",
    )
    parser.add_argument("--experiment", default="multiclass-classification-xgb")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Data not found at {args.data}. Run `datagen multiclass-classification` first."
        )

    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    truth = sidecar.get("ground_truth", {})

    # --- ibis: load + materialize ---
    table = ibis.duckdb.connect().read_parquet(str(args.data))
    feature_cols = [c for c in table.columns if c.startswith("feature_")]
    data = (
        table
        .select(*feature_cols, "target")
        .execute()
    )
    X = data[feature_cols]
    y = data["target"].astype(int)
    n_classes = int(y.max()) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    sample_weight = (
        compute_sample_weight(class_weight="balanced", y=y_train)
        if args.use_sample_weights
        else None
    )

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "n_rows": len(data),
                "n_features": len(feature_cols),
                "n_classes": n_classes,
                "seed": args.seed,
                "test_size": args.test_size,
                "use_sample_weights": args.use_sample_weights,
                "top_k": args.top_k,
                "model": "XGBClassifier",
                "objective": "multi:softprob",
                "n_estimators": 300,
                "max_depth": 5,
                "learning_rate": 0.05,
            }
        )
        mlflow.set_tag("data_hash", data_hash(data))
        for k in ("class_sep", "n_informative"):
            if k in truth:
                mlflow.set_tag(f"truth.{k}", str(truth[k]))

        # --- fit ---
        pipeline = build_pipeline(feature_cols, n_classes, args.seed)
        if sample_weight is not None:
            pipeline.fit(X_train, y_train, clf__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)

        # --- predict ---
        y_proba = pipeline.predict_proba(X_test)
        y_pred = pipeline.predict(X_test)

        # --- metrics ---
        acc = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        f1_micro = float(f1_score(y_test, y_pred, average="micro"))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
        ll = float(log_loss(y_test, y_proba, labels=list(range(n_classes))))
        top_k = float(
            top_k_accuracy_score(y_test, y_proba, k=args.top_k, labels=list(range(n_classes)))
        )

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_f1_micro", f1_micro)
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        mlflow.log_metric("test_log_loss", ll)
        mlflow.log_metric(f"test_top_{args.top_k}_accuracy", top_k)

        # --- log model ---
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=X_train.head(5),
        )

        # --- plots ---
        plots_dir = TEMPLATE_DIR / "_tmp_plots"
        plots_dir.mkdir(exist_ok=True)

        def save(fig, name):
            path = plots_dir / name
            fig.savefig(path, dpi=120, bbox_inches="tight")
            mlflow.log_artifact(str(path), artifact_path="plots")

        save(class_balance_plot(y_test.to_numpy(), n_classes), "class_balance.png")
        save(confusion_matrix_plot(y_test.to_numpy(), y_pred, n_classes), "confusion_matrix.png")
        save(
            confusion_matrix_plot(y_test.to_numpy(), y_pred, n_classes, normalize=True),
            "confusion_matrix_normalized.png",
        )
        save(per_class_metrics_plot(y_test.to_numpy(), y_pred, n_classes), "per_class_metrics.png")
        save(roc_ovr_plot(y_test.to_numpy(), y_proba, n_classes), "roc_ovr.png")

        # SHAP per class — show summary for the most-confused class
        preprocessor = pipeline.named_steps["preprocess"]
        clf = pipeline.named_steps["clf"]
        X_test_t = preprocessor.transform(X_test.iloc[:200])
        save(shap_summary(clf, X_test_t, feature_cols, class_idx=0), "shap_class_0.png")

        # --- sidecar artifact ---
        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        # --- summary ---
        print(f"run_id:           {run.info.run_id}")
        print(f"experiment:       {args.experiment}")
        print(f"n_classes:        {n_classes}")
        print(f"sample_weights:   {args.use_sample_weights}")
        print()
        print(f"test accuracy:    {acc:.4f}")
        print(f"test top-{args.top_k} accuracy: {top_k:.4f}")
        print(f"test log-loss:    {ll:.4f}")
        print()
        print("F1 averaging:")
        print(f"  macro:    {f1_macro:.4f}  (unweighted mean across classes)")
        print(f"  micro:    {f1_micro:.4f}  (= accuracy on multiclass single-label)")
        print(f"  weighted: {f1_weighted:.4f}  (weighted by class support)")


if __name__ == "__main__":
    main()
