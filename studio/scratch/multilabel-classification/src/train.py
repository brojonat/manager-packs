"""Train an XGBoost multilabel classifier wrapped in MultiOutputClassifier.

Specializes the multiclass template for the multilabel case:

- Target is (n_samples, n_labels) — multiple binary columns
- Wrap XGBClassifier in MultiOutputClassifier (one model per label)
- Hamming loss is the primary metric, NOT subset accuracy
- Per-label F1 + macro/micro/weighted (different semantics from multiclass)
- Label co-occurrence heatmap to spot label dependencies
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
    hamming_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "multilabel-classification.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"

sys.path.insert(0, str(THIS_DIR))
from plots import (  # noqa: E402
    cardinality_plot,
    label_balance_plot,
    label_cooccurrence,
    per_label_metrics_plot,
)


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def build_pipeline(feature_cols: list[str], seed: int) -> Pipeline:
    """XGBClassifier wrapped in MultiOutputClassifier inside a sklearn Pipeline.

    MultiOutputClassifier fits one independent XGBoost model per label.
    For correlated labels, ClassifierChain is the alternative — it
    feeds each label's prediction as a feature to the next, in a
    user-chosen order. ClassifierChain is more powerful but slower
    and order-sensitive.
    """
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
                MultiOutputClassifier(
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=seed,
                        n_jobs=-1,
                    ),
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
    parser.add_argument("--experiment", default="multilabel-classification-xgb")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Data not found at {args.data}. Run `datagen multilabel-classification` first."
        )

    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    truth = sidecar.get("ground_truth", {})

    # --- ibis: load + materialize ---
    table = ibis.duckdb.connect().read_parquet(str(args.data))
    feature_cols = [c for c in table.columns if c.startswith("feature_")]
    label_cols = [c for c in table.columns if c.startswith("label_")]

    data = (
        table
        .select(*feature_cols, *label_cols)
        .execute()
    )
    X = data[feature_cols]
    Y = data[label_cols].to_numpy().astype(int)
    n_labels = Y.shape[1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=args.test_size, random_state=args.seed
    )

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "n_rows": len(data),
                "n_features": len(feature_cols),
                "n_labels": n_labels,
                "label_columns": ",".join(label_cols),
                "seed": args.seed,
                "test_size": args.test_size,
                "model": "MultiOutputClassifier(XGBClassifier)",
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
            }
        )
        mlflow.set_tag("data_hash", data_hash(data))
        for k in ("avg_labels_per_sample_target", "label_cardinality", "label_density"):
            if k in truth:
                mlflow.set_tag(f"truth.{k}", str(truth[k]))

        # --- fit ---
        pipeline = build_pipeline(feature_cols, args.seed)
        pipeline.fit(X_train, Y_train)

        # --- predict ---
        Y_pred = pipeline.predict(X_test)

        # --- metrics ---
        ham_loss = float(hamming_loss(Y_test, Y_pred))
        subset_acc = float(accuracy_score(Y_test, Y_pred))  # exact-match accuracy
        f1_macro = float(f1_score(Y_test, Y_pred, average="macro", zero_division=0))
        f1_micro = float(f1_score(Y_test, Y_pred, average="micro", zero_division=0))
        f1_weighted = float(
            f1_score(Y_test, Y_pred, average="weighted", zero_division=0)
        )
        f1_samples = float(
            f1_score(Y_test, Y_pred, average="samples", zero_division=0)
        )

        mlflow.log_metric("test_hamming_loss", ham_loss)
        mlflow.log_metric("test_subset_accuracy", subset_acc)
        mlflow.log_metric("test_f1_macro", f1_macro)
        mlflow.log_metric("test_f1_micro", f1_micro)
        mlflow.log_metric("test_f1_weighted", f1_weighted)
        mlflow.log_metric("test_f1_samples", f1_samples)

        # Per-label metrics
        for i, lbl in enumerate(label_cols):
            f1_i = float(
                f1_score(Y_test[:, i], Y_pred[:, i], average="binary", zero_division=0)
            )
            mlflow.log_metric(f"test_f1__{lbl}", f1_i)

        # Cardinality stats
        true_card = float(Y_test.sum(axis=1).mean())
        pred_card = float(Y_pred.sum(axis=1).mean())
        mlflow.log_metric("true_label_cardinality_test", true_card)
        mlflow.log_metric("pred_label_cardinality_test", pred_card)

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

        save(label_balance_plot(Y_test, label_cols), "label_balance.png")
        save(label_cooccurrence(Y_test, label_cols), "label_cooccurrence.png")
        save(per_label_metrics_plot(Y_test, Y_pred, label_cols), "per_label_metrics.png")
        save(cardinality_plot(Y_test, Y_pred), "label_cardinality.png")

        # --- sidecar artifact ---
        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        # --- summary ---
        print(f"run_id:           {run.info.run_id}")
        print(f"experiment:       {args.experiment}")
        print(f"n_labels:         {n_labels}")
        print()
        print(f"hamming loss:     {ham_loss:.4f}  (lower = better; primary metric)")
        print(f"subset accuracy:  {subset_acc:.4f}  (very strict — all labels must match)")
        print(f"label cardinality: true={true_card:.2f}, pred={pred_card:.2f}")
        print()
        print("F1 averaging:")
        print(f"  macro:    {f1_macro:.4f}  (unweighted mean across labels)")
        print(f"  micro:    {f1_micro:.4f}  (pooled across all label predictions)")
        print(f"  weighted: {f1_weighted:.4f}  (weighted by label support)")
        print(f"  samples:  {f1_samples:.4f}  (per-row F1, then averaged)")
        print()
        print("Per-label F1:")
        for i, lbl in enumerate(label_cols):
            f1_i = float(f1_score(Y_test[:, i], Y_pred[:, i], average="binary", zero_division=0))
            pos_rate = float(Y_test[:, i].mean())
            print(f"  {lbl}: F1={f1_i:.4f}  (positive rate {pos_rate:.1%})")


if __name__ == "__main__":
    main()
