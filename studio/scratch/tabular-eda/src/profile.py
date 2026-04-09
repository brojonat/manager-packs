"""Profile a tabular dataset and log everything to MLflow.

This is the EDA workflow we want every project to start with: shape +
dtypes + missing + distributions + correlations + leakage detection +
high-cardinality detection + mutual-information vs Pearson. The output
is a tracked MLflow run that captures the state of the data at the
moment you started modeling, plus a `findings.json` artifact that
lists every suspicious thing the workflow found.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ibis
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "messy-binary.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"

sys.path.insert(0, str(THIS_DIR))
from plots import (  # noqa: E402
    categorical_cardinality,
    correlation_heatmap,
    mi_vs_pearson,
    missing_data_bar,
    numeric_distributions,
    outlier_boxplot,
)


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def infer_target_type(y: pd.Series) -> str:
    """Heuristic: binary, multiclass (≤20 classes), or regression."""
    if y.dtype.kind in "biu":  # bool / int
        n_unique = y.nunique()
        if n_unique == 2:
            return "binary"
        if n_unique <= 20:
            return "multiclass"
        return "regression"  # high-cardinality int — probably a count or ID
    if y.dtype.kind == "f":
        return "regression"
    return "categorical"  # object / category dtype with > 2 levels


def find_leakage_candidates(
    df: pd.DataFrame, target_col: str, numeric_cols: list[str], threshold: float = 0.95
) -> list[dict]:
    """Features with |Pearson| > threshold to the target are leakage suspects."""
    if df[target_col].dtype.kind not in "biuf":
        return []
    out = []
    target_values = df[target_col].astype(float)
    for col in numeric_cols:
        if col == target_col:
            continue
        try:
            corr = float(df[[col, target_col]].dropna().corr().iloc[0, 1])
        except Exception:
            continue
        if not np.isfinite(corr):
            continue
        if abs(corr) > threshold:
            out.append({"feature": col, "pearson": round(corr, 4)})
    return out


def find_high_cardinality(
    df: pd.DataFrame, cat_cols: list[str], threshold: int = 50
) -> list[dict]:
    """Categoricals with too many unique values would explode a OneHotEncoder."""
    out = []
    for col in cat_cols:
        n_unique = int(df[col].nunique())
        if n_unique > threshold:
            out.append({"feature": col, "n_unique": n_unique, "n_rows": len(df)})
    return out


def find_near_constant(
    df: pd.DataFrame, all_cols: list[str], threshold: float = 0.98
) -> list[dict]:
    """Columns where one value covers > threshold of rows are useless."""
    out = []
    for col in all_cols:
        try:
            top_freq = float(df[col].value_counts(normalize=True).iloc[0])
        except Exception:
            continue
        if top_freq > threshold:
            out.append({"feature": col, "top_value_freq": round(top_freq, 4)})
    return out


def find_redundant_pairs(
    df: pd.DataFrame, numeric_cols: list[str], threshold: float = 0.95
) -> list[dict]:
    """Pairs of numeric features with |Pearson| > threshold are redundant."""
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr().abs()
    out = []
    seen = set()
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i + 1 :]:
            v = float(corr.loc[c1, c2])
            if v > threshold and (c1, c2) not in seen:
                out.append({"pair": [c1, c2], "pearson": round(v, 4)})
                seen.add((c1, c2))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--target", default="target", help="Target column name.")
    parser.add_argument("--experiment", default="tabular-eda")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data not found at {args.data}.")

    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}

    # ibis: read once, materialize
    table = ibis.duckdb.connect().read_parquet(str(args.data))
    df = table.execute()

    target_col = args.target
    if target_col not in df.columns:
        raise SystemExit(f"Target column `{target_col}` not in dataframe columns.")

    numeric_cols = [c for c in df.columns if df[c].dtype.kind in "biuf"]
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    target_type = infer_target_type(df[target_col])

    # --- profiling stats ---
    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    missing_pct_overall = missing_cells / total_cells if total_cells else 0.0
    duplicate_rows = int(df.duplicated().sum())

    # --- findings ---
    leakage = find_leakage_candidates(df, target_col, numeric_cols)
    high_card = find_high_cardinality(df, cat_cols)
    near_const = find_near_constant(df, list(df.columns))
    redundant = find_redundant_pairs(
        df, [c for c in numeric_cols if c != target_col]
    )

    # --- mutual information vs Pearson ---
    feature_numeric = [c for c in numeric_cols if c != target_col]
    pearson_corr = (
        df[feature_numeric + [target_col]]
        .corr()[target_col]
        .drop(target_col)
        .abs()
    )
    # MI requires no missing values; impute with median for the score
    X_for_mi = df[feature_numeric].copy()
    for col in feature_numeric:
        if X_for_mi[col].isna().any():
            X_for_mi[col] = X_for_mi[col].fillna(X_for_mi[col].median())
    if target_type in ("binary", "multiclass"):
        mi_scores = mutual_info_classif(
            X_for_mi.values, df[target_col].values, random_state=0
        )
    else:
        mi_scores = mutual_info_regression(
            X_for_mi.values, df[target_col].values, random_state=0
        )
    mi_series = pd.Series(mi_scores, index=feature_numeric)

    # --- MLflow ---
    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "n_numeric": len(numeric_cols),
                "n_categorical": len(cat_cols),
                "target_col": target_col,
                "target_type": target_type,
            }
        )

        mlflow.set_tag("data_hash", data_hash(df))

        mlflow.log_metric("missing_cells", missing_cells)
        mlflow.log_metric("missing_pct_overall", round(missing_pct_overall, 4))
        mlflow.log_metric("duplicate_rows", duplicate_rows)
        mlflow.log_metric("n_leakage_candidates", len(leakage))
        mlflow.log_metric("n_high_cardinality", len(high_card))
        mlflow.log_metric("n_near_constant", len(near_const))
        mlflow.log_metric("n_redundant_pairs", len(redundant))

        # --- artifacts ---
        plots_dir = TEMPLATE_DIR / "_tmp_plots"
        plots_dir.mkdir(exist_ok=True)

        def save(fig, name):
            path = plots_dir / name
            fig.savefig(path, dpi=120, bbox_inches="tight")
            mlflow.log_artifact(str(path), artifact_path="plots")

        save(missing_data_bar(df), "missing.png")
        save(numeric_distributions(df, [c for c in numeric_cols if c != target_col]), "numeric_distributions.png")
        save(categorical_cardinality(df, cat_cols), "categorical_cardinality.png")
        save(correlation_heatmap(df, numeric_cols, target_col), "correlation_heatmap.png")
        save(mi_vs_pearson(pearson_corr, mi_series, target_col), "mi_vs_pearson.png")
        save(outlier_boxplot(df, [c for c in numeric_cols if c != target_col]), "outliers.png")

        findings = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "target_col": target_col,
            "target_type": target_type,
            "missing_cells": missing_cells,
            "missing_pct_overall": missing_pct_overall,
            "duplicate_rows": duplicate_rows,
            "leakage_candidates": leakage,
            "high_cardinality": high_card,
            "near_constant": near_const,
            "redundant_pairs": redundant,
        }
        findings_path = plots_dir / "findings.json"
        findings_path.write_text(json.dumps(findings, indent=2, default=str))
        mlflow.log_artifact(str(findings_path), artifact_path="report")

        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        # --- summary ---
        print(f"run_id:        {run.info.run_id}")
        print(f"experiment:    {args.experiment}")
        print()
        print(f"Shape:         {len(df)} rows × {len(df.columns)} cols")
        print(f"Numeric:       {len(numeric_cols)} cols")
        print(f"Categorical:   {len(cat_cols)} cols")
        print(f"Target:        `{target_col}` → inferred type = {target_type}")
        print(f"Missing:       {missing_cells} cells ({missing_pct_overall:.1%})")
        print(f"Duplicates:    {duplicate_rows} rows")
        print()
        print("=" * 60)
        print("FINDINGS")
        print("=" * 60)
        if leakage:
            print(f"\n⚠  TARGET LEAKAGE ({len(leakage)}):")
            for item in leakage:
                print(f"   {item['feature']}: Pearson = {item['pearson']:+.4f}")
        if high_card:
            print(f"\n⚠  HIGH CARDINALITY ({len(high_card)}):")
            for item in high_card:
                print(f"   {item['feature']}: {item['n_unique']} unique values out of {item['n_rows']} rows")
        if near_const:
            print(f"\n⚠  NEAR-CONSTANT ({len(near_const)}):")
            for item in near_const:
                print(f"   {item['feature']}: top value covers {item['top_value_freq']:.1%}")
        if redundant:
            print(f"\n⚠  REDUNDANT PAIRS ({len(redundant)}):")
            for item in redundant:
                print(f"   {item['pair'][0]} ↔ {item['pair'][1]}: Pearson = {item['pearson']:.4f}")

        # Mutual information vs Pearson — surface the gap
        print("\n📊 MI vs Pearson (top 5 features by MI):")
        top_mi = mi_series.sort_values(ascending=False).head(5)
        for feat, mi_val in top_mi.items():
            pearson_val = float(pearson_corr.get(feat, 0.0))
            gap = "⚡ MI >> |Pearson|" if mi_val > 0.05 and pearson_val < 0.1 else ""
            print(f"   {feat}: MI={mi_val:.4f}, |Pearson|={pearson_val:.4f}  {gap}")


if __name__ == "__main__":
    main()
