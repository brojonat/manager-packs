"""Train an XGBoost regressor on the Friedman1 dataset, log to MLflow.

Specializes the sklearn-pipeline template with the things that make
tabular regression actually useful in practice:

- XGBoost as the default tabular regressor (beats linear on non-linear
  structure like Friedman1's sin/quadratic terms)
- **Quantile regression** for prediction intervals — fit q=0.10 and
  q=0.90 models alongside the point model to get an 80% interval per
  prediction, no Gaussian assumptions required
- **Interval coverage** validation: empirical % of test points inside
  the interval should approximate the nominal level
- Residual diagnostics (residual vs predicted, histogram, QQ plot)
- SHAP feature importance + recovery check against known informative
  vs noise features

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "friedman.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"

sys.path.insert(0, str(THIS_DIR))
from plots import (  # noqa: E402
    interval_coverage_plot,
    predicted_vs_actual,
    residual_diagnostics,
    shap_summary,
)


def data_hash(df: pd.DataFrame) -> str:
    """Stable short hash of the materialized dataframe."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def build_xgb_regressor(
    feature_cols: list[str],
    seed: int,
    objective: str = "reg:squarederror",
    quantile_alpha: float | None = None,
) -> Pipeline:
    """XGBRegressor inside a sklearn Pipeline.

    Set `objective="reg:quantileerror"` and pass a `quantile_alpha` in
    (0, 1) to fit a quantile regression for that quantile. The point
    estimator (default) uses squared error.
    """
    kwargs = dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective=objective,
        random_state=seed,
        n_jobs=-1,
    )
    if quantile_alpha is not None:
        kwargs["quantile_alpha"] = quantile_alpha

    return Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[("num", StandardScaler(), feature_cols)],
                    remainder="drop",
                ),
            ),
            ("clf", XGBRegressor(**kwargs)),
        ]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--lower-quantile", type=float, default=0.10)
    parser.add_argument("--upper-quantile", type=float, default=0.90)
    parser.add_argument("--experiment", default="regression-xgb")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data not found at {args.data}. Run `datagen friedman` first.")

    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    truth = sidecar.get("ground_truth", {})
    irreducible_rmse = float(truth.get("irreducible_rmse", float("nan")))
    informative_features = set(truth.get("informative_features", []))

    # --- ibis: load + materialize for sklearn ---
    table = ibis.duckdb.connect().read_parquet(str(args.data))
    feature_cols = [c for c in table.columns if c.startswith("feature_")]

    target_stats = (
        table
        .aggregate(
            target_mean=table.target.mean(),
            target_std=table.target.std(),
            n_total=table.count(),
        )
        .execute()
        .iloc[0]
    )

    data = (
        table
        .select(*feature_cols, "target")
        .execute()
    )
    X = data[feature_cols]
    y = data["target"]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    # Split a calibration set off the training data for conformal prediction
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=args.seed
    )

    nominal_coverage = args.upper_quantile - args.lower_quantile

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        # --- params ---
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "n_rows": len(data),
                "n_features": len(feature_cols),
                "target_mean": round(float(target_stats["target_mean"]), 4),
                "target_std": round(float(target_stats["target_std"]), 4),
                "seed": args.seed,
                "test_size": args.test_size,
                "model": "XGBRegressor",
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "lower_quantile": args.lower_quantile,
                "upper_quantile": args.upper_quantile,
                "nominal_coverage": nominal_coverage,
            }
        )

        # --- tags ---
        mlflow.set_tag("data_hash", data_hash(data))
        for k in ("noise_std", "function_form", "n_informative"):
            if k in truth:
                mlflow.set_tag(f"truth.{k}", str(truth[k]))

        # --- fit point model + two quantile models ---
        xgb_point = build_xgb_regressor(feature_cols, args.seed)
        xgb_point.fit(X_train, y_train)

        xgb_lower = build_xgb_regressor(
            feature_cols, args.seed,
            objective="reg:quantileerror",
            quantile_alpha=args.lower_quantile,
        )
        xgb_lower.fit(X_train, y_train)

        xgb_upper = build_xgb_regressor(
            feature_cols, args.seed,
            objective="reg:quantileerror",
            quantile_alpha=args.upper_quantile,
        )
        xgb_upper.fit(X_train, y_train)

        # --- conformal calibration ---
        # On the held-out calibration set, compute conformity scores:
        #   E_i = max(q_low(x_i) - y_i,  y_i - q_high(x_i))
        # E_i > 0 when y_i is OUTSIDE the predicted interval. The right
        # quantile of E gives the additive correction that achieves
        # marginal coverage on test data.
        cal_low = xgb_lower.predict(X_calib)
        cal_high = xgb_upper.predict(X_calib)
        conformity = np.maximum(cal_low - y_calib.to_numpy(), y_calib.to_numpy() - cal_high)
        # Quantile level for finite-sample coverage guarantee
        n_cal = len(y_calib)
        q_level = min(1.0, np.ceil((nominal_coverage) * (n_cal + 1)) / n_cal)
        conformal_q = float(np.quantile(conformity, q_level))

        # --- predictions ---
        y_pred = xgb_point.predict(X_test)
        y_low_raw = xgb_lower.predict(X_test)
        y_high_raw = xgb_upper.predict(X_test)
        # Conformalized: expand the interval by q on both sides
        y_low = y_low_raw - conformal_q
        y_high = y_high_raw + conformal_q

        # --- metrics ---
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_metric("test_rmse", rmse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)

        if not np.isnan(irreducible_rmse):
            mlflow.log_metric("irreducible_rmse", irreducible_rmse)
            mlflow.log_metric("rmse_above_irreducible", rmse - irreducible_rmse)

        # Empirical interval coverage on test (raw and conformalized)
        y_test_arr = y_test.to_numpy()
        inside_raw = (y_test_arr >= y_low_raw) & (y_test_arr <= y_high_raw)
        inside = (y_test_arr >= y_low) & (y_test_arr <= y_high)
        coverage_raw = float(inside_raw.mean())
        coverage_conformal = float(inside.mean())

        mlflow.log_metric("conformal_q", conformal_q)
        mlflow.log_metric("coverage_raw", coverage_raw)
        mlflow.log_metric("coverage_conformal", coverage_conformal)
        mlflow.log_metric("coverage_error_raw", abs(coverage_raw - nominal_coverage))
        mlflow.log_metric("coverage_error_conformal", abs(coverage_conformal - nominal_coverage))
        mlflow.log_metric("interval_width_raw", float((y_high_raw - y_low_raw).mean()))
        mlflow.log_metric("interval_width_conformal", float((y_high - y_low).mean()))

        # --- log models ---
        mlflow.sklearn.log_model(sk_model=xgb_point, name="model_point", input_example=X_train.head(5))
        mlflow.sklearn.log_model(sk_model=xgb_lower, name="model_lower", input_example=X_train.head(5))
        mlflow.sklearn.log_model(sk_model=xgb_upper, name="model_upper", input_example=X_train.head(5))

        # --- plots ---
        plots_dir = TEMPLATE_DIR / "_tmp_plots"
        plots_dir.mkdir(exist_ok=True)

        f_pva = predicted_vs_actual(y_test.to_numpy(), y_pred, y_low, y_high)
        f_pva.savefig(plots_dir / "predicted_vs_actual.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "predicted_vs_actual.png"), artifact_path="plots")

        f_res = residual_diagnostics(y_test.to_numpy(), y_pred)
        f_res.savefig(plots_dir / "residual_diagnostics.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "residual_diagnostics.png"), artifact_path="plots")

        f_cov, _ = interval_coverage_plot(y_test_arr, y_low, y_high, nominal_coverage)
        f_cov.savefig(plots_dir / "interval_coverage.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "interval_coverage.png"), artifact_path="plots")

        f_cov_raw, _ = interval_coverage_plot(y_test_arr, y_low_raw, y_high_raw, nominal_coverage)
        f_cov_raw.savefig(plots_dir / "interval_coverage_raw.png", dpi=120)
        mlflow.log_artifact(str(plots_dir / "interval_coverage_raw.png"), artifact_path="plots")

        # SHAP on the point model
        preprocessor = xgb_point.named_steps["preprocess"]
        clf = xgb_point.named_steps["clf"]
        X_test_t = preprocessor.transform(X_test.iloc[:200])
        f_shap = shap_summary(clf, X_test_t, feature_cols)
        f_shap.savefig(plots_dir / "shap_summary.png", dpi=120, bbox_inches="tight")
        mlflow.log_artifact(str(plots_dir / "shap_summary.png"), artifact_path="plots")

        # --- sidecar artifact ---
        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        # --- summary ---
        print(f"run_id:           {run.info.run_id}")
        print(f"experiment:       {args.experiment}")
        print(f"target std:       {float(target_stats['target_std']):.4f}")
        print(f"test RMSE:        {rmse:.4f}")
        if not np.isnan(irreducible_rmse):
            print(f"  irreducible:    {irreducible_rmse:.4f}  (best possible)")
            print(f"  excess:         {rmse - irreducible_rmse:+.4f}")
        print(f"test MAE:         {mae:.4f}")
        print(f"test R²:          {r2:.4f}")
        print()
        print(f"Prediction interval [{args.lower_quantile}, {args.upper_quantile}]:")
        print(f"  nominal coverage:        {nominal_coverage:.0%}")
        print(f"  raw empirical coverage:  {coverage_raw:.1%}  ({inside_raw.sum()}/{len(inside_raw)})")
        print(f"  conformal q correction:  ±{conformal_q:.4f}")
        print(f"  conformal coverage:      {coverage_conformal:.1%}  ({inside.sum()}/{len(inside)})")
        print(f"  interval width raw:      {(y_high_raw - y_low_raw).mean():.4f}")
        print(f"  interval width conformal:{(y_high - y_low).mean():.4f}")


if __name__ == "__main__":
    main()
