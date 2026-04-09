"""Marimo notebook: load the trained coin-flip model and explore it interactively.

Run with:
    marimo edit notebooks/coin_flip_demo.py
or for app mode (hidden code):
    marimo run notebooks/coin_flip_demo.py
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import mlflow
    import mlflow.sklearn
    import numpy as np
    import pandas as pd
    from scipy.special import expit

    return Path, expit, json, mlflow, mo, pd


@app.cell
def title(mo):
    mo.md(r"""
    # Fair coin via logistic regression

    This notebook loads the model trained by `src/train.py` from the local
    MLflow store and explores it interactively. The model is a logistic
    regression on `flip_index → outcome`, so:

    - The **intercept** (in logit space) maps to $P(\text{heads})$ at the
      mean of the indices via $\sigma(\text{intercept})$.
    - The **slope on `flip_index`** is a non-stationarity detector. For a
      fair stationary coin it should be ≈ 0; for a drifting coin it
      captures the drift.
    """)
    return


@app.cell
def setup_mlflow(Path, mlflow):
    template_dir = Path(__file__).resolve().parent.parent
    tracking_uri = f"file:{template_dir / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    return


@app.cell
def list_runs(mlflow):
    """Find the most recent run from the coin-flip-sklearn experiment."""
    experiment = mlflow.get_experiment_by_name("coin-flip-sklearn")
    if experiment is None:
        latest_run_id = None
        runs_df = None
    else:
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10,
        )
        latest_run_id = runs_df.iloc[0]["run_id"] if len(runs_df) else None
    return latest_run_id, runs_df


@app.cell
def show_runs(latest_run_id, mo, runs_df):
    if latest_run_id is None:
        runs_display = mo.md(
            "**No runs found.** Run `python src/train.py` from the template "
            "directory first."
        )
    else:
        runs_cols = [
            c
            for c in [
                "run_id",
                "metrics.cv_log_loss_mean",
                "metrics.intercept_logit",
                "metrics.coef_flip_index_standardized",
                "metrics.p_at_mean_index",
                "metrics.p_recovery_error",
                "tags.true_p",
                "tags.drift",
                "start_time",
            ]
            if c in runs_df.columns
        ]
        runs_display = mo.vstack(
            [
                mo.md(f"**Latest run:** `{latest_run_id}`"),
                mo.ui.table(runs_df[runs_cols]),
            ]
        )
    runs_display
    return


@app.cell
def load_model(latest_run_id, mlflow):
    if latest_run_id is None:
        loaded_model = None
    else:
        loaded_model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
    return (loaded_model,)


@app.cell
def coefficients(expit, loaded_model, mo):
    if loaded_model is None:
        intercept = None
        coef_std = None
        p_at_mean = None
        coef_display = mo.md("_(no model loaded)_")
    else:
        clf = loaded_model.named_steps["clf"]
        intercept = float(clf.intercept_[0])
        coef_std = float(clf.coef_[0][0])
        p_at_mean = float(expit(intercept))
        coef_display = mo.md(
            f"""
    ## Coefficients (recovered from model)

    | Quantity | Value |
    |---|---|
    | intercept (logit space) | `{intercept:+.4f}` |
    | coef on standardized `flip_index` | `{coef_std:+.4f}` |
    | $P(\\text{{heads}})$ at mean index | **`{p_at_mean:.4f}`** |
    """
        )
    coef_display
    return coef_std, p_at_mean


@app.cell
def slider_section(mo):
    mo.md("""
    ## Predict the next flip

    Pick a `flip_index` and see what the model predicts.
    """)
    return


@app.cell
def slider(mo):
    flip_idx_slider = mo.ui.slider(
        start=0,
        stop=1000,
        step=1,
        value=0,
        label="flip_index",
        full_width=True,
    )
    flip_idx_slider
    return (flip_idx_slider,)


@app.cell
def predict_at_idx(flip_idx_slider, loaded_model, mo, pd):
    if loaded_model is None:
        proba_heads = None
        prediction_display = mo.md("_(no model loaded)_")
    else:
        idx_value = int(flip_idx_slider.value)
        prediction_input = pd.DataFrame({"flip_index": [idx_value]})
        proba_pair = loaded_model.predict_proba(prediction_input)[0]
        proba_heads = float(proba_pair[1])
        prediction_display = mo.md(
            f"""
    For **`flip_index = {idx_value}`**:

    $$P(\\text{{heads}}) = {proba_heads:.4f}$$

    $$P(\\text{{tails}}) = {(1 - proba_heads):.4f}$$
    """
        )
    prediction_display
    return


@app.cell
def comparison_to_truth(coef_std, json, latest_run_id, mlflow, mo, p_at_mean):
    """If the run has a sidecar with ground truth, show the comparison."""
    truth_display = mo.md("_(no run loaded)_")
    if latest_run_id is not None and p_at_mean is not None:
        client = mlflow.tracking.MlflowClient()
        sidecar_truth = None
        try:
            local_path = client.download_artifacts(latest_run_id, "data/sidecar.json")
            sidecar_obj = json.loads(open(local_path).read())
            sidecar_truth = sidecar_obj.get("ground_truth", {})
        except Exception:
            sidecar_truth = None

        if sidecar_truth:
            true_p_value = sidecar_truth.get("true_p")
            drift_value = sidecar_truth.get("drift", 0.0)
            error_value = (
                abs(p_at_mean - true_p_value) if true_p_value is not None else None
            )
            error_str = f"{error_value:.4f}" if error_value is not None else "—"
            truth_display = mo.md(
                f"""
    ## Recovery vs ground truth

    | Quantity | Truth | Recovered |
    |---|---|---|
    | `true_p` | `{true_p_value}` | `{p_at_mean:.4f}` |
    | `drift` | `{drift_value}` | slope=`{coef_std:+.4f}` |
    | `\\|error\\|` | — | `{error_str}` |

    For a stationary coin, the slope should be ≈ 0.
    For a drifting coin, the slope captures the drift direction.
    """
            )
        else:
            truth_display = mo.md("_(no sidecar artifact found for this run)_")
    truth_display
    return


if __name__ == "__main__":
    app.run()
