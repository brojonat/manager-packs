"""Plot helpers — produce matplotlib figures we log as MLflow artifacts."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def empirical_vs_predicted(
    df: pd.DataFrame,
    pred_proba: np.ndarray,
    window: int = 25,
) -> plt.Figure:
    """Sliding-window empirical P(heads) vs the model's predicted P.

    For a fair stationary coin with index as the only feature, both lines
    should hover around `true_p` and the model line should be nearly flat.
    For a drifting coin, the empirical line will trend and the model will
    track it (revealing whether the slope on index picked up the drift).
    """
    n = len(df)
    indices = df["flip_index"].to_numpy()
    outcomes = df["outcome"].to_numpy()

    rolling = np.convolve(outcomes, np.ones(window) / window, mode="valid")
    rolling_idx = indices[window - 1 :]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(rolling_idx, rolling, label=f"empirical (window={window})", lw=2)
    ax.plot(indices, pred_proba, label="model P(heads)", lw=2, ls="--")
    ax.axhline(0.5, color="grey", lw=1, alpha=0.5, label="0.5")
    ax.set_xlabel("flip_index")
    ax.set_ylabel("P(heads)")
    ax.set_title("Empirical vs predicted P(heads) over the sequence")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def calibration_plot(y_true: np.ndarray, pred_proba: np.ndarray) -> plt.Figure:
    """Reliability diagram. For a coin this is mostly a sanity check."""
    fig, ax = plt.subplots(figsize=(5, 5))
    frac_pos, mean_pred = calibration_curve(y_true, pred_proba, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect")
    ax.plot(mean_pred, frac_pos, marker="o", lw=2, label="model")
    ax.set_xlabel("mean predicted P(heads)")
    ax.set_ylabel("fraction of heads observed")
    ax.set_title("Calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def coefficient_plot(intercept: float, coef: float, feature_name: str) -> plt.Figure:
    """Bar of intercept + coefficient on the standardized feature."""
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(["intercept", f"coef({feature_name})"], [intercept, coef], color=["#4477aa", "#ee6677"])
    ax.axvline(0, color="black", lw=1)
    for bar, val in zip(bars, [intercept, coef]):
        ax.text(
            val,
            bar.get_y() + bar.get_height() / 2,
            f"  {val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
        )
    ax.set_title("Logistic regression coefficients (standardized features)")
    fig.tight_layout()
    return fig
