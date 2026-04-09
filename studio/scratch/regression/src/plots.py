"""Plot helpers for the regression scratch project."""

import matplotlib.pyplot as plt
import numpy as np
import shap
from scipy import stats


def predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: np.ndarray | None = None,
    y_upper: np.ndarray | None = None,
) -> plt.Figure:
    """Scatter of predicted vs actual with optional prediction-interval shading."""
    fig, ax = plt.subplots(figsize=(7, 6))
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))

    if y_lower is not None and y_upper is not None:
        # Sort by actual to draw interval bands
        order = np.argsort(y_true)
        ax.fill_between(
            y_true[order],
            y_lower[order],
            y_upper[order],
            alpha=0.2,
            color="#4477aa",
            label="prediction interval",
        )

    ax.scatter(y_true, y_pred, alpha=0.5, s=15, color="#222222", label="point prediction")
    ax.plot([lo, hi], [lo, hi], color="red", lw=1.5, ls="--", label="y = x")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Predicted vs actual")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def residual_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Three-panel residual diagnostic: residual vs predicted, histogram, QQ plot."""
    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    ax_rvp, ax_hist, ax_qq = axes

    # Residual vs predicted
    ax_rvp.scatter(y_pred, residuals, alpha=0.5, s=15)
    ax_rvp.axhline(0, color="red", lw=1, ls="--")
    ax_rvp.set_xlabel("predicted")
    ax_rvp.set_ylabel("residual (actual - predicted)")
    ax_rvp.set_title("Residual vs predicted")
    ax_rvp.text(
        0.02,
        0.98,
        "Should be flat band around 0.\nFunnel = heteroscedasticity.\nCurve = missing non-linearity.",
        transform=ax_rvp.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="grey",
    )

    # Residual histogram
    ax_hist.hist(residuals, bins=40, color="#4477aa", alpha=0.7, density=True)
    res_std = float(residuals.std())
    res_mean = float(residuals.mean())
    xs = np.linspace(residuals.min(), residuals.max(), 200)
    ax_hist.plot(
        xs,
        stats.norm.pdf(xs, loc=res_mean, scale=res_std),
        color="red",
        lw=2,
        label=f"N({res_mean:.2f}, {res_std:.2f})",
    )
    ax_hist.set_xlabel("residual")
    ax_hist.set_ylabel("density")
    ax_hist.set_title("Residual distribution")
    ax_hist.legend(loc="best")

    # QQ plot
    stats.probplot(residuals, dist="norm", plot=ax_qq)
    ax_qq.set_title("QQ plot vs Normal")
    ax_qq.get_lines()[0].set_markersize(4)
    ax_qq.get_lines()[1].set_color("red")

    fig.tight_layout()
    return fig


def interval_coverage_plot(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    nominal_coverage: float,
) -> tuple[plt.Figure, float]:
    """Sort by actual, plot the prediction interval band, color points by inside/outside.

    Returns the figure and the empirical coverage (fraction inside the interval).
    """
    inside = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = float(inside.mean())

    order = np.argsort(y_true)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(
        np.arange(len(y_true)),
        y_lower[order],
        y_upper[order],
        alpha=0.25,
        color="#4477aa",
        label=f"prediction interval (nominal {nominal_coverage:.0%})",
    )
    ax.scatter(
        np.arange(len(y_true))[inside[order]],
        y_true[order][inside[order]],
        s=10,
        color="#228833",
        label=f"inside ({inside.sum()})",
    )
    ax.scatter(
        np.arange(len(y_true))[~inside[order]],
        y_true[order][~inside[order]],
        s=15,
        color="#cc3311",
        label=f"outside ({(~inside).sum()})",
    )
    ax.set_xlabel("test sample (sorted by true value)")
    ax.set_ylabel("y")
    ax.set_title(
        f"Interval coverage: {coverage:.1%} empirical vs {nominal_coverage:.0%} nominal"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, coverage


def shap_summary(model, X_sample, feature_names: list[str]) -> plt.Figure:
    """Global SHAP summary (beeswarm) for an XGBRegressor."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
