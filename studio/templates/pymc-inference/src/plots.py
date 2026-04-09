"""Plot helpers for the PyMC template — logged as MLflow artifacts."""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta as beta_dist


def posterior_plot(idata, var_name: str = "p", true_value: float | None = None) -> plt.Figure:
    """ArviZ posterior plot with optional ground-truth marker."""
    ax = az.plot_posterior(idata, var_names=[var_name], hdi_prob=0.95)
    fig = ax.figure if hasattr(ax, "figure") else plt.gcf()
    if true_value is not None:
        for axis in fig.axes:
            axis.axvline(true_value, color="red", lw=2, ls="--", label=f"true {var_name}={true_value}")
            axis.legend(loc="best")
    fig.tight_layout()
    return fig


def trace_plot(idata, var_name: str = "p") -> plt.Figure:
    """ArviZ trace plot for chain mixing diagnosis."""
    axes = az.plot_trace(idata, var_names=[var_name])
    fig = axes.ravel()[0].figure
    fig.tight_layout()
    return fig


def prior_vs_posterior(
    prior_alpha: float,
    prior_beta: float,
    n: int,
    k: int,
    true_value: float | None = None,
) -> plt.Figure:
    """Show the Beta prior and the conjugate Beta posterior on the same plot.

    Beta-Binomial is conjugate, so the posterior is closed-form:
        Beta(alpha + k, beta + n - k)

    This is the same answer NUTS would give you (modulo finite-sample
    Monte Carlo error) but instant — useful for interactive exploration.
    """
    post_alpha = prior_alpha + k
    post_beta = prior_beta + (n - k)

    x = np.linspace(0.0, 1.0, 500)
    prior_pdf = beta_dist.pdf(x, prior_alpha, prior_beta)
    post_pdf = beta_dist.pdf(x, post_alpha, post_beta)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(x, prior_pdf, alpha=0.3, color="#888888", label=f"prior Beta({prior_alpha}, {prior_beta})")
    ax.plot(x, prior_pdf, color="#888888", lw=1)
    ax.fill_between(x, post_pdf, alpha=0.5, color="#4477aa", label=f"posterior Beta({post_alpha}, {post_beta})")
    ax.plot(x, post_pdf, color="#4477aa", lw=2)

    if true_value is not None:
        ax.axvline(true_value, color="red", lw=2, ls="--", label=f"true p = {true_value}")

    ax.set_xlabel("p (heads probability)")
    ax.set_ylabel("density")
    ax.set_title(f"Prior vs posterior  ({k} heads in {n} flips)")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
