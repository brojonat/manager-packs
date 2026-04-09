"""Marimo notebook: load the PyMC posterior and explore conjugate priors interactively.

Run with:
    marimo edit notebooks/coin_flip_demo.py
"""

import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")


@app.cell
def imports():
    import json
    from pathlib import Path

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import mlflow
    import numpy as np
    import pandas as pd
    from scipy.stats import beta as beta_dist

    return Path, az, beta_dist, json, mlflow, mo, np, plt


@app.cell
def title(mo):
    mo.md(r"""
    # Fair coin via Beta-Binomial (PyMC)

    This notebook loads the posterior fit by `src/train.py` and explores it
    interactively. The Beta-Binomial model is conjugate, so given a Beta(α, β)
    prior and `k` heads in `n` flips, the posterior is:

    $$p \mid \text{data} \sim \text{Beta}(\alpha + k,\ \beta + n - k)$$

    Closed form means we can let you slide the prior around and see the
    posterior update **instantly**, without re-running NUTS.

    This is the Bayesian counterpart to `template-sklearn-pipeline`. The
    sklearn version gives you a point estimate of `p`; this version gives
    you a **distribution** over `p`, plus a direct answer to "what's the
    probability the coin is fair?"
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
    """Find the most recent run from the coin-flip-pymc experiment."""
    experiment = mlflow.get_experiment_by_name("coin-flip-pymc")
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
                "metrics.posterior_mean",
                "metrics.posterior_sd",
                "metrics.hdi_94_low",
                "metrics.hdi_94_high",
                "metrics.rhat",
                "metrics.prob_fair_within_tol",
                "metrics.p_recovery_error",
                "tags.true_p",
                "params.n_flips",
                "params.n_heads",
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
def load_idata(az, latest_run_id, mlflow):
    if latest_run_id is None:
        loaded_idata = None
    else:
        idata_client = mlflow.tracking.MlflowClient()
        idata_path = idata_client.download_artifacts(latest_run_id, "posterior/idata.nc")
        loaded_idata = az.from_netcdf(idata_path)
    return (loaded_idata,)


@app.cell
def load_run_facts(json, latest_run_id, mlflow):
    """Pull the data sidecar + run params so the conjugate update has the truth."""
    n_flips_value = None
    n_heads_value = None
    true_p_value = None
    if latest_run_id is not None:
        facts_client = mlflow.tracking.MlflowClient()
        run = facts_client.get_run(latest_run_id)
        n_flips_value = int(run.data.params.get("n_flips", 0))
        n_heads_value = int(run.data.params.get("n_heads", 0))
        try:
            sidecar_local_path = facts_client.download_artifacts(latest_run_id, "data/sidecar.json")
            sidecar_obj = json.loads(open(sidecar_local_path).read())
            true_p_value = sidecar_obj.get("ground_truth", {}).get("true_p")
        except Exception:
            true_p_value = None
    return n_flips_value, n_heads_value, true_p_value


@app.cell
def posterior_summary(az, loaded_idata, mo):
    if loaded_idata is None:
        summary_display = mo.md("_(no posterior loaded)_")
        post_mean_value = None
        hdi_low_value = None
        hdi_high_value = None
    else:
        summary_df = az.summary(loaded_idata, var_names=["p"])
        post_mean_value = float(summary_df.loc["p", "mean"])
        post_sd_value = float(summary_df.loc["p", "sd"])
        hdi_low_value = float(summary_df.loc["p", "hdi_3%"])
        hdi_high_value = float(summary_df.loc["p", "hdi_97%"])
        rhat_value = float(summary_df.loc["p", "r_hat"])
        ess_value = float(summary_df.loc["p", "ess_bulk"])
        summary_display = mo.md(
            f"""
    ## Posterior summary (from NUTS)

    | Quantity | Value |
    |---|---|
    | posterior mean | `{post_mean_value:.4f}` |
    | posterior sd | `{post_sd_value:.4f}` |
    | 94% HDI | `[{hdi_low_value:.4f}, {hdi_high_value:.4f}]` |
    | R-hat | `{rhat_value:.4f}` |
    | ESS bulk | `{ess_value:.0f}` |
    """
        )
    summary_display
    return


@app.cell
def slider_section(mo):
    mo.md("""
    ## Conjugate prior playground

    Beta-Binomial is conjugate, so the posterior is closed-form. Slide the
    prior parameters around and watch the posterior update instantly — no
    re-sampling.

    A `Beta(1, 1)` prior is uniform on `[0, 1]` (i.e., "I have no idea").
    Larger values concentrate the prior; equal `α` and `β` center it on 0.5.
    """)
    return


@app.cell
def prior_sliders(mo):
    prior_alpha_slider = mo.ui.slider(
        start=0.5, stop=20.0, step=0.5, value=1.0, label="prior α"
    )
    prior_beta_slider = mo.ui.slider(
        start=0.5, stop=20.0, step=0.5, value=1.0, label="prior β"
    )
    fair_tol_slider = mo.ui.slider(
        start=0.01, stop=0.20, step=0.01, value=0.05, label="fairness tolerance ±"
    )
    sliders_widget = mo.vstack([prior_alpha_slider, prior_beta_slider, fair_tol_slider])
    sliders_widget
    return fair_tol_slider, prior_alpha_slider, prior_beta_slider


@app.cell
def conjugate_plot(
    beta_dist,
    fair_tol_slider,
    mo,
    n_flips_value,
    n_heads_value,
    np,
    plt,
    prior_alpha_slider,
    prior_beta_slider,
    true_p_value,
):
    if n_flips_value is None or n_heads_value is None:
        conjugate_display = mo.md("_(no run loaded)_")
        post_alpha_value = None
        post_beta_value = None
        prob_fair_value = None
    else:
        prior_alpha_value = float(prior_alpha_slider.value)
        prior_beta_value = float(prior_beta_slider.value)
        post_alpha_value = prior_alpha_value + n_heads_value
        post_beta_value = prior_beta_value + (n_flips_value - n_heads_value)

        x_grid = np.linspace(0.0, 1.0, 500)
        prior_pdf_values = beta_dist.pdf(x_grid, prior_alpha_value, prior_beta_value)
        post_pdf_values = beta_dist.pdf(x_grid, post_alpha_value, post_beta_value)

        fig_conj, ax_conj = plt.subplots(figsize=(7, 4))
        ax_conj.fill_between(
            x_grid, prior_pdf_values, alpha=0.3, color="#888888",
            label=f"prior Beta({prior_alpha_value:.1f}, {prior_beta_value:.1f})",
        )
        ax_conj.fill_between(
            x_grid, post_pdf_values, alpha=0.5, color="#4477aa",
            label=f"posterior Beta({post_alpha_value:.1f}, {post_beta_value:.1f})",
        )
        ax_conj.plot(x_grid, post_pdf_values, color="#4477aa", lw=2)
        if true_p_value is not None:
            ax_conj.axvline(true_p_value, color="red", lw=2, ls="--", label=f"true p={true_p_value}")

        # Shade the 'fair' region
        fair_tol_value = float(fair_tol_slider.value)
        ax_conj.axvspan(0.5 - fair_tol_value, 0.5 + fair_tol_value, alpha=0.1, color="green",
                        label=f"'fair' = [{0.5 - fair_tol_value:.2f}, {0.5 + fair_tol_value:.2f}]")

        ax_conj.set_xlabel("p (heads probability)")
        ax_conj.set_ylabel("density")
        ax_conj.set_title(f"{n_heads_value} heads in {n_flips_value} flips")
        ax_conj.set_xlim(0, 1)
        ax_conj.legend(loc="best", fontsize=9)
        fig_conj.tight_layout()

        # P(p in [0.5-tol, 0.5+tol]) using the Beta CDF
        prob_fair_value = float(
            beta_dist.cdf(0.5 + fair_tol_value, post_alpha_value, post_beta_value)
            - beta_dist.cdf(0.5 - fair_tol_value, post_alpha_value, post_beta_value)
        )

        conjugate_display = mo.vstack(
            [
                mo.as_html(fig_conj),
                mo.md(
                    f"### P(coin is fair within ±{fair_tol_value:.2f}) = **`{prob_fair_value:.4f}`**"
                ),
            ]
        )
    conjugate_display
    return


@app.cell
def explainer(mo):
    mo.md(r"""
    ## Why this is the right answer

    The closed-form conjugate update is **literally** the same answer that
    NUTS gave you in `train.py` (modulo Monte Carlo noise of a few decimal
    places). NUTS is overkill for a Beta-Binomial — but the same code shape
    works for any model PyMC can express, conjugate or not.

    So the workflow is:

    1. **Define the model in PyMC** — works for non-conjugate models too.
    2. **Sample with NUTS, log idata + diagnostics to MLflow** — the
       reference plumbing.
    3. **Reload the posterior anywhere** via `arviz.from_netcdf` — no need
       for the original training code.
    4. **Answer decision-relevant questions directly** from posterior
       samples (e.g. P(fair), P(better than baseline), expected loss under
       a utility function).

    Compare this to the sklearn template, which only ever gives you point
    estimates and Wald-style confidence intervals. Both have their place;
    the contrast is the selling point.
    """)
    return


if __name__ == "__main__":
    app.run()
