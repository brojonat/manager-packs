"""Fit a Beta-Binomial model to the coin-flip dataset, log to MLflow.

This is the reference template every Bayesian bundle copies. The model
is trivial (Beta prior on p, Binomial likelihood) but the plumbing —
PyMC sampling + ArviZ diagnostics + MLflow logging + idata persistence
— is the actual point.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import arviz as az
import mlflow
import numpy as np
import pandas as pd
import pymc as pm

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "coin-flip.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"

# So we can import plots.py without packaging
sys.path.insert(0, str(THIS_DIR))
from plots import posterior_plot, prior_vs_posterior, trace_plot  # noqa: E402


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def build_model(n: int, k: int, prior_alpha: float, prior_beta: float) -> pm.Model:
    """Beta prior on the bias, Binomial likelihood.

    Beta-Binomial is conjugate, so we could compute this in closed form
    (`Beta(alpha+k, beta+n-k)`). We use PyMC anyway as the reference
    pattern: real bundles will use models that aren't conjugate.

    Note: we use Binomial with sufficient statistics (n, k) instead of
    N independent Bernoulli observations. The likelihood is identical
    but the sampler is much faster — and it documents the right move
    when sufficient statistics exist.
    """
    with pm.Model() as model:
        p = pm.Beta("p", alpha=prior_alpha, beta=prior_beta)
        pm.Binomial("flips", n=n, p=p, observed=k)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--prior-beta", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fair-tolerance", type=float, default=0.05,
                        help="An absolute tolerance: 'fair' = p in [0.5-tol, 0.5+tol]")
    parser.add_argument("--experiment", default="coin-flip-pymc")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Data not found at {args.data}. Run `datagen coin-flip` first."
        )

    df = pd.read_parquet(args.data)
    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    truth = sidecar.get("ground_truth", {})
    true_p = truth.get("true_p")

    n_flips = int(len(df))
    n_heads = int(df["outcome"].sum())

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        # --- params ---
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "n_flips": n_flips,
                "n_heads": n_heads,
                "prior_alpha": args.prior_alpha,
                "prior_beta": args.prior_beta,
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "seed": args.seed,
                "fair_tolerance": args.fair_tolerance,
                "model": "Beta-Binomial",
                "sampler": "NUTS",
            }
        )

        # --- tags ---
        mlflow.set_tag("data_hash", data_hash(df))
        if true_p is not None:
            mlflow.set_tag("true_p", str(true_p))
        if "drift" in truth:
            mlflow.set_tag("drift", str(truth["drift"]))

        # --- fit ---
        model = build_model(n_flips, n_heads, args.prior_alpha, args.prior_beta)
        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                random_seed=args.seed,
                progressbar=False,
            )

        # --- diagnostics ---
        summary = az.summary(idata, var_names=["p"])
        post_mean = float(summary.loc["p", "mean"])
        post_sd = float(summary.loc["p", "sd"])
        hdi_low = float(summary.loc["p", "hdi_3%"])
        hdi_high = float(summary.loc["p", "hdi_97%"])
        rhat = float(summary.loc["p", "r_hat"])
        ess_bulk = float(summary.loc["p", "ess_bulk"])
        ess_tail = float(summary.loc["p", "ess_tail"])

        # Posterior probability that the coin is "fair" within tolerance
        p_samples = idata.posterior["p"].to_numpy().flatten()
        fair_lo = 0.5 - args.fair_tolerance
        fair_hi = 0.5 + args.fair_tolerance
        prob_fair = float(((p_samples >= fair_lo) & (p_samples <= fair_hi)).mean())

        mlflow.log_metric("posterior_mean", post_mean)
        mlflow.log_metric("posterior_sd", post_sd)
        mlflow.log_metric("hdi_94_low", hdi_low)
        mlflow.log_metric("hdi_94_high", hdi_high)
        mlflow.log_metric("hdi_94_width", hdi_high - hdi_low)
        mlflow.log_metric("rhat", rhat)
        mlflow.log_metric("ess_bulk", ess_bulk)
        mlflow.log_metric("ess_tail", ess_tail)
        mlflow.log_metric("prob_fair_within_tol", prob_fair)
        if true_p is not None:
            mlflow.log_metric("p_recovery_error", float(abs(post_mean - true_p)))
            mlflow.log_metric(
                "true_p_in_94_hdi",
                float(hdi_low <= true_p <= hdi_high),
            )

        # --- artifacts: idata + plots + sidecar ---
        plots_dir = TEMPLATE_DIR / "_tmp_plots"
        plots_dir.mkdir(exist_ok=True)

        idata_path = plots_dir / "idata.nc"
        idata.to_netcdf(idata_path)
        mlflow.log_artifact(str(idata_path), artifact_path="posterior")

        f1 = posterior_plot(idata, var_name="p", true_value=true_p)
        f1_path = plots_dir / "posterior.png"
        f1.savefig(f1_path, dpi=120)
        mlflow.log_artifact(str(f1_path), artifact_path="plots")

        f2 = trace_plot(idata, var_name="p")
        f2_path = plots_dir / "trace.png"
        f2.savefig(f2_path, dpi=120)
        mlflow.log_artifact(str(f2_path), artifact_path="plots")

        f3 = prior_vs_posterior(
            args.prior_alpha,
            args.prior_beta,
            n_flips,
            n_heads,
            true_value=true_p,
        )
        f3_path = plots_dir / "prior_vs_posterior.png"
        f3.savefig(f3_path, dpi=120)
        mlflow.log_artifact(str(f3_path), artifact_path="plots")

        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        print(f"run_id:           {run.info.run_id}")
        print(f"experiment:       {args.experiment}")
        print(f"posterior mean:   {post_mean:.4f}")
        print(f"posterior sd:     {post_sd:.4f}")
        print(f"94% HDI:          [{hdi_low:.4f}, {hdi_high:.4f}]")
        print(f"R-hat:            {rhat:.4f}")
        print(f"ESS (bulk/tail):  {ess_bulk:.0f} / {ess_tail:.0f}")
        print(f"P(p∈[0.5±{args.fair_tolerance}]) = {prob_fair:.4f}  (the 'is it fair?' answer)")
        if true_p is not None:
            print(f"true p:           {true_p}")
            print(f"|recovery error|: {abs(post_mean - true_p):.4f}")
            print(f"true p in 94% HDI: {hdi_low <= true_p <= hdi_high}")


if __name__ == "__main__":
    main()
