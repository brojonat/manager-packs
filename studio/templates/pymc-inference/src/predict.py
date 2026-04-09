"""Load a posterior idata from MLflow and answer questions about p.

The PyMC counterpart to predict.py in the sklearn template. Reload
path: download the idata.nc artifact, load with ArviZ, summarize.
"""

import argparse
from pathlib import Path

import arviz as az
import mlflow

DEFAULT_TRACKING = Path(__file__).resolve().parent.parent / "mlruns"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="MLflow run ID from train.py output.")
    parser.add_argument("--fair-tolerance", type=float, default=0.05)
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")

    client = mlflow.tracking.MlflowClient()
    local_path = client.download_artifacts(args.run_id, "posterior/idata.nc")
    idata = az.from_netcdf(local_path)

    p_samples = idata.posterior["p"].to_numpy().flatten()
    summary = az.summary(idata, var_names=["p"])

    fair_lo = 0.5 - args.fair_tolerance
    fair_hi = 0.5 + args.fair_tolerance
    prob_fair = float(((p_samples >= fair_lo) & (p_samples <= fair_hi)).mean())

    print(f"posterior mean:   {summary.loc['p', 'mean']:.4f}")
    print(f"posterior sd:     {summary.loc['p', 'sd']:.4f}")
    print(f"94% HDI:          [{summary.loc['p', 'hdi_3%']:.4f}, {summary.loc['p', 'hdi_97%']:.4f}]")
    print(f"R-hat:            {summary.loc['p', 'r_hat']:.4f}")
    print()
    print(f"P(p∈[0.5±{args.fair_tolerance}]) = {prob_fair:.4f}")
    print("  → the 'is it fair?' answer")


if __name__ == "__main__":
    main()
