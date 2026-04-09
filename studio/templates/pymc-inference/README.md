# template-pymc-inference

The minimal end-to-end PyMC project that every Bayesian bundle copies.
The point isn't the model — it's the **plumbing**.

This template is the Bayesian counterpart to `template-sklearn-pipeline`.
Both consume `studio/data/coin-flip.parquet`. The sklearn version fits a
logistic regression and produces a point estimate; the PyMC version
fits a Beta-Binomial conjugate model and produces a full posterior over
the bias `p`. The two bundles answer the same question two different
ways, and that comparison is the selling point.

## What this verifies

- Project layout: `src/` for code, `notebooks/` for marimo, `mlruns/` for tracking
- PyMC model definition, NUTS sampling, ArviZ diagnostics
- MLflow logging of:
  - **Params**: priors, n_draws, n_tune, n_chains, seed, data hash
  - **Metrics**: posterior mean/std, 95% HDI, R-hat, ESS, posterior probability of fairness, recovery error against ground truth
  - **Tags**: data hash, true_p
  - **Artifacts**: `idata.nc` (NetCDF posterior), trace plot, posterior plot, prior-vs-posterior plot, sidecar JSON
- Posterior reload via `arviz.from_netcdf("idata.nc")` from anywhere
- A marimo notebook that loads the posterior and demonstrates conjugate
  prior updating with an interactive slider

## The worked example: fair coin

The dataset is `studio/data/coin-flip.parquet` from `datagen coin-flip`.
We model:

$$p \sim \text{Beta}(\alpha, \beta)$$
$$\text{flips} \sim \text{Binomial}(n, p)$$

The posterior over `p` is what we report. For a Uniform prior
($\alpha=\beta=1$) and `k` heads in `n` flips, the posterior is
$\text{Beta}(\alpha + k, \beta + n - k)$ in closed form — the marimo
notebook uses this for instant interactive prior tweaking.

## Run it

```bash
# 1. Generate data (if you haven't already)
datagen coin-flip --n 500 --p 0.7 --seed 42

# 2. Fit and log to MLflow
cd studio/templates/pymc-inference
python src/train.py

# 3. Inspect via the MLflow UI (optional)
mlflow ui --backend-store-uri ./mlruns

# 4. Load the posterior and answer questions
python src/predict.py --run-id <run_id>

# 5. Open the marimo demo notebook
marimo edit notebooks/coin_flip_demo.py
```

## Layout

```
pymc-inference/
├── README.md
├── SKILL.md              # what an agent learns from this template
├── src/
│   ├── train.py          # PyMC fit + ArviZ diagnostics + MLflow logging
│   ├── predict.py        # load idata, summarize, answer "is it fair?"
│   └── plots.py          # plot helpers (logged as MLflow artifacts)
├── notebooks/
│   └── coin_flip_demo.py # marimo notebook with conjugate prior slider
└── mlruns/               # MLflow tracking store (gitignored)
```
