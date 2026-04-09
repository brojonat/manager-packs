# ManagerPack Content Roadmap

## Phase 0 — Foundational templates (verify the workflow)

Before building real content, build two minimal end-to-end projects that
exercise the full workflow we want every bundle to follow. Treat these
as templates: every later bundle copies their structure.

**The two tracks share the same problem and the same dataset:** "is this
coin fair?" Each flip is one row of `(flip_index, outcome)` tabular data.
The sklearn track fits a logistic regression on `index → outcome`; the
PyMC track fits a Beta-Binomial conjugate model. Same data, same
question, two methodologies. Pedagogically this also gives buyers a
direct comparison and a reason to buy both.

**Notebooks are always marimo.** Never Jupyter. We have the
`marimo-notebooks`, `anywidget`, and `wigglystuff` skills available for
exactly this purpose.

Each minimal project must verify:
- Project layout (`pyproject.toml`, `src/`, `notebooks/`, `artifacts/`)
- Reproducible deps (uv lock)
- MLflow logging: params, metrics, tags, artifacts (model + plots + data hashes)
- Artifact persistence (model + preprocessing + metadata travel together)
- **Marimo** notebook that loads the trained artifact and demonstrates inference
- A skill file (`SKILL.md`) that tells an agent how to instantiate this template

### [ ] Shared dataset: `coin-flips`

A single synthetic dataset both templates consume. Lives under
`bundles/_shared-data/coin-flips/` (or wherever data generation lands —
see Phase 4).

- [ ] Generator takes `(n_flips, true_p, seed)` and emits a parquet file
  with columns `flip_index`, `outcome`
- [ ] Sidecar JSON with `true_p`, `seed`, `n_flips` (ground truth for validation)
- [ ] Optional drift mode: generator can vary `p` linearly with index, so
  later we can show that the logistic-regression slope detects non-stationarity

### [x] `template-sklearn-pipeline` — fair coin via logistic regression

Lives at `studio/templates/sklearn-pipeline/`. Treats `(flip_index,
outcome)` as tabular data. Logistic regression on the index feature.
The point is *not* the model — it's the plumbing.

- [x] `Pipeline` with `ColumnTransformer` + `LogisticRegression`
- [x] Train/val/test split with fixed seed
- [x] Cross-validated metric (log-loss)
- [x] Coefficients interpreted: intercept = `logit(p)`, slope on index = drift signal
- [x] MLflow run: params, metrics, model, plots (empirical-vs-predicted,
  calibration, coefficients), data hash, sidecar artifact
- [x] Persist via `mlflow.sklearn.log_model` (NOT bare joblib — MLflow
  wraps the model with signature + conda env + requirements)
- [x] Reload + predict via `mlflow.sklearn.load_model` in `src/predict.py`
- [x] **Marimo** notebook (`notebooks/coin_flip_demo.py`) loads the
  latest run, shows recovered coefficients, and has a `mo.ui.slider`
  for interactive prediction. Validates clean with `marimo check`.
- [x] Recovery verified: `true_p=0.7 → recovered 0.71` (error 0.01)
- [x] Drift detection verified: `p=0.3, drift=0.4` produces strong
  positive slope on `flip_index`, correctly detecting non-stationarity
- [x] `SKILL.md` describing the template so an agent can reproduce it

### [x] `template-pymc-inference` — fair coin via Beta-Binomial

Lives at `studio/templates/pymc-inference/`. Same dataset as the sklearn
template, modeled as `p ~ Beta(α, β); flips ~ Binomial(n, p)` (sufficient
statistics; same likelihood as N Bernoulli but much faster). The PyMC
counterpart of the sklearn template.

- [x] PyMC model with NUTS sampling (4 chains, 2000 draws, 1000 tune)
- [x] ArviZ posterior summary, trace plot, posterior plot, prior-vs-posterior
- [x] Posterior probability that `p ∈ [0.5±tol]` (the "is it fair" answer)
- [x] MLflow run: priors, draws/tune/chains, posterior_mean/sd/HDI/R-hat/ESS,
  prob_fair_within_tol, p_recovery_error, true_p_in_94_hdi, sidecar artifact
- [x] Persist `idata` (NetCDF) as artifact under `posterior/idata.nc`
- [x] Reload via `arviz.from_netcdf` in `src/predict.py`
- [x] **Marimo** notebook (`notebooks/coin_flip_demo.py`) loads idata,
  shows posterior summary, has α/β/tolerance sliders that drive a
  closed-form **conjugate update** for instant prior exploration without
  re-sampling. Validates clean with `marimo check`.
- [x] Recovery verified: `true_p=0.7 → posterior mean 0.7090` (error 0.009);
  true value lands in 94% HDI; R-hat 1.000; ESS 3267
- [x] Decision output verified:
  - Biased coin (p=0.7): `P(fair within ±0.05) = 0.0000`
  - Fair coin (p=0.5): `P(fair within ±0.05) = 0.5981`
- [x] `SKILL.md` describing the template (with the comparison-to-sklearn
  selling point baked in)

---

## Phase 1 — Tabular DS bundles (canonical sklearn problems)

Each bundle copies `template-sklearn-pipeline` and specializes it.
**Default model: XGBoost** (see `feedback_xgboost_for_tabular.md` memory).

- [x] `tabular-eda` — profile a new dataset before modeling. Detects
  target leakage (>0.95 |Pearson| to target), high-cardinality
  categoricals (>50 unique → OHE explosion), near-constant features,
  redundant pairs, missing data per column, skewed distributions, and
  outliers (IQR-based). Pairs **mutual information vs Pearson** to
  catch non-linear signal Pearson misses (the Friedman1 lesson).
  Includes target type inference (binary/multiclass/regression) so the
  workflow ends with "what model do I train next?". Uses a new
  `datagen messy-binary` synthetic dataset with **7 planted issues**
  for the demo. Studio scratch at `studio/scratch/tabular-eda/`,
  bundle at `bundles/tabular-eda/`. Verified: profiler caught all 4
  flagged issues (leakage, high-card, near-const, redundant pair) and
  surfaced the rest in plots.
- [x] `binary-classification` — XGBoost with `scale_pos_weight`, threshold
  tuning, calibration verification (Brier + reliability diagram), SHAP
  feature importance, baseline LogisticRegression comparison.
  Studio scratch at `studio/scratch/binary-classification/`. Bundle at
  `bundles/binary-classification/` (manifest, SKILL.md, self-contained
  demo.py). Verified: ROC-AUC 0.969, PR-AUC 0.912 (vs 0.152 baseline),
  Brier 0.040, F1 @ tuned 0.84 vs F1 @ 0.5 = 0.82.
- [ ] `multiclass-classification` — OvR vs softmax, confusion matrices, per-class metrics
- [ ] `multilabel-classification` — label correlations, classifier chains, hamming loss
- [x] `regression` — XGBoost point estimator + **conformalized quantile
  regression** for prediction intervals that actually achieve nominal
  coverage (raw quantile XGBoost undercovers ~15-20pp on Friedman1; CQR
  fixes it). Plus residual diagnostics, SHAP, and a LinearRegression
  baseline that fails on Friedman1's `sin`/quadratic terms. Studio
  scratch at `studio/scratch/regression/`. Bundle at `bundles/regression/`.
  New `datagen friedman` subcommand added for the dataset. Verified:
  RMSE 1.30 vs irreducible 1.00 (excess 0.30); R² 0.93; conformal
  coverage 89% vs nominal 90% (raw was 72%).
- [x] `unsupervised` — KMeans / GMM / DBSCAN clustering with **K
  selection via silhouette** (not visual elbow), **stability checks**
  via pairwise ARI across random seeds, **IsolationForest** for
  anomaly detection, PCA for dimensionality reduction. Demo
  contrasts blobs (KMeans wins) vs moons (DBSCAN wins) so the
  algorithm-shape matching lesson is impossible to miss. Studio
  scratch at `studio/scratch/unsupervised/`, bundle at
  `bundles/unsupervised/`. Verified: K=4 chosen on blobs (true K=4),
  stability ARI = 1.000, KMeans/GMM ARI vs truth = 1.000,
  IsolationForest precision 0.97 / recall 1.00 on 30 planted
  outliers.

---

## Phase 2 — PyMC / Bayesian bundles

Each bundle copies `template-pymc-inference` and specializes it.

- [ ] `bayesian-ab-testing` — posterior of conversion-rate diff, expected loss
- [ ] `bayesian-bandits` — Thompson sampling, contextual bandits, regret
- [ ] `bayesian-regression` — hierarchical models, partial pooling
- [ ] `bayesian-mixture-models` — soft clustering with uncertainty
- [ ] `bayesian-decision-analysis` — utility functions, EVOI

---

## Phase 3 — Reward-driven sequence optimization

The "in-between" project. Specifically targeted at OnlyFans creators (and
similar pay-to-view content businesses): given a feed of items shown to
users in some random ordering, with a per-visit reward (dollars spent),
recommend the optimal ordering — including which **sequences** of items
are worth more than the sum of their parts.

### Problem

- **Inputs:** events of `(visit_id, ordering, per_item_clicks, total_revenue)`
- **Outputs:**
  - Marginal item value (which items drive revenue)
  - Pair / triple / quad sequence value (which adjacencies amplify revenue)
  - Recommended top-k orderings with credible intervals
- **Pitch:** "Run this experiment for two weeks and we'll tell you how to
  order your content."

### Approach (Bayesian, PyMC)

- Position-aware reward model: `revenue = f(items, positions, sequences)`
- Item-level latent value `β_i` + position bias `α_p` + sequence-bonus
  terms `γ_{ij}`, `γ_{ijk}` (with shrinkage to keep them honest)
- Treat each visit as an observation; fit with NUTS
- Decode top orderings by sampling from posterior (Thompson-style) and
  scoring permutations
- Bonus: contextual bandit version that updates priors as new data comes in

### Tasks

- [ ] Pick a click model variant (PBM vs cascade vs hybrid)
- [ ] Decide how to model sequence bonuses without combinatorial blowup
  (probably L1/horseshoe priors on pair/triple terms)
- [ ] Build a synthetic data generator (see Phase 4) that knows ground
  truth so we can validate
- [ ] PyMC model + ArviZ diagnostics
- [ ] Decoding step: sample from posterior, score top-k orderings
- [ ] Skill bundle + Marimo notebook walkthrough
- [ ] Evaluate against simpler baselines (raw CTR, position-corrected CTR)

---

## Phase 4 — Data generation (cross-cutting)

Every bundle needs a dataset. Real datasets are noisy and slow to
acquire; synthetic data with known ground truth is faster to iterate on
and lets us validate models against the truth they should recover.

Lives at `studio/datagen/` as a standalone CLI (`datagen <problem>`).
Each subcommand maps to one problem type, writes parquet + sidecar JSON
with ground-truth parameters. Defaults to `studio/data/<problem>.parquet`.

Each dataset:
- Is deterministic given a seed
- Saves to parquet (works with the parquet-analysis skill)
- Has a sidecar JSON with the ground-truth parameters used to generate it

### Implemented

- [x] `coin-flip` — Bernoulli sequence, optional linear drift
- [x] `binary-classification` — sklearn `make_classification`, 2 classes
- [x] `multiclass-classification` — sklearn `make_classification`, N classes
- [x] `multilabel-classification` — sklearn `make_multilabel_classification`
- [x] `regression` — sklearn `make_regression`, exposes true coefficients
- [x] `friedman` — sklearn `make_friedman1`, non-linear regression with
  known informative vs noise features (used by the `regression` bundle)
- [x] `messy-binary` — binary classification with 7 planted EDA
  issues (leakage, high-cardinality, near-constant, missing, skewed,
  outliers, redundant pair) for the `tabular-eda` bundle
- [x] `blobs` — sklearn `make_blobs` for clustering

### Still TODO

- [ ] **High-dim tabular**: low intrinsic dimensionality embedded in
  high-dim noise (for PCA/UMAP demos)
- [ ] **Bandit environments**: Bernoulli arms, contextual arms with
  known reward functions
- [ ] **Sequence/ranking data**: simulated user visits where each user
  has a latent preference vector, position bias is configurable, and
  pair/triple bonuses are injected so we can verify the model recovers them
- [ ] **Regression with outliers / heteroscedastic noise**
- [ ] **A/B test streams**: paired Bernoulli arms with known lift, for
  the bayesian-ab-testing bundle
- [ ] **Corruptions module**: layer missing data, leakage, label noise on
  top of any clean dataset

---

## Open questions

- Pricing: do we charge $5 for foundational templates (low value to
  buyer, high value to us as foot-in-the-door), or only price the
  domain-specific bundles?
- Bundle granularity: one big "tabular-classification" bundle vs three
  small ones (binary / multiclass / multilabel)?
