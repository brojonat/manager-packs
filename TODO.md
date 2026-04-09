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
- [x] `multiclass-classification` — XGBoost with `multi:softprob`,
  per-class metrics (precision/recall/F1), macro vs micro vs weighted
  F1 averaging, **`sample_weight` for class imbalance** (no
  `scale_pos_weight` for multiclass), confusion matrix as the primary
  diagnostic, top-K accuracy, per-class SHAP. Demo generates a 5-class
  imbalanced dataset (40/25/15/12/8) and contrasts unweighted vs
  weighted training to show how `sample_weight` rescues minority-class
  F1 even though overall accuracy barely moves. Studio scratch at
  `studio/scratch/multiclass-classification/`. Verified: 5-class
  balanced test acc 0.84, top-3 acc 0.98, F1 macro/micro/weighted all
  ≈ 0.84 (balanced data; demo uses imbalanced to show the divergence).
- [x] `multilabel-classification` — XGBoost wrapped in
  `MultiOutputClassifier` (one independent model per label,
  parallelized via `n_jobs=-1`). **Hamming loss as the primary metric**
  (not subset accuracy, which is brutally strict). Four F1 averaging
  strategies (macro/micro/weighted/**samples** — multilabel-only),
  per-label F1 monitoring, label co-occurrence heatmap to decide
  whether `ClassifierChain` would help, label cardinality histogram
  (true vs predicted) to catch under-prediction. Studio scratch at
  `studio/scratch/multilabel-classification/`. Verified on a 6-label
  dataset with positive rates 13-53%: hamming loss 0.170, subset
  accuracy 0.415, F1 macro 0.645, F1 micro 0.717. Per-label F1 ranges
  from 0.45 (rarest, 12.5% positive) to 0.80 (most common, 53%
  positive) — exactly the imbalance pattern the bundle teaches to
  monitor.
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

## Phase 5 — Local LLM fine-tuning with Unsloth

Fine-tune small open-weight LLMs locally for text classification,
translation, and other NLP tasks. The pitch: providers (OpenAI,
Anthropic) offer fine-tuning as an API, but you can do it **locally**
with models you own, keep your data private, and pay nothing per
inference once trained. Unlike TF-IDF + XGBoost (which requires heavy
hyperparameter tuning and doesn't generalize to tasks like
translation), a fine-tuned LLM transfers the same pre-trained language
understanding to any text task.

**Library:** [unsloth](https://github.com/unslothai/unsloth) — 2×
faster QLoRA fine-tuning with 70% less VRAM. Wraps HuggingFace
ecosystem (transformers, PEFT, TRL) with custom Triton kernels.

**Primary model:** **Gemma 4** — Google's recent release, reportedly
very effective at small sizes. Start here and validate; fall back to
Phi-4-mini or Llama-3.2 if needed.

**Primary use case:** text classification (the multiclass problem from
Phase 1, but using an LLM on raw text instead of XGBoost on tabular
features). Frame classification as instruction-tuning: the prompt is
the text, the completion is the label. Then extend to translation and
other text tasks to show the generalization advantage.

**When to use LLM fine-tuning instead of XGBoost:**
- The input is **text** (reviews, tickets, emails, descriptions),
  not tabular features
- You want to leverage the LLM's pre-trained language understanding
  rather than engineering features manually
- You have at least a few hundred labeled examples (not zero-shot or
  few-shot — that's prompting, not fine-tuning)
- You want a model you **own** (no API dependency, no per-token cost,
  data never leaves your machine)
- You want the **same model** to later handle related text tasks
  (translation, summarization, extraction) without starting over

**When to still use XGBoost:**
- The features are already structured/tabular — no text involved
- You don't need language understanding, just feature patterns

### Inference runtime

Use **llama.cpp** directly for production inference (not Ollama).
Convert the fine-tuned model to GGUF, run via llama.cpp's CLI or
Python bindings. The trend is toward using llama.cpp directly for
more control over quantization, context length, and batching.

### Dependency isolation: marimo `--sandbox` + PEP 723

**Do NOT add LLM deps to `studio/pyproject.toml`.** Each LLM
notebook should be self-contained via PEP 723 inline script metadata.
Marimo's `--sandbox` mode reads the metadata, creates an isolated
venv, and installs exactly the declared deps automatically:

```python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "unsloth",
#     "marimo",
#     "torch",
#     "transformers",
#     "trl",
#     "datasets",
# ]
# ///
```

Run with `marimo edit --sandbox demo.py`. The notebook IS the
dependency declaration — no separate requirements.txt. This is the
pattern for any bundle with heavy or GPU-specific deps.

### Workflow

```
1. Pick a Gemma-4 variant from HuggingFace (e.g. unsloth/gemma-4-4b-it-bnb-4bit)
2. Format classification data as instruction→label pairs (Alpaca format)
3. Load with unsloth + QLoRA (4-bit quantization)
4. Fine-tune with HuggingFace TRL's SFTTrainer
5. Evaluate: per-class F1, confusion matrix (same metrics as the
   multiclass-classification bundle — we already know how to do this)
6. Compare: zero-shot prompting vs fine-tuned on the same eval set
7. Merge LoRA adapter + quantize to GGUF
8. Serve locally with llama.cpp
```

### Tasks

- [ ] Build `studio/scratch/llm-text-classification/`:
  - Marimo notebook (PEP 723 sandbox) demonstrating the full workflow
  - Load Gemma-4 via unsloth, format data, fine-tune, evaluate
  - Compare zero-shot prompting vs fine-tuned on the same eval set
    to show the value of fine-tuning
- [ ] Build `bundles/llm-text-classification/`:
  - SKILL.md — when to use LLM fine-tuning vs XGBoost (text vs
    tabular), the unsloth/Gemma-4 workflow, data formatting, eval
    metrics (reuse multiclass: per-class F1, confusion matrix), GGUF
    export + llama.cpp serving
  - demo.py — PEP 723 sandbox notebook, fully self-contained
- [ ] `datagen` subcommand for synthetic text classification data
  (generate labeled text snippets with known categories — could use
  an LLM to generate the text, or use a small public dataset like
  a subset of AG News)
- [ ] Evaluate whether Gemma-4 at 2B or 4B is best for the demo
  (tradeoff: smaller = faster training loop in the notebook, larger
  = better accuracy; the demo should be impressive but not take an
  hour to train)
- [ ] Ship training code only, NOT LoRA adapter weights (weights are
  large and model-version-specific; the reproducible code is the
  product)
- [ ] Extension: after text classification, add a translation task
  to show "same fine-tuning pipeline, different task, same model
  family" — this is the generalization advantage over TF-IDF

---

## Retrofit — PEP 723 inline script metadata on all existing bundles

All 6 Phase 1 bundle `demo.py` files currently list deps in a
docstring ("Required deps: pip install ..."). Replace with a PEP 723
`# /// script` block so buyers can run `marimo edit --sandbox demo.py`
and get an auto-created isolated env with zero manual install.

- [x] `bundles/binary-classification/demo.py`
- [x] `bundles/multiclass-classification/demo.py`
- [x] `bundles/multilabel-classification/demo.py`
- [x] `bundles/regression/demo.py`
- [x] `bundles/tabular-eda/demo.py`
- [x] `bundles/unsupervised/demo.py`

The `building-bundles` skill has been updated to make PEP 723 the
standard. All future bundles should use PEP 723 from the start.

---

## Open questions

- Pricing: do we charge $5 for foundational templates (low value to
  buyer, high value to us as foot-in-the-door), or only price the
  domain-specific bundles?
- Bundle granularity: one big "tabular-classification" bundle vs three
  small ones (binary / multiclass / multilabel)?
