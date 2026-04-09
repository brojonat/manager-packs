---
name: building-bundles
description: How to design, build, validate, and ship a new ManagerPack skill bundle. Use when starting any new bundle in the studio. Captures the workflow from picking a model family through to a published, marimo-validated, MLflow-tracked deliverable.
---

# Building a new ManagerPack bundle

The procedure for taking a problem statement to a sellable bundle. This
is v1 — it's likely incomplete and will be iterated on as we learn from
each bundle we ship. **Update this skill after each bundle** with
anything that surprised you, broke, or worked unexpectedly well.

The phases below assume you've already decided *what* bundle to build.
That's a separate skill (`picking-bundles`, doesn't exist yet). This
skill is about *how* to build it once the topic is locked in.

## 0. Preconditions

Before you start coding:

- [ ] You can name the bundle in one sentence ("how to do X correctly")
- [ ] You can name the buyer in one sentence ("a [role] who needs to [task]")
- [ ] You know which template to copy from (`sklearn-pipeline` or
      `pymc-inference` — see "Choosing a model family" below)
- [ ] You know what synthetic dataset the bundle will use (must already
      exist in `datagen`, or be added there first)
- [ ] You can list 3-5 things the bundle teaches that aren't obvious
      from reading the sklearn / PyMC docs

If any of these is fuzzy, stop and clarify. The hardest bundles to ship
are the ones with vague scope.

## 1. Choose a model family

Default decisions:

| Problem type | Model family | Template to copy |
|---|---|---|
| Binary / multiclass / multilabel classification on tabular data | **XGBoost** | `sklearn-pipeline` |
| Regression on tabular data | **XGBoost** | `sklearn-pipeline` |
| EDA / unsupervised / clustering on tabular data | scikit-learn (PCA, UMAP, KMeans, IsolationForest) | `sklearn-pipeline` |
| A/B testing, hypothesis testing | PyMC | `pymc-inference` |
| Bandits / sequential decisions | PyMC | `pymc-inference` |
| Hierarchical / multilevel models | PyMC | `pymc-inference` |
| Decision analysis with explicit utilities | PyMC | `pymc-inference` |
| Mixture models with uncertainty | PyMC | `pymc-inference` |
| Time series with structural components | PyMC (or statsmodels for SARIMAX) | `pymc-inference` |

**XGBoost is the default for tabular prediction** — see the
xgboost-for-tabular feedback memory. Logistic / linear models appear
in Phase 1 bundles only as **baselines** for the marimo notebook to
contrast against XGBoost, never as the headline estimator.

**Bayesian methods are for when uncertainty quantification matters more
than raw predictive accuracy.** If a buyer just wants "predict this
column," they want XGBoost. If they want "what's the probability
treatment B is at least 5% better than A?" they want PyMC.

## 2. Develop in `studio/scratch/<bundle-name>/`

This is the lab notebook. Heavy, full of MLflow runs, free to be messy.
The shipping version comes later.

```bash
cp -r studio/templates/sklearn-pipeline studio/scratch/<bundle-name>
cd studio/scratch/<bundle-name>
```

(or `pymc-inference` for Bayesian bundles.)

Now specialize:

- [ ] Update `src/train.py` to load the right dataset from `studio/data/`
- [ ] Replace the model in `build_pipeline()` (e.g. `LogisticRegression`
      → `XGBClassifier`)
- [ ] Add the bundle-specific value adds (calibration, threshold tuning,
      class imbalance handling, SHAP, etc.) — **these are the reason the
      bundle is worth $5**, not just "XGBoost on a dataframe"
- [ ] Update `src/plots.py` with domain-appropriate plots
- [ ] Update `src/predict.py` if the prediction shape differs
- [ ] Update the marimo notebook to load the new model and show the new
      diagnostics

## 3. Validate end-to-end (the studio version)

Before touching `bundles/`:

- [ ] `python src/train.py` runs to completion without warnings you
      don't understand
- [ ] The MLflow run logs everything you'd want to see in the UI
      (params, metrics, all plots, the model artifact, the data sidecar)
- [ ] **Recovery against ground truth is good** — the model actually
      recovers what we know to be true (within reason). For
      classification this means high ROC-AUC against the synthetic
      labels; for regression it means low MAE/RMSE; for Bayesian this
      means the posterior covers the true value.
- [ ] `python src/predict.py --run-id <id> ...` reloads the model from
      MLflow and produces sensible output
- [ ] `marimo check notebooks/<name>_demo.py` is clean
- [ ] `marimo export html notebooks/<name>_demo.py -o /tmp/check.html`
      runs end-to-end without errors

If any of these fails, fix it in `studio/scratch/` before moving on.
The bundle should never ship something the studio scratch can't run.

## 4. Distill into `bundles/<bundle-name>/`

The shipping artifact. **Self-contained.** No datagen dependency, no
MLflow dependency, nothing the buyer doesn't strictly need.

What goes in `bundles/<bundle-name>/`:

```
bundles/<bundle-name>/
├── manifest.json     # sale metadata (price, tags, files, Stripe IDs)
├── SKILL.md          # the actual agent skill — generic guidance
└── demo.py           # self-contained marimo notebook with worked example
```

Optional extras only if they earn their keep:

- `EXAMPLES.md` — extra worked examples in markdown (not runnable)
- `REFERENCE.md` — cheat sheet / API reference an agent can grep
- `data/example.parquet` — only if generating data inline is awkward

### `SKILL.md`

This is what an agent reads. Write it for *future-you's agent that has
never seen this domain*. Structure:

1. Frontmatter (`name`, `description` — must specify trigger keywords)
2. One-paragraph what it does + when to use it
3. Core principles (3-7 bullets, the things you'd say if you only had
   30 seconds to brief someone)
4. Project layout the agent should produce
5. Pipeline shape with concrete code (the agent will copy this verbatim)
6. The bundle's value-adds, each with a code block showing how to do it
7. Common pitfalls / things not to do
8. Where to look for more — pointers to the demo notebook, references

**Generic vs problem-specific:** SKILL.md should describe the *pattern*
(e.g., "for class imbalance use `scale_pos_weight = n_neg / n_pos`"),
not just narrate the specific dataset. The agent will be applying this
to the buyer's actual problem, not to our synthetic data.

### `demo.py` (self-contained marimo notebook)

This is what a buyer runs to see "yes, this works." Constraints:

- **No `datagen` dependency.** Generate the data inline with
  `sklearn.datasets.make_*` and a fixed seed. The buyer should be able
  to `marimo edit demo.py` with only the bundle's stated dependencies.
- **No MLflow.** MLflow is dev-time tooling for us; the buyer doesn't
  need it. Train the model right in the notebook.
- **One narrative thread.** Generate data → fit → diagnose → interpret.
  The buyer reads top-to-bottom and understands the workflow.
- **At least one interactive cell** (`mo.ui.slider` or similar) so the
  buyer feels the value of marimo's reactivity.
- **Validates clean** with `marimo check`. Renders end-to-end with
  `marimo export html` (test before committing).

### `manifest.json`

Already templated by other bundles. New ones go in with all
`stripe_*` and `reddit_*` fields set to `null` — those get populated
later by the store CLI when you actually publish.

```json
{
  "name": "binary-classification",
  "title": "Binary Classification with XGBoost",
  "description": "...",
  "price_cents": 500,
  "tags": ["data-science", "classification", "xgboost"],
  "files": ["SKILL.md", "demo.py"],
  "stripe_product_id": null,
  "stripe_price_id": null,
  "stripe_payment_link": null,
  "reddit_post_id": null
}
```

## 5. Validate the bundle (the shipping version)

- [ ] `managerpack bundles validate <name>` passes
- [ ] `marimo check bundles/<name>/demo.py` is clean
- [ ] `marimo export html bundles/<name>/demo.py -o /tmp/x.html` runs
      end-to-end with no errors
- [ ] You can read `bundles/<name>/SKILL.md` cold and understand the
      whole pattern in 5 minutes
- [ ] `bundles/<name>/demo.py` is genuinely self-contained — try it in
      a fresh shell with only `pip install marimo xgboost scikit-learn
      pandas numpy matplotlib` (and nothing else from the studio)

## 6. Commit

On the right branch (`phase-N` or feature branch). Commit message
explains *what* the bundle teaches and *what* differentiates it from
"just read the docs":

```
Add binary-classification bundle (XGBoost + calibration + threshold tuning)

The first Phase 1 bundle. Specializes the sklearn-pipeline template
with XGBClassifier and adds the things that make tabular classification
work in practice but rarely show up in tutorials:

- scale_pos_weight for imbalance (no resampling)
- Early stopping with held-out validation
- Threshold tuning for the buyer's actual cost function (not 0.5)
- Calibration check via reliability diagram + Brier score
- SHAP for feature importance
- A baseline LogisticRegression cell so the buyer sees why XGBoost wins
...
```

## 7. Publish (later, manually)

When ready to actually sell:

```bash
managerpack stripe create <bundle-name>     # creates product + payment link
managerpack bundles upload <bundle-name>    # upload files to R2
managerpack reddit post <bundle-name>       # post to r/rayab
```

This step is **not part of "building the bundle"** — it's a separate
publish step you do when you're confident the bundle is ready and
you've decided the price tier. Don't auto-publish from the build
workflow.

## Iterating on this skill

Every bundle teaches us something. After each bundle, ask:

- What was harder than expected?
- What was easier than expected?
- What did the bundle's `SKILL.md` need that this skill didn't tell me to write?
- What validation step would have caught a bug earlier?
- What "should be obvious" thing did I forget?

Then update the relevant section above. **This skill is the loop's
memory.**
