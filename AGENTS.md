# Agent notes — manager-pack

Persistent context for AI agents working on this project. Append (don't
replace) when adding new facts.

## Studio skills

Workflow / process skills for building ManagerPack content. Read these
before starting any new bundle.

- `studio/skills/building-bundles/SKILL.md` — the end-to-end procedure
  for taking a problem statement to a published, marimo-validated,
  MLflow-tracked bundle. Includes the model-family decision table
  (XGBoost for tabular prediction, PyMC for inference/decisions) and
  the develop-in-scratch → distill-into-bundle handoff. Iterate on this
  after every bundle ships.

## Tools / libraries for future phases

### Unsloth + Gemma 4 (Phase 5 — LLM fine-tuning)

[unsloth](https://github.com/unslothai/unsloth) — 2× faster QLoRA
fine-tuning with 70% less VRAM. Primary model: **Gemma 4** (Google).
Primary use case: fine-tune locally for text classification, then
extend to translation and other text tasks to show generalization.

Key conventions for Phase 5:
- **Gemma 4** as the default model (not Phi, not Llama)
- **llama.cpp** for inference (not Ollama)
- **marimo `--sandbox`** with PEP 723 inline script metadata for
  dependency isolation (NOT in studio pyproject.toml)
- **Ship training code only**, not LoRA weights
- TF-IDF is the strawman baseline, not a serious contender

See `TODO.md` Phase 5 for the full plan.

## External references on disk

### Bayesian Methods for Hackers (cloned locally)

`/home/brojonat/clones/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/`

Cameron Davidson-Pilon's *Probabilistic Programming and Bayesian Methods
for Hackers*, cloned locally with all chapters and example notebooks.
Use this as the primary reference when designing **Phase 2 PyMC bundles**
(`bayesian-ab-testing`, `bayesian-bandits`, `bayesian-regression`,
`bayesian-mixture-models`, `bayesian-decision-analysis`).

Layout:

- `Chapter1_Introduction/` — basic PyMC, Beta-Binomial, switchpoint detection
- `Chapter2_MorePyMC/` — modeling tricks, deterministic variables, A/B testing
- `Chapter3_MCMC/` — sampler diagnostics, convergence
- `Chapter4_TheGreatestTheoremNeverTold/` — Law of Large Numbers, Monte Carlo
- `Chapter5_LossFunctions/` — Bayesian decision analysis, loss functions
- `Chapter6_Priorities/` — prior selection, conjugate priors, elicitation
- `Chapter7_BayesianMachineLearning/` — Bayesian regression, ML applications
- `ExamplesFromChapters/` — standalone runnable examples
- `sandbox/` — additional one-off examples

When designing a Phase 2 bundle, **read the relevant chapter(s) first**
to ground the bundle in the canonical examples (Chapter 5 for
decision-analysis, Chapter 7 for Bayesian regression, etc.). The book's
notebooks are written in Jupyter — convert to marimo when copying any
example into a bundle.
