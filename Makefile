HOST ?= 0.0.0.0
PORT ?= 2718
LOGDIR := logs
MARIMO_FLAGS := --host $(HOST) --port $(PORT) --no-token --headless --include-code

$(LOGDIR):
	mkdir -p $(LOGDIR)

# Phase 1 bundles (sklearn/xgboost — use marimo --sandbox)
.PHONY: demo-binary-classification
demo-binary-classification: | $(LOGDIR)
	marimo run --sandbox bundles/binary-classification/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/binary-classification.log

.PHONY: demo-multiclass-classification
demo-multiclass-classification: | $(LOGDIR)
	marimo run --sandbox bundles/multiclass-classification/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/multiclass-classification.log

.PHONY: demo-multilabel-classification
demo-multilabel-classification: | $(LOGDIR)
	marimo run --sandbox bundles/multilabel-classification/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/multilabel-classification.log

.PHONY: demo-regression
demo-regression: | $(LOGDIR)
	marimo run --sandbox bundles/regression/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/regression.log

.PHONY: demo-tabular-eda
demo-tabular-eda: | $(LOGDIR)
	marimo run --sandbox bundles/tabular-eda/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/tabular-eda.log

.PHONY: demo-unsupervised
demo-unsupervised: | $(LOGDIR)
	marimo run --sandbox bundles/unsupervised/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/unsupervised.log

# Phase 2 bundles (PyMC/Bayesian — use marimo --sandbox)
.PHONY: demo-bayesian-ab-testing
demo-bayesian-ab-testing: | $(LOGDIR)
	marimo run --sandbox bundles/bayesian-ab-testing/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/bayesian-ab-testing.log

.PHONY: demo-bayesian-bandits
demo-bayesian-bandits: | $(LOGDIR)
	marimo run --sandbox bundles/bayesian-bandits/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/bayesian-bandits.log

.PHONY: demo-bayesian-regression
demo-bayesian-regression: | $(LOGDIR)
	marimo run --sandbox bundles/bayesian-regression/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/bayesian-regression.log

.PHONY: demo-bayesian-mixture-models
demo-bayesian-mixture-models: | $(LOGDIR)
	marimo run --sandbox bundles/bayesian-mixture-models/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/bayesian-mixture-models.log

.PHONY: demo-bayesian-decision-analysis
demo-bayesian-decision-analysis: | $(LOGDIR)
	marimo run --sandbox bundles/bayesian-decision-analysis/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/bayesian-decision-analysis.log

# Phase 5 bundles (unsloth/torch — use .venv-llm, skip sandbox)
.PHONY: demo-llm-finetuning
demo-llm-finetuning: | $(LOGDIR)
	.venv-llm/bin/marimo run --no-sandbox bundles/llm-finetuning/scripts/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/llm-finetuning.log

# Edit mode (for development)
.PHONY: edit-llm-finetuning
edit-llm-finetuning: | $(LOGDIR)
	.venv-llm/bin/marimo edit --no-sandbox bundles/llm-finetuning/scripts/demo.py --host $(HOST) --port $(PORT) --no-token 2>&1 | tee $(LOGDIR)/llm-finetuning.log

# Utilities
.PHONY: check
check:
	@for f in bundles/*/scripts/demo.py; do \
		echo "checking $$f ..."; \
		marimo check "$$f" || exit 1; \
	done
	@echo "all demos pass"

# Export HTML previews — regenerates assets/*.html from scripts/*.py
# Uses --sandbox so PEP 723 metadata installs deps automatically.
# Usage: make export-html-<bundle>  OR  make export-html (all bundles)
.PHONY: export-html
export-html:
	@for d in bundles/*/scripts; do \
		bundle=$$(dirname $$d | xargs basename); \
		mkdir -p bundles/$$bundle/assets; \
		for py in $$d/*.py; do \
			[ -f "$$py" ] || continue; \
			name=$$(basename $$py .py); \
			echo "exporting $$bundle/$$name ..."; \
			marimo export html --sandbox --no-include-code "$$py" -o bundles/$$bundle/assets/$$name.html; \
		done; \
	done

export-html-%:
	@bundle=$*; \
	mkdir -p bundles/$$bundle/assets; \
	for py in bundles/$$bundle/scripts/*.py; do \
		[ -f "$$py" ] || continue; \
		name=$$(basename $$py .py); \
		echo "exporting $$bundle/$$name ..."; \
		marimo export html --sandbox --no-include-code "$$py" -o bundles/$$bundle/assets/$$name.html; \
	done

# Sync to llmsrules.
# `make sync` is fast — just copies whatever HTMLs are on disk (does NOT
# regenerate). Use `make export-html` separately if you want to refresh
# previews for all bundles (slow; each one spins up a sandbox).
# `make sync-<bundle>` regenerates that one bundle's HTML then syncs —
# safe to chain because it's narrowly scoped.
.PHONY: sync
sync:
	./sync-skills.sh

sync-%: export-html-%
	./sync-skills.sh $*
