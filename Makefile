HOST ?= 0.0.0.0
PORT ?= 2718
LOGDIR := logs
MARIMO_FLAGS := --host $(HOST) --port $(PORT) --no-token --headless --include-code

$(LOGDIR):
	mkdir -p $(LOGDIR)

# Phase 1 bundles (sklearn/xgboost — use marimo --sandbox)
.PHONY: demo-binary-classification
demo-binary-classification: | $(LOGDIR)
	marimo run --sandbox bundles/binary-classification/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/binary-classification.log

.PHONY: demo-multiclass-classification
demo-multiclass-classification: | $(LOGDIR)
	marimo run --sandbox bundles/multiclass-classification/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/multiclass-classification.log

.PHONY: demo-multilabel-classification
demo-multilabel-classification: | $(LOGDIR)
	marimo run --sandbox bundles/multilabel-classification/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/multilabel-classification.log

.PHONY: demo-regression
demo-regression: | $(LOGDIR)
	marimo run --sandbox bundles/regression/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/regression.log

.PHONY: demo-tabular-eda
demo-tabular-eda: | $(LOGDIR)
	marimo run --sandbox bundles/tabular-eda/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/tabular-eda.log

.PHONY: demo-unsupervised
demo-unsupervised: | $(LOGDIR)
	marimo run --sandbox bundles/unsupervised/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/unsupervised.log

# Phase 5 bundles (unsloth/torch — use .venv-llm, skip sandbox)
.PHONY: demo-llm-finetuning
demo-llm-finetuning: | $(LOGDIR)
	.venv-llm/bin/marimo run --no-sandbox bundles/llm-finetuning/demo.py $(MARIMO_FLAGS) 2>&1 | tee $(LOGDIR)/llm-finetuning.log

# Edit mode (for development)
.PHONY: edit-llm-finetuning
edit-llm-finetuning: | $(LOGDIR)
	.venv-llm/bin/marimo edit --no-sandbox bundles/llm-finetuning/demo.py --host $(HOST) --port $(PORT) --no-token 2>&1 | tee $(LOGDIR)/llm-finetuning.log

# Utilities
.PHONY: check
check:
	@for f in bundles/*/demo.py; do \
		echo "checking $$f ..."; \
		marimo check "$$f" || exit 1; \
	done
	@echo "all demos pass"
