# template-sklearn-pipeline

The minimal end-to-end sklearn project that every tabular bundle copies.
The point isn't the model — it's the **plumbing**.

## What this verifies

- Project layout: `src/` for code, `notebooks/` for marimo, `mlruns/` for tracking
- `Pipeline` + `ColumnTransformer` (preprocessing travels with the model)
- Train / val / test split with fixed seed
- Cross-validated metric (log-loss) for model selection
- MLflow logging:
  - **Params**: every CLI arg + all model hyperparameters
  - **Metrics**: cv mean/std, test score, recovery error against ground truth
  - **Tags**: data hash, data path, true_p (when known)
  - **Artifacts**: the model (via `mlflow.sklearn.log_model`), plots, sidecar
- Model loading via `mlflow.sklearn.load_model("runs:/<id>/model")`
- A marimo notebook that loads the trained artifact and demonstrates inference
  with an interactive `anywidget` slider

## The worked example: fair coin

The dataset is `studio/data/coin-flip.parquet` from `datagen coin-flip`.
Each row is `(flip_index, outcome)`. We treat this as tabular and fit a
logistic regression on `flip_index → outcome`.

Why this works:
- For a stationary biased coin, `intercept ≈ logit(true_p)` and the slope
  on `flip_index` is ≈ 0.
- For a coin whose bias **drifts**, the slope is non-zero — so the slope
  becomes a free non-stationarity detector.

## Run it

```bash
# 1. Generate data (if you haven't already)
datagen coin-flip --n 500 --p 0.7 --seed 42

# 2. Train and log to MLflow
cd studio/templates/sklearn-pipeline
python src/train.py

# 3. Inspect via the MLflow UI (optional)
mlflow ui --backend-store-uri ./mlruns

# 4. Load the trained model and predict
python src/predict.py --run-id <run_id> --flip-index 100

# 5. Open the marimo demo notebook
marimo edit notebooks/coin_flip_demo.py
```

## Layout

```
sklearn-pipeline/
├── README.md
├── SKILL.md              # what an agent learns from this template
├── src/
│   ├── train.py          # train + log to MLflow
│   ├── predict.py        # load model and predict
│   └── plots.py          # plot helpers (logged as MLflow artifacts)
├── notebooks/
│   └── coin_flip_demo.py # marimo notebook with anywidget slider
└── mlruns/               # MLflow tracking store (created on first run)
```
