# ManagerPack Studio

Content development workspace. Everything in here is for **building**
ManagerPack bundles, not for running the store. Heavy ML dependencies
(scikit-learn, pymc, mlflow, marimo) live here so the store CLI stays
lean.

## Layout

```
studio/
├── pyproject.toml       # studio dependencies (sklearn, pymc, mlflow, marimo)
├── datagen/             # data generation CLI
│   ├── cli.py
│   ├── output.py        # shared parquet + sidecar writer
│   └── problems/        # one module per problem type
└── data/                # generated datasets (parquet + sidecar JSON)
```

## Install

From the repo root:

```bash
pip install -e ./studio
```

## datagen

Each subcommand maps to a different problem type. Every command writes a
parquet file plus a sidecar JSON containing the ground-truth parameters
used to generate the data, so we can validate that models recover what
they're supposed to recover.

```bash
datagen --help                                        # list problems
datagen coin-flip --n 200 --p 0.7 --seed 42           # Bernoulli sequence
datagen binary-classification --n 1000 --features 10  # sklearn make_classification
datagen multiclass-classification --classes 5         # sklearn make_classification
datagen multilabel-classification --labels 5          # sklearn make_multilabel_classification
datagen regression --n 500 --features 8 --noise 0.5   # sklearn make_regression
datagen blobs --centers 4                             # sklearn make_blobs
```

Default output: `studio/data/<problem>.parquet` (+ `studio/data/<problem>.json`).
Override with `--output <path>`.

## Adding a new problem

1. Create `datagen/problems/<problem>.py` with a Click command
2. Register it in `datagen/cli.py`
3. The command should call `write_dataset(problem, df, ground_truth, output)`
   from `datagen/output.py`
