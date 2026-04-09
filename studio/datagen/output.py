"""Shared output utilities for datagen subcommands.

Every dataset is written as:
  - <output>.parquet     # the actual data
  - <output>.json        # ground-truth params used to generate it

The sidecar JSON lets us validate models against the truth they should
recover. This is the whole point of synthetic data.
"""

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def resolve_output(problem: str, output: str | None) -> Path:
    """Default to studio/data/<problem>.parquet; allow override."""
    if output:
        p = Path(output)
        if p.suffix != ".parquet":
            p = p.with_suffix(".parquet")
        return p
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_DATA_DIR / f"{problem}.parquet"


def write_dataset(
    problem: str,
    df: pd.DataFrame,
    ground_truth: dict[str, Any],
    output: str | None = None,
) -> Path:
    """Write parquet + sidecar JSON. Returns the parquet path."""
    parquet_path = resolve_output(problem, output)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    sidecar_path = parquet_path.with_suffix(".json")
    sidecar = {
        "problem": problem,
        "n_rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "ground_truth": ground_truth,
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2, default=str) + "\n")

    click.echo(f"  wrote {parquet_path}  ({len(df)} rows, {len(df.columns)} cols)")
    click.echo(f"  wrote {sidecar_path}")
    return parquet_path
