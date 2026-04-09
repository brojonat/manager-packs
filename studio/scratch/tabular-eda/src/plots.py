"""Plot helpers for the tabular-eda scratch project."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def missing_data_bar(df: pd.DataFrame) -> plt.Figure:
    """Per-column missing percentage. Sorted descending so the worst
    offenders are at the top."""
    missing_pct = (df.isna().mean() * 100).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(missing_pct))))
    bars = ax.barh(missing_pct.index, missing_pct.values, color="#cc3311")
    for bar, val in zip(bars, missing_pct.values):
        if val > 0:
            ax.text(val, bar.get_y() + bar.get_height() / 2, f"  {val:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("missing %")
    ax.set_title("Missing data per column")
    ax.set_xlim(0, max(100, missing_pct.max() * 1.15))
    fig.tight_layout()
    return fig


def numeric_distributions(df: pd.DataFrame, numeric_cols: list[str]) -> plt.Figure:
    """Histograms of all numeric columns. Highlights skewed columns."""
    n_feat = len(numeric_cols)
    n_cols = 4
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        values = df[col].dropna()
        ax.hist(values, bins=40, color="#4477aa", alpha=0.7)
        skew = float(values.skew())
        title = f"{col}\nskew={skew:+.2f}"
        if abs(skew) > 1.0:
            title += " ⚠"
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(n_feat, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle("Numeric distributions (⚠ marks |skew| > 1)", fontsize=10)
    fig.tight_layout()
    return fig


def categorical_cardinality(df: pd.DataFrame, cat_cols: list[str]) -> plt.Figure:
    """Bar chart of unique-value counts per categorical column. Highlights
    columns that would explode a OneHotEncoder."""
    if not cat_cols:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No categorical columns", ha="center", va="center")
        ax.axis("off")
        return fig

    counts = pd.Series({c: df[c].nunique() for c in cat_cols}).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(counts))))
    colors = ["#cc3311" if v > 50 else "#4477aa" for v in counts.values]
    bars = ax.barh(counts.index, counts.values, color=colors)
    for bar, val in zip(bars, counts.values):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f"  {val}", va="center", fontsize=8)
    ax.axvline(50, color="black", lw=1, ls="--", alpha=0.5)
    ax.set_xlabel("unique value count")
    ax.set_title("Categorical cardinality (red = > 50, OHE explosion risk)")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], target_col: str) -> plt.Figure:
    """Pearson correlation heatmap with the target column annotated."""
    cols = [c for c in numeric_cols if c != target_col] + [target_col]
    corr = df[cols].corr().values
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)
    target_idx = len(cols) - 1
    for i in range(len(cols)):
        ax.text(
            target_idx, i, f"{corr[i, target_idx]:.2f}",
            ha="center", va="center", fontsize=7,
            color="black" if abs(corr[i, target_idx]) < 0.5 else "white",
        )
    ax.set_title("Pearson correlations (annotated column = correlation with target)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def mi_vs_pearson(
    pearson: pd.Series, mutual_info: pd.Series, target_col: str
) -> plt.Figure:
    """Side-by-side bar chart of |Pearson correlation| and mutual
    information for every feature. Catches features that have low linear
    correlation but high mutual information (non-linear relationships)."""
    features = sorted(set(pearson.index) | set(mutual_info.index))
    pearson_abs = [abs(float(pearson.get(f, 0.0))) for f in features]
    mi_vals = [float(mutual_info.get(f, 0.0)) for f in features]

    x = np.arange(len(features))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, pearson_abs, width, label="|Pearson|", color="#4477aa")
    ax.bar(x + width / 2, mi_vals, width, label="Mutual Info", color="#ee8866")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=8)
    ax.set_title(
        f"Linear vs non-linear relationship with `{target_col}` "
        "(MI catches what Pearson misses)"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def outlier_boxplot(df: pd.DataFrame, numeric_cols: list[str]) -> plt.Figure:
    """Box plots per numeric column. Outliers stick out."""
    n_feat = len(numeric_cols)
    n_cols = 4
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 2.5 * n_rows))
    axes = np.atleast_2d(axes)
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        values = df[col].dropna()
        ax.boxplot(values, vert=True, showfliers=True)
        # IQR-based outlier count
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        n_out = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        title = f"{col}\noutliers (IQR): {n_out}"
        if n_out > 0.05 * len(values):
            title += " ⚠"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)
    for idx in range(n_feat, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")
    fig.suptitle("Outliers per numeric column (⚠ marks > 5% outliers)", fontsize=10)
    fig.tight_layout()
    return fig
