"""Plot helpers for the multilabel-classification scratch project."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def label_balance_plot(Y: np.ndarray, label_names: list[str]) -> plt.Figure:
    """Per-label positive rate. Different from multiclass: each label is
    independently 0/1 and can have its own imbalance level."""
    pos_rates = Y.mean(axis=0)
    pos_counts = Y.sum(axis=0).astype(int)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(label_names))))
    bars = ax.barh(label_names, pos_rates, color="#4477aa")
    for bar, rate, cnt in zip(bars, pos_rates, pos_counts):
        ax.text(
            rate, bar.get_y() + bar.get_height() / 2,
            f"  {rate:.1%}  (n={cnt})", va="center", fontsize=8,
        )
    ax.set_xlabel("positive rate")
    ax.set_title(f"Per-label positive rate ({Y.shape[0]} samples)")
    ax.set_xlim(0, max(1.0, float(pos_rates.max()) * 1.2))
    fig.tight_layout()
    return fig


def label_cooccurrence(Y: np.ndarray, label_names: list[str]) -> plt.Figure:
    """Conditional probability matrix: P(label_j = 1 | label_i = 1).

    Reading: row i, column j = "given label_i is on, how often is label_j
    also on?" The diagonal is always 1.0. Off-diagonal entries reveal
    label dependencies that ClassifierChain can exploit.
    """
    n_labels = Y.shape[1]
    cooc = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        i_count = int(Y[:, i].sum())
        if i_count == 0:
            continue
        for j in range(n_labels):
            cooc[i, j] = float(((Y[:, i] == 1) & (Y[:, j] == 1)).sum() / i_count)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cooc, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n_labels))
    ax.set_yticks(range(n_labels))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("label_j")
    ax.set_ylabel("label_i (conditioned on)")
    ax.set_title("Conditional co-occurrence: P(label_j | label_i)")
    for i in range(n_labels):
        for j in range(n_labels):
            ax.text(
                j, i, f"{cooc[i, j]:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if cooc[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def per_label_metrics_plot(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]
) -> plt.Figure:
    """Per-label precision / recall / F1 — analogous to per-class for
    multiclass but each label has its own positive rate so the bars
    can vary wildly."""
    n_labels = len(label_names)
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for i in range(n_labels):
        p, r, f, _ = precision_recall_fscore_support(
            y_true[:, i], y_pred[:, i], average="binary", zero_division=0
        )
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
        supports.append(int(y_true[:, i].sum()))

    x = np.arange(n_labels)
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_prf, ax_sup = axes

    ax_prf.bar(x - width, precisions, width, label="precision", color="#4477aa")
    ax_prf.bar(x, recalls, width, label="recall", color="#ee8866")
    ax_prf.bar(x + width, f1s, width, label="F1", color="#228833")
    ax_prf.set_xticks(x)
    ax_prf.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax_prf.set_ylim(0, 1.05)
    ax_prf.set_ylabel("score")
    ax_prf.set_title("Per-label precision / recall / F1")
    ax_prf.legend(loc="best", fontsize=9)
    for i, v in enumerate(f1s):
        ax_prf.text(i + width, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax_sup.bar(x, supports, color="#888888")
    ax_sup.set_xticks(x)
    ax_sup.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax_sup.set_ylabel("support (positive count in test set)")
    ax_sup.set_title("Per-label support")
    for i, v in enumerate(supports):
        ax_sup.text(i, v, str(v), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def cardinality_plot(Y_true: np.ndarray, Y_pred: np.ndarray) -> plt.Figure:
    """Distribution of labels-per-row, true vs predicted."""
    true_card = Y_true.sum(axis=1)
    pred_card = Y_pred.sum(axis=1)
    max_c = int(max(true_card.max(), pred_card.max()))
    bins = np.arange(0, max_c + 2) - 0.5

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(true_card, bins=bins, alpha=0.6, color="#4477aa", label="true")
    ax.hist(pred_card, bins=bins, alpha=0.6, color="#cc3311", label="predicted")
    ax.set_xlabel("labels per row")
    ax.set_ylabel("count")
    ax.set_xticks(range(0, max_c + 1))
    ax.set_title(
        f"Label cardinality — true mean {true_card.mean():.2f}, "
        f"pred mean {pred_card.mean():.2f}"
    )
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
