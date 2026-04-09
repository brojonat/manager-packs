"""Plot helpers for the multiclass-classification scratch project."""

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def class_balance_plot(y: np.ndarray, n_classes: int) -> plt.Figure:
    counts = np.bincount(y, minlength=n_classes)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(range(n_classes), counts, color="#4477aa")
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, val, str(int(val)),
            ha="center", va="bottom", fontsize=8,
        )
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"class {i}" for i in range(n_classes)])
    ax.set_ylabel("count")
    ax.set_title(f"Class balance ({len(y)} samples, {n_classes} classes)")
    fig.tight_layout()
    return fig


def confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, normalize: bool = False
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    if normalize:
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        display = cm_norm
        fmt = ".2f"
        title = "Normalized confusion matrix (row-wise)"
    else:
        display = cm
        fmt = "d"
        title = "Confusion matrix"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display, cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels([f"c{i}" for i in range(n_classes)])
    ax.set_yticklabels([f"c{i}" for i in range(n_classes)])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    threshold = display.max() / 2
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, format(display[i, j], fmt),
                ha="center", va="center",
                color="white" if display[i, j] > threshold else "black",
                fontsize=10,
            )
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def per_class_metrics_plot(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> plt.Figure:
    """Per-class precision / recall / F1 + support, side by side."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    x = np.arange(n_classes)
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_prf, ax_sup = axes

    ax_prf.bar(x - width, precision, width, label="precision", color="#4477aa")
    ax_prf.bar(x, recall, width, label="recall", color="#ee8866")
    ax_prf.bar(x + width, f1, width, label="F1", color="#228833")
    ax_prf.set_xticks(x)
    ax_prf.set_xticklabels([f"c{i}" for i in range(n_classes)])
    ax_prf.set_ylim(0, 1.05)
    ax_prf.set_ylabel("score")
    ax_prf.set_title("Per-class precision / recall / F1")
    ax_prf.legend(loc="best", fontsize=9)
    for i, v in enumerate(f1):
        ax_prf.text(i + width, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    ax_sup.bar(x, support, color="#888888")
    ax_sup.set_xticks(x)
    ax_sup.set_xticklabels([f"c{i}" for i in range(n_classes)])
    ax_sup.set_ylabel("support (count in test set)")
    ax_sup.set_title("Per-class support")
    for i, v in enumerate(support):
        ax_sup.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig


def roc_ovr_plot(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> plt.Figure:
    """One-vs-rest ROC curve per class."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = plt.colormaps.get_cmap("tab10")
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=2, color=cmap(i % 10),
                label=f"class {i} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def shap_summary(model, X_sample, feature_names: list[str], class_idx: int) -> plt.Figure:
    """SHAP summary for one specific class. Multiclass SHAP returns
    a 3D array (n_samples, n_features, n_classes); pick a class to plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    # shap_values.values shape: (n_samples, n_features, n_classes)
    # Slice to one class
    sliced = shap.Explanation(
        values=shap_values.values[:, :, class_idx],
        base_values=shap_values.base_values[:, class_idx]
        if shap_values.base_values.ndim > 1
        else shap_values.base_values,
        data=shap_values.data,
        feature_names=feature_names,
    )
    plt.figure(figsize=(8, 5))
    shap.summary_plot(sliced, X_sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP summary — class {class_idx}", fontsize=10)
    fig.tight_layout()
    return fig
