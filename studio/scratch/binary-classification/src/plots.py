"""Plot helpers for the binary-classification scratch project."""

import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray) -> plt.Figure:
    """ROC and precision-recall side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_roc, ax_pr = axes

    ax_roc.plot(fpr, tpr, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="random")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")

    pos_rate = float(y_true.mean())
    ax_pr.plot(recall, precision, lw=2, label=f"PR AUC = {pr_auc:.3f}")
    ax_pr.axhline(pos_rate, color="grey", ls="--", lw=1, label=f"baseline = {pos_rate:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-recall curve (better for imbalanced)")
    ax_pr.legend(loc="lower left")

    fig.tight_layout()
    return fig


def calibration_plot(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> plt.Figure:
    """Reliability diagram. For binary classification this is critical:
    a model can have great ROC-AUC and still be miscalibrated."""
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_cal, ax_hist = axes

    ax_cal.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="perfect")
    ax_cal.plot(mean_pred, frac_pos, marker="o", lw=2, label="model")
    ax_cal.set_xlabel("mean predicted P(positive)")
    ax_cal.set_ylabel("fraction of positives observed")
    ax_cal.set_title("Calibration (reliability diagram)")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.legend(loc="best")

    ax_hist.hist(y_score, bins=30, color="#4477aa", alpha=0.7)
    ax_hist.set_xlabel("predicted P(positive)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Predicted probability histogram")

    fig.tight_layout()
    return fig


def threshold_sweep(y_true: np.ndarray, y_score: np.ndarray) -> tuple[plt.Figure, dict]:
    """Sweep classification thresholds and plot precision/recall/F1.
    Returns the figure and a dict of best-threshold-by-F1.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions = []
    recalls = []
    f1s = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)

    best_f1_idx = int(np.argmax(f1s))
    best_threshold = float(thresholds[best_f1_idx])
    best_f1 = float(f1s[best_f1_idx])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresholds, precisions, lw=2, label="precision")
    ax.plot(thresholds, recalls, lw=2, label="recall")
    ax.plot(thresholds, f1s, lw=2, label="F1")
    ax.axvline(best_threshold, color="red", lw=1, ls="--",
               label=f"best F1 = {best_f1:.3f} @ t={best_threshold:.2f}")
    ax.axvline(0.5, color="grey", lw=1, ls=":", label="default 0.5")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("metric value")
    ax.set_title("Threshold sweep — 0.5 is rarely the right answer")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best")
    fig.tight_layout()

    best = {
        "best_f1_threshold": best_threshold,
        "best_f1": best_f1,
        "precision_at_best_f1": float(precisions[best_f1_idx]),
        "recall_at_best_f1": float(recalls[best_f1_idx]),
    }
    return fig, best


def confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=14)
    ax.set_title(f"Confusion matrix @ threshold={threshold:.2f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def shap_summary(model, X_sample, feature_names: list[str]) -> plt.Figure:
    """Global SHAP summary (beeswarm). XGBoost has fast TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig
