"""Plot helpers for the unsupervised scratch project."""

import matplotlib.pyplot as plt
import numpy as np


def k_selection_plot(
    k_values: list[int],
    inertias: list[float],
    silhouettes: list[float],
    best_k: int,
) -> plt.Figure:
    """Two-panel: elbow curve (inertia vs K) + silhouette curve (silhouette vs K)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax_elbow, ax_sil = axes

    ax_elbow.plot(k_values, inertias, marker="o", lw=2, color="#4477aa")
    ax_elbow.axvline(best_k, color="red", lw=1, ls="--", label=f"chosen K = {best_k}")
    ax_elbow.set_xlabel("K (number of clusters)")
    ax_elbow.set_ylabel("inertia (within-cluster sum of squares)")
    ax_elbow.set_title("Elbow — visual but subjective")
    ax_elbow.legend(loc="best")

    ax_sil.plot(k_values, silhouettes, marker="o", lw=2, color="#cc3311")
    ax_sil.axvline(best_k, color="red", lw=1, ls="--", label=f"chosen K = {best_k}")
    ax_sil.set_xlabel("K (number of clusters)")
    ax_sil.set_ylabel("silhouette score")
    ax_sil.set_title("Silhouette — quantifiable, prefer this")
    ax_sil.legend(loc="best")

    fig.tight_layout()
    return fig


def pca_clusters(
    X_pca: np.ndarray,
    labels_dict: dict[str, np.ndarray],
    explained_variance: tuple[float, float],
) -> plt.Figure:
    """One subplot per labeling method, points colored by cluster id."""
    n_methods = len(labels_dict)
    n_cols = min(n_methods, 3)
    n_rows = (n_methods + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (name, labels) in enumerate(labels_dict.items()):
        ax = axes[idx]
        unique_labels = sorted(set(labels))
        cmap = plt.colormaps.get_cmap("tab10")
        for li, lab in enumerate(unique_labels):
            mask = labels == lab
            color = "#888888" if lab == -1 else cmap(li % 10)
            label_str = "noise (-1)" if lab == -1 else f"cluster {lab}"
            ax.scatter(
                X_pca[mask, 0], X_pca[mask, 1],
                s=12, alpha=0.7, color=color, label=label_str,
            )
        ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%})")
        ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%})")
        ax.set_title(f"{name} — {len(unique_labels)} clusters")
        ax.legend(loc="best", fontsize=8)
    for ax_blank in axes[n_methods:]:
        ax_blank.axis("off")
    fig.tight_layout()
    return fig


def stability_heatmap(ari_matrix: np.ndarray, n_runs: int) -> plt.Figure:
    """Heatmap of pairwise ARI between runs of the same algorithm. High and
    consistent = stable. Low or noisy = wrong K or wrong algorithm."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(ari_matrix, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n_runs))
    ax.set_yticks(range(n_runs))
    ax.set_xticklabels([f"r{i}" for i in range(n_runs)], fontsize=8)
    ax.set_yticklabels([f"r{i}" for i in range(n_runs)], fontsize=8)
    ax.set_title(
        f"Stability — pairwise ARI across {n_runs} random_states\n"
        f"(mean off-diag = {np.mean(ari_matrix[~np.eye(n_runs, dtype=bool)]):.3f})"
    )
    for i in range(n_runs):
        for j in range(n_runs):
            ax.text(j, i, f"{ari_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if ari_matrix[i, j] < 0.7 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def anomaly_scatter(
    X_pca: np.ndarray,
    is_anomaly: np.ndarray,
    score: np.ndarray | None = None,
) -> plt.Figure:
    """Anomaly detection result in PCA space."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    normal_mask = ~is_anomaly
    ax.scatter(
        X_pca[normal_mask, 0], X_pca[normal_mask, 1],
        s=10, alpha=0.5, color="#4477aa", label=f"normal ({normal_mask.sum()})",
    )
    ax.scatter(
        X_pca[is_anomaly, 0], X_pca[is_anomaly, 1],
        s=40, alpha=0.9, color="#cc3311", marker="x", label=f"anomaly ({is_anomaly.sum()})",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("IsolationForest anomaly detection in PCA space")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig
