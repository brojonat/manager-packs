"""Unsupervised clustering + anomaly detection on the blobs dataset.

Specializes the sklearn-pipeline template for the unlabeled case:

- Choose K with **silhouette score**, not just visual elbow
- **Stability check** across multiple random_states (pairwise ARI)
- Compare KMeans / GMM / DBSCAN side by side in PCA space
- **IsolationForest** for anomaly detection
- Validate against known cluster IDs (from the synthetic data sidecar)
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ibis
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR.parent
DEFAULT_DATA = TEMPLATE_DIR.parent.parent / "data" / "blobs.parquet"
DEFAULT_TRACKING = TEMPLATE_DIR / "mlruns"

sys.path.insert(0, str(THIS_DIR))
from plots import (  # noqa: E402
    anomaly_scatter,
    k_selection_plot,
    pca_clusters,
    stability_heatmap,
)


def data_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--stability-runs", type=int, default=8)
    parser.add_argument("--anomaly-contamination", type=float, default=0.02)
    parser.add_argument("--experiment", default="unsupervised-blobs")
    parser.add_argument("--tracking-uri", default=str(DEFAULT_TRACKING))
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(f"Data not found at {args.data}. Run `datagen blobs` first.")

    sidecar_path = args.data.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    truth = sidecar.get("ground_truth", {})
    true_n_centers = int(truth.get("n_centers", 0))

    # --- ibis: load + materialize ---
    table = ibis.duckdb.connect().read_parquet(str(args.data))
    feature_cols = [c for c in table.columns if c.startswith("feature_")]

    data = (
        table
        .select(*feature_cols, "cluster")
        .execute()
    )
    X_raw = data[feature_cols].to_numpy()
    y_true = data["cluster"].to_numpy()  # held back at fit time, used for validation only

    # Standardize before clustering — distance metrics need it
    X = StandardScaler().fit_transform(X_raw)

    mlflow.set_tracking_uri(f"file:{args.tracking_uri}")
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "data_path": str(args.data),
                "n_rows": len(data),
                "n_features": len(feature_cols),
                "seed": args.seed,
                "k_min": args.k_min,
                "k_max": args.k_max,
                "stability_runs": args.stability_runs,
                "anomaly_contamination": args.anomaly_contamination,
            }
        )
        mlflow.set_tag("data_hash", data_hash(data))
        if true_n_centers:
            mlflow.set_tag("truth.n_centers", str(true_n_centers))

        # ============================================================
        # Step 1: choose K via silhouette score
        # ============================================================
        k_values = list(range(args.k_min, args.k_max + 1))
        inertias = []
        silhouettes = []
        for k in k_values:
            km = KMeans(n_clusters=k, n_init=10, random_state=args.seed)
            labels_k = km.fit_predict(X)
            inertias.append(float(km.inertia_))
            silhouettes.append(float(silhouette_score(X, labels_k)))

        best_k = k_values[int(np.argmax(silhouettes))]
        best_silhouette = float(max(silhouettes))
        mlflow.log_metric("best_k_by_silhouette", best_k)
        mlflow.log_metric("best_silhouette", best_silhouette)
        if true_n_centers:
            mlflow.log_metric("k_recovery_error", abs(best_k - true_n_centers))

        # ============================================================
        # Step 2: stability check at best K
        # ============================================================
        stability_labels = []
        for run_idx in range(args.stability_runs):
            km_run = KMeans(n_clusters=best_k, n_init=10, random_state=run_idx * 7 + 1)
            stability_labels.append(km_run.fit_predict(X))
        ari_matrix = np.zeros((args.stability_runs, args.stability_runs))
        for i in range(args.stability_runs):
            for j in range(args.stability_runs):
                ari_matrix[i, j] = adjusted_rand_score(stability_labels[i], stability_labels[j])
        off_diag_mean = float(
            ari_matrix[~np.eye(args.stability_runs, dtype=bool)].mean()
        )
        mlflow.log_metric("stability_ari_mean", off_diag_mean)

        # ============================================================
        # Step 3: KMeans / GMM / DBSCAN comparison at best K
        # ============================================================
        kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=args.seed)
        kmeans_labels = kmeans_final.fit_predict(X)

        gmm_final = GaussianMixture(n_components=best_k, random_state=args.seed)
        gmm_labels = gmm_final.fit_predict(X)

        # DBSCAN: eps via heuristic (median pairwise distance / 4 is a rough start;
        # in production tune this against your data scale)
        dbscan_final = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan_final.fit_predict(X)

        # Internal evaluation metrics (no labels needed)
        sil_kmeans = float(silhouette_score(X, kmeans_labels))
        sil_gmm = float(silhouette_score(X, gmm_labels))
        ch_kmeans = float(calinski_harabasz_score(X, kmeans_labels))
        ch_gmm = float(calinski_harabasz_score(X, gmm_labels))
        mlflow.log_metric("kmeans_silhouette", sil_kmeans)
        mlflow.log_metric("gmm_silhouette", sil_gmm)
        mlflow.log_metric("kmeans_ch", ch_kmeans)
        mlflow.log_metric("gmm_ch", ch_gmm)

        # External validation against the true labels (only possible because
        # we held them back from fitting)
        ari_kmeans = float(adjusted_rand_score(y_true, kmeans_labels))
        ari_gmm = float(adjusted_rand_score(y_true, gmm_labels))
        ari_dbscan = float(adjusted_rand_score(y_true, dbscan_labels))
        mlflow.log_metric("kmeans_ari_vs_truth", ari_kmeans)
        mlflow.log_metric("gmm_ari_vs_truth", ari_gmm)
        mlflow.log_metric("dbscan_ari_vs_truth", ari_dbscan)

        # ============================================================
        # Step 4: anomaly detection with IsolationForest
        # ============================================================
        # Inject some far-away outliers to give the detector something to find
        rng = np.random.default_rng(args.seed)
        n_anom = max(5, int(args.anomaly_contamination * len(X)))
        outliers = rng.uniform(low=-15, high=15, size=(n_anom, X.shape[1]))
        X_with_anom = np.vstack([X, outliers])
        y_anom_truth = np.concatenate([np.zeros(len(X)), np.ones(n_anom)])

        iso = IsolationForest(
            contamination=args.anomaly_contamination,
            random_state=args.seed,
            n_jobs=-1,
        )
        iso_pred = iso.fit_predict(X_with_anom)  # -1 = outlier, 1 = inlier
        is_anomaly = (iso_pred == -1).astype(int)

        # Precision / recall vs the planted outliers
        tp = int(((is_anomaly == 1) & (y_anom_truth == 1)).sum())
        fp = int(((is_anomaly == 1) & (y_anom_truth == 0)).sum())
        fn = int(((is_anomaly == 0) & (y_anom_truth == 1)).sum())
        anom_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        anom_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        mlflow.log_metric("anomaly_precision", anom_precision)
        mlflow.log_metric("anomaly_recall", anom_recall)
        mlflow.log_metric("n_planted_outliers", n_anom)

        # ============================================================
        # Plots
        # ============================================================
        plots_dir = TEMPLATE_DIR / "_tmp_plots"
        plots_dir.mkdir(exist_ok=True)

        def save(fig, name):
            path = plots_dir / name
            fig.savefig(path, dpi=120, bbox_inches="tight")
            mlflow.log_artifact(str(path), artifact_path="plots")

        save(k_selection_plot(k_values, inertias, silhouettes, best_k), "k_selection.png")

        # PCA for visualization
        pca = PCA(n_components=2, random_state=args.seed)
        X_pca = pca.fit_transform(X)
        var_ratio = (
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )

        save(
            pca_clusters(
                X_pca,
                {
                    "KMeans": kmeans_labels,
                    "GMM": gmm_labels,
                    "DBSCAN": dbscan_labels,
                    "Truth (held out)": y_true,
                },
                var_ratio,
            ),
            "clusters_pca.png",
        )
        save(stability_heatmap(ari_matrix, args.stability_runs), "stability.png")

        X_anom_pca = pca.transform(X_with_anom)
        save(anomaly_scatter(X_anom_pca, is_anomaly == 1), "anomaly.png")

        if sidecar:
            sidecar_out = plots_dir / "sidecar.json"
            sidecar_out.write_text(json.dumps(sidecar, indent=2))
            mlflow.log_artifact(str(sidecar_out), artifact_path="data")

        # ============================================================
        # Summary
        # ============================================================
        print(f"run_id:                {run.info.run_id}")
        print(f"experiment:            {args.experiment}")
        print()
        print(f"K selection (silhouette):")
        for k, sil in zip(k_values, silhouettes):
            marker = "  ←" if k == best_k else ""
            print(f"  K={k}: silhouette={sil:.4f}{marker}")
        print(f"  → chosen K = {best_k}", end="")
        if true_n_centers:
            print(f"  (true K = {true_n_centers})")
        else:
            print()
        print()
        print(f"Stability (mean off-diag ARI across {args.stability_runs} seeds):")
        print(f"  {off_diag_mean:.4f}  (1.0 = identical, < 0.7 = unstable)")
        print()
        print("Method comparison at best K:")
        print(f"  {'method':<10} {'silhouette':>12} {'CH score':>12} {'ARI vs truth':>14}")
        print(f"  {'-' * 50}")
        print(f"  {'KMeans':<10} {sil_kmeans:>12.4f} {ch_kmeans:>12.1f} {ari_kmeans:>14.4f}")
        print(f"  {'GMM':<10} {sil_gmm:>12.4f} {ch_gmm:>12.1f} {ari_gmm:>14.4f}")
        print(f"  {'DBSCAN':<10} {'n/a':>12} {'n/a':>12} {ari_dbscan:>14.4f}")
        print()
        print(f"Anomaly detection (IsolationForest):")
        print(f"  planted outliers: {n_anom}")
        print(f"  precision:        {anom_precision:.4f}")
        print(f"  recall:           {anom_recall:.4f}")


if __name__ == "__main__":
    main()
