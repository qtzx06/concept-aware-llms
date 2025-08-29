#!/usr/bin/env python3
"""
Plot coherence improvement: baseline (Agglomerative) vs improved (DBSCAN + normalization + word-level).
Saves plots to decoder/plots/coherence_improvement.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from pathlib import Path

from cluster_coherence_analyzer import ClusterCoherenceAnalyzer


def make_synthetic_embeddings(num_clusters: int = 5, points_per_cluster: int = 30, dim: int = 256, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((num_clusters, dim))
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    X = []
    labels_true = []
    for k in range(num_clusters):
        # cluster spread
        spread = 0.25 + 0.05 * k
        pts = centers[k] + rng.normal(0, spread, size=(points_per_cluster, dim))
        X.append(pts)
        labels_true.extend([k] * points_per_cluster)
    X = np.vstack(X).astype(np.float32)
    labels_true = np.array(labels_true)
    return torch.from_numpy(X), labels_true


def cluster_baseline(X: np.ndarray, distance_threshold: float = 0.20, pca_components: int = 10):
    Z = X
    if Z.shape[1] > pca_components and Z.shape[0] > 2:
        pca = PCA(n_components=min(pca_components, Z.shape[0]-1), random_state=42)
        Z = pca.fit_transform(Z)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
    return clustering.fit_predict(Z)


def cluster_improved(X: np.ndarray, eps: float = 0.28, min_samples: int = 2, pca_components: int = 8):
    # normalize
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    if Z.shape[1] > pca_components and Z.shape[0] > 2:
        pca = PCA(n_components=min(pca_components, Z.shape[0]-1), random_state=42)
        Z = pca.fit_transform(Z)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = clustering.fit_predict(Z)
    # fallback if too many noise
    if np.sum(labels == -1) > max(2, int(0.3 * len(labels))):
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.15, metric="cosine", linkage="average")
        labels = clustering.fit_predict(Z)
    return labels


def plot_improvement(baseline_res, improved_res, save_path: Path):
    bw, bi = baseline_res['mean_within_similarity'], baseline_res['mean_inter_similarity']
    iw, ii = improved_res['mean_within_similarity'], improved_res['mean_inter_similarity']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of means
    axes[0].bar(["Baseline within", "Baseline inter"], [bw, bi], color=["#4C78A8", "#F58518"], alpha=0.8)
    axes[0].bar(["Improved within", "Improved inter"], [iw, ii], color=["#54A24B", "#E45756"], alpha=0.8)
    axes[0].axhline(0.41, color='green', linestyle='--', linewidth=1, label='Target within 0.41')
    axes[0].axhline(0.12, color='red', linestyle='--', linewidth=1, label='Target inter 0.12')
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mean similarities")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Histogram overlay of within similarities
    axes[1].hist(baseline_res['within_similarities'], bins=30, alpha=0.5, label='Baseline within', color="#4C78A8")
    axes[1].hist(improved_res['within_similarities'], bins=30, alpha=0.5, label='Improved within', color="#54A24B")
    axes[1].set_title("Within-cluster similarity distribution")
    axes[1].set_xlabel("cosine similarity")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved coherence improvement plot to {save_path}")


def main():
    print("Building coherence improvement plot...")
    analyzer = ClusterCoherenceAnalyzer()

    # Synthetic consistent data for A/B
    X, _ = make_synthetic_embeddings(num_clusters=6, points_per_cluster=25, dim=analyzer.embedding_matrix.shape[1])
    X_np = X.float().cpu().numpy()

    # Baseline clustering
    labels_base = cluster_baseline(X_np, distance_threshold=0.20, pca_components=10)
    res_base = analyzer.analyze_clustering_quality(X, labels_base)
    print(f"Baseline within={res_base['mean_within_similarity']:.3f}, inter={res_base['mean_inter_similarity']:.3f}")

    # Improved clustering
    labels_imp = cluster_improved(X_np, eps=0.28, min_samples=2, pca_components=8)
    # Use normalized tensor for analysis (analyzer normalizes internally too)
    res_imp = analyzer.analyze_clustering_quality(X, labels_imp)
    print(f"Improved within={res_imp['mean_within_similarity']:.3f}, inter={res_imp['mean_inter_similarity']:.3f}")

    # Plot
    out_path = Path(__file__).parent / "plots" / "coherence_improvement.png"
    plot_improvement(res_base, res_imp, out_path)


if __name__ == "__main__":
    main()
