import numpy as np
import matplotlib.pyplot as plt

try:
    embeddings = np.load("embeddings.npy")        # shape (n, d)
except FileNotFoundError:
    embeddings = None

try:
    S = np.load("similarity.npy")                 # shape (n, n)
except FileNotFoundError:
    S = None

try:
    labels = np.loadtxt("labels.csv", delimiter=",", dtype=int)  # shape (n,)
except FileNotFoundError:
    labels = None

def compute_cluster_similarity(similarity: np.ndarray, labels: np.ndarray):
    """
    similarity: (n, n) symmetric matrix (e.g., cosine similarity), 1.0 on diagonal.
    labels: (n,) array-like of cluster ids (int/str).
    Returns:
        mean_within, mean_inter, per_cluster_within (dict: cluster -> mean)
    """
    labels = np.asarray(labels)
    n = similarity.shape[0]
    if similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    if labels.shape[0] != n:
        raise ValueError("labels length must match similarity size")

    same = labels[:, None] == labels[None, :]
    np.fill_diagonal(same, False)  # exclude self-similarity from within
    diff = ~same

    within_vals = similarity[same]
    inter_vals  = similarity[diff]

    mean_within = float(np.mean(within_vals)) if within_vals.size else float("nan")
    mean_inter  = float(np.mean(inter_vals))  if inter_vals.size  else float("nan")

    # Per-cluster within means (useful diagnostics)
    per_cluster = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            block = similarity[np.ix_(idx, idx)]
            mask = ~np.eye(len(idx), dtype=bool)
            per_cluster[c] = float(np.mean(block[mask]))
        else:
            per_cluster[c] = float("nan")

    return mean_within, mean_inter, per_cluster


def ensure_similarity_matrix(embeddings: np.ndarray = None,
                             similarity: np.ndarray = None,
                             normalize_embeddings: bool = True) -> np.ndarray:
    """
    Provide either `similarity` directly or `embeddings` to compute cosine similarity.
    Returns a (n, n) similarity matrix.
    """
    if similarity is not None:
        return np.asarray(similarity)

    if embeddings is None:
        raise ValueError("Provide either `similarity` or `embeddings`.")
    X = np.asarray(embeddings)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D: (n_samples, dim)")

    if normalize_embeddings:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X @ X.T  # cosine similarity if normalized


def plot_within_vs_inter(labels,
                         embeddings: np.ndarray = None,
                         similarity: np.ndarray = None,
                         ax: plt.Axes = None,
                         title: str = "Mean Within-Cluster vs Inter-Cluster Similarity",
                         save_path: str = None):
    """
    High-level wrapper: accepts either embeddings or a similarity matrix + labels,
    computes the metrics, and plots a simple two-bar chart (no hard-coded colors).
    """
    S = ensure_similarity_matrix(embeddings=embeddings, similarity=similarity)
    mean_within, mean_inter, per_cluster = compute_cluster_similarity(S, np.asarray(labels))

    # Plot
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        created_ax = True

    ax.bar(["Mean Within", "Mean Inter"], [mean_within, mean_inter])
    ax.set_title(title)
    ax.set_ylabel("Similarity")
    ax.set_ylim(0, 1)  # optional; remove if your similarity isn't in [0,1]
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if save_path:
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if created_ax:
        plt.tight_layout()
        plt.show()

    return {
        "mean_within": mean_within,
        "mean_inter": mean_inter,
        "per_cluster_within": per_cluster
    }
