import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from numpy.typing import NDArray

# Constants adapted from the baseline example
PCA_COMPONENTS = 10
TSNE_COMPONENTS = 2
# This threshold might need tuning for different embedding spaces/models
DISTANCE_THRESHOLD = 0.15 

def reduce_and_cluster_embeddings(embeddings: NDArray) -> NDArray:
    """
    Performs dimensionality reduction and clustering on token embeddings.
    Returns the cluster labels for each embedding.
    """
    if embeddings.shape[0] <= 1:
        return np.array([0]) if embeddings.shape[0] == 1 else np.array([])

    # --- Dynamic PCA for pre-reduction ---
    # Ensure n_components is not greater than n_samples
    n_samples = embeddings.shape[0]
    dynamic_pca_components = min(PCA_COMPONENTS, n_samples)
    
    if embeddings.shape[1] > dynamic_pca_components and dynamic_pca_components > 1:
        pca = PCA(n_components=dynamic_pca_components)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    # --- t-SNE to 2D for visualization and clustering ---
    # Perplexity must be less than n_samples
    perplexity = min(30.0, float(embeddings_reduced.shape[0] - 1))
    
    if perplexity > 0:
        tsne = TSNE(
            n_components=TSNE_COMPONENTS,
            init="pca",
            perplexity=perplexity,
            random_state=42,
            method='exact' # Slower but better for small datasets
        )
        embeddings_2d = tsne.fit_transform(embeddings_reduced)
    else:
        # Fallback if not enough points for t-SNE
        embeddings_2d = embeddings_reduced[:, :TSNE_COMPONENTS] if embeddings_reduced.shape[1] >= TSNE_COMPONENTS else embeddings_reduced

    # --- Agglomerative Clustering on the 2D embeddings ---
    # Using cosine distance as it's often effective for high-dimensional vectors
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric="cosine",
        linkage="average"
    )
    cluster_labels = clustering.fit_predict(embeddings_2d)

    return cluster_labels
