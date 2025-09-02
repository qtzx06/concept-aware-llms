# concept-qwen/concept_qwen_utilities.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nltk.corpus import stopwords
import nltk

# A set of tokens that are critical for grammar and should sometimes bypass concept clustering.
# This includes common punctuation and a few essential stopwords.
GRAMMAR_CRITICAL_TOKENS = {
    '.', ',', '?', '!', "'s", "'", '"',
    'a', 'an', 'the', 'and', 'but', 'or', 'so', 'if', 'for', 'of', 'in', 'to', 'on', 'with', 'by', 'at', 'from',
    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
    'he', 'she', 'it', 'they', 'we', 'you', 'I',
    'that', 'which', 'who', 'what', 'when', 'where', 'why', 'how'
}

# ensure the stopwords dataset from nltk is available
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def get_top_k_candidates(logits, k):
    """Gets the top-k most likely tokens and their probabilities from the logits."""
    probs = torch.softmax(logits[:, -1, :], dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k, dim=-1)
    return top_k_ids.squeeze(), top_k_probs.squeeze()

def filter_candidates(token_ids, token_probs, tokenizer):
    """
    A much more relaxed filter. The main goal is to remove tokens that are not
    decodable as meaningful text (e.g., pure whitespace, control characters).
    The "lobotomy" of removing stopwords and short words is GONE.
    """
    filtered_ids = []
    filtered_probs = []
    for token_id, prob in zip(token_ids, token_probs):
        token_str = tokenizer.decode(token_id)
        # The main goal is to filter out non-linguistic or purely functional tokens.
        # We keep it simple: if it has at least one letter or number, it's a candidate.
        if re.search(r'[a-zA-Z0-9]', token_str):
            filtered_ids.append(token_id)
            filtered_probs.append(prob)
    
    if not filtered_ids:
        return torch.tensor([], device=token_ids.device, dtype=torch.long), \
               torch.tensor([], device=token_probs.device, dtype=torch.float)

    return torch.tensor(filtered_ids, device=token_ids.device), torch.tensor(filtered_probs, device=token_probs.device)


def get_candidate_embeddings(model, token_ids):
    """Gets the embedding vectors for the filtered candidate tokens."""
    if token_ids.dim() == 0:
        token_ids = token_ids.unsqueeze(0)
    if token_ids.nelement() == 0:
        return torch.tensor([], device=model.device)
    embedding_matrix = model.get_output_embeddings().weight
    return embedding_matrix[token_ids]

def cluster_embeddings(embeddings, distance_threshold=0.45):
    """Performs dimensionality reduction and clustering, returns cluster labels."""
    if embeddings.shape[0] <= 1:
        return np.array([0]) if embeddings.shape[0] == 1 else np.array([])

    with torch.no_grad():
        embeddings_np = embeddings.cpu().to(torch.float32).numpy()

    # Use PCA for initial dimensionality reduction before clustering
    # This is more stable than t-SNE for the actual clustering process
    if embeddings_np.shape[0] > 10: # Heuristic to avoid PCA on tiny samples
        pca = PCA(n_components=min(10, embeddings_np.shape[0]))
        reduced_embeddings = pca.fit_transform(embeddings_np)
    else:
        reduced_embeddings = embeddings_np

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    return clustering.fit_predict(reduced_embeddings)

def find_centroid_token(cluster_embeddings, cluster_token_ids, tokenizer):
    """Finds the token whose embedding is closest to the cluster's centroid."""
    if cluster_embeddings.nelement() == 0:
        return "N/A"
    
    with torch.no_grad():
        centroid = torch.mean(cluster_embeddings.to(torch.float32), dim=0)
        distances = torch.cdist(cluster_embeddings.to(torch.float32), centroid.unsqueeze(0))
        closest_token_index = torch.argmin(distances)
        return tokenizer.decode(cluster_token_ids[closest_token_index])

def visualize_clusters(embeddings, cluster_labels, token_ids, tokenizer, log_path):
    """Saves a t-SNE visualization of the clusters."""
    if embeddings.shape[0] <= 1:
        return

    with torch.no_grad():
        embeddings_np = embeddings.cpu().to(torch.float32).numpy()

    perplexity = max(min(5, embeddings_np.shape[0] - 1), 1)
    tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, method='exact', random_state=42)
    reduced_embeddings_2d = tsne.fit_transform(embeddings_np)
    
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.family'] = 'sans-serif'
    scatter = plt.scatter(reduced_embeddings_2d[:, 0], reduced_embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    
    for i, token_id in enumerate(token_ids):
        decoded_token = tokenizer.decode(token_id)
        # Sanitize the token to remove non-printable characters before plotting
        cleaned_token = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', decoded_token)
        plt.annotate(cleaned_token, (reduced_embeddings_2d[i, 0], reduced_embeddings_2d[i, 1]), fontsize=8)
        
    plt.title(f'Token Clusters (Step saved in {log_path.name})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.savefig(log_path / 'clusters.png')
    plt.close()