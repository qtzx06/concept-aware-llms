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
GRAMMAR_CRITICAL_TOKENS = {'.', ',', '?', '!', "'s", "'", "of", "in", "to", "a", "the"}

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

def filter_candidates(token_ids, token_probs, tokenizer, min_length=3):
    """
    Filters out stopwords, short tokens, and other non-meaningful tokens.
    """
    filtered_ids = []
    filtered_probs = []
    for token_id, prob in zip(token_ids, token_probs):
        token_str = tokenizer.decode(token_id, skip_special_tokens=True).strip()
        # Stricter filter: ensure the token contains at least one letter
        if (token_str and 
            len(token_str) >= min_length and 
            token_str.lower() not in STOPWORDS and 
            re.search(r'[a-zA-Z]', token_str)):
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

def cluster_and_visualize(embeddings, token_ids, tokenizer, log_path, pca_components=100, tsne_components=2, distance_threshold=0.45):
    """
    Performs dimensionality reduction, clustering, and saves a visualization.
    Returns the cluster labels.
    """
    if embeddings.shape[0] <= 1:
        return np.array([0]) if embeddings.shape[0] == 1 else np.array([])

    # fix: convert bfloat16 to float32 before converting to numpy
    with torch.no_grad():
        embeddings_np = embeddings.cpu().to(torch.float32).numpy()

    # dimensionality reduction for visualization (t-sne to 2d)
    perplexity = max(min(5, embeddings_np.shape[0] - 1), 1)
    tsne = TSNE(n_components=tsne_components, init='pca', perplexity=perplexity, method='exact', random_state=42)
    reduced_embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(reduced_embeddings_2d)

    # visualization
    plt.figure(figsize=(12, 10))
    
    # Use a font that supports a wider range of unicode characters to avoid warnings
    plt.rcParams['font.family'] = 'sans-serif'
    
    scatter = plt.scatter(reduced_embeddings_2d[:, 0], reduced_embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    
    # annotate points with their decoded token
    for i, token_id in enumerate(token_ids):
        plt.annotate(tokenizer.decode(token_id), (reduced_embeddings_2d[i, 0], reduced_embeddings_2d[i, 1]), fontsize=8)
        
    plt.title(f'Token Clusters (Step saved in {log_path.name})')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.savefig(log_path / 'clusters.png')
    plt.close()

    return cluster_labels
