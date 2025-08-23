import torch
import numpy as np
import re
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from pathlib import Path

# --- Constants ---
PCA_COMPONENTS = 100
TSNE_COMPONENTS = 10
DISTANCE_THRESHOLD = 0.45
ALPHA = 0.7
DEFAULT_EMBEDDING_MODEL = 'all-mpnet-base-v2'

# A set of common English stopwords
STOPWORDS = {
    'a', 'an', 'the', 'and', 'but', 'or', 'so', 'if', 'for', 'of', 'in', 'to', 'on', 'with', 'by', 'at', 'from',
    'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
    'he', 'she', 'it', 'they', 'we', 'you', 'i',
    'that', 'which', 'who', 'what', 'when', 'where', 'why', 'how',
    'about', 'above', 'after', 'below', 'between', 'down', 'during', 'into', 'out', 'over', 'through', 'under', 'up',
    'no', 'not', 'only', 'very', 's', 't'
}

def visualize_clusters(embeddings: NDArray, cluster_labels: NDArray, tokens: list[str], log_path: Path):
    """Saves a t-SNE visualization of the clusters."""
    if embeddings.shape[0] <= 1:
        return

    with torch.no_grad():
        embeddings_np = embeddings.cpu().to(torch.float32).numpy()

    perplexity = max(min(5, embeddings_np.shape[0] - 1), 1)
    tsne = TSNE(n_components=2, init='pca', perplexity=perplexity, method='exact', random_state=42)
    reduced_embeddings_2d = tsne.fit_transform(embeddings_np)
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_embeddings_2d[:, 0], reduced_embeddings_2d[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
    
    for i, token in enumerate(tokens):
        plt.annotate(token.strip(), (reduced_embeddings_2d[i, 0], reduced_embeddings_2d[i, 1]), fontsize=8)
        
    plt.title('Token Clusters Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.savefig(log_path / 'clusters.png')
    plt.close()

def filter_and_clean_tokens(tokens: list[str], probabilities: list[float]) -> tuple[list[str], list[float]]:
    """
    Performs a stricter cleaning of tokens to remove stopwords, punctuation, and nonsensical fragments.
    """
    cleaned_tokens = []
    cleaned_probs = []

    for token, prob in zip(tokens, probabilities):
        token_str = token.strip()
        
        # Rule 1: Skip if empty after stripping
        if not token_str:
            continue
            
        # Rule 2: Check against stopwords (case-insensitive)
        if token_str.lower() in STOPWORDS:
            continue
            
        # Rule 3: Ensure it contains at least one alphabetic character
        if not re.search(r'[a-zA-Z]', token_str):
            continue
            
        # Rule 4: Filter out single characters that are not 'a' or 'i' (case-insensitive)
        if len(token_str) == 1 and token_str.lower() not in ['a', 'i']:
             continue

        cleaned_tokens.append(token_str) # Append the cleaned, stripped string
        cleaned_probs.append(prob)
    
    return cleaned_tokens, cleaned_probs

def filter_by_frequency(list_of_token_lists: list[list[str]], min_occurrences: int) -> tuple[list[str], Counter]:
    """
    Filters tokens to keep only those that appear in at least `min_occurrences` paraphrases.
    Also returns the counts of all tokens for the ranking formula.
    """
    if not list_of_token_lists:
        return [], Counter()
    
    token_counts = Counter(token for sublist in list_of_token_lists for token in sublist)
    frequent_tokens = [token for token, count in token_counts.items() if count >= min_occurrences]
    
    return frequent_tokens, token_counts

import requests

def get_contextual_embeddings(prompt: str, tokens: list[str], vllm_url: str, model_name: str) -> NDArray | None:
    """
    Gets sentence embeddings for tokens by querying our custom server endpoint.
    """
    try:
        # The base URL is the same, we just hit our new endpoint
        embedding_url = vllm_url.replace("/v1/completions", "/get_embeddings")
        
        json_payload = {
            "prompt": prompt,
            "tokens": tokens,
        }

        response = requests.post(embedding_url, json=json_payload, timeout=60) # Increased timeout
        response.raise_for_status()
        data = response.json()
        
        # Extract embeddings from the response
        embeddings = data['embeddings']
        
        # Convert to a torch tensor to match the rest of the pipeline
        return torch.tensor(embeddings, device="cpu")

    except requests.exceptions.RequestException as e:
        print(f"API Error while fetching embeddings: {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"Error parsing embedding response: {e}")
        return None

def reduce_and_cluster_embeddings(embeddings: NDArray) -> NDArray:
    """
    Performs dimensionality reduction (PCA then t-SNE) and clustering, as per the paper.
    """
    if embeddings.shape[0] <= 1:
        return np.array([0]) if embeddings.shape[0] == 1 else np.array([])

    with torch.no_grad():
        embeddings_np = embeddings.cpu().to(torch.float32).numpy()

    if embeddings_np.shape[1] > PCA_COMPONENTS:
        pca_components = min(PCA_COMPONENTS, embeddings_np.shape[0])
        if pca_components > 0:
            pca = PCA(n_components=pca_components, svd_solver='full')
            reduced_pca = pca.fit_transform(embeddings_np)
        else:
            reduced_pca = embeddings_np
    else:
        reduced_pca = embeddings_np

    if reduced_pca.shape[1] > TSNE_COMPONENTS:
        tsne_components = min(TSNE_COMPONENTS, reduced_pca.shape[0])
        perplexity = min(30.0, float(reduced_pca.shape[0] - 1))
        if tsne_components > 0 and perplexity > 0:
            tsne = TSNE(n_components=tsne_components, init='pca', perplexity=perplexity, method='exact', random_state=42)
            reduced_tsne = tsne.fit_transform(reduced_pca)
        else:
            reduced_tsne = reduced_pca
    else:
        reduced_tsne = reduced_pca

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=DISTANCE_THRESHOLD,
        metric='cosine',
        linkage='average'
    )
    return clustering.fit_predict(reduced_tsne)

def rank_concepts_by_paper_formula(tokens: list[str], probabilities: list[float], cluster_labels: NDArray, repetitions: Counter, alpha: float = ALPHA) -> list[dict]:
    """
    Ranks clusters by summing the probabilities of the tokens within them.
    This is a more robust method than the original paper's formula.
    """
    unique_clusters = np.unique(cluster_labels)
    ranked_concepts = []
    
    token_to_prob = {token: prob for token, prob in zip(tokens, probabilities)}
    
    for cluster_id in unique_clusters:
        mask = (cluster_labels == cluster_id)
        concept_tokens = [token for i, token in enumerate(tokens) if mask[i]]
        
        if not concept_tokens:
            continue

        # Sum the probabilities of all tokens in the cluster
        cluster_score = sum(token_to_prob.get(token, 0) for token in concept_tokens)
        
        ranked_concepts.append({
            "id": cluster_id,
            "tokens": concept_tokens,
            "score": cluster_score,
            "details": f"(sum_prob: {cluster_score:.4f})"
        })
        
    ranked_concepts.sort(key=lambda x: x["score"], reverse=True)
    return ranked_concepts

def select_token_from_best_concept(best_concept: dict, all_predictions: dict[str, float]) -> str | None:
    """
    Selects the token with the highest original probability from the winning concept cluster.
    """
    best_token = None
    highest_prob = -1.0

    for token in best_concept["tokens"]:
        if all_predictions.get(token, -1) > highest_prob:
            highest_prob = all_predictions[token]
            best_token = token
            
    return best_token
