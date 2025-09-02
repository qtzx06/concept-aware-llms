import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import numpy as np
from typing import Tuple
import logging
import os

def setup_logger():
    """Sets up a logger to write to concept_decoder.log inside the script's directory."""
    logger = logging.getLogger('ConceptDecoderLogger')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, 'concept_decoder.log')

    handler = logging.FileHandler(log_file_path, mode='w')
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger

def get_top_k_candidates(model, tokenizer, input_ids, k: int, min_token_prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]
    
    probs = torch.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k, dim=-1)
    
    top_k_ids = top_k_ids.squeeze()
    top_k_probs = top_k_probs.squeeze()

    if min_token_prob > 0:
        mask = top_k_probs >= min_token_prob
        top_k_ids = top_k_ids[mask]
        top_k_probs = top_k_probs[mask]

    return top_k_ids, top_k_probs

def get_candidate_embeddings(model, top_k_ids):
    embedding_matrix = model.get_output_embeddings().weight
    return embedding_matrix[top_k_ids]

def reduce_dimensionality(embeddings, n_components=32):
    embeddings_np = embeddings.detach().cpu().float().numpy()
    if embeddings_np.shape[0] <= n_components:
        return embeddings
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_np)
    return torch.tensor(reduced_embeddings, device=embeddings.device)

def cluster_candidates(embeddings, algorithm='agglomerative', distance_threshold=0.4, eps=0.5) -> np.ndarray:
    embeddings_np = embeddings.detach().cpu().float().numpy()
    if algorithm == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    elif algorithm == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=2)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    clustering.fit(embeddings_np)
    return clustering.labels_

def generate_with_concept_decoder(
    model, tokenizer, prompt: str, max_new_tokens: int, k: int, 
    distance_threshold: float, enable_thinking: bool,
    clustering_algo: str, eps: float, use_dim_reduction: bool,
    dim_reduction_components: int, concept_ranking_method: str, min_token_prob: float,
    logger: logging.Logger
) -> str:
    device = model.device
    think_end_token_id = 151668

    messages = [{"role": "system", "content": "Name something parents would criticize their children for having."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    
    full_text = ""
    in_thinking_block = False
    thinking_token_count = 0

    print("Output: ", end="", flush=True)

    for i in range(max_new_tokens):
        logger.info(f"\n{'='*20} Step {i+1}/{max_new_tokens} {'='*20}")
        
        top_k_ids, top_k_probs = get_top_k_candidates(model, tokenizer, input_ids, k, min_token_prob)
        
        if top_k_ids.nelement() == 0:
            print("\n[Warning: No candidates left after filtering. Stopping.]")
            logger.warning("No candidates left after filtering.")
            break

        logger.info(f"\n--- Top {k} Candidates (after prob filter) ---")
        for token_id, prob in zip(top_k_ids, top_k_probs):
            logger.info(f"Token: '{tokenizer.decode(token_id)}' (ID: {token_id.item()}), Prob: {prob.item():.4f}")

        embeddings = get_candidate_embeddings(model, top_k_ids)
        
        # Safeguard for clustering: must have at least 2 samples.
        if embeddings.shape[0] < 2:
            next_token_id = top_k_ids[0]
            logger.info("\nOnly one candidate left after filtering. Selecting it directly.")
        else:
            if use_dim_reduction:
                embeddings = reduce_dimensionality(embeddings, n_components=dim_reduction_components)

            cluster_labels = cluster_candidates(embeddings, algorithm=clustering_algo, distance_threshold=distance_threshold, eps=eps)
            
            logger.info(f"\n--- Clustering Results ({clustering_algo}) ---")
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(tokenizer.decode(top_k_ids[i]))
            for label, tokens in clusters.items():
                logger.info(f"Cluster {label}: {tokens}")

            unique_clusters = np.unique(cluster_labels)
            if clustering_algo == 'dbscan':
                unique_clusters = unique_clusters[unique_clusters != -1]

            if len(unique_clusters) == 0:
                next_token_id = top_k_ids[0]
                logger.info("\nNo valid clusters found. Falling back to greedy selection.")
            else:
                cluster_probs = torch.zeros(len(unique_clusters), device=device)
                logger.info(f"\n--- Concept Ranking ({concept_ranking_method}) ---")
                for j, cluster_id in enumerate(unique_clusters):
                    mask = cluster_labels == cluster_id
                    if concept_ranking_method == 'sum':
                        prob = top_k_probs[mask].sum()
                    elif concept_ranking_method == 'max':
                        prob = top_k_probs[mask].max()
                    cluster_probs[j] = prob
                    logger.info(f"Concept (Cluster {cluster_id}) Prob: {prob:.4f}")

                if torch.sum(cluster_probs) > 0:
                    chosen_cluster_idx = torch.multinomial(cluster_probs, 1).item()
                    chosen_cluster_id = unique_clusters[chosen_cluster_idx]
                else:
                    chosen_cluster_id = unique_clusters[0]
                
                logger.info(f"\nSelected Concept: Cluster {chosen_cluster_id}")

                cluster_mask = cluster_labels == chosen_cluster_id
                token_ids_in_cluster = top_k_ids[cluster_mask]
                token_probs_in_cluster = top_k_probs[cluster_mask]
                
                renormalized_probs = token_probs_in_cluster / token_probs_in_cluster.sum()
                next_token_id = token_ids_in_cluster[torch.multinomial(renormalized_probs, 1).item()]
        
        next_token_item = next_token_id.item()
        logger.info(f"Selected Token: '{tokenizer.decode(next_token_item)}' (ID: {next_token_item})")

        if enable_thinking:
            if not in_thinking_block and len(full_text) == 0 and "think" in tokenizer.decode([next_token_item]):
                in_thinking_block = True
            if in_thinking_block:
                thinking_token_count += 1
                if thinking_token_count > 50:
                    next_token_item = think_end_token_id
                    in_thinking_block = False
            if next_token_item == think_end_token_id:
                in_thinking_block = False

        input_ids = torch.cat([input_ids, torch.tensor([[next_token_item]], device=device)], dim=-1)
        
        decoded_token = tokenizer.decode([next_token_item], skip_special_tokens=True)
        print(decoded_token, end="", flush=True)
        full_text += decoded_token
        
        if next_token_item == tokenizer.eos_token_id:
            break
            
    return full_text
