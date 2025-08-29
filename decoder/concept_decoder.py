import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import numpy as np
from typing import Tuple, List, Dict, Optional
import logging
import os
import re

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
    device = logits.device
    probs = torch.softmax(logits, dim=-1)
    
    top_k_ids = top_k_ids.squeeze()
    top_k_probs = top_k_probs.squeeze()

    if min_token_prob > 0:
        mask = top_k_probs >= min_token_prob
        top_k_ids = top_k_ids[mask]
        top_k_probs = top_k_probs[mask]

    return top_k_ids, top_k_probs

def is_word_boundary(tokenizer, token_text: str) -> bool:
    """
    Determine if a token represents a word boundary.
    This is tokenizer-specific heuristic.
    """
    if not token_text:
        return False
    
    # Common patterns for word boundaries
    # Tokens that start with space or are standalone
    if token_text.startswith(' ') or token_text.startswith('Ġ'):  # GPT-style space prefix
        return True
    
    # For tokenizers that don't use space prefixes, check if it's likely a word start
    # This is a heuristic - tokens that are alphabetic and don't look like subword continuations
    if token_text.isalpha() and len(token_text) > 1:
        return True
    
    # Special tokens
    if token_text.startswith('<') and token_text.endswith('>'):
        return True
        
    return False

def group_tokens_into_words(tokenizer, token_ids: torch.Tensor, token_probs: torch.Tensor) -> List[Dict]:
    """
    Group consecutive tokens into complete words.
    
    Returns:
        List of word dictionaries containing:
        - 'tokens': list of token IDs in the word
        - 'text': decoded word text
        - 'prob': combined probability
        - 'embedding_indices': indices in original token list
    """
    words = []
    current_word = {
        'tokens': [],
        'text': '',
        'prob': 1.0,
        'embedding_indices': []
    }
    
    for i, (token_id, token_prob) in enumerate(zip(token_ids, token_probs)):
        token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True)
        
        # Check if this token starts a new word
        is_boundary = is_word_boundary(tokenizer, token_text)
        
        if is_boundary and current_word['tokens']:
            # Finish current word and start new one
            current_word['text'] = tokenizer.decode(current_word['tokens'], skip_special_tokens=True).strip()
            if current_word['text']:  # Only add non-empty words
                words.append(current_word)
            
            # Start new word
            current_word = {
                'tokens': [token_id.item()],
                'text': '',
                'prob': token_prob.item(),
                'embedding_indices': [i]
            }
        else:
            # Continue current word
            current_word['tokens'].append(token_id.item())
            current_word['prob'] *= token_prob.item()  # Multiply probabilities for word
            current_word['embedding_indices'].append(i)
    
    # Don't forget the last word
    if current_word['tokens']:
        current_word['text'] = tokenizer.decode(current_word['tokens'], skip_special_tokens=True).strip()
        if current_word['text']:
            words.append(current_word)
    
    # Fallback: if no words detected, treat each token as a separate word
    if not words:
        for i, (token_id, token_prob) in enumerate(zip(token_ids, token_probs)):
            token_text = tokenizer.decode([token_id.item()], skip_special_tokens=True).strip()
            if token_text:
                words.append({
                    'tokens': [token_id.item()],
                    'text': token_text,
                    'prob': token_prob.item(),
                    'embedding_indices': [i]
                })
    
    return words

def get_word_embeddings(model, words: List[Dict], token_embeddings: torch.Tensor) -> torch.Tensor:
    """
    Get embeddings for words by combining token embeddings.
    Uses average pooling of constituent token embeddings.
    """
    word_embeddings = []
    
    for word in words:
        # Get embeddings for all tokens in this word
        token_indices = word['embedding_indices']
        word_token_embeddings = token_embeddings[token_indices]
        
        # Average pooling (you could also try max pooling or weighted average)
        if len(word_token_embeddings.shape) > 1:
            word_embedding = torch.mean(word_token_embeddings, dim=0)
        else:
            word_embedding = word_token_embeddings
            
        word_embeddings.append(word_embedding)
    
    return torch.stack(word_embeddings)

def get_candidate_embeddings(model, top_k_ids):
    """Gets token embeddings from the model with proper device handling"""
    embedding_matrix = model.get_output_embeddings().weight
    # Ensure top_k_ids is on same device as embedding matrix
    if top_k_ids.device != embedding_matrix.device:
        top_k_ids = top_k_ids.to(embedding_matrix.device)
    return embedding_matrix[top_k_ids]

def reduce_dimensionality(embeddings, n_components=32):
    embeddings_np = embeddings.detach().cpu().float().numpy()
    if embeddings_np.shape[0] <= n_components:
        return embeddings
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings_np)
    return torch.tensor(reduced_embeddings, device=embeddings.device)

def cluster_candidates(embeddings, algorithm='agglomerative', distance_threshold=0.4, eps=0.5) -> np.ndarray:
    """Clusters embeddings using the specified algorithm"""
    embeddings_np = embeddings.detach().cpu().float().numpy()
    if algorithm == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    elif algorithm == 'dbscan':
        clustering = DBSCAN(eps=eps, min_samples=2)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    clustering.fit(embeddings_np)
    return clustering.labels_

def select_word_from_cluster(words: List[Dict], cluster_indices: List[int], ranking_method: str = 'sum') -> Dict:
    """
    Select the best word from a cluster based on the ranking method.
    """
    cluster_words = [words[i] for i in cluster_indices]
    
    if ranking_method == 'sum':
        # Select word with highest probability
        best_word = max(cluster_words, key=lambda w: w['prob'])
    elif ranking_method == 'max':
        # Could implement other ranking methods here
        best_word = max(cluster_words, key=lambda w: w['prob'])
    else:
        # Default to first word
        best_word = cluster_words[0]
    
    return best_word

def generate_with_concept_decoder(
    model, tokenizer, prompt: str, max_new_tokens: int, k: int, 
    distance_threshold: float, enable_thinking: bool,
    clustering_algo: str, eps: float, use_dim_reduction: bool,
    dim_reduction_components: int, concept_ranking_method: str, min_token_prob: float,
    logger: logging.Logger
) -> str:
    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    think_end_token_id = 151668

    messages = [{"role": "system", "content": "Name something parents would criticize their children for having."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    
    full_text = ""
    in_thinking_block = False
    thinking_token_count = 0

    print("Output: ", end="", flush=True)

    for step in range(max_new_tokens):
        logger.info(f"\n{'='*20} Step {step+1}/{max_new_tokens} {'='*20}")
        
        # Step 1: Get top-k token candidates
        top_k_ids, top_k_probs = get_top_k_candidates(model, tokenizer, input_ids, k, min_token_prob)
        
        if top_k_ids.nelement() == 0:
            print("\n[Warning: No candidates left after filtering. Stopping.]")
            logger.warning("No candidates left after filtering.")
            break

        logger.info(f"\n--- Top {len(top_k_ids)} Token Candidates (after prob filter) ---")
        for token_id, prob in zip(top_k_ids, top_k_probs):
            logger.info(f"Token: '{tokenizer.decode(token_id)}' (ID: {token_id.item()}), Prob: {prob.item():.4f}")

        # Step 2: Group tokens into words
        words = group_tokens_into_words(tokenizer, top_k_ids, top_k_probs)
        
        logger.info(f"\n--- Word Grouping Results ---")
        logger.info(f"Grouped {len(top_k_ids)} tokens into {len(words)} words:")
        for i, word in enumerate(words):
            logger.info(f"Word {i}: '{word['text']}' (tokens: {len(word['tokens'])}, prob: {word['prob']:.4f})")

        if len(words) < 2:
            # Not enough words for clustering, select the best one
            selected_word = words[0] if words else None
            if selected_word:
                next_token_id = selected_word['tokens'][0]  # Use first token of the word
                logger.info(f"\nOnly {len(words)} word(s) available. Selecting: '{selected_word['text']}'")
            else:
                # Fallback to first token
                next_token_id = top_k_ids[0]
                logger.info("\nNo valid words found. Falling back to first token.")
        else:
            # Step 3: Get token embeddings and create word embeddings
            token_embeddings = get_candidate_embeddings(model, top_k_ids)
            word_embeddings = get_word_embeddings(model, words, token_embeddings)
            
            # Step 4: Apply dimensionality reduction if requested
            if use_dim_reduction:
                word_embeddings = reduce_dimensionality(word_embeddings, n_components=dim_reduction_components)
                logger.info(f"Applied PCA reduction to {word_embeddings.shape[1]} components")

            # Step 5: Cluster words using the existing cluster_candidates function
            cluster_labels = cluster_candidates(word_embeddings, algorithm=clustering_algo, 
                                              distance_threshold=distance_threshold, eps=eps)
            
            logger.info(f"\n--- Word Clustering Results ({clustering_algo}) ---")
            word_clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in word_clusters:
                    word_clusters[label] = []
                word_clusters[label].append(words[i]['text'])
            
            for label, word_texts in word_clusters.items():
                logger.info(f"Cluster {label}: {word_texts}")

            # Step 6: Select concept cluster
            unique_clusters = np.unique(cluster_labels)
            if clustering_algo == 'dbscan':
                unique_clusters = unique_clusters[unique_clusters != -1]

            if len(unique_clusters) == 0:
                selected_word = words[0]
                logger.info("\nNo valid clusters found. Falling back to first word.")
            else:
                # Compute cluster probabilities based on word probabilities
                cluster_probs = torch.zeros(len(unique_clusters), device=device)
                logger.info(f"\n--- Concept Ranking ({concept_ranking_method}) ---")
                
                for j, cluster_id in enumerate(unique_clusters):
                    cluster_word_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_word_probs = [words[i]['prob'] for i in cluster_word_indices]
                    
                    if concept_ranking_method == 'sum':
                        prob = sum(cluster_word_probs)
                    elif concept_ranking_method == 'max':
                        prob = max(cluster_word_probs)
                    else:
                        prob = sum(cluster_word_probs)
                    
                    cluster_probs[j] = prob
                    cluster_words = [words[i]['text'] for i in cluster_word_indices]
                    logger.info(f"Concept (Cluster {cluster_id}): {cluster_words}, Prob: {prob:.4f}")

                # Sample from concept clusters
                if torch.sum(cluster_probs) > 0:
                    chosen_cluster_idx = torch.multinomial(cluster_probs, 1).item()
                    chosen_cluster_id = unique_clusters[chosen_cluster_idx]
                else:
                    chosen_cluster_id = unique_clusters[0]
                
                logger.info(f"\nSelected Concept: Cluster {chosen_cluster_id}")

                # Step 7: Select word from chosen cluster
                cluster_word_indices = [i for i, label in enumerate(cluster_labels) if label == chosen_cluster_id]
                selected_word = select_word_from_cluster(words, cluster_word_indices, concept_ranking_method)
                
                logger.info(f"Selected Word from Cluster: '{selected_word['text']}' (prob: {selected_word['prob']:.4f})")

            # Use the first token of the selected word
            next_token_id = selected_word['tokens'][0]
        
        next_token_item = next_token_id.item() if hasattr(next_token_id, 'item') else next_token_id
        logger.info(f"Next Token: '{tokenizer.decode([next_token_item])}' (ID: {next_token_item})")

        # Handle thinking mode
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

        # Update input_ids and generate output
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_item]], device=device)], dim=-1)
        
        decoded_token = tokenizer.decode([next_token_item], skip_special_tokens=True)
        print(decoded_token, end="", flush=True)
        full_text += decoded_token
        
        if next_token_item == tokenizer.eos_token_id:
            break
            
    return full_text