#!/usr/bin/env python3
"""
Concept-Guided Beam Search implementation.
Uses concept clustering to guide beam search selection for better human-likeness.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
from typing import List, Tuple, Dict
import time

def get_top_k_candidates(model, input_ids, k: int, min_token_prob: float = 0.001) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get top-k token candidates with their probabilities."""
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
    """Get embeddings for candidate tokens."""
    embedding_matrix = model.get_output_embeddings().weight
    return embedding_matrix[top_k_ids]

def cluster_candidates(embeddings, algorithm='agglomerative', distance_threshold=0.4):
    """Cluster candidate tokens into concepts."""
    embeddings_np = embeddings.detach().cpu().float().numpy()
    
    if embeddings_np.shape[0] < 2:
        return np.array([0] * embeddings_np.shape[0])
    
    if algorithm == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    clustering.fit(embeddings_np)
    return clustering.labels_

def calculate_concept_scores(cluster_labels, token_probs, concept_ranking_method='sum'):
    """Calculate scores for each concept cluster."""
    unique_clusters = np.unique(cluster_labels)
    concept_scores = {}
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        if concept_ranking_method == 'sum':
            score = token_probs[mask].sum().item()
        elif concept_ranking_method == 'max':
            score = token_probs[mask].max().item()
        else:
            score = token_probs[mask].mean().item()
        concept_scores[cluster_id] = score
    
    return concept_scores

def concept_guided_beam_search(
    model, 
    tokenizer, 
    prompt: str, 
    max_new_tokens: int, 
    num_beams: int = 5,
    k: int = 50,
    distance_threshold: float = 0.4,
    concept_ranking_method: str = 'sum',
    min_token_prob: float = 0.001,
    concept_diversity_weight: float = 0.3,
    concept_quality_weight: float = 0.7
):
    """
    Generate text using concept-guided beam search.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams to maintain
        k: Number of top-k candidates to consider
        distance_threshold: Clustering distance threshold
        concept_ranking_method: How to rank concepts ('sum', 'max', 'mean')
        min_token_prob: Minimum token probability threshold
        concept_diversity_weight: Weight for concept diversity in beam scoring
        concept_quality_weight: Weight for concept quality in beam scoring
    """
    device = next(model.parameters()).device
    
    messages = [{"role": "system", "content": "Output one word responses to the question."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    
    # Initialize beams: (sequence, score, concept_history)
    beams = [([], 0.0, [])]  # (token_sequence, log_prob, concept_history)
    
    print("Output: ", end="", flush=True)
    
    for step in range(max_new_tokens):
        new_beams = []
        
        for beam_sequence, beam_score, concept_history in beams:
            if len(beam_sequence) == 0:
                current_input_ids = input_ids
            else:
                current_input_ids = torch.cat([input_ids, torch.tensor([beam_sequence], device=device)], dim=-1)
            
            # Get top-k candidates
            top_k_ids, top_k_probs = get_top_k_candidates(model, current_input_ids, k, min_token_prob)
            
            if top_k_ids.nelement() == 0:
                continue
            
            # Get embeddings and cluster into concepts
            embeddings = get_candidate_embeddings(model, top_k_ids)
            cluster_labels = cluster_candidates(embeddings, 'agglomerative', distance_threshold)
            concept_scores = calculate_concept_scores(cluster_labels, top_k_probs, concept_ranking_method)
            
            # Calculate concept diversity for this beam
            current_concepts = set(concept_history)
            
            # Score each candidate token
            for i, (token_id, token_prob) in enumerate(zip(top_k_ids, top_k_probs)):
                cluster_id = cluster_labels[i]
                concept_score = concept_scores.get(cluster_id, 0.0)
                
                # Calculate concept diversity score
                if cluster_id in current_concepts:
                    diversity_score = 0.0  # Penalize repeating concepts
                else:
                    diversity_score = 1.0  # Reward new concepts
                
                # Combined scoring: concept quality + diversity
                combined_score = (
                    concept_quality_weight * concept_score +
                    concept_diversity_weight * diversity_score
                )
                
                # Final beam score (combine with token probability)
                new_beam_score = beam_score + torch.log(token_prob) + combined_score
                
                # Create new beam
                new_sequence = beam_sequence + [token_id.item()]
                new_concept_history = concept_history + [cluster_id]
                
                new_beams.append((new_sequence, new_beam_score.item(), new_concept_history))
        
        # Select top beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:num_beams]
        
        if not beams:
            break
        
        # Print the best token from the best beam
        best_beam = beams[0]
        if best_beam[0]:
            last_token = best_beam[0][-1]
            decoded_token = tokenizer.decode([last_token], skip_special_tokens=True)
            print(decoded_token, end="", flush=True)
    
    # Return the best beam's sequence
    if beams:
        best_sequence = beams[0][0]
        output_ids = best_sequence
        return tokenizer.decode(output_ids, skip_special_tokens=True)
    else:
        return ""

def generate_with_standard_beam_search(model, tokenizer, prompt, max_new_tokens, num_beams=5):
    """Generate text using standard beam search for comparison."""
    device = next(model.parameters()).device
    
    messages = [{"role": "system", "content": "Output one word responses to the question."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=2,
            do_sample=False
        )
    
    output_ids = generated_ids[0][len(input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def compare_concept_guided_vs_standard():
    """Compare concept-guided beam search with standard beam search."""
    
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test prompt
    prompt = "Name something parents would criticize their children for having."
    max_new_tokens = 20
    
    print(f"\nPrompt: {prompt}")
    print("="*80)
    
    # Test different concept-guided beam search configurations
    configs = [
        {
            'name': 'Standard Beam Search',
            'function': lambda: generate_with_standard_beam_search(model, tokenizer, prompt, max_new_tokens, 5),
            'params': {}
        },
        {
            'name': 'Concept-Guided (Balanced)',
            'function': lambda: concept_guided_beam_search(
                model, tokenizer, prompt, max_new_tokens, 
                num_beams=5, k=50, distance_threshold=0.4,
                concept_ranking_method='sum',
                concept_diversity_weight=0.3, concept_quality_weight=0.7
            ),
            'params': {'diversity_weight': 0.3, 'quality_weight': 0.7}
        },
        {
            'name': 'Concept-Guided (Diversity-Focused)',
            'function': lambda: concept_guided_beam_search(
                model, tokenizer, prompt, max_new_tokens, 
                num_beams=5, k=50, distance_threshold=0.4,
                concept_ranking_method='sum',
                concept_diversity_weight=0.7, concept_quality_weight=0.3
            ),
            'params': {'diversity_weight': 0.7, 'quality_weight': 0.3}
        },
        {
            'name': 'Concept-Guided (Quality-Focused)',
            'function': lambda: concept_guided_beam_search(
                model, tokenizer, prompt, max_new_tokens, 
                num_beams=5, k=50, distance_threshold=0.4,
                concept_ranking_method='sum',
                concept_diversity_weight=0.1, concept_quality_weight=0.9
            ),
            'params': {'diversity_weight': 0.1, 'quality_weight': 0.9}
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\n[{config['name']}]")
        print("-" * 50)
        
        start_time = time.time()
        output = config['function']()
        generation_time = time.time() - start_time
        
        print(f"\nOutput: {output}")
        print(f"Time: {generation_time:.2f}s")
        
        results[config['name']] = {
            'output': output,
            'time': generation_time,
            'params': config['params']
        }
    
    # BERTScore evaluation
    print("\n" + "="*80)
    print("BERTScore Evaluation")
    print("="*80)
    
    from bertscore_evaluator import BERTScoreEvaluator, get_human_references_for_prompt
    
    evaluator = BERTScoreEvaluator()
    human_references = get_human_references_for_prompt(prompt)
    
    print(f"\n{'Method':<35} {'Best F1':<10} {'Avg F1':<10} {'Best Precision':<15} {'Best Recall':<12} {'Time':<8}")
    print("-" * 100)
    
    best_method = None
    best_f1 = -1
    
    for method_name, result in results.items():
        eval_result = evaluator.evaluate_with_multiple_references(result['output'].strip(), human_references)
        
        print(f"{method_name:<35} {eval_result['best']['f1']:<10.4f} {eval_result['average']['f1']:<10.4f} "
              f"{eval_result['best']['precision']:<15.4f} {eval_result['best']['recall']:<12.4f} {result['time']:<8.2f}s")
        
        if eval_result['best']['f1'] > best_f1:
            best_f1 = eval_result['best']['f1']
            best_method = method_name
    
    print(f"\n🏆 Best Method: {best_method} (F1: {best_f1:.4f})")
    
    # Improvement analysis
    standard_f1 = None
    for method_name, result in results.items():
        if 'Standard Beam Search' in method_name:
            eval_result = evaluator.evaluate_with_multiple_references(result['output'].strip(), human_references)
            standard_f1 = eval_result['best']['f1']
            break
    
    if standard_f1:
        print(f"\nImprovement Analysis (vs Standard Beam Search):")
        for method_name, result in results.items():
            if 'Standard Beam Search' not in method_name:
                eval_result = evaluator.evaluate_with_multiple_references(result['output'].strip(), human_references)
                improvement = eval_result['best']['f1'] - standard_f1
                print(f"{method_name}: {improvement:+.4f}")
    
    print(f"\nHuman References: {human_references}")
    
    return results

if __name__ == "__main__":
    compare_concept_guided_vs_standard()
