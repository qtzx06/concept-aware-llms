#!/usr/bin/env python3
"""
Analyze Concept Clusters
Integrates cluster coherence analysis with concept decoder to measure clustering quality.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from cluster_coherence_analyzer import ClusterCoherenceAnalyzer
from concept_decoder import generate_with_concept_decoder, setup_logger
import time
from typing import List, Dict, Tuple

def get_top_k_candidates(model, tokenizer, input_ids, k: int, min_token_prob: float = 0.001):
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

def analyze_concept_decoder_clustering(prompt: str, max_new_tokens: int = 20, k: int = 50, 
                                     distance_threshold: float = 0.4):
    """
    Analyze clustering quality during concept decoder generation.
    
    Args:
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        k: Number of top-k candidates
        distance_threshold: Clustering distance threshold
    """
    print("Loading model and analyzer...")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    
    # Initialize coherence analyzer
    analyzer = ClusterCoherenceAnalyzer("Qwen/Qwen3-0.6B")
    
    print(f"\nPrompt: {prompt}")
    print("="*80)
    
    # Prepare input
    messages = [{"role": "system", "content": "Output one word responses to the question."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)
    
    clustering_analyses = []
    
    print("Generating with concept decoder and analyzing clusters...")
    print("Output: ", end="", flush=True)
    
    for step in range(max_new_tokens):
        # Get top-k candidates
        top_k_ids, top_k_probs = get_top_k_candidates(model, tokenizer, input_ids, k, min_token_prob=0.001)
        
        if top_k_ids.nelement() == 0:
            break
        
        # Get embeddings and cluster
        embedding_matrix = model.get_output_embeddings().weight
        embeddings = embedding_matrix[top_k_ids]
        
        # Cluster candidates
        cluster_labels = cluster_candidates(embeddings, 'agglomerative', distance_threshold)
        
        # Get token texts
        token_texts = [tokenizer.decode([tid]) for tid in top_k_ids]
        
        # Analyze clustering quality
        analysis = analyzer.analyze_clustering_quality(
            top_k_ids.tolist(), cluster_labels, token_texts
        )
        
        # Add step information
        analysis['step'] = step + 1
        analysis['selected_token'] = None  # Will be filled after selection
        
        clustering_analyses.append(analysis)
        
        # Select next token (simplified - just pick first token for demonstration)
        next_token_id = top_k_ids[0]
        analysis['selected_token'] = tokenizer.decode([next_token_id])
        
        # Print selected token
        print(analysis['selected_token'], end="", flush=True)
        
        # Update input for next step
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=-1)
        
        # Stop if EOS token
        if next_token_id == tokenizer.eos_token_id:
            break
    
    print("\n\n" + "="*80)
    print("CLUSTERING QUALITY ANALYSIS SUMMARY")
    print("="*80)
    
    # Aggregate results
    avg_coherences = [analysis['avg_coherence'] for analysis in clustering_analyses]
    silhouette_scores = [analysis['silhouette_score'] for analysis in clustering_analyses]
    num_clusters = [analysis['num_clusters'] for analysis in clustering_analyses]
    
    print(f"\nOverall Statistics:")
    print(f"- Average Coherence: {np.mean(avg_coherences):.4f} ± {np.std(avg_coherences):.4f}")
    print(f"- Average Silhouette Score: {np.mean(silhouette_scores):.4f} ± {np.std(silhouette_scores):.4f}")
    print(f"- Average Number of Clusters: {np.mean(num_clusters):.1f} ± {np.std(num_clusters):.1f}")
    print(f"- Min Coherence: {np.min(avg_coherences):.4f}")
    print(f"- Max Coherence: {np.max(avg_coherences):.4f}")
    
    # Step-by-step analysis
    print(f"\nStep-by-Step Analysis:")
    for i, analysis in enumerate(clustering_analyses):
        print(f"\nStep {analysis['step']}:")
        print(f"  - Selected Token: '{analysis['selected_token']}'")
        print(f"  - Coherence: {analysis['avg_coherence']:.4f}")
        print(f"  - Silhouette: {analysis['silhouette_score']:.4f}")
        print(f"  - Clusters: {analysis['num_clusters']}")
        
        # Show cluster details for interesting steps
        if analysis['avg_coherence'] < 0.5 or analysis['silhouette_score'] < 0.0:
            print(f"  ⚠️  Poor clustering quality detected!")
            for cluster_id, coherence in analysis['cluster_coherence'].items():
                cluster_mask = analysis['cluster_labels'] == cluster_id
                cluster_tokens = [analysis['token_texts'][j] for j, mask in enumerate(cluster_mask) if mask]
                print(f"    Cluster {cluster_id}: {cluster_tokens[:5]} (coherence: {coherence:.4f})")
    
    # Quality assessment
    print(f"\nQuality Assessment:")
    overall_avg_coherence = np.mean(avg_coherences)
    overall_silhouette = np.mean(silhouette_scores)
    
    if overall_avg_coherence > 0.7:
        print("✅ High semantic coherence - clustering is working well")
    elif overall_avg_coherence > 0.5:
        print("⚠️  Moderate semantic coherence - some tuning may help")
    else:
        print("❌ Low semantic coherence - clustering parameters need adjustment")
    
    if overall_silhouette > 0.2:
        print("✅ Good cluster separation - distinct semantic groups")
    elif overall_silhouette > 0.0:
        print("⚠️  Moderate cluster separation - some overlap between clusters")
    else:
        print("❌ Poor cluster separation - clusters are not well-distinguished")
    
    # Recommendations
    print(f"\nRecommendations:")
    if overall_avg_coherence < 0.6:
        print("- Consider increasing distance_threshold for tighter clusters")
    if overall_silhouette < 0.1:
        print("- Consider decreasing distance_threshold for better separation")
    if np.mean(num_clusters) < 2:
        print("- Consider decreasing distance_threshold to create more clusters")
    if np.mean(num_clusters) > 8:
        print("- Consider increasing distance_threshold to reduce cluster fragmentation")
    
    return clustering_analyses

def compare_clustering_parameters(prompt: str, max_new_tokens: int = 15):
    """
    Compare clustering quality across different parameters.
    """
    print("Comparing clustering parameters...")
    
    distance_thresholds = [0.2, 0.4, 0.6, 0.8]
    results = {}
    
    for threshold in distance_thresholds:
        print(f"\n{'='*60}")
        print(f"Testing distance_threshold = {threshold}")
        print(f"{'='*60}")
        
        try:
            analysis = analyze_concept_decoder_clustering(
                prompt, max_new_tokens, k=50, distance_threshold=threshold
            )
            
            # Aggregate results
            avg_coherences = [a['avg_coherence'] for a in analysis]
            silhouette_scores = [a['silhouette_score'] for a in analysis]
            num_clusters = [a['num_clusters'] for a in analysis]
            
            results[threshold] = {
                'avg_coherence': np.mean(avg_coherences),
                'silhouette_score': np.mean(silhouette_scores),
                'avg_clusters': np.mean(num_clusters),
                'std_coherence': np.std(avg_coherences),
                'std_silhouette': np.std(silhouette_scores)
            }
            
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")
            results[threshold] = None
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("PARAMETER COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Threshold':<12} {'Avg Coherence':<15} {'Silhouette':<12} {'Avg Clusters':<12}")
    print("-" * 60)
    
    for threshold, result in results.items():
        if result is not None:
            print(f"{threshold:<12} {result['avg_coherence']:<15.4f} {result['silhouette_score']:<12.4f} {result['avg_clusters']:<12.1f}")
    
    # Find best parameters
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_coherence = max(valid_results.items(), key=lambda x: x[1]['avg_coherence'])
        best_silhouette = max(valid_results.items(), key=lambda x: x[1]['silhouette_score'])
        
        print(f"\nBest Parameters:")
        print(f"- Best Coherence: distance_threshold = {best_coherence[0]} (coherence: {best_coherence[1]['avg_coherence']:.4f})")
        print(f"- Best Silhouette: distance_threshold = {best_silhouette[0]} (silhouette: {best_silhouette[1]['silhouette_score']:.4f})")
    
    return results

if __name__ == "__main__":
    # Test with a sample prompt
    test_prompt = "Name something parents would criticize their children for having."
    
    print("Testing cluster coherence analysis...")
    analyze_concept_decoder_clustering(test_prompt, max_new_tokens=10)
    
    print("\n" + "="*80)
    print("Comparing different clustering parameters...")
    compare_clustering_parameters(test_prompt, max_new_tokens=8)
