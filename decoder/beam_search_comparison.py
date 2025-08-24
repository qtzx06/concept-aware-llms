#!/usr/bin/env python3
"""
Comparison script to test if beam search improves results over concept-aware sampling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bertscore_evaluator import BERTScoreEvaluator, get_human_references_for_prompt
from concept_decoder import generate_with_concept_decoder, setup_logger
import time

def generate_with_beam_search(model, tokenizer, prompt, max_new_tokens, num_beams=5):
    """Generate text using beam search."""
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
            temperature=0.7,
            do_sample=False  # Deterministic beam search
        )
    
    output_ids = generated_ids[0][len(input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def generate_with_sampling(model, tokenizer, prompt, max_new_tokens):
    """Generate text using standard sampling (baseline)."""
    device = next(model.parameters()).device
    
    messages = [{"role": "system", "content": "Output one word responses to the question."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            do_sample=True
        )
    
    output_ids = generated_ids[0][len(input_ids[0]):].tolist()
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def compare_methods():
    """Compare beam search, sampling, and concept decoder."""
    
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
    
    # Generate with different methods
    print("\n[1] STANDARD SAMPLING")
    start_time = time.time()
    sampling_output = generate_with_sampling(model, tokenizer, prompt, max_new_tokens)
    sampling_time = time.time() - start_time
    print(f"Output: {sampling_output}")
    print(f"Time: {sampling_time:.2f}s")
    
    print("\n[2] BEAM SEARCH (num_beams=5)")
    start_time = time.time()
    beam_output = generate_with_beam_search(model, tokenizer, prompt, max_new_tokens, num_beams=5)
    beam_time = time.time() - start_time
    print(f"Output: {beam_output}")
    print(f"Time: {beam_time:.2f}s")
    
    print("\n[3] CONCEPT-AWARE DECODER")
    start_time = time.time()
    logger = setup_logger()
    concept_output = generate_with_concept_decoder(
        model, tokenizer, prompt, max_new_tokens, k=50, 
        distance_threshold=0.4, enable_thinking=True, clustering_algo="agglomerative", 
        eps=0.5, use_dim_reduction=False, dim_reduction_components=32,
        concept_ranking_method="sum", min_token_prob=0.001, logger=logger
    )
    concept_time = time.time() - start_time
    print(f"Output: {concept_output}")
    print(f"Time: {concept_time:.2f}s")
    
    # BERTScore evaluation
    print("\n[4] BERTScore Evaluation")
    print("="*50)
    
    evaluator = BERTScoreEvaluator()
    human_references = get_human_references_for_prompt(prompt)
    
    # Evaluate each method
    sampling_eval = evaluator.evaluate_with_multiple_references(sampling_output.strip(), human_references)
    beam_eval = evaluator.evaluate_with_multiple_references(beam_output.strip(), human_references)
    concept_eval = evaluator.evaluate_with_multiple_references(concept_output.strip(), human_references)
    
    # Print results
    print("\n" + "="*80)
    print("BERTScore Comparison Results")
    print("="*80)
    
    print(f"\n{'Method':<20} {'Best F1':<10} {'Avg F1':<10} {'Best Precision':<15} {'Best Recall':<12} {'Time':<8}")
    print("-" * 80)
    
    print(f"{'Sampling':<20} {sampling_eval['best']['f1']:<10.4f} {sampling_eval['average']['f1']:<10.4f} "
          f"{sampling_eval['best']['precision']:<15.4f} {sampling_eval['best']['recall']:<12.4f} {sampling_time:<8.2f}s")
    
    print(f"{'Beam Search':<20} {beam_eval['best']['f1']:<10.4f} {beam_eval['average']['f1']:<10.4f} "
          f"{beam_eval['best']['precision']:<15.4f} {beam_eval['best']['recall']:<12.4f} {beam_time:<8.2f}s")
    
    print(f"{'Concept Decoder':<20} {concept_eval['best']['f1']:<10.4f} {concept_eval['average']['f1']:<10.4f} "
          f"{concept_eval['best']['precision']:<15.4f} {concept_eval['best']['recall']:<12.4f} {concept_time:<8.2f}s")
    
    # Find best method
    methods = [
        ("Sampling", sampling_eval['best']['f1']),
        ("Beam Search", beam_eval['best']['f1']),
        ("Concept Decoder", concept_eval['best']['f1'])
    ]
    best_method = max(methods, key=lambda x: x[1])
    
    print(f"\n🏆 Best Method: {best_method[0]} (F1: {best_method[1]:.4f})")
    
    # Improvement analysis
    print(f"\nImprovement Analysis:")
    print(f"Beam Search vs Sampling: {beam_eval['best']['f1'] - sampling_eval['best']['f1']:+.4f}")
    print(f"Concept Decoder vs Sampling: {concept_eval['best']['f1'] - sampling_eval['best']['f1']:+.4f}")
    print(f"Concept Decoder vs Beam Search: {concept_eval['best']['f1'] - beam_eval['best']['f1']:+.4f}")
    
    print(f"\nHuman References: {human_references}")
    
    return {
        'sampling': sampling_eval,
        'beam': beam_eval,
        'concept': concept_eval,
        'times': {'sampling': sampling_time, 'beam': beam_time, 'concept': concept_time}
    }

if __name__ == "__main__":
    compare_methods()
