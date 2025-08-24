#!/usr/bin/env python3
"""
Test script for the travel prompt: "What is something people often forget to bring when traveling?"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from bertscore_evaluator import BERTScoreEvaluator, get_human_references_for_prompt
from concept_decoder import generate_with_concept_decoder, setup_logger
from concept_guided_beam_search import concept_guided_beam_search, generate_with_standard_beam_search
import time

def generate_with_sampling(model, tokenizer, prompt, max_new_tokens):
    """Generate text using standard sampling."""
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

def test_travel_prompt():
    """Test all methods with the travel prompt."""
    
    print("Loading model and tokenizer...")
    model_name = "Qwen/Qwen3-0.6B"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test prompt
    prompt = "What is something people often forget to bring when traveling?"
    max_new_tokens = 20
    
    print(f"\nPrompt: {prompt}")
    print("="*80)
    
    # Define all methods to test
    methods = [
        {
            'name': 'Standard Sampling',
            'function': lambda: generate_with_sampling(model, tokenizer, prompt, max_new_tokens),
            'description': 'Traditional sampling with temperature and top-p'
        },
        {
            'name': 'Standard Beam Search',
            'function': lambda: generate_with_standard_beam_search(model, tokenizer, prompt, max_new_tokens, 5),
            'description': 'Traditional beam search with 5 beams'
        },
        {
            'name': 'Concept Decoder',
            'function': lambda: generate_with_concept_decoder(
                model, tokenizer, prompt, max_new_tokens, k=50, 
                distance_threshold=0.4, enable_thinking=True, clustering_algo="agglomerative", 
                eps=0.5, use_dim_reduction=False, dim_reduction_components=32,
                concept_ranking_method="sum", min_token_prob=0.001, logger=setup_logger()
            ),
            'description': 'Concept-aware sampling with clustering'
        },
        {
            'name': 'Concept-Guided Beam (Balanced)',
            'function': lambda: concept_guided_beam_search(
                model, tokenizer, prompt, max_new_tokens, 
                num_beams=5, k=50, distance_threshold=0.4,
                concept_ranking_method='sum',
                concept_diversity_weight=0.3, concept_quality_weight=0.7
            ),
            'description': 'Beam search guided by concept clustering (balanced)'
        },
        {
            'name': 'Concept-Guided Beam (Diversity)',
            'function': lambda: concept_guided_beam_search(
                model, tokenizer, prompt, max_new_tokens, 
                num_beams=5, k=50, distance_threshold=0.4,
                concept_ranking_method='sum',
                concept_diversity_weight=0.7, concept_quality_weight=0.3
            ),
            'description': 'Beam search guided by concept clustering (diversity-focused)'
        }
    ]
    
    results = {}
    
    # Run all methods
    for i, method in enumerate(methods, 1):
        print(f"\n[{i}] {method['name']}")
        print(f"Description: {method['description']}")
        print("-" * 60)
        
        start_time = time.time()
        output = method['function']()
        generation_time = time.time() - start_time
        
        print(f"Output: {output}")
        print(f"Time: {generation_time:.2f}s")
        
        results[method['name']] = {
            'output': output,
            'time': generation_time,
            'description': method['description']
        }
    
    # BERTScore evaluation
    print("\n" + "="*100)
    print("BERTScore Evaluation Results")
    print("="*100)
    
    evaluator = BERTScoreEvaluator()
    human_references = get_human_references_for_prompt(prompt)
    
    print(f"\n{'Method':<30} {'Best F1':<10} {'Avg F1':<10} {'Best Precision':<15} {'Best Recall':<12} {'Time':<8}")
    print("-" * 100)
    
    best_method = None
    best_f1 = -1
    baseline_f1 = None
    
    for method_name, result in results.items():
        eval_result = evaluator.evaluate_with_multiple_references(result['output'].strip(), human_references)
        
        print(f"{method_name:<30} {eval_result['best']['f1']:<10.4f} {eval_result['average']['f1']:<10.4f} "
              f"{eval_result['best']['precision']:<15.4f} {eval_result['best']['recall']:<12.4f} {result['time']:<8.2f}s")
        
        if eval_result['best']['f1'] > best_f1:
            best_f1 = eval_result['best']['f1']
            best_method = method_name
        
        if method_name == 'Standard Sampling':
            baseline_f1 = eval_result['best']['f1']
    
    print(f"\n🏆 Best Method: {best_method} (F1: {best_f1:.4f})")
    
    # Improvement analysis
    if baseline_f1:
        print(f"\nImprovement Analysis (vs Standard Sampling baseline):")
        for method_name, result in results.items():
            eval_result = evaluator.evaluate_with_multiple_references(result['output'].strip(), human_references)
            improvement = eval_result['best']['f1'] - baseline_f1
            print(f"{method_name}: {improvement:+.4f}")
    
    # Performance vs Speed analysis
    print(f"\nPerformance vs Speed Analysis:")
    for method_name, result in results.items():
        eval_result = evaluator.evaluate_with_multiple_references(result['output'].strip(), human_references)
        f1_score = eval_result['best']['f1']
        speed = result['time']
        efficiency = f1_score / speed  # F1 per second
        print(f"{method_name}: F1={f1_score:.4f}, Time={speed:.2f}s, Efficiency={efficiency:.4f} F1/s")
    
    # Recommendations
    print(f"\n" + "="*100)
    print("Recommendations")
    print("="*100)
    
    # Find best performance
    best_performance = max(results.items(), key=lambda x: 
        evaluator.evaluate_with_multiple_references(x[1]['output'].strip(), human_references)['best']['f1'])
    
    # Find best efficiency
    best_efficiency = max(results.items(), key=lambda x: 
        evaluator.evaluate_with_multiple_references(x[1]['output'].strip(), human_references)['best']['f1'] / x[1]['time'])
    
    print(f"🎯 Best Performance: {best_performance[0]} (F1: {evaluator.evaluate_with_multiple_references(best_performance[1]['output'].strip(), human_references)['best']['f1']:.4f})")
    print(f"⚡ Best Efficiency: {best_efficiency[0]} (F1/s: {evaluator.evaluate_with_multiple_references(best_efficiency[1]['output'].strip(), human_references)['best']['f1'] / best_efficiency[1]['time']:.4f})")
    
    print(f"\nHuman References: {human_references}")
    
    return results

if __name__ == "__main__":
    test_travel_prompt()
