#!/usr/bin/env python3
"""
Test script for BERTScore evaluation functionality.
This script demonstrates how to use the BERTScore evaluator to compare
different decoding methods against human intuition.
"""

from bertscore_evaluator import BERTScoreEvaluator, get_human_references_for_prompt

def test_bertscore_evaluation():
    """Test the BERTScore evaluation with sample outputs."""
    
    # Sample prompt and outputs
    prompt = "Name something parents would criticize their children for having."
    
    # Simulated outputs from different decoding methods
    baseline_output = "lazy attitude and not doing homework"
    concept_output = "messy room and bad grades"
    
    print("Testing BERTScore Evaluation")
    print("="*50)
    print(f"Prompt: {prompt}")
    print(f"Baseline Output: {baseline_output}")
    print(f"Concept Decoder Output: {concept_output}")
    print()
    
    try:
        # Initialize evaluator
        evaluator = BERTScoreEvaluator()
        
        # Get human references
        references = get_human_references_for_prompt(prompt)
        print(f"Human References: {references}")
        print()
        
        # Perform evaluation
        comparison = evaluator.compare_decoding_methods(
            baseline_output=baseline_output,
            concept_output=concept_output,
            references=references,
            prompt=prompt
        )
        
        # Print results
        evaluator.print_comparison_summary(comparison)
        
        # Save results
        results_file = evaluator.save_evaluation_results(comparison)
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        print("Make sure bert-score is installed: pip install bert-score")

def test_single_evaluation():
    """Test single pair evaluation."""
    
    print("\n" + "="*50)
    print("Testing Single Pair Evaluation")
    print("="*50)
    
    try:
        evaluator = BERTScoreEvaluator()
        
        candidate = "messy room"
        reference = "bad grades"
        
        result = evaluator.evaluate_single_pair(candidate, reference)
        
        print(f"Candidate: {candidate}")
        print(f"Reference: {reference}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1: {result['f1']:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_bertscore_evaluation()
    test_single_evaluation()
