import torch
from bert_score import BERTScorer
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import json
import os
from datetime import datetime

class BERTScoreEvaluator:
    """
    A class to evaluate generated text against human reference responses using BERTScore.
    This helps assess how well the outputs match human intuition.
    """
    
    def __init__(self, model_type: str = "bert-base-uncased", device: str = None):
        """
        Initialize the BERTScore evaluator.
        
        Args:
            model_type: The BERT model to use for scoring (default: microsoft/deberta-v3-large)
            device: Device to run the model on (default: auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_type = model_type
        self.scorer = None
        self.logger = logging.getLogger('BERTScoreEvaluator')
        
    def _load_scorer(self):
        """Load the BERTScore model if not already loaded."""
        if self.scorer is None:
            self.logger.info(f"Loading BERTScore model: {self.model_type}")
            self.scorer = BERTScorer(model_type=self.model_type, device=self.device)
            self.logger.info("BERTScore model loaded successfully")
    
    def evaluate_single_pair(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        Evaluate a single candidate-reference pair.
        
        Args:
            candidate: The generated text to evaluate
            reference: The human reference text
            
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        self._load_scorer()
        
        P, R, F1 = self.scorer.score([candidate], [reference])
        
        return {
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item(),
            'candidate': candidate,
            'reference': reference
        }
    
    def evaluate_multiple_pairs(self, candidates: List[str], references: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate multiple candidate-reference pairs.
        
        Args:
            candidates: List of generated texts
            references: List of human reference texts
            
        Returns:
            Dictionary containing lists of precision, recall, and F1 scores
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates must match number of references")
        
        self._load_scorer()
        
        P, R, F1 = self.scorer.score(candidates, references)
        
        return {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist(),
            'candidates': candidates,
            'references': references
        }
    
    def evaluate_with_multiple_references(self, candidate: str, references: List[str]) -> Dict[str, float]:
        """
        Evaluate a candidate against multiple reference responses.
        
        Args:
            candidate: The generated text to evaluate
            references: List of human reference texts
            
        Returns:
            Dictionary containing best and average scores
        """
        self._load_scorer()
        
        # Score against each reference
        P, R, F1 = self.scorer.score([candidate] * len(references), references)
        
        scores = {
            'precision': P.tolist(),
            'recall': R.tolist(),
            'f1': F1.tolist()
        }
        
        # Calculate best and average scores
        best_scores = {
            'precision': max(scores['precision']),
            'recall': max(scores['recall']),
            'f1': max(scores['f1'])
        }
        
        avg_scores = {
            'precision': np.mean(scores['precision']),
            'recall': np.mean(scores['recall']),
            'f1': np.mean(scores['f1'])
        }
        
        return {
            'best': best_scores,
            'average': avg_scores,
            'all_scores': scores,
            'candidate': candidate,
            'references': references
        }
    
    def compare_decoding_methods(self, 
                                baseline_output: str, 
                                concept_output: str, 
                                references: List[str],
                                prompt: str) -> Dict[str, Dict]:
        """
        Compare baseline and concept decoder outputs against human references.
        
        Args:
            baseline_output: Output from standard decoding
            concept_output: Output from concept-aware decoding
            references: List of human reference responses
            prompt: The original prompt used for generation
            
        Returns:
            Dictionary containing comparison results
        """
        baseline_eval = self.evaluate_with_multiple_references(baseline_output, references)
        concept_eval = self.evaluate_with_multiple_references(concept_output, references)
        
        comparison = {
            'prompt': prompt,
            'baseline': {
                'output': baseline_output,
                'best_f1': baseline_eval['best']['f1'],
                'avg_f1': baseline_eval['average']['f1'],
                'best_precision': baseline_eval['best']['precision'],
                'best_recall': baseline_eval['best']['recall']
            },
            'concept_decoder': {
                'output': concept_output,
                'best_f1': concept_eval['best']['f1'],
                'avg_f1': concept_eval['average']['f1'],
                'best_precision': concept_eval['best']['precision'],
                'best_recall': concept_eval['best']['recall']
            },
            'improvement': {
                'f1_improvement': concept_eval['best']['f1'] - baseline_eval['best']['f1'],
                'precision_improvement': concept_eval['best']['precision'] - baseline_eval['best']['precision'],
                'recall_improvement': concept_eval['best']['recall'] - baseline_eval['best']['recall']
            },
            'human_references': references,
            'timestamp': datetime.now().isoformat()
        }
        
        return comparison
    
    def save_evaluation_results(self, results: Dict, filename: str = None):
        """
        Save evaluation results to a JSON file.
        
        Args:
            results: The evaluation results dictionary
            filename: Optional filename, defaults to timestamped name
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bertscore_evaluation_{timestamp}.json"
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation results saved to: {filepath}")
        return filepath
    
    def print_comparison_summary(self, comparison: Dict):
        """
        Print a formatted summary of the comparison results.
        
        Args:
            comparison: The comparison results dictionary
        """
        print("\n" + "="*80)
        print("BERTScore Evaluation Results")
        print("="*80)
        print(f"Prompt: {comparison['prompt']}")
        print(f"Timestamp: {comparison['timestamp']}")
        
        print("\n" + "-"*40)
        print("BASELINE DECODER")
        print("-"*40)
        print(f"Output: {comparison['baseline']['output']}")
        print(f"Best F1: {comparison['baseline']['best_f1']:.4f}")
        print(f"Avg F1: {comparison['baseline']['avg_f1']:.4f}")
        print(f"Best Precision: {comparison['baseline']['best_precision']:.4f}")
        print(f"Best Recall: {comparison['baseline']['best_recall']:.4f}")
        
        print("\n" + "-"*40)
        print("CONCEPT DECODER")
        print("-"*40)
        print(f"Output: {comparison['concept_decoder']['output']}")
        print(f"Best F1: {comparison['concept_decoder']['best_f1']:.4f}")
        print(f"Avg F1: {comparison['concept_decoder']['avg_f1']:.4f}")
        print(f"Best Precision: {comparison['concept_decoder']['best_precision']:.4f}")
        print(f"Best Recall: {comparison['concept_decoder']['best_recall']:.4f}")
        
        print("\n" + "-"*40)
        print("IMPROVEMENT")
        print("-"*40)
        f1_improvement = comparison['improvement']['f1_improvement']
        precision_improvement = comparison['improvement']['precision_improvement']
        recall_improvement = comparison['improvement']['recall_improvement']
        
        print(f"F1 Improvement: {f1_improvement:+.4f}")
        print(f"Precision Improvement: {precision_improvement:+.4f}")
        print(f"Recall Improvement: {recall_improvement:+.4f}")
        
        if f1_improvement > 0:
            print("✅ Concept decoder performs better than baseline")
        elif f1_improvement < 0:
            print("❌ Baseline performs better than concept decoder")
        else:
            print("➖ Both methods perform similarly")
        
        print("\n" + "-"*40)
        print("HUMAN REFERENCES")
        print("-"*40)
        for i, ref in enumerate(comparison['human_references'], 1):
            print(f"{i}. {ref}")
        
        print("="*80)


def get_human_references_for_prompt(prompt: str) -> List[str]:
    """
    Get human reference responses for a given prompt.
    This is a simple implementation - in practice, you might want to use
    a database or API to get real human responses.
    
    Args:
        prompt: The prompt to get references for
        
    Returns:
        List of human reference responses
    """
    # This is a simple mapping - in practice, you'd want a more sophisticated system
    reference_mappings = {
        "Name something parents would criticize their children for having.": [
            "messy room",
            "bad grades",
            "lazy attitude",
            "disrespectful behavior",
            "spending too much time on phone",
            "not doing homework",
            "talking back",
            "being late",
            "not cleaning up",
            "wasting money"
        ],
        "What is something people often forget to bring when traveling?": [
            "phone charger",
            "toothbrush",
            "passport",
            "medication",
            "underwear",
            "phone",
            "wallet",
            "keys",
            "toothpaste",
            "deodorant"
        ],
        "Name a food that people either love or hate.": [
            "cilantro",
            "blue cheese",
            "olives",
            "anchovies",
            "licorice",
            "marmite",
            "durian",
            "black licorice",
            "brussels sprouts",
            "liver"
        ]
    }
    
    # Try exact match first
    if prompt in reference_mappings:
        return reference_mappings[prompt]
    
    # Try partial matching for similar prompts
    for key_prompt, references in reference_mappings.items():
        if any(word in prompt.lower() for word in key_prompt.lower().split()):
            return references
    
    # Default references for unknown prompts
    return [
        "common response 1",
        "common response 2", 
        "common response 3",
        "common response 4",
        "common response 5"
    ]
