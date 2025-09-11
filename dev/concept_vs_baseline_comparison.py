#!/usr/bin/env python3
"""
Concept-Aware vs Baseline Comparison using Pairwise LLM Judge
This script generates outputs from both baseline and concept-aware models,
then uses Gemini to compare them pairwise (LLMArena style).
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pairwise_llm_judge import PairwiseLLMJudge
from realistic_qwen_baseline import RealisticQwenBaseline
from src.data.folio_loader import FolioLoader
from src.data.truthfulqa_loader import TruthfulQALoader
from src.models.simple_concept_decoder import create_concept_decoder


class ConceptVsBaselineComparison:
    """
    System that compares concept-aware decoding vs baseline using pairwise LLM judge.
    """
    
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B", device="auto"):
        """
        Initialize the comparison system.
        
        Args:
            model_name: Name of the model to use
            device: Device to run the model on
        """
        self.model_name = model_name
        self.device = device
        
        # Initialize the pairwise judge
        self.judge = PairwiseLLMJudge()
        
        # Initialize the baseline generator (use actual PyTorch model)
        self.baseline_generator = self._setup_baseline_generator()
        
        # Initialize concept decoder (if available)
        self.concept_decoder = None
        self._setup_concept_decoder()
    
    def _setup_baseline_generator(self):
        """Set up the baseline generator using actual PyTorch model."""
        try:
            # Try to use the robust PyTorch-based baseline generator
            from src.models.robust_concept_decoder import create_robust_concept_decoder
            # Force CPU to avoid MPS issues on macOS
            device = "cpu" if self.device == "auto" or "mps" in str(self.device) else self.device
            baseline_generator = create_robust_concept_decoder(
                model_name=self.model_name,
                device=device
            )
            print("✅ PyTorch baseline generator initialized successfully")
            return baseline_generator
        except Exception as e:
            print(f"⚠️ PyTorch baseline generator initialization failed: {e}")
            print("⚠️ Falling back to simulated baseline...")
            # Fallback to simulated baseline
            return RealisticQwenBaseline(model_name=self.model_name, device=self.device)
    
    def _setup_concept_decoder(self):
        """Set up the concept decoder with fallback to baseline."""
        try:
            # Try to use the robust PyTorch-based concept decoder
            from src.models.robust_concept_decoder import create_robust_concept_decoder
            # Force CPU to avoid MPS issues on macOS
            device = "cpu" if self.device == "auto" or "mps" in str(self.device) else self.device
            self.concept_decoder = create_robust_concept_decoder(
                model_name=self.model_name,
                device=device
            )
            print("✅ PyTorch concept decoder initialized successfully")
        except Exception as e:
            print(f"⚠️ PyTorch concept decoder initialization failed: {e}")
            print("⚠️ Falling back to simple concept decoder...")
            try:
                # Fallback to simple concept decoder
                self.concept_decoder = create_concept_decoder(
                    model_name=self.model_name,
                    device=self.device
                )
                print("✅ Simple concept decoder initialized successfully")
            except Exception as e2:
                print(f"⚠️ Simple concept decoder also failed: {e2}")
                print("⚠️ Will use baseline generation for both models")
                self.concept_decoder = None
    
    def load_sample_data(self, dataset_type="folio", num_examples=10):
        """Load sample data for evaluation."""
        if dataset_type.lower() == "folio":
            return self._load_folio_data(num_examples)
        elif dataset_type.lower() == "truthfulqa":
            return self._load_truthfulqa_data(num_examples)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def _load_folio_data(self, num_examples=10):
        """Load real FOLIO data."""
        try:
            loader = FolioLoader()
            raw_data = loader.load_data(split="validation", num_examples=num_examples)
            
            # Convert FOLIO format to our expected format
            formatted_data = []
            for item in raw_data:
                # Create question from premises and conclusion
                premises = "\n- ".join(item['premises'])
                conclusion = item['conclusion']
                question = f"Given the following premises:\n- {premises}\n\nIs the following conclusion True, False, or Unknown?\nConclusion: {conclusion}"
                
                # Map FOLIO labels to our format
                ground_truth = item['label']  # FOLIO uses 'label' field
                
                formatted_data.append({
                    "question": question,
                    "ground_truth": ground_truth
                })
            
            return formatted_data
            
        except Exception as e:
            print(f"Warning: Could not load FOLIO data: {e}")
            print("Falling back to sample data...")
            # Fallback to sample data
            sample_data = [
                {
                    "question": "Is the following statement true or false: All birds can fly.",
                    "ground_truth": "False"
                },
                {
                    "question": "If it's raining, then the ground is wet. It's raining. Is the ground wet?",
                    "ground_truth": "True"
                }
            ]
            return sample_data[:num_examples]
    
    def _load_truthfulqa_data(self, num_examples=10):
        """Load real TruthfulQA data."""
        try:
            loader = TruthfulQALoader()
            raw_data = loader.load_data(num_examples=num_examples)
            
            # Convert TruthfulQA format to our expected format
            formatted_data = []
            for item in raw_data:
                formatted_data.append({
                    "question": item['Question'],
                    "ground_truth": item['Best Answer']
                })
            
            return formatted_data
            
        except Exception as e:
            print(f"Warning: Could not load TruthfulQA data: {e}")
            print("Falling back to sample data...")
            # Fallback to sample data
            sample_data = [
                {
                    "question": "What is the capital of France?",
                    "ground_truth": "Paris"
                },
                {
                    "question": "Who wrote the novel '1984'?",
                    "ground_truth": "George Orwell"
                }
            ]
            return sample_data[:num_examples]
    
    def generate_baseline_outputs(self, questions: List[str]) -> List[str]:
        """Generate outputs using baseline model."""
        print("Generating baseline outputs...")
        baseline_outputs = []
        
        for i, question in enumerate(questions):
            print(f"Baseline {i+1}/{len(questions)}...", end=" ", flush=True)
            try:
                # Handle both PyTorch and simulated baseline generators
                if hasattr(self.baseline_generator, 'generate_answers'):
                    # PyTorch-based generator
                    result = self.baseline_generator.generate_answers([question])
                    output = result[0]["prediction"]
                else:
                    # Simulated generator
                    output = self.baseline_generator.generate_single(question)
                baseline_outputs.append(output)
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                baseline_outputs.append(f"Error generating baseline output: {e}")
        
        return baseline_outputs
    
    def generate_concept_outputs(self, questions: List[str]) -> List[str]:
        """Generate outputs using concept-aware model."""
        print("Generating concept-aware outputs...")
        concept_outputs = []
        
        if self.concept_decoder is None:
            print("⚠️ Concept decoder not available, using baseline for both")
            return self.generate_baseline_outputs(questions)
        
        for i, question in enumerate(questions):
            print(f"Concept {i+1}/{len(questions)}...", end=" ", flush=True)
            try:
                # Use concept decoder if available
                outputs = self.concept_decoder.generate_answers([question], max_new_tokens=50)
                concept_outputs.append(outputs[0])
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                # Fallback to baseline
                try:
                    # Handle both PyTorch and simulated baseline generators
                    if hasattr(self.baseline_generator, 'generate_answers'):
                        # PyTorch-based generator
                        result = self.baseline_generator.generate_answers([question])
                        output = result[0]["prediction"]
                    else:
                        # Simulated generator
                        output = self.baseline_generator.generate_single(question)
                    concept_outputs.append(output)
                    print("✓ (fallback)")
                except Exception as e2:
                    print(f"✗ Fallback error: {e2}")
                    concept_outputs.append(f"Error generating concept output: {e2}")
        
        return concept_outputs
    
    def run_comparison(self, dataset_type="folio", num_examples=10):
        """Run the complete comparison experiment."""
        print(f"🚀 Running Concept-Aware vs Baseline Comparison")
        print(f"Dataset: {dataset_type.upper()}")
        print(f"Number of examples: {num_examples}")
        print("=" * 60)
        
        # Load data
        print("📊 Loading sample data...")
        data = self.load_sample_data(dataset_type, num_examples)
        questions = [item["question"] for item in data]
        ground_truths = [item["ground_truth"] for item in data]
        
        # Generate outputs
        baseline_outputs = self.generate_baseline_outputs(questions)
        concept_outputs = self.generate_concept_outputs(questions)
        
        # Run pairwise comparisons
        print("\n⚖️ Running pairwise comparisons...")
        results = self.judge.compare_batch(
            questions, ground_truths, baseline_outputs, concept_outputs,
            dataset_type=dataset_type, delay=0.5
        )
        
        # Add generation details to results
        results.update({
            "dataset_type": dataset_type,
            "num_examples": num_examples,
            "model_name": self.model_name,
            "device": self.device,
            "baseline_outputs": baseline_outputs,
            "concept_outputs": concept_outputs,
            "questions": questions,
            "ground_truths": ground_truths
        })
        
        # Display results
        print(f"\n📈 COMPARISON RESULTS:")
        print(f"Total Comparisons: {results['total_comparisons']}")
        print(f"Successful Comparisons: {results['successful_comparisons']}")
        print(f"Failed Comparisons: {results['failed_comparisons']}")
        print(f"\n🏆 WIN RATES:")
        print(f"Baseline Wins: {results['baseline_wins']} ({results['baseline_win_rate']:.1%})")
        print(f"Concept-Aware Wins: {results['concept_wins']} ({results['concept_win_rate']:.1%})")
        print(f"Ties: {results['ties']} ({results['tie_rate']:.1%})")
        
        # Show some individual comparisons
        print(f"\n📝 Sample Comparisons:")
        for i, comp in enumerate(results['individual_comparisons'][:3]):
            if 'error' not in comp:
                print(f"\n  Example {i+1}:")
                print(f"    Question: {comp['question'][:60]}...")
                print(f"    Winner: {comp['winner']} ({comp['confidence']} confidence)")
                print(f"    Reasoning: {comp['reasoning']}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"concept_vs_baseline_comparison_{timestamp}.json"
        filepath = self.judge.save_results(results, filename)
        
        print(f"\n💾 Results saved to: {filepath}")
        print(f"\n✅ Comparison completed successfully!")
        
        return results


def main():
    """Main function to run the comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare Concept-Aware vs Baseline using Pairwise LLM Judge")
    parser.add_argument("--dataset", type=str, default="folio", 
                       choices=["folio", "truthfulqa"],
                       help="Dataset to use for evaluation")
    parser.add_argument("--num_examples", type=int, default=10,
                       help="Number of examples to evaluate")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Model name to use")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run the model on")
    
    args = parser.parse_args()
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        print("\nYou can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize and run comparison
    comparison = ConceptVsBaselineComparison(
        model_name=args.model,
        device=args.device
    )
    
    results = comparison.run_comparison(
        dataset_type=args.dataset,
        num_examples=args.num_examples
    )


if __name__ == "__main__":
    main()
