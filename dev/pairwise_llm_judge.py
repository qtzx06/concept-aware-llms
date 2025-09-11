#!/usr/bin/env python3
"""
Pairwise LLM Judge - LLMArena-style comparison
This script compares baseline vs concept-aware outputs using Gemini as a judge,
providing pairwise comparisons instead of quantitative scores.
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai


class PairwiseLLMJudge:
    """
    LLMArena-style pairwise comparison judge using OpenAI's API.
    Compares two model outputs and determines which is better.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the pairwise LLM judge.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable OPENAI_API_KEY
            model_name: Name of the OpenAI model to use for evaluation
        """
        self.model_name = model_name
        
        # Set up API key
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError(
                    "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=api_key)
    
    def _create_comparison_prompt(self, question: str, ground_truth: str, 
                                baseline_output: str, concept_output: str,
                                dataset_type: str = "general") -> str:
        """Create pairwise comparison prompt."""
        
        if dataset_type.lower() == "folio":
            return f"""You are an expert evaluator comparing two model responses to a logical reasoning question. Please compare the following:

**Question:** {question}

**Ground Truth:** {ground_truth}

**Model A (Baseline) Response:** {baseline_output}

**Model B (Concept-Aware) Response:** {concept_output}

**Task:** Determine which response is better for this logical reasoning question.

**Evaluation Criteria:**
1. **Correctness**: Which response is more accurate?
2. **Logical Reasoning**: Which response demonstrates better logical reasoning?
3. **Clarity**: Which response is clearer and more understandable?
4. **Completeness**: Which response provides more complete information?

**Instructions:**
- Respond with ONLY a JSON object in this exact format:
{{"winner": "<A/B/TIE>", "reasoning": "<brief explanation of why this response is better>", "confidence": "<high/medium/low>"}}

- Choose A if the baseline response is better
- Choose B if the concept-aware response is better  
- Choose TIE if they are equally good
- Be fair and focus on the quality of reasoning and accuracy"""

        elif dataset_type.lower() == "truthfulqa":
            return f"""You are an expert evaluator comparing two model responses to a factual question. Please compare the following:

**Question:** {question}

**Reference Answer:** {ground_truth}

**Model A (Baseline) Response:** {baseline_output}

**Model B (Concept-Aware) Response:** {concept_output}

**Task:** Determine which response is better for this factual question.

**Evaluation Criteria:**
1. **Factual Accuracy**: Which response is more factually correct?
2. **Truthfulness**: Which response avoids misconceptions better?
3. **Helpfulness**: Which response is more useful and informative?
4. **Clarity**: Which response is clearer and better structured?

**Instructions:**
- Respond with ONLY a JSON object in this exact format:
{{"winner": "<A/B/TIE>", "reasoning": "<brief explanation of why this response is better>", "confidence": "<high/medium/low>"}}

- Choose A if the baseline response is better
- Choose B if the concept-aware response is better
- Choose TIE if they are equally good
- Focus on factual accuracy and truthfulness"""

        else:  # General evaluation
            return f"""You are an expert evaluator comparing two model responses. Please compare the following:

**Question:** {question}

**Reference Answer:** {ground_truth}

**Model A (Baseline) Response:** {baseline_output}

**Model B (Concept-Aware) Response:** {concept_output}

**Task:** Determine which response is better overall.

**Evaluation Criteria:**
1. **Accuracy**: Which response is more accurate?
2. **Relevance**: Which response better addresses the question?
3. **Completeness**: Which response provides more complete information?
4. **Clarity**: Which response is clearer and better structured?

**Instructions:**
- Respond with ONLY a JSON object in this exact format:
{{"winner": "<A/B/TIE>", "reasoning": "<brief explanation of why this response is better>", "confidence": "<high/medium/low>"}}

- Choose A if the baseline response is better
- Choose B if the concept-aware response is better
- Choose TIE if they are equally good
- Be fair and consistent in your evaluation"""

    def _parse_comparison_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini's comparison response."""
        try:
            # Try to extract JSON from the response
            response_text = response_text.strip()
            
            # Find JSON object in the response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                result = json.loads(json_str)
                
                # Validate required fields
                if 'winner' not in result:
                    raise ValueError("Missing 'winner' field in response")
                
                # Ensure winner is valid
                if result['winner'] not in ['A', 'B', 'TIE']:
                    raise ValueError(f"Invalid winner: {result['winner']}")
                
                return result
            else:
                raise ValueError("No valid JSON found in response")
                
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse comparison response: {e}")
            print(f"Raw response: {response_text}")
            
            # Return default comparison
            return {
                "winner": "TIE",
                "reasoning": "Failed to parse comparison response",
                "confidence": "low",
                "error": str(e)
            }
    
    def compare_outputs(self, question: str, ground_truth: str, 
                       baseline_output: str, concept_output: str,
                       dataset_type: str = "general") -> Dict[str, Any]:
        """Compare two model outputs using OpenAI judge."""
        try:
            # Create comparison prompt
            prompt = self._create_comparison_prompt(
                question, ground_truth, baseline_output, concept_output, dataset_type
            )
            
            # Generate comparison using OpenAI
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator comparing model responses. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=512,
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            comparison = self._parse_comparison_response(response_text)
            
            # Add metadata
            comparison.update({
                "question": question,
                "ground_truth": ground_truth,
                "baseline_output": baseline_output,
                "concept_output": concept_output,
                "dataset_type": dataset_type
            })
            
            return comparison
            
        except Exception as e:
            print(f"Error comparing outputs: {e}")
            return {
                "winner": "TIE",
                "reasoning": f"Comparison failed: {str(e)}",
                "confidence": "low",
                "error": str(e),
                "question": question,
                "ground_truth": ground_truth,
                "baseline_output": baseline_output,
                "concept_output": concept_output,
                "dataset_type": dataset_type
            }
    
    def compare_batch(self, questions: List[str], ground_truths: List[str],
                     baseline_outputs: List[str], concept_outputs: List[str],
                     dataset_type: str = "general", delay: float = 0.5) -> Dict[str, Any]:
        """
        Compare batches of model outputs.
        
        Args:
            questions: List of questions
            ground_truths: List of ground truth answers
            baseline_outputs: List of baseline model outputs
            concept_outputs: List of concept-aware model outputs
            dataset_type: Type of dataset ("folio", "truthfulqa", or "general")
            delay: Delay between API calls in seconds
            
        Returns:
            Dictionary containing comparison results
        """
        if not all(len(lst) == len(questions) for lst in [ground_truths, baseline_outputs, concept_outputs]):
            raise ValueError("All input lists must have the same length")
        
        print(f"Comparing {len(questions)} pairs of outputs using Gemini judge...")
        
        comparisons = []
        baseline_wins = 0
        concept_wins = 0
        ties = 0
        successful_comparisons = 0
        
        for i, (question, ground_truth, baseline_output, concept_output) in enumerate(
            zip(questions, ground_truths, baseline_outputs, concept_outputs)
        ):
            print(f"Comparing {i+1}/{len(questions)}...", end=" ", flush=True)
            
            # Compare outputs
            comparison = self.compare_outputs(
                question, ground_truth, baseline_output, concept_output, dataset_type
            )
            comparisons.append(comparison)
            
            # Count results
            if 'error' not in comparison:
                successful_comparisons += 1
                winner = comparison['winner']
                if winner == 'A':
                    baseline_wins += 1
                    print(f"Baseline wins")
                elif winner == 'B':
                    concept_wins += 1
                    print(f"Concept-aware wins")
                else:  # TIE
                    ties += 1
                    print(f"Tie")
            else:
                print(f"Error: {comparison.get('error', 'Unknown error')}")
            
            # Add delay to avoid rate limiting
            if i < len(questions) - 1:  # Don't delay after the last item
                time.sleep(delay)
        
        # Calculate win rates
        total_comparisons = baseline_wins + concept_wins + ties
        baseline_win_rate = baseline_wins / total_comparisons if total_comparisons > 0 else 0
        concept_win_rate = concept_wins / total_comparisons if total_comparisons > 0 else 0
        tie_rate = ties / total_comparisons if total_comparisons > 0 else 0
        
        results = {
            "total_comparisons": len(questions),
            "successful_comparisons": successful_comparisons,
            "failed_comparisons": len(questions) - successful_comparisons,
            "baseline_wins": baseline_wins,
            "concept_wins": concept_wins,
            "ties": ties,
            "baseline_win_rate": baseline_win_rate,
            "concept_win_rate": concept_win_rate,
            "tie_rate": tie_rate,
            "individual_comparisons": comparisons
        }
        
        print(f"\nComparison complete!")
        print(f"Baseline wins: {baseline_wins} ({baseline_win_rate:.1%})")
        print(f"Concept-aware wins: {concept_wins} ({concept_win_rate:.1%})")
        print(f"Ties: {ties} ({tie_rate:.1%})")
        
        return results
    
    def save_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Save comparison results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pairwise_comparison_{timestamp}.json"
        
        # Ensure results directory exists
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
        return filepath


def main():
    """Example usage of the pairwise LLM judge."""
    print("Pairwise LLM Judge - LLMArena Style (OpenAI)")
    print("=" * 45)
    
    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your_api_key_here'")
        print("\nYou can get an API key from: https://platform.openai.com/api-keys")
        return
    
    # Initialize judge
    judge = PairwiseLLMJudge()
    
    # Example data
    questions = [
        "Is the following statement true or false: All birds can fly.",
        "What is the capital of France?"
    ]
    
    ground_truths = [
        "False",
        "Paris"
    ]
    
    baseline_outputs = [
        "False, not all birds can fly.",
        "The capital of France is Paris."
    ]
    
    concept_outputs = [
        "False, not all birds can fly. For example, penguins and ostriches cannot fly.",
        "The capital of France is Paris, which is located in the north-central part of the country."
    ]
    
    # Compare
    print("Running pairwise comparisons...")
    results = judge.compare_batch(
        questions, ground_truths, baseline_outputs, concept_outputs, 
        dataset_type="general"
    )
    
    # Display results
    print(f"\nResults:")
    print(f"Baseline Win Rate: {results['baseline_win_rate']:.1%}")
    print(f"Concept-Aware Win Rate: {results['concept_win_rate']:.1%}")
    print(f"Tie Rate: {results['tie_rate']:.1%}")
    
    # Save results
    judge.save_results(results, "example_pairwise_comparison.json")
    
    print("\nComparison complete!")


if __name__ == "__main__":
    main()
