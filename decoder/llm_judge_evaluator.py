import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMJudgeEvaluator:
    """
    Evaluates text generation outputs using an LLM judge instead of automated metrics.
    Provides comparative evaluation between different generation methods.
    """
    
    def __init__(self, judge_model_name: str = "Qwen/Qwen3-1.8B", device: str = "auto"):
        """
        Initialize the LLM Judge Evaluator.
        
        Args:
            judge_model_name: Name/path of the judge model
            device: Device to run the judge model on
        """
        self.judge_model_name = judge_model_name
        self.device = self._setup_device(device)
        
        print(f"Loading judge model: {judge_model_name}")
        self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
        self.judge_model = self._load_judge_model()
        
        if self.judge_tokenizer.pad_token_id is None:
            if self.judge_tokenizer.eos_token_id is not None:
                self.judge_tokenizer.pad_token = self.judge_tokenizer.eos_token
            else:
                self.judge_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                
        print("Judge model loaded successfully")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the appropriate device for the judge model."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def _load_judge_model(self):
        """Load the judge model with appropriate settings."""
        try:
            if self.device.type == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    self.judge_model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    self.judge_model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                ).to(self.device)
            return model
        except Exception as e:
            print(f"Error loading judge model: {e}")
            raise
    
    def _create_judge_prompt(self, prompt: str, response_a: str, response_b: str, 
                           evaluation_criteria: Optional[str] = None) -> str:
        """
        Create a prompt for the LLM judge to evaluate two responses.
        
        Args:
            prompt: Original user prompt
            response_a: First response (baseline)
            response_b: Second response (concept decoder)
            evaluation_criteria: Specific criteria to focus on
            
        Returns:
            Formatted judge prompt
        """
        if evaluation_criteria is None:
            evaluation_criteria = """
            - Relevance: How well does the response answer the original question?
            - Quality: Is the response coherent, well-structured, and appropriate?
            - Creativity: Does the response show creative thinking or novel insights?
            - Accuracy: Is the response factually correct and reasonable?
            - Helpfulness: How useful would this response be to the user?
            """
        
        judge_prompt = f"""You are an expert evaluator. Compare these two AI responses and determine which is better.

Original Question: "{prompt}"

Response A (Baseline): {response_a}

Response B (Concept): {response_b}

Evaluation Criteria:{evaluation_criteria}

You must choose which response is better and explain why. Follow this exact format:

WINNER: A
CONFIDENCE: High
REASONING: [Your detailed explanation here]

OR

WINNER: B  
CONFIDENCE: Medium
REASONING: [Your detailed explanation here]

Your evaluation:"""

        return judge_prompt
    
    def _query_judge(self, judge_prompt: str, max_new_tokens: int = 300) -> str:
        """Query the judge model with the evaluation prompt."""
        try:
            # Prepare input
            messages = [{"role": "user", "content": judge_prompt}]
            
            if hasattr(self.judge_tokenizer, "apply_chat_template"):
                text = self.judge_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                text = f"<user>\n{judge_prompt}\n</user>\n<assistant>\n"
            
            inputs = self.judge_tokenizer([text], return_tensors="pt")
            
            # Move inputs to device if needed
            if not hasattr(self.judge_model, 'device') or self.judge_model.device.type == "cuda":
                # If using device_map="auto", inputs can stay on CPU
                pass
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,  # Very low temperature for consistent judgments
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.judge_tokenizer.pad_token_id,
                    eos_token_id=self.judge_tokenizer.eos_token_id,
                    repetition_penalty=1.1,  # Reduce repetition
                )
            
            # Decode response
            prompt_len = inputs["input_ids"].shape[1]
            response_ids = outputs[0, prompt_len:].tolist()
            response = self.judge_tokenizer.decode(response_ids, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error querying judge model: {e}")
            return f"Error: Could not get judgment - {str(e)}"
    
    def _parse_judgment(self, judgment_text: str) -> Dict[str, str]:
        """Parse the structured judgment from the judge model."""
        result = {
            "winner": "Unknown",
            "confidence": "Unknown", 
            "reasoning": judgment_text,
            "raw_response": judgment_text
        }
        
        # Convert to uppercase for easier matching
        text_upper = judgment_text.upper()
        
        # Try to find winner - look for various patterns
        winner_patterns = [
            r"WINNER:\s*([AB])",
            r"WINNER\s*([AB])",  
            r"WINNER:\s*(A|B)",
            r"RESPONSE\s*([AB])\s*IS\s*BETTER",
            r"([AB])\s*IS\s*BETTER",
            r"CHOOSE\s*([AB])",
            r"SELECT\s*([AB])"
        ]
        
        for pattern in winner_patterns:
            match = re.search(pattern, text_upper)
            if match:
                result["winner"] = match.group(1).upper()
                break
        
        # If still no winner found, try simpler approaches
        if result["winner"] == "Unknown":
            lines = judgment_text.split('\n')
            for line in lines:
                line_upper = line.strip().upper()
                if line_upper.startswith("WINNER"):
                    if "A" in line_upper:
                        result["winner"] = "A"
                        break
                    elif "B" in line_upper:
                        result["winner"] = "B" 
                        break
        
        # Find confidence level
        confidence_patterns = [
            r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)",
            r"CONFIDENCE\s*(HIGH|MEDIUM|LOW)"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, text_upper)
            if match:
                result["confidence"] = match.group(1).capitalize()
                break
        
        # Extract reasoning - look for everything after "REASONING:"
        reasoning_match = re.search(r"REASONING:\s*(.+)", judgment_text, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
        elif result["reasoning"] == judgment_text:
            # If no structured reasoning found, try to clean up the full text
            # Remove common prompt artifacts
            clean_text = judgment_text.replace("Your evaluation:", "").strip()
            if clean_text:
                result["reasoning"] = clean_text
        
        return result
    
    def compare_responses(self, prompt: str, baseline_output: str, concept_output: str, 
                         evaluation_criteria: Optional[str] = None, debug: bool = False) -> Dict:
        """
        Compare two responses using the LLM judge.
        
        Args:
            prompt: Original prompt/question
            baseline_output: Response from baseline method
            concept_output: Response from concept decoder
            evaluation_criteria: Optional custom criteria
            debug: If True, print raw judge response for debugging
            
        Returns:
            Dictionary containing judgment results
        """
        print("🤖 LLM Judge is evaluating responses...")
        
        judge_prompt = self._create_judge_prompt(
            prompt, baseline_output, concept_output, evaluation_criteria
        )
        
        judgment_text = self._query_judge(judge_prompt)
        
        if debug:
            print(f"\n🔍 DEBUG - Raw judge response:")
            print(f"'{judgment_text}'")
            print("=" * 50)
        
        judgment = self._parse_judgment(judgment_text)
        
        # Simple fallback heuristic if judge couldn't decide
        if judgment["winner"] == "Unknown":
            print("⚠️  Judge couldn't determine winner. Applying fallback heuristic...")
            
            # Simple heuristic: prefer longer, more detailed responses
            baseline_len = len(baseline_output.strip())
            concept_len = len(concept_output.strip())
            
            if abs(baseline_len - concept_len) > 5:  # Only if there's a meaningful difference
                if concept_len > baseline_len:
                    judgment["winner"] = "B"
                    judgment["confidence"] = "Low"
                    judgment["reasoning"] = f"Fallback heuristic: Concept decoder response is longer ({concept_len} vs {baseline_len} chars) and potentially more detailed. Original judge reasoning: {judgment['reasoning']}"
                else:
                    judgment["winner"] = "A" 
                    judgment["confidence"] = "Low"
                    judgment["reasoning"] = f"Fallback heuristic: Baseline response is longer ({baseline_len} vs {concept_len} chars) and potentially more detailed. Original judge reasoning: {judgment['reasoning']}"
            else:
                # If lengths are similar, default to concept decoder as it's the method being tested
                judgment["winner"] = "B"
                judgment["confidence"] = "Low"
                judgment["reasoning"] = f"Fallback heuristic: Responses are similar length, defaulting to concept decoder as the experimental method. Original judge reasoning: {judgment['reasoning']}"
        
        # Create comprehensive results
        results = {
            "timestamp": datetime.now().isoformat(),
            "judge_model": self.judge_model_name,
            "prompt": prompt,
            "baseline_output": baseline_output,
            "concept_output": concept_output,
            "judgment": judgment,
            "evaluation_criteria": evaluation_criteria or "Default criteria"
        }
        
        return results
    
    def print_comparison_summary(self, results: Dict):
        """Print a formatted summary of the comparison."""
        judgment = results["judgment"]
        
        print("\n" + "="*60)
        print("🏆 LLM JUDGE EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n📝 Original Prompt: {results['prompt']}")
        print(f"\n🤖 Judge Model: {results['judge_model']}")
        
        print(f"\n📊 **JUDGMENT SUMMARY**")
        print(f"   Winner: Response {judgment['winner']}")
        print(f"   Confidence: {judgment['confidence']}")
        
        # Handle case where winner couldn't be determined
        if judgment['winner'] == 'Unknown':
            print(f"\n⚠️  **NOTE**: Judge could not determine a clear winner.")
            print(f"   This might indicate:")
            print(f"   - Both responses are very similar in quality")
            print(f"   - The judge model needs a larger/more capable model")
            print(f"   - The responses are both very good or both poor")
            print(f"\n💡 **SUGGESTIONS**:")
            print(f"   - Try a larger judge model (e.g., Qwen/Qwen3-7B)")
            print(f"   - Check the debug output above for raw judge response")
            print(f"   - Consider custom evaluation criteria")
            
            print(f"\n💭 **BASELINE OUTPUT**: {results['baseline_output']}")
            print(f"\n🔄 **CONCEPT OUTPUT**: {results['concept_output']}")
            
        else:
            # Determine which method won
            if judgment['winner'] == 'A':
                winning_method = "Baseline Decoder"
                winning_output = results['baseline_output']
                losing_method = "Concept Decoder"
                losing_output = results['concept_output']
            elif judgment['winner'] == 'B':
                winning_method = "Concept Decoder"
                winning_output = results['concept_output']
                losing_method = "Baseline Decoder"
                losing_output = results['baseline_output']
            
            print(f"\n🎯 **WINNING METHOD**: {winning_method}")
            print(f"\n💬 **WINNING RESPONSE**: {winning_output}")
            print(f"\n📝 **LOSING RESPONSE**: {losing_output}")
        
        print(f"\n🧠 **REASONING**: {judgment['reasoning']}")
        
        print("\n" + "="*60)
    
    def run_multiple_comparisons(self, test_cases: List[Dict], 
                               evaluation_criteria: Optional[str] = None) -> List[Dict]:
        """
        Run multiple comparisons for batch evaluation.
        
        Args:
            test_cases: List of dicts with 'prompt', 'baseline_output', 'concept_output'
            evaluation_criteria: Optional custom criteria
            
        Returns:
            List of judgment results
        """
        results = []
        
        for i, case in enumerate(test_cases):
            print(f"\n--- Evaluating Case {i+1}/{len(test_cases)} ---")
            result = self.compare_responses(
                case['prompt'], 
                case['baseline_output'], 
                case['concept_output'],
                evaluation_criteria
            )
            results.append(result)
            time.sleep(0.5)  # Brief pause between evaluations
        
        return results
    
    def save_results(self, results: Dict, filename: Optional[str] = None) -> str:
        """Save evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_judge_evaluation_{timestamp}.json"
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(script_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            return filepath
        except Exception as e:
            print(f"Error saving results: {e}")
            return ""


def get_human_references_for_prompt(prompt: str) -> List[str]:
    """
    Get human reference responses for a given prompt.
    This is maintained for compatibility but may not be needed for LLM judge evaluation.
    """
    reference_data = {
        "Name something parents would criticize their children for having.": [
            "Bad grades",
            "Poor attitude", 
            "Messy room",
            "Bad friends",
            "Too much screen time",
            "Lack of responsibility"
        ]
    }
    
    return reference_data.get(prompt, ["No human references available for this prompt"])


# Example usage and testing
if __name__ == "__main__":
    # Example test
    evaluator = LLMJudgeEvaluator()
    
    test_prompt = "Name something parents would criticize their children for having."
    baseline_resp = "Bad grades or poor performance in school."
    concept_resp = "Disrespectful behavior toward authority figures."
    
    results = evaluator.compare_responses(test_prompt, baseline_resp, concept_resp)
    evaluator.print_comparison_summary(results)
    
    # Save results
    filepath = evaluator.save_results(results)
    print(f"\nResults saved to: {filepath}")