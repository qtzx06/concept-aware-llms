import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class RobustConceptDecoder:
    """
    A robust, simplified concept decoder that avoids infinite loops and MPS issues.
    """
    
    def __init__(self, model_name, device="cpu", temperature=0.7, top_p=0.95):
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        
        print(f"Loading model: {self.model_name}...")
        print(f"Device: {self.device}, Temperature: {self.temperature}, Top-p: {self.top_p}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 for stability
                device_map="cpu",  # Force CPU to avoid MPS issues
                trust_remote_code=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def generate_answers(self, prompts: List[str], max_new_tokens: int = 50) -> List[Dict[str, Any]]:
        """
        Generate answers for a list of prompts using robust generation.
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}...")
            
            try:
                # Encode the prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                
                # Generate with robust settings
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=min(max_new_tokens, 30),  # Limit tokens
                        temperature=self.temperature,
                        top_p=self.top_p,
                        do_sample=True if self.temperature > 0 else False,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Prevent repetition
                        no_repeat_ngram_size=2,
                        early_stopping=True
                    )
                
                # Decode the generated tokens, excluding the prompt
                prompt_length = input_ids.shape[1]
                generated_ids = generated_ids[0][prompt_length:]
                full_prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                # Safety check: limit response length
                if len(full_prediction) > 300:
                    full_prediction = full_prediction[:300] + "..."
                
                results.append({
                    "prediction": full_prediction.strip(),
                    "concepts": [],  # Simplified - no complex concepts
                    "generation_log": [{"decision": "robust_generation"}]
                })
                
            except Exception as e:
                print(f"Error during generation for prompt '{prompt}': {e}")
                results.append({
                    "prediction": "Error in generation",
                    "concepts": [],
                    "generation_log": [{"error": str(e)}]
                })
        
        return results

def create_robust_concept_decoder(model_name: str, device: str) -> RobustConceptDecoder:
    """Factory function to create a RobustConceptDecoder instance."""
    return RobustConceptDecoder(model_name=model_name, device=device)
