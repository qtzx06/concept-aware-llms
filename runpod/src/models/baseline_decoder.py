import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class BaselineDecoder:
    def __init__(self, model_name, device="auto"):
        self.model_name = model_name
        print(f"Loading model: {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device
            )
            # After loading, get the actual device the model is on
            self.device = self.model.device
            print(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_answers(self, prompts, max_new_tokens=50, temperature=0.7, top_p=0.95):
        answers = []
        for prompt in prompts:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                # Handle deterministic vs. sampling case
                do_sample = temperature > 0.0
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode and clean up the output
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = generated_text[len(prompt):].strip()
                answers.append(answer)
            except Exception as e:
                print(f"ERROR during generation for prompt '{prompt}': {e}")
                answers.append("") # Append empty string on error
        return answers

if __name__ == '__main__':
    # Example usage
    try:
        decoder = BaselineDecoder(model_name="Qwen/Qwen2.5-0.5B-Instruct")
        
        # Example ProtoQA-style prompts
        example_prompts = [
            "Q: What is a common cause of a traffic jam?\nA:",
            "Q: Name something you would find in a kitchen.\nA:"
        ]
        
        print("\nGenerating answers for example prompts...")
        generated_answers = decoder.generate_answers(example_prompts)
        
        for i, (prompt, answer) in enumerate(zip(example_prompts, generated_answers)):
            print("---")
            print(f"Prompt {i+1}: {prompt}")
            print(f"Generated Answer: {answer}")

    except Exception as e:
        print(f"Failed to run baseline decoder example: {e}")
