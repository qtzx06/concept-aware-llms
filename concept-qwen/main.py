# concept-qwen/main.py
import torch
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from concept_decoder import ConceptDecoder

def get_user_input(prompt_text, default='n', type_func=str):
    prompt = f"{prompt_text} (default: {default}): "
    response = input(prompt).lower().strip()
    if not response:
        return default
    return type_func(response)

def get_user_choice(prompt_text, choices, default_choice):
    choices_str = "/".join(choices)
    prompt = f"{prompt_text} ({choices_str}) (default: {default_choice}): "
    response = input(prompt).lower().strip()
    if not response:
        return default_choice
    while response not in choices:
        print(f"Invalid choice. Please select from: {choices_str}")
        response = input(prompt).lower().strip()
    return response

def run_baseline(prompt, model, tokenizer, config):
    strategy = config.get('strategy', 'sampling')
    print(f"\n--- BASELINE DECODER ({strategy.capitalize()}) ---")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    gen_params = {
        "max_new_tokens": config.get('max_new_tokens', 100),
    }

    if strategy == 'sampling':
        gen_params.update({
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.92,
        })
    elif strategy == 'beam':
        gen_params.update({
            "num_beams": config.get('num_beams', 3),
            "early_stopping": True,
        })
    # Greedy is default if do_sample=False and num_beams=1

    generated_ids = model.generate(**inputs, **gen_params)
    output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(output)
    return output[len(prompt):].strip()

def run_benchmark(model, tokenizer, device):
    # (Benchmark function remains the same as before)
    pass

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2-0.5B"
    
    print(f"Loading model: {model_name} on device: {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded.")

    print("\n--- Configuration ---")
    if get_user_input("Run automatic benchmark?", 'n') == 'y':
        run_benchmark(model, tokenizer, device)
        return

    # --- Interactive Mode ---
    baseline_config = {}
    concept_config = {}

    baseline_config['strategy'] = get_user_choice(
        "Select baseline decoding strategy",
        ['sampling', 'greedy', 'beam'],
        'sampling'
    )
    if baseline_config['strategy'] == 'beam':
        baseline_config['num_beams'] = get_user_input("Enter baseline num beams", 3, int)

    concept_config['logging_enabled'] = get_user_input("Enable detailed concept decoder logging?", 'y') == 'y'
    
    concept_config['strategy'] = get_user_choice(
        "Select concept decoding strategy",
        ['sampling', 'beam'],
        'sampling'
    )

    if concept_config['strategy'] == 'sampling':
        concept_config['top_p'] = get_user_input("Enter top-p for nucleus sampling", 0.92, float)
    elif concept_config['strategy'] == 'beam':
        concept_config['num_beams'] = get_user_input("Enter concept num beams", 3, int)

    concept_config['top_k'] = get_user_input("Enter top-k for candidate selection", 100, int)
    concept_config['distance_threshold'] = get_user_input("Enter distance threshold for clustering", 0.45, float)
    
    prompt = "The ancient library was filled with books, their pages smelling of dust and"
    
    run_baseline(prompt, model, tokenizer, baseline_config)
    
    concept_decoder = ConceptDecoder(model, tokenizer, concept_config)
    concept_decoder.generate(prompt)

if __name__ == "__main__":
    main()
