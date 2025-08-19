# concept-qwen/main.py
import torch
import os
import pandas as pd
import warnings
import re
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress the specific hipBLASLt warning
warnings.filterwarnings("ignore", message="Attempting to use hipBLASLt on an unsupported architecture!")

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

def run_benchmark(model, tokenizer, device, benchmark_config):
    """Runs a benchmark on the WritingPrompts dataset."""
    print("\n--- Starting Automatic Benchmark ---")

    # Clear previous benchmark logs if logging is enabled for this run
    if benchmark_config.get('logging_enabled', False):
        script_dir = Path(__file__).parent.resolve()
        benchmark_log_dir = script_dir / 'benchmark_logs'
        if benchmark_log_dir.exists():
            print(f"Clearing old logs in {benchmark_log_dir}...")
            shutil.rmtree(benchmark_log_dir)

    print("Loading benchmark dataset (krisha05/story-generation-dataset)...")
    num_prompts = benchmark_config.get('num_prompts', 10)
    # Use the 'train' split as this dataset doesn't have a 'test' split.
    dataset = load_dataset("krisha05/story-generation-dataset", split="train").shuffle(seed=42).select(range(num_prompts))
    
    print("Loading semantic similarity model (all-MiniLM-L6-v2)...")

    similarity_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    results = []
    # Use default 'sampling' for both decoders for a fair comparison
    baseline_config = {'strategy': 'sampling', 'max_new_tokens': 150}
    
    # --- Prepare Concept Decoder with Benchmark Config ---
    concept_config = {'strategy': 'sampling', 'max_new_tokens': 150}
    concept_config.update(benchmark_config) # Add logging settings
    if benchmark_config.get('logging_enabled', False):
        # FIX: Create an absolute path for the benchmark logs directory
        script_dir = Path(__file__).parent.resolve()
        concept_config['log_dir_base'] = script_dir / 'benchmark_logs'
    
    concept_decoder = ConceptDecoder(model, tokenizer, concept_config)

    for i, item in enumerate(dataset):
        prompt = item['instruction']
        reference_story = item['output']
        
        if benchmark_config.get('logging_enabled', False):
            concept_decoder.set_log_prompt_id(i + 1) # Tell decoder which prompt it's on

        print(f"\n--- Benchmarking Prompt {i+1}/{len(dataset)} ---")
        print(f"Prompt: {prompt[:100]}...")

        # Generate with both methods
        baseline_gen = run_baseline(prompt, model, tokenizer, baseline_config)
        concept_gen = concept_decoder.generate(prompt)

        # Calculate semantic similarity
        ref_embedding = similarity_model.encode(reference_story, convert_to_tensor=True)
        base_embedding = similarity_model.encode(baseline_gen, convert_to_tensor=True)
        concept_embedding = similarity_model.encode(concept_gen, convert_to_tensor=True)

        score_base = util.pytorch_cos_sim(ref_embedding, base_embedding).item()
        score_concept = util.pytorch_cos_sim(ref_embedding, concept_embedding).item()
        
        results.append({
            "prompt": prompt,
            "reference_story": reference_story,
            "baseline_generation": baseline_gen,
            "concept_generation": concept_gen,
            "similarity_score_baseline": score_base,
            "similarity_score_concept": score_concept,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("benchmark_results.csv", index=False)
    print("\n--- Benchmark Complete ---")
    print("Results saved to benchmark_results.csv")
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(results_df[['similarity_score_baseline', 'similarity_score_concept']].describe())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-0.6B"
    
    print(f"Loading model: {model_name} on device: {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded.")

    print("\n--- Configuration ---")
    if get_user_input("Run automatic benchmark?", 'n') == 'y':
        benchmark_config = {}
        benchmark_config['logging_enabled'] = get_user_input("Enable benchmark logging?", 'n') == 'y'
        if benchmark_config['logging_enabled']:
            benchmark_config['log_mode'] = get_user_choice(
                "Benchmark log mode",
                ['logs', 'visuals', 'both'],
                'both'
            )
        benchmark_config['num_prompts'] = get_user_input("Enter number of prompts to run", 10, int)
        run_benchmark(model, tokenizer, device, benchmark_config)
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
    
    ################################
    ####### MANUAL PROMPTING #######
    ################################
    
    prompt = "The moon is actually a giant egg , and it has just started to hatch ."
    
    run_baseline(prompt, model, tokenizer, baseline_config)
    
    concept_decoder = ConceptDecoder(model, tokenizer, concept_config)
    concept_decoder.generate(prompt)

if __name__ == "__main__":
    main()