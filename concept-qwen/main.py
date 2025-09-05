# concept-qwen/main.py
import torch
import os
import pandas as pd
import warnings
import re
import shutil
from pathlib import Path
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure the NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK's 'punkt' tokenizer...")
    nltk.download('punkt')


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
        "repetition_penalty": config.get('repetition_penalty', 1.0)
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
    """Runs a more robust benchmark on the WikiText dataset for factual continuation."""
    print("\n--- Starting Automatic Benchmark (Factual Continuation) ---")

    if benchmark_config.get('logging_enabled', False):
        script_dir = Path(__file__).parent.resolve()
        benchmark_log_dir = script_dir / 'benchmark_logs'
        if benchmark_log_dir.exists():
            print(f"Clearing old logs in {benchmark_log_dir}...")
            shutil.rmtree(benchmark_log_dir)

    print("Loading benchmark dataset (databricks/databricks-dolly-15k)...")
    num_prompts = benchmark_config.get('num_prompts', 10)
    # Using 'train' split as it's the standard for this dataset
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train") 
    
    # --- Pre-process dataset to create prompt/completion pairs ---
    tasks = []
    response_length_filter = benchmark_config.get('response_length_filter', 'none')

    for item in dataset:
        if len(tasks) >= num_prompts:
            break
        
        response = item['response']
        
        # Apply the user's length filter before adding the task
        reference_length = len(tokenizer(response).input_ids)
        if response_length_filter == 'short' and reference_length > 10:
            continue
        if response_length_filter == 'long' and reference_length <= 10:
            continue

        instruction = item['instruction']
        context = item['context']

        # Clean Wikipedia citation numbers from the context, as recommended
        if context:
            context = re.sub(r'\\[\\d+\\]', '', context).strip()
            prompt = f"Instruction: {instruction}\n\nContext: {context}\n\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\n\nResponse:"

        tasks.append({'prompt': prompt, 'reference_story': response})
    
    if not tasks:
        print("\nNo prompts found matching the specified filter. Exiting benchmark.")
        return
    
    print(f"Created {len(tasks)} factual continuation tasks.")
    print("Loading semantic similarity model (all-MiniLM-L6-v2)...")
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

    results = []
    
    # --- Create Decoder Configs from Benchmark Config ---
    strategy = benchmark_config.get('strategy', 'sampling')
    
    baseline_config = {
        'strategy': strategy
    }
    
    concept_config = benchmark_config.copy() # Start with all benchmark settings
    
    if strategy == 'beam':
        baseline_config['num_beams'] = benchmark_config.get('num_beams', 3)
    
    if benchmark_config.get('logging_enabled', False):
        script_dir = Path(__file__).parent.resolve()
        concept_config['log_dir_base'] = script_dir / 'benchmark_logs'
    
    concept_decoder = ConceptDecoder(model, tokenizer, concept_config)

    for i, task in enumerate(tasks):
        prompt = task['prompt']
        reference_story = task['reference_story']
        
        reference_length = len(tokenizer(reference_story).input_ids)
        desired_tokens = int((reference_length + 10 ) * 1.1) # maybe hard code +10 for coherent answer
        max_new_tokens = min(desired_tokens, 250) # hard cap so i dont crash
        
        baseline_config['max_new_tokens'] = max_new_tokens
        concept_decoder.config['max_new_tokens'] = max_new_tokens # Update the decoder's config directly
        
        if benchmark_config.get('logging_enabled', False):
            concept_decoder.set_log_prompt_id(i + 1)

        print(f"\n--- Benchmarking Prompt {i+1}/{len(tasks)} ---")
        print(f"Prompt: {prompt[:150]}...")
        print(f"(Reference length: {reference_length} tokens, Generating max: {max_new_tokens} tokens)")

        baseline_gen = run_baseline(prompt, model, tokenizer, baseline_config)
        concept_gen = concept_decoder.generate(prompt)

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
    results_df.to_csv("benchmark_results_wikitext.csv", index=False)
    print("\n--- Benchmark Complete ---")
    print("Results saved to benchmark_results_wikitext.csv")
    
    print("\n--- Summary Statistics ---")
    print(results_df[['similarity_score_baseline', 'similarity_score_concept']].describe())



def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen3-8B"
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
        benchmark_config['response_length_filter'] = get_user_choice(
            "Filter prompts by response length?",
            ['none', 'short', 'long'],
            'none'
        )
        
        print("\n--- Benchmark Decoding Settings ---")
        # Use the same strategy for both for a fair comparison
        strategy = get_user_choice("Select decoding strategy for benchmark", ['sampling', 'beam'], 'sampling')
        benchmark_config['strategy'] = strategy
        
        if strategy == 'beam':
            benchmark_config['num_beams'] = get_user_input("Enter num beams", 2, int)
        elif strategy == 'sampling':
            benchmark_config['top_p'] = get_user_input("Enter top-p for nucleus sampling", 0.92, float)
        
        print("\n--- Concept Decoder Settings ---")
        benchmark_config['top_k'] = get_user_input("Enter top-k for candidate selection", 100, int)
        benchmark_config['distance_threshold'] = get_user_input("Enter distance threshold for clustering", 0.45, float)

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
        baseline_config['num_beams'] = get_user_input("Enter baseline num beams", 2, int)

    concept_config['logging_enabled'] = get_user_input("Enable detailed concept decoder logging?", 'y') == 'y'
    
    concept_config['strategy'] = get_user_choice(
        "Select concept decoding strategy",
        ['sampling', 'beam'],
        'sampling'
    )

    if concept_config['strategy'] == 'sampling':
        concept_config['top_p'] = get_user_input("Enter top-p for nucleus sampling", 0.92, float)
    elif concept_config['strategy'] == 'beam':
        concept_config['num_beams'] = get_user_input("Enter concept num beams", 2, int)

    concept_config['top_k'] = get_user_input("Enter top-k for candidate selection", 100, int)
    concept_config['distance_threshold'] = get_user_input("Enter distance threshold for clustering", 0.45, float)
    
    ################################
    ####### MANUAL PROMPTING #######
    ################################
    
    prompt = "It was stormy and the water was cold cocaine drugs bad stuff"
    
    run_baseline(prompt, model, tokenizer, baseline_config)
    
    concept_decoder = ConceptDecoder(model, tokenizer, concept_config)
    concept_decoder.generate(prompt)

if __name__ == "__main__":
    main()