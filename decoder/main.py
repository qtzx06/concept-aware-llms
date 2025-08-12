import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concept_decoder import generate_with_concept_decoder, setup_logger
import argparse
import sys
import os

def get_user_config(parser):
    """Gets configuration from the user interactively."""
    args, _ = parser.parse_known_args()
    
    print("\n--- Configure Generation Settings (press Enter to accept default) ---")

    think_input = input(f"Enable thinking mode? (yes/no) [default: {'yes' if args.enable_thinking else 'no'}]: ").lower()
    if think_input:
        args.enable_thinking = think_input not in ["no", "n"]

    tokens_input = input(f"Max new tokens [default: {args.max_new_tokens}]: ")
    if tokens_input:
        args.max_new_tokens = int(tokens_input)

    print("\n--- Concept Decoder Settings ---")

    k_input = input(f"Top-k candidates [default: {args.k}]: ")
    if k_input:
        args.k = int(k_input)

    prob_input = input(f"Minimum token probability (0 to disable) [default: {args.min_token_prob}]: ")
    if prob_input:
        args.min_token_prob = float(prob_input)

    cluster_input = input(f"Clustering algorithm (agglomerative/dbscan) [default: {args.clustering_algo}]: ").lower()
    if cluster_input:
        args.clustering_algo = cluster_input
    
    if args.clustering_algo == 'agglomerative':
        dist_input = input(f"Agglomerative distance threshold [default: {args.distance_threshold}]: ")
        if dist_input:
            args.distance_threshold = float(dist_input)
    elif args.clustering_algo == 'dbscan':
        eps_input = input(f"DBSCAN eps (similarity radius) [default: {args.eps}]: ")
        if eps_input:
            args.eps = float(eps_input)

    dim_red_input = input(f"Use PCA for dimensionality reduction? (yes/no) [default: {'yes' if args.use_dim_reduction else 'no'}]: ").lower()
    if dim_red_input:
        args.use_dim_reduction = dim_red_input in ["yes", "y"]
    
    if args.use_dim_reduction:
        comp_input = input(f"PCA components [default: {args.dim_reduction_components}]: ")
        if comp_input:
            args.dim_reduction_components = int(comp_input)

    ranking_input = input(f"Concept ranking method (sum/max) [default: {args.concept_ranking_method}]: ").lower()
    if ranking_input:
        args.concept_ranking_method = ranking_input

    print("\n" + "-"*64 + "\n")
    return args

def main():
    parser = argparse.ArgumentParser(description="Concept-Aware Decoding for Large Language Models")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="The forgotten library was filled with ancient books, their pages smelling of dust and...")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--enable_thinking", type=bool, default=True)
    parser.add_argument('--non_interactive', action='store_true')
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--min_token_prob", type=float, default=0.001)
    parser.add_argument("--clustering_algo", type=str, default="agglomerative", choices=['agglomerative', 'dbscan'])
    parser.add_argument("--distance_threshold", type=float, default=0.4)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--use_dim_reduction", type=bool, default=False)
    parser.add_argument("--dim_reduction_components", type=int, default=32)
    parser.add_argument("--concept_ranking_method", type=str, default="sum", choices=['sum', 'max'])

    is_interactive = '--non_interactive' not in sys.argv
    
    if is_interactive:
        args = get_user_config(parser)
    else:
        args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model and tokenizer: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print("Model loaded.")
    
    print(f"\nPrompt: \"{args.prompt}\"\n")
    print("---")

    print("[1] BASELINE DECODER (Standard Sampling)")
    start_time = time.time()
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens, temperature=0.7, top_p=0.8)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    
    baseline_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    end_time = time.time()
    print(f"Output: {baseline_output}")
    print(f"\nGeneration Time: {end_time - start_time:.2f} seconds")
    print("---")

    print("[2] CONCEPT-DECODER")
    logger = setup_logger()
    start_time = time.time()
    
    generate_with_concept_decoder(
        model, tokenizer, args.prompt, args.max_new_tokens, args.k, 
        args.distance_threshold, args.enable_thinking, args.clustering_algo, 
        args.eps, args.use_dim_reduction, args.dim_reduction_components,
        args.concept_ranking_method, args.min_token_prob, logger
    )
    
    end_time = time.time()
    print(f"\n\nGeneration Time: {end_time - start_time:.2f} seconds")

    print("\n--- Last 10 lines of concept_decoder.log ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, 'concept_decoder.log')
    os.system(f"tail -n 10 {log_file_path}")


if __name__ == "__main__":
    main()
