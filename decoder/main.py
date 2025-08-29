#!/usr/bin/env python3
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from concept_decoder import generate_with_concept_decoder, setup_logger
from llm_judge_evaluator import LLMJudgeEvaluator
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

    print("\n--- LLM Judge Evaluation Settings ---")

    judge_input = input(f"Enable LLM judge evaluation? (yes/no) [default: {'yes' if args.enable_llm_judge and not args.disable_llm_judge else 'no'}]: ").lower()
    if judge_input:
        args.enable_llm_judge = judge_input in ["yes", "y"]
        args.disable_llm_judge = judge_input not in ["yes", "y"]

    if args.enable_llm_judge:
        judge_model_input = input(f"Judge model [default: {args.judge_model}]: ")
        if judge_model_input:
            args.judge_model = judge_model_input
            
        criteria_input = input(f"Use custom evaluation criteria? (yes/no) [default: no]: ").lower()
        if criteria_input in ["yes", "y"]:
            print("\nEnter custom evaluation criteria (press Enter twice when done):")
            criteria_lines = []
            while True:
                line = input()
                if line == "" and len(criteria_lines) > 0:
                    break
                criteria_lines.append(line)
            if criteria_lines:
                args.custom_criteria = "\n".join(criteria_lines)

    print("\n" + "-" * 64 + "\n")
    return args


def pick_device_and_dtype():
    """Return (device_str, torch_dtype, use_device_map_auto) chosen safely."""
    if torch.cuda.is_available():
        return "cuda", torch.float16, True
    if torch.backends.mps.is_available():
        # MPS prefers float32 for stability
        return "mps", torch.float32, False
    return "cpu", torch.float32, False


def ensure_pad_token(tokenizer):
    """Make sure pad_token exists; fall back to eos_token if needed."""
    if tokenizer.pad_token_id is None:
        # If eos exists, use it as pad; otherwise add a token as pad
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def render_chat_text(tokenizer, system_prompt, user_prompt, enable_thinking: bool):
    """
    Render a chat prompt robustly across tokenizers:
    - Use chat template if available
    - If 'enable_thinking' isn't supported, silently drop it
    - Else fall back to a simple concatenation
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,  # some tokenizers (e.g., Qwen) support this
            )
        except TypeError:
            # Older/other templates don't accept enable_thinking
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    # Fallback: naive formatting
    return f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"


def main():
    parser = argparse.ArgumentParser(description="Concept-Aware Decoding for Large Language Models")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", type=str, default="Name something parents would criticize their children for having.")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--enable_thinking", type=bool, default=True)
    parser.add_argument("--non_interactive", action="store_true")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--min_token_prob", type=float, default=0.001)
    parser.add_argument("--clustering_algo", type=str, default="agglomerative", choices=["agglomerative", "dbscan"])
    parser.add_argument("--distance_threshold", type=float, default=0.4)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--use_dim_reduction", type=bool, default=False)
    parser.add_argument("--dim_reduction_components", type=int, default=32)
    parser.add_argument("--concept_ranking_method", type=str, default="sum", choices=["sum", "max"])
    parser.add_argument("--enable_llm_judge", action="store_true", default=True, help="Enable LLM judge evaluation")
    parser.add_argument("--disable_llm_judge", action="store_true", help="Disable LLM judge evaluation")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen3-0.6B", help="LLM model for judging")
    parser.add_argument("--custom_criteria", type=str, default=None, help="Custom evaluation criteria")

    is_interactive = "--non_interactive" not in sys.argv
    args = get_user_config(parser) if is_interactive else parser.parse_args()

    # Device & dtype policy (robust for CUDA vs MPS vs CPU)
    device, torch_dtype, use_device_map_auto = pick_device_and_dtype()

    print(f"Loading model and tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ensure_pad_token(tokenizer)

    if use_device_map_auto:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model_device = next(model.parameters()).device
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        model_device = torch.device(device)

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id or eos_token_id

    print("Model loaded.")
    print(f"\nPrompt: \"{args.prompt}\"\n")
    print("---")

    # ---------------------------
    # [1] Baseline Decoding
    # ---------------------------
    print("[1] BASELINE DECODER (Standard Sampling)")
    start_time = time.time()

    system_prompt = "Output one word responses to the question."
    text = render_chat_text(tokenizer, system_prompt, args.prompt, args.enable_thinking)
    model_inputs = tokenizer([text], return_tensors="pt")
    # If model is sharded (device_map='auto'), you can leave inputs on CPU; otherwise move to model_device
    if not use_device_map_auto:
        model_inputs = {k: v.to(model_device) for k, v in model_inputs.items()}

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        do_sample=True,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
    )

    # Slice off the prompt portion robustly
    prompt_len = model_inputs["input_ids"].shape[1]
    output_ids = generated_ids[0, prompt_len:].tolist()
    baseline_output = tokenizer.decode(output_ids, skip_special_tokens=True)

    end_time = time.time()
    print(f"Output: {baseline_output.strip()}")
    print(f"\nGeneration Time: {end_time - start_time:.2f} seconds")
    print("---")

    # ---------------------------
    # [2] Concept Decoder
    # ---------------------------
    print("[2] CONCEPT-DECODER")
    logger = setup_logger()
    start_time = time.time()

    concept_output = generate_with_concept_decoder(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.k,
        args.distance_threshold,
        args.enable_thinking,
        args.clustering_algo,
        args.eps,
        args.use_dim_reduction,
        args.dim_reduction_components,
        args.concept_ranking_method,
        args.min_token_prob,
        logger,
    )

    end_time = time.time()
    print(f"\n\nGeneration Time: {end_time - start_time:.2f} seconds")

    print("\n--- Last 10 lines of concept_decoder.log ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, "concept_decoder.log")
    os.system(f"tail -n 10 {log_file_path}")

    # ---------------------------
    # [3] LLM Judge Evaluation
    # ---------------------------
    if args.enable_llm_judge and not args.disable_llm_judge:
        print("\n[3] LLM Judge Evaluation")
        print("=" * 50)

        try:
            # Initialize the LLM judge evaluator
            judge_evaluator = LLMJudgeEvaluator(
                judge_model_name=args.judge_model,
                device=str(model_device)
            )
            
            # Perform the comparison
            evaluation_results = judge_evaluator.compare_responses(
                prompt=args.prompt,
                baseline_output=baseline_output.strip(),
                concept_output=str(concept_output).strip(),
                evaluation_criteria=getattr(args, 'custom_criteria', None),
                debug=True  # Enable debug mode to see raw judge output
            )
            
            # Print the results
            judge_evaluator.print_comparison_summary(evaluation_results)
            
            # Save results to file
            results_file = judge_evaluator.save_results(evaluation_results)
            print(f"\nDetailed evaluation results saved to: {results_file}")
            
        except Exception as e:
            print(f"LLM Judge evaluation failed: {e}")
            print("Continuing without evaluation...")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()