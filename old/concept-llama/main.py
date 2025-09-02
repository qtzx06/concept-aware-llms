import os
import re
import math
import shutil
import argparse
import logging
from pathlib import Path
from collections import Counter
import requests

from concept_decoder import (
    filter_and_clean_tokens,
    get_contextual_embeddings,
    reduce_and_cluster_embeddings,
    rank_concepts_by_paper_formula,
    select_token_from_best_concept,
    visualize_clusters,
)

# --- Configuration ---
LOG_DIR_BASE = Path(__file__).parent / "logs"

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConceptAwareDecoder:
    """
    A decoder that uses concept clustering to select the next token.
    This is a complete rewrite focused on a single, efficient API call.
    """
    def __init__(self, vllm_url: str, model: str, k: int):
        self.vllm_url = vllm_url
        self.model = model
        self.k = k

    def _format_prompt(self, prompt: str) -> str:
        """Wraps the user's prompt in the official Llama 3 chat template."""
        return (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _get_top_k_logprobs(self, prompt: str) -> dict[str, float] | None:
        """
        Gets the top-k log probabilities for the next token from a single API call.
        """
        logger.info(f"--- Step 1: Getting top-{self.k} predictions from a single API call ---")
        formatted_prompt = self._format_prompt(prompt)

        json_payload = {
            "model": self.model,
            "prompt": formatted_prompt,
            "max_tokens": 1,
            "logprobs": self.k,
            "temperature": 0.0, # Use 0.0 for deterministic, ground-truth probabilities
        }

        try:
            response = requests.post(self.vllm_url, json=json_payload, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("choices") and data["choices"][0].get("logprobs"):
                token_logprobs = data["choices"][0]["logprobs"]["top_logprobs"][0]
                # Clean tokens immediately by stripping whitespace
                return {token.strip(): math.exp(logprob) for token, logprob in token_logprobs.items()}
            logger.error("API response was successful but did not contain logprobs.")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API Error while fetching logprobs: {e}")
            return None

    def decode(self, prompt: str) -> tuple[str | None, dict]:
        """
        Performs the full decoding process based on a single model output.
        """
        # Step 1: Get the model's probability distribution for the next token.
        ground_truth_probs = self._get_top_k_logprobs(prompt)
        if not ground_truth_probs:
            return None, {"error": "Failed to get ground-truth probabilities from the model."}

        # Step 2: Clean the initial list of candidates from the model's output.
        logger.info("--- Step 2: Cleaning the initial candidate tokens ---")
        initial_tokens = list(ground_truth_probs.keys())
        initial_probs = list(ground_truth_probs.values())
        
        final_candidates, final_probs = filter_and_clean_tokens(initial_tokens, initial_probs)
        logger.info(f"--> {len(final_candidates)} candidates remain after cleaning.")

        if not final_candidates:
            # If cleaning removes everything, fall back to the original most likely token.
            top_token = max(ground_truth_probs, key=ground_truth_probs.get, default=None)
            return top_token, {"error": "No valid candidates remained after cleaning."}

        # Step 3: Embed, cluster, and rank the high-quality final candidates.
        logger.info("--- Step 3: Embedding, clustering, and ranking concepts ---")
        embeddings = get_contextual_embeddings(prompt, final_candidates, self.vllm_url, self.model)
        if embeddings is None:
            return None, {"error": "Failed to get embeddings for candidates."}
            
        cluster_labels = reduce_and_cluster_embeddings(embeddings)
        ranked_concepts = rank_concepts_by_paper_formula(final_candidates, final_probs, cluster_labels, Counter())
        if not ranked_concepts:
            top_token = max(ground_truth_probs, key=ground_truth_probs.get, default=None)
            return top_token, {"error": "No concepts were formed after clustering."}

        # Step 4: Select the best token from the winning concept.
        logger.info("--- Step 4: Selecting best token from top-ranked concept ---")
        best_concept = ranked_concepts[0]
        chosen_token = select_token_from_best_concept(best_concept, ground_truth_probs)
        
        # Prepare data for logging
        log_data = {
            "final_candidates": final_candidates,
            "ranked_concepts": ranked_concepts,
            "embeddings": embeddings,
            "cluster_labels": cluster_labels,
        }
        
        return chosen_token, log_data


def log_results(chosen_token: str, log_data: dict):
    """Logs the details of the decoding process to a file and creates a visualization."""
    if not chosen_token:
        logger.warning("No token was chosen, skipping log generation.")
        return

    safe_token_name = "".join(c if c.isalnum() else '_' for c in chosen_token)
    log_dir = LOG_DIR_BASE / safe_token_name
    
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    with open(log_dir / "log.txt", "w", encoding="utf-8") as f:
        f.write("--- Concept-Aware Decoding Log ---")
        f.write(f"Chosen Token: '{chosen_token}'\n\n")

        f.write("== Final Candidates (from Top-K, after cleaning) ==\n")
        for token in sorted(log_data.get("final_candidates", [])):
            f.write(f"- {token}\n")
            
        f.write("\n== Ranked Concepts (Clusters) ==\n")
        for i, concept in enumerate(log_data.get("ranked_concepts", [])):
            tokens_str = ", ".join(f"'{t}'" for t in concept['tokens'])
            f.write(f"Rank {i+1:02d} | Score: {concept['score']:.6f} {concept['details']}\n")
            f.write(f"  Tokens: [{tokens_str}]\n")

    logger.info(f"--> Concept decoding log saved to: {log_dir}")

    # Generate visualization if data is available
    if all(k in log_data for k in ["embeddings", "cluster_labels", "final_candidates"]):
        if log_data["embeddings"].shape[0] > 0 and len(log_data["final_candidates"]) > 0:
            visualize_clusters(
                embeddings=log_data["embeddings"],
                cluster_labels=log_data["cluster_labels"],
                tokens=log_data["final_candidates"],
                log_path=log_dir
            )
            logger.info(f"--> Cluster visualization saved to: {log_dir}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Concept-Aware Decoding with Llama (Single-Call Method)")
    parser.add_argument("--prompt", type=str, default="From the ingredients milk, eggs, and flour, I can make a single food item. That item is a ",
                        help="The prompt for the decoder.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="The name of the model to use.")
    parser.add_argument("--k", type=int, default=100,
                        help="The number of top-k predictions to retrieve and analyze.")
    parser.add_argument("--vllm_url", type=str, default=os.getenv("VLLM_URL", "https://upatpmb38ag5v3-8000.proxy.runpod.net/v1/completions"),
                        help="The URL of the vLLM server.")
    args = parser.parse_args()

    if not args.vllm_url:
        logger.error("VLLM_URL environment variable not set and --vllm_url not provided.")
        return

    # Clear previous logs for a clean run
    if LOG_DIR_BASE.exists():
        shutil.rmtree(LOG_DIR_BASE)
    LOG_DIR_BASE.mkdir(exist_ok=True)

    decoder = ConceptAwareDecoder(vllm_url=args.vllm_url, model=args.model, k=args.k)
    
    logger.info("--- Applying Concept-Aware Decoding for Prompt ---")
    chosen_token, log_data = decoder.decode(args.prompt)
    
    if chosen_token:
        logger.info(f"\n---> Concept-Aware Choice: '{chosen_token}' <---")
        log_results(chosen_token, log_data)
    else:
        error_msg = log_data.get("error", "An unknown error occurred.")
        logger.warning(f"\nCould not determine a token using concept-aware decoding: {error_msg}")


if __name__ == "__main__":
    main()