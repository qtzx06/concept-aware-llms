import torch
import numpy as np
import re
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.clustering import reduce_and_cluster_embeddings

class ConceptDecoder:
    def __init__(self, model_name, device="auto", decoding_strategy="concept_aware", entropy_threshold=2.5, temperature=0.7, top_p=0.95, alpha=0.7):
        self.model_name = model_name
        self.decoding_strategy = decoding_strategy
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature
        self.top_p = top_p
        self.alpha = alpha
        
        print(f"Loading model: {self.model_name}...")
        print(f"Decoding strategy: {self.decoding_strategy}, Entropy threshold: {self.entropy_threshold}, Temp: {self.temperature}, Top-p: {self.top_p}, Alpha: {self.alpha}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=device
            )
            self.device = self.model.device
            print(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _get_embeddings(self, token_ids):
        """
        Gets the embedding vectors for the provided token IDs from the model's embedding matrix.
        """
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        
        token_ids = token_ids.to(self.model.device)
        
        # Access the embedding matrix (works for most decoder models)
        embedding_matrix = self.model.get_input_embeddings().weight
        
        return embedding_matrix[token_ids].float().cpu().detach().numpy()

    def _clean_token(self, token_str):
        """
        Cleans a single token by removing special characters and extra whitespace.
        """
        # Rule 0: reject any special control tokens like <|eot_id|>
        if re.match(r"^<\|.*\|>$", token_str):
            return None
        
        # Rule 1: reject strings with no letters
        if not re.search(r"[a-zA-Z]", token_str):
            return None
            
        return token_str.strip()

    def _get_concept_aware_token(self, logits, top_k=100):
        """
        Performs concept-aware filtering on a tensor of logits.
        """
        if self.temperature > 0.0:
            # Apply temperature scaling and top-p sampling
            scaled_logits = logits / self.temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[0, indices_to_remove] = 0
            probs = probs / torch.sum(probs)
        else:
            # Deterministic case, no sampling
            probs = torch.nn.functional.softmax(logits, dim=-1)

        # Get the top_k candidates from the final probability distribution
        top_k_probs, top_k_indices = torch.topk(probs, top_k, sorted=True)
        
        top_k_tokens_raw = self.tokenizer.convert_ids_to_tokens(top_k_indices[0])
        top_k_indices_list = top_k_indices[0].tolist()
        top_k_probs_list = top_k_probs[0].tolist()

        # --- Clean Tokens ---
        candidates = []
        for token, prob, token_id in zip(top_k_tokens_raw, top_k_probs_list, top_k_indices_list):
            cleaned_token = self._clean_token(token)
            if cleaned_token:
                candidates.append({"id": token_id, "token": cleaned_token, "prob": prob})
        
        if not candidates:
            # Fallback to greedy if all tokens are cleaned
            fallback_id = top_k_indices[0][0].item()
            fallback_token = self.tokenizer.decode(fallback_id)
            return { "best_token_id": fallback_id, "best_token": fallback_token, "ranked_concepts": [], "candidates": [] }

        # --- Get Embeddings and Cluster ---
        candidate_ids = [c['id'] for c in candidates]
        embeddings = self._get_embeddings(candidate_ids)
        cluster_labels = reduce_and_cluster_embeddings(embeddings)

        # --- Rank Concepts ---
        unique_clusters = np.unique(cluster_labels)
        ranked_concepts = []
        for i, c in enumerate(candidates):
            c['cluster'] = cluster_labels[i]

        for cluster_id in unique_clusters:
            cluster_tokens = [c for c in candidates if c['cluster'] == cluster_id]
            cluster_score = sum(c['prob'] for c in cluster_tokens)
            ranked_concepts.append({
                "id": int(cluster_id),
                "tokens": [c['token'] for c in cluster_tokens],
                "score": cluster_score
            })
        
        ranked_concepts.sort(key=lambda x: x["score"], reverse=True)

        # --- Select Best Token from Best Concept ---
        best_concept_id = ranked_concepts[0]['id']
        tokens_in_best_concept = [c for c in candidates if c['cluster'] == best_concept_id]
        
        best_token_info = max(tokens_in_best_concept, key=lambda c: c['prob'])
        
        return {
            "best_token_id": best_token_info['id'],
            "best_token": best_token_info['token'],
            "ranked_concepts": ranked_concepts,
            "candidates": candidates
        }

    def generate_answers(self, prompts, max_new_tokens=50):
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing prompt {i+1}/{len(prompts)}...")
            
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            all_concepts = []
            generation_log = []
            
            try:
                for _ in range(max_new_tokens):
                    # Get logits for the very next token
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        next_token_logits = outputs.logits[:, -1, :]
                    
                    # --- Entropy Gating Logic ---
                    # Always calculate entropy on the true, unscaled distribution
                    unscaled_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                    entropy = -torch.sum(unscaled_probs * torch.log(unscaled_probs + 1e-9), dim=-1).item()
                    
                    use_concept_logic = True
                    if self.decoding_strategy == 'entropy_gated' and entropy < self.entropy_threshold:
                        use_concept_logic = False

                    if use_concept_logic:
                        # Intervention: Use concept-aware logic
                        output = self._get_concept_aware_token(next_token_logits)
                        best_token_id = output['best_token_id']
                        decision = "concept"
                    else:
                        # No intervention: Use standard sampling or greedy
                        if self.temperature > 0.0:
                            scaled_logits = next_token_logits / self.temperature
                            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
                            best_token_id = torch.multinomial(probs, num_samples=1).item()
                            decision = "sample"
                        else: # Deterministic case
                            best_token_id = torch.argmax(next_token_logits, dim=-1).item()
                            decision = "greedy"
                        output = {"ranked_concepts": []}
                    
                    # Log the step
                    generation_log.append({
                        "token": self.tokenizer.decode(best_token_id),
                        "entropy": entropy,
                        "decision": decision
                    })
                    
                    # Store concepts from the first step for analysis
                    if not all_concepts and use_concept_logic:
                        all_concepts = output['ranked_concepts']

                    # Stop if EOS token is generated
                    if best_token_id == self.tokenizer.eos_token_id:
                        break
                    
                    # Append the chosen token ID to the input for the next iteration
                    input_ids = torch.cat([input_ids, torch.tensor([[best_token_id]], device=self.model.device)], dim=-1)

                # Decode the generated tokens, excluding the prompt
                prompt_token_count = len(self.tokenizer.encode(prompt))
                generated_ids = input_ids[0][prompt_token_count:]
                full_prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                results.append({
                    "prediction": full_prediction.strip(),
                    "concepts": all_concepts,
                    "generation_log": generation_log
                })

            except Exception as e:
                print(f"Error during concept-aware generation for prompt '{prompt}': {e}")
                results.append({"prediction": "", "concepts": [], "generation_log": []})
        return results
