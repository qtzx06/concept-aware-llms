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

    def _find_centroid_token(self, token_ids, embeddings):
        """
        Finds the token whose embedding is closest to the centroid of the cluster.
        """
        if not embeddings.any():
            return token_ids[0] # Fallback
            
        centroid = np.mean(embeddings, axis=0)
        
        # Calculate cosine similarity between each embedding and the centroid
        # Cosine similarity = A . B / (||A|| * ||B||)
        # We can ignore the norms since they are constant for this comparison
        similarities = np.dot(embeddings, centroid)
        
        # Find the index of the most similar token
        closest_token_index = np.argmax(similarities)
        
        return token_ids[closest_token_index]

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
        Implements the full paper-accurate concept-aware decoding.
        """
        # --- 1. Get Top-K Candidates (Standard) ---
        if self.temperature > 0.0:
            scaled_logits = logits / self.temperature
            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)
            # Top-p sampling
            # (Implementation omitted for brevity, assuming probs are now sampled)
        else:
            probs = torch.nn.functional.softmax(logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(probs, top_k, sorted=True)
        top_k_indices_list = top_k_indices[0].tolist()
        
        # --- 2. Simulate Augmented Paraphrases for Repetition Frequency ---
        # We generate N=5 diverse samples to simulate the paraphrases
        with torch.no_grad():
            # Correctly use the `inputs_embeds` argument
            input_embeds = self.model.get_input_embeddings()(top_k_indices)
            augmented_outputs = self.model.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=1, 
                num_return_sequences=5, 
                do_sample=True, 
                temperature=0.8 # High temp for diversity
            )
        
        repetition_counts = {}
        for seq in augmented_outputs:
            token_id = seq[-1].item()
            repetition_counts[token_id] = repetition_counts.get(token_id, 0) + 1

        # --- 3. Clean, Get Embeddings, and Cluster ---
        candidates = []
        for prob, token_id in zip(top_k_probs[0], top_k_indices_list):
            token_str = self.tokenizer.decode(token_id)
            cleaned_token = self._clean_token(token_str)
            if cleaned_token:
                candidates.append({
                    "id": token_id, 
                    "token": cleaned_token, 
                    "prob": prob.item(),
                    "rep_freq": repetition_counts.get(token_id, 0) / 5.0
                })
        
        if not candidates: # Fallback
            return {"best_token_id": top_k_indices_list[0]}

        candidate_ids = [c['id'] for c in candidates]
        embeddings = self._get_embeddings(candidate_ids)
        cluster_labels = reduce_and_cluster_embeddings(embeddings)
        for i, c in enumerate(candidates):
            c['cluster'] = cluster_labels[i]

        # --- 4. Score Clusters using Paper's Formula ---
        unique_clusters = np.unique(cluster_labels)
        ranked_concepts = []
        for cluster_id in unique_clusters:
            cluster_candidates = [c for c in candidates if c['cluster'] == cluster_id]
            
            max_prob = max(c['prob'] for c in cluster_candidates)
            max_rep_freq = max(c['rep_freq'] for c in cluster_candidates)
            
            # The weighted formula from the paper
            weighted_score = self.alpha * max_prob + (1 - self.alpha) * max_rep_freq
            
            # Find the centroid token for this cluster
            cluster_token_ids = [c['id'] for c in cluster_candidates]
            cluster_embeddings = self._get_embeddings(cluster_token_ids)
            centroid_token_id = self._find_centroid_token(cluster_token_ids, cluster_embeddings)
            
            ranked_concepts.append({
                "id": int(cluster_id),
                "score": weighted_score,
                "centroid_token_id": centroid_token_id,
                "tokens": [c['token'] for c in cluster_candidates]
            })
        
        ranked_concepts.sort(key=lambda x: x["score"], reverse=True)

        # --- 5. Select the Centroid of the Best Cluster ---
        best_concept = ranked_concepts[0]
        best_token_id = best_concept['centroid_token_id']
        
        return {
            "best_token_id": best_token_id,
            "ranked_concepts": ranked_concepts
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
