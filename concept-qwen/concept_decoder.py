# concept-qwen/concept_decoder.py
import torch
import torch.nn.functional as F
import numpy as np
import shutil
from pathlib import Path

from concept_qwen_utilities import (
    get_top_k_candidates,
    filter_candidates,
    get_candidate_embeddings,
    cluster_embeddings,
    visualize_clusters,
    find_centroid_token,
    GRAMMAR_CRITICAL_TOKENS
)

class ConceptDecoder:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.config = config
        
        # Advanced Logging Setup
        self.logging_enabled = config.get('logging_enabled', False)
        self.log_mode = config.get('log_mode', 'both')
        self.log_dir_base = config.get('log_dir_base', Path(__file__).parent.resolve())
        self.log_dir = self.log_dir_base / "logs" # Default for interactive
        self.current_prompt_id = None

    def set_log_prompt_id(self, prompt_id):
        """Sets the prompt ID for benchmark logging to create prompt_n folders."""
        self.current_prompt_id = f"prompt_{prompt_id}"
        self.log_dir = self.log_dir_base / self.current_prompt_id

    def _setup_logging(self):
        if self.logging_enabled:
            if self.current_prompt_id is None:
                self.log_dir = self.log_dir_base / "logs"
            
            if self.log_dir.exists():
                shutil.rmtree(self.log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _log_step(self, step_num, embeddings, filtered_ids, filtered_probs, cluster_labels, ranked_concepts, chosen_concept_rank):
        if not self.logging_enabled:
            return
        
        step_path = self.log_dir / str(step_num)
        step_path.mkdir(exist_ok=True)

        if self.log_mode in ['visuals', 'both']:
            visualize_clusters(embeddings, cluster_labels, filtered_ids, self.tokenizer, step_path)

        if self.log_mode in ['logs', 'both']:
            with open(step_path / "log.txt", "w") as f:
                f.write(f"--- Step {step_num} Log ---\n\n")
                f.write("== Filtered Candidates ==\n")
                for tid, prob in zip(filtered_ids, filtered_probs):
                    f.write(f"Token: '{self.tokenizer.decode(tid)}' (ID: {tid.item()}), Prob: {prob.item():.4f}\n")

                f.write("\n== Ranked Concepts (Clusters) ==\n")
                for i, (prob, concept_tokens) in enumerate(ranked_concepts):
                    cluster_mask = torch.from_numpy(np.isin(filtered_ids.cpu(), concept_tokens.cpu())).to(self.device)
                    cluster_embeds = embeddings[cluster_mask]
                    centroid_token = find_centroid_token(cluster_embeds, concept_tokens, self.tokenizer)
                    
                    f.write(f"Rank {i+1} | Centroid: '{centroid_token}' | Score: {prob:.4f}\n")
                    f.write(f"  Tokens: {[self.tokenizer.decode(t) for t in concept_tokens]}\n")
                
                f.write(f"\n**Chosen Concept:** Rank {chosen_concept_rank}\n")

    def _get_next_token_dist(self, logits):
        k = self.config.get('top_k', 100)
        top_k_ids, top_k_probs = get_top_k_candidates(logits, k)
        
        # "Grammar First" Hybrid Approach:
        # Check if the top candidate is a critical grammar token.
        top_token_str = self.tokenizer.decode(top_k_ids[0], skip_special_tokens=True).strip()
        if top_token_str in GRAMMAR_CRITICAL_TOKENS:
            return top_k_ids[0] # If so, use it directly and skip clustering.

        # If not grammar, proceed with concept clustering as before.
        filtered_ids, filtered_probs = filter_candidates(top_k_ids, top_k_probs, self.tokenizer)
        if filtered_ids.nelement() == 0:
            return top_k_ids[0] # Fallback to top token if filter is too aggressive

        embeddings = get_candidate_embeddings(self.model, filtered_ids)
        
        cluster_labels = cluster_embeddings(embeddings, self.config.get('distance_threshold', 0.45)) if embeddings.shape[0] > 1 else np.array([0])

        unique_clusters = np.unique(cluster_labels)
        ranked_concepts = []
        if len(unique_clusters) > 0:
            for cluster_id in unique_clusters:
                mask = torch.from_numpy(cluster_labels == cluster_id).to(self.device)
                concept_prob = filtered_probs[mask].sum().item()
                concept_tokens = filtered_ids[mask]
                ranked_concepts.append((concept_prob, concept_tokens))
            
            ranked_concepts.sort(key=lambda x: x[0], reverse=True)
            
            if self.logging_enabled:
                self._log_step(self.step_num + 1, embeddings, filtered_ids, filtered_probs, cluster_labels, ranked_concepts, 1)

            best_concept_tokens = ranked_concepts[0][1]
            best_concept_probs = filtered_probs[torch.from_numpy(np.isin(filtered_ids.cpu(), best_concept_tokens.cpu())).to(self.device)]
            return best_concept_tokens, best_concept_probs
        
        return top_k_ids[0]

    def _nucleus_sample(self, token_ids, token_probs):
        top_p = self.config.get('top_p', 0.92)
        sorted_probs, sorted_indices = torch.sort(token_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        token_probs[indices_to_remove] = 0.0
        
        renormalized_probs = token_probs / token_probs.sum()
        return token_ids[torch.multinomial(renormalized_probs, 1).item()]

    def _sampling_step(self, generated_ids):
        outputs = self.model(generated_ids)
        dist = self._get_next_token_dist(outputs.logits)
        
        if isinstance(dist, tuple):
            token_ids, token_probs = dist
            next_token_id = self._nucleus_sample(token_ids, token_probs)
        else:
            next_token_id = dist
            
        return torch.cat([generated_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=-1)

    def _beam_search_step(self, beams):
        all_candidates = []
        for seq, score in beams:
            outputs = self.model(seq)
            dist = self._get_next_token_dist(outputs.logits)

            if isinstance(dist, tuple):
                token_ids, token_probs = dist
                token_probs = token_probs / token_probs.sum()
                for token_id, prob in zip(token_ids, token_probs):
                    if prob > 0:
                        new_seq = torch.cat([seq, token_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                        new_score = score - torch.log(prob)
                        all_candidates.append((new_seq, new_score))
            else:
                probs = F.softmax(outputs.logits[:, -1, :], dim=-1)
                token_prob = probs[0, dist]
                if token_prob > 0:
                    new_seq = torch.cat([seq, dist.unsqueeze(0).unsqueeze(0)], dim=-1)
                    new_score = score - torch.log(token_prob)
                    all_candidates.append((new_seq, new_score))
        
        ordered = sorted(all_candidates, key=lambda x: x[1])
        return ordered[:self.config.get('num_beams', 3)]

    def generate(self, prompt):
        strategy = self.config.get('strategy', 'sampling')
        print(f"\n--- CONCEPT-AWARE DECODER ({strategy.capitalize()}) ---")
        self._setup_logging()
        
        input_ids = self.tokenizer([prompt], return_tensors="pt").to(self.device).input_ids
        
        print(prompt, end="", flush=True)

        if strategy == 'sampling':
            generated_ids = input_ids
            for i in range(self.config.get('max_new_tokens', 100)):
                self.step_num = i
                generated_ids = self._sampling_step(generated_ids)
                
                decoded_token = self.tokenizer.decode(generated_ids[0, -1])
                print(decoded_token, end="", flush=True)
                if generated_ids[0, -1] == self.tokenizer.eos_token_id:
                    break
            final_output = generated_ids

        elif strategy == 'beam':
            num_beams = self.config.get('num_beams', 3)
            beams = [(input_ids, 0.0)]
            for i in range(self.config.get('max_new_tokens', 100)):
                self.step_num = i
                beams = self._beam_search_step(beams)
                if not beams or beams[0][0][0, -1] == self.tokenizer.eos_token_id:
                    break
            final_output = beams[0][0]
            full_text = self.tokenizer.decode(final_output[0])
            print(full_text[len(prompt):], end="", flush=True)

        print()
        return self.tokenizer.decode(final_output[0], skip_special_tokens=True)
