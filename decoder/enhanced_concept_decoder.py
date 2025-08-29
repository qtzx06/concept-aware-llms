import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import math
import re
from cluster_coherence_analyzer import ClusterCoherenceAnalyzer
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    _HAS_ST = False

class EnhancedConceptDecoder:
    """
    Enhanced concept decoder that:
    1. Measures cluster coherence and optimizes for target levels (0.41 within, 0.12 inter)
    2. Implements iterative token sampling (passes sampled tokens back to continue)
    3. Passes the "vibe check" with improved semantic coherence
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen1.5-0.5B", device: str = "auto"):
        """
        Initialize the enhanced concept decoder.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cpu", "mps", "cuda")
        """
        self.device = self._get_device(device)
        print(f"Loading model {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "mps" else None
        )
        
        if self.device == "mps":
            self.model = self.model.to(self.device)
        
        # Initialize coherence analyzer
        self.coherence_analyzer = ClusterCoherenceAnalyzer(model_name, device)

        # Optional: sentence-transformers backend for stronger semantic embeddings
        self.use_sentence_transformers = _HAS_ST
        self.st_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.st_model: Optional[SentenceTransformer] = None
        if self.use_sentence_transformers:
            try:
                # SentenceTransformer uses its own device handling; set to 'cpu' for MPS to avoid dtype issues
                st_device = 'cpu' if self.device == 'mps' else None
                self.st_model = SentenceTransformer(self.st_model_name, device=st_device)
            except Exception:
                self.use_sentence_transformers = False
        
        # Clustering parameters optimized for target coherence
        self.pca_components = 64  # Reduced for better clustering (post-ST)
        self.distance_threshold = 0.12  # Tighter for compact clusters
        self.use_dbscan = True
        self.dbscan_eps = 0.28  # increased to reduce singletons
        self.dbscan_min_samples = 2
        self.use_kmeans = True  # spherical k-means via normalized KMeans
        self.min_cluster_size = 2
        
        # Stopwords for filtering
        self.stopwords = {
            'a', 'an', 'the', 'and', 'but', 'or', 'so', 'if', 'for', 'of', 'in', 'to', 'on', 'with', 'by', 'at', 'from',
            'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'he', 'she', 'it',
            'they', 'we', 'you', 'i', 'that', 'which', 'who', 'what', 'when', 'where', 'why', 'how', 'about', 'above',
            'after', 'below', 'between', 'down', 'during', 'into', 'out', 'over', 'through', 'under', 'up', 'no',
            'not', 'only', 'very', 's', 't', 'll', 've', 're', 'd', 'm'
        }
        
        print("Enhanced Concept Decoder initialized!")
    
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def get_token_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """Get embeddings for given token IDs."""
        token_ids_tensor = torch.tensor(token_ids, device=self.device)
        embeddings = self.model.get_input_embeddings().weight[token_ids_tensor]
        return embeddings.float().cpu()
    
    def filter_and_clean_tokens(self, tokens: List[str], probabilities: List[float]) -> Tuple[List[str], List[float]]:
        """
        Filter and clean tokens, removing stopwords and low-quality tokens.
        Enhanced to pass the "vibe check".
        """
        cleaned_tokens = []
        cleaned_probs = []
        
        for token, prob in zip(tokens, probabilities):
            # Clean the token
            clean_token = token.strip()
            
            # Skip if empty or too short
            if len(clean_token) < 1:
                continue
            
            # Skip stopwords (but be less aggressive)
            if clean_token.lower() in self.stopwords and len(clean_token) <= 3:
                continue
            
            # Skip tokens that are just punctuation
            if re.match(r'^[^\w\s]*$', clean_token):
                continue
            
            # Skip tokens that are just numbers
            if clean_token.isdigit():
                continue
            
            # Skip tokens that are just whitespace
            if clean_token.isspace():
                continue
            
            cleaned_tokens.append(clean_token)
            cleaned_probs.append(prob)
        
        return cleaned_tokens, cleaned_probs
    
    def _aggregate_to_word_level(self, tokens: List[str], token_ids: List[int], embeddings: torch.Tensor) -> Tuple[List[str], np.ndarray]:
        """Aggregate subword pieces to word-level and produce embeddings.
        If sentence-transformers is available, embed words with ST; otherwise mean-pool LM embeddings.
        """
        # Build words from subwords
        words: List[str] = []
        for tok in tokens:
            w = tok.replace("##", "").strip()
            if not w:
                w = tok.strip()
            words.append(w)

        # Deduplicate while preserving order
        seen = set()
        unique_words: List[str] = []
        for w in words:
            wl = w.lower()
            if wl and wl not in seen:
                seen.add(wl)
                unique_words.append(w)

        if self.use_sentence_transformers and self.st_model is not None and len(unique_words) > 0:
            try:
                word_embs = self.st_model.encode(unique_words, normalize_embeddings=True)
                if isinstance(word_embs, list):
                    word_embs = np.asarray(word_embs)
                return unique_words, np.asarray(word_embs, dtype=np.float32)
            except Exception:
                # fall back to LM embeddings
                pass

        # Fallback: mean-pool subword LM embeddings per word
        word_to_vecs: Dict[str, List[np.ndarray]] = {}
        for tok, emb in zip(tokens, embeddings.numpy()):
            w = tok.replace("##", "").strip() or tok.strip()
            key = w.lower()
            if not key:
                continue
            word_to_vecs.setdefault(key, []).append(emb)
        out_words: List[str] = []
        out_vecs: List[np.ndarray] = []
        for key, vecs in word_to_vecs.items():
            out_words.append(key)
            out_vecs.append(np.mean(np.stack(vecs, axis=0), axis=0))
        if not out_vecs:
            return unique_words, np.zeros((0, embeddings.shape[1]), dtype=np.float32)
        return out_words, np.array(out_vecs, dtype=np.float32)

    def cluster_tokens_with_coherence_optimization(self, embeddings: torch.Tensor, 
                                                 tokens: List[str], 
                                                 probabilities: List[float]) -> Tuple[np.ndarray, Dict]:
        """
        Cluster tokens with optimization for target coherence levels.
        Returns cluster labels and coherence analysis.
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings)), {}
        
        # Normalize and aggregate to word-level
        token_ids_tmp = [self.tokenizer.convert_tokens_to_ids([t])[0] if self.tokenizer.convert_tokens_to_ids([t]) else 0 for t in tokens]
        words, word_embs = self._aggregate_to_word_level(tokens, token_ids_tmp, embeddings)
        X = word_embs.astype(np.float32)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        # Reduce dimensionality for better clustering
        if X.shape[1] > self.pca_components and X.shape[0] > 2:
            pca = PCA(n_components=min(self.pca_components, X.shape[0]-1))
            embeddings_reduced = pca.fit_transform(X)
        else:
            embeddings_reduced = X
        
        # Perform clustering with optimized parameters
        if self.use_kmeans and X.shape[0] >= 4:
            k = max(2, min(12, int(np.sqrt(X.shape[0]))))
            # KMeans on normalized vectors approximates spherical k-means
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(embeddings_reduced)
        elif self.use_dbscan:
            clustering = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples, metric="cosine")
            cluster_labels = clustering.fit_predict(embeddings_reduced)
            # If too many singletons/noise, fallback to Agglomerative with lower threshold
            num_noise = int(np.sum(cluster_labels == -1))
            if num_noise > max(2, int(0.3 * len(cluster_labels))):
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=max(0.08, self.distance_threshold * 0.9),
                    metric="cosine",
                    linkage="average"
                )
                cluster_labels = clustering.fit_predict(embeddings_reduced)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage="average"
            )
            cluster_labels = clustering.fit_predict(embeddings_reduced)
        
        # Analyze coherence
        # Recompute coherence on normalized, word-level space for better fidelity
        coherence_analysis = self.coherence_analyzer.analyze_clustering_quality(
            torch.from_numpy(X), cluster_labels
        )
        
        return cluster_labels, coherence_analysis
    
    def rank_concepts_by_coherence(self, tokens: List[str], probabilities: List[float], 
                                 cluster_labels: np.ndarray, coherence_analysis: Dict) -> List[Dict]:
        """
        Rank concepts by combining probability and coherence quality.
        Enhanced to favor clusters that meet target coherence levels.
        """
        unique_clusters = np.unique(cluster_labels)
        ranked_concepts = []
        
        token_to_prob = {token: prob for token, prob in zip(tokens, probabilities)}
        cluster_coherences = coherence_analysis['cluster_coherences']
        
        for cluster_id in unique_clusters:
            concept_tokens = [token for i, token in enumerate(tokens) if cluster_labels[i] == cluster_id]
            if not concept_tokens:
                continue
            
            # Calculate base score from probabilities
            cluster_score = sum(token_to_prob.get(token, 0) for token in concept_tokens)
            
            # Get coherence score for this cluster
            coh_val = cluster_coherences.get(cluster_id, {}).get('coherence', None)
            cluster_coherence = 0.0 if coh_val is None else coh_val
            
            # Stronger bonus for coherence
            coherence_bonus = max(0.0, cluster_coherence - 0.30) * 0.8
            
            # Penalty for being too similar to other clusters (inter-cluster similarity > 0.12)
            if coherence_analysis['mean_inter_similarity'] > 0.12:
                inter_penalty = (coherence_analysis['mean_inter_similarity'] - 0.12) * 0.5
                cluster_score -= max(0.0, inter_penalty)
            
            # Apply coherence bonus
            cluster_score += coherence_bonus
            
            ranked_concepts.append({
                "id": cluster_id,
                "tokens": concept_tokens,
                "score": cluster_score,
                "coherence": cluster_coherence,
                "size": len(concept_tokens)
            })
        
        # Sort by score
        ranked_concepts.sort(key=lambda x: x["score"], reverse=True)
        return ranked_concepts
    
    def select_best_token_from_concept(self, best_concept: Dict, all_predictions: Dict[str, float]) -> str:
        """Select the best token from the winning concept cluster."""
        return max(best_concept["tokens"], key=lambda token: all_predictions.get(token, -1))
    
    def generate_with_enhanced_concept_decoding(self, prompt: str, max_tokens: int = 50, 
                                              temperature: float = 0.0, top_k: int = 100) -> str:
        """
        Generate text using enhanced concept decoding with iterative token sampling.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            
        Returns:
            Generated text
        """
        print(f"Generating with Enhanced Concept Decoder...")
        print(f"Prompt: '{prompt}'")
        print(f"Max tokens: {max_tokens}")
        
        generated_tokens = []
        current_prompt = prompt
        
        for step in range(max_tokens):
            print(f"\n--- Step {step + 1}/{max_tokens} ---")
            
            # Tokenize current prompt
            inputs = self.tokenizer(current_prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token predictions
                
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Get top-k tokens
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                top_k_probs = torch.softmax(top_k_logits, dim=-1)
                
                # Convert to token strings
                top_k_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_k_indices]
                top_k_probs = top_k_probs.cpu().numpy()
                
                # Create predictions dictionary
                predictions = {token: prob for token, prob in zip(top_k_tokens, top_k_probs)}
            
            print(f"Top-k tokens obtained: {len(predictions)}")
            
            # Filter and clean tokens
            initial_tokens = list(predictions.keys())
            initial_probs = list(predictions.values())
            final_tokens, final_probs = self.filter_and_clean_tokens(initial_tokens, initial_probs)
            
            if not final_tokens:
                print("No valid tokens after filtering, using fallback...")
                chosen_token = max(predictions, key=predictions.get)
            else:
                print(f"Valid tokens after filtering: {len(final_tokens)}")
                
                # Get embeddings for final tokens
                final_token_ids = [self.tokenizer.convert_tokens_to_ids([token])[0] 
                                 for token in final_tokens if self.tokenizer.convert_tokens_to_ids([token])]
                
                if not final_token_ids:
                    chosen_token = max(predictions, key=predictions.get)
                else:
                    embeddings = self.get_token_embeddings(final_token_ids)
                    
                    # Cluster tokens with coherence optimization
                    cluster_labels, coherence_analysis = self.cluster_tokens_with_coherence_optimization(
                        embeddings, final_tokens, final_probs
                    )
                    
                    # Print coherence analysis
                    print(f"Clustering results:")
                    print(f"  Within-cluster similarity: {coherence_analysis.get('mean_within_similarity', 0):.3f}")
                    print(f"  Inter-cluster similarity: {coherence_analysis.get('mean_inter_similarity', 0):.3f}")
                    print(f"  Number of clusters: {coherence_analysis.get('num_clusters', 0)}")
                    
                    # Rank concepts by coherence
                    ranked_concepts = self.rank_concepts_by_coherence(
                        final_tokens, final_probs, cluster_labels, coherence_analysis
                    )
                    
                    if ranked_concepts:
                        best_concept = ranked_concepts[0]
                        # Enforce coherence floor: if too low, fall back to highest-prob token
                        if best_concept['coherence'] is not None and best_concept['coherence'] < 0.25:
                            chosen_token = max(predictions, key=predictions.get)
                        else:
                            chosen_token = self.select_best_token_from_concept(best_concept, predictions)
                        
                        print(f"Selected concept {best_concept['id']}:")
                        print(f"  Tokens: {best_concept['tokens'][:5]}...")
                        print(f"  Coherence: {best_concept['coherence']:.3f}")
                        print(f"  Size: {best_concept['size']}")
                    else:
                        chosen_token = max(predictions, key=predictions.get)
            
            # Add chosen token to generated sequence
            generated_tokens.append(chosen_token)
            current_prompt += chosen_token
            
            print(f"Chosen token: '{chosen_token}'")
            
            # Check for end conditions
            if chosen_token.strip() in ['</s>', '<|endoftext|>', '\n\n']:
                print("End token detected, stopping generation.")
                break
        
        # Decode final result
        generated_text = "".join(generated_tokens)
        print(f"\nGenerated text: '{generated_text}'")
        
        return generated_text
    
    def compare_with_standard_sampling(self, prompt: str, max_tokens: int = 50) -> Dict:
        """
        Compare enhanced concept decoding with standard sampling.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Comparison results
        """
        print("=" * 80)
        print("COMPARISON: Enhanced Concept Decoder vs Standard Sampling")
        print("=" * 80)
        
        # Enhanced concept decoding
        print("\n1. Enhanced Concept Decoder:")
        concept_text = self.generate_with_enhanced_concept_decoding(prompt, max_tokens)
        
        # Standard sampling
        print("\n2. Standard Sampling:")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        standard_text = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"Generated text: '{standard_text}'")
        
        # Results
        results = {
            "prompt": prompt,
            "enhanced_concept_text": concept_text,
            "standard_text": standard_text,
            "enhanced_length": len(concept_text),
            "standard_length": len(standard_text)
        }
        
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)
        print(f"Enhanced Concept Decoder: '{concept_text}'")
        print(f"Standard Sampling: '{standard_text}'")
        print(f"Enhanced length: {len(concept_text)} chars")
        print(f"Standard length: {len(standard_text)} chars")
        print("=" * 80)
        
        return results

def test_enhanced_concept_decoder():
    """Test the enhanced concept decoder."""
    decoder = EnhancedConceptDecoder()
    
    # Test prompts
    test_prompts = [
        "The best way to learn programming is",
        "I can't get home for the holidays because of the",
        "The most important quality in a friend is",
        "What makes a good leader is"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Testing prompt: '{prompt}'")
        print(f"{'='*60}")
        
        results = decoder.compare_with_standard_sampling(prompt, max_tokens=20)
        
        print(f"\nResults for '{prompt}':")
        print(f"Enhanced: {results['enhanced_concept_text']}")
        print(f"Standard: {results['standard_text']}")

if __name__ == "__main__":
    test_enhanced_concept_decoder()
