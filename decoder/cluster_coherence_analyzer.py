#!/usr/bin/env python3
"""
Cluster Coherence Analyzer
Measures semantic coherence of clusters using cosine similarity of token embeddings.
Robust to None/special tokens; friendly to CPU/MPS (Apple Silicon) and CUDA.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt


class ClusterCoherenceAnalyzer:
    """
    Analyzes the semantic coherence of token clusters using input-embedding weights
    from a causal LM. Measures within-cluster similarity vs inter-cluster similarity.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "auto"):
        """
        Initialize the analyzer with a language model.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("auto", "cpu", "mps", "cuda")
        """
        self.device = self._get_device(device)
        print(f"Loading model {model_name} on {self.device}...")

        # Precision / placement policy:
        # - CUDA: float16 + device_map='auto' typically OK
        # - MPS/CPU: prefer float32 and move explicitly
        if self.device == "cuda":
            torch_dtype = torch.float16
            device_map = "auto"
        else:
            torch_dtype = torch.float32
            device_map = None  # we'll move explicitly

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            device_map=device_map
        )
        if device_map is None:
            self.model = self.model.to(self.device)

        # Get embedding matrix
        self.embedding_matrix = self.model.get_input_embeddings().weight
        print(f"Model loaded. Embedding dimension: {tuple(self.embedding_matrix.shape)}")

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

    # ----------------------------
    # Tokenization & normalization
    # ----------------------------
    def _normalize_token_ids(self, token_ids: List[int]) -> List[int]:
        """
        Ensure a clean List[int] of token ids:
        - Replace None with unk_token_id
        - Drop special tokens
        - Cast to int
        """
        unk = self.tokenizer.unk_token_id
        special = set(getattr(self.tokenizer, "all_special_ids", []) or [])
        cleaned: List[int] = []
        for tid in token_ids:
            if tid is None:
                tid = unk
            try:
                tid = int(tid)
            except Exception:
                tid = unk
            if tid in special:
                continue
            cleaned.append(tid)
        if not cleaned:
            raise ValueError("No valid (non-special) token ids after normalization.")
        return cleaned

    def tokens_from_prompt(self, text: str) -> List[int]:
        """
        Convert a text prompt into a flat list of token ids (no special tokens).
        """
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False
        )
        ids = enc["input_ids"]
        if isinstance(ids[0], list):  # batched case
            ids = [x for row in ids for x in row]
        return self._normalize_token_ids(ids)

    def tokens_from_prompts(self, prompts: List[str]) -> List[List[int]]:
        """
        Batch convert multiple prompts to token ids (each prompt -> list of ids).
        """
        enc = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False
        )
        ids_list = enc["input_ids"]
        # Normalize each row
        return [self._normalize_token_ids(ids) for ids in ids_list]

    # ----------------------------
    # Embedding & similarity utils
    # ----------------------------
    def get_token_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """
        Get embeddings for given token IDs (robust to None/specials).

        Args:
            token_ids: List[int]

        Returns:
            Tensor of shape [num_tokens, embedding_dim] (float32, on CPU)
        """
        token_ids = self._normalize_token_ids(token_ids)
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        embeddings = self.embedding_matrix[token_ids_tensor].to(
            dtype=torch.float32
        ).detach().cpu()
        return embeddings

    def calculate_pairwise_similarities(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Calculate cosine similarities between all pairs of embeddings.

        Args:
            embeddings: Tensor [num_tokens, embedding_dim]

        Returns:
            Similarity matrix [num_tokens, num_tokens]
        """
        X = embeddings.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X = X / norms
        return cosine_similarity(X)

    # ----------------------------
    # Coherence / separation
    # ----------------------------
    def calculate_cluster_coherence(self, embeddings: torch.Tensor, cluster_labels: np.ndarray) -> Dict:
        """
        Calculate coherence metrics for each cluster.

        Args:
            embeddings: Token embeddings
            cluster_labels: Cluster assignments for each token

        Returns:
            Dictionary with coherence metrics
        """
        similarity_matrix = self.calculate_pairwise_similarities(embeddings)
        unique_clusters = np.unique(cluster_labels)

        cluster_coherences: Dict[int, Dict] = {}
        within_similarities: List[float] = []
        inter_similarities: List[float] = []

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) < 2:
                cluster_coherences[cluster_id] = {
                    "coherence": None,
                    "size": int(len(cluster_indices)),
                    "tokens": cluster_indices.tolist(),
                }
                continue

            # Within-cluster similarities
            cluster_similarities = similarity_matrix[cluster_indices][:, cluster_indices]
            within_sim = cluster_similarities[np.triu_indices(len(cluster_indices), k=1)]
            within_similarities.extend(within_sim.tolist())

            # Inter-cluster similarities
            other_indices = np.where(~cluster_mask)[0]
            if len(other_indices) > 0:
                inter_sim = similarity_matrix[cluster_indices][:, other_indices].flatten()
                inter_similarities.extend(inter_sim.tolist())

            cluster_coherences[cluster_id] = {
                "coherence": float(np.mean(within_sim)) if len(within_sim) > 0 else None,
                "size": int(len(cluster_indices)),
                "tokens": cluster_indices.tolist(),
            }

        return {
            "cluster_coherences": cluster_coherences,
            "mean_within_similarity": float(np.mean(within_similarities)) if within_similarities else 0.0,
            "mean_inter_similarity": float(np.mean(inter_similarities)) if inter_similarities else 0.0,
            "within_similarities": [float(x) for x in within_similarities],
            "inter_similarities": [float(x) for x in inter_similarities],
        }

    def calculate_separation_score(self, embeddings: torch.Tensor, cluster_labels: np.ndarray) -> float:
        """
        Calculate how well-separated different clusters are.

        Args:
            embeddings: Token embeddings
            cluster_labels: Cluster assignments

        Returns:
            Separation score (mean inter-cluster similarity; lower is better)
        """
        similarity_matrix = self.calculate_pairwise_similarities(embeddings)
        unique_clusters = np.unique(cluster_labels)

        if len(unique_clusters) < 2:
            return 0.0

        inter_cluster_similarities: List[float] = []
        for i, cluster1 in enumerate(unique_clusters):
            for cluster2 in unique_clusters[i + 1 :]:
                mask1 = cluster_labels == cluster1
                mask2 = cluster_labels == cluster2
                sims = similarity_matrix[mask1][:, mask2].flatten()
                inter_cluster_similarities.extend(sims.tolist())

        return float(np.mean(inter_cluster_similarities)) if inter_cluster_similarities else 0.0

    def analyze_clustering_quality(self, embeddings: torch.Tensor, cluster_labels: np.ndarray) -> Dict:
        """
        Comprehensive analysis of clustering quality.

        Args:
            embeddings: Token embeddings
            cluster_labels: Cluster assignments

        Returns:
            Dictionary with comprehensive quality metrics
        """
        coherence_results = self.calculate_cluster_coherence(embeddings, cluster_labels)
        separation_score = self.calculate_separation_score(embeddings, cluster_labels)
        silhouette_like = coherence_results["mean_within_similarity"] - separation_score

        quality_assessment = self._assess_quality(
            coherence_results["mean_within_similarity"],
            separation_score,
            silhouette_like
        )

        return {
            **coherence_results,
            "separation_score": separation_score,
            "silhouette_score": silhouette_like,
            "quality_assessment": quality_assessment,
            "num_clusters": int(len(np.unique(cluster_labels))),
            "total_tokens": int(len(cluster_labels)),
        }

    def _assess_quality(self, within_sim: float, separation: float, silhouette: float) -> str:
        """Assess the overall quality of clustering."""
        if within_sim >= 0.4 and separation <= 0.15:
            return "EXCELLENT - High coherence, good separation"
        elif within_sim >= 0.3 and separation <= 0.2:
            return "GOOD - Balanced coherence and separation"
        elif within_sim >= 0.2 and separation <= 0.25:
            return "FAIR - Moderate quality"
        else:
            return "POOR - Low coherence or poor separation"

    # ----------------------------
    # Visualization & reporting
    # ----------------------------
    def visualize_clusters(
        self,
        embeddings: torch.Tensor,
        cluster_labels: np.ndarray,
        analysis_results: Dict,
        save_path: Optional[str] = None,
    ):
        """
        Create visualizations of cluster analysis.

        Args:
            embeddings: Token embeddings
            cluster_labels: Cluster assignments
            analysis_results: Results from analyze_clustering_quality
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1) Similarity distribution
        axes[0, 0].hist(analysis_results["within_similarities"], bins=20, alpha=0.7, label="Within-cluster")
        axes[0, 0].hist(analysis_results["inter_similarities"], bins=20, alpha=0.7, label="Inter-cluster")
        axes[0, 0].axvline(
            analysis_results["mean_within_similarity"], linestyle="--", label=f'Mean within: {analysis_results["mean_within_similarity"]:.3f}'
        )
        axes[0, 0].axvline(
            analysis_results["mean_inter_similarity"], linestyle="--", label=f'Mean inter: {analysis_results["mean_inter_similarity"]:.3f}'
        )
        axes[0, 0].set_xlabel("Cosine Similarity")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title("Similarity Distribution")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2) Cluster sizes
        cluster_sizes = [analysis_results["cluster_coherences"][i]["size"] for i in sorted(analysis_results["cluster_coherences"].keys())]
        axes[0, 1].bar(range(len(cluster_sizes)), cluster_sizes)
        axes[0, 1].set_xlabel("Cluster ID")
        axes[0, 1].set_ylabel("Number of Tokens")
        axes[0, 1].set_title("Cluster Sizes")
        axes[0, 1].grid(True, alpha=0.3)

        # 3) Cluster coherences
        cluster_cohs = [analysis_results["cluster_coherences"][i]["coherence"] for i in sorted(analysis_results["cluster_coherences"].keys())]
        # Replace None with np.nan for plotting
        cluster_cohs = [np.nan if v is None else v for v in cluster_cohs]
        axes[1, 0].bar(range(len(cluster_cohs)), cluster_cohs)
        axes[1, 0].axhline(0.41, linestyle="--", label="Target coherence (0.41)")
        axes[1, 0].set_xlabel("Cluster ID")
        axes[1, 0].set_ylabel("Coherence Score")
        axes[1, 0].set_title("Cluster Coherences")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4) Summary
        summary_text = f"""
Quality Assessment: {analysis_results['quality_assessment']}

Mean Within-Cluster Similarity: {analysis_results['mean_within_similarity']:.3f}
Mean Inter-Cluster Similarity: {analysis_results['mean_inter_similarity']:.3f}
Separation Score: {analysis_results['separation_score']:.3f}
Silhouette-like Score: {analysis_results['silhouette_score']:.3f}

Number of Clusters: {analysis_results['num_clusters']}
Total Tokens: {analysis_results['total_tokens']}

Target Metrics:
- Within-cluster: 0.41 (your reference)
- Inter-cluster: 0.12 (your reference)
- BERT top-10: 0.22 (reference)
        """.strip("\n")
        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis("off")
        axes[1, 1].set_title("Analysis Summary")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plots saved to {save_path}")

        plt.show()

    def print_detailed_analysis(self, analysis_results: Dict):
        """Print a detailed analysis of clustering quality."""
        print("=" * 80)
        print("CLUSTER COHERENCE ANALYSIS")
        print("=" * 80)

        print(f"Quality Assessment: {analysis_results['quality_assessment']}")
        print(f"Number of Clusters: {analysis_results['num_clusters']}")
        print(f"Total Tokens: {analysis_results['total_tokens']}")
        print()

        print("SIMILARITY METRICS:")
        print(f"  Mean Within-Cluster Similarity: {analysis_results['mean_within_similarity']:.3f}")
        print(f"  Mean Inter-Cluster Similarity: {analysis_results['mean_inter_similarity']:.3f}")
        print(f"  Separation Score: {analysis_results['separation_score']:.3f}")
        print(f"  Silhouette-like Score: {analysis_results['silhouette_score']:.3f}")
        print()

        print("TARGET COMPARISON:")
        print(f"  Your Target Within-Cluster: 0.41")
        print(f"  Your Target Inter-Cluster: 0.12")
        print(f"  BERT Top-10 Reference: 0.22")
        print()

        print("PERFORMANCE vs TARGETS:")
        within_diff = analysis_results["mean_within_similarity"] - 0.41
        inter_diff = analysis_results["mean_inter_similarity"] - 0.12

        print(f"  Within-cluster difference: {within_diff:+.3f} ({'✓' if within_diff >= 0 else '✗'})")
        print(f"  Inter-cluster difference: {inter_diff:+.3f} ({'✓' if inter_diff <= 0 else '✗'})")
        print()

        print("INDIVIDUAL CLUSTER ANALYSIS:")
        for cluster_id, cluster_data in sorted(analysis_results["cluster_coherences"].items()):
            coh = cluster_data["coherence"]
            coh_str = f"{coh:.3f}" if coh is not None else "N/A"
            print(f"  Cluster {cluster_id}: coherence={coh_str}, size={cluster_data['size']}")

        print("=" * 80)


# ----------------------------
# Simple test/demo harness
# ----------------------------
def analyze_single_prompt(analyzer: ClusterCoherenceAnalyzer, prompt: str) -> Dict:
    """
    Tokenize a prompt, fetch embeddings, put all tokens in one cluster (for demo),
    and run the analysis.
    """
    token_ids = analyzer.tokens_from_prompt(prompt)
    emb = analyzer.get_token_embeddings(token_ids)
    labels = np.zeros(len(token_ids), dtype=int)  # all tokens in one cluster for demo
    results = analyzer.analyze_clustering_quality(emb, labels)
    return results


def analyze_multiple_prompts_as_clusters(analyzer: ClusterCoherenceAnalyzer, prompts: List[str]) -> Dict:
    """
    Treat each prompt as its own cluster: concatenate token embeddings and assign cluster labels
    per prompt, then analyze coherence/separation across prompts.
    """
    token_lists = analyzer.tokens_from_prompts(prompts)
    embeddings_list = [analyzer.get_token_embeddings(toks) for toks in token_lists]

    # Concatenate
    all_embeddings = torch.cat(embeddings_list, dim=0)
    # Build labels: 0 for first prompt's tokens, 1 for second, etc.
    labels = []
    for i, toks in enumerate(token_lists):
        labels.extend([i] * len(toks))
    labels = np.array(labels, dtype=int)

    return analyzer.analyze_clustering_quality(all_embeddings, labels)


def test_cluster_coherence():
    """
    Legacy-style synthetic test (random embeddings + round-robin labels).
    """
    analyzer = ClusterCoherenceAnalyzer()

    # Simulate some test data
    num_tokens = 50
    embedding_dim = analyzer.embedding_matrix.shape[1]

    # Create synthetic embeddings with some clustering structure
    embeddings = torch.randn(num_tokens, embedding_dim)

    # Create cluster labels (5 clusters)
    cluster_labels = np.array([i % 5 for i in range(num_tokens)])

    # Analyze clustering quality
    results = analyzer.analyze_clustering_quality(embeddings, cluster_labels)

    # Print results
    analyzer.print_detailed_analysis(results)

    # Create visualizations
    analyzer.visualize_clusters(embeddings, cluster_labels, results)


if __name__ == "__main__":
    # Example 1: analyze a single prompt (all tokens as one cluster)
    analyzer = ClusterCoherenceAnalyzer(device="auto")
    prompt = "Something people often forget to bring when traveling is"
    print("\n--- Single-prompt analysis ---")
    single_results = analyze_single_prompt(analyzer, prompt)
    analyzer.print_detailed_analysis(single_results)

    # Example 2: treat each prompt as its own cluster and compare across prompts
    print("\n--- Multi-prompt (each prompt = one cluster) analysis ---")
    prompts = [
        "Something people often forget to bring when traveling is",
        "A quick brown fox jumps over the lazy dog",
        "Deep learning models can capture complex data distributions"
    ]
    multi_results = analyze_multiple_prompts_as_clusters(analyzer, prompts)

    # Build concatenated embeddings/labels again for visualization
    token_lists = analyzer.tokens_from_prompts(prompts)
    emb_list = [analyzer.get_token_embeddings(toks) for toks in token_lists]
    all_emb = torch.cat(emb_list, dim=0)
    all_labels = np.concatenate([np.full(len(toks), i, dtype=int) for i, toks in enumerate(token_lists)])

    analyzer.print_detailed_analysis(multi_results)
    analyzer.visualize_clusters(all_emb, all_labels, multi_results)
