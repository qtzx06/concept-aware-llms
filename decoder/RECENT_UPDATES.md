### Recent Updates (Concept-Aware Decoding + Coherence)

- **New: `ClusterCoherenceAnalyzer` (`decoder/cluster_coherence_analyzer.py`)**
  - Measures semantic coherence of clusters using cosine similarity over token embeddings (word2vec-style via model input embeddings).
  - Reports: mean within-cluster similarity, mean inter-cluster similarity, separation score, silhouette-like score, per-cluster stats, and visualizations.
  - Device auto-detection supports Apple Silicon (`mps`), CUDA, or CPU.
  - Default model updated to `Qwen/Qwen1.5-8B` (user change).

- **New: Enhanced Concept Decoder (`decoder/enhanced_concept_decoder.py`)**
  - Iterative token sampling: after a token (single or multi-piece) is selected, it is appended to the prompt and decoding continues.
  - Concept ranking combines token probability with cluster coherence targets (aims for ≥0.41 within-cluster, ≤0.12 inter-cluster similarity).
  - Tunable clustering (PCA components, cosine-average agglomerative clustering with distance threshold).
  - "Vibe check" filtering: removes low-signal tokens (punctuation-only, whitespace, purely numeric, overly-common short stopwords).
  - Apple Silicon friendly (uses `mps` if available).

- **New: End-to-End Test Script (`decoder/test_cluster_coherence_and_decoder.py`)**
  - Part 1: Sweeps clustering params to approach target coherence levels; prints best config and summary; saves plots.
  - Part 2: Compares Enhanced Concept Decoding vs. standard sampling on several prompts.
  - Part 3: Demonstrates iterative token sampling behavior on a longer prompt.

- **Notebook note (`llama.ipynb`)**
  - vLLM was replaced by `transformers` imports to avoid compile-time install issues on macOS; if you still use vLLM cells, prefer running the Python scripts above for Apple Silicon.

### How to Run

- Measure coherence + run enhanced decoder:
```bash
cd decoder
python test_cluster_coherence_and_decoder.py
```

- Use analyzer directly:
```python
from decoder.cluster_coherence_analyzer import ClusterCoherenceAnalyzer
analyzer = ClusterCoherenceAnalyzer(model_name="Qwen/Qwen1.5-8B")
# embeddings: torch.Tensor [num_tokens, hidden_dim]
# cluster_labels: np.ndarray [num_tokens]
results = analyzer.analyze_clustering_quality(embeddings, cluster_labels)
analyzer.print_detailed_analysis(results)
```

- Use enhanced decoder:
```python
from decoder.enhanced_concept_decoder import EnhancedConceptDecoder
decoder = EnhancedConceptDecoder(model_name="Qwen/Qwen1.5-8B")
text = decoder.generate_with_enhanced_concept_decoding(
  "I can't get home for the holidays because of the ",
  max_tokens=20,
  temperature=0.0,
  top_k=100,
)
```

### Targets and Interpretation

- **Targets:** within-cluster ≈ 0.41; inter-cluster ≈ 0.12; BERT top-10 ≈ 0.22 (reference).
- The analyzer prints whether results meet/approach targets and provides per-cluster coherence and separation.
- Enhanced decoder biases concept selection toward clusters with higher coherence and reasonable separation.

### Apple Silicon Notes

- The analyzer and decoder auto-select `mps` when available; no CUDA required.
- Avoid `vllm` on macOS if wheels fail; the provided scripts rely on `transformers` and run on MPS/CPU.

### Files Touched

- Added: `decoder/cluster_coherence_analyzer.py`
- Added: `decoder/enhanced_concept_decoder.py`
- Added: `decoder/test_cluster_coherence_and_decoder.py`
- Updated default analyzer model (user change): `Qwen/Qwen1.5-8B`
