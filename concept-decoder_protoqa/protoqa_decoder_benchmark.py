#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ProtoQA decoder benchmark (baseline vs. concept-aware) for HF causal LMs.

- Baseline: greedy top-k next-token seeds -> complete to a SINGLE WORD (word-boundary stop),
            keep top-10 unique words by next-token probability.
- Concept: same top-k WORD candidates, embed words (avg of sub-token embeddings),
           reduce + cluster (Agglomerative on cosine distance), rank clusters,
           pick cluster-centroid words (by cosine-to-centroid) for top-10.

Both paths output *words* (not BPE fragments), matching the Concept-BERT ProtoQA setup
(“mask last word and fill one word”). This aligns with Chen Shani’s guidance.

CSV columns:
id,question,prefix,gold_count,gold_answers,baseline_preds,concept_preds,
recall@10_exact_baseline,recall@10_exact_concept,cluster_intra_sim,cluster_inter_sim,cluster_coherence
"""

import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional but helpful. If not available in your venv, comment these two imports
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# Utilities
# ---------------------------

WORD_BOUNDARY = re.compile(r"[ \t\n\r\f\v.,;:!?)\]\}\"\u3002\uFF0C\uFF01\uFF1F]")
VALID_WORD = re.compile(r"^[a-z][a-z\-']{0,30}$")
STOP_WORDS = {"the", "a", "an", ""}

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9\s\-']", "", s.lower()).strip()

def is_valid_word(w: str) -> bool:
    if not w: return False
    if w in STOP_WORDS: return False
    return bool(VALID_WORD.match(w))

def safe_device_map(device_arg: str):
    if device_arg == "auto":
        return "auto"
    return None

def to_device_ids(tokenizer, text: str, device):
    return tokenizer(text, return_tensors="pt").to(device)["input_ids"]

def decode_suffix_to_first_word(tok, ids: List[int]) -> str:
    """Decode a sequence of *generated* ids to the first complete word substring."""
    if not ids:
        return ""
    s = tok.decode(ids, skip_special_tokens=True)
    # Stop at first boundary or space
    if " " in s:
        s = s.split(" ", 1)[0]
    m = WORD_BOUNDARY.search(s)
    if m:
        s = s[:m.start()]
    w = re.sub(r"[^A-Za-z\-']", "", s).lower()
    return w

def complete_word_after_seed(
    model, tok, seeded_ids: torch.Tensor, max_steps: int = 6
) -> str:
    """
    Continue generation from seeded_ids until we complete one word (or hit max_steps).
    Return the *word* (normalized/cleaned).
    seeded_ids: shape [1, T]
    """
    device = seeded_ids.device
    gen_ids = []
    for _ in range(max_steps):
        with torch.no_grad():
            logits = model(seeded_ids).logits[:, -1, :]
        next_id = torch.argmax(logits, dim=-1)  # greedy continuation
        seeded_ids = torch.cat([seeded_ids, next_id.view(1, 1)], dim=-1)
        gen_ids.append(next_id.item())

        w = decode_suffix_to_first_word(tok, gen_ids)
        # If we already formed a non-empty word that looks valid or a boundary occurred, stop.
        if w and (is_valid_word(w) or WORD_BOUNDARY.search(tok.decode(gen_ids, skip_special_tokens=True))):
            return w
    # fallback: whatever we got
    return decode_suffix_to_first_word(tok, gen_ids)

def topk_words_from_logits(
    model, tok, input_ids: torch.Tensor, logits: torch.Tensor, k: int = 100, max_steps: int = 6
) -> List[Tuple[str, float]]:
    """
    Pull top-k *next tokens*, seed each, then continue until the first word boundary.
    Return a list of (word, prob). Deduplicate words, keep first (highest-prob) occurrence.
    """
    probs = torch.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, k, sorted=True)
    words: List[Tuple[str, float]] = []
    seen = set()
    for p, tid in zip(top_p[0], top_i[0]):
        # Seed with this next-token id
        seeded = torch.cat([input_ids, tid.view(1, 1)], dim=-1)
        w = complete_word_after_seed(model, tok, seeded, max_steps=max_steps)
        w = normalize(w)
        if not is_valid_word(w):
            continue
        if w in seen:
            continue
        seen.add(w)
        words.append((w, float(p.item())))
        if len(words) >= k:
            break
    return words

def embed_word_avg(model, tok, word: str) -> np.ndarray:
    """
    Average the input embeddings of the sub-tokens that compose the word.
    Prefix with a space to encourage BPE to start a new token.
    """
    if not word:
        # tiny random to avoid zeros
        return np.random.normal(scale=1e-6, size=(model.get_input_embeddings().embedding_dim,))
    enc = tok.encode(" " + word, add_special_tokens=False)
    if len(enc) == 0:
        enc = tok.encode(word, add_special_tokens=False)
    W = model.get_input_embeddings().weight
    idx = torch.tensor(enc, device=W.device)
    vec = W[idx].float().mean(dim=0).detach().cpu().numpy()
    return vec

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def cluster_words(
    word_vecs: np.ndarray,
    distance_threshold: float = 0.45
) -> np.ndarray:
    """
    Agglomerative clustering on cosine distance (1 - cosine).
    Returns labels array (shape [N]); -1 is not used here (all points get a label).
    """
    if word_vecs.shape[0] <= 1:
        return np.zeros((word_vecs.shape[0],), dtype=int)

    # Dim. reduction for stability when N < dim
    n = word_vecs.shape[0]
    rdim = min(min(50, word_vecs.shape[1]), max(2, n - 1))
    try:
        X = PCA(n_components=rdim, random_state=0).fit_transform(word_vecs)
    except Exception:
        X = word_vecs  # fallback

    # Precompute cosine distance matrix, normalized to [0,1]
    sim = cosine_similarity(X)
    D = 1.0 - sim
    dmin, dmax = D.min(), D.max()
    if dmax > dmin:
        D = (D - dmin) / (dmax - dmin)

    # sklearn >= 1.2 uses `metric`; older uses `affinity`
    try:
        model = AgglomerativeClustering(
            metric="precomputed",
            linkage="complete",
            distance_threshold=distance_threshold,
            n_clusters=None
        )
    except TypeError:
        model = AgglomerativeClustering(
            affinity="precomputed",
            linkage="complete",
            distance_threshold=distance_threshold,
            n_clusters=None
        )

    labels = model.fit_predict(D)
    return labels

def rank_clusters_and_pick_centroids(
    words: List[str],
    probs: List[float],
    vecs: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.7
) -> List[str]:
    """
    Score each cluster with alpha*max_prob + (1-alpha)*(cluster_size / max_size).
    For each cluster, pick the word closest to the cluster centroid (cosine).
    Return centroid words sorted by cluster score (desc).
    """
    words = np.array(words)
    probs = np.array(probs)
    unique_labels = np.unique(labels)
    # cluster sizes
    sizes = {lab: int(np.sum(labels == lab)) for lab in unique_labels}
    max_size = max(sizes.values()) if sizes else 1

    centroid_words = []
    cluster_scores = []
    for lab in unique_labels:
        idx = np.where(labels == lab)[0]
        if len(idx) == 0:
            continue
        cluster_vecs = vecs[idx]
        centroid = np.mean(cluster_vecs, axis=0)
        # pick closest to centroid
        sims = np.array([cosine(v, centroid) for v in cluster_vecs])
        best_idx_local = idx[int(np.argmax(sims))]
        best_word = words[best_idx_local]

        # cluster score
        max_prob = float(np.max(probs[idx]))
        size_term = sizes[lab] / max_size
        score = alpha * max_prob + (1.0 - alpha) * size_term

        centroid_words.append(best_word)
        cluster_scores.append(score)

    order = np.argsort(cluster_scores)[::-1]
    ranked = [centroid_words[i] for i in order]
    # Dedup in order (should already be unique)
    seen, out = set(), []
    for w in ranked:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out

def compute_coherence(vecs: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns (intra, inter, coherence = intra - inter), averaged over pairs.
    If single cluster or <2 points, returns (0,0,0).
    """
    n = vecs.shape[0]
    if n < 2:
        return (0.0, 0.0, 0.0)
    sims = cosine_similarity(vecs)

    # intra-cluster avg
    intra_vals = []
    inter_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                intra_vals.append(sims[i, j])
            else:
                inter_vals.append(sims[i, j])
    intra = float(np.mean(intra_vals)) if intra_vals else 0.0
    inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    return (intra, inter, intra - inter)


# ---------------------------
# ProtoQA loading
# ---------------------------

def load_protoqa(protoqa_dir: str, split: str) -> List[Dict[str, Any]]:
    """
    Load ProtoQA *crowdsourced* split jsonl and return a list of items:
    {id, question, gold:list[str]}
    """
    base = Path(protoqa_dir)
    path = base / split / f"{split}.crowdsourced.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"ProtoQA file not found: {path}")

    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            ex = json.loads(line)
            qid = ex["metadata"]["id"]
            qtext = ex["question"]["original"].strip()
            # gold answers from answers.raw keys (exclude counts and weird fields)
            raw = ex.get("answers", {}).get("raw", {})
            # Some datasets include numeric counts mapped to strings; keys are the answers
            gold = []
            for k in raw.keys():
                s = normalize(str(k))
                if not s: continue
                gold.append(s)
            # dedup
            gold = sorted(set(gold))
            if not gold:
                continue
            items.append({"id": qid, "question": qtext, "gold": gold})
    if not items:
        raise RuntimeError(f"No valid items parsed from {path}")
    return items


# ---------------------------
# Decoding helpers (one-word intent)
# ---------------------------

def get_next_logits(model, tok, prefix_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        out = model(prefix_ids)
        return out.logits[:, -1, :]

def get_prefix_for_question(q: str) -> str:
    # Minimal instruction, but decoding enforces one-word via boundary stopping.
    return f"{q.strip()} Answer: "

def baseline_top10_words(model, tok, prefix_ids, k=100, max_steps=6) -> List[str]:
    logits = get_next_logits(model, tok, prefix_ids)
    cands = topk_words_from_logits(model, tok, prefix_ids, logits, k=k, max_steps=max_steps)
    # sort by prob desc, keep top-10 unique words
    cands.sort(key=lambda x: x[1], reverse=True)
    words = []
    seen = set()
    for w, p in cands:
        if w in seen: continue
        seen.add(w)
        words.append(w)
        if len(words) >= 10:
            break
    return words

def concept_top10_words(model, tok, prefix_ids, k=100, max_steps=6, alpha=0.7, cluster_dist_thresh=0.45) -> Tuple[List[str], Tuple[float,float,float]]:
    logits = get_next_logits(model, tok, prefix_ids)
    cands = topk_words_from_logits(model, tok, prefix_ids, logits, k=k, max_steps=max_steps)
    if not cands:
        return [], (0.0, 0.0, 0.0)
    words, probs = zip(*cands)
    # embed words
    vecs = np.vstack([embed_word_avg(model, tok, w) for w in words])
    # cluster
    labels = cluster_words(vecs, distance_threshold=cluster_dist_thresh)
    # rank clusters and pick centroids
    ranked = rank_clusters_and_pick_centroids(list(words), list(probs), vecs, labels, alpha=alpha)
    top10 = ranked[:10]
    # coherence
    intra, inter, coh = compute_coherence(vecs, labels)
    return top10, (intra, inter, coh)

def recall_at_10_exact(preds: List[str], gold: List[str]) -> float:
    if not preds: return 0.0
    P = set(normalize(p) for p in preds if p)
    G = set(normalize(g) for g in gold if g)
    if not G: return 0.0
    hits = len(P.intersection(G))
    # standard ProtoQA "recall at 10" (fraction of gold covered by top-10 preds)
    # Some papers define as hits/|gold|; keep that here.
    return hits / max(1, len(G))


# ---------------------------
# Main runner
# ---------------------------

def run(
    model_name: str,
    protoqa_dir: str,
    split: str,
    top_k_tokens: int,
    out_path: str,
    device: str = "auto",
    alpha: float = 0.7,
    cluster_dist_thresh: float = 0.45,
    max_steps_word: int = 6
):
    items = load_protoqa(protoqa_dir, split)
    print(f"Loaded {len(items)} ProtoQA items from {protoqa_dir} [{split}]")
    print(f"Sample: {items[0]['question']} | gold size: {len(items[0]['gold'])}")

    print(f"Loading HF model: {model_name} on {device}")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=safe_device_map(device)
    )
    device0 = model.device
    model.eval()

    rows = []
    r10_base_list = []
    r10_conc_list = []
    coh_list = []

    for idx, ex in enumerate(items, 1):
        qid = ex["id"]
        q = ex["question"]
        gold = ex["gold"]

        prefix = get_prefix_for_question(q)
        prefix_ids = to_device_ids(tok, prefix, device0)

        # Baseline: top-10 greedy-completed words by top-k next-token probs
        base_words = baseline_top10_words(model, tok, prefix_ids, k=top_k_tokens, max_steps=max_steps_word)

        # Concept: cluster words
        conc_words, (intra, inter, coh) = concept_top10_words(
            model, tok, prefix_ids,
            k=top_k_tokens, max_steps=max_steps_word,
            alpha=alpha, cluster_dist_thresh=cluster_dist_thresh
        )

        r10_base = recall_at_10_exact(base_words, gold)
        r10_conc = recall_at_10_exact(conc_words, gold)

        r10_base_list.append(r10_base)
        r10_conc_list.append(r10_conc)
        coh_list.append(coh)

        rows.append({
            "id": qid,
            "question": q,
            "prefix": prefix,
            "gold_count": len(gold),
            "gold_answers": "|".join(gold[:200]),  # cap to keep row manageable
            "baseline_preds": "|".join(base_words),
            "concept_preds": "|".join(conc_words),
            "recall@10_exact_baseline": r10_base,
            "recall@10_exact_concept": r10_conc,
            "cluster_intra_sim": intra,
            "cluster_inter_sim": inter,
            "cluster_coherence": coh
        })

        if idx % 25 == 0 or idx == len(items):
            mean_b = float(np.mean(r10_base_list)) if r10_base_list else 0.0
            mean_c = float(np.mean(r10_conc_list)) if r10_conc_list else 0.0
            mean_coh = float(np.mean(coh_list)) if coh_list else 0.0
            print(f"[{idx}/{len(items)}] mean r@10 (base,concept): "
                  f"{mean_b:.3f} {mean_c:.3f} | mean coh: {mean_coh:.3f}")

    # Write CSV
    out = Path(out_path)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nSaved per-item metrics -> {out}")

    print("\n=== Summary ===")
    print(f"Mean Recall@10 EXACT (baseline): {np.mean(r10_base_list)}")
    print(f"Mean Recall@10 EXACT (concept) : {np.mean(r10_conc_list)}")
    print(f"Mean cluster coherence         : {np.mean(coh_list)}")


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model name (causal LM).")
    ap.add_argument("--protoqa_dir", type=str, required=True, help="Path to ProtoQA data root (containing split folders).")
    ap.add_argument("--split", type=str, default="dev", help="Split folder name (dev/test/train).")
    ap.add_argument("--top_k_tokens", type=int, default=100, help="How many top next tokens to expand into word candidates.")
    ap.add_argument("--out", type=str, default="protoqa_decoder_results.csv", help="CSV output path.")
    ap.add_argument("--device", type=str, default="auto", help="HF accelerate device map (e.g., auto).")
    ap.add_argument("--alpha", type=float, default=0.7, help="Cluster score blend: alpha*max_prob + (1-alpha)*size_term.")
    ap.add_argument("--cluster_dist_thresh", type=float, default=0.45, help="Agglomerative distance_threshold (0..1 after normalization).")
    ap.add_argument("--max_steps_word", type=int, default=6, help="Max continuation steps to complete one word.")
    args = ap.parse_args()

    run(
        model_name=args.model,
        protoqa_dir=args.protoqa_dir,
        split=args.split,
        top_k_tokens=args.top_k_tokens,
        out_path=args.out,
        device=args.device,
        alpha=args.alpha,
        cluster_dist_thresh=args.cluster_dist_thresh,
        max_steps_word=args.max_steps_word
    )


if __name__ == "__main__":
    main()