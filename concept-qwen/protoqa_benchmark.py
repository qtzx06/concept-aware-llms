# concept-qwen/protoqa_benchmark.py
# Cross-platform ProtoQA benchmark for decoder LLMs (Qwen/Qwen3).
# Top-k clustering, recall@k coverage, and cluster coherence.
#
# Backends:
#   --backend hf      : local HuggingFace model (CPU/MPS/CUDA)
#   --backend runpod  : serverless endpoint (OpenAI/vLLM-like). Set:
#                       export RUNPOD_ENDPOINT="https://api.runpod.ai/v2/<ENDPOINT_ID>"
#                       export RUNPOD_API_KEY="rp_api_..."
#
# Notes:
# - Exact Recall@10 is strict string match over normalized forms (like ProtoQA dev golds).
# - Semantic Recall@10 gives credit if cosine(token, any gold) >= --sem_thresh (default 0.62).
# - Cluster coherence uses SentenceTransformers MiniLM on CPU for portability.

import argparse
import json
import os
from pathlib import Path
import re
from collections import defaultdict
import time
import random

import numpy as np
import pandas as pd
import torch

# Sklearn (handle older/newer AgglomerativeClustering arg name)
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from sentence_transformers import SentenceTransformer, util

# Optional NLTK light lemmatization
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# -------------------- NLTK setup --------------------
def _ensure_nltk():
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')

_ensure_nltk()
LEMM = WordNetLemmatizer()


# -------------------- Text utils --------------------
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"[\"“”‘’]", "", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[.,;:!?]+$", "", s)
    toks = [LEMM.lemmatize(t) for t in s.split()]
    return " ".join(toks)

def first_word_only(text: str) -> str:
    """
    Extract exactly one word from a decoder completion (allow letters, hyphen).
    """
    if not text:
        return ""
    # Take the first non-empty line after trimming the prompt
    cut = text.strip().splitlines()[0].strip()
    # Remove leading labels like "Answer:" or "- "
    cut = re.sub(r"^(answer:|a:|-)\s*", "", cut, flags=re.I)
    # Keep only the first token-ish word
    m = re.match(r"([a-zA-Z][a-zA-Z\-']*)", cut)
    return m.group(1) if m else ""

def to_first_person_prompt(question: str) -> str:
    """
    Heuristically convert ProtoQA question into a decoder-friendly, short-answer prompt.
    We push the model to return ONE common noun (1 word).
    Also convert 2nd person to 1st person where easy.
    """
    q = question.strip()
    # quick pronoun shifts
    repl = [
        (r"\byou would\b", "I would"),
        (r"\byou might\b", "I might"),
        (r"\byou are\b", "I am"),
        (r"\byour\b", "my"),
        (r"\byou\b", "I"),
    ]
    for pat, rep in repl:
        q = re.sub(pat, rep, q, flags=re.I)

    # Encourage single-word noun
    return (
        f"{q}\n"
        "Answer with exactly ONE common noun (one word). No punctuation, no extra words."
    )


# -------------------- ProtoQA data --------------------
def find_protoqa_jsonl(protoqa_dir: Path, split: str) -> Path:
    cands = list(protoqa_dir.rglob(f"*{split}*.jsonl"))
    if not cands:
        raise FileNotFoundError(
            f"Could not find a *{split}*.jsonl under {protoqa_dir}. Clone https://github.com/iesl/protoqa-data and pass --protoqa_dir."
        )
    for c in cands:
        if c.name == f"{split}.jsonl":
            return c
    return cands[0]

def load_protoqa(protoqa_dir: str, split: str):
    """
    Load ProtoQA split into [{'question': str, 'gold': set[str]}].
    For dev/train, golds exist; for test, golds may be empty.
    """
    path = find_protoqa_jsonl(Path(protoqa_dir), split)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            q = ex.get("question") or ex.get("prompt") or ""
            ans_raw = ex.get("answers") or ex.get("answers_raw") or {}
            if isinstance(ans_raw, dict):
                golds = list(ans_raw.keys())
            elif isinstance(ans_raw, list):
                golds = ans_raw
            else:
                golds = []
            gold_norm = {normalize_text(a) for a in golds if a and len(a) < 64}
            gold_norm = {a for a in gold_norm if a and a not in {"", "none"}}
            rows.append({"question": q, "gold": gold_norm})
    return rows


# -------------------- Device helpers --------------------
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# -------------------- RunPod client --------------------
class RunPodClient:
    """
    Minimal RunPod client for OpenAI/vLLM-like endpoints.
    Supports:
      - /run (async queue) + /status polling
      - /runsync (synchronous)
    We expect the endpoint to understand keys: model (optional), prompt, max_tokens, temperature, top_p, logprobs, n.
    """
    def __init__(self, endpoint: str | None = None, api_key: str | None = None):
        import requests  # local import for portability
        self.requests = requests
        self.endpoint = endpoint or os.getenv("RUNPOD_ENDPOINT")
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY")
        if not (self.endpoint and self.api_key):
            raise RuntimeError("Set RUNPOD_ENDPOINT and RUNPOD_API_KEY in your environment.")

        # Derive base and mode
        self.base = self.endpoint.rstrip("/")
        if self.base.endswith("/run") or self.base.endswith("/runsync"):
            self.base = self.base.rsplit("/", 1)[0]
        self.mode = "runsync" if os.getenv("RUNPOD_RUNSYNC", "0") == "1" else "run"

        self.hdrs = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post(self, path: str, json_payload: dict):
        url = f"{self.base}/{path.lstrip('/')}"
        r = self.requests.post(url, headers=self.hdrs, json=json_payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str):
        url = f"{self.base}/{path.lstrip('/')}"
        r = self.requests.get(url, headers=self.hdrs, timeout=60)
        r.raise_for_status()
        return r.json()

    def _extract_texts(self, data) -> list[str]:
        """
        Try to pull texts from a few known shapes:
          - OpenAI-like: data["output"]["choices"][i]["text"]
          - Simple: data["output"]["text"] or ["texts"]
        """
        out = []
        # OpenAI-ish
        try:
            choices = data["output"]["choices"]
            for ch in choices:
                t = ch.get("text") or ch.get("message", {}).get("content", "")
                if t:
                    out.append(t)
        except Exception:
            pass
        # Simpler shapes
        if not out and isinstance(data.get("output"), dict):
            if "text" in data["output"]:
                out = [data["output"]["text"]]
            elif "texts" in data["output"] and isinstance(data["output"]["texts"], list):
                out = data["output"]["texts"]

        # Some services return raw string in "output"
        if not out and isinstance(data.get("output"), str):
            out = [data["output"]]

        return [str(t) for t in out if t is not None]

    def _extract_topk_logprobs(self, data) -> dict[str, float] | None:
        """
        Look for OpenAI/vLLM logprobs:
          data["output"]["choices"][0]["logprobs"]["top_logprobs"][0] ~ {token: logprob}
        """
        try:
            top = data["output"]["choices"][0]["logprobs"]["top_logprobs"][0]
            # Convert logprobs -> probs
            return {str(tok).strip(): float(np.exp(float(lp))) for tok, lp in top.items()}
        except Exception:
            return None

    def generate_texts(self, prompt: str, n: int, max_tokens: int, temperature: float, top_p: float) -> list[str]:
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": n
            }
        }
        if self.mode == "runsync":
            data = self._post("runsync", payload)
            return self._extract_texts(data)
        # async queue
        job = self._post("run", payload)
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError(f"Unexpected RunPod response: {job}")
        # poll
        for _ in range(120):  # up to ~2 minutes
            time.sleep(1.0)
            status = self._get(f"status/{job_id}")
            s = status.get("status")
            if s in {"FAILED", "CANCELLED", "CANCELED"}:
                raise RuntimeError(f"RunPod job failed: {status}")
            if s == "COMPLETED":
                return self._extract_texts(status)
        raise RuntimeError("RunPod job timed out.")

    def topk_logprobs(self, prompt: str, k: int = 100) -> dict[str, float] | None:
        """
        Ask endpoint for top-k logprobs for next token. If not supported, returns None.
        """
        payload = {
            "input": {
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0.0,
                "logprobs": k
            }
        }
        if self.mode == "runsync":
            data = self._post("runsync", payload)
            lp = self._extract_topk_logprobs(data)
            return lp
        job = self._post("run", payload)
        job_id = job.get("id")
        if not job_id:
            return None
        for _ in range(120):
            time.sleep(1.0)
            status = self._get(f"status/{job_id}")
            if status.get("status") == "COMPLETED":
                return self._extract_topk_logprobs(status)
        return None


# -------------------- HF backend (local) --------------------
def hf_load(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device()
    print(f"Loading model: {model_name} on {device}")
    # Torch dtype: use float32 on CPU/MPS for safety
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if device == "cuda":
        model = model.cuda()
    elif device == "mps":
        model = model.to("mps")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, device

@torch.no_grad()
def hf_next_token_topk(model, tokenizer, prompt: str, k: int):
    inputs = tokenizer(prompt, return_tensors="pt")
    if next(model.parameters()).device.type in {"cuda", "mps"}:
        inputs = {k_: v.to(next(model.parameters()).device) for k_, v in inputs.items()}
    out = model(**inputs)
    logits = out.logits[:, -1, :]  # [1, V]
    probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
    topk_probs, topk_ids = torch.topk(probs, k)
    # return lists
    toks = [tokenizer.decode([int(i)]).strip() for i in topk_ids]
    probs = [float(p) for p in topk_probs]
    return toks, probs

@torch.no_grad()
def hf_sample_one_word(model, tokenizer, prompt: str, max_new_tokens: int = 2, temperature: float = 0.7, top_p: float = 0.92) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    if next(model.parameters()).device.type in {"cuda", "mps"}:
        inputs = {k_: v.to(next(model.parameters()).device) for k_, v in inputs.items()}
    gen = model.generate(
        **inputs,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    full = tokenizer.decode(gen[0], skip_special_tokens=True)
    ans = full[len(prompt):]
    return first_word_only(ans)

def hf_output_embeddings(model) -> torch.Tensor:
    W = model.get_output_embeddings().weight.detach().cpu()
    # Make sure float32 before numpy
    return W.float()


# -------------------- Concept clustering (two paths) --------------------
def cluster_topk_tokens_with_output_emb(top_tokens: list[str],
                                        top_probs: list[float],
                                        tokenizer,
                                        W_out: torch.Tensor,
                                        distance_threshold: float = 0.45):
    """
    Cluster candidate tokens using the model's output embedding matrix (HF backend).
    """
    # Map token -> id carefully (some decoders need encode/decode properly)
    token_ids = []
    for tok in top_tokens:
        # Try to get single-token id; if multi-token, take the first piece
        ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        tid = ids[0] if ids else None
        if tid is None or tid >= W_out.shape[0]:
            token_ids.append(None)
        else:
            token_ids.append(tid)

    # Filter tokens we couldn't map
    valid = [(t, p, i) for t, p, i in zip(top_tokens, top_probs, token_ids) if i is not None]
    if not valid:
        return [(0, [{"token_text": top_tokens[0], "prob": top_probs[0]}])], {0: top_probs[0]}

    toks, probs, ids = zip(*valid)
    emb = W_out[list(ids)]  # [k, d] float32
    X = emb.numpy().astype(np.float32)

    X_red = X
    if X.shape[1] > 64 and X.shape[0] > 2:
        pca = PCA(n_components=64)
        X_red = pca.fit_transform(X)

    # Handle sklearn API changes for metric/affinity
    try:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="average", metric="cosine"
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="average", affinity="cosine"
        )
    labels = clustering.fit_predict(X_red)

    clusters = defaultdict(list)
    scores = defaultdict(float)
    for t, p, lab in zip(toks, probs, labels):
        clusters[int(lab)].append({"token_text": t, "prob": float(p)})
        scores[int(lab)] += float(p)

    ranked = sorted(clusters.items(), key=lambda kv: scores[kv[0]], reverse=True)
    return ranked, scores

def cluster_topk_tokens_with_text_emb(top_tokens: list[str],
                                      top_probs: list[float],
                                      embedder: SentenceTransformer,
                                      distance_threshold: float = 0.45):
    """
    Cluster tokens using a generic text embedder (RunPod fallback when output embeddings are unavailable).
    """
    if not top_tokens:
        return [], {}

    embs = embedder.encode(top_tokens, convert_to_numpy=True, normalize_embeddings=True)
    X = embs.astype(np.float32)

    X_red = X
    if X.shape[1] > 64 and X.shape[0] > 2:
        pca = PCA(n_components=64)
        X_red = pca.fit_transform(X)

    try:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="average", metric="cosine"
        )
    except TypeError:
        clustering = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold, linkage="average", affinity="cosine"
        )
    labels = clustering.fit_predict(X_red)

    clusters = defaultdict(list)
    scores = defaultdict(float)
    for t, p, lab in zip(top_tokens, top_probs, labels):
        clusters[int(lab)].append({"token_text": t, "prob": float(p)})
        scores[int(lab)] += float(p)

    ranked = sorted(clusters.items(), key=lambda kv: scores[kv[0]], reverse=True)
    return ranked, scores


# -------------------- Predictions --------------------
def baseline_predictions_hf(model, tokenizer, prompt: str, n: int, max_new_tokens: int = 2) -> list[str]:
    outs = []
    for _ in range(n):
        w = hf_sample_one_word(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        if w:
            outs.append(w)
    # unique-preserving
    seen, uniq = set(), []
    for a in outs:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

def baseline_predictions_runpod(rp: RunPodClient, prompt: str, n: int, max_new_tokens: int = 2) -> list[str]:
    texts = rp.generate_texts(prompt, n=n, max_tokens=max_new_tokens, temperature=0.7, top_p=0.92)
    words = [first_word_only(t) for t in texts]
    seen, uniq = set(), []
    for a in words:
        if a and a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

def concept_predictions_hf(model, tokenizer, prompt: str, top_k_tokens: int, distance_threshold: float, max_per_cluster: int) -> list[str]:
    toks, probs = hf_next_token_topk(model, tokenizer, prompt, k=top_k_tokens)
    # Clean tokens: strip leading whitespace/punct
    toks = [re.sub(r"^[\s#.,:;!?]+", "", t) for t in toks]
    toks, probs = zip(*[(t, p) for t, p in zip(toks, probs) if t])
    W = hf_output_embeddings(model)
    ranked, _ = cluster_topk_tokens_with_output_emb(list(toks), list(probs), tokenizer, W, distance_threshold)
    answers = []
    for _, items in ranked:
        items_sorted = sorted(items, key=lambda x: x["prob"], reverse=True)[:max_per_cluster]
        for it in items_sorted:
            tok = it["token_text"]
            tok = re.sub(r"^[\s#.,:;!?]+", "", tok)
            if tok:
                answers.append(tok)
    seen, uniq = set(), []
    for a in answers:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

def concept_predictions_runpod(rp: RunPodClient, prompt: str, top_k_tokens: int, distance_threshold: float, max_per_cluster: int, embedder: SentenceTransformer) -> list[str]:
    top = rp.topk_logprobs(prompt, k=top_k_tokens)
    if not top:
        # Fallback: sample many and take first word frequency as proxy
        texts = rp.generate_texts(prompt, n=min(20, top_k_tokens), max_tokens=1, temperature=0.0, top_p=1.0)
        words = [first_word_only(t) for t in texts]
        cnt = defaultdict(float)
        for w in words:
            if w:
                cnt[w] += 1.0
        toks, probs = zip(*sorted(cnt.items(), key=lambda kv: kv[1], reverse=True))
        toks, probs = list(toks), list(probs)
    else:
        toks, probs = zip(*sorted(top.items(), key=lambda kv: kv[1], reverse=True))
        toks, probs = list(toks), list(probs)

    toks = [re.sub(r"^[\s#.,:;!?]+", "", t) for t in toks]
    toks, probs = zip(*[(t, p) for t, p in zip(toks, probs) if t])

    ranked, _ = cluster_topk_tokens_with_text_emb(list(toks), list(probs), embedder, distance_threshold)
    answers = []
    for _, items in ranked:
        items_sorted = sorted(items, key=lambda x: x["prob"], reverse=True)[:max_per_cluster]
        for it in items_sorted:
            tok = it["token_text"]
            tok = re.sub(r"^[\s#.,:;!?]+", "", tok)
            if tok:
                answers.append(tok)
    # dedupe
    seen, uniq = set(), []
    for a in answers:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


# -------------------- Metrics --------------------
def recall_at_k_exact(preds: list[str], gold_set: set[str], k: int) -> float:
    if not preds or not gold_set or k <= 0:
        return 0.0
    preds_k = [normalize_text(p) for p in preds[:k] if p]
    hits = sum(1 for p in preds_k if p in gold_set)
    denom = min(k, len(gold_set))
    return hits / denom if denom > 0 else 0.0

def recall_at_k_semantic(preds: list[str], gold_set: set[str], k: int, embedder: SentenceTransformer, thresh: float = 0.62) -> float:
    """
    Soft coverage: credit if max cosine(pred, any gold) >= thresh.
    """
    preds_k = [normalize_text(p) for p in preds[:k] if p]
    if not preds_k or not gold_set:
        return 0.0
    all_texts = preds_k + list(gold_set)
    embs = embedder.encode(all_texts, convert_to_tensor=True, normalize_embeddings=True, device="cpu")
    n_pred = len(preds_k)
    pred_embs = embs[:n_pred]
    gold_embs = embs[n_pred:]
    hits = 0
    for i in range(n_pred):
        sims = util.cos_sim(pred_embs[i], gold_embs)[0].cpu().numpy()
        if np.max(sims) >= thresh:
            hits += 1
    denom = min(k, len(gold_set))
    return hits / denom if denom > 0 else 0.0

def cluster_coherence(tokens_by_cluster: list[list[str]], embedder: SentenceTransformer) -> tuple[float, float, float]:
    """
    Average intra-cluster cosine minus inter-cluster cosine over token *strings* (MiniLM).
    """
    all_tokens = [t for cl in tokens_by_cluster for t in cl]
    if len(all_tokens) < 2:
        return 0.0, 0.0, 0.0
    embs = embedder.encode(all_tokens, convert_to_tensor=True, normalize_embeddings=True, device="cpu")
    intra, inter, n_intra, n_inter = 0.0, 0.0, 0, 0
    # index bookkeeping
    idx = 0
    starts = []
    for cl in tokens_by_cluster:
        starts.append(idx)
        idx += len(cl)
    for c_id, cl in enumerate(tokens_by_cluster):
        for i in range(len(cl)):
            ii = starts[c_id] + i
            for d_id, cl2 in enumerate(tokens_by_cluster):
                for j in range(len(cl2)):
                    jj = starts[d_id] + j
                    if ii >= jj:
                        continue
                    sim = float(util.cos_sim(embs[ii], embs[jj]).item())
                    if c_id == d_id:
                        intra += sim; n_intra += 1
                    else:
                        inter += sim; n_inter += 1
    intra_m = intra / n_intra if n_intra else 0.0
    inter_m = inter / n_inter if n_inter else 0.0
    return intra_m, inter_m, (intra_m - inter_m)


# -------------------- Main loop --------------------
def run_protoqa(
    backend: str,
    model_name: str,
    protoqa_dir: str,
    split: str,
    out_csv: str,
    top_n: int,
    top_k_tokens: int,
    distance_threshold: float,
    sem_thresh: float,
):
    print(f"Backend: {backend}")
    print(f"Loading ProtoQA from {protoqa_dir} [{split}] …")
    data = load_protoqa(protoqa_dir, split)
    print(f"Loaded {len(data)} items.")

    # Sentence-Transformer on CPU for portability
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # Prepare backends
    model = tokenizer = device = None
    rp = None

    if backend == "hf":
        model, tokenizer, device = hf_load(model_name)
    elif backend == "runpod":
        print("Using RunPod remote endpoint (set RUNPOD_ENDPOINT and RUNPOD_API_KEY).")
        rp = RunPodClient()
    else:
        raise ValueError("--backend must be 'hf' or 'runpod'.")

    rows = []
    for idx, ex in enumerate(data, 1):
        q = ex["question"]; gold = ex["gold"]
        prompt = to_first_person_prompt(q)

        # --- Baseline one-word predictions ---
        if backend == "hf":
            base_preds = baseline_predictions_hf(model, tokenizer, prompt, n=top_n)
            # Concept preds via output-embedding clustering
            concept_preds = concept_predictions_hf(
                model, tokenizer, prompt,
                top_k_tokens=top_k_tokens,
                distance_threshold=distance_threshold,
                max_per_cluster=max(1, top_n // 3),
            )
            # For coherence, reconstruct clusters from HF path
            toks, probs = hf_next_token_topk(model, tokenizer, prompt, k=top_k_tokens)
            toks = [re.sub(r"^[\s#.,:;!?]+", "", t) for t in toks if t.strip()]
            W = hf_output_embeddings(model)
            ranked, _ = cluster_topk_tokens_with_output_emb(toks, probs, tokenizer, W, distance_threshold)
        else:
            base_preds = baseline_predictions_runpod(rp, prompt, n=top_n)
            concept_preds = concept_predictions_runpod(
                rp, prompt,
                top_k_tokens=top_k_tokens,
                distance_threshold=distance_threshold,
                max_per_cluster=max(1, top_n // 3),
                embedder=embedder,
            )
            # For coherence, rebuild cluster token buckets from runpod path
            top = rp.topk_logprobs(prompt, k=top_k_tokens)
            if not top:
                # fallback: most frequent single words from n=top_k_tokens samples
                texts = rp.generate_texts(prompt, n=min(20, top_k_tokens), max_tokens=1, temperature=0.0, top_p=1.0)
                words = [first_word_only(t) for t in texts]
                cnt = defaultdict(float)
                for w in words:
                    if w:
                        cnt[w] += 1.0
                toks, probs = zip(*sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)) if cnt else ([], [])
            else:
                toks, probs = zip(*sorted(top.items(), key=lambda kv: kv[1], reverse=True))
            toks = [re.sub(r"^[\s#.,:;!?]+", "", t) for t in toks if str(t).strip()]
            ranked, _ = cluster_topk_tokens_with_text_emb(toks, list(probs) if toks else [], embedder, distance_threshold)

        # Metrics
        r10_base_exact = recall_at_k_exact(base_preds, gold, k=10)
        r10_conc_exact = recall_at_k_exact(concept_preds, gold, k=10)
        r10_base_sem = recall_at_k_semantic(base_preds, gold, k=10, embedder=embedder, thresh=sem_thresh)
        r10_conc_sem = recall_at_k_semantic(concept_preds, gold, k=10, embedder=embedder, thresh=sem_thresh)

        # Cluster coherence (use top few clusters’ token strings)
        cluster_tokens_lists = []
        for _, items in ranked[: min(5, len(ranked))]:
            toks_in = [re.sub(r"^[\s#.,:;!?]+", "", it["token_text"]) for it in items]
            toks_in = [t for t in toks_in if t]
            if toks_in:
                cluster_tokens_lists.append(toks_in)
        intra, inter, coh = cluster_coherence(cluster_tokens_lists, embedder)

        rows.append({
            "id": idx,
            "question": q,
            "prompt": prompt,
            "gold_count": len(gold),
            "gold_answers": "|".join(sorted(gold)),

            "baseline_preds": "|".join(base_preds[:10]),
            "concept_preds": "|".join(concept_preds[:10]),

            "recall@10_exact_baseline": r10_base_exact,
            "recall@10_exact_concept": r10_conc_exact,
            "recall@10_semantic_baseline": r10_base_sem,
            "recall@10_semantic_concept": r10_conc_sem,

            "cluster_intra_sim": intra,
            "cluster_inter_sim": inter,
            "cluster_coherence": coh,
        })

        if idx % 25 == 0:
            print(f"[{idx}/{len(data)}] Mean r@10_exact (base,concept): "
                  f"{np.mean([r['recall@10_exact_baseline'] for r in rows]):.3f} "
                  f"{np.mean([r['recall@10_exact_concept'] for r in rows]):.3f}   "
                  f"Mean r@10_sem (base,concept): "
                  f"{np.mean([r['recall@10_semantic_baseline'] for r in rows]):.3f} "
                  f"{np.mean([r['recall@10_semantic_concept'] for r in rows]):.3f}")

    df = pd.DataFrame(rows)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nSaved per-item metrics -> {out}")

    print("\n=== Summary ===")
    print("Mean Recall@10 EXACT (baseline) :", float(df['recall@10_exact_baseline'].mean()))
    print("Mean Recall@10 EXACT (concept)  :", float(df['recall@10_exact_concept'].mean()))
    if 'recall@10_semantic_baseline' in df.columns:
        print("Mean Recall@10 SEMANTIC (baseline) :", float(df['recall@10_semantic_baseline'].mean()))
        print("Mean Recall@10 SEMANTIC (concept)  :", float(df['recall@10_semantic_concept'].mean()))
    print("Mean cluster coherence            :", float(df['cluster_coherence'].mean()))


# -------------------- CLI --------------------
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf", "runpod"], default="hf")
    p.add_argument("--model", default="Qwen/Qwen3-1.8B", help="HF model id (when --backend hf)")
    p.add_argument("--protoqa_dir", required=True, help="Path to protoqa-data/data")
    p.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    p.add_argument("--out", default="protoqa_results.csv")
    p.add_argument("--top_n", type=int, default=10, help="# single-word answers to return (baseline)")
    p.add_argument("--top_k_tokens", type=int, default=100, help="top-k next-token candidates for clustering")
    p.add_argument("--distance_threshold", type=float, default=0.45, help="agglomerative clustering distance threshold")
    p.add_argument("--sem_thresh", type=float, default=0.62, help="semantic match threshold for Recall@10 (MiniLM cosine)")
    args = p.parse_args()

    run_protoqa(
        backend=args.backend,
        model_name=args.model,
        protoqa_dir=args.protoqa_dir,
        split=args.split,
        out_csv=args.out,
        top_n=args.top_n,
        top_k_tokens=args.top_k_tokens,
        distance_threshold=args.distance_threshold,
        sem_thresh=args.sem_thresh,
    )

if __name__ == "__main__":
    cli()
