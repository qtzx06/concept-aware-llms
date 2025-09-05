# concept-qwen/judge_eval.py
import os
import time
import argparse
import random
import pandas as pd
from dotenv import load_dotenv
from typing import Optional, Tuple
from openai import OpenAI

PAIR_SYSTEM_PROMPT = (
    "You are a strict evaluator that scores TWO model completions for plausibility in context.\n"
    "For EACH completion, output exactly one of: 0, 0.5, or 1.\n"
    "Scoring rubric:\n"
    " - 1: Clearly plausible/likely continuation for the given prompt.\n"
    " - 0.5: Possible but unlikely; tenuous fit but not nonsensical.\n"
    " - 0: Does not make sense in context / incoherent / irrelevant.\n"
    "Output format (no extra words):\n"
    "A: <score>\nB: <score>"
)

def get_openai_client() -> OpenAI:
    load_dotenv(dotenv_path=".env.local")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found. Put it in .env.local as OPENAI_API_KEY=sk-...")
    return OpenAI()

def parse_pair_scores(text: str) -> Optional[Tuple[float, float]]:
    t = (text or "").strip().lower()
    # common patterns: "A: 1\nB: 0.5" or "a=1;b=0.5" etc.
    # extract first two numbers in order
    import re
    nums = re.findall(r"(?:^|[^\d])(\d(?:\.\d)?)", t)
    # coerce to allowed set
    vals = []
    for n in nums:
        if n in {"0", "0.0"}: vals.append(0.0)
        elif n in {"0.5", ".5"}: vals.append(0.5)
        elif n in {"1", "1.0"}: vals.append(1.0)
        if len(vals) == 2: break
    if len(vals) == 2:
        return (vals[0], vals[1])
    return None

def llm_judge_score_pair(
    client: OpenAI,
    prompt_text: str,
    cand_a: str,
    cand_b: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
    retry_sleep: float = 2.0,
) -> Optional[Tuple[float,float]]:
    user_msg = (
        "Evaluate two candidate completions for the given prompt.\n\n"
        f"PROMPT:\n{prompt_text}\n\n"
        f"COMPLETION A:\n{cand_a}\n\n"
        f"COMPLETION B:\n{cand_b}\n\n"
        "Respond with exactly two lines:\nA: <0/0.5/1>\nB: <0/0.5/1>"
    )
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": PAIR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=8,
            )
            raw = (resp.choices[0].message.content or "").strip()
            parsed = parse_pair_scores(raw)
            if parsed:
                return parsed
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[LLM Judge] Failed after {max_retries} attempts: {e}")
                return None
            time.sleep(retry_sleep)
    return None

def _first_from_pipe(value: str) -> str:
    """If the cell contains a '|' separated list, take the first item."""
    s = str(value) if value is not None else ""
    parts = s.split("|")
    return parts[0].strip() if parts else s.strip()

def main():
    ap = argparse.ArgumentParser(description="LLM-as-a-judge paired scoring (randomized A/B) for concept-aware evals.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV (e.g., protoqa_results.csv)")
    ap.add_argument("--out", dest="out_csv", required=True, help="Output CSV with judge scores")
    ap.add_argument("--prompt_col", default="prompt", help="Column name containing the prompt")
    ap.add_argument("--baseline_col", default="baseline_preds", help="Column with baseline outputs (pipe-separated, take first)")
    ap.add_argument("--concept_col", default="concept_preds", help="Column with concept outputs (pipe-separated, take first)")
    ap.add_argument("--judge_model", default="gpt-4o-mini", help="OpenAI judge model (e.g., gpt-4o-mini)")
    args = ap.parse_args()

    client = get_openai_client()
    df = pd.read_csv(args.in_csv)

    for col in [args.prompt_col, args.baseline_col, args.concept_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    base_scores, concept_scores, ab_order = [], [], []
    for _, row in df.iterrows():
        prompt = str(row[args.prompt_col])
        base_out = _first_from_pipe(row[args.baseline_col])
        concept_out = _first_from_pipe(row[args.concept_col])

        # randomize order to avoid positional bias
        if random.random() < 0.5:
            A = base_out
            B = concept_out
            ab_order.append("baseline_first")
        else:
            A = concept_out
            B = base_out
            ab_order.append("concept_first")

        pair = llm_judge_score_pair(client, prompt, A, B, model=args.judge_model)
        if pair is None:
            base_scores.append(None)
            concept_scores.append(None)
            continue

        score_a, score_b = pair
        if ab_order[-1] == "baseline_first":
            base_scores.append(score_a)
            concept_scores.append(score_b)
        else:
            base_scores.append(score_b)
            concept_scores.append(score_a)

    df["judge_order"] = ab_order
    df["judge_score_baseline"] = base_scores
    df["judge_score_concept"] = concept_scores
    df.to_csv(args.out_csv, index=False)
    print(f"Saved judge-scored results → {args.out_csv}")

if __name__ == "__main__":
    main(