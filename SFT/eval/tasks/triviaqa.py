"""
TriviaQA closed-book evaluation: model is given a question, must produce an
answer string. Metric: EM and F1 against the alias list (best-alias scoring),
identical to the nq_open task.
"""

import json
import os
from typing import Dict, List

from SFT.data.get_val_dataset import render_chat
from .nq_open import normalize_answer, f1_score, best_alias_score
from ..utils import generate_completions, get_eos_token_ids


def load_test(data_dir: str, k: int = -1) -> List[Dict]:
    f = os.path.join(data_dir, "eval", "triviaqa", "triviaqa_test_data.jsonl")
    if not os.path.exists(f):
        raise FileNotFoundError(f"TriviaQA test data not found: {f}")
    out = []
    with open(f) as fp:
        for line in fp:
            out.append(json.loads(line.strip()))
            if k > 0 and len(out) >= k:
                break
    return out


def compute_accuracy(args, model, tokenizer, batch_size: int = 4, max_new_tokens: int = 32) -> Dict:
    data_dir = getattr(args, "data_dir", "./data")
    n_test = getattr(args, "n_test", -1)
    if n_test <= 0:
        n_test = 100000

    test = load_test(data_dir, k=n_test)
    print(f"Loaded {len(test)} TriviaQA test examples")

    prompts = [render_chat(tokenizer, r['messages'][0]['content']) for r in test]
    print("Generating answers...")
    gens = generate_completions(
        model, tokenizer, prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=get_eos_token_ids(tokenizer),
        do_sample=False,
        temperature=1.0,
        disable_tqdm=False,
    )

    em_total, f1_total, n = 0.0, 0.0, 0
    for r, g in zip(test, gens):
        out = g.strip()
        for stop in ["<|im_end|>", "<|im_start|>", "\n"]:
            if stop in out:
                out = out[:out.find(stop)]
        out = out.strip()

        aliases = r["metadata"].get("aliases") or [r["metadata"].get("primary_answer", "")]
        em = best_alias_score(out, aliases, lambda p, g: float(normalize_answer(p) == normalize_answer(g)))
        f1 = best_alias_score(out, aliases, f1_score)
        em_total += em
        f1_total += f1
        n += 1

    em_mean = em_total / n if n else 0.0
    f1_mean = f1_total / n if n else 0.0
    print(f"\nTriviaQA Results:")
    print(f"  EM: {em_mean*100:.2f}  F1: {f1_mean*100:.2f}  n={n}")
    return {
        "f1_score": f1_mean * 100.0,
        "exact_match": em_mean * 100.0,
        "f1": f1_mean,
        "em": em_mean,
        "accuracy": em_mean,
        "n_test": n,
    }
