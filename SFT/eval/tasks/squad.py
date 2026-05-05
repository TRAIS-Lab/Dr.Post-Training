"""
SQuAD closed-book evaluation (no context): model is given the question only,
must produce the answer string. Metric: EM and F1 against the answers.text list
(typically 1-3 clean gold strings per question).

NOTE: SQuAD questions are usually context-dependent — closed-book performance
will be low compared to the open-book setting.
"""

import json
import os
from typing import Dict, List

from .nq_open import normalize_answer, f1_score, best_alias_score
from ..utils import generate_completions


def load_test(data_dir: str, k: int = -1) -> List[Dict]:
    f = os.path.join(data_dir, "eval", "squad", "squad_test_data.jsonl")
    if not os.path.exists(f):
        raise FileNotFoundError(f"SQuAD test data not found: {f}")
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
    print(f"Loaded {len(test)} SQuAD (closed-book) test examples")

    prompts = [f"<|user|>\n{r['messages'][0]['content']}\n<|assistant|>\n" for r in test]
    print("Generating answers...")
    gens = generate_completions(
        model, tokenizer, prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature=1.0,
        disable_tqdm=False,
    )

    em_total = 0.0
    f1_total = 0.0
    n = 0
    for r, g in zip(test, gens):
        out = g.strip()
        for stop in ["<|user|>", "<|assistant|>", "\n"]:
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
    print(f"\nSQuAD (closed-book) Results:")
    print(f"  EM: {em_mean:.4f}  F1: {f1_mean:.4f}  n={n}")
    return {"accuracy": em_mean, "em": em_mean, "f1": f1_mean, "n_test": n}
