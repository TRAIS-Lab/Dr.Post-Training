"""
NaturalQuestions-open (closed-book) evaluation: model is given a question, must
produce the answer string from its own knowledge. Metric: EM and F1 against the
alias list (best-alias scoring).
"""

import json
import os
import re
import string
from collections import Counter
from typing import Dict, List

from ..utils import generate_completions


def normalize_answer(s: str) -> str:
    """Lowercase, strip articles + punctuation + extra whitespace (SQuAD-style)."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_alias_score(pred: str, aliases: List[str], scorer):
    """Return the best score across all aliases."""
    return max((scorer(pred, a) for a in aliases), default=0.0)


def load_test(data_dir: str, k: int = -1) -> List[Dict]:
    f = os.path.join(data_dir, "eval", "nq_open", "nq_open_test_data.jsonl")
    if not os.path.exists(f):
        raise FileNotFoundError(f"NQ-open test data not found: {f}")
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
    print(f"Loaded {len(test)} NQ-open test examples")

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
    print(f"\nNQ-open Results:")
    print(f"  EM: {em_mean:.4f}  F1: {f1_mean:.4f}  n={n}")
    return {"accuracy": em_mean, "em": em_mean, "f1": f1_mean, "n_test": n}
