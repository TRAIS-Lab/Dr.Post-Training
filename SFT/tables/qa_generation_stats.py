#!/usr/bin/env python
"""Format-vs-content decomposition of the question-answering metrics from the saved generations (Llama-3.2-1B, LoRA).

Reads <run dir>/<task>_generations.jsonl (written by SFT/eval: prediction, raw generation, references, per-example
score, n_tokens) for the LoRA arms with exact scoring, Target-Only (full budget) and the 50-step Target-Only probe, and reports per
setting x arm, mean over seeds:
  metric        the paper metric (ROUGE-L x100 / F1) recomputed from the file
  words         mean words per cleaned prediction (references: samsum ~20, tydiqa ~6 (median 3), nq_open ~2, squad ~3)
  run-on        share of generations that hit max_new_tokens (never emitted a stop token)
  script match  tydiqa only: share of predictions whose letters are mostly in the reference's script (Arabic vs Latin)
  metric|trunc  metric after truncating every prediction to the reference median length (tydiqa 3 / nq_open 2 / squad 2 words,
                samsum 18): how much of an arm's gap is verbosity rather than content
Usage: python SFT/tables/qa_generation_stats.py [--results-root DIR] [--md OUT.md]   (default --md SFT/tables/data/qa_generation_stats.md)
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import REPO, results_root  # noqa: E402
from SFT.eval.tasks.nq_open import best_alias_score, f1_score  # noqa: E402
from SFT.eval.tasks.tydiqa import _f1  # noqa: E402

SETTINGS = [("alpaca_samsum", "alpaca_samsum", "samsum"), ("less_tydiqa", "less_tydiqa", "tydiqa"),
            ("triviaqa_nq", "triviaqa_nq_open", "nq_open"), ("less_squad", "less_squad", "squad")]
SEEDS = [42, 2, 22, 62, 82]
TRUNC = {"samsum": 18, "tydiqa": 3, "nq_open": 2, "squad": 2}
MAX_NEW = {"samsum": 128, "tydiqa": 50, "nq_open": 32, "squad": 32}
ARMS = [("Full-Training", "FullTraining-LoRA"), ("Global Subset", "GlobalSubset-LoRA-gip"), ("Layer-Wise Subset", "LayerWiseSubset-LoRA-gip"),
        ("Target-Only (full budget)", "TO"), ("Target-Only (50 steps)", "TO50")]


def arabic_share(s: str) -> float | None:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return None
    return sum("ARABIC" in unicodedata.name(c, "") for c in letters) / len(letters)


def score(task, pred, refs):
    if task == "tydiqa":
        return 100 * _f1(pred, refs[0])
    if task == "samsum":
        return None   # ROUGE-L needs the corpus-level scorer; use the per-example value stored in the file
    return 100 * best_alias_score(pred, refs, f1_score)


def rouge_l(preds, refs):
    try:
        import evaluate
        r = evaluate.load("rouge").compute(predictions=preds, references=refs, use_stemmer=True)
        return 100 * r["rougeL"]
    except Exception:
        return None


def truncate(pred: str, k: int) -> str:
    w = pred.split()
    return " ".join(w[:k])


def stats(task, rows):
    preds = [r["prediction"] for r in rows]
    refs = [r["references"] for r in rows]
    words = [len(p.split()) for p in preds]
    run_on = [r.get("n_tokens", 0) >= MAX_NEW[task] - 1 for r in rows]
    out = {"words": st.mean(words), "run_on": 100 * st.mean(run_on)}
    tr = [truncate(p, TRUNC[task]) for p in preds]
    if task == "samsum":
        out["metric"] = rouge_l(preds, [r[0] for r in refs])
        out["metric_trunc"] = rouge_l(tr, [r[0] for r in refs])
    else:
        out["metric"] = st.mean(score(task, p, r) for p, r in zip(preds, refs))
        out["metric_trunc"] = st.mean(score(task, p, r) for p, r in zip(tr, refs))
    if task == "tydiqa":
        sh = [arabic_share(p) for p in preds]
        ref_sh = [arabic_share(r[0]) for r in refs]
        match = [(s is not None and rs is not None and (s >= 0.5) == (rs >= 0.5)) for s, rs in zip(sh, ref_sh)]
        out["script_match"] = 100 * st.mean(match)
    return out


def find(runs: Path, prefix, task, arm, seed):
    if arm == "TO":
        pat = f"{runs}/{task}_val_{task}-Llama-3.2-1B-FullTraining-LoRA-ms[0-9][0-9][0-9]*-lr*-b8-v16-s{seed}/{task}_generations.jsonl"
        m = [p for p in sorted(glob.glob(pat)) if not re.search(r"-ms50-", p)]
    elif arm == "TO50":
        m = sorted(glob.glob(f"{runs}/target_only_ms50/{task}_val_{task}-Llama-3.2-1B-FullTraining-LoRA-ms50-lr*-b8-v16-s{seed}/{task}_generations.jsonl"))
    else:
        m = sorted(glob.glob(f"{runs}/{prefix}-Llama-3.2-1B-{arm}-p*-lr*-b8-v16-s{seed}/{task}_generations.jsonl"))
    return Path(m[0]) if m else None


def fmt(v, d=1):
    return "-" if v is None else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--md", type=Path, default=REPO / "SFT" / "tables" / "data" / "qa_generation_stats.md")
    a = ap.parse_args()
    runs = results_root(a.results_root) / "SFT" / "runs_v2"
    out = ["# Generation statistics (LoRA, exact scoring; mean over seeds)\n", f"runs: `{runs}`\n"]
    for setting, prefix, task in SETTINGS:
        cols = ["metric", "words", "run-on %"] + (["script match %"] if task == "tydiqa" else []) + [f"metric | first {TRUNC[task]} words"]
        out.append(f"\n## {setting} ({task})\n")
        out.append("| arm | seeds | " + " | ".join(cols) + " |")
        out.append("|---|---|" + "---|" * len(cols))
        for label, arm in ARMS:
            per = []
            for seed in SEEDS:
                f = find(runs, prefix, task, arm, seed)
                if f is None:
                    continue
                rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
                if rows:
                    per.append(stats(task, rows))
            if not per:
                out.append(f"| {label} | 0 | " + " | ".join("-" for _ in cols) + " |")
                continue
            agg = lambda k, d=1: fmt(st.mean(p[k] for p in per) if all(p.get(k) is not None for p in per) else None, d)   # noqa: E731
            vals = [agg("metric", 2), agg("words"), agg("run_on")] + ([agg("script_match")] if task == "tydiqa" else []) + [agg("metric_trunc", 2)]
            out.append(f"| {label} | {len(per)} | " + " | ".join(vals) + " |")
    text = "\n".join(out)
    print(text)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(text + "\n")


if __name__ == "__main__":
    main()
