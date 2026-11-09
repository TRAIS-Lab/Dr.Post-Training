#!/usr/bin/env python
"""Tables of the GREATS settings at 7-8B (configs/less_tydiqa_mistral7b_greats, configs/alpaca_samsum_llama3_8b_greats).

Reads the run dirs written by SFT/train/greats_task.sh (adapters evaluated unmerged) and prints, per setting, mean +- standard error
over seeds of
  - TyDiQA: 1-shot macro / micro F1 and EM on GREATS's 500 test rows, and the held-out / D* perplexity at step 3,000;
  - SAMSum: held-out / D* perplexity at step 1,000 (GREATS's plotted horizon) and at the end of the epoch (GREATS reports this
    task by perplexity only);
  - training wall-clock: the trainer's train_wall_time (training loop incl. target gradients and scoring, minus every in-loop
    evaluation), with the overhead of each curation arm over Full-Training.

    python SFT/tables/greats_settings.py [--runs $SCRATCH_DIR/Dr.Post-Training/SFT/runs] [--per-seed]
"""
import argparse
import glob
import json
import math
import os
import statistics as st

ARMS = [("FullTraining-LoRA", "Full-Training"), ("GlobalSubset-LoRA", "Global Subset (= GREATS)"),
        ("LayerWiseSubset-LoRA", "Layer-Wise Subset")]
SETTINGS = [
    ("TyDiQA (Mistral-7B, LESS mixture, step 3,000)", "less_official_tydiqa_less-Mistral-7B-v0.1", "tydiqa_less", 3000),
    ("SAMSum (Llama-3-8B, Alpaca, 1 epoch)", "alpaca_tt_samsum_tt-Meta-Llama-3-8B", "samsum_tt", None),
]


def mse(xs, fmt="{:.2f}"):
    if not xs:
        return "-"
    if len(xs) == 1:
        return fmt.format(xs[0])
    return (fmt + " ± " + fmt).format(st.mean(xs), st.stdev(xs) / math.sqrt(len(xs)))


def at_step(curve, step, key):
    pts = [e for e in curve if e["step"] == step]
    return pts[-1][key] if pts else None


def load(runs, prefix, target, stop):
    out = {}
    for arm, _ in ARMS:
        for d in sorted(glob.glob(os.path.join(runs, f"{prefix}-{arm}-p*"))):
            res_path = os.path.join(d, f"{target}_results.json")
            if not os.path.exists(res_path):
                continue
            res = json.load(open(res_path))
            if res.get("lora_merge", True):
                continue  # merged-adapter results are not comparable (SFT/train/greats_task.sh)
            curve = json.load(open(os.path.join(d, "evaluation_results.json")))
            last = curve[-1]
            row = {"seed": int(d.rsplit("-s", 1)[1]), "ppl": last["eval_perplexity"], "dppl": last["val_perplexity"],
                   "train_min": last["train_wall_time"] / 60}
            if target == "tydiqa_less":
                row.update(f1=res["f1_score"], f1_micro=res["f1_micro"], em=res["exact_match"])
            else:
                row.update(ppl1k=at_step(curve, 1000, "eval_perplexity"), dppl1k=at_step(curve, 1000, "val_perplexity"))
            if stop is not None:
                assert last["step"] == stop, f"{d}: last step {last['step']} != {stop}"
            out.setdefault(arm, []).append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=os.path.join(os.environ.get("SCRATCH_DIR", "."), "Dr.Post-Training", "SFT", "runs"))
    ap.add_argument("--per-seed", action="store_true")
    args = ap.parse_args()
    for title, prefix, target, stop in SETTINGS:
        rows = load(args.runs, prefix, target, stop)
        if not rows:
            continue
        print(f"\n## {title}\n")
        col = lambda arm, k, f="{:.2f}": mse([r[k] for r in rows.get(arm, []) if r.get(k) is not None], f)  # noqa: E731
        if target == "tydiqa_less":
            print("| Method | Seeds | macro-F1 | micro-F1 | EM | Held-out ppl | D* ppl |\n|---|---|---|---|---|---|---|")
            for arm, name in ARMS:
                print(f"| {name} | {len(rows.get(arm, []))} | {col(arm, 'f1')} | {col(arm, 'f1_micro')} | {col(arm, 'em')} | "
                      f"{col(arm, 'ppl', '{:.4f}')} | {col(arm, 'dppl', '{:.3f}')} |")
        else:
            print("| Method | Seeds | Held-out ppl @1K | Held-out ppl @end | D* ppl @1K | D* ppl @end |\n|---|---|---|---|---|---|")
            for arm, name in ARMS:
                print(f"| {name} | {len(rows.get(arm, []))} | {col(arm, 'ppl1k')} | {col(arm, 'ppl')} | {col(arm, 'dppl1k')} | "
                      f"{col(arm, 'dppl')} |")
        base = st.mean(r["train_min"] for r in rows["FullTraining-LoRA"]) if rows.get("FullTraining-LoRA") else None
        print("\n| Method | Training wall-clock (min) | Overhead vs Full-Training |\n|---|---|---|")
        for arm, name in ARMS:
            if not rows.get(arm):
                continue
            over = "-" if arm == "FullTraining-LoRA" or base is None else \
                f"{100 * (st.mean(r['train_min'] for r in rows[arm]) / base - 1):+.0f}%"
            print(f"| {name} | {col(arm, 'train_min')} | {over} |")
        if args.per_seed:
            keys = ["f1", "f1_micro", "em", "ppl", "dppl"] if target == "tydiqa_less" else ["ppl1k", "ppl", "dppl1k", "dppl"]
            print("\n| Method | Seed | " + " | ".join(keys) + " | train min |\n|---|---|" + "---|" * (len(keys) + 1))
            for arm, name in ARMS:
                for r in sorted(rows.get(arm, []), key=lambda r: r["seed"]):
                    print(f"| {name} | {r['seed']} | " + " | ".join(f"{r[k]:.4g}" for k in keys) + f" | {r['train_min']:.1f} |")


if __name__ == "__main__":
    main()
