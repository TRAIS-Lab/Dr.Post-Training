#!/usr/bin/env python
"""Target-Only 50-step probe (Llama-3.2-1B question answering): the probe runs next to the full-budget rows of the run table.

Not a paper table. The probe trains the Target-Only arm on the 16 target examples for 50 steps instead of the main run's budget
(the held-out perplexity of the full-budget runs is minimal within the first ~50 steps, read off the test set, so the probe is an
oracle early stop) and evaluates with the paper protocol (SFT/eval/eval.sh --batch_size 64 --n_test 500).
Probe runs:   <results root>/SFT/runs_v2/target_only_ms50/<task>_val_<task>-Llama-3.2-1B-FullTraining-<FT>-ms50-lr<lr>-b8-v16-s<seed>/
              (SFT/train/qa_target_only_ms50_task.sh over SFT/train/manifests/qa_target_only_ms50{,_meso}.txt; 3 fine-tuning
              methods x 4 settings x 5 seeds) -> <task>_results.json (downstream metric), evaluation_results.json (held-out
              perplexity every 5 steps).
Baselines:    SFT/tables/data/qa_runs.csv (SFT/tables/qa_downstream.py --scan): Target-Only (full budget), Full-Training,
              Global Subset / Layer-Wise Subset with exact scoring. The `protocol` column says which evaluation encoding produced
              each baseline row; rows still on the earlier encoding (leading <|begin_of_text|>, sampled decoding for
              samsum / tydiqa) are counted per cell, since the probe was evaluated with the fixed one.
Usage: python SFT/tables/qa_target_only_probe.py [--results-root DIR] [--csv FILE] [--md OUT.md]
       (default --md: SFT/tables/data/qa_target_only_ms50.md, gitignored like the csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import REPO, results_root  # noqa: E402

SETTING_OF_TASK = {"samsum": "alpaca_samsum", "tydiqa": "less_tydiqa", "nq_open": "triviaqa_nq", "squad": "less_squad"}
TO_RE = re.compile(r"^(?P<task>samsum|tydiqa|nq_open|squad)_val_(?P=task)-Llama-3\.2-1B-FullTraining-(?P<ft>Full|LoRA|MeSO)-ms(?P<ms>\d+)-lr[\d.e+-]+-b8-v16-s(?P<seed>\d+)$")
SETTINGS = ["alpaca_samsum", "less_tydiqa", "triviaqa_nq", "less_squad"]
FTS = ["LoRA", "Full", "MeSO"]
SEEDS = [42, 2, 22, 62, 82]
FIXED = "chat_template_no_bos+greedy"
BASE = [("TargetOnly", "-", "", "Target-Only (full budget)"), ("FullTraining", "-", "", "Full-Training"),
        ("GlobalSubset", "gip", "", "Global Subset (exact)"), ("LayerWiseSubset", "gip", "", "Layer-Wise Subset (exact)")]


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return st.mean(xs), (st.pstdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0), len(xs)


def fmt(m, se, n):
    return "-" if m is None else f"{m:.2f} ± {se:.2f} (n={n})"


def read_probe(root: Path):
    runs = {}
    for d in sorted(root.iterdir()) if root.exists() else []:
        t = TO_RE.match(d.name)
        if not t or not d.is_dir():
            continue
        task, ft, seed = t["task"], t["ft"], int(t["seed"])
        rec = {"dir": d.name, "metric": None, "protocol": None, "ppl": [], "final_ppl": None, "min_ppl": None, "min_step": None}
        rp = d / f"{task}_results.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            rec["metric"] = 100 * r["rougeL"] if task == "samsum" else r["f1_score"]
            rec["protocol"] = r.get("prompt_encoding", "bos")
        ep = d / "evaluation_results.json"
        if ep.exists():
            ev = json.loads(ep.read_text())
            ev = ev if isinstance(ev, list) else ev.get("results", [])
            pts = sorted({(e["step"], e["eval_perplexity"]) for e in ev if e.get("eval_perplexity") is not None})
            rec["ppl"] = pts
            if pts:
                rec["final_ppl"] = pts[-1][1]
                rec["min_ppl"], rec["min_step"] = min((p, s) for s, p in pts)
        runs[(SETTING_OF_TASK[task], ft, seed)] = rec
    return runs


def read_csv(path: Path):
    rows = defaultdict(dict)   # (setting, ft, method, scoring, variant) -> {seed: row}
    for r in csv.DictReader(open(path)):
        rows[(r["setting"], r["finetuning"], r["method"], r["scoring"], r["variant"])][int(r["seed"])] = r
    return rows


def n_old(rows) -> int:
    """Baseline rows evaluated with the pre-fix encoding (csv exports without a protocol column count as pre-fix)."""
    return sum(1 for r in rows.values() if r.get("protocol", "bos") != FIXED)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--csv", type=Path, default=REPO / "SFT" / "tables" / "data" / "qa_runs.csv")
    ap.add_argument("--md", type=Path, default=REPO / "SFT" / "tables" / "data" / "qa_target_only_ms50.md")
    a = ap.parse_args()
    root = results_root(a.results_root) / "SFT" / "runs_v2" / "target_only_ms50"
    probe = read_probe(root)
    base = read_csv(a.csv)
    out = []
    P = out.append
    P(f"# Target-Only 50-step probe vs full-budget rows\n\nprobe runs: {len(probe)} under `{root}`; baselines: `{a.csv}`.\n")
    P("Metric = ROUGE-L x100 (samsum) or F1 (tydiqa / nq_open / squad) on the 500-example test protocol; mean ± SE over seeds "
      "(pstdev/sqrt(n)). Paired Δ = probe − baseline on the same seed. A trailing `[k old]` marks a baseline cell with k rows still "
      "on the earlier evaluation encoding (the probe rows all use the fixed one).\n")
    old_probe = [rec["dir"] for rec in probe.values() if rec["metric"] is not None and rec["protocol"] != FIXED]
    if old_probe:
        P(f"WARNING: {len(old_probe)} probe run(s) evaluated with the pre-fix encoding.\n")
    for ft in FTS:
        P(f"\n## {ft}\n")
        P("| setting | Target-Only ms50 (probe) | " + " | ".join(b[3] for b in BASE) + " | paired Δ probe − LWS | paired Δ probe − TO(full) |")
        P("|---|" + "---|" * (len(BASE) + 3))
        for s in SETTINGS:
            pr = {seed: rec for (ss, f, seed), rec in probe.items() if ss == s and f == ft and rec["metric"] is not None}
            cells = [fmt(*mean_se([r["metric"] for r in pr.values()]))]
            for m, sc, var, _ in BASE:
                rows = base.get((s, ft, m, sc, var), {})
                k = n_old(rows)
                cells.append(fmt(*mean_se([float(r["metric"]) for r in rows.values()])) + (f" [{k} old]" if k else ""))
            deltas = []
            for m, sc, var in (("LayerWiseSubset", "gip", ""), ("TargetOnly", "-", "")):
                rows = base.get((s, ft, m, sc, var), {})
                d = [pr[k]["metric"] - float(rows[k]["metric"]) for k in pr if k in rows]
                mm, se, n = mean_se(d)
                deltas.append("-" if mm is None else f"{mm:+.2f} ± {se:.2f} (n={n})")
            P(f"| {s} | " + " | ".join(cells + deltas) + " |")
        P(f"\n### held-out perplexity along the 50 steps ({ft}; mean over seeds)\n")
        steps = sorted({s for (ss, f, _), rec in probe.items() if f == ft and rec["metric"] is not None for s, _ in rec["ppl"]})
        if steps:
            P("| setting | " + " | ".join(f"s{s}" for s in steps) + " | min ppl @ step | TO(full) final ppl | Full-Training final ppl | LWS final ppl |")
            P("|---|" + "---|" * (len(steps) + 4))
            for s in SETTINGS:
                recs = [rec for (ss, f, _), rec in probe.items() if ss == s and f == ft and rec["ppl"] and rec["metric"] is not None]
                if not recs:
                    continue
                by = defaultdict(list)
                for rec in recs:
                    for stp, p in rec["ppl"]:
                        by[stp].append(p)
                traj = [f"{st.mean(by[stp]):.2f}" if by.get(stp) else "-" for stp in steps]
                tail = []
                for m, sc, var in (("TargetOnly", "-", ""), ("FullTraining", "-", ""), ("LayerWiseSubset", "gip", "")):
                    rows = base.get((s, ft, m, sc, var), {})
                    v = [float(r["final_eval_ppl"]) for r in rows.values() if r["final_eval_ppl"] not in ("", "nan")]
                    tail.append(f"{st.mean(v):.2f}" if v else "-")
                P(f"| {s} | " + " | ".join(traj) + f" | {st.mean(r['min_ppl'] for r in recs):.2f} @ {st.mean(r['min_step'] for r in recs):.0f} | "
                  + " | ".join(tail) + " |")
    missing = [(s, ft, seed) for s in SETTINGS for ft in FTS for seed in SEEDS if probe.get((s, ft, seed), {}).get("metric") is None]
    P(f"\nmissing / unfinished probe cells: {len(missing)}" + (": " + ", ".join(f"{s}/{ft}/s{seed}" for s, ft, seed in missing) if missing else ""))
    text = "\n".join(out)
    print(text)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(text + "\n")
        print(f"\n-> {a.md.relative_to(REPO) if a.md.is_relative_to(REPO) else a.md}")


if __name__ == "__main__":
    main()
