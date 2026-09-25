#!/usr/bin/env python
"""Target-set size sweep (Llama-3.2-1B question answering, LoRA): downstream metric of the curated arms and Target-Only at 2 / 4 / 8 / 16
target examples, next to Full-Training (which does not see the target set).

Not a paper table (the paper's n_val ablation is the Qwen one, SFT/tables/qwen_capability.py). Tests the explanation of the Target-Only
results: on the format-dominated targets (samsum, tydiqa) a handful of target examples should already carry Target-Only, while the
curated arms' scoring signal degrades with fewer target examples; on the knowledge targets (nq_open, squad) Target-Only should stay
below the curated arms at every size.
Runs:   <results root>/SFT/runs_v2/<pool>_<target>-Llama-3.2-1B-<arm>-p<pct>-lr1.00e-04-b8-v<k>-s<seed>/<target>_results.json
        (SFT/train/qa_matrix_task.sh with N_VAL=k over manifests/qa_nval_sweep.txt; arms LayerWiseSubset-LoRA-gip, GlobalSubset-LoRA-gip)
        <target>_val_<target>-Llama-3.2-1B-FullTraining-LoRA-ms<budget>-lr1.00e-04-b8-v<k>-s<seed>/ (qa_target_only_task.sh with N_VAL=k
        over manifests/qa_nval_sweep_target_only.txt); k = 16 rows are the paper's runs.
Usage:  python SFT/tables/qa_nval_sweep.py [--results-root DIR] [--md OUT.md]   (default --md SFT/tables/data/qa_nval_sweep.md)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import REPO, results_root  # noqa: E402

SETTINGS = [("alpaca_samsum", "alpaca_samsum", "samsum"), ("less_tydiqa", "less_tydiqa", "tydiqa"),
            ("triviaqa_nq", "triviaqa_nq_open", "nq_open"), ("less_squad", "less_squad", "squad")]
NVALS = [2, 4, 8, 16]
SEEDS = [42, 2, 22, 62, 82]
ARMS = [("Full-Training", "FullTraining-LoRA", False), ("Global Subset (exact)", "GlobalSubset-LoRA-gip", True),
        ("Layer-Wise Subset (exact)", "LayerWiseSubset-LoRA-gip", True), ("Target-Only", None, True)]


def metric(path: Path, task: str):
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    return 100 * r["rougeL"] if task == "samsum" else r["f1_score"]


def cell(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    se = st.pstdev(vals) / len(vals) ** 0.5 if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.2f} ± {se:.2f}" + ("" if len(vals) == len(SEEDS) else f" (n={len(vals)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--md", type=Path, default=REPO / "SFT" / "tables" / "data" / "qa_nval_sweep.md")
    a = ap.parse_args()
    runs = results_root(a.results_root) / "SFT" / "runs_v2"
    out = ["# Target-set size sweep (LoRA, exact scoring): downstream metric, mean ± SE over 5 seeds\n",
           f"runs: `{runs}`; ROUGE-L x100 (samsum) or F1 (others), 500-example test protocol.\n"]
    for setting, prefix, task in SETTINGS:
        out.append(f"\n## {setting}\n")
        out.append("| target examples | " + " | ".join(n for n, _, _ in ARMS) + " |")
        out.append("|---|" + "---|" * len(ARMS))
        for k in NVALS:
            row = []
            for _, yaml_name, depends in ARMS:
                kk = k if depends else 16
                vals = []
                for seed in SEEDS:
                    if yaml_name is None:
                        pat = f"{runs}/{task}_val_{task}-Llama-3.2-1B-FullTraining-LoRA-ms*-lr*-b8-v{kk}-s{seed}/{task}_results.json"
                    else:
                        pat = f"{runs}/{prefix}-Llama-3.2-1B-{yaml_name}-p*-lr*-b8-v{kk}-s{seed}/{task}_results.json"
                    m = sorted(glob.glob(pat))
                    vals.append(metric(Path(m[0]), task) if m else None)
                row.append(cell(vals))
            out.append(f"| {k} | " + " | ".join(row) + " |")
    text = "\n".join(out)
    print(text)
    if a.md:
        a.md.parent.mkdir(parents=True, exist_ok=True)
        a.md.write_text(text + "\n")


if __name__ == "__main__":
    main()
