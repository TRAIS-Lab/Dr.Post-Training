"""Paper table: rlvr_accuracy.tex (adxtab:rlvr).

Reads the RLVR runs of W&B project verl_grpo_math (GRPO on MATH, Qwen3-1.7B-Base), one run per method and seed:
    Full-Training      Qwen3-1.7B_s<seed>
    Global Subset      Qwen3-1.7B_Global_s<seed>_reward
    Layer-Wise Subset  Qwen3-1.7B_LayerWise_s<seed>_reward
and reports the evaluation accuracy (%) at rounds 30, 60 and 78 (the last evaluation within the first 80 rounds;
evaluation runs every 3 rounds) as mean +- SE over the seeds (SE = population std / sqrt n).

Usage: python RLVR/tables/accuracy.py [--paper-root DIR] [--venues ICLR] [--check] [--stdout]   (needs wandb login)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import add_common_args, emit, mean_se  # noqa: E402

TABLE = "rlvr_accuracy.tex"
PROJECT = "verl_grpo_math"
EVAL_KEY = "val-aux/DigitalLearningGmbH/MATH-lighteval/reward/mean@1"   # eval accuracy (fraction) on the MATH test split
PATTERNS = {  # method -> W&B run-name pattern (seed in group 1)
    "FullTraining": re.compile(r"^Qwen3-1\.7B_s(\d+)$"),
    "GlobalSubset": re.compile(r"^Qwen3-1\.7B_Global_s(\d+)_reward$"),
    "LayerWiseSubset": re.compile(r"^Qwen3-1\.7B_LayerWise_s(\d+)_reward$"),
}
COLUMNS = [("FullTraining", "Full-Training"), ("GlobalSubset", "Global Subset"), ("LayerWiseSubset", "Layer-Wise Subset")]
ROUNDS = [("Round 30", 30), ("Round 60", 60), ("Final", 78)]


def load_runs() -> dict[str, dict[int, dict[int, float]]]:
    """{method: {seed: {round: accuracy %}}} for every run whose name matches a pattern."""
    import wandb

    out: dict[str, dict[int, dict[int, float]]] = {m: {} for m in PATTERNS}
    for run in wandb.Api(timeout=120).runs(PROJECT, per_page=100):
        for method, pat in PATTERNS.items():
            m = pat.match(run.name)
            if not m:
                continue
            seed = int(m.group(1))
            if seed in out[method]:
                sys.exit(f"two W&B runs match {method} seed {seed} (e.g. {run.name}); rename one of them")
            acc = {int(h["_step"]): 100.0 * h[EVAL_KEY] for h in run.scan_history(keys=["_step", EVAL_KEY], page_size=10000)
                   if h.get(EVAL_KEY) is not None}
            out[method][seed] = acc
            print(f"{run.name:40s} {len(acc)} evaluations")
    return out


def cell(values) -> str:
    ms = mean_se(values)
    if ms is None:
        return "--"
    m, se, n = ms
    return f"\\({m:.1f} \\pm {se:.1f}\\)" if n > 1 else f"\\({m:.1f}\\)"


def build(runs: dict[str, dict[int, dict[int, float]]]) -> str:
    lines = ["\\begin{tabular}{lccc}", "\t\\toprule",
             "\t & " + " & ".join(f"\\textbf{{{h}}}" for _, h in COLUMNS) + " \\\\", "\t\\midrule"]
    for label, rnd in ROUNDS:
        cells = []
        for method, _ in COLUMNS:
            vals = {s: acc[rnd] for s, acc in runs[method].items() if rnd in acc}
            print(f"{label:8s} {method:16s} n={len(vals)} seeds={sorted(vals)}")
            cells.append(cell(vals))
        lines.append(f"\t{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\t\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args = add_common_args(ap).parse_args()
    body = build(load_runs())
    return emit({TABLE: body}, args, __file__, Path(f"W&B project {PROJECT}"))


if __name__ == "__main__":
    sys.exit(main())
