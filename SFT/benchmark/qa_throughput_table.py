#!/usr/bin/env python
"""Summarise the real-job step throughput runs of run_qa_throughput.sh (Markdown to stdout; a diagnostic, not a paper table).

Run directories are named <Method>-<Finetuning>_<scoring>_<backend>.  Per run, ms/step = slope of train_wall_time between the
first and the last logged evaluation (kernel compilation and warm-up fall in the first interval; `load_run` lives in
SFT/tables/qa_timing.py, which writes the paper's qa_timing_grid_* tables from the same runs).  The final training loss is listed
as a sanity check that fused and reference runs follow the same trajectory.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))
from SFT.tables.qa_timing import load_run  # noqa: E402

FINETUNING = ("Full", "LoRA", "MeSO")
METHODS = ("LayerWiseSubset", "GlobalSubset")
SCORINGS = ("compress", "gip", "pip")
LABEL = {"FullTraining": "Full-Training", "LayerWiseSubset": "Layer-Wise Subset", "GlobalSubset": "Global Subset (1P)"}
SLABEL = {"compress": "Compressed", "gip": "GIP", "pip": "PIP", "na": ""}
NAME_RE = re.compile(r"^(\w+)-(\w+)_(\w+)_(\w+)$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "paper", "qa_throughput"))
    args = ap.parse_args()
    runs = {}
    for d in sorted(glob.glob(os.path.join(args.dir, "*"))):
        m = NAME_RE.match(os.path.basename(d))
        if not m or not os.path.isdir(d):
            continue
        res = load_run(d)
        if res:
            runs[(m.group(1), m.group(2), m.group(3), m.group(4))] = res
    if not runs:
        print("no finished runs in", args.dir)
        return
    backends = sorted({k[3] for k in runs}, key=lambda b: (b != "cute", b))
    print("alpaca -> samsum, Llama-3.2-1B, n=8 T<=512 m=1 k=4 (ms/step, slope over the logged evaluations; "
          "overhead vs the Full-Training row of the same fine-tuning mode)\n")
    hdr = "| Fine-tuning | Method | Scoring | " + " | ".join(f"{b} ms/step" for b in backends) + " | vs Full-Training | final loss |"
    print(hdr); print("|" + "---|" * (hdr.count("|") - 1))
    rows = []   # (ft, method, scoring, {backend: run}, base)
    for ft in FINETUNING:
        base = next((runs[k] for k in runs if k[0] == "FullTraining" and k[1] == ft), None)
        if base:
            rows.append((ft, "FullTraining", "na", {backends[0]: base}, base))
        for meth in METHODS:
            for sc in SCORINGS:
                per_b = {b: runs[(meth, ft, sc, b)] for b in backends if (meth, ft, sc, b) in runs}
                if per_b:
                    rows.append((ft, meth, sc, per_b, base))
    for ft, meth, sc, per_b, base in rows:
        cells, over, losses = [], [], []
        for b in backends:
            r = per_b.get(b)
            if r is None:
                cells.append("—"); continue
            cells.append(f"{r['ms_per_step']:.0f}")
            if base and meth != "FullTraining":
                over.append(f"{b}: {100 * (r['ms_per_step'] / base['ms_per_step'] - 1):+.1f}%")
            if r["final_loss"] is not None:
                losses.append(f"{r['final_loss']:.3f}")
        print(f"| {ft} | {LABEL[meth]} | {SLABEL[sc]} | " + " | ".join(cells) + f" | {', '.join(over) or '—'} | {'/'.join(losses) or '—'} |")


if __name__ == "__main__":
    main()
