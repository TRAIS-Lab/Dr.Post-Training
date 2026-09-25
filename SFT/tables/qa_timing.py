"""Paper tables: qa_timing_grid_full.tex, qa_timing_grid_lora.tex, qa_timing_grid_meso.tex (adxtab:qa-timing-grid).

Experiments run on : one node with 4x A40
Launcher           : SFT/benchmark/run_qa_throughput.sh (real alpaca -> samsum steps of the production launcher, Llama-3.2-1B, n = 8,
                     T <= 512, m = 1, k = 4; Full / LoRA / MeSO x {Full-Training, Layer-Wise Subset, Global Subset (1P)} x
                     {compressed, GIP, PIP}; CuTe kernels) and PROFILE=1 for the torch.profiler runs (steps 30-39 attributed by
                     SFT/benchmark/trace_attribution.py)
Results read from  : SFT/benchmark/results/paper/qa_profile/<Method>-<ft>_<scoring>_cute/profile.json (GPU-busy ms per component) and
                     SFT/benchmark/results/paper/qa_throughput/<same name>/evaluation_results.json (un-profiled wall ms per step = slope of
                     train_wall_time over the logged evaluations); kept locally under the repo (gitignored), so --results-root is not used
Seeds / statistics : one run per cell; GPU-busy ms per component over 10 profiled steps at one decimal; Host / idle = un-profiled wall
                     minus all GPU-busy time; Total = un-profiled wall ms per step with the overhead over Full-Training in parentheses
Usage              : python SFT/tables/qa_timing.py [--benchmark-dir DIR] [--paper-root DIR] [--venues ICLR] [--check]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import REPO, add_common_args, emit, mrow  # noqa: E402

BENCH = REPO / "SFT" / "benchmark" / "results" / "paper"
FT = (("Full", "full"), ("LoRA", "lora"), ("MeSO", "meso"))
COLS = (("FullTraining", "na"), ("LayerWiseSubset", "compress"), ("LayerWiseSubset", "gip"), ("LayerWiseSubset", "pip"),
        ("GlobalSubset", "compress"), ("GlobalSubset", "gip"), ("GlobalSubset", "pip"))
ROWS = (("Forward", "forward", False), ("Backward", "Backward", False), ("a.grad", "a.grad", True), ("scoring", "scoring", True),
        ("w.grad", "w.grad", True), ("autograd", "autograd", True), ("Optimizer", "optimizer", False), ("Host / idle", "idle", False))


def load_run(run_dir: Path):
    """ms/step of a throughput run = slope of train_wall_time between the first and the last logged evaluation (the first interval
    absorbs kernel compilation and warm-up); also the per-window slopes, the step count and the final training loss."""
    p = Path(run_dir) / "evaluation_results.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    r = r if isinstance(r, list) else r.get("results", r.get("evaluation_results", []))
    seen, uniq = set(), []
    for e in r:   # the final explicit evaluate() repeats the last step; keep the in-loop entry
        if e.get("step", 0) > 0 and e["step"] not in seen:
            seen.add(e["step"])
            uniq.append(e)
    if len(uniq) < 2:
        return None
    first, last = uniq[0], uniq[-1]
    ms = 1000.0 * (last["train_wall_time"] - first["train_wall_time"]) / (last["step"] - first["step"])
    windows = [1000.0 * (b["train_wall_time"] - a["train_wall_time"]) / (b["step"] - a["step"]) for a, b in zip(uniq, uniq[1:])]
    log = Path(run_dir) / "train.log"
    losses = re.findall(r"'loss': ([0-9.]+)", log.read_text(errors="ignore")) if log.exists() else []
    return {"ms_per_step": ms, "windows": windows, "steps": last["step"], "final_loss": float(losses[-1]) if losses else None}


def column(bench: Path, method: str, ft: str, scoring: str, backend: str):
    name = f"{method}-{ft}_{scoring}_{backend}"
    pp = bench / "qa_profile" / name / "profile.json"
    if not pp.exists():
        return None
    prof = json.loads(pp.read_text())
    tr = load_run(bench / "qa_throughput" / name)
    wall = tr["ms_per_step"] if tr else None
    b = dict(prof["busy_ms"])
    b["Backward"] = sum(b[c] for c in ("a.grad", "scoring", "w.grad", "autograd"))
    b["idle"] = (wall - sum(prof["busy_ms"].values())) if wall is not None else None
    return {"busy": b, "wall": wall}


def fmt(v):
    return "---" if v is None else f"{v:.1f}"


def grid(bench: Path, ft: str, backend: str):
    cols = [column(bench, m, ft, sc, backend) for m, sc in COLS]
    if all(c is None for c in cols):
        return None
    base = cols[0]["wall"] if cols[0] else None
    L = [r"\begin{tabular}{l c ccc ccc}", "\t\\toprule",
         "\t" + mrow(2, r"\textbf{Component}") + " & " + mrow(2, r"\textbf{Full-Training}")
         + r" & \multicolumn{3}{c}{\textbf{Layer-Wise Subset}} & \multicolumn{3}{c}{\textbf{Global Subset (1P)}} \\",
         "\t\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}", "\t & & Compressed & GIP & PIP & Compressed & GIP & PIP \\\\", "\t\\midrule"]
    for label, key, shaded in ROWS:
        cells = [fmt(c["busy"].get(key)) if c else "---" for c in cols]
        if key == "scoring":
            cells[0] = "---"   # Full-Training does not score
        if shaded:
            L.append("\t\\rowcolor{black!6}")
        name = f"\\quad \\texttt{{{label}}}" if shaded else f"\\textbf{{{label}}}"
        L.append(f"\t{name} & " + " & ".join(cells) + r" \\")
    totals = []
    for i, c in enumerate(cols):
        if c is None or c["wall"] is None:
            totals.append("---")
        elif i == 0 or not base:
            totals.append(f"\\textbf{{{c['wall']:.0f}}}")
        else:
            totals.append(f"\\textbf{{{c['wall']:.0f}}} \\scriptsize({100 * (c['wall'] / base - 1):+.1f}\\%)")
    L += ["\t\\midrule", "\t\\textbf{Total} & " + " & ".join(totals) + r" \\", "\t\\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--benchmark-dir", default=str(BENCH), help="directory with qa_profile/ and qa_throughput/ (default SFT/benchmark/results/paper)")
    ap.add_argument("--backend", default="cute", help="kernel backend of the profiled runs (cute | off)")
    args = ap.parse_args()
    bench = Path(args.benchmark_dir)
    files = {f"qa_timing_grid_{tag}.tex": grid(bench, ft, args.backend) for ft, tag in FT}
    missing = [k for k, v in files.items() if v is None]
    if missing:
        sys.exit(f"no profiled runs for {missing}")
    return emit(files, args, __file__, bench.relative_to(REPO) if bench.is_relative_to(REPO) else bench,
                note="alpaca->samsum real steps, Llama-3.2-1B, torch.profiler GPU-busy ms + un-profiled wall")


if __name__ == "__main__":
    sys.exit(main())
