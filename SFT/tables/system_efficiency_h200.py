"""Paper tables: system_overhead_h200.tex (adxtab:system-overhead-h200), peak_memory_h200.tex (adxtab:peak-memory-h200),
score_cost_h200.tex (adxtab:score-cost-h200).

Experiments run on : one H200 (141 GB) GPU per job on the Slurm cluster
Launcher           : SFT/benchmark/slurm/launch_h200.sh -> benchmark_run.py (per-component step timing; bf16, fused AdamW, the CuTe
                     backend with the Hopper TMA+wgmma kernels of drpt/kernels/hopper_ops.py, every scoring backend, m = 1, k = n/2;
                     no activation checkpointing by default, CKPT=1 for the checkpointing suite) and benchmark_scoring.py (standalone
                     scoring cost on synthetic tensors, T-sweep to 32768 and m-sweep). Primary model Qwen3-8B-Base (--model-tag
                     qwen3-8b --subdir breakdown); Qwen3-14B-Base ran with checkpointing (--model-tag qwen3-14b --subdir
                     breakdown_checkpointing); the A40 models at larger n are available under their tags.
Results read from  : SFT/benchmark/results/paper/h200/<subdir>/<tag>_n<n>_T<T>_m1.json and .../h200/scoring/scoring_<tag>.json
                     (the launcher writes to SFT/benchmark/results/h200/; the snapshot the tables are built from sits under paper/
                     like the A40 suite, local and gitignored; --benchmark-dir overrides)
Seeds / statistics : single timed run per cell (mean over the benchmark's timed steps, see benchmark.py); ms per step at one decimal,
                     totals rounded with the overhead over Full-Training in parentheses; peak memory in GB; scoring cost in ms with the
                     cheapest exact backend per column in bold; OOM marks a combination that did not fit on one GPU
Totals             : --totals plain (default) takes the Total row and the overhead from `plain_step_ms`, the same step timed without
                     the per-layer timing patches (the patched Full-Training step is ~4 % slower than the native step at 8-14B on an
                     H200, see the README's Hopper note), so the component rows (patched pass) do not sum to the Total;
                     --totals phases sums the per-phase timers as the A40 tables do.
Usage              : python SFT/tables/system_efficiency_h200.py [--benchmark-dir DIR] [--model-tag TAG]
                     [--subdir breakdown|breakdown_checkpointing] [--totals plain|phases] [--paper-root DIR] [--venues ICLR] [--check]
                     Curated columns use exact GIP scoring like the A40 tables; --scoring compress|pip|direct prints the other backends.

Same component definitions as SFT/tables/system_efficiency.py (which this module imports); the columns are batch
configurations of one model instead of models of one configuration.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import REPO, add_common_args, emit  # noqa: E402
import SFT.tables.system_efficiency as se  # noqa: E402

BENCH = REPO / "SFT" / "benchmark" / "results" / "paper" / "h200"
MODEL_TAG = "qwen3-8b"   # json tag of the model the tables quote; --model-tag overrides
TOTALS = "plain"         # "plain": plain_step_ms (unpatched step) | "phases": sum of the per-phase timers
T_SWEEP = [512, 2048, 8192, 16384, 32768]
M_SWEEP = [1, 4, 8]


def step_total(method: str, r: dict) -> float:
    """Total ms of a method's step: the unpatched timing when requested and recorded, else the phase sum."""
    if TOTALS == "plain" and r.get("plain_step_ms") is not None:
        return float(r["plain_step_ms"])
    return se.total_ms(method, r)


def configs_of(bd: dict, tag: str) -> list[str]:
    """Config labels present for a model, ordered by (T, n)."""
    def key(label):
        n, T = re.match(r"n=(\d+) T=(\d+)", label).groups()
        return int(T), int(n)
    return sorted(bd.get(tag, {}), key=key)


def cfg_math(label: str) -> str:
    n, T = re.match(r"n=(\d+) T=(\d+)", label).groups()
    return rf"\(n={n},\; \seqlen={T}\)"


def system_overhead(bd: dict, tag: str, labels: list[str]) -> str:
    k = len(labels)
    head = [rf"\begin{{tabular}}{{l {' '.join(['ccc'] * k)}}}", r"\toprule",
            "\t& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{cfg_math(l)}}}" for l in labels) + r" \\",
            "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(k)),
            "\t" + r"\textbf{Component} & " + " & ".join(["Full-Training & Layer-Wise Subset & Global Subset"] * k) + r" \\"]
    cols, totals = [], []
    for label in labels:
        combos = bd[tag][label]
        full, lw, op = combos.get("full_training/pip"), combos.get(f"layer_wise_subset/{se.SCORING}"), combos.get(f"global_subset_one_pass/{se.SCORING}")
        if not full:
            sys.exit(f"missing Full-Training result for {tag} {label}")
        base = step_total("full_training", full)
        cols.append(se.column("full_training", full))
        totals.append(se.tot(base))
        for method, r in (("layer_wise_subset", lw), ("global_subset_one_pass", op)):
            if r:
                cols.append(se.column(method, r))
                totals.append(se.tot(step_total(method, r), base))
            else:
                cols.append({key: None for key, _, _ in se.ROWS})
                totals.append("OOM")
    return se.tabular(head, cols, totals)


def peak_memory(bd: dict, tag: str, labels: list[str]) -> str:
    methods = ["full_training/pip", f"layer_wise_subset/{se.SCORING}", f"global_subset_one_pass/{se.SCORING}", f"global_subset/{se.SCORING}"]
    L = [r"\begin{tabular}{l cccc}", r"\toprule",
         "\t" + r"\textbf{Configuration} & Full-Training & Layer-Wise Subset & Global Subset (1P) & Global Subset (2P) \\", r"\midrule"]
    for label in labels:
        combos = bd[tag][label]
        cells = [f"{combos[m]['peak_memory_gb']:.1f}" if combos.get(m) else "OOM" for m in methods]
        L.append("\t" + cfg_math(label) + " & " + " & ".join(cells) + r" \\")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def score_cost(scoring_dir: Path, tag: str):
    p = scoring_dir / f"scoring_{tag}.json"
    if not p.exists():
        return None
    cfg = json.loads(p.read_text())["configs"]
    labels = [f"n=8 T={T} m=1" for T in T_SWEEP] + [f"n=8 T=512 m={m}" for m in M_SWEEP]
    rows = [("Direct", "direct"), ("GIP", "gip"), ("PIP", "pip")]
    nt, nm = len(T_SWEEP), len(M_SWEEP)
    L = [rf"\begin{{tabular}}{{l {'r' * nt} {'r' * nm}}}", r"\toprule",
         "\t& " + rf"\multicolumn{{{nt}}}{{c}}{{sweep \(\seqlen\) (\(m=1\))}} & \multicolumn{{{nm}}}{{c}}{{sweep \(m\) (\(\seqlen=512\))}} \\",
         rf"\cmidrule(lr){{2-{1 + nt}}}\cmidrule(lr){{{2 + nt}-{1 + nt + nm}}}",
         "\t" + r"\textbf{Scoring} & " + " & ".join(str(T) for T in T_SWEEP) + " & " + " & ".join(str(m) for m in M_SWEEP) + r" \\", r"\midrule"]
    cells, comp = {name: [] for name, _ in rows}, []
    for label in labels:
        vals = {k: (cfg.get(label, {}).get(k, {}) or {}).get("total_ms") for _, k in rows}
        exact = {k: v for k, v in vals.items() if v is not None}
        best = min(exact, key=exact.get) if exact else None
        for name, k in rows:
            s = "OOM" if vals[k] is None else f"{vals[k]:.0f}"
            cells[name].append(rf"\textbf{{{s}}}" if k == best else s)
        c = (cfg.get(label, {}).get("compress", {}) or {}).get("total_ms")
        comp.append("OOM" if c is None else f"{c:.0f}")
    L += [f"\t{name} & " + " & ".join(cells[name]) + r" \\" for name, _ in rows]
    return "\n".join(L + [r"\midrule", "\tCompressed & " + " & ".join(comp) + r" \\", r"\bottomrule", r"\end{tabular}"]) + "\n"


def main() -> int:
    global TOTALS
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--benchmark-dir", default=str(BENCH),
                    help="directory with breakdown/, breakdown_checkpointing/ and scoring/ (default SFT/benchmark/results/paper/h200)")
    ap.add_argument("--model-tag", default=MODEL_TAG, help=f"json tag of the model (default {MODEL_TAG}; e.g. qwen3-14b, llama-3.2-3b, smollm2-360m)")
    ap.add_argument("--subdir", default="breakdown", choices=["breakdown", "breakdown_checkpointing"],
                    help="results subdirectory: breakdown (no activation checkpointing, default) or breakdown_checkpointing")
    ap.add_argument("--scoring", default="gip", choices=["compress", "pip", "gip", "direct"],
                    help="scoring backend of the curated columns (paper: gip; others print to stdout)")
    ap.add_argument("--totals", default=TOTALS, choices=["plain", "phases"], help="Total row / overhead from plain_step_ms (default) or from the phase sums")
    args = ap.parse_args()
    TOTALS = args.totals
    se.SCORING = args.scoring
    if args.scoring != "gip":
        args.stdout = True   # the paper tables are the exact-GIP grids; other backends are diagnostics
    bench = Path(args.benchmark_dir)
    bd = se.load_breakdown(bench / args.subdir)
    labels = configs_of(bd, args.model_tag)
    if not labels:
        sys.exit(f"no breakdown results for {args.model_tag} under {bench / args.subdir}")
    files = {"system_overhead_h200.tex": system_overhead(bd, args.model_tag, labels),
             "peak_memory_h200.tex": peak_memory(bd, args.model_tag, labels),
             "score_cost_h200.tex": score_cost(bench / "scoring", args.model_tag)}
    missing = [k for k, v in files.items() if v is None]
    for k in missing:
        print(f"warning: no benchmark results for {k}, skipped", file=sys.stderr)
        del files[k]
    root = bench.relative_to(REPO) if bench.is_relative_to(REPO) else bench
    ckpt = "activation checkpointing" if args.subdir == "breakdown_checkpointing" else "no activation checkpointing"
    totals = "totals from the unpatched step" if TOTALS == "plain" else "totals from the phase sums"
    rc = emit(files, args, __file__, root,
              note=f"H200 suite, {args.model_tag}, CuTe backend with the Hopper TMA+wgmma kernels, bf16, fused AdamW, {ckpt}, exact GIP scoring, {totals}")
    return rc or (1 if missing else 0)


if __name__ == "__main__":
    sys.exit(main())
