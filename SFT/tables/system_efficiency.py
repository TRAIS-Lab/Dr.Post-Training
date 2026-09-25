"""Paper tables: system_overhead.tex (tab:system-overhead), score_cost.tex (adxtab:score-cost), peak_memory.tex (adxtab:peak-memory),
timing_grid_{smollm2,tinyllama,llama3b}_{n8,n2}.tex (adxtab:timing-grid), timing_grid_ckpt_*_{n8,n2}.tex (adxtab:timing-grid-ckpt).

Experiments run on : one node with 4x A40 (repeated runs of the suite agree within 1 %)
Launcher           : SFT/benchmark/slurm/a40_paper_suite.sbatch -> benchmark_run.py (per-component step timing, with and without
                     activation checkpointing; CuTe fused kernels, every scoring backend, m = 1, k = n/2) and benchmark_scoring.py
                     (standalone scoring cost on synthetic tensors, T- and m-sweeps); models SmolLM2-360M, TinyLlama-1.1B, Llama-3.2-3B
Results read from  : SFT/benchmark/results/paper/breakdown/<tag>_n<n>_T<T>_m1.json, .../breakdown_checkpointing/ (same names),
                     .../scoring/scoring_<tag>.json; kept locally under the repo (gitignored, 13 MB), so --results-root is not used by this generator.
                     Llama-3.2-3B also has <tag>_n<n>_T<T>_m1_full_training_cublaslt.json (Full-Training with cuBLASLt), quoted as a comment.
Seeds / statistics : single timed run per cell (median step of the benchmark's repetitions, see benchmark.py); ms per step at one decimal,
                     totals rounded with the overhead over Full-Training in parentheses; peak memory in GB; scoring cost in ms with the
                     cheapest exact backend per column in bold
Usage              : python SFT/tables/system_efficiency.py [--benchmark-dir DIR] [--paper-root DIR] [--venues ICLR] [--check]
                     Paper tables use exact GIP scoring (the backend of the QA accuracy runs). --scoring compress|pip|direct prints the same grids for the other backends to stdout (diagnostic, not paper tables)

Components follow the paper: a.grad = activation gradient, scoring = compress + score + select (+ retain for one-pass), w.grad = selected
weight gradient (one-pass: post-hoc assembly), autograd = the remainder of the backward pass. For Global Subset (1P) the Backward row
includes the post-hoc assembly and the global selection, so that Forward + Backward + Optimizer = Total for every column.
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
MODELS = [("smollm2-360m", r"\textsc{SmolLM2-360M}", "smollm2"), ("tinyllama-1.1b", r"\textsc{TinyLlama-1.1B}", "tinyllama"),
          ("llama-3.2-3b", r"\textsc{Llama-3.2-3B}", "llama3b")]   # (json tag, display, file short)
CONFIGS = [("n=8 T=512 m=1", "n8"), ("n=2 T=1024 m=1", "n2")]
SCORING = "gip"
NAME_RE = re.compile(r"^(?P<tag>.+?)_n(?P<n>\d+)_T(?P<T>\d+)_m(?P<m>\d+)\.json$")
ROWS = [("forward", r"\textbf{Forward}", False), ("backward", r"\textbf{Backward}", False), ("act", r"\quad \texttt{a.grad}", True),
        ("scoring", r"\quad \texttt{scoring}", True), ("wgrad", r"\quad \texttt{w.grad}", True), ("autograd", r"\quad \texttt{autograd}", True),
        ("optimizer", r"\textbf{Optimizer}", False)]


# --------------------------------------------------------------------------------------------------------------- loading
def load_breakdown(d: Path) -> dict:
    """{tag: {config label: {method/scoring: result}}} from the per-config JSONs of benchmark_run.py."""
    out: dict = {}
    for p in sorted(d.glob("*.json")):
        m = NAME_RE.match(p.name)
        if m:
            for label, combos in json.loads(p.read_text()).get("results", {}).items():
                out.setdefault(m["tag"], {})[label] = combos
    return out


def load_extra(d: Path, tag: str, label: str, suffix: str):
    n, T = re.match(r"n=(\d+) T=(\d+)", label).groups()
    p = d / f"{tag}_n{n}_T{T}_m1_{suffix}.json"
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------------------------------------------------ components
def g(r, *keys):
    return sum((r.get(k) or 0.0) for k in keys)


def total_ms(method, r):
    if method in ("full_training", "layer_wise_subset"):
        return g(r, "forward", "backward", "optimizer")
    if method == "global_subset_one_pass":
        return g(r, "forward", "backward", "selection", "wgrad", "optimizer")
    if method == "global_subset":
        return g(r, "pass1_forward", "pass1_backward", "selection", "pass2_forward", "pass2_backward", "optimizer")
    raise ValueError(method)


def column(method, r):
    """Row values forward / backward / act / scoring / wgrad / autograd / optimizer (None -> ---)."""
    if method == "full_training":
        act, wg = g(r, "act_grad"), g(r, "wgrad")
        return dict(forward=r["forward"], backward=r["backward"], act=act, scoring=None, wgrad=wg, autograd=r["backward"] - act - wg, optimizer=r["optimizer"])
    if method == "layer_wise_subset":
        act, sc, wg = g(r, "act_grad"), g(r, "compress", "score", "select", "emb_score", "emb_select", "select_wgrad"), g(r, "wgrad", "emb_wgrad")
        return dict(forward=r["forward"], backward=r["backward"], act=act, scoring=sc, wgrad=wg, autograd=r["backward"] - act - sc - wg, optimizer=r["optimizer"])
    if method == "global_subset_one_pass":
        act, sc, asm = g(r, "act_grad"), g(r, "compress", "score", "retain", "emb_score", "emb_retain"), g(r, "wgrad")
        return dict(forward=r["forward"], backward=r["backward"] + g(r, "selection") + asm, act=act, scoring=sc, wgrad=asm,
                    autograd=r["backward"] - act - sc, optimizer=r["optimizer"])
    if method == "global_subset_score":   # two-pass, pass 1
        act, sc = g(r, "p1_act_grad"), g(r, "p1_compress", "p1_score", "p1_emb_score")
        return dict(forward=r["pass1_forward"], backward=r["pass1_backward"] + g(r, "selection"), act=act, scoring=sc, wgrad=None,
                    autograd=r["pass1_backward"] - act - sc, optimizer=None)
    if method == "global_subset_grad":    # two-pass, pass 2
        act, wg = g(r, "p2_act_grad"), g(r, "p2_wgrad")
        return dict(forward=r["pass2_forward"], backward=r["pass2_backward"], act=act, scoring=None, wgrad=wg, autograd=r["pass2_backward"] - act - wg, optimizer=r["optimizer"])
    raise ValueError(method)


def f1(v):
    return "---" if v is None else f"{v:.1f}"


def tot(v, base=None, pct="{:+.0f}"):
    s = rf"\textbf{{{v:.0f}}}"
    return s + (rf" \scriptsize({pct.format(100 * (v / base - 1))}\%)" if base is not None else "")


def tabular(head, cols, totals):
    L = list(head) + [r"\midrule"]
    for key, name, shaded in ROWS:
        if shaded:
            L.append(r"\rowcolor{black!6}")
        L.append("\t" + name + " & " + " & ".join(f1(c[key]) for c in cols) + r" \\")
    L += [r"\midrule", "\t" + r"\textbf{Total} & " + " & ".join(totals) + r" \\", r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------------------------ tables
def system_overhead(bd: dict, bench: Path) -> str:
    label = CONFIGS[0][0]
    head = [r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
            "\t& " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{disp}}}" for _, disp, _ in MODELS) + r" \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
            "\t" + r"\textbf{Component} & " + " & ".join(["Full-Training & Layer-Wise Subset & Global Subset"] * 3) + r" \\"]
    cols, totals, notes = [], [], []
    for tag, _, _ in MODELS:
        combos = bd.get(tag, {}).get(label, {})
        full, lw, op = combos.get("full_training/pip"), combos.get(f"layer_wise_subset/{SCORING}"), combos.get(f"global_subset_one_pass/{SCORING}")
        if not (full and lw and op):
            sys.exit(f"missing {label} results for {tag}: have {sorted(k for k, v in combos.items() if v)}")
        base = total_ms("full_training", full)
        cols += [column("full_training", full), column("layer_wise_subset", lw), column("global_subset_one_pass", op)]
        totals += [tot(base), tot(total_ms("layer_wise_subset", lw), base), tot(total_ms("global_subset_one_pass", op), base)]
        lt = load_extra(bench / "breakdown", tag, label, "full_training_cublaslt")
        if lt:
            notes.append(f"% {tag}: Full-Training with cuBLASLt = {total_ms('full_training', lt):.0f} ms (w.grad {g(lt, 'wgrad'):.1f} ms) vs {base:.0f} ms default cuBLAS")
    return "\n".join(notes + [""]) * bool(notes) + tabular(head, cols, totals)


def timing_grid(bd: dict, tag: str, label: str):
    combos = bd.get(tag, {}).get(label, {})
    full, lw = combos.get("full_training/pip"), combos.get(f"layer_wise_subset/{SCORING}")
    op, tp = combos.get(f"global_subset_one_pass/{SCORING}"), combos.get(f"global_subset/{SCORING}")
    if not (full and lw and op and tp):
        return None
    base = total_ms("full_training", full)
    head = [r"\begin{tabular}{l ccc cc}", r"\toprule",
            "\t" + mrow(2, r"\textbf{Component}") + " & " + mrow(2, r"\textbf{Full-Training}") + " & " + mrow(2, r"\textbf{Layer-Wise Subset}")
            + " & " + mrow(2, r"\textbf{Global Subset (1P)}") + r" & \multicolumn{2}{c}{\textbf{Global Subset (2P)}} \\",
            r"\cmidrule(lr){5-6}", "\t" + r"& & & & Score & Grad \\"]
    cols = [column("full_training", full), column("layer_wise_subset", lw), column("global_subset_one_pass", op), column("global_subset_score", tp), column("global_subset_grad", tp)]
    totals = [tot(base), tot(total_ms("layer_wise_subset", lw), base, "{:+.1f}"), tot(total_ms("global_subset_one_pass", op), base, "{:+.1f}"),
              rf"\multicolumn{{2}}{{c}}{{{tot(total_ms('global_subset', tp), base, '{:+.1f}')}}}"]
    return tabular(head, cols, totals)


def peak_memory(bd: dict, bd_ckpt: dict) -> str:
    L = [r"\begin{tabular}{cc cccc cccc}", r"\toprule",
         "\t& & " + r"\multicolumn{4}{c}{\(n=8,\; T=512\)} & \multicolumn{4}{c}{\(n=2,\; T=1024\)} \\", r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
         "\t" + r"\textbf{Model} & \textbf{Setting} & " + " & ".join(["Full-Training & Layer-Wise Subset & Global Subset (1P) & Global Subset (2P)"] * 2) + r" \\", r"\midrule"]
    methods = ["full_training/pip", f"layer_wise_subset/{SCORING}", f"global_subset_one_pass/{SCORING}", f"global_subset/{SCORING}"]
    for i, (tag, disp, _) in enumerate(MODELS):
        for setting, src in (("No ckpt", bd), ("With ckpt", bd_ckpt)):
            cells = []
            for label, _ in CONFIGS:
                combos = src.get(tag, {}).get(label, {})
                cells += [f"{combos[k]['peak_memory_gb']:.1f}" if combos.get(k) else "---" for k in methods]
            L.append("\t" + (mrow(2, disp) if setting == "No ckpt" else "") + f" & {setting} & " + " & ".join(cells) + r" \\")
        if i < len(MODELS) - 1:
            L.append(r"\midrule")
    return "\n".join(L + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def score_cost(scoring_dir: Path):
    """T-sweep (512, 2048, 8192; m=1) and m-sweep (1, 4, 8; T=512): Direct / GIP / PIP (cheapest exact bold) and Compressed."""
    data = {tag: json.loads((scoring_dir / f"scoring_{tag}.json").read_text())["configs"] for tag, _, _ in MODELS if (scoring_dir / f"scoring_{tag}.json").exists()}
    if not data:
        return None
    labels = ["n=8 T=512 m=1", "n=8 T=2048 m=1", "n=8 T=8192 m=1", "n=8 T=512 m=1", "n=8 T=512 m=4", "n=8 T=512 m=8"]
    rows = [("Direct", "direct"), ("GIP", "gip"), ("PIP", "pip")]
    L = [r"\begin{tabular}{l rrr rrr rrr rrr rrr rrr}", r"\toprule",
         "\t& " + " & ".join(rf"\multicolumn{{6}}{{c}}{{{disp}}}" for _, disp, _ in MODELS) + r" \\", r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(lr){14-19}",
         "\t& " + " & ".join([r"\multicolumn{3}{c}{sweep \(\seqlen\)} & \multicolumn{3}{c}{sweep \(m\)}"] * 3) + r" \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}\cmidrule(lr){17-19}",
         "\t" + r"\textbf{Scoring} & " + " & ".join(["512 & 2048 & 8192 & 1 & 4 & 8"] * 3) + r" \\", r"\midrule"]
    cells, comp = {name: [] for name, _ in rows}, []
    for tag, _, _ in MODELS:
        cfg = data.get(tag, {})
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


# -------------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--benchmark-dir", default=str(BENCH), help="directory with breakdown/, breakdown_checkpointing/ and scoring/ (default SFT/benchmark/results/paper)")
    ap.add_argument("--scoring", default="gip", choices=["compress", "pip", "gip", "direct"], help="scoring backend of the curated columns (paper: gip; others print to stdout)")
    args = ap.parse_args()
    global SCORING
    SCORING = args.scoring
    if SCORING != "gip":
        args.stdout = True   # the paper tables are the exact-GIP grids; other backends are diagnostics
    bench = Path(args.benchmark_dir)
    bd, bd_ckpt = load_breakdown(bench / "breakdown"), load_breakdown(bench / "breakdown_checkpointing")
    files = {"system_overhead.tex": system_overhead(bd, bench), "peak_memory.tex": peak_memory(bd, bd_ckpt), "score_cost.tex": score_cost(bench / "scoring")}
    for tag, _, short in MODELS:
        for label, cfg in CONFIGS:
            files[f"timing_grid_{short}_{cfg}.tex"] = timing_grid(bd, tag, label)
            files[f"timing_grid_ckpt_{short}_{cfg}.tex"] = timing_grid(bd_ckpt, tag, label)
    missing = [k for k, v in files.items() if v is None]
    if missing:
        sys.exit(f"missing benchmark results for {missing}")
    return emit(files, args, __file__, bench.relative_to(REPO) if bench.is_relative_to(REPO) else bench, note="A40 suite, CuTe kernels, exact GIP scoring")


if __name__ == "__main__":
    sys.exit(main())
