"""Paper tables: sft_qwen_main.tex (tab:SFT-qwen), sft_qwen_nval_if.tex (adxtab:sft-qwen-nval-if), sft_qwen_nval_math.tex
(adxtab:sft-qwen-nval-math). k_table() builds k-only tables at 128 target examples (the 128 block of the nval tables, plus the
IFEval-loose column) for ad-hoc use; they are not paper tables.

Experiments run on : 8x H200 nodes
Launcher           : SFT/train/train.sh -c configs/<setting> -m <arm> --runs_root <results root>/SFT/runs_v2 ; SFT/eval/eval.sh --target <target>
                     settings dolci_inst_if, dolci_mixed_if, tulu3_if, dolci_reason_mathpersona, dolci_mixed_mathpersona, tulu3_mathpersona,
                     dolci_reason_mathref128_gen32b; arms FullTraining-Full-eot, {Global,LayerWise}Subset-Full[-f25|-f75]-eot; --n_val 128/64/32/16
Results read from  : <results root>/SFT/runs_v2/<pool>_<target>-Qwen3-1.7B-Base-<arm>-p1.0-lr1.00e-05-b8-v<n_val>-s<seed>/
                     {ifeval,ifbench,math500,gsm8k}_results.greedy.json
Seeds / statistics : 42, 2, 22, 62, 82; mean +- SE (population std / sqrt n); best per column bold (ties at one decimal all bold)
Usage              : python SFT/tables/qwen_capability.py [--results-root DIR] [--paper-root DIR] [--venues ICLR] [--check]

Layouts. The main table (sft_qwen_main) stacks two blocks in ONE tabular on a shared 24-column grid: the Precise-IF settings
(IFEval prompt-level strict, IFBench) on top and the Math settings (MATH500, GSM8K) below; rows Full-Training / Global Subset /
Layer-Wise Subset at k = n/2 = 4 of the n = 8 candidates and 32 target examples (Full-Training does not depend on the target set).
The appendix tables sft_qwen_nval_if / _math have rows = arms x k (2/4/6; Full-Training = 8) and columns = setting x metric
(IFEval strict, IFBench; MATH500, GSM8K), with one block per target size (128/64/32/16), Full-Training once at the top; bold = best
curated arm per column within a block. Labels spanning several rows are centred with multirow (mrow); a cmidrule separates the Global and
Layer-Wise groups inside each target-size block. k is printed as a count and the target-set size in words (no D* symbol).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import add_common_args, best_mask, emit, fmt_cell, mrow, results_root  # noqa: E402

MODEL, LR, SEEDS = "Qwen3-1.7B-Base", "1.00e-05", [42, 2, 22, 62, 82]
NVALS = [128, 64, 32, 16]          # target-set sizes of the robustness grid
NVAL_DEFAULT = 128                 # target-set size of the appendix k tables (and of every Full-Training run)
MAIN_K_SUFFIX, MAIN_NVAL = "", 32  # main table: k = n/2 (run-dir suffix "") and 32 target examples
FULL_ARM = "FullTraining-Full-eot"
K = [("f25", "2"), ("", "4"), ("f75", "6")]   # (run-dir suffix, k as a count of the n = 8 candidates)
METHODS = [("GlobalSubset", "Global Subset"), ("LayerWiseSubset", "Layer-Wise Subset")]

# metric = (column label, result file, JSON key)
IFEVAL = (r"\texttt{IFEval}", "ifeval_results.greedy.json", "prompt_level_strict_acc")
IFEVAL_LOOSE = (r"\texttt{IFEval} (loose)", "ifeval_results.greedy.json", "prompt_level_loose_acc")
IFBENCH = (r"\texttt{IFBench}", "ifbench_results.greedy.json", "prompt_level_strict_acc")
MATH500 = (r"\texttt{MATH500}", "math500_results.greedy.json", "accuracy")
GSM8K = (r"\texttt{GSM8K}", "gsm8k_results.greedy.json", "accuracy")
IF_M, IF_MAIN_M, MATH_M = [IFEVAL, IFEVAL_LOOSE, IFBENCH], [IFEVAL, IFBENCH], [MATH500, GSM8K]

# setting = (key, pool label, target label, run-dir prefix of the curated arms, run-dir prefix of Full-Training, metrics).
# Full-Training ignores the target set, so the curated-target setting shares the persona Full-Training runs.
IF_TARGET, MATH_TARGET = r"Precise IF (\texttt{Dolci})", r"Math (\texttt{Dolci})"
IF_SETTINGS = [
    ("dolci_inst_if", r"\texttt{Dolci} instruction", IF_TARGET, "dolci_instruction_precise_if", "dolci_instruction_precise_if", IF_M),
    ("dolci_mixed_if", r"\texttt{Dolci} mixed", IF_TARGET, "dolci_mixed_precise_if", "dolci_mixed_precise_if", IF_M),
    ("tulu3_if", r"\texttt{Tulu 3}", IF_TARGET, "tulu3_general_precise_if", "tulu3_general_precise_if", IF_M),
]
MATH_SETTINGS = [
    ("dolci_reason_math", r"\texttt{Dolci} reasoning", MATH_TARGET, "dolci_reasoning_math_persona", "dolci_reasoning_math_persona", MATH_M),
    ("dolci_mixed_math", r"\texttt{Dolci} mixed", MATH_TARGET, "dolci_mixed_math_persona", "dolci_mixed_math_persona", MATH_M),
    ("tulu3_math", r"\texttt{Tulu 3}", MATH_TARGET, "tulu3_general_math_persona", "tulu3_general_math_persona", MATH_M),
    ("mathgen_32b", r"\texttt{Dolci} reasoning", "Math (curated)", "dolci_reasoning_math_ref128_gen32b", "dolci_reasoning_math_persona", MATH_M),
]
SETTINGS = IF_SETTINGS + MATH_SETTINGS


# --------------------------------------------------------------------------------------------------------------- reading
class Runs:
    """Reads one metric of one arm over all seeds from <results root>/SFT/runs_v2."""

    def __init__(self, root: Path):
        self.dir = root / "SFT" / "runs_v2"

    def scores(self, prefix: str, arm: str, nval: int, metric) -> dict[int, float]:
        _, fn, key = metric
        out = {}
        for seed in SEEDS:
            f = self.dir / f"{prefix}-{MODEL}-{arm}-p1.0-lr{LR}-b8-v{nval}-s{seed}" / fn
            if f.exists():
                v = json.loads(f.read_text()).get(key)
                if v is not None:
                    out[seed] = float(v)
        return out

    def full(self, setting, metric):
        return self.scores(setting[4], FULL_ARM, NVAL_DEFAULT, metric)

    def curated(self, setting, method: str, k_suffix: str, nval: int, metric):
        arm = f"{method}-Full" + (f"-{k_suffix}" if k_suffix else "") + "-eot"
        return self.scores(setting[3], arm, nval, metric)


def completeness_report(runs: Runs) -> None:
    """One line per setting: seeds found for Full-Training and the least complete curated arm (primary metric, 128 targets)."""
    for s in SETTINGS:
        primary = s[5][0]
        n_full = len(runs.full(s, primary))
        n_cur = min(len(runs.curated(s, m, suf, NVAL_DEFAULT, primary)) for m, _ in METHODS for suf, _ in K)
        print(f"{s[0]:18s} Full n={n_full}  curated min n={n_cur}")


def header_cmidrules(first_col: int, nset: int, span: int) -> str:
    return " ".join(f"\\cmidrule(lr){{{first_col + i * span}-{first_col - 1 + (i + 1) * span}}}" for i in range(nset))


# ---------------------------------------------------------------------------------------------------------- main table
def main_table(runs: Runs) -> str:
    """Two stacked blocks (IF settings / Math settings) in one tabular on a 24-column grid, so the blocks have equal width
    and aligned edges: an IF setting spans 8 grid columns (2 metrics x 4), a Math setting 6 (2 metrics x 3). Every cell is
    an exact multiple of the grid unit W (the widest header label divided by its span, +1pt) so that TeX's span allocation
    closes on a uniform grid; the natural width ~590pt is scaled by adjustbox to the line width. Curated arms at
    MAIN_K_SUFFIX / MAIN_NVAL, Full-Training at 128 targets (independent of the target set)."""
    grid, unit = 24, 21.5
    arms = [("Full-Training", lambda s, m: runs.full(s, m))]
    arms += [(lab, lambda s, m, meth=meth: runs.curated(s, meth, MAIN_K_SUFFIX, MAIN_NVAL, m)) for meth, lab in METHODS]

    def mc(n: int, txt: str) -> str:
        return f"\\multicolumn{{{n}}}{{@{{}}c@{{}}}}{{\\makebox[{n * unit:g}pt]{{{txt}}}}}"

    def block(settings, metrics) -> list[str]:
        nset, nm = len(settings), len(metrics)
        gs = grid // nset
        gm = gs // nm
        assert gs * nset == grid and gm * nm == gs, (nset, nm)
        lines = [mrow(2, r"\textbf{Method}", rule=True) + " & "
                 + " & ".join(mc(gs, f"{pool}/{tgt}") for _, pool, tgt, *_ in settings) + r" \\",
                 header_cmidrules(2, nset, gs),
                 " & " + " & ".join(mc(gm, m[0]) for _ in settings for m in metrics) + r" \\",
                 r"\midrule"]
        cols = [(s, m) for s in settings for m in metrics]
        vals = [[get(s, m) for s, m in cols] for _, get in arms]
        best = [best_mask([vals[a][c] for a in range(len(arms))]) for c in range(len(cols))]
        for a, (lab, _) in enumerate(arms):
            if lab == "Layer-Wise Subset":
                lines.append(r"\midrule")
            lines.append(f"{lab} & " + " & ".join(mc(gm, fmt_cell(vals[a][c], best[c][a])) for c in range(len(cols))) + r" \\")
        return lines

    pre = r"\begin{tabular}{@{}l@{\hspace{\tabcolsep}}*{%d}{@{}>{\centering\arraybackslash}p{%gpt}}@{}}" % (grid, unit)
    lines = ([r"\begin{adjustbox}{max width=\linewidth}", pre, r"\toprule"] + block(IF_SETTINGS, IF_MAIN_M)
             + [r"\bottomrule", r"\noalign{\vskip 3pt}", r"\toprule"] + block(MATH_SETTINGS, MATH_M)
             + [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"])
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------ appendix tables
def k_table(runs: Runs, settings, metrics) -> str:
    """Rows = Full-Training (k = 8) and the curated arms x k, columns = setting x metric, 128 target examples."""
    ncol = len(settings) * len(metrics)
    heads = [f"\\multicolumn{{{len(metrics)}}}{{c}}{{{pool}/{tgt}}}" for _, pool, tgt, *_ in settings]
    lines = [r"\begin{tabular}{l c " + " ".join(["c"] * ncol) + "}", r"\toprule",
             mrow(2, r"\textbf{Method}", rule=True) + " & " + mrow(2, r"\(k\)", rule=True) + " & " + " & ".join(heads) + r" \\",
             header_cmidrules(3, len(settings), len(metrics)),
             r" & & " + " & ".join(m[0] for _ in settings for m in metrics) + r" \\",
             r"\midrule"]
    rows = [("Full-Training", "8", lambda s, m: runs.full(s, m))]
    rows += [(lab, kk, lambda s, m, meth=meth, suf=suf: runs.curated(s, meth, suf, NVAL_DEFAULT, m))
             for meth, lab in METHODS for suf, kk in K]
    cols = [(s, m) for s in settings for m in metrics]
    vals = [[get(s, m) for s, m in cols] for _, _, get in rows]
    best = [best_mask([vals[r][c] for r in range(len(rows))]) for c in range(len(cols))]
    nrows = {lab: sum(1 for l, _, _ in rows if l == lab) for lab, _, _ in rows}
    prev = None
    for r, (lab, kk, _) in enumerate(rows):
        if prev is not None and lab != prev:
            lines.append(r"\midrule")
        show = "" if lab == prev else (mrow(nrows[lab], lab) if nrows[lab] > 1 else lab)
        lines.append(f"{show} & {kk} & " + " & ".join(fmt_cell(vals[r][c], best[c][r]) for c in range(len(cols))) + r" \\")
        prev = lab
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def nval_table(runs: Runs, settings, metrics) -> str:
    """Target-size robustness: the k_table layout with one block per target size (Full-Training once at the top);
    bold = best curated arm per column within a block."""
    ncol = len(settings) * len(metrics)
    heads = [f"\\multicolumn{{{len(metrics)}}}{{c}}{{{pool}/{tgt}}}" for _, pool, tgt, *_ in settings]
    top = [mrow(2, r"\textbf{Target size}", rule=True), mrow(2, r"\textbf{Method}", rule=True), mrow(2, r"\(k\)", rule=True)]
    lines = [r"\begin{tabular}{l l c " + " ".join(["c"] * ncol) + "}", r"\toprule",
             " & ".join(top) + " & " + " & ".join(heads) + r" \\",
             header_cmidrules(4, len(settings), len(metrics)),
             r" & & & " + " & ".join(m[0] for _ in settings for m in metrics) + r" \\",
             r"\midrule"]
    cols = [(s, m) for s in settings for m in metrics]
    lines.append(r"-- & Full-Training & 8 & " + " & ".join(fmt_cell(runs.full(s, m)) for s, m in cols) + r" \\")
    arms = [(meth, lab, suf, kk) for meth, lab in METHODS for suf, kk in K]
    for nval in NVALS:
        lines.append(r"\midrule")
        vals = [[runs.curated(s, meth, suf, nval, m) for s, m in cols] for meth, _, suf, _ in arms]
        best = [best_mask([vals[a][c] for a in range(len(arms))]) for c in range(len(cols))]
        prev = None
        for a, (_, lab, _, kk) in enumerate(arms):
            if a and lab != prev:
                # light rule between the Global and Layer-Wise groups, spanning Method .. last column so that the
                # target-size column (its label is centred over all 6 rows, i.e. at this rule's level) stays open
                lines.append(f"\\cmidrule(lr){{2-{3 + ncol}}}")
            size = mrow(len(arms), f"\\({nval}\\)", rule=True) if a == 0 else ""
            show = "" if lab == prev else mrow(len(K), lab)
            cells = " & ".join(fmt_cell(vals[a][c], best[c][a]) for c in range(len(cols)))
            lines.append(f"{size} & {show} & {kk} & {cells} \\\\")
            prev = lab
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    args = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0])).parse_args()
    root = results_root(args.results_root)
    runs = Runs(root)
    completeness_report(runs)
    files = {
        "sft_qwen_main.tex": main_table(runs),
        "sft_qwen_nval_if.tex": nval_table(runs, IF_SETTINGS, IF_MAIN_M),
        "sft_qwen_nval_math.tex": nval_table(runs, MATH_SETTINGS, MATH_M),
    }
    return emit(files, args, __file__, root)


if __name__ == "__main__":
    sys.exit(main())
