"""Paper tables: sft_downstream.tex (tab:SFT-downstream), sft_qa_ppl.tex (adxtab:sft-qa-ppl), sft_qa_matrix_lora.tex /
sft_qa_matrix_full.tex / sft_qa_matrix_meso.tex (adxtab:sft-qa-matrix-lora / -full / -meso), sft_ablation.tex (adxtab:sft-ablation).

Experiments run on : one node with 4x A40
Launcher           : SFT/train/qa_matrix_task.sh <setting> <yaml> <seed> (train.sh, then eval.sh --batch_size 64 --n_test 500) over the
                     manifests of SFT/train/qa_matrix_manifest.py; Target-Only rows: SFT/train/qa_target_only_task.sh (train_val_ablation.sh);
                     settings alpaca_samsum, less_tydiqa, triviaqa_nq, less_squad; arms FullTraining-<FT>, <Method>-<FT>[-gip|-cmp|-random|-lnorm]
Results read from  : <results root>/SFT/runs_v2/<pool>_<target>-Llama-3.2-1B-<yaml>-p<pct>-lr<lr>-b8-v16-s<seed>/{<target>_results.json,
                     evaluation_results.json} and <target>_val_<target>-Llama-3.2-1B-FullTraining-<FT>-ms<steps>-lr<lr>-b8-v16-s<seed>/ (Target-Only, full budget)
                     and target_only_ms50/<the same>-ms50-... (Target-Only, early-stopped: the rows of the tables).
                     --scan reads them and rewrites the local per-run export SFT/tables/data/qa_runs.csv (gitignored); without --scan
                     the tables are built from that export, so they can be regenerated on a machine without the run dirs.
Seeds / statistics : 42, 2, 22, 62, 82; downstream metric at the final checkpoint (Rouge-L x100 for samsum, F1 % otherwise); mean +- SE
                     (population std / sqrt n); a cell with fewer than 5 seeds is printed as --; best per column bold (ties at one decimal
                     all bold). Perplexity = final evaluation perplexity on the 500 held-out rows, mean +- SE over seeds at two decimals,
                     lowest per row bold.
Usage              : python SFT/tables/qa_downstream.py [--scan] [--report] [--results-root DIR] [--paper-root DIR] [--venues ICLR] [--check]

Backends: exact = GIP, compressed = normal-64*64 (k = 4 of n = 8, 16 target examples, m = 1 target example per step). Layouts:
sft_downstream = the three main update rules under LoRA with exact scoring + Target-Only; sft_qa_matrix_<ft> = five update rules (+ Target-Only) x setting x
{Exact, Compressed}; sft_ablation = Full-Training, the two Random Subset controls, Global Subset, its layer-normalized variant and
Layer-Wise Subset (all curated rows compressed, the backend of the layer-normalized control) x setting x fine-tuning method;
sft_qa_ppl = Full-Training / Global (exact) / Layer-Wise (exact) / Target-Only, final evaluation perplexity, rows = setting x fine-tuning.
Target-Only (no pool, no scoring): in Table 2 (sft_downstream) it is a plain fourth row that competes for bold like every other row; in the
appendix tables it is listed after a rule and bold = best pool-based arm. Table 2 shows its early-stopped runs (50 steps, runs_v2/target_only_ms50, SFT/train/qa_target_only_ms50_task.sh; csv variant ms50), the appendix
tables show both the same-budget runs (csv variant "") and the early-stopped ones.
--report prints the markdown matrix, the Target-Only perplexity trajectory (minimum vs final) and seconds per step by backend.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import REPO, add_common_args, best_mask, emit, fmt_cell, mean_se, mrow, results_root  # noqa: E402

CSV = Path(__file__).resolve().parent / "data" / "qa_runs.csv"
CFG = REPO / "SFT" / "train" / "configs"
MODEL, SEEDS, MIN_SEEDS = "Llama-3.2-1B", [42, 2, 22, 62, 82], 5
SETTINGS = ["alpaca_samsum", "less_tydiqa", "triviaqa_nq", "less_squad"]
SETTING_OF_PREFIX = {"alpaca_samsum": "alpaca_samsum", "less_tydiqa": "less_tydiqa", "triviaqa_nq_open": "triviaqa_nq", "less_squad": "less_squad"}
TARGET = {"alpaca_samsum": "samsum", "less_tydiqa": "tydiqa", "triviaqa_nq": "nq_open", "less_squad": "squad"}
SETTING_OF_TASK = {v: k for k, v in TARGET.items()}
SETTING_TEX = {"alpaca_samsum": r"\texttt{alpaca}/\texttt{samsum}", "less_tydiqa": r"\texttt{less-mix}/\texttt{tydiqa}",
               "triviaqa_nq": r"\texttt{triviaqa}/\texttt{nq\_open}", "less_squad": r"\texttt{less-mix}/\texttt{squad}"}
METHODS = ["FullTraining", "GlobalSubset", "BlockWiseSubset", "SublayerWiseSubset", "LayerWiseSubset"]
TO_DIR, TO_VARIANT = "target_only_ms50", "ms50"   # early-stopped (50-step) Target-Only runs (csv variant ms50); the same-budget runs are variant ""
TO_ROWS = [("Target-Only (same budget)", "TargetOnly", "-", ""), ("Target-Only (early stop)", "TargetOnly", "-", TO_VARIANT)]   # appendix tables; Table 1: early stop only
LABEL = {"FullTraining": "Full-Training", "TargetOnly": "Target-Only", "GlobalSubset": "Global Subset", "BlockWiseSubset": "Block-Wise Subset",
         "SublayerWiseSubset": "Sublayer-Wise Subset", "LayerWiseSubset": "Layer-Wise Subset"}
FTS = ["Full", "LoRA", "MeSO"]
FT_LABEL = {"Full": "Full parameter", "LoRA": "LoRA", "MeSO": "MeSO"}
FT_SHORT = {"Full": "Full param.", "LoRA": "LoRA", "MeSO": "MeSO"}
EXACT, CMP = "gip", "cmp"
BACKEND_LABEL = {"gip": "Exact (GIP)", "cmp": "Compressed 64x64", "-": "-"}
RUN_RE = re.compile(r"^(?P<prefix>alpaca_samsum|less_tydiqa|triviaqa_nq_open|less_squad)-Llama-3\.2-1B-(?P<yaml>.+?)-p[\d.]+-lr[\d.e+-]+-b8-v16-s(?P<seed>\d+)$")
TO_RE = re.compile(r"^(?P<task>samsum|tydiqa|nq_open|squad)_val_(?P=task)-Llama-3\.2-1B-FullTraining-(?P<ft>Full|LoRA|MeSO)-ms\d+-lr[\d.e+-]+-b8-v16-s(?P<seed>\d+)$")
COLS = ["setting", "finetuning", "method", "scoring", "variant", "yaml", "seed", "steps", "final_eval_ppl", "min_eval_ppl", "min_ppl_step",
        "metric_name", "metric", "protocol", "train_wall_s", "run"]   # protocol: prompt_encoding of the result file ("bos" = the earlier encoding with a leading <|begin_of_text|>)


# --------------------------------------------------------------------------------------------------------------- scanning
def classify(setting: str, yaml_name: str):
    """(method family, fine-tuning, scoring backend, variant) of a run from its yaml; None if the yaml is not a matrix cell."""
    p = CFG / setting / f"{yaml_name}.yaml"
    if not p.exists():
        return None
    y = yaml.safe_load(p.read_text()) or {}
    d = yaml.safe_load((CFG / setting / "defaults.yaml").read_text()) or {}
    fam, ft = y.get("method"), y.get("finetuning", "Full")
    if fam not in METHODS or ft not in FTS:
        return None
    sc = y.get("scoring", d.get("scoring", {})) or {}
    variant, backend = "", {"compress": CMP, "pip": "pip", "gip": EXACT}.get(sc.get("method", "pip"), sc.get("method"))
    if y.get("selection_mode", d.get("selection_mode", "topk")) == "random":
        variant, backend = "random", "-"
    else:
        if backend == CMP and sc.get("compression", "") != "normal-64*64":
            backend = "cmp:" + str(sc.get("compression"))
        if y.get("score_normalization", "none") not in ("none", None):
            variant = "lnorm"
    if fam == "FullTraining":
        backend = "-"
    return fam, ft, backend, variant


def read_run(d: Path, setting: str):
    """Final-checkpoint metric, evaluation-perplexity curve summary and training wall time of one run dir; None if not evaluated."""
    tgt = TARGET[setting]
    rp = d / f"{tgt}_results.json"
    if not rp.exists():
        return None
    r = json.loads(rp.read_text())
    metric_name, metric = ("rougeL", 100 * r["rougeL"]) if tgt == "samsum" else ("f1", r["f1_score"])
    steps = ppl = wall = min_ppl = min_step = None
    ep = d / "evaluation_results.json"
    if ep.exists():
        ev = json.loads(ep.read_text())
        ev = ev if isinstance(ev, list) else ev.get("results", [])
        if ev:
            last = ev[-1]
            steps, ppl, wall = last.get("step"), last.get("eval_perplexity"), last.get("train_wall_time")
            pts = [(e["eval_perplexity"], e["step"]) for e in ev if e.get("eval_perplexity") is not None]
            if pts:
                min_ppl, min_step = min(pts)
    if wall is None and (d / "train.log").exists():
        m = re.findall(r"'train_runtime': ([\d.]+)", (d / "train.log").read_text(errors="ignore"))
        wall = float(m[-1]) if m else None
    return dict(steps=steps, final_eval_ppl=ppl, min_eval_ppl=min_ppl, min_ppl_step=min_step, metric_name=metric_name, metric=metric,
                protocol=r.get("prompt_encoding", "bos"), train_wall_s=wall, run=d.name)


def scan(runs: Path) -> list[dict]:
    rows = []
    for d in sorted((runs / TO_DIR).iterdir()) if (runs / TO_DIR).is_dir() else []:   # early-stopped Target-Only (50 steps)
        t = TO_RE.match(d.name)
        if t and d.is_dir():
            setting = SETTING_OF_TASK[t["task"]]
            r = read_run(d, setting)
            if r:
                rows.append(dict(setting=setting, finetuning=t["ft"], method="TargetOnly", scoring="-", variant=TO_VARIANT, yaml=f"FullTraining-{t['ft']}", seed=int(t["seed"]), **r))
    for d in sorted(runs.iterdir()):
        if not d.is_dir():
            continue
        t = TO_RE.match(d.name)
        if t:
            setting = SETTING_OF_TASK[t["task"]]
            r = read_run(d, setting)
            if r:
                rows.append(dict(setting=setting, finetuning=t["ft"], method="TargetOnly", scoring="-", variant="", yaml=f"FullTraining-{t['ft']}", seed=int(t["seed"]), **r))
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        setting = SETTING_OF_PREFIX[m["prefix"]]
        cls = classify(setting, m["yaml"])
        r = read_run(d, setting) if cls else None
        if r:
            fam, ft, backend, variant = cls
            rows.append(dict(setting=setting, finetuning=ft, method=fam, scoring=backend, variant=variant, yaml=m["yaml"], seed=int(m["seed"]), **r))
    return rows


def write_csv(rows: list[dict]) -> None:
    CSV.parent.mkdir(exist_ok=True)
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["setting"], r["finetuning"], r["method"], r["scoring"], r["variant"], r["seed"])):
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in COLS})


def load_csv() -> list[dict]:
    rows = []
    for r in csv.DictReader(CSV.open()):
        for k in ("seed", "steps", "min_ppl_step"):
            r[k] = int(float(r[k])) if r[k] else None
        for k in ("final_eval_ppl", "min_eval_ppl", "metric", "train_wall_s"):
            r[k] = float(r[k]) if r[k] else None
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------------------ aggregation
class Cells:
    """{seed: metric} and {seed: final perplexity} per (setting, fine-tuning, method, backend, variant)."""

    def __init__(self, rows):
        self.metric, self.ppl, self.rows = {}, {}, {}
        for r in rows:
            k = (r["setting"], r["finetuning"], r["method"], r["scoring"], r["variant"])
            self.metric.setdefault(k, {})[r["seed"]] = r["metric"]
            self.rows.setdefault(k, []).append(r)
            if r["final_eval_ppl"] is not None:
                self.ppl.setdefault(k, {})[r["seed"]] = r["final_eval_ppl"]

    def v(self, s, ft, m, b="-", var=""):
        return self.metric.get((s, ft, m, "-" if m in ("FullTraining", "TargetOnly") else b, var), {})

    def p(self, s, ft, m, b="-", var=""):
        return self.ppl.get((s, ft, m, "-" if m in ("FullTraining", "TargetOnly") else b, var), {})


def full(v):
    """A cell is reported only with all seeds present."""
    return v if len(v) >= MIN_SEEDS else {}


def cell(v, bold=False):
    return fmt_cell(full(v), bold, missing="--")


def bolds(cols):
    return best_mask([full(c) for c in cols])


def cmid(first, groups, width):
    return "".join(f"\\cmidrule(lr){{{first + i * width}-{first + i * width + width - 1}}}" for i in range(groups))


# ------------------------------------------------------------------------------------------------------------------ tables
def downstream(c: Cells) -> str:
    """Table 2: Full-Training / Global Subset / Layer-Wise Subset / Target-Only (early stop) under LoRA with exact scoring; no rule, bold = best of all four."""
    rows = [("Full-Training", "FullTraining", "-", ""), ("Global Subset", "GlobalSubset", EXACT, ""), ("Layer-Wise Subset", "LayerWiseSubset", EXACT, ""),
            ("Target-Only", "TargetOnly", "-", TO_VARIANT)]
    L = [r"\begin{tabular}{l c c c c}", "\t\\toprule",
         "\t\\textbf{Method} & " + " & ".join(SETTING_TEX[s] for s in SETTINGS) + r" \\", "\t\\midrule"]
    arms = [r for r in rows if r]
    best = {s: bolds([c.v(s, "LoRA", m, b, v) for _, m, b, v in arms]) for s in SETTINGS}   # bold = best of all rows, Target-Only included
    for r in rows:
        if r is None:
            L.append("\t\\midrule")
            continue
        label, m, b, v = r
        i = arms.index(r)
        L.append(f"\t{label} & " + " & ".join(cell(c.v(s, 'LoRA', m, b, v), best[s][i]) for s in SETTINGS) + r" \\")
    return "\n".join(L + ["\t\\bottomrule", r"\end{tabular}"]) + "\n"


def matrix(c: Cells, ft: str) -> str:
    """Five update rules x setting x {Exact, Compressed} for one fine-tuning method."""
    keys = [(s, b) for s in SETTINGS for b in (EXACT, CMP)]
    L = [r"\begin{tabular}{l " + " ".join("c c" for _ in SETTINGS) + "}", "\t\\toprule",
         "\t" + mrow(2, r"\textbf{Method}") + " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{SETTING_TEX[s]}}}" for s in SETTINGS) + r" \\",
         "\t" + cmid(2, len(SETTINGS), 2),
         "\t & " + " & ".join("Exact & Compressed" for _ in SETTINGS) + r" \\", "\t\\midrule"]
    best = {k: bolds([c.v(k[0], ft, m, k[1]) for m in METHODS]) for k in keys}
    for i, m in enumerate(METHODS):
        if m == "FullTraining":
            cells = [f"\\multicolumn{{2}}{{c}}{{{cell(c.v(s, ft, m), all(best[(s, b)][i] for b in (EXACT, CMP)))}}}" for s in SETTINGS]
            L += [f"\t{LABEL[m]} & " + " & ".join(cells) + r" \\", "\t\\midrule"]
        else:
            L.append(f"\t{LABEL[m]} & " + " & ".join(cell(c.v(s, ft, m, b), best[(s, b)][i]) for s, b in keys) + r" \\")
    to = [r for r in TO_ROWS if any(full(c.v(s, ft, r[1], r[2], r[3])) for s in SETTINGS)]   # pool-free arm: spans both scoring columns, never bold
    if to:
        L.append("\t\\midrule")
        for label, m, b, v in to:
            L.append(f"\t{label} & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{cell(c.v(s, ft, m, b, v))}}}" for s in SETTINGS) + r" \\")
    return "\n".join(L + ["\t\\bottomrule", r"\end{tabular}"]) + "\n"


CONTROL_ROWS = [("Full-Training", "FullTraining", "-", ""), ("Global Random Subset", "GlobalSubset", "-", "random"),
                ("Layer-Wise Random Subset", "LayerWiseSubset", "-", "random"), None,
                ("Global Subset", "GlobalSubset", CMP, ""), ("Global Subset, layer-normalized scores", "GlobalSubset", CMP, "lnorm"),
                ("Layer-Wise Subset", "LayerWiseSubset", CMP, ""), None, *TO_ROWS]


def controls(c: Cells) -> str:
    """Random Subset and layer-normalized controls next to the compressed curated endpoints, columns = setting x fine-tuning."""
    keys = [(s, ft) for s in SETTINGS for ft in FTS]
    rows = [r for r in CONTROL_ROWS if r is None or any(full(c.v(s, ft, r[1], r[2], r[3])) for s, ft in keys)]
    arms = [r for r in rows if r]
    pool = [r for r in arms if r[1] != "TargetOnly"]   # Target-Only is shown but never bold
    L = [r"\begin{tabular}{l " + " ".join("c" * len(FTS) for _ in SETTINGS) + "}", "\t\\toprule",
         "\t" + mrow(2, r"\textbf{Method}") + " & " + " & ".join(f"\\multicolumn{{{len(FTS)}}}{{c}}{{{SETTING_TEX[s]}}}" for s in SETTINGS) + r" \\",
         "\t" + cmid(2, len(SETTINGS), len(FTS)),
         "\t & " + " & ".join(FT_SHORT[ft] for _ in SETTINGS for ft in FTS) + r" \\", "\t\\midrule"]
    best = {k: bolds([c.v(k[0], k[1], m, b, v) for _, m, b, v in pool]) for k in keys}
    for r in rows:
        if r is None:
            L.append("\t\\midrule")
            continue
        label, m, b, v = r
        bo = [best[k][pool.index(r)] if r in pool else False for k in keys]
        L.append(f"\t{label} & " + " & ".join(cell(c.v(s, ft, m, b, v), bo[j]) for j, (s, ft) in enumerate(keys)) + r" \\")
    return "\n".join(L + ["\t\\bottomrule", r"\end{tabular}"]) + "\n"


ARMS_PPL = [("Full-Training", "FullTraining", "-", ""), ("Global Subset", "GlobalSubset", EXACT, ""), ("Layer-Wise Subset", "LayerWiseSubset", EXACT, ""), *TO_ROWS]


def lowest_mask(cols, prec: int = 2):
    """best_mask for a lower-is-better statistic: True where the mean is minimal at the displayed precision (ties -> all True);
    nothing is bold in a row with a single filled cell."""
    means = [round(mean_se(c)[0], prec) if mean_se(c) else None for c in cols]
    filled = [m for m in means if m is not None]
    mn = min(filled) if len(filled) > 1 else None
    return [m is not None and mn is not None and abs(m - mn) < 1e-9 for m in means]


def qa_ppl(c: Cells) -> str:
    """Final evaluation perplexity of the three main update rules and Target-Only (same budget / early stop), rows = setting x fine-tuning; lowest pool-based arm bold."""
    L = [r"\begin{tabular}{l l " + " ".join("c" for _ in ARMS_PPL) + "}", "\t\\toprule",
         "\t\\textbf{Setting} & \\textbf{Fine-tuning} & " + " & ".join(f"\\textbf{{{lab}}}" for lab, _, _, _ in ARMS_PPL) + r" \\"]
    for s in SETTINGS:
        block = []
        for ft in FTS:
            vals = [full(c.p(s, ft, m, b, v)) if full(c.v(s, ft, m, b, v)) else {} for _, m, b, v in ARMS_PPL]
            npool = sum(1 for _, m, _, _ in ARMS_PPL if m != "TargetOnly")
            mask = lowest_mask(vals[:npool]) + [False] * (len(vals) - npool)   # Target-Only shown, never bold
            cells = [fmt_cell(v, bo, prec=2, missing="--") for v, bo in zip(vals, mask)]
            block.append((FT_LABEL[ft], cells))
        if all(x == "--" for _, cells in block for x in cells):
            continue
        L.append("\t\\midrule")
        for i, (lab, cells) in enumerate(block):
            L.append(f"\t{mrow(len(block), SETTING_TEX[s]) if i == 0 else ''} & {lab} & " + " & ".join(cells) + r" \\")
    return "\n".join(L + ["\t\\bottomrule", r"\end{tabular}"]) + "\n"


# ------------------------------------------------------------------------------------------------------------------ report
def report(c: Cells) -> None:
    def ms(v):
        r = mean_se(v)
        return f"{r[0]:.1f} +- {r[1]:.1f}" + (f" (n={r[2]})" if r[2] < MIN_SEEDS else "") if r else ""
    print("# QA matrix: downstream metric at the final checkpoint, mean +- SE over seeds (Rouge-L x100 for samsum, F1 % otherwise)\n")
    for s in SETTINGS:
        print(f"## {s} ({TARGET[s]})\n")
        cols = [(ft, b) for ft in FTS for b in (EXACT, CMP)]
        print("| Update rule | " + " | ".join(f"{FT_LABEL[ft]}: {BACKEND_LABEL[b]}" for ft, b in cols) + " |\n|" + "---|" * (1 + len(cols)))
        for m in METHODS + ["TargetOnly"]:
            print(f"| {LABEL[m]} | " + " | ".join(ms(c.v(s, ft, m, b)) for ft, b in cols) + " |")
        extras = sorted(k for k in c.metric if k[0] == s and k[4])
        if extras:
            print("\nControls / variants:\n")
            for k in extras:
                print(f"- {FT_LABEL[k[1]]}, {LABEL.get(k[4], LABEL[k[2]] + ' (' + k[4] + ')')}"
                      f"{', ' + BACKEND_LABEL.get(k[3], k[3]) if k[3] != '-' else ''}: {ms(c.metric[k])}")
        print()
    print("# Target-Only Update: evaluation perplexity minimum vs final (mean over seeds)\n")
    print("| setting | finetuning | min ppl | at step | of budget | final ppl | final/min | metric |\n|---|---|---|---|---|---|---|---|")
    for k, rows in sorted(c.rows.items()):
        if k[2] != "TargetOnly":
            continue
        cur = [r for r in rows if r["min_eval_ppl"] is not None and r["steps"]]
        if cur:
            print(f"| {k[0]} | {k[1]} | {st.mean(r['min_eval_ppl'] for r in cur):.2f} | {st.mean(r['min_ppl_step'] for r in cur):.0f} | "
                  f"{st.mean(r['min_ppl_step'] / r['steps'] for r in cur):.0%} | {st.mean(r['final_eval_ppl'] for r in cur):.2f} | "
                  f"{st.mean(r['final_eval_ppl'] / r['min_eval_ppl'] for r in cur):.1f}x | {ms(c.metric[k])} |")
    print("\n# Seconds per training step (median over runs; training wall time / steps, periodic evaluation excluded)\n")
    print("| setting | finetuning | method | scoring | runs | s/step |\n|---|---|---|---|---|---|")
    for k, rows in sorted(c.rows.items()):
        sp = [r["train_wall_s"] / r["steps"] for r in rows if r["train_wall_s"] and r["steps"]]
        if sp:
            print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]}{'/' + k[4] if k[4] else ''} | {len(sp)} | {st.median(sp):.3f} |")


def completeness(c: Cells) -> None:
    for s in SETTINGS:
        parts = []
        for ft in FTS:
            n = [len(c.v(s, ft, m, b)) for m in METHODS for b in ((EXACT, CMP) if m != "FullTraining" else ("-",))]
            parts.append(f"{ft} {min(n)}-{max(n)}/{MIN_SEEDS}")
        print(f"{s:14s} seeds per cell: " + ", ".join(parts))


# -------------------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.splitlines()[0]))
    ap.add_argument("--scan", action="store_true", help=f"rescan the run dirs under <results root>/SFT/runs_v2 and rewrite {CSV.relative_to(REPO)}")
    ap.add_argument("--report", action="store_true", help="print the markdown matrix, the Target-Only trajectory and s/step by backend")
    args = ap.parse_args()
    if args.scan or not CSV.exists():
        runs = results_root(args.results_root) / "SFT" / "runs_v2"
        rows = scan(runs)
        write_csv(rows)
        print(f"{len(rows)} runs under {runs} -> {CSV.relative_to(REPO)}")
        src = runs
    else:
        rows, src = load_csv(), CSV.relative_to(REPO)
    c = Cells(rows)
    completeness(c)
    if args.report:
        report(c)
    files = {"sft_downstream.tex": downstream(c), "sft_qa_ppl.tex": qa_ppl(c), "sft_ablation.tex": controls(c),
             **{f"sft_qa_matrix_{ft.lower()}.tex": matrix(c, ft) for ft in FTS}}
    return emit(files, args, __file__, src, note="Llama-3.2-1B question answering, 5 seeds, final checkpoint")


if __name__ == "__main__":
    sys.exit(main())
