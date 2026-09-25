#!/usr/bin/env python
"""Manifests of the question-answering runs (Llama-3.2-1B), one line per Slurm array task; finished runs are skipped.

Writes to SFT/train/manifests/ (gitignored), for SFT/train/slurm/qa_array.sbatch:
  qa_matrix_<stage>.txt   <setting> <yaml> <seed>            TASK=SFT/train/qa_matrix_task.sh      (stages below, launch order)
  qa_matrix_all.txt       every stage concatenated
  qa_target_only.txt      <setting> FullTraining-<FT> <seed>  TASK=SFT/train/qa_target_only_task.sh (adxtab:sft-qa-ppl rows)
  case_study.txt          run.sh flags                        TASK=SFT/case_study/run.sh            (fig:case-study)
An existing file is never overwritten (a running array indexes into it by line number): the fresh version goes to <name>.new.txt
unless --force is given.

Matrix: exact scoring = GIP (PIP computes the same inner product, SFT/benchmark/check_pip_gip_equivalence.py),
compressed scoring = normal-64*64 for every fine-tuning method. Yaml names: FullTraining-<FT>; LayerWiseSubset-<FT> (Full / LoRA) and
{Block,Sublayer}WiseSubset-LoRA are the compressed cells without a suffix; every other cell is <Method>-<FT>-<gip|cmp>;
controls <Method>-<FT>-{random,lnorm} (Random Subset; layer-normalized Global Subset, compressed). MeSO arms use learning rate 5e-5.
Run dirs: <results root>/SFT/runs_v2 (tables/common.py: --results-root, $DRPT_RESULTS or $SCRATCH_DIR/Dr.Post-Training).
Usage: python SFT/train/qa_matrix_manifest.py [--force] [--results-root DIR]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import REPO, results_root  # noqa: E402

CFG = REPO / "SFT" / "train" / "configs"
OUT = REPO / "SFT" / "train" / "manifests"
SEEDS = [42, 2, 22, 62, 82]
SETTINGS = ["alpaca_samsum", "less_tydiqa", "triviaqa_nq", "less_squad"]
OTHER = ["less_tydiqa", "triviaqa_nq", "less_squad"]
TASK = {"alpaca_samsum": "samsum", "less_tydiqa": "tydiqa", "triviaqa_nq": "nq_open", "less_squad": "squad"}
MIN_LORA = {"alpaca_samsum": 45, "less_tydiqa": 42, "triviaqa_nq": 6, "less_squad": 17}   # measured train+eval minutes, LoRA, A40
CASE_STUDY = [("alpaca", "samsum", "0.4"), ("less", "tydiqa", "0.005"), ("triviaqa", "nq_open", "0.05"), ("less", "squad", "0.005")]


def arms(ft):
    def name(m, b):
        if b == "gip":
            return f"{m}-{ft}-gip"
        legacy = (m == "LayerWiseSubset" and ft != "MeSO") or (m in ("BlockWiseSubset", "SublayerWiseSubset") and ft == "LoRA")
        return f"{m}-{ft}" if legacy else f"{m}-{ft}-cmp"
    return {"full": f"FullTraining-{ft}", **{f"{m}:{b}": name(m, b) for m in ("GlobalSubset", "BlockWiseSubset", "SublayerWiseSubset", "LayerWiseSubset") for b in ("gip", "cmp")}}


STAGES = [
    ("A1", [(s, arms("LoRA")[f"{m}:gip"]) for s in SETTINGS for m in ("LayerWiseSubset", "GlobalSubset")]),
    ("A2", [("alpaca_samsum", a) for a in (arms("Full")["full"], arms("Full")["GlobalSubset:gip"], arms("Full")["GlobalSubset:cmp"], arms("Full")["LayerWiseSubset:gip"], arms("Full")["LayerWiseSubset:cmp"])]),
    ("A3", [("alpaca_samsum", a) for a in (arms("MeSO")["full"], arms("MeSO")["GlobalSubset:gip"], arms("MeSO")["GlobalSubset:cmp"], arms("MeSO")["LayerWiseSubset:gip"], arms("MeSO")["LayerWiseSubset:cmp"])]),
    ("B", [(s, arms("LoRA")[f"{m}:gip"]) for s in SETTINGS for m in ("BlockWiseSubset", "SublayerWiseSubset")]),
    ("C", [(s, arms("LoRA")[k]) for s in SETTINGS for k in ("full", "LayerWiseSubset:cmp", "GlobalSubset:cmp")]),   # LoRA baselines / compressed cells
    ("D", [(s, a) for s in OTHER for ft in ("Full", "MeSO") for a in arms(ft).values()]),
    ("E", [("alpaca_samsum", arms(ft)[f"{m}:{b}"]) for ft in ("Full", "MeSO") for m in ("BlockWiseSubset", "SublayerWiseSubset") for b in ("gip", "cmp")]),
    ("F", [(s, f"{m}-{ft}-{v}") for s in SETTINGS for ft in ("Full", "LoRA", "MeSO") for m, v in (("GlobalSubset", "random"), ("LayerWiseSubset", "random"), ("GlobalSubset", "lnorm"))]),
]


def raw(path, key):
    """yaml scalar as written (PyYAML would turn learning_rate 1.00e-04 into 0.0001; run dirs use the literal text)."""
    m = re.search(rf"^{key}:\s*(\S+)", Path(path).read_text(), re.M)
    return m.group(1) if m else None


def run_dir(runs, setting, method, seed):
    d = yaml.safe_load((CFG / setting / "defaults.yaml").read_text())
    lr = raw(CFG / setting / f"{method}.yaml", "learning_rate") or raw(CFG / setting / "defaults.yaml", "learning_rate")
    name = f"{d['train_dataset']}_{d['target_task']}-{os.path.basename(d['model'])}-{method}-p{d['percentage']}-lr{lr}-b{d['batch_size']}-v{d['n_val']}-s{seed}"
    return runs / name, d["target_task"]


def write(name, lines, force):
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.txt"
    if path.exists() and not force:
        path = OUT / f"{name}.new.txt"   # never rewrite a manifest an array may still be indexing into
    path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="overwrite existing manifest files (only when no array uses them)")
    ap.add_argument("--results-root", default=None)
    a = ap.parse_args()
    runs = results_root(a.results_root) / "SFT" / "runs_v2"
    out_all = []
    for stage, cells in STAGES:
        lines, minutes = [], 0.0
        for seed in SEEDS:
            for setting, method in cells:
                if not (CFG / setting / f"{method}.yaml").exists():
                    sys.exit(f"missing yaml {setting}/{method}")
                d, target = run_dir(runs, setting, method, seed)
                if (d / f"{target}_results.json").exists():
                    continue
                lines.append(f"{setting} {method} {seed}")
                minutes += MIN_LORA[setting] * (1.0 if "-LoRA" in method else 1.3)
        p = write(f"qa_matrix_{stage}", lines, a.force)
        out_all += lines
        print(f"stage {stage}: {len(lines):4d} runs, ~{minutes / 60:6.1f} GPU-h -> {p.relative_to(REPO)}")
    write("qa_matrix_all", out_all, a.force)
    # Target-Only rows (train_val_ablation.sh run dirs)
    to = []
    for seed in SEEDS:
        for setting in SETTINGS:
            for ft in ("Full", "LoRA", "MeSO"):
                t = TASK[setting]
                if not glob.glob(str(runs / f"{t}_val_{t}-Llama-3.2-1B-FullTraining-{ft}-ms*-lr*-b8-v16-s{seed}" / f"{t}_results.json")):
                    to.append(f"{setting} FullTraining-{ft} {seed}")
    print(f"target-only: {len(to):4d} runs -> {write('qa_target_only', to, a.force).relative_to(REPO)}")
    # Case study (SFT/case_study/run.sh; full-parameter, lr 1e-5, records every step)
    cs = []
    for seed in SEEDS:
        for train, task, pct in CASE_STUDY:
            if not (runs / "case_study" / f"{train}_{task}-Llama-3.2-1B-p{pct}-lr1e-05-b8-v16-s{seed}" / "selection_records.json").exists():
                cs.append(f"--train {train} --task {task} --percentage {pct} --seed {seed}")
    print(f"case study : {len(cs):4d} runs -> {write('case_study', cs, a.force).relative_to(REPO)}")
    print(f"total matrix runs pending: {len(out_all)}")


if __name__ == "__main__":
    main()
