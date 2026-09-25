#!/usr/bin/env python
"""Exact vs compressed Layer-Wise scores on identical batches (alpaca/samsum, Llama-3.2-1B LoRA, selection_records.json of the
runs_diag runs): per-layer Pearson correlation and top-k overlap vs compressor size, plus the Global-style sum over layers."""
import json, sys, os, numpy as np
R = os.environ.get("SCRATCH_DIR", os.path.expanduser("~/scratch/Project")) + "/Dr.Post-Training/SFT/runs_diag/alpaca_samsum-Llama-3.2-1B-LayerWiseSubset-LoRA-%s-rec-p0.004-lr1.00e-04-b8-v16-s42/selection_records.json"
ref = json.load(open(R % "pip"))["steps"]
print(f"{'compressor':12s} {'per-layer corr (median)':>24s} {'top-4 overlap':>14s} {'Global-sum corr':>16s} {'Global top-4 overlap':>20s}")
for tag in ["cmp", "cmp128", "cmp256", "cmp512"]:
    p = R % tag
    if not os.path.exists(p): print(f"{tag:12s} (missing)"); continue
    S = json.load(open(p))["steps"]; corr, ovl, gc, go = [], [], [], []
    for a, b in zip(ref, S):
        E = np.array([l["scores"] for l in a["layers"]], float); C = np.array([l["scores"] for l in b["layers"]], float)
        k = len(a["layers"][0]["selected_indices"])
        for e, c, la, lb in zip(E, C, a["layers"], b["layers"]):
            if e.std() == 0 or c.std() == 0: continue
            corr.append(np.corrcoef(e, c)[0, 1]); ovl.append(len(set(la["selected_indices"]) & set(lb["selected_indices"])) / k)
        e, c = E.sum(0), C.sum(0); gc.append(np.corrcoef(e, c)[0, 1]); go.append(len(set(np.argsort(-e)[:k]) & set(np.argsort(-c)[:k])) / k)
    d = {"cmp": "64x64"}.get(tag, tag.replace("cmp", "") + "x" + tag.replace("cmp", ""))
    print(f"{d:12s} {np.median(corr):24.2f} {np.mean(ovl):14.2f} {np.median(gc):16.2f} {np.mean(go):20.2f}")
