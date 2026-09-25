#!/usr/bin/env python
"""PIP vs GIP: both return <g_i, g_target> per candidate; this prints their max relative difference and the top-k agreement
on random 3-D factors (B=8 candidates, V=1 target, T tokens, O x I layer) in fp32 on CPU (reference path) and, if available,
bf16 on CUDA (fused kernels). Evidence for treating the two exact backends as interchangeable."""
import torch
from drpt.selection.utils import compute_scores_and_similarity, compute_scores_gip

def run(device, dtype, B=8, V=1, T=96, O=256, I=384, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=dtype)
    tgo, tin, vgo, vin = mk(B, T, O), mk(B, T, I), mk(V, T, O), mk(V, T, I)
    pip, _ = compute_scores_and_similarity(tgo, tin, vgo, vin, None, False)
    gip, _ = compute_scores_gip(tgo, tin, vgo, vin, None, False)
    pip, gip = pip.float().flatten(), gip.float().flatten()
    rel = ((pip - gip).abs().max() / pip.abs().max()).item()
    same = set(pip.topk(k).indices.tolist()) == set(gip.topk(k).indices.tolist())
    print(f"{device}/{str(dtype).split('.')[-1]}: max|pip-gip|/max|pip| = {rel:.2e}, top-{k} identical = {same}")
    return rel, same

ok = True
for s in range(3):
    r, same = run("cpu", torch.float32, seed=s); ok &= r < 1e-4 and same
if torch.cuda.is_available():
    for s in range(3):
        r, same = run("cuda", torch.bfloat16, seed=s); ok &= r < 5e-2 and same
print("PASS" if ok else "FAIL")
