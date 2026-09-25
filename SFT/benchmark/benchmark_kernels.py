#!/usr/bin/env python
"""
Fused kernels (drpt/kernels backends: cute, triton) vs PyTorch reference ops on synthetic per-layer tensors.

For every Linear layer type of a model (q/k/v/o/gate/up/down) at a given
(n, T, m), times the reference and each available backend's implementation of
the three custom-backward steps that drpt/kernels fuses — ghost inner product
(gip), per-token inner product (pip) and the selected-sample weight gradient —
reports the relative error of each against an fp64 reference, and sums over the
blocks.

    python SFT/benchmark/benchmark_kernels.py --model-tag qwen3-1.7b --n 8 --T 512 --m 1
"""
import argparse
import os
import sys

import torch

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(BENCHMARK_DIR, "..", "..")))

from drpt import kernels  # noqa: E402
from drpt.selection.utils import (  # noqa: E402
    _compute_scores_and_similarity_ref, compute_scores_gip, compute_selected_gradients,
)
from SFT.benchmark.benchmark_scoring import MODEL_DEFS, get_layer_dims  # noqa: E402

MODEL_DEFS = dict(MODEL_DEFS)
MODEL_DEFS.update({
    "qwen3-1.7b": {"name": "Qwen/Qwen3-1.7B", "h": 2048, "i": 6144, "L": 28, "heads": 16, "kv": 8, "hd": 128},
    "qwen3-4b": {"name": "Qwen/Qwen3-4B", "h": 2560, "i": 9728, "L": 36, "heads": 32, "kv": 8, "hd": 128},
    "qwen3-8b": {"name": "Qwen/Qwen3-8B", "h": 4096, "i": 12288, "L": 36, "heads": 32, "kv": 8, "hd": 128},
})


def timeit(fn, warm=3, iters=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for j in range(iters):
        st[j].record(); fn(); en[j].record()
    torch.cuda.synchronize()
    t = sorted(st[j].elapsed_time(en[j]) for j in range(iters))
    return t[len(t) // 2]


def relerr(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-tag", default="qwen3-1.7b", choices=sorted(MODEL_DEFS))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--m", type=int, default=1)
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    backends = [b for b, ok in (("cute", kernels.HAS_CUTE), ("triton", kernels.HAS_TRITON)) if ok]
    if not backends:
        sys.exit("no fused-kernel backend importable")
    mods = {b: __import__(f"drpt.kernels.{b}_ops", fromlist=["x"]) for b in backends}

    d = MODEL_DEFS[args.model_tag]
    n, T, m, L = args.n, args.T, args.m, d["L"]
    dev, dt = "cuda", torch.bfloat16
    k_sel = max(1, int(n * args.frac))
    tot = {}
    print(f"{d['name']}  n={n} T={T} m={m}  |sel|={k_sel}  backends={backends}  (per-layer ms; totals x{L} blocks)")
    cols = "".join(f"{b:>8}" for b in backends)
    hdr = (f"{'layer':<10}{'O':>6}{'I':>6} | {'pip ref':>8}{cols} | {'gip ref':>8}{cols} | {'wg ref':>8}{cols} | "
           f"{'cmp ref':>8}{'cute':>7} | max rel.err vs fp64 (pip, gip, wgrad)")
    print(hdr); print("-" * len(hdr))
    for name, O, I in get_layer_dims(d):
        go = torch.randn(n, T, O, device=dev, dtype=dt) * 1e-2
        inp = torch.randn(n, T, I, device=dev, dtype=dt)
        vgo = torch.randn(m, T, O, device=dev, dtype=dt) * 1e-2
        vinp = torch.randn(m, T, I, device=dev, dtype=dt)
        G64 = torch.einsum("vto,vti->oi", vgo.double(), vinp.double())
        ref64 = torch.einsum("bso,bsi,oi->b", go.double(), inp.double(), G64)
        G = torch.einsum("vto,vti->oi", vgo, vinp)
        sel = torch.randperm(n, device=dev)[:k_sel].sort()[0]
        scale = torch.tensor(1.7, device=dev)
        w64 = torch.einsum("kso,ksi->oi", go[sel].double(), inp[sel].double()) * 1.7

        t_pip_ref = timeit(lambda: _compute_scores_and_similarity_ref(go, inp, vgo, vinp, None, False))
        kernels.set_fused_kernels(False)
        t_gip_ref = timeit(lambda: compute_scores_gip(go, inp, vgo, vinp, None, False)) if n * m * T * T * 4 < 16e9 else float("nan")
        t_wg_ref = timeit(lambda: compute_selected_gradients(go, inp, sel, False, scale))
        errs = {"pip": relerr(_compute_scores_and_similarity_ref(go, inp, vgo, vinp, None, False)[0], ref64),
                "gip": float("nan"), "wgrad": relerr(compute_selected_gradients(go, inp, sel, False, scale)[0], w64)}
        kernels.set_backend(backends[0])
        row = {"pip": [], "gip": [], "wgrad": []}
        for b in backends:
            ops = mods[b]
            row["pip"].append(timeit(lambda: ops.pip_scores(go, inp, G)))
            row["gip"].append(timeit(lambda: ops.gip_scores(go, inp, vgo, vinp)))
            row["wgrad"].append(timeit(lambda: ops.selected_wgrad(go, inp, sel, scale, False)))
            errs["pip"] = max(errs["pip"], relerr(ops.pip_scores(go, inp, G), ref64))
            errs["gip"] = max(errs["gip"] if errs["gip"] == errs["gip"] else 0.0, relerr(ops.gip_scores(go, inp, vgo, vinp), ref64))
            errs["wgrad"] = max(errs["wgrad"], relerr(ops.selected_wgrad(go, inp, sel, scale, False)[0], w64))
        # compress projection (kappa = 64 x 64): reference = the two projections + bmm as in Compressor.forward
        P_O = torch.randn(O, 64, device=dev, dtype=dt); P_I = torch.randn(I, 64, device=dev, dtype=dt)
        def cmp_ref():
            c1 = (go.reshape(-1, O) @ P_O) / 8.0
            c2 = (inp.reshape(-1, I) @ P_I) / 8.0
            return torch.einsum("bsi,bsj->bij", c1.reshape(n, T, 64), c2.reshape(n, T, 64)).reshape(n, -1)
        t_cmp_ref = timeit(cmp_ref)
        t_cmp_cute = timeit(lambda: mods["cute"].compressed_grad(go, inp, P_O, P_I, scale=1.0 / 64)) if "cute" in mods else float("nan")
        fmt = lambda xs: "".join(f"{x:>8.2f}" for x in xs)
        print(f"{name:<10}{O:>6}{I:>6} | {t_pip_ref:>8.2f}{fmt(row['pip'])} | {t_gip_ref:>8.2f}{fmt(row['gip'])} | "
              f"{t_wg_ref:>8.2f}{fmt(row['wgrad'])} | {t_cmp_ref:>8.2f}{t_cmp_cute:>7.2f} | "
              f"{errs['pip']:.0e} {errs['gip']:.0e} {errs['wgrad']:.0e}")
        for key, val in [("pip_ref", t_pip_ref), ("gip_ref", t_gip_ref), ("wgrad_ref", t_wg_ref),
                         ("compress_ref", t_cmp_ref), ("compress_cute", t_cmp_cute)] + \
                        [(f"{op}_{b}", row[op][i]) for op in ("pip", "gip", "wgrad") for i, b in enumerate(backends)]:
            tot[key] = tot.get(key, 0.0) + val * L
        del go, inp, vgo, vinp, G64, G
        torch.cuda.empty_cache()
    print("-" * len(hdr))
    print("TOTAL over blocks (ms): " + "   ".join(f"{k} {v:.1f}" for k, v in tot.items()))


if __name__ == "__main__":
    main()
