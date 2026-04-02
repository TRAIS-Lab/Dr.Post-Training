#!/usr/bin/env python
"""
Standalone scoring benchmark — times scoring functions directly with synthetic tensors.

No model loading, no forward/backward. Creates random (n, T, O, I) tensors
and calls each scoring function, summed across all linear layers in the model.

Usage:
    python benchmark_scoring.py --gpu 0 --model-tag qwen3-0.6b
    python benchmark_scoring.py --gpu 0 --model-tag qwen3-0.6b --output results/scoring_qwen3-0.6b.json
"""

import argparse
import json
import math
import os
import sys
import time

import torch

# Add project root to path
BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BENCHMARK_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from drpt.selection.utils import (
    compute_scores_and_similarity,
    compute_scores_full_ghost,
    compute_scores_direct_materialization,
)

# =============================================================================
# Model definitions
# =============================================================================

MODEL_DEFS = {
    "qwen3-0.6b": {
        "name": "Qwen/Qwen3-0.6B",
        "h": 1024, "i": 3072, "L": 28, "heads": 16, "kv": 8, "hd": 128,
    },
    "qwen3-1.7b": {
        "name": "Qwen/Qwen3-1.7B",
        "h": 2048, "i": 6144, "L": 28, "heads": 16, "kv": 8, "hd": 128,
    },
    "llama-3.2-3b": {
        "name": "meta-llama/Llama-3.2-3B",
        "h": 3072, "i": 8192, "L": 28, "heads": 24, "kv": 8, "hd": 128,
    },
}


def get_layer_dims(d):
    """Return list of (name, O, I) for all linear layers in one transformer block."""
    h, i = d["h"], d["i"]
    q_dim = d["heads"] * d["hd"]
    kv_dim = d["kv"] * d["hd"]
    return [
        ("q_proj", q_dim, h),
        ("k_proj", kv_dim, h),
        ("v_proj", kv_dim, h),
        ("o_proj", h, q_dim),
        ("gate_proj", i, h),
        ("up_proj", i, h),
        ("down_proj", h, i),
    ]


def compute_thresholds(d):
    """Compute V* coefficient and T* for a model."""
    layers = get_layer_dims(d)
    L = d["L"]
    sum_OI = sum(O * I for _, O, I in layers) * L
    sum_OpI = sum(O + I for _, O, I in layers) * L
    sum_sqrtOI = sum(math.sqrt(O * I) for _, O, I in layers) * L
    vstar_coeff = sum_OI / sum_OpI  # V* = vstar_coeff / T
    tstar = sum_OI / sum_sqrtOI     # T where reduced_ghost == direct
    return vstar_coeff, tstar


# =============================================================================
# Scoring timer
# =============================================================================

def estimate_full_ghost_memory(n, T, m, O):
    """Estimate peak memory for full_ghost go_dot tensor in bytes."""
    return n * m * T * T * 2  # bf16


def time_scoring_one_layer(method, n, T, m, O, I, device,
                           num_warmup=5, num_iters=10, direct_batch_size=0):
    """Time a single scoring call for one layer. Returns ms or None if OOM."""
    try:
        train_go = torch.randn(n, T, O, dtype=torch.bfloat16, device=device)
        train_inp = torch.randn(n, T, I, dtype=torch.bfloat16, device=device)
        val_go = torch.randn(m, T, O, dtype=torch.bfloat16, device=device)
        val_inp = torch.randn(m, T, I, dtype=torch.bfloat16, device=device)
    except torch.cuda.OutOfMemoryError:
        return None

    if method == "reduced_ghost":
        fn = lambda: compute_scores_and_similarity(
            train_go, train_inp, val_go, val_inp, None, False)
    elif method == "full_ghost":
        mem = estimate_full_ghost_memory(n, T, m, O)
        if mem > 20e9:  # skip if >20GB for this single layer
            del train_go, train_inp, val_go, val_inp
            torch.cuda.empty_cache()
            return None
        fn = lambda: compute_scores_full_ghost(
            train_go, train_inp, val_go, val_inp, None, False)
    elif method == "direct":
        fn = lambda: compute_scores_direct_materialization(
            train_go, train_inp, val_go, val_inp, None, False,
            batch_size=direct_batch_size)[:2]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Warmup (includes torch.compile JIT for reduced_ghost)
    for _ in range(num_warmup):
        try:
            fn()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            del train_go, train_inp, val_go, val_inp
            return None

    # Timed iterations
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    for j in range(num_iters):
        try:
            starts[j].record()
            fn()
            ends[j].record()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            del train_go, train_inp, val_go, val_inp
            return None

    torch.cuda.synchronize()
    times = [starts[j].elapsed_time(ends[j]) for j in range(num_iters)]

    del train_go, train_inp, val_go, val_inp
    torch.cuda.empty_cache()

    return sum(times) / len(times)


def time_scoring_all_layers(method, n, T, m, model_def, device,
                            num_warmup=5, num_iters=10, direct_batch_size=0):
    """Time scoring across all linear layers, return total ms or None if OOM."""
    layer_types = get_layer_dims(model_def)
    L = model_def["L"]
    total_ms = 0.0

    # Group identical layers to avoid redundant timing
    # All L transformer blocks have the same 7 layer types
    for name, O, I in layer_types:
        ms = time_scoring_one_layer(
            method, n, T, m, O, I, device,
            num_warmup=num_warmup, num_iters=num_iters,
            direct_batch_size=direct_batch_size)
        if ms is None:
            return None
        total_ms += ms * L  # Same dims repeated L times

    return total_ms


# =============================================================================
# Compress scoring simulation
# =============================================================================

def time_compress_scoring(n, T, m, O, I, device, kappa=4096,
                          num_warmup=5, num_iters=10):
    """Simulate compress scoring: project to R^kappa, then inner product.

    Real compress does: sparsify(go) ⊗ sparsify(inp) → R^kappa per sample,
    then s = train_compressed @ val_compressed.T
    We simulate the dominant cost: two random projections + Kronecker + matmul.
    """
    k = int(math.sqrt(kappa))  # e.g., 64 for kappa=4096
    try:
        # Simulate: project go (n, T, O) → (n, k) and inp (n, T, I) → (n, k)
        # Then Kronecker: (n, k*k) = (n, kappa)
        train_c = torch.randn(n, kappa, dtype=torch.bfloat16, device=device)
        val_c = torch.randn(1, kappa, dtype=torch.bfloat16, device=device)

        # Projection cost simulation: two matmuls per sample
        proj_O = torch.randn(O, k, dtype=torch.bfloat16, device=device)
        proj_I = torch.randn(I, k, dtype=torch.bfloat16, device=device)
        train_go = torch.randn(n, T, O, dtype=torch.bfloat16, device=device)
        train_inp = torch.randn(n, T, I, dtype=torch.bfloat16, device=device)
    except torch.cuda.OutOfMemoryError:
        return None

    def fn():
        # Project: (n, T, O) @ (O, k) → (n, T, k) → sum over T → (n, k)
        go_proj = (train_go @ proj_O).sum(dim=1)  # (n, k)
        inp_proj = (train_inp @ proj_I).sum(dim=1)  # (n, k)
        # Kronecker: (n, k) x (n, k) → (n, k*k) via outer product
        compressed = (go_proj.unsqueeze(2) * inp_proj.unsqueeze(1)).flatten(1)
        # Score: matmul with val
        return (compressed * val_c).sum(dim=1)

    for _ in range(num_warmup):
        try:
            fn()
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    for j in range(num_iters):
        starts[j].record()
        fn()
        ends[j].record()
    torch.cuda.synchronize()
    times = [starts[j].elapsed_time(ends[j]) for j in range(num_iters)]

    del train_go, train_inp, train_c, val_c, proj_O, proj_I
    torch.cuda.empty_cache()
    return sum(times) / len(times)


def time_compress_all_layers(n, T, m, model_def, device,
                             num_warmup=5, num_iters=10, kappa=4096):
    """Time compress scoring across all layers."""
    layer_types = get_layer_dims(model_def)
    L = model_def["L"]
    total_ms = 0.0
    for name, O, I in layer_types:
        ms = time_compress_scoring(n, T, m, O, I, device, kappa,
                                   num_warmup, num_iters)
        if ms is None:
            return None
        total_ms += ms * L
    return total_ms


# =============================================================================
# Main
# =============================================================================

METHODS = ["compress", "full_ghost", "reduced_ghost", "direct"]


def run_config(n, T, m, model_def, device, num_warmup=5, num_iters=10):
    """Run all scoring methods for one (n, T, m) config. Returns dict."""
    results = {}
    for method in METHODS:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if method == "compress":
            ms = time_compress_all_layers(n, T, m, model_def, device,
                                          num_warmup, num_iters)
        else:
            ms = time_scoring_all_layers(method, n, T, m, model_def, device,
                                         num_warmup, num_iters,
                                         direct_batch_size=0)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        results[method] = {"total_ms": round(ms, 1) if ms is not None else None,
                           "peak_gb": round(peak_gb, 2)}
        status = f"{ms:.1f} ms" if ms is not None else "OOM"
        print(f"      {method:<14s} {status:>10s}  ({peak_gb:.1f} GB)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Standalone scoring benchmark")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model-tag", type=str, required=True,
                        choices=list(MODEL_DEFS.keys()))
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=10)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda"

    model_def = MODEL_DEFS[args.model_tag]
    vstar_coeff, tstar = compute_thresholds(model_def)

    print(f"Model: {model_def['name']} ({args.model_tag})")
    print(f"V* = {vstar_coeff:.0f}/T,  T* = {tstar:.0f}")
    print()

    # Define configs: m-sweep at T=512, T-sweep at m=1
    configs = []
    n = 8

    # Axis 1: m sweep at T=512
    for m in [1, 2, 4, 8, 16]:
        configs.append({"n": n, "T": 512, "m": m, "axis": "m_sweep"})

    # Axis 2: T sweep at m=1
    for T in [256, 512, 1024, 2048, 4096, 8192]:
        if T == 512:
            continue  # already covered by m_sweep with m=1
        configs.append({"n": n, "T": T, "m": 1, "axis": "T_sweep"})

    all_results = {}
    total = len(configs)

    for idx, cfg in enumerate(configs):
        n_c, T_c, m_c = cfg["n"], cfg["T"], cfg["m"]
        label = f"n={n_c} T={T_c} m={m_c}"
        vstar = vstar_coeff / T_c
        print(f"  [{idx+1}/{total}] {label}  (V*={vstar:.2f})")
        all_results[label] = run_config(n_c, T_c, m_c, model_def, device,
                                        args.num_warmup, args.num_iters)

    # Summary
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {model_def['name']}  (T*={tstar:.0f})")
    print(f"{'='*80}")
    print(f"\n  {'Config':<22s}  {'compress':>10s}  {'full_gh':>10s}  {'red_ghost':>10s}  {'direct':>10s}  {'Best exact':>12s}")
    print(f"  {'-'*78}")

    for label, r in all_results.items():
        parts = [f"  {label:<22s}"]
        for method in METHODS:
            ms = r[method]["total_ms"]
            parts.append(f"{ms:>9.1f}ms" if ms is not None else f"{'OOM':>10s}")
        exact = {m: r[m]["total_ms"] for m in ["full_ghost", "reduced_ghost", "direct"]
                 if r[m]["total_ms"] is not None}
        best = min(exact, key=exact.get) if exact else "N/A"
        parts.append(f"  {best:>10s}")
        print("  ".join(parts))

    # Save
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        out = {
            "model": model_def["name"],
            "model_tag": args.model_tag,
            "model_dims": {k: model_def[k] for k in ["h", "i", "L", "heads", "kv", "hd"]},
            "T_star": round(tstar, 0),
            "V_star_coeff": round(vstar_coeff, 1),
            "configs": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
