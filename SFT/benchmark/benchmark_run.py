#!/usr/bin/env python
"""
Unified benchmark runner for Dr. Post-Training.

Runs all method x scoring combinations for a given model and config,
with detailed per-component timing breakdown.

Usage:
  # Run all configs for Llama-3.2-1B on GPU 2
  python benchmark_run.py --gpu 2

  # Single config
  python benchmark_run.py --gpu 2 --config 0

  # Smaller model for wider sweeps
  python benchmark_run.py --gpu 2 --model Qwen/Qwen2.5-0.5B

  # Custom config
  python benchmark_run.py --gpu 2 --batch-size 16 --seq-length 1024 --val-batch-size 4

  # Save results to JSON
  python benchmark_run.py --gpu 2 --output results/llama_all.json
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_SH = os.path.join(BENCHMARK_DIR, "benchmark.sh")
PROJECT_ROOT = os.path.abspath(os.path.join(BENCHMARK_DIR, "..", ".."))

# Default configs covering both sides of the reduced_ghost/full_ghost crossover.
# All use dummy dataset for guaranteed full-length sequences.
DEFAULT_CONFIGS = [
    {"label": "n=8 T=512 m=1",   "batch_size": 8,  "seq_length": 512,  "val_batch_size": 1},
    {"label": "n=8 T=512 m=4",   "batch_size": 8,  "seq_length": 512,  "val_batch_size": 4},
    {"label": "n=4 T=1024 m=1",  "batch_size": 4,  "seq_length": 1024, "val_batch_size": 1},
    {"label": "n=4 T=1024 m=4",  "batch_size": 4,  "seq_length": 1024, "val_batch_size": 4},
    {"label": "n=2 T=2048 m=1",  "batch_size": 2,  "seq_length": 2048, "val_batch_size": 1},
]

# All method x scoring combinations to benchmark.
# (method, scoring_method, score_compression)
COMBOS = [
    ("standard",        "reduced_ghost",  "none"),
    ("layerwise",       "compress",       "normal-64*64"),
    ("layerwise",       "full_ghost",     "none"),
    ("layerwise",       "reduced_ghost",  "none"),
    ("layerwise",       "direct",         "none"),
    ("subset",          "compress",       "normal-64*64"),
    ("subset",          "full_ghost",     "none"),
    ("subset",          "reduced_ghost",  "none"),
    ("subset",          "direct",         "none"),
    ("subset_one_pass", "compress",       "normal-64*64"),
    ("subset_one_pass", "full_ghost",     "none"),
    ("subset_one_pass", "reduced_ghost",  "none"),
    ("subset_one_pass", "direct",         "none"),
]


# =============================================================================
# Subprocess runner
# =============================================================================

def run_single(method, scoring, score_comp, config, gpu, num_warmup, num_iterations, model=None, direct_batch_size=0, gradient_checkpointing=False):
    """Run a single benchmark via bash (clean process, no CUDA context leak)."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT + ":" + env.get("PYTHONPATH", "")
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    tmp = tempfile.mktemp(suffix='.json')
    args = ["--method", method, "--dataset", "dummy"]
    if model:
        args += ["--model", model]
    args += [
        "--batch-size", str(config["batch_size"]),
        "--seq-length", str(config["seq_length"]),
        "--val-batch-size", str(config["val_batch_size"]),
        "--scoring-method", scoring,
        "--score-compression", score_comp,
        "--num-warmup", str(num_warmup),
        "--num-iterations", str(num_iterations),
        "--direct-batch-size", str(direct_batch_size),
        "--output", tmp,
    ]
    if gradient_checkpointing:
        args += ["--gradient-checkpointing"]
    cmd = ["bash", BENCHMARK_SH] + args
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr_tail = proc.stderr.strip().split('\n')[-3:]
        return None, '\n'.join(stderr_tail)
    with open(tmp) as f:
        result = json.load(f)
    os.remove(tmp)
    return result, None


# =============================================================================
# Total time computation
# =============================================================================

def compute_total(r, method):
    """Compute total step time from result dict."""
    if r is None:
        return None
    if method in ("standard", "layerwise"):
        return r.get("forward", 0) + r.get("backward", 0) + r.get("optimizer", 0)
    elif method == "subset":
        return (r.get("pass1_forward", 0) + r.get("pass1_backward", 0) +
                r.get("selection", 0) + r.get("pass2_forward", 0) +
                r.get("pass2_backward", 0) + r.get("optimizer", 0))
    elif method == "subset_one_pass":
        return (r.get("forward", 0) + r.get("backward", 0) +
                r.get("selection", 0) + r.get("wgrad", 0) +
                r.get("optimizer", 0))
    return 0


# =============================================================================
# Detailed breakdown printer
# =============================================================================

def print_breakdown(method, scoring, r):
    """Print detailed per-component breakdown."""
    if r is None:
        print(f"    OOM")
        return

    mem = r.get("peak_memory_gb", 0)

    if method == "standard":
        fwd, bwd, opt = r.get("forward", 0), r.get("backward", 0), r.get("optimizer", 0)
        act, wg = r.get("act_grad", 0), r.get("wgrad", 0)
        print(f"    forward         {fwd:>8.1f} ms")
        print(f"    backward        {bwd:>8.1f} ms")
        print(f"      act_grad      {act:>8.1f} ms")
        print(f"      w.grad        {wg:>8.1f} ms")
        print(f"      autograd      {bwd - act - wg:>8.1f} ms")
        print(f"    optimizer       {opt:>8.1f} ms")
        print(f"    TOTAL           {fwd + bwd + opt:>8.1f} ms  |  mem={mem:.2f} GB")

    elif method == "layerwise":
        fwd, bwd, opt = r.get("forward", 0), r.get("backward", 0), r.get("optimizer", 0)
        act = r.get("act_grad", 0)
        comp, score, sel, wg = r.get("compress", 0), r.get("score", 0), r.get("select", 0), r.get("wgrad", 0)
        sw = r.get("select_wgrad", 0)
        measured = act + comp + score + sel + wg + sw
        print(f"    forward         {fwd:>8.1f} ms")
        print(f"    backward        {bwd:>8.1f} ms")
        print(f"      act_grad      {act:>8.1f} ms")
        if comp > 0.01: print(f"      compress      {comp:>8.1f} ms")
        print(f"      score         {score:>8.1f} ms")
        if sw > 0.01:
            print(f"      sel+wgrad     {sw:>8.1f} ms")
        else:
            print(f"      select        {sel:>8.1f} ms")
            print(f"      w.grad        {wg:>8.1f} ms")
        print(f"      autograd      {bwd - measured:>8.1f} ms")
        print(f"    optimizer       {opt:>8.1f} ms")
        print(f"    TOTAL           {fwd + bwd + opt:>8.1f} ms  |  mem={mem:.2f} GB")

    elif method == "subset":
        p1f, p1b = r.get("pass1_forward", 0), r.get("pass1_backward", 0)
        sel_t = r.get("selection", 0)
        p2f, p2b = r.get("pass2_forward", 0), r.get("pass2_backward", 0)
        opt = r.get("optimizer", 0)
        p1_act, p1_comp, p1_score = r.get("p1_act_grad", 0), r.get("p1_compress", 0), r.get("p1_score", 0)
        p2_act, p2_wg = r.get("p2_act_grad", 0), r.get("p2_wgrad", 0)
        total = p1f + p1b + sel_t + p2f + p2b + opt
        print(f"    pass1_forward   {p1f:>8.1f} ms")
        print(f"    pass1_backward  {p1b:>8.1f} ms")
        print(f"      act_grad      {p1_act:>8.1f} ms")
        if p1_comp > 0.01: print(f"      compress      {p1_comp:>8.1f} ms")
        print(f"      score         {p1_score:>8.1f} ms")
        print(f"      autograd      {p1b - p1_act - p1_comp - p1_score:>8.1f} ms")
        print(f"    selection       {sel_t:>8.1f} ms")
        print(f"    pass2_forward   {p2f:>8.1f} ms")
        print(f"    pass2_backward  {p2b:>8.1f} ms")
        if p2_act > 0.01:
            print(f"      act_grad      {p2_act:>8.1f} ms")
            print(f"      w.grad        {p2_wg:>8.1f} ms")
            print(f"      autograd      {p2b - p2_act - p2_wg:>8.1f} ms")
        print(f"    optimizer       {opt:>8.1f} ms")
        print(f"    TOTAL           {total:>8.1f} ms  |  mem={mem:.2f} GB")

    elif method == "subset_one_pass":
        fwd, bwd = r.get("forward", 0), r.get("backward", 0)
        sel_t, asm, opt = r.get("selection", 0), r.get("wgrad", 0), r.get("optimizer", 0)
        act, comp, score, retain = r.get("act_grad", 0), r.get("compress", 0), r.get("score", 0), r.get("retain", 0)
        measured = act + comp + score + retain
        total = fwd + bwd + sel_t + asm + opt
        print(f"    forward         {fwd:>8.1f} ms")
        print(f"    backward        {bwd:>8.1f} ms")
        print(f"      act_grad      {act:>8.1f} ms")
        if comp > 0.01: print(f"      compress      {comp:>8.1f} ms")
        print(f"      score         {score:>8.1f} ms")
        print(f"      retain        {retain:>8.1f} ms")
        print(f"      autograd      {bwd - measured:>8.1f} ms")
        print(f"    selection       {sel_t:>8.1f} ms")
        print(f"    assembly        {asm:>8.1f} ms")
        print(f"    optimizer       {opt:>8.1f} ms")
        print(f"    TOTAL           {total:>8.1f} ms  |  mem={mem:.2f} GB")


# =============================================================================
# Summary printer
# =============================================================================

def print_summary(all_results, model_name):
    """Print summary tables across all configs."""
    print(f"\n{'='*80}")
    print(f"  SUMMARY: {model_name}")
    print(f"{'='*80}")

    # Subset 1P overhead vs standard
    print(f"\n  Subset 1P Overhead vs Standard:")
    print(f"  {'Config':<22} {'compress':>10} {'full_gh':>10} {'red_ghost':>10} {'direct':>10}")
    print(f"  {'-'*62}")
    for label, config_results in all_results.items():
        std_r = config_results.get("standard/reduced_ghost")
        std_total = compute_total(std_r, "standard") if std_r else None
        if std_total is None:
            continue
        vals = []
        for scoring in ["compress", "full_ghost", "reduced_ghost", "direct"]:
            key = f"subset_one_pass/{scoring}"
            r = config_results.get(key)
            total = compute_total(r, "subset_one_pass")
            if total is not None:
                ovh = (total / std_total - 1) * 100
                vals.append(f"{ovh:>+9.1f}%")
            else:
                vals.append(f"{'OOM':>10}")
        print(f"  {label:<22} {'  '.join(vals)}")

    # Score cost
    print(f"\n  Score Cost (ms):")
    print(f"  {'Config':<22} {'compress':>10} {'full_gh':>10} {'red_ghost':>10} {'direct':>10}")
    print(f"  {'-'*62}")
    for label, config_results in all_results.items():
        vals = []
        for scoring in ["compress", "full_ghost", "reduced_ghost", "direct"]:
            key = f"subset_one_pass/{scoring}"
            r = config_results.get(key)
            if r is not None:
                score = r.get("score", 0) + r.get("compress", 0)
                vals.append(f"{score:>9.1f}")
            else:
                vals.append(f"{'OOM':>10}")
        print(f"  {label:<22} {'  '.join(vals)}")

    # One-pass speedup vs two-pass
    print(f"\n  One-Pass Speedup vs Two-Pass:")
    print(f"  {'Config':<22} {'compress':>10} {'full_gh':>10} {'red_ghost':>10} {'direct':>10}")
    print(f"  {'-'*62}")
    for label, config_results in all_results.items():
        vals = []
        for scoring in ["compress", "full_ghost", "reduced_ghost", "direct"]:
            r_2p = config_results.get(f"subset/{scoring}")
            r_1p = config_results.get(f"subset_one_pass/{scoring}")
            t_2p = compute_total(r_2p, "subset")
            t_1p = compute_total(r_1p, "subset_one_pass")
            if t_2p and t_1p and t_2p > 0:
                speedup = (1 - t_1p / t_2p) * 100
                vals.append(f"{speedup:>9.1f}%")
            else:
                vals.append(f"{'N/A':>10}")
        print(f"  {label:<22} {'  '.join(vals)}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for Dr. Post-Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all default configs for Llama-3.2-1B
  python benchmark_run.py --gpu 2

  # Single config
  python benchmark_run.py --gpu 2 --config 0

  # Smaller model for wider sweeps
  python benchmark_run.py --gpu 2 --model Qwen/Qwen2.5-0.5B

  # Custom config (overrides default configs)
  python benchmark_run.py --gpu 2 --batch-size 16 --seq-length 1024 --val-batch-size 4

  # Save results
  python benchmark_run.py --gpu 2 --output results/llama.json
        """)
    parser.add_argument("--gpu", type=int, default=None, help="GPU index (default: use CUDA_VISIBLE_DEVICES from env)")
    parser.add_argument("--model", type=str, default=None, help="Model name (default: Llama-3.2-1B)")
    parser.add_argument("--config", type=int, default=None, help="Run only this config index (0-4)")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size (custom config)")
    parser.add_argument("--seq-length", type=int, default=None, help="Override sequence length")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Override val batch size")
    parser.add_argument("--num-warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--num-iterations", type=int, default=20, help="Timed iterations (default: 20)")
    parser.add_argument("--direct-batch-size", type=int, default=0, help="Chunk size for batched direct scoring (0=all at once)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient (activation) checkpointing")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    # Determine configs to run
    if args.batch_size is not None:
        # Custom single config
        n, T, m = args.batch_size, args.seq_length or 512, args.val_batch_size or 1
        configs = [{"label": f"n={n} T={T} m={m}", "batch_size": n, "seq_length": T, "val_batch_size": m}]
    elif args.config is not None:
        configs = [DEFAULT_CONFIGS[args.config]]
    else:
        configs = DEFAULT_CONFIGS

    model = args.model
    model_name = model or "meta-llama/Llama-3.2-1B"

    all_results = {}
    total_runs = len(configs) * len(COMBOS)
    run_idx = 0

    for config in configs:
        label = config["label"]
        print(f"\n{'='*70}")
        print(f"  {label} | {model_name} (dummy, full-length)")
        print(f"{'='*70}")

        config_results = {}
        for method, scoring, score_comp in COMBOS:
            run_idx += 1
            tag = f"{method}/{scoring}"
            print(f"\n  [{run_idx}/{total_runs}] {tag}")

            r, err = run_single(method, scoring, score_comp, config, args.gpu,
                                args.num_warmup, args.num_iterations, model,
                                direct_batch_size=args.direct_batch_size,
                                gradient_checkpointing=args.gradient_checkpointing)
            if r is None:
                print(f"    FAILED: {err}")
            else:
                print_breakdown(method, scoring, r)

            config_results[tag] = r
        all_results[label] = config_results

    # Print summary
    print_summary(all_results, model_name)

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({"model": model_name, "results": all_results}, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
