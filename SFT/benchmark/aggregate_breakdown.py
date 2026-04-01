#!/usr/bin/env python
"""
Aggregate Benchmark 1 (breakdown) results into formatted tables.

Usage:
    python SFT/benchmark/aggregate_breakdown.py
    python SFT/benchmark/aggregate_breakdown.py --results-dir SFT/benchmark/results/breakdown
"""

import argparse
import json
import glob
import os
import sys


def compute_total(r, method):
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
    return None


def load_results(results_dir):
    """Load all breakdown JSON files. Returns sorted list of (tag, label, model, combos)."""
    entries = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path).replace(".json", "")
        idx = name.find("_n")
        if idx < 0:
            continue
        tag = name[:idx]
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        model = data.get("model", tag).split("/")[-1]
        for label, combos in data.get("results", {}).items():
            entries.append((tag, label, model, combos))
    return entries


def fmt(v, width=8):
    if v is None or v == 0:
        return " " * width
    return f"{v:>{width}.1f}"


def print_breakdown_table(entries, out, scoring_filter="compress"):
    """Print detailed breakdown table filtered to one scoring method."""
    out.write("=" * 90 + "\n")
    out.write(f"  Benchmark 1: Per-Component Breakdown\n")
    out.write(f"  Scoring: {scoring_filter} (score_compression=normal-64x64, kappa=4096)\n")
    out.write(f"  Configs: n=8 T=512 m=1, n=2 T=1024 m=1 (same across all models)\n")
    out.write("=" * 90 + "\n")
    out.write(f"""
  Components:
    forward    — merged-batch forward pass (n+m samples)
    backward   — merged-batch backward pass, containing:
      act_grad   — activation gradient: dL/da = W^T @ dL/de
      compress   — score compressor projection (go, inp) -> R^kappa
      score      — inner product in compressed space
      select     — top-k selection from scores
      w.grad     — weight gradient: dL/dW = dL/de^T @ a (for selected samples)
      retain     — save (go, inp) for post-hoc assembly (1P only)
      autograd   — PyTorch autograd overhead (hooks, graph traversal, alloc)
    Note: embedding layer costs (gather-dot-sum scoring, scatter gradients)
    are folded into the corresponding phases (score, select, w.grad, retain).
    selection  — global top-k across accumulated per-layer scores (subset only)
    assembly   — post-hoc w.grad from retained data (1P only)
    pass1/2    — two-pass variant runs scoring pass then gradient pass (2P only)
    optimizer  — Adam optimizer step
""")

    METHOD_DISPLAY = {
        "standard": "Standard",
        "layerwise": "Layerwise",
        "subset": "Subset 2P",
        "subset_one_pass": "Subset 1P",
    }

    for tag, label, model, combos in entries:
        std_key = "standard/reduced_ghost"
        std_r = combos.get(std_key)
        std_total = compute_total(std_r, "standard")

        out.write(f"\n--- {model} | {label} ---\n")
        if std_total and std_r:
            mem = std_r.get("peak_memory_gb", 0)
            out.write(f"Standard baseline: {std_total:.1f} ms | {mem:.1f} GB\n")

        out.write("\n")

        methods = [
            ("standard", "reduced_ghost"),
            ("layerwise", scoring_filter),
            ("subset", scoring_filter),
            ("subset_one_pass", scoring_filter),
        ]

        for method, scoring in methods:
            key = f"{method}/{scoring}"
            r = combos.get(key)
            total = compute_total(r, method) if r else None
            disp = METHOD_DISPLAY.get(method, method)

            if r is None or total is None:
                out.write(f"{disp:<12s}  {'OOM':>8s}\n")
                continue

            mem = r.get("peak_memory_gb", 0)
            if std_total and method != "standard":
                ovh = f"+{(total / std_total - 1) * 100:.1f}%"
            else:
                ovh = "--"

            optim = r.get("optimizer", 0)

            if method == "subset":
                # Two-pass: fold embedding into corresponding phases
                p1f = r.get("pass1_forward", 0)
                p1b = r.get("pass1_backward", 0)
                p1_act = r.get("p1_act_grad", 0)
                p1_comp = r.get("p1_compress", 0) or 0
                p1_score = (r.get("p1_score", 0) or 0) + (r.get("p1_emb_score", 0) or 0)
                p1_auto = p1b - p1_act - p1_comp - p1_score
                sel = r.get("selection", 0) or 0
                p2f = r.get("pass2_forward", 0)
                p2b = r.get("pass2_backward", 0)
                p2_act = r.get("p2_act_grad", 0) or 0
                p2_wg = r.get("p2_wgrad", 0) or 0
                p2_auto = p2b - p2_act - p2_wg

                out.write(f"{disp:<12s}  pass1_fwd   {p1f:>8.1f}ms\n")
                out.write(f"{'':12s}  pass1_bwd   {p1b:>8.1f}ms\n")
                out.write(f"{'':12s}    act_grad  {p1_act:>8.1f}ms\n")
                if p1_comp > 0.1:
                    out.write(f"{'':12s}    compress  {p1_comp:>8.1f}ms\n")
                out.write(f"{'':12s}    score     {p1_score:>8.1f}ms\n")
                out.write(f"{'':12s}    autograd  {p1_auto:>8.1f}ms\n")
                out.write(f"{'':12s}  selection   {sel:>8.1f}ms\n")
                out.write(f"{'':12s}  pass2_fwd   {p2f:>8.1f}ms\n")
                out.write(f"{'':12s}  pass2_bwd   {p2b:>8.1f}ms\n")
                out.write(f"{'':12s}    act_grad  {p2_act:>8.1f}ms\n")
                out.write(f"{'':12s}    w.grad    {p2_wg:>8.1f}ms\n")
                out.write(f"{'':12s}    autograd  {p2_auto:>8.1f}ms\n")
                out.write(f"{'':12s}  optimizer   {optim:>8.1f}ms\n")
                out.write(f"{'':12s}  TOTAL       {total:>8.1f}ms  {ovh:>9s}  {mem:>5.1f}G\n")
            elif method == "subset_one_pass":
                # Fold embedding into score/retain
                fwd = r.get("forward", 0)
                bwd = r.get("backward", 0)
                act = r.get("act_grad", 0)
                comp = r.get("compress", 0) or 0
                score = (r.get("score", 0) or 0) + (r.get("emb_score", 0) or 0)
                retain = (r.get("retain", 0) or 0) + (r.get("emb_retain", 0) or 0)
                auto = bwd - act - comp - score - retain
                sel = r.get("selection", 0) or 0
                asm = r.get("wgrad", 0) or 0

                out.write(f"{disp:<12s}  forward     {fwd:>8.1f}ms\n")
                out.write(f"{'':12s}  backward    {bwd:>8.1f}ms\n")
                out.write(f"{'':12s}    act_grad  {act:>8.1f}ms\n")
                if comp > 0.1:
                    out.write(f"{'':12s}    compress  {comp:>8.1f}ms\n")
                out.write(f"{'':12s}    score     {score:>8.1f}ms\n")
                if retain > 0.1:
                    out.write(f"{'':12s}    retain    {retain:>8.1f}ms\n")
                out.write(f"{'':12s}    autograd  {auto:>8.1f}ms\n")
                out.write(f"{'':12s}  selection   {sel:>8.1f}ms\n")
                out.write(f"{'':12s}  assembly    {asm:>8.1f}ms\n")
                out.write(f"{'':12s}  optimizer   {optim:>8.1f}ms\n")
                out.write(f"{'':12s}  TOTAL       {total:>8.1f}ms  {ovh:>9s}  {mem:>5.1f}G\n")
            else:
                # Standard and Layerwise: fold embedding into score/select/wgrad
                fwd = r.get("forward", 0)
                bwd = r.get("backward", 0)
                act = r.get("act_grad", 0)
                comp = r.get("compress", 0) or 0
                score = (r.get("score", 0) or 0) + (r.get("emb_score", 0) or 0)
                sel = (r.get("select", 0) or 0) + (r.get("emb_select", 0) or 0)
                wg = (r.get("wgrad", 0) or 0) + (r.get("emb_wgrad", 0) or 0)
                sw = r.get("select_wgrad", 0) or 0
                measured = act + comp + score + sel + wg + sw
                auto = bwd - measured

                out.write(f"{disp:<12s}  forward     {fwd:>8.1f}ms\n")
                out.write(f"{'':12s}  backward    {bwd:>8.1f}ms\n")
                out.write(f"{'':12s}    act_grad  {act:>8.1f}ms\n")
                if comp > 0.1:
                    out.write(f"{'':12s}    compress  {comp:>8.1f}ms\n")
                if score > 0.1:
                    out.write(f"{'':12s}    score     {score:>8.1f}ms\n")
                if sel > 0.1:
                    out.write(f"{'':12s}    select    {sel:>8.1f}ms\n")
                if wg > 0.1:
                    out.write(f"{'':12s}    w.grad    {wg:>8.1f}ms\n")
                if sw > 0.1:
                    out.write(f"{'':12s}    sel+wgrad {sw:>8.1f}ms\n")
                out.write(f"{'':12s}    autograd  {auto:>8.1f}ms\n")
                out.write(f"{'':12s}  optimizer   {optim:>8.1f}ms\n")
                out.write(f"{'':12s}  TOTAL       {total:>8.1f}ms  {ovh:>9s}  {mem:>5.1f}G\n")
        out.write("\n")


def print_overhead_summary(entries, out, scoring_filter="compress"):
    """Print compact overhead summary table."""
    out.write("=" * 90 + "\n")
    out.write(f"  Overhead Summary (scoring={scoring_filter})\n")
    out.write("=" * 90 + "\n\n")

    header = (f"  {'Model':<15s}  {'Config':<22s}  {'Std (ms)':>9s}  "
              f"{'Layerwise':>11s}  {'Subset 2P':>11s}  {'Subset 1P':>11s}  "
              f"{'1P Mem':>7s}")
    out.write(header + "\n")
    out.write("  " + "-" * (len(header) - 2) + "\n")

    for tag, label, model, combos in entries:
        std_r = combos.get("standard/reduced_ghost")
        std_total = compute_total(std_r, "standard")
        if std_total is None:
            continue

        parts = [f"  {model:<15s}", f"{label:<22s}", f"{std_total:>8.0f}ms"]
        for method in ["layerwise", "subset", "subset_one_pass"]:
            key = f"{method}/{scoring_filter}"
            r = combos.get(key)
            total = compute_total(r, method)
            if total is not None:
                ovh = (total / std_total - 1) * 100
                parts.append(f"{ovh:>+10.1f}%")
            else:
                parts.append(f"{'OOM':>11s}")

        # 1P memory
        r_1p = combos.get(f"subset_one_pass/{scoring_filter}")
        if r_1p:
            mem = r_1p.get("peak_memory_gb", 0)
            parts.append(f"{mem:>6.1f}G")
        else:
            parts.append(f"{'':>7s}")

        out.write("  ".join(parts) + "\n")
    out.write("\n")


def print_onepass_speedup(entries, out, scoring_filter="compress"):
    """Print 1P vs 2P speedup."""
    out.write("=" * 70 + "\n")
    out.write(f"  One-Pass Speedup vs Two-Pass (scoring={scoring_filter})\n")
    out.write("=" * 70 + "\n\n")

    header = f"  {'Model':<15s}  {'Config':<22s}  {'2P (ms)':>9s}  {'1P (ms)':>9s}  {'Speedup':>8s}"
    out.write(header + "\n")
    out.write("  " + "-" * (len(header) - 2) + "\n")

    for tag, label, model, combos in entries:
        t_2p = compute_total(combos.get(f"subset/{scoring_filter}"), "subset")
        t_1p = compute_total(combos.get(f"subset_one_pass/{scoring_filter}"), "subset_one_pass")
        if t_2p and t_1p:
            speedup = (1 - t_1p / t_2p) * 100
            out.write(f"  {model:<15s}  {label:<22s}  {t_2p:>8.0f}ms  {t_1p:>8.0f}ms  {speedup:>7.1f}%\n")
        else:
            out.write(f"  {model:<15s}  {label:<22s}  {'OOM':>9s}  {'OOM':>9s}\n")
    out.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="SFT/benchmark/results/breakdown")
    parser.add_argument("--output", default="SFT/benchmark/results/breakdown/breakdown_tables.txt")
    parser.add_argument("--scoring", default="compress")
    args = parser.parse_args()

    entries = load_results(args.results_dir)
    if not entries:
        print(f"No results in {args.results_dir}")
        sys.exit(1)

    # Filter to clean configs only (at least 4 methods OK)
    clean = [(tag, label, model, combos) for tag, label, model, combos in entries
             if sum(1 for v in combos.values() if v is not None) >= 4]

    with open(args.output, "w") as out:
        print_overhead_summary(clean, out, args.scoring)
        print_onepass_speedup(clean, out, args.scoring)
        print_breakdown_table(clean, out, args.scoring)

    print(f"Written to: {args.output}")
    print(f"Clean configs: {len(clean)}/{len(entries)}")


if __name__ == "__main__":
    main()
