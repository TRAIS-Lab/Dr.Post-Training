#!/usr/bin/env python
"""
Aggregate benchmark results into formatted tables and save to txt.

Usage:
    python SFT/benchmark/aggregate_txt.py
    python SFT/benchmark/aggregate_txt.py --results-dir SFT/benchmark/results --output results.txt
"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────

METHODS = ["standard", "layerwise", "subset", "subset_one_pass"]
SCORINGS = ["compress", "ghost_greats", "ghost", "direct"]
METHOD_DISPLAY = {
    "standard": "Standard",
    "layerwise": "Layerwise",
    "subset": "Subset 2P",
    "subset_one_pass": "Subset 1P",
}

# Model dimensions: tag -> (hidden_size, intermediate_size, num_layers,
#                           num_heads, num_kv_heads, head_dim)
MODEL_DIMS = {
    "qwen3-0.6b":   {"h": 1024, "i": 3072,  "L": 28, "heads": 16, "kv": 8,  "hd": 128},
    "qwen3-1.7b":   {"h": 2048, "i": 6144,  "L": 28, "heads": 16, "kv": 8,  "hd": 128},
    "qwen3-4b":     {"h": 2560, "i": 9728,  "L": 36, "heads": 32, "kv": 8,  "hd": 128},
    "qwen3-8b":     {"h": 4096, "i": 12288, "L": 36, "heads": 32, "kv": 8,  "hd": 128},
    "llama-3.2-3b": {"h": 3072, "i": 8192,  "L": 28, "heads": 24, "kv": 8,  "hd": 128},
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_all_results(results_dir):
    """Load all JSON result files.
    Returns: {tag: {config_label: {method/scoring: result_dict}, "_model": name}}
    """
    models = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        basename = os.path.basename(path)
        # Parse filename: tag_nX_TY_mZ.json
        name = basename.replace(".json", "")
        # Find _n pattern to split tag from config
        idx = name.find("_n")
        if idx < 0:
            continue
        tag = name[:idx]
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        model_name = data.get("model", tag)
        models[tag]["_model"] = model_name
        for label, combos in data.get("results", {}).items():
            models[tag][label] = combos
    return dict(models)


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


def get_score_cost(r):
    if r is None:
        return None
    return r.get("score", 0) + r.get("compress", 0) + r.get("p1_score", 0) + r.get("p1_compress", 0)


def parse_config(label):
    parts = {}
    for tok in label.split():
        k, v = tok.split("=")
        parts[k] = int(v)
    return parts.get("n"), parts.get("T"), parts.get("m")


def sort_configs(configs):
    return sorted([c for c in configs if c != "_model"], key=lambda l: parse_config(l))


def compute_vstar(dims, T):
    """Compute effective V* = sum(O*I) / (S * sum(O+I)) across all linear layers."""
    h, i, L = dims["h"], dims["i"], dims["L"]
    heads, kv, hd = dims["heads"], dims["kv"], dims["hd"]
    q_dim = heads * hd
    kv_dim = kv * hd

    # Per-layer linear dimensions: (output_dim, input_dim)
    layers = [
        (q_dim, h),    # q_proj
        (kv_dim, h),   # k_proj
        (kv_dim, h),   # v_proj
        (h, q_dim),    # o_proj
        (i, h),        # gate_proj
        (i, h),        # up_proj
        (h, i),        # down_proj
    ]
    sum_oi = sum(o * inp for o, inp in layers) * L
    sum_opI = sum(o + inp for o, inp in layers) * L
    return sum_oi / (T * sum_opI) if T > 0 else 0


# ── Table generators ────────────────────────────────────────────────────────

def table_overhead(all_models, out):
    """Table 1: Subset 1P Overhead vs Standard."""
    out.write("=" * 100 + "\n")
    out.write("  TABLE 1: Subset 1P Overhead vs Standard\n")
    out.write("=" * 100 + "\n\n")

    header = f"{'Model':<15s}  {'Config':<20s}  {'Std (ms)':>9s}"
    for s in SCORINGS:
        header += f"  {s:>13s}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    for tag in sorted(all_models):
        data = all_models[tag]
        model = data["_model"].split("/")[-1]
        for label in sort_configs(data):
            combos = data[label]
            std_r = combos.get("standard/ghost")
            std_total = compute_total(std_r, "standard") if std_r else None
            if std_total is None:
                continue
            line = f"{model:<15s}  {label:<20s}  {std_total:>8.0f}ms"
            for scoring in SCORINGS:
                r = combos.get(f"subset_one_pass/{scoring}")
                total = compute_total(r, "subset_one_pass")
                if total is not None:
                    ovh = (total / std_total - 1) * 100
                    line += f"  {ovh:>+12.1f}%"
                else:
                    line += f"  {'OOM':>13s}"
            out.write(line + "\n")
        out.write("\n")


def table_score_cost(all_models, out):
    """Table 2: Score Cost (ms) for each scoring method."""
    out.write("=" * 100 + "\n")
    out.write("  TABLE 2: Score Cost (ms) — Subset 1P\n")
    out.write("=" * 100 + "\n\n")

    header = f"{'Model':<15s}  {'Config':<20s}"
    for s in SCORINGS:
        header += f"  {s:>13s}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    for tag in sorted(all_models):
        data = all_models[tag]
        model = data["_model"].split("/")[-1]
        for label in sort_configs(data):
            combos = data[label]
            line = f"{model:<15s}  {label:<20s}"
            for scoring in SCORINGS:
                r = combos.get(f"subset_one_pass/{scoring}")
                cost = get_score_cost(r)
                if cost is not None:
                    line += f"  {cost:>12.1f}ms"
                else:
                    line += f"  {'OOM':>13s}"
            out.write(line + "\n")
        out.write("\n")


def table_onepass_speedup(all_models, out):
    """Table 3: One-Pass Speedup vs Two-Pass."""
    out.write("=" * 100 + "\n")
    out.write("  TABLE 3: One-Pass Speedup vs Two-Pass\n")
    out.write("=" * 100 + "\n\n")

    header = f"{'Model':<15s}  {'Config':<20s}"
    for s in SCORINGS:
        header += f"  {s:>13s}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    for tag in sorted(all_models):
        data = all_models[tag]
        model = data["_model"].split("/")[-1]
        for label in sort_configs(data):
            combos = data[label]
            line = f"{model:<15s}  {label:<20s}"
            for scoring in SCORINGS:
                t_2p = compute_total(combos.get(f"subset/{scoring}"), "subset")
                t_1p = compute_total(combos.get(f"subset_one_pass/{scoring}"), "subset_one_pass")
                if t_2p and t_1p and t_2p > 0:
                    speedup = (1 - t_1p / t_2p) * 100
                    line += f"  {speedup:>12.1f}%"
                else:
                    line += f"  {'—':>13s}"
            out.write(line + "\n")
        out.write("\n")


def table_memory(all_models, out):
    """Table 4: Peak Memory (GB)."""
    out.write("=" * 100 + "\n")
    out.write("  TABLE 4: Peak Memory (GB)\n")
    out.write("=" * 100 + "\n\n")

    header = f"{'Model':<15s}  {'Config':<20s}  {'Standard':>10s}  {'Layerwise':>12s}  {'Subset 2P':>12s}  {'Subset 1P':>12s}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    for tag in sorted(all_models):
        data = all_models[tag]
        model = data["_model"].split("/")[-1]
        for label in sort_configs(data):
            combos = data[label]
            std_r = combos.get("standard/ghost")
            std_mem = f"{std_r.get('peak_memory_gb', 0):.1f}" if std_r else "—"
            parts = [f"{model:<15s}", f"{label:<20s}", f"{std_mem:>10s}"]
            for method in ["layerwise", "subset", "subset_one_pass"]:
                mems = []
                for s in SCORINGS:
                    r = combos.get(f"{method}/{s}")
                    if r and "peak_memory_gb" in r:
                        mems.append(r["peak_memory_gb"])
                if mems:
                    parts.append(f"{min(mems):.1f}–{max(mems):.1f}".rjust(12))
                else:
                    parts.append("—".rjust(12))
            out.write("  ".join(parts) + "\n")
        out.write("\n")


def table_crossover(all_models, out):
    """Table 5: Ghost vs Ghost_greats Crossover Validation."""
    out.write("=" * 100 + "\n")
    out.write("  TABLE 5: Ghost vs Ghost_greats Crossover Validation\n")
    out.write("=" * 100 + "\n\n")

    header = f"{'Model':<15s}  {'Config':<20s}  {'ghost (ms)':>11s}  {'greats (ms)':>12s}  {'V*':>6s}  {'m/V*':>6s}  {'Predicted':>13s}  {'Actual':>13s}  {'Match':>5s}"
    out.write(header + "\n")
    out.write("-" * len(header) + "\n")

    for tag in sorted(all_models):
        data = all_models[tag]
        model = data["_model"].split("/")[-1]
        dims = MODEL_DIMS.get(tag)
        for label in sort_configs(data):
            combos = data[label]
            n, T, m = parse_config(label)
            r_g = combos.get("subset_one_pass/ghost")
            r_gg = combos.get("subset_one_pass/ghost_greats")
            cost_g = get_score_cost(r_g)
            cost_gg = get_score_cost(r_gg)
            if cost_g is None and cost_gg is None:
                continue

            vstar = compute_vstar(dims, T) if dims else None
            mv = f"{m / vstar:.2f}" if vstar else "—"

            if cost_g is not None and cost_gg is not None:
                if abs(cost_gg / cost_g - 1) < 0.05:
                    actual = "~tie"
                elif cost_gg < cost_g:
                    actual = "ghost_greats"
                else:
                    actual = "ghost"
            else:
                actual = "OOM"

            predicted = "ghost_greats" if (vstar and m < vstar) else "ghost" if vstar else "—"
            match = "Y" if actual == predicted or actual == "~tie" else ("" if actual == "OOM" else "N")

            g_str = f"{cost_g:.1f}" if cost_g else "OOM"
            gg_str = f"{cost_gg:.1f}" if cost_gg else "OOM"
            vs_str = f"{vstar:.2f}" if vstar else "—"

            out.write(f"{model:<15s}  {label:<20s}  {g_str:>10s}ms  {gg_str:>11s}ms  {vs_str:>6s}  {mv:>6s}  {predicted:>13s}  {actual:>13s}  {match:>5s}\n")
        out.write("\n")


def table_breakdown(all_models, out):
    """Table 6: Detailed per-component breakdown for each model/config."""
    out.write("=" * 100 + "\n")
    out.write("  TABLE 6: Detailed Per-Component Breakdown\n")
    out.write("=" * 100 + "\n")

    for tag in sorted(all_models):
        data = all_models[tag]
        model = data["_model"].split("/")[-1]

        for label in sort_configs(data):
            combos = data[label]
            n, T, m = parse_config(label)
            std_r = combos.get("standard/ghost")
            std_total = compute_total(std_r, "standard") if std_r else None

            out.write(f"\n--- {model} | {label} ---\n")
            if std_total:
                std_mem = std_r.get("peak_memory_gb", 0)
                out.write(f"Standard: {std_total:.1f} ms | {std_mem:.2f} GB\n\n")

            header = (f"{'Method':<12s}  {'Scoring':<13s}  {'forward':>8s}  {'act_grad':>8s}  "
                      f"{'compress':>8s}  {'score':>8s}  {'select':>8s}  {'w.grad':>8s}  "
                      f"{'p2_fwd':>8s}  {'p2_bwd':>8s}  {'assembly':>8s}  {'optim':>8s}  "
                      f"{'Total':>9s}  {'Ovhd':>8s}  {'Mem':>7s}")
            out.write(header + "\n")
            out.write("-" * len(header) + "\n")

            for method in METHODS:
                scorings = ["--"] if method == "standard" else SCORINGS
                for scoring in scorings:
                    key = f"{method}/{'ghost' if scoring == '--' else scoring}"
                    r = combos.get(key)
                    total = compute_total(r, method) if r else None

                    if r is None or total is None:
                        out.write(f"{METHOD_DISPLAY[method]:<12s}  {scoring:<13s}  {'OOM':>8s}\n")
                        continue

                    mem = r.get("peak_memory_gb", 0)
                    ovh = f"+{(total / std_total - 1) * 100:.1f}%" if std_total and method != "standard" else "--"

                    def fmt(v):
                        return f"{v:>8.1f}" if v else f"{'':>8s}"

                    fwd = r.get("forward", r.get("pass1_forward", 0))
                    act = r.get("act_grad", r.get("p1_act_grad", 0))
                    comp = r.get("compress", r.get("p1_compress", 0)) or 0
                    score = r.get("score", r.get("p1_score", 0)) or 0
                    sel = r.get("select", r.get("selection", 0)) or 0
                    wg = r.get("wgrad", r.get("p2_wgrad", 0)) or 0
                    p2f = r.get("pass2_forward", 0) or 0
                    p2b = r.get("pass2_backward", 0) or 0
                    asm = 0
                    if method == "subset_one_pass":
                        asm = r.get("wgrad", 0) or 0
                        wg = 0  # wgrad is assembly for 1P
                        sel = r.get("selection", 0) or 0

                    out.write(
                        f"{METHOD_DISPLAY[method]:<12s}  {scoring:<13s}  "
                        f"{fmt(fwd)}  {fmt(act)}  {fmt(comp) if comp > 0.1 else fmt(0)}  "
                        f"{fmt(score) if score > 0.1 else fmt(0)}  "
                        f"{fmt(sel) if sel > 0.1 else fmt(0)}  "
                        f"{fmt(wg) if wg > 0.1 else fmt(0)}  "
                        f"{fmt(p2f) if p2f > 0.1 else fmt(0)}  "
                        f"{fmt(p2b) if p2b > 0.1 else fmt(0)}  "
                        f"{fmt(asm) if asm > 0.1 else fmt(0)}  "
                        f"{fmt(r.get('optimizer', 0))}  "
                        f"{total:>8.1f}ms  {ovh:>8s}  {mem:>6.1f}G\n"
                    )
            out.write("\n")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="SFT/benchmark/results")
    parser.add_argument("--output", default="SFT/benchmark/results/benchmark_tables.txt")
    args = parser.parse_args()

    all_models = load_all_results(args.results_dir)
    if not all_models:
        print(f"No results found in {args.results_dir}")
        sys.exit(1)

    tags = sorted(all_models.keys())
    total_configs = sum(len(sort_configs(all_models[t])) for t in tags)
    total_combos = sum(
        sum(1 for v in all_models[t][c].values() if v is not None and not isinstance(v, str))
        for t in tags for c in sort_configs(all_models[t])
    )

    with open(args.output, "w") as out:
        out.write("=" * 100 + "\n")
        out.write("  Dr. Post-Training — Benchmark Results\n")
        out.write(f"  Models: {', '.join(all_models[t]['_model'] for t in tags)}\n")
        out.write(f"  Configs: {total_configs} total, {total_combos} successful combos\n")
        out.write("=" * 100 + "\n\n")

        table_overhead(all_models, out)
        table_score_cost(all_models, out)
        table_onepass_speedup(all_models, out)
        table_memory(all_models, out)
        table_crossover(all_models, out)
        table_breakdown(all_models, out)

    print(f"Results written to: {args.output}")
    print(f"Models: {len(tags)}, Configs: {total_configs}")

    # Also print crossover table to stdout for quick check
    import io
    buf = io.StringIO()
    table_crossover(all_models, buf)
    print(buf.getvalue())


if __name__ == "__main__":
    main()
