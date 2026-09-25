#!/usr/bin/env python
"""
Fused-kernel A/B table: drpt.kernels backends (cute, triton) vs the PyTorch reference ops.

Reads the per-run JSONs written by benchmark.py into results/fused_ab/
(named <tag>_n<n>_T<T>_<method>_<scoring>_<ref|triton|cute>[_lt].json) and prints,
per (model, config, method, scoring): step time per backend, the phases the kernels
touch (score / compress / w.grad / assembly), and the overhead over the Standard
(full_training) step.  ``_lt`` variants ran with cuBLASLt preferred for all GEMMs.

    python SFT/benchmark/aggregate_fused_ab.py [--results-dir SFT/benchmark/results/fused_ab]
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

NAME_RE = re.compile(r"^(?P<tag>.+?)_n(?P<n>\d+)_T(?P<T>\d+)_(?P<method>full_training|layer_wise_subset|global_subset_one_pass|global_subset)_(?P<scoring>\w+?)_(?P<variant>ref|triton|cute|fused)(?P<lt>_lt)?\.json$")
BACKENDS = ("triton", "cute")


def total_ms(method, r):
    if method in ("full_training", "layer_wise_subset"):
        return r["forward"] + r["backward"] + r["optimizer"]
    if method == "global_subset_one_pass":
        return r["forward"] + r["backward"] + r.get("selection", 0) + r.get("wgrad", 0) + r["optimizer"]
    if method == "global_subset":
        return (r["pass1_forward"] + r["pass1_backward"] + r.get("selection", 0)
                + r["pass2_forward"] + r["pass2_backward"] + r["optimizer"])
    raise ValueError(method)


def kernel_phases(method, r):
    """Phases whose kernels the fused path replaces (plus compress for context)."""
    if method == "layer_wise_subset":
        return {"score": r.get("score", 0) + r.get("emb_score", 0), "compress": r.get("compress", 0),
                "w.grad": r.get("wgrad", 0) + r.get("emb_wgrad", 0)}
    if method == "global_subset_one_pass":
        return {"score": r.get("score", 0) + r.get("emb_score", 0), "compress": r.get("compress", 0),
                "assembly": r.get("wgrad", 0)}
    if method == "full_training":
        return {"w.grad": r.get("wgrad", 0)}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "fused_ab"))
    args = ap.parse_args()

    runs = defaultdict(dict)  # (tag, n, T) -> {(method, scoring, variant): result}
    for path in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        with open(path) as f:
            r = json.load(f)
        g = m.groupdict()
        variant = ("triton" if g["variant"] == "fused" else g["variant"]) + (g["lt"] or "")
        runs[(g["tag"], int(g["n"]), int(g["T"]))][(g["method"], g["scoring"], variant)] = r

    for (tag, n, T), res in sorted(runs.items()):
        std = next((res[k] for k in res if k[0] == "full_training" and not k[2].endswith("_lt")), None)
        std_lt = next((res[k] for k in res if k[0] == "full_training" and k[2].endswith("_lt")), None)
        std_ms = total_ms("full_training", std) if std else None
        print(f"\n=== {tag}  n={n} T={T} m=1 ===")
        if std:
            line = f"Standard: {std_ms:.0f} ms (w.grad {std['wgrad']:.0f} ms, peak {std['peak_memory_gb']:.1f} GB)"
            if std_lt:
                line += f"   | with cuBLASLt: {total_ms('full_training', std_lt):.0f} ms (w.grad {std_lt['wgrad']:.0f} ms)"
            print(line)
        hdr = (f"{'method':<24}{'scoring':<10}{'ref ms':>8}{'triton':>8}{'cute':>8}   {'ovh ref':>8}{'triton':>8}{'cute':>8}"
               f"   kernel phases ms (ref -> triton -> cute)")
        print(hdr)
        print("-" * len(hdr))
        keys = sorted({(m_, sc) for (m_, sc, v) in res if m_ != "full_training"})

        def fmt_ms(r, method):
            return f"{total_ms(method, r):>8.0f}" if r else f"{'-':>8}"

        def fmt_ovh(r, method):
            return f"{100 * (total_ms(method, r) / std_ms - 1):>+7.1f}%" if (r and std_ms) else f"{'-':>8}"

        for method, scoring in keys:
            ref = res.get((method, scoring, "ref"))
            variants = [res.get((method, scoring, b)) for b in BACKENDS]
            if ref is None and not any(variants):
                continue
            phases = {}
            for r in [ref] + variants:
                if r is not None:
                    for k, v in kernel_phases(method, r).items():
                        phases.setdefault(k, [])
            ph = []
            for k in sorted(phases):
                vals = [kernel_phases(method, r).get(k) if r is not None else None for r in [ref] + variants]
                if all((v or 0) < 0.05 for v in vals if v is not None):
                    continue
                ph.append(f"{k} " + "->".join(f"{v:.0f}" if v is not None else "-" for v in vals))
            print(f"{method:<24}{scoring:<10}{fmt_ms(ref, method)}{''.join(fmt_ms(r, method) for r in variants)}"
                  f"   {fmt_ovh(ref, method)}{''.join(fmt_ovh(r, method) for r in variants)}   {', '.join(ph)}")
            ref_lt = res.get((method, scoring, "ref_lt"))
            if ref_lt:
                print(f"{'':<24}{'(ref+cuBLASLt)':<10}{fmt_ms(ref_lt, method)}{'':>16}   {fmt_ovh(ref_lt, method)}"
                      f"{'':>16}   w.grad {kernel_phases(method, ref_lt).get('w.grad', 0):.0f}")


if __name__ == "__main__":
    main()
