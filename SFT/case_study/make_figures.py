"""Paper figures: Figures/SFT/case_study/{magnitude,correlation}_<setting>.pdf and legend.pdf (fig:case-study, appendix
adxsubsubsec:case-study): per-layer scores of Layer-Wise Subset Update, averaged over training steps and seeds, by transformer block
and layer type; left = mean absolute score (log scale), right = Spearman rank correlation rho_S with the Global Subset ranking.

Experiments run on : one node with 4x A40
Launcher           : SFT/case_study/run.sh --train <pool> --task <target> --percentage <pct> --seed <seed> (manifest case_study.txt of
                     SFT/train/qa_matrix_manifest.py with SFT/train/slurm/qa_array.sbatch): full-parameter Llama-3.2-1B, lr 1e-5, the
                     step budget of the setting, both selection rules scored at every step without being applied
Results read from  : <results root>/SFT/runs_v2/case_study/<pool>_<target>-Llama-3.2-1B-p<pct>-lr1e-05-b8-v16-s<seed>/selection_records.json
Seeds / statistics : 42, 2, 22, 62, 82; per (block, layer type) the per-step values of all seeds are pooled; lines = mean, bands =
                     25th-75th percentile; the annotation marks the down_proj peak (magnitude ratio to the next type; rho_S at the peak)
Usage              : python SFT/case_study/make_figures.py [--results-root DIR] [--paper-root DIR] [--venues ICLR] [--out DIR] [--summary]

Style and layout follow the notebook version of the figures (SFT/case_study/result.ipynb).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tables.common import paper_root, results_root  # noqa: E402

plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 12
plt.style.use("seaborn-v0_8-darkgrid")

SEEDS = [42, 2, 22, 62, 82]
SETTINGS = [("alpaca_samsum", "0.4"), ("less_tydiqa", "0.005"), ("triviaqa_nq_open", "0.05"), ("less_squad", "0.005")]   # (run prefix, percentage)
LAYER_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TYPE_LABELS = {"q_proj": "Q", "k_proj": "K", "v_proj": "V", "o_proj": "O", "gate_proj": "Gate", "up_proj": "Up", "down_proj": "Down"}
COLORS = {"q_proj": "#E41A1C", "k_proj": "#FF7F00", "v_proj": "#FFD700", "o_proj": "#984EA3", "gate_proj": "#377EB8", "up_proj": "#4DAF4A", "down_proj": "#A65628"}
ATTN = {"q_proj": "o", "k_proj": "s", "v_proj": "^", "o_proj": "v"}
MARKERS = {**ATTN, "gate_proj": "D", "up_proj": "P", "down_proj": "*"}
RED = "#CC0000"
BBOX = dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=RED, linewidth=1.2, alpha=0.95)
ARROW = dict(arrowstyle="->", color=RED, lw=1.8, connectionstyle="arc3,rad=-0.2", shrinkA=0, shrinkB=14)


# --------------------------------------------------------------------------------------------------------------- reading
def classify_layers(names):
    info = {}
    for i, name in enumerate(names):
        parts = name.split(".")
        block = next((int(parts[j + 1]) for j, p in enumerate(parts) if p == "layers" and j + 1 < len(parts)), -1)
        ltype = next((t for t in LAYER_TYPES if t in name), "lm_head" if name == "lm_head" else "unknown")
        info[i] = (block, ltype)
    return info


def load_records(path: Path):
    """Per (block, type): the per-step mean |score| and the per-step Spearman rho with the Global Subset scores of one run."""
    data = json.loads(path.read_text())
    info = classify_layers(data["metadata"]["layer_names"])
    bs = data["metadata"]["train_batch_size"]
    nb = max(b for b, _ in info.values()) + 1
    mag = {b: {t: [] for t in LAYER_TYPES} for b in range(nb)}
    rho = {b: {t: [] for t in LAYER_TYPES} for b in range(nb)}
    for sd in data["steps"]:
        glob_scores = np.array(sd["subset"]["selection"]["scores"])
        for layer in sd["layer_wise_subset"]["layers"]:
            scores = layer.get("scores", [])
            block, ltype = info[layer["layer_idx"]]
            if len(scores) != bs or ltype not in LAYER_TYPES or block < 0:
                continue
            s = np.array(scores)
            mag[block][ltype].append(float(np.mean(np.abs(s))))
            rho[block][ltype].append(float(spearmanr(s, glob_scores)[0]))
    return dict(mag=mag, rho=rho, num_blocks=nb, n_steps=len(data["steps"]), batch_size=bs)


def merge(records):
    """Pool the per-step values of all seeds (more samples -> tighter percentile bands)."""
    nb = records[0]["num_blocks"]
    out = dict(mag={b: {t: [] for t in LAYER_TYPES} for b in range(nb)}, rho={b: {t: [] for t in LAYER_TYPES} for b in range(nb)},
               num_blocks=nb, n_steps=sum(r["n_steps"] for r in records), batch_size=records[0]["batch_size"], n_seeds=len(records))
    for r in records:
        for b in range(nb):
            for t in LAYER_TYPES:
                out["mag"][b][t] += r["mag"][b][t]
                out["rho"][b][t] += r["rho"][b][t]
    return out


# -------------------------------------------------------------------------------------------------------------- plotting
def style_ax(ax):
    ax.grid(True, alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(axis="both", which="major", labelsize=20, direction="out", length=5)


def curves(ax, table, blocks):
    """Mean line and 25-75 % band per layer type; returns {type: [mean per block]}."""
    means_by_type = {}
    for t in LAYER_TYPES:
        means, lo, hi = [], [], []
        for b in blocks:
            vals = table[b][t]
            means.append(np.mean(vals) if vals else 0)
            lo.append(np.percentile(vals, 25) if vals else 0)
            hi.append(np.percentile(vals, 75) if vals else 0)
        means_by_type[t] = means
        ax.plot(blocks, means, color=COLORS[t], linewidth=2.2, marker=MARKERS[t], markersize=6, linestyle="-" if t in ATTN else "--")
        ax.fill_between(blocks, lo, hi, color=COLORS[t], alpha=0.12)
    return means_by_type


def new_fig():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def magnitude_figure(res, path: Path):
    blocks = list(range(res["num_blocks"]))
    fig, ax = new_fig()
    means = curves(ax, res["mag"], blocks)
    down = means["down_proj"]
    pb = int(np.argmax(down))
    peak = down[pb]
    others = {t: means[t][pb] for t in LAYER_TYPES if t != "down_proj"}
    second = max(others, key=others.get)
    ratio = peak / others[second]
    ax.scatter([pb, pb], [peak, others[second]], s=450, facecolors="none", edgecolors=RED, linewidths=2.5, zorder=10)
    ax.plot([pb, pb], [others[second], peak], color=RED, lw=1.5, ls=":", alpha=0.7, zorder=9)
    mid = np.sqrt(peak * others[second])
    ax.annotate(f"{ratio:.0f}$\\times$", xy=(pb, mid), xytext=(pb + 3.5, mid), fontsize=16, fontfamily="serif", color=RED, ha="left", va="center",
                bbox=BBOX, arrowprops=dict(arrowstyle="->", color=RED, lw=1.8, connectionstyle="arc3,rad=0.15", shrinkA=0, shrinkB=2))
    ax.set_xlabel("Transformer Block", fontsize=24)
    ax.set_ylabel("Average Score", fontsize=24)
    ax.set_yscale("log")
    ax.set_xticks(blocks)
    style_ax(ax)
    plt.tight_layout()
    save(fig, path)
    return second, ratio


def correlation_figure(res, path: Path):
    blocks = list(range(res["num_blocks"]))
    fig, ax = new_fig()
    means = curves(ax, res["rho"], blocks)
    pb = int(np.argmax(means["down_proj"]))
    peak = means["down_proj"][pb]
    ax.scatter([pb], [peak], s=450, facecolors="none", edgecolors=RED, linewidths=2.5, zorder=10)
    ax.annotate(f"$\\rho_S$ = {peak:.2f}", xy=(pb, peak), xytext=(pb + 4, peak - 0.18), fontsize=16, fontfamily="serif", color=RED,
                arrowprops=ARROW, ha="center", va="top", bbox=BBOX)
    ax.set_xlabel("Transformer Block", fontsize=24)
    ax.set_ylabel("Rank Correlation", fontsize=24)
    ax.set_xticks(blocks)
    ax.set_ylim(-0.15, 1.05)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    style_ax(ax)
    plt.tight_layout()
    save(fig, path)


def legend_figure(path: Path):
    fig, ax = plt.subplots(figsize=(10, 0.6))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    handles = [plt.Line2D([0], [0], color=COLORS[t], linewidth=2.2, marker=MARKERS[t], markersize=7, linestyle="-" if t in ATTN else "--") for t in LAYER_TYPES]
    labels = [f"{TYPE_LABELS[t]} {'[Attn]' if t in ATTN else '[MLP]'}" for t in LAYER_TYPES]
    leg = fig.legend(handles, labels, loc="center", ncol=len(LAYER_TYPES), fontsize=20, frameon=True, edgecolor="black", fancybox=False, framealpha=1.0, facecolor="white")
    leg.get_frame().set_linewidth(1.0)
    save(fig, path)


def summary(key, res):
    """Type-averaged magnitude (with the down_proj ratio) and Spearman rho_S with the Global Subset ranking, the numbers quoted in the text."""
    print(f"\n{key}: {res['n_seeds']} seeds x {res['n_steps'] // res['n_seeds']} steps, batch {res['batch_size']}, {res['num_blocks']} blocks")
    mag = {t: np.mean(sum((res["mag"][b][t] for b in range(res["num_blocks"])), [])) for t in LAYER_TYPES}
    rho = {t: np.mean(sum((res["rho"][b][t] for b in range(res["num_blocks"])), [])) for t in LAYER_TYPES}
    for t in LAYER_TYPES:
        print(f"  {TYPE_LABELS[t]:<5s} mean |score| {mag[t]:.6f} (Down is {mag['down_proj'] / mag[t]:.0f}x)   rho_S with Global Subset {rho[t]:.3f}")


# ------------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=None)
    ap.add_argument("--paper-root", default=None)
    ap.add_argument("--venues", default="ICLR")
    ap.add_argument("--out", default=None, help="write the PDFs here instead of <paper root>/<venue>/Figures/SFT/case_study")
    ap.add_argument("--summary", action="store_true", help="print the type-averaged magnitudes and correlations")
    args = ap.parse_args()
    runs = results_root(args.results_root) / "SFT" / "runs_v2" / "case_study"
    outs = [Path(args.out)] if args.out else [paper_root(args.paper_root) / v / "Figures" / "SFT" / "case_study" for v in args.venues.split(",") if v]
    missing = 0
    for prefix, pct in SETTINGS:
        recs = []
        for seed in SEEDS:
            p = runs / f"{prefix}-Llama-3.2-1B-p{pct}-lr1e-05-b8-v16-s{seed}" / "selection_records.json"
            if p.exists():
                recs.append(load_records(p))
            else:
                missing += 1
                print(f"[missing] {p}")
        if not recs:
            continue
        res = merge(recs)
        print(f"{prefix}: {len(recs)}/{len(SEEDS)} seeds")
        for out in outs:
            second, ratio = magnitude_figure(res, out / f"magnitude_{prefix}.pdf")
            correlation_figure(res, out / f"correlation_{prefix}.pdf")
        print(f"  down_proj peak is {ratio:.1f}x the next type ({TYPE_LABELS[second]})")
        if args.summary:
            summary(prefix, res)
    for out in outs:
        legend_figure(out / "legend.pdf")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
