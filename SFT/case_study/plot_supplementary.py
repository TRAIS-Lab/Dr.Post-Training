#!/usr/bin/env python3
"""
Supplementary figures (2 rows × 4 cols):
  (a,e) Score decisiveness by layer depth
  (b,f) Block score magnitude (who dominates Subset)
  (c,g) Pairwise ranking flip rate by block distance
  (d,h) Attention vs MLP feature preference comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr


plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


def smooth(arr, w=7):
    if len(arr) <= w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid')


def analyze(records_path):
    with open(records_path) as f:
        data = json.load(f)

    steps = data['steps']
    num_layers = data['metadata']['num_layers']
    layer_names = data['metadata']['layer_names']
    batch_size = data['metadata']['train_batch_size']

    sel_to_block = {}
    for i, name in enumerate(layer_names):
        parts = name.split('.')
        for j, p in enumerate(parts):
            if p == 'layers' and j + 1 < len(parts):
                sel_to_block[i] = int(parts[j + 1])
                break

    # ── Score std per layer ──
    layer_score_std = defaultdict(list)
    for sd in steps:
        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if scores and len(scores) == batch_size:
                layer_score_std[li].append(np.std(scores))

    layer_indices = sorted(layer_score_std.keys())
    mean_stds = np.array([np.mean(layer_score_std[li]) for li in layer_indices])

    # ── Block magnitude ──
    block_magnitudes = defaultdict(list)
    for sd in steps:
        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores:
                continue
            block = sel_to_block.get(li, -1)
            if block < 0:
                continue
            block_magnitudes[block].append(np.mean(np.abs(scores)))

    all_blocks = sorted(block_magnitudes.keys())
    block_means = np.array([np.mean(block_magnitudes[b]) for b in all_blocks])
    block_shares = block_means / block_means.sum() * 100

    # ── Pairwise flip rate by block distance ──
    # For each pair of blocks at distance d, what fraction of sample pairs
    # have their ranking flipped?
    block_pair_flips = defaultdict(lambda: {'flip': 0, 'total': 0})

    for sd in steps:
        # Compute per-block mean scores
        bscores = defaultdict(lambda: np.zeros(batch_size))
        bcounts = defaultdict(int)
        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores or len(scores) != batch_size:
                continue
            block = sel_to_block.get(li, -1)
            if block < 0:
                continue
            bscores[block] += np.array(scores)
            bcounts[block] += 1

        valid = sorted(b for b in bscores if bcounts[b] > 0)
        for b in valid:
            bscores[b] /= bcounts[b]

        # Count flips for each block pair
        for i_b, bi in enumerate(valid):
            for bj in valid[i_b+1:]:
                dist = abs(bi - bj)
                for si in range(batch_size):
                    for sj in range(si+1, batch_size):
                        bi_prefers_si = bscores[bi][si] > bscores[bi][sj]
                        bj_prefers_si = bscores[bj][si] > bscores[bj][sj]
                        block_pair_flips[dist]['total'] += 1
                        if bi_prefers_si != bj_prefers_si:
                            block_pair_flips[dist]['flip'] += 1

    distances = sorted(block_pair_flips.keys())
    flip_rates = [block_pair_flips[d]['flip'] / block_pair_flips[d]['total'] * 100
                  for d in distances]

    return {
        'layer_indices': layer_indices,
        'mean_stds': mean_stds,
        'all_blocks': all_blocks,
        'block_means': block_means,
        'block_shares': block_shares,
        'num_layers': num_layers,
        'distances': distances,
        'flip_rates': flip_rates,
    }


def main():
    scratch = "/scratch/pbb/Project/Gradient-Streaming/SFT"
    datasets = [
        (f"{scratch}/case_study_tulu3_tydiqa-Llama-3.2-1B-p0.005-lr3.61e-05-b16-v8-s42/selection_records.json",
         r'tulu3 $\rightarrow$ tydiqa'),
        (f"{scratch}/case_study_alpaca_samsum-Llama-3.2-1B-p0.2-lr1.47e-06-b16-v8-s42/selection_records.json",
         r'alpaca $\rightarrow$ samsum'),
    ]

    results = [analyze(p) for p, _ in datasets]
    labels = [l for _, l in datasets]

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 3, wspace=0.4, hspace=0.45)

    panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    pi = 0

    for row, (res, label) in enumerate(zip(results, labels)):
        n_third = res['num_layers'] // 3

        # ── Col 0: Score decisiveness ──
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor('white')

        li = res['layer_indices']
        stds = res['mean_stds']
        s = smooth(stds, w=7)
        x_s = li[3:3+len(s)]
        ax.plot(x_s, s, color='#2166AC', linewidth=2.5)
        ax.set_yscale('log')

        if row == 1:
            ax.set_xlabel('Layer Index')
        ax.set_ylabel('Score std (log)')
        ax.set_title(f'{panel_labels[pi]} {label}\nScore decisiveness', fontsize=11, pad=6)

        early_std = np.mean(stds[:n_third])
        late_std = np.mean(stds[2*n_third:])
        ax.text(0.97, 0.95, f'Early/Late = {early_std/late_std:.1f}x',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')
        pi += 1

        # ── Col 1: Block magnitude ──
        ax = fig.add_subplot(gs[row, 1])
        ax.set_facecolor('white')

        blocks = res['all_blocks']
        shares = res['block_shares']
        colors = ['#B2182B' if s > 20 else '#FF7F00' if s > 5 else '#2166AC' for s in shares]
        ax.bar(blocks, shares, color=colors, edgecolor='black', linewidth=0.6)
        max_idx = np.argmax(shares)
        ax.text(blocks[max_idx], shares[max_idx] + 1, f'{shares[max_idx]:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        if row == 1:
            ax.set_xlabel('Transformer Block')
        ax.set_ylabel('Score magnitude share (%)')
        ax.set_title(f'{panel_labels[pi]} {label}\nBlock contribution to Subset', fontsize=11, pad=6)
        ax.set_xticks(blocks)

        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')
        pi += 1

        # ── Col 2: Pairwise flip rate by distance ──
        ax = fig.add_subplot(gs[row, 2])
        ax.set_facecolor('white')

        dists = res['distances']
        rates = res['flip_rates']
        ax.plot(dists, rates, color='#B2182B', linewidth=2.5, marker='o', markersize=5)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1.0)
        ax.text(max(dists) * 0.95, 51, '50% (random)', fontsize=9, ha='right', va='bottom', color='gray')

        if row == 1:
            ax.set_xlabel('Block Distance')
        ax.set_ylabel('Pairwise ranking flip rate (%)')
        ax.set_title(f'{panel_labels[pi]} {label}\nRanking disagreement vs distance', fontsize=11, pad=6)
        ax.set_ylim(20, 55)

        # Annotate adjacent vs max distance
        ax.text(0.03, 0.05, f'd=1: {rates[0]:.1f}%\nd={dists[-1]}: {rates[-1]:.1f}%',
                transform=ax.transAxes, fontsize=9, va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))

        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')
        pi += 1

    fig.patch.set_facecolor('white')

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / 'figures'
    save_dir.mkdir(exist_ok=True)

    path = save_dir / 'fig_supplementary.pdf'
    fig.savefig(str(path), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)

    for res, label in zip(results, labels):
        n_third = res['num_layers'] // 3
        e = np.mean(res['mean_stds'][:n_third])
        l = np.mean(res['mean_stds'][2*n_third:])
        print(f"\n{label}:")
        print(f"  Score std: Early={e:.4f}, Late={l:.4f}, ratio={e/l:.1f}x")
        print(f"  Top block: Block {res['all_blocks'][np.argmax(res['block_shares'])]} = {res['block_shares'].max():.1f}%")
        print(f"  Flip rate: d=1 → {res['flip_rates'][0]:.1f}%, d={res['distances'][-1]} → {res['flip_rates'][-1]:.1f}%")


if __name__ == "__main__":
    main()
