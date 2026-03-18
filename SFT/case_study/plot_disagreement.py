#!/usr/bin/env python3
"""
Figure: Top-1 disagreement between Layerwise and Subset selection.

Left column:  Violin plots — rank of Subset's top-1 within each block
Right column: Grouped bars — block agreement & negative score rate
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path


plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


def analyze_disagreement(records_path):
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

    all_blocks = sorted(set(sel_to_block.values()))
    n_blocks = len(all_blocks)

    # Per-step: rank of subset's top-1 in each block, agreement, negative score
    rank_matrix = []  # (n_steps, n_blocks)
    agree_matrix = []  # (n_steps, n_blocks) bool
    neg_matrix = []  # (n_steps, n_blocks) bool
    distinct_top1s = []  # per step: number of distinct top-1 samples across blocks
    # Controversy: rank std across blocks for subset's top-1 vs others
    subset_top1_controversy = []  # per step: rank std of subset's top-1
    other_controversy = []  # per step: list of rank stds for non-top-1 samples

    for sd in steps:
        # Subset top-1
        subset_scores = sd['subset']['selection'].get('scores', [])
        if not subset_scores or len(subset_scores) != batch_size:
            continue
        subset_top1 = int(np.argmax(subset_scores))

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
        if len(valid) != n_blocks:
            continue
        for b in valid:
            bscores[b] /= bcounts[b]

        step_ranks = []
        step_agree = []
        step_neg = []
        block_top1s = set()
        # Full rank matrix for all samples: (batch_size, n_blocks)
        full_ranks = np.zeros((batch_size, n_blocks))
        for bi, b in enumerate(all_blocks):
            scores_b = bscores[b]
            ranking = np.argsort(-scores_b)  # descending
            # Assign rank to each sample
            for r, sample_idx in enumerate(ranking):
                full_ranks[sample_idx, bi] = r + 1
            rank = int(full_ranks[subset_top1, bi])
            step_ranks.append(rank)
            block_top1 = int(ranking[0])
            step_agree.append(block_top1 == subset_top1)
            block_top1s.add(block_top1)
            step_neg.append(scores_b[subset_top1] < 0)

        rank_matrix.append(step_ranks)
        agree_matrix.append(step_agree)
        neg_matrix.append(step_neg)
        distinct_top1s.append(len(block_top1s))

        # Controversy: rank std across blocks for each sample
        sample_rank_stds = np.std(full_ranks, axis=1)  # (batch_size,)
        subset_top1_controversy.append(sample_rank_stds[subset_top1])
        others = [sample_rank_stds[s] for s in range(batch_size) if s != subset_top1]
        other_controversy.extend(others)

    rank_matrix = np.array(rank_matrix)  # (n_steps, n_blocks)
    agree_matrix = np.array(agree_matrix)
    neg_matrix = np.array(neg_matrix)

    agree_rates = agree_matrix.mean(axis=0) * 100  # (n_blocks,)
    neg_rates = neg_matrix.mean(axis=0) * 100
    mean_rank = rank_matrix.mean()
    mean_distinct = np.mean(distinct_top1s)

    # Per-step dynamics
    mean_rank_per_step = rank_matrix.mean(axis=1)  # (n_steps,)
    neg_rate_per_step = neg_matrix.mean(axis=1) * 100  # (n_steps,)

    return {
        'rank_matrix': rank_matrix,
        'agree_rates': agree_rates,
        'neg_rates': neg_rates,
        'mean_rank': mean_rank,
        'mean_distinct': mean_distinct,
        'all_blocks': all_blocks,
        'n_blocks': n_blocks,
        'batch_size': batch_size,
        'n_steps': rank_matrix.shape[0],
        # Dynamics
        'mean_rank_per_step': mean_rank_per_step,
        'neg_rate_per_step': neg_rate_per_step,
        # Controversy
        'subset_top1_controversy': np.array(subset_top1_controversy),
        'other_controversy': np.array(other_controversy),
    }


def main():
    scratch = "/scratch/pbb/Project/Dr.Post-Training/SFT"
    datasets = [
        (f"{scratch}/case_study_tulu3_tydiqa-Llama-3.2-1B-p0.005-lr3.61e-05-b16-v8-s42/selection_records.json",
         r'tulu3 $\rightarrow$ tydiqa'),
        (f"{scratch}/case_study_alpaca_samsum-Llama-3.2-1B-p0.2-lr1.47e-06-b16-v8-s42/selection_records.json",
         r'alpaca $\rightarrow$ samsum'),
    ]

    results = []
    for path, label in datasets:
        print(f"Analyzing {label}...")
        results.append(analyze_disagreement(path))

    # Block-gradient colormap
    cmap = plt.cm.RdBu_r
    block_colors = [cmap(v) for v in np.linspace(0.15, 0.85, 16)]

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 0.9], wspace=0.35, hspace=0.45)

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    pi = 0

    for row, ((_, label), res) in enumerate(zip(datasets, results)):
        blocks = res['all_blocks']
        n_b = res['n_blocks']

        # ── Left: Violin plots — rank of subset's top-1 within each block ──
        ax = fig.add_subplot(gs[row, 0])
        ax.set_facecolor('white')

        # Build data for violinplot: list of arrays, one per block
        violin_data = [res['rank_matrix'][:, bi] for bi in range(n_b)]
        parts = ax.violinplot(violin_data, positions=blocks, showmeans=False,
                              showmedians=False, showextrema=False)

        for i, body in enumerate(parts['bodies']):
            body.set_facecolor(block_colors[i])
            body.set_edgecolor('black')
            body.set_linewidth(0.6)
            body.set_alpha(0.85)

        # Median (white dot) and IQR (thin black line)
        for bi in range(n_b):
            data = violin_data[bi]
            q1, med, q3 = np.percentile(data, [25, 50, 75])
            ax.vlines(blocks[bi], q1, q3, color='black', linewidth=1.5, zorder=3)
            ax.scatter(blocks[bi], med, color='white', edgecolor='black',
                       s=25, zorder=4, linewidth=0.8)

        # Reference line: rank 8.5 ("bottom half" boundary)
        ax.axhline(y=res['batch_size'] / 2 + 0.5, color='gray', linestyle='--',
                    linewidth=1.0, alpha=0.6)

        ax.set_ylim(res['batch_size'] + 0.8, 0.2)  # inverted: rank 1 at top
        ax.set_xticks(blocks)
        if row == 1:
            ax.set_xlabel('Transformer Block')
        ax.set_ylabel('Rank (1 = best)')
        ax.set_title(f'{panel_labels[pi]} {label}\nRank of Subset\'s top-1 within each block',
                      fontsize=12, pad=8)

        ax.text(0.97, 0.05, f'Mean rank: {res["mean_rank"]:.1f} / {res["batch_size"]}',
                transform=ax.transAxes, fontsize=10, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        pi += 1

        # ── Right: Grouped bar — agreement & negative score rate ──
        ax = fig.add_subplot(gs[row, 1])
        ax.set_facecolor('white')

        x = np.arange(n_b)
        width = 0.38

        # Agreement bars (block-gradient colored)
        for i in range(n_b):
            ax.bar(x[i] - width / 2, res['agree_rates'][i], width,
                   color=block_colors[i], edgecolor='black', linewidth=0.5,
                   label='Agreement rate' if i == 0 else None, alpha=0.85)

        # Negative score bars (muted red, hatched)
        ax.bar(x + width / 2, res['neg_rates'], width,
               color='#D6604D', edgecolor='black', linewidth=0.5,
               hatch='//', alpha=0.7, label='Negative score rate')

        # Reference lines
        mean_agree = res['agree_rates'].mean()
        ax.axhline(y=mean_agree, color='#2166AC', linestyle='--', linewidth=1.0,
                    alpha=0.7, label=f'Mean agreement ({mean_agree:.1f}%)')
        ax.axhline(y=100 / res['batch_size'], color='gray', linestyle=':',
                    linewidth=1.0, alpha=0.6, label=f'Random chance ({100/res["batch_size"]:.1f}%)')

        ax.set_xticks(x)
        ax.set_xticklabels(blocks)
        ax.set_ylim(0, max(res['agree_rates'].max(), res['neg_rates'].max()) * 1.25)
        if row == 1:
            ax.set_xlabel('Transformer Block')
        ax.set_ylabel('Percentage (%)')
        ax.set_title(f'{panel_labels[pi]} {label}\nBlock agreement with Subset\'s top-1',
                      fontsize=12, pad=8)

        ax.text(0.97, 0.95, f'Distinct top-1s: {res["mean_distinct"]:.1f} / {res["batch_size"]}',
                transform=ax.transAxes, fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.9))

        ax.legend(loc='upper left', frameon=True, edgecolor='black',
                  fancybox=False, framealpha=1.0, facecolor='white', fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        pi += 1

    fig.patch.set_facecolor('white')

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / 'figures'
    save_dir.mkdir(exist_ok=True)

    path = save_dir / 'fig_disagreement.pdf'
    fig.savefig(str(path), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close(fig)

    # ── Figure 2: Dynamics & Controversy (2 rows × 2 cols) ──
    fig2 = plt.figure(figsize=(14, 10))
    gs2 = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.45)
    panel_labels2 = ['(a)', '(b)', '(c)', '(d)']
    pi2 = 0

    def smooth(arr, w=15):
        if len(arr) <= w:
            return np.arange(len(arr)), arr
        s = np.convolve(arr, np.ones(w) / w, mode='valid')
        x = np.arange(w // 2, w // 2 + len(s))
        return x, s

    for row, ((_, label), res) in enumerate(zip(datasets, results)):
        # ── Left: Dynamics over training ──
        ax = fig2.add_subplot(gs2[row, 0])
        ax.set_facecolor('white')

        x_r, s_r = smooth(res['mean_rank_per_step'])
        ax.plot(x_r, s_r, color='#2166AC', linewidth=2.0, label='Mean rank')
        ax.axhline(y=res['batch_size'] / 2 + 0.5, color='gray', linestyle='--',
                    linewidth=1.0, alpha=0.5, label='Bottom-half boundary')
        ax.set_ylabel('Mean rank of Subset\'s top-1', color='#2166AC')
        ax.set_ylim(1, res['batch_size'])
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelcolor='#2166AC')

        ax2 = ax.twinx()
        x_n, s_n = smooth(res['neg_rate_per_step'])
        ax2.plot(x_n, s_n, color='#D6604D', linewidth=2.0, linestyle='--',
                 label='Negative rate')
        ax2.set_ylabel('Blocks giving negative score (%)', color='#D6604D')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='y', labelcolor='#D6604D')

        if row == 1:
            ax.set_xlabel('Training Step')
        ax.set_title(f'{panel_labels2[pi2]} {label}\nDisagreement dynamics over training',
                      fontsize=12, pad=8)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True,
                  edgecolor='black', fancybox=False, framealpha=1.0,
                  facecolor='white', fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        for spine in ax2.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        pi2 += 1

        # ── Right: Controversy — rank std of subset's top-1 vs others ──
        ax = fig2.add_subplot(gs2[row, 1])
        ax.set_facecolor('white')

        bins = np.linspace(0, res['batch_size'] / 2, 25)
        ax.hist(res['other_controversy'], bins=bins, density=True,
                color='#BDBDBD', edgecolor='black', linewidth=0.5,
                alpha=0.7, label='Other samples')
        ax.hist(res['subset_top1_controversy'], bins=bins, density=True,
                color='#2166AC', edgecolor='black', linewidth=0.5,
                alpha=0.7, label="Subset's top-1")

        mean_sub = res['subset_top1_controversy'].mean()
        mean_oth = res['other_controversy'].mean()
        ax.axvline(x=mean_sub, color='#2166AC', linestyle='--', linewidth=1.5,
                    label=f'Mean top-1 ({mean_sub:.2f})')
        ax.axvline(x=mean_oth, color='#666666', linestyle='--', linewidth=1.5,
                    label=f'Mean other ({mean_oth:.2f})')

        if row == 1:
            ax.set_xlabel('Rank std across blocks')
        ax.set_ylabel('Density')
        ax.set_title(f'{panel_labels2[pi2]} {label}\nSample controversy (rank variability)',
                      fontsize=12, pad=8)
        ax.legend(loc='upper right', frameon=True, edgecolor='black',
                  fancybox=False, framealpha=1.0, facecolor='white', fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')
        pi2 += 1

    fig2.patch.set_facecolor('white')
    path2 = save_dir / 'fig_disagreement_dynamics.pdf'
    fig2.savefig(str(path2), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"Saved: {path2}")
    plt.close(fig2)

    # Print summary stats
    for (_, label), res in zip(datasets, results):
        print(f"\n{label} (b={res['batch_size']}, {res['n_steps']} steps):")
        print(f"  Mean rank of subset's top-1: {res['mean_rank']:.1f} / {res['batch_size']}")
        print(f"  Mean distinct top-1s: {res['mean_distinct']:.1f} / {res['batch_size']}")
        print(f"  Overall agreement rate: {res['agree_rates'].mean():.1f}%")
        print(f"  Overall negative rate: {res['neg_rates'].mean():.1f}%")
        print(f"  Agreement by block: {', '.join(f'{r:.1f}%' for r in res['agree_rates'])}")
        print(f"  Negative by block:  {', '.join(f'{r:.1f}%' for r in res['neg_rates'])}")
        print(f"  Controversy — subset top-1 mean rank std: {res['subset_top1_controversy'].mean():.2f}")
        print(f"  Controversy — other samples mean rank std: {res['other_controversy'].mean():.2f}")
        # Dynamics: first half vs second half
        half = res['n_steps'] // 2
        r1 = res['mean_rank_per_step'][:half].mean()
        r2 = res['mean_rank_per_step'][half:].mean()
        n1 = res['neg_rate_per_step'][:half].mean()
        n2 = res['neg_rate_per_step'][half:].mean()
        print(f"  Dynamics — mean rank: first half {r1:.1f}, second half {r2:.1f}")
        print(f"  Dynamics — neg rate: first half {n1:.1f}%, second half {n2:.1f}%")


if __name__ == "__main__":
    main()
