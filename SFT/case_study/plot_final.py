#!/usr/bin/env python3
"""
Final paper figure for case study.

Top row:    Heatmaps (one per dataset)
Bottom row: Per-layer-group feature preference bars (one per dataset)

Feature preference: For each step, compute Spearman(layer_scores, feature)
for each layer separately, then average across steps. Group into early/mid/late.
Show as grouped bars with base rate reference.
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr


plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


def extract_user_text(raw):
    clean = raw.replace('<|begin_of_text|>', '').strip()
    if '<|user|>' in clean:
        user_part = clean.split('<|user|>')[-1]
        if '<|assistant|>' in user_part:
            user_part = user_part.split('<|assistant|>')[0]
        return user_part.strip()
    return clean


def get_nonlatin_scripts(text):
    scripts = set()
    for c in text[:500]:
        cp = ord(c)
        if 0x0600 <= cp <= 0x06FF: scripts.add('arabic')
        elif 0x4E00 <= cp <= 0x9FFF: scripts.add('chinese')
        elif 0x0400 <= cp <= 0x04FF: scripts.add('cyrillic')
        elif 0x0900 <= cp <= 0x097F: scripts.add('devanagari')
        elif 0x3040 <= cp <= 0x30FF: scripts.add('japanese')
        elif 0xAC00 <= cp <= 0xD7AF: scripts.add('korean')
    return scripts


def compute_common_features(train_text, val_text):
    """Unified feature set for both datasets. Same features, same order."""
    train_user = extract_user_text(train_text)
    tl = train_user.lower()
    vl = extract_user_text(val_text).lower()

    t_scripts = get_nonlatin_scripts(train_text)
    v_scripts = get_nonlatin_scripts(val_text)

    stopwords = {'the','a','an','is','are','was','were','in','on','at','to','for','of',
                 'and','or','but','not','with','this','that','it','be','as','by','from',
                 'has','have','had','do','does','did','will','would','can','could'}
    t_words = set(re.findall(r'[a-zA-Z]+', tl)) - stopwords
    v_words = set(re.findall(r'[a-zA-Z]+', vl)) - stopwords
    word_overlap = len(t_words & v_words) / max(1, len(t_words | v_words))

    return {
        # Ordered lexical → structural → semantic → irrelevant
        'Word overlap': word_overlap,
        'QA format': float(('question' in tl or '?' in train_user[:300]) and
                           ('context' in tl or 'passage' in tl or 'given' in tl[:100])),
        'Conversation': float(any(k in train_text for k in [':\n', 'said', 'asked', 'replied',
                                                              'Person', 'User:', 'Assistant:'])),
        'Has examples': float(any(k in tl for k in ['example:', 'input:', 'output:', 'e.g.',
                                                      'for instance', 'for example'])),
        'Has code': float(any(k in train_text for k in ['def ', 'class ', '```', 'function ', 'import '])),
    }


def smooth(arr, w=7):
    if len(arr) <= w:
        return arr
    return np.convolve(arr, np.ones(w)/w, mode='valid')


def analyze(records_path, feature_fn):
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

    # ── Heatmap ──
    block_pairs_corr = defaultdict(list)
    for sd in steps:
        block_scores = defaultdict(lambda: np.zeros(batch_size))
        block_counts = defaultdict(int)
        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores or len(scores) != batch_size:
                continue
            block = sel_to_block.get(li, -1)
            if block < 0:
                continue
            block_scores[block] += np.array(scores)
            block_counts[block] += 1
        blocks = sorted(b for b in block_scores if block_counts[b] > 0)
        for b in blocks:
            block_scores[b] /= block_counts[b]
        for i, bi in enumerate(blocks):
            for bj in blocks[i+1:]:
                rho, _ = spearmanr(block_scores[bi], block_scores[bj])
                block_pairs_corr[(bi, bj)].append(rho)

    all_blocks = sorted(set(b for pair in block_pairs_corr for b in pair))
    n_blocks = len(all_blocks)
    corr_matrix = np.ones((n_blocks, n_blocks))
    for (bi, bj), corrs in block_pairs_corr.items():
        i = all_blocks.index(bi)
        j = all_blocks.index(bj)
        corr_matrix[i, j] = np.mean(corrs)
        corr_matrix[j, i] = np.mean(corrs)

    # ── Feature-score correlation: per-step Spearman averaged across steps ──
    feature_names = None
    n_third = num_layers // 3

    # layer_idx -> feature -> list of per-step Spearman rho
    layer_step_corrs = defaultdict(lambda: defaultdict(list))
    # Subset scores -> feature correlation
    subset_step_corrs = defaultdict(list)

    for sd in steps:
        val_text = sd['val_samples'][0]
        sample_feats = [feature_fn(tt, val_text) for tt in sd['train_samples']]
        if feature_names is None:
            feature_names = list(sample_feats[0].keys())

        # Layerwise
        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores or len(scores) != batch_size:
                continue

            scores_arr = np.array(scores)
            for fname in feature_names:
                fvals = np.array([sf[fname] for sf in sample_feats])
                if fvals.std() < 1e-10:
                    continue
                rho, _ = spearmanr(scores_arr, fvals)
                layer_step_corrs[li][fname].append(rho)

        # Subset
        subset_scores = sd['subset']['selection'].get('scores', [])
        if subset_scores and len(subset_scores) == batch_size:
            subset_arr = np.array(subset_scores)
            for fname in feature_names:
                fvals = np.array([sf[fname] for sf in sample_feats])
                if fvals.std() < 1e-10:
                    continue
                rho, _ = spearmanr(subset_arr, fvals)
                subset_step_corrs[fname].append(rho)

    layer_indices = sorted(layer_step_corrs.keys())

    # Compute mean per-step Spearman for each layer
    feature_corr_by_layer = {}
    for fname in feature_names:
        corrs = []
        for li in layer_indices:
            vals = layer_step_corrs[li].get(fname, [])
            corrs.append(np.mean(vals) if vals else 0.0)
        feature_corr_by_layer[fname] = np.array(corrs)

    # Group into early/mid/late + subset
    feature_group_corr = {}
    for fname in feature_names:
        c = feature_corr_by_layer[fname]
        feature_group_corr[fname] = {
            'early': np.mean(c[:n_third]),
            'middle': np.mean(c[n_third:2*n_third]),
            'late': np.mean(c[2*n_third:]),
            'subset': np.mean(subset_step_corrs[fname]) if subset_step_corrs[fname] else 0.0,
        }

    return {
        'corr_matrix': corr_matrix,
        'all_blocks': all_blocks,
        'feature_names': feature_names,
        'feature_corr_by_layer': feature_corr_by_layer,
        'feature_group_corr': feature_group_corr,
        'layer_indices': layer_indices,
        'num_layers': num_layers,
        'batch_size': batch_size,
        'n_steps': len(steps),
    }


def main():
    scratch = "/scratch/pbb/Project/Dr.Post-Training/SFT"
    datasets = [
        {
            'path': f"{scratch}/case_study_tulu3_tydiqa-Llama-3.2-1B-p0.005-lr3.61e-05-b16-v8-s42/selection_records.json",
            'label': r'tulu3 $\rightarrow$ tydiqa',
            'feature_fn': compute_common_features,
        },
        {
            'path': f"{scratch}/case_study_alpaca_samsum-Llama-3.2-1B-p0.2-lr1.47e-06-b16-v8-s42/selection_records.json",
            'label': r'alpaca $\rightarrow$ samsum',
            'feature_fn': compute_common_features,
        },
    ]

    results = []
    for ds in datasets:
        print(f"Analyzing {ds['label']}...")
        results.append(analyze(ds['path'], ds['feature_fn']))

    # ── Figure: 2 rows × 2 cols ──
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.8, 1.2], wspace=0.4, hspace=0.55)

    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    pi = 0

    group_colors = {'early': '#2166AC', 'middle': '#4DAF4A', 'late': '#B2182B', 'subset': '#984EA3'}

    for row, (ds, res) in enumerate(zip(datasets, results)):
        # ── Left: heatmap ──
        ax_h = fig.add_subplot(gs[row, 0])
        ax_h.set_facecolor('white')

        cm = res['corr_matrix']
        n_b = len(res['all_blocks'])
        offdiag = cm[~np.eye(n_b, dtype=bool)]
        im = ax_h.imshow(cm, cmap='RdBu_r', vmin=offdiag.min(), vmax=1.0, aspect='equal')
        ax_h.set_xticks(range(n_b))
        ax_h.set_yticks(range(n_b))
        ax_h.set_xticklabels(res['all_blocks'], fontsize=8)
        ax_h.set_yticklabels(res['all_blocks'], fontsize=8)
        if row == 1:
            ax_h.set_xlabel('Transformer Block')
        ax_h.set_ylabel('Transformer Block')
        ax_h.set_title(f'{panel_labels[pi]} {ds["label"]}\nBlock ranking correlation',
                       fontsize=12, pad=8)
        cbar = fig.colorbar(im, ax=ax_h, shrink=0.82, pad=0.02)
        cbar.ax.tick_params(labelsize=8)
        if row == 0:
            cbar.set_label('Spearman $\\rho$', fontsize=10)
        pi += 1

        for spine in ax_h.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')

        # ── Right: grouped bar chart of feature preference by layer group ──
        ax_b = fig.add_subplot(gs[row, 1])
        ax_b.set_facecolor('white')

        # Filter features with any signal
        fnames = [f for f in res['feature_names']
                  if max(abs(res['feature_group_corr'][f]['early']),
                         abs(res['feature_group_corr'][f]['middle']),
                         abs(res['feature_group_corr'][f]['late'])) > 0.003]

        n_feats = len(fnames)
        x = np.arange(n_feats)
        width = 0.19
        groups = [('Early', 'early'), ('Middle', 'middle'), ('Late', 'late'), ('Subset', 'subset')]

        for gi, (gname, gkey) in enumerate(groups):
            vals = [res['feature_group_corr'][f][gkey] for f in fnames]
            bars = ax_b.bar(x + (gi - 1.5) * width, vals, width,
                           color=group_colors[gkey], edgecolor='black', linewidth=0.5,
                           label=gname, alpha=0.85)

        ax_b.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(fnames, fontsize=9, ha='right', rotation=45)
        ax_b.set_ylabel('Mean within-step\nSpearman $\\rho$ (score, feature)')
        ax_b.set_title(f'{panel_labels[pi]} {ds["label"]}\nFeature preference by layer group',
                       fontsize=12, pad=8)
        ax_b.legend(loc='best', frameon=True, edgecolor='black',
                    fancybox=False, framealpha=1.0, facecolor='white', fontsize=9,
                    ncol=4)
        pi += 1

        for spine in ax_b.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')

    fig.patch.set_facecolor('white')

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / 'figures'
    save_dir.mkdir(exist_ok=True)

    path = save_dir / 'fig_case_study_final.pdf'
    fig.savefig(str(path), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close(fig)

    # Print summary
    for ds, res in zip(datasets, results):
        print(f"\n{ds['label']} (b={res['batch_size']}, {res['n_steps']} steps):")
        for fname in res['feature_names']:
            gc = res['feature_group_corr'][fname]
            if max(abs(gc['early']), abs(gc['middle']), abs(gc['late'])) > 0.003:
                print(f"  {fname.replace(chr(10),' '):<35s}: E={gc['early']:+.4f}  M={gc['middle']:+.4f}  L={gc['late']:+.4f}  (E-L={gc['early']-gc['late']:+.4f})")


if __name__ == "__main__":
    main()
