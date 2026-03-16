#!/usr/bin/env python3
"""
Attention vs MLP: MLP layers are 2-3x more opinionated about sample utility.

Figure: For each feature, show Attn vs MLP bar pairs at each depth group.
Two panels (one per dataset).
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
        u = clean.split('<|user|>')[-1]
        if '<|assistant|>' in u:
            u = u.split('<|assistant|>')[0]
        return u.strip()[:512]
    return clean[:512]


def compute_features(tt, vt):
    tu = extract_user_text(tt)
    tl = tu.lower()
    vl = extract_user_text(vt).lower()
    sw = {'the','a','an','is','are','was','were','in','on','at','to','for','of',
          'and','or','but','not','with','this','that','it','be','as','by','from',
          'has','have','had','do','does','did','will','would','can','could'}
    tw = set(re.findall(r'[a-zA-Z]+', tl)) - sw
    vw = set(re.findall(r'[a-zA-Z]+', vl)) - sw
    return {
        'Word overlap': len(tw & vw) / max(1, len(tw | vw)),
        'QA format': float(('question' in tl or '?' in tu[:300]) and
                           ('context' in tl or 'passage' in tl)),
        'Conversation': float(any(k in tt for k in [':\n', 'said', 'asked', 'replied'])),
        'Has examples': float(any(k in tl for k in ['example:', 'input:', 'output:', 'e.g.'])),
        'Has code': float(any(k in tt for k in ['def ', 'class ', '```', 'function ', 'import '])),
    }


def analyze(records_path):
    with open(records_path) as f:
        data = json.load(f)

    steps = data['steps']
    num_layers = data['metadata']['num_layers']
    layer_names = data['metadata']['layer_names']
    batch_size = data['metadata']['train_batch_size']

    lt = {}
    sb = {}
    for i, name in enumerate(layer_names):
        for j, p in enumerate(name.split('.')):
            if p == 'layers' and j + 1 < len(name.split('.')):
                sb[i] = int(name.split('.')[j + 1])
                break
        lt[i] = 'attn' if 'self_attn' in name else 'mlp'

    nb = max(sb.values()) + 1
    bt = nb // 3

    gc = defaultdict(lambda: defaultdict(list))

    for sd in steps:
        vt = sd['val_samples'][0]
        feats = [compute_features(t, vt) for t in sd['train_samples']]
        fnames = list(feats[0].keys())

        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores or len(scores) != batch_size:
                continue
            if li not in sb:
                continue

            b = sb[li]
            d = 'Early' if b < bt else ('Middle' if b < 2 * bt else 'Late')
            sa = np.array(scores)

            for fn in fnames:
                fv = np.array([f[fn] for f in feats])
                if fv.std() < 1e-10:
                    continue
                rho, _ = spearmanr(sa, fv)
                gc[(d, lt[li])][fn].append(rho)

    # Compute means
    result = {}
    for (depth, ltype), feat_corrs in gc.items():
        for fn, vals in feat_corrs.items():
            result[(depth, ltype, fn)] = np.mean(vals)

    return result, fnames


def main():
    scratch = "/scratch/pbb/Project/Gradient-Streaming/SFT"
    datasets = [
        (f"{scratch}/case_study_tulu3_tydiqa-Llama-3.2-1B-p0.005-lr3.61e-05-b16-v8-s42/selection_records.json",
         r'tulu3 $\rightarrow$ tydiqa'),
        (f"{scratch}/case_study_alpaca_samsum-Llama-3.2-1B-p0.2-lr1.47e-06-b16-v8-s42/selection_records.json",
         r'alpaca $\rightarrow$ samsum'),
    ]

    results = []
    labels = []
    for path, label in datasets:
        print(f"Analyzing {label}...")
        r, fnames = analyze(path)
        results.append(r)
        labels.append(label)

    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    attn_color = '#377EB8'
    mlp_color = '#E41A1C'
    depths = ['Early', 'Middle', 'Late']

    for col, (res, label) in enumerate(zip(results, labels)):
        ax = fig.add_subplot(gs[col])
        ax.set_facecolor('white')

        # For each depth group, show attn vs mlp bars for each feature
        # Layout: features on x-axis, grouped by depth
        # Each feature has 6 bars: Early-Attn, Early-MLP, Mid-Attn, Mid-MLP, Late-Attn, Late-MLP

        n_feats = len(fnames)
        n_groups = 3  # Early, Mid, Late
        width = 0.12
        x = np.arange(n_feats)

        for gi, depth in enumerate(depths):
            offset_attn = (gi - 1) * (2 * width + 0.04) - width / 2
            offset_mlp = offset_attn + width

            attn_vals = [res.get((depth, 'attn', fn), 0) for fn in fnames]
            mlp_vals = [res.get((depth, 'mlp', fn), 0) for fn in fnames]

            attn_label = f'{depth} Attn' if col == 0 else None
            mlp_label = f'{depth} MLP' if col == 0 else None

            # Use different alpha for each depth
            alpha = [0.5, 0.75, 1.0][gi]

            ax.bar(x + offset_attn, attn_vals, width, color=attn_color, alpha=alpha,
                   edgecolor='black', linewidth=0.4, label=attn_label)
            ax.bar(x + offset_mlp, mlp_vals, width, color=mlp_color, alpha=alpha,
                   edgecolor='black', linewidth=0.4, label=mlp_label, hatch='//' if gi == 0 else ('\\\\' if gi == 2 else ''))

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(fnames, fontsize=9, ha='right', rotation=35)
        ax.set_ylabel('Mean within-step\nSpearman $\\rho$ (score, feature)')
        ax.set_title(f'{chr(97 + col)}) {label}', fontsize=13, pad=8)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color('black')

    # Shared legend - simpler: just Attn vs MLP
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=attn_color, edgecolor='black', label='Attention layers'),
        Patch(facecolor=mlp_color, edgecolor='black', label='MLP layers'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2,
               frameon=True, edgecolor='black', fancybox=False, framealpha=1.0,
               facecolor='white', fontsize=11, bbox_to_anchor=(0.5, 1.02))

    fig.patch.set_facecolor('white')

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / 'figures'
    save_dir.mkdir(exist_ok=True)

    path = save_dir / 'fig_attn_vs_mlp.pdf'
    fig.savefig(str(path), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close(fig)

    # Print summary
    for res, label in zip(results, labels):
        print(f"\n{label}:")
        for fn in fnames:
            attn_mean = np.mean([res.get((d, 'attn', fn), 0) for d in depths])
            mlp_mean = np.mean([res.get((d, 'mlp', fn), 0) for d in depths])
            if max(abs(attn_mean), abs(mlp_mean)) > 0.005:
                ratio = abs(mlp_mean / attn_mean) if abs(attn_mean) > 0.001 else float('inf')
                print(f"  {fn:<16s}: Attn={attn_mean:+.4f}  MLP={mlp_mean:+.4f}  MLP/Attn={ratio:.1f}x")


if __name__ == "__main__":
    main()
