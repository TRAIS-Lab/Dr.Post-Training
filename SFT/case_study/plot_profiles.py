#!/usr/bin/env python3
"""
Sample utility profiles: cluster samples by how early/mid/late layers rank them.

Figure: For each dataset, show cluster centroids as grouped bars or line profiles,
plus the fraction of samples in each cluster.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.cluster import KMeans


plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


def compute_profiles(records_path):
    with open(records_path) as f:
        data = json.load(f)

    steps = data['steps']
    num_layers = data['metadata']['num_layers']
    batch_size = data['metadata']['train_batch_size']
    n_third = num_layers // 3

    profiles = []
    for sd in steps:
        early_scores = np.zeros(batch_size)
        mid_scores = np.zeros(batch_size)
        late_scores = np.zeros(batch_size)
        n_e, n_m, n_l = 0, 0, 0
        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores or len(scores) != batch_size:
                continue
            s = np.array(scores)
            if li < n_third:
                early_scores += s; n_e += 1
            elif li < 2 * n_third:
                mid_scores += s; n_m += 1
            else:
                late_scores += s; n_l += 1

        if n_e == 0 or n_m == 0 or n_l == 0:
            continue
        early_scores /= n_e
        mid_scores /= n_m
        late_scores /= n_l

        # Z-score within this step
        for arr in [early_scores, mid_scores, late_scores]:
            if arr.std() > 1e-10:
                arr[:] = (arr - arr.mean()) / arr.std()

        for ti in range(batch_size):
            profiles.append([early_scores[ti], mid_scores[ti], late_scores[ti]])

    return np.array(profiles)


def name_cluster(centroid):
    """Name a cluster based on its centroid pattern."""
    e, m, l = centroid
    high = 0.3
    low = -0.3

    if e > high and l > high:
        return 'Universally\npreferred'
    if e < low and l < low:
        return 'Universally\nrejected'
    if e > high and l < low:
        return 'Early-layer\nspecialist'
    if e < low and l > high:
        return 'Late-layer\nspecialist'
    if m > high and e < high and l < high:
        return 'Middle-layer\nspecialist'
    return 'Mixed'


def main():
    scratch = "/scratch/pbb/Project/Dr.Post-Training/SFT"
    datasets = [
        (f"{scratch}/case_study_tulu3_tydiqa-Llama-3.2-1B-p0.005-lr3.61e-05-b16-v8-s42/selection_records.json",
         r'tulu3 $\rightarrow$ tydiqa'),
        (f"{scratch}/case_study_alpaca_samsum-Llama-3.2-1B-p0.2-lr1.47e-06-b16-v8-s42/selection_records.json",
         r'alpaca $\rightarrow$ samsum'),
    ]

    all_profiles = []
    labels = []
    for path, label in datasets:
        print(f"Computing profiles for {label}...")
        all_profiles.append(compute_profiles(path))
        labels.append(label)

    n_clusters = 4

    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    cluster_colors = ['#2166AC', '#B2182B', '#4DAF4A', '#984EA3']
    group_labels = ['Early', 'Middle', 'Late']

    for col, (profiles, label) in enumerate(zip(all_profiles, labels)):
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(profiles)
        centroids = km.cluster_centers_

        # Sort clusters by early-layer score (descending) for consistent ordering
        order = np.argsort(-centroids[:, 0])
        centroids = centroids[order]
        # Remap cluster IDs
        remap = {old: new for new, old in enumerate(order)}
        cluster_ids = np.array([remap[c] for c in cluster_ids])

        ax = fig.add_subplot(gs[col])
        ax.set_facecolor('white')

        x = np.arange(3)  # Early, Middle, Late
        total = len(profiles)
        width = 0.18

        for ci in range(n_clusters):
            n_in_cluster = (cluster_ids == ci).sum()
            pct = n_in_cluster / total * 100
            cname = name_cluster(centroids[ci])
            label_str = f'{cname} ({pct:.0f}%)'

            offsets = np.array([-1.5, -0.5, 0.5, 1.5])
            ax.bar(x + offsets[ci] * width, centroids[ci], width,
                   color=cluster_colors[ci], edgecolor='black', linewidth=0.5,
                   label=label_str, alpha=0.85)

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=12)
        ax.set_ylabel('Mean z-scored utility')
        ax.set_title(f'{chr(97+col)}) {label}', fontsize=13, pad=8)
        ax.legend(loc='lower right', frameon=True, edgecolor='black',
                  fancybox=False, framealpha=1.0, facecolor='white', fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')

        # Print summary
        print(f"\n  {label}:")
        for ci in range(n_clusters):
            n = (cluster_ids == ci).sum()
            c = centroids[ci]
            print(f"    {name_cluster(c).replace(chr(10),' '):<25s}: {n:>5d} ({n/total*100:.1f}%)  "
                  f"E={c[0]:+.3f} M={c[1]:+.3f} L={c[2]:+.3f}")

    fig.patch.set_facecolor('white')

    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / 'figures'
    save_dir.mkdir(exist_ok=True)

    path = save_dir / 'fig_profiles.pdf'
    fig.savefig(str(path), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
