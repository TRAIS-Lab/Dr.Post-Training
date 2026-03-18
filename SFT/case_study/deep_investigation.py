#!/usr/bin/env python3
"""
Formalizing concept-level preferences.

1. Embedding PCA: project samples into embedding space, PCA, check which
   principal components predict each layer's score.
2. Attention vs MLP: do attention and MLP layers within the same block
   prefer different samples?
3. Intra-block diversity: how many unique selection patterns exist within
   each block? (7 layers per block — do they all agree?)
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict, Counter
from pathlib import Path
from scipy.stats import spearmanr

import torch
from transformers import AutoTokenizer, AutoModel


plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 13, 'axes.titlesize': 12,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.family': 'serif',
    'mathtext.fontset': 'cm',
})


def extract_user_text(raw, max_len=512):
    clean = raw.replace('<|begin_of_text|>', '').strip()
    if '<|user|>' in clean:
        user_part = clean.split('<|user|>')[-1]
        if '<|assistant|>' in user_part:
            user_part = user_part.split('<|assistant|>')[0]
        return user_part.strip()[:max_len]
    return clean[:max_len]


def analyze(records_path, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")

    with open(records_path) as f:
        data = json.load(f)

    steps = data['steps']
    num_layers = data['metadata']['num_layers']
    layer_names = data['metadata']['layer_names']
    batch_size = data['metadata']['train_batch_size']
    n_third = num_layers // 3

    # Classify layers as attention vs MLP
    layer_type = {}  # layer_idx -> 'attn' or 'mlp'
    sel_to_block = {}
    for i, name in enumerate(layer_names):
        parts = name.split('.')
        for j, p in enumerate(parts):
            if p == 'layers' and j + 1 < len(parts):
                sel_to_block[i] = int(parts[j + 1])
                break
        if 'self_attn' in name:
            layer_type[i] = 'attn'
        elif any(k in name for k in ['gate_proj', 'up_proj', 'down_proj']):
            layer_type[i] = 'mlp'
        else:
            layer_type[i] = 'other'

    # ══════════════════════════════════════════════════════════════
    # 1. EMBEDDING PCA AS CONCEPT AXES
    # ══════════════════════════════════════════════════════════════
    print(f"\n--- 1. Embedding PCA as Concept Axes ---")
    print(f"  Embedding all samples...")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").cuda().eval()

    # Collect all unique texts and embed them
    # For each step: embed val + train samples, compute per-PC correlation with layer scores
    from sklearn.decomposition import PCA

    # First pass: embed all texts and find PCs
    all_texts = []
    for sd in steps:
        all_texts.append(extract_user_text(sd['val_samples'][0]))
        for tt in sd['train_samples']:
            all_texts.append(extract_user_text(tt))

    # Encode in batches
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(all_texts), 64):
            batch = all_texts[i:i+64]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128,
                           return_tensors='pt').to('cuda')
            out = emb_model(**enc)
            mask = enc['attention_mask'].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)

    # PCA on all embeddings
    n_pcs = 10
    pca = PCA(n_components=n_pcs)
    all_pcs = pca.fit_transform(all_embs)

    print(f"  Variance explained by first {n_pcs} PCs: {pca.explained_variance_ratio_[:n_pcs].sum():.3f}")
    for i in range(min(5, n_pcs)):
        print(f"    PC{i+1}: {pca.explained_variance_ratio_[i]:.3f}")

    # Now for each layer, compute within-step Spearman between scores and each PC
    # PC values for train samples at each step
    layer_pc_corrs = defaultdict(lambda: defaultdict(list))  # layer -> pc_idx -> list of rho

    text_idx = 0
    for si, sd in enumerate(steps):
        val_idx = text_idx
        train_start = text_idx + 1
        text_idx += 1 + batch_size

        # PC values for the train samples in this step
        train_pcs = all_pcs[train_start:train_start + batch_size]  # (batch_size, n_pcs)

        # Also compute: similarity of each train sample to val in embedding space
        val_emb = all_embs[val_idx]
        train_embs = all_embs[train_start:train_start + batch_size]
        sim_to_val = train_embs @ val_emb  # cosine similarity

        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores or len(scores) != batch_size:
                continue
            scores_arr = np.array(scores)

            for pc_idx in range(n_pcs):
                pc_vals = train_pcs[:, pc_idx]
                if pc_vals.std() > 1e-10:
                    rho, _ = spearmanr(scores_arr, pc_vals)
                    layer_pc_corrs[li][pc_idx].append(rho)

            # Also correlate with embedding similarity to val
            if sim_to_val.std() > 1e-10:
                rho, _ = spearmanr(scores_arr, sim_to_val)
                layer_pc_corrs[li]['sim_to_val'].append(rho)

    # Aggregate: for each PC, what's the early/mid/late mean correlation?
    layer_indices = sorted(layer_pc_corrs.keys())
    print(f"\n  Per-layer-group correlation with embedding PCs:")
    print(f"  {'Dimension':<15s} {'Early':>8s} {'Middle':>8s} {'Late':>8s} {'|E-L|':>8s}")
    print(f"  {'-'*48}")

    pc_results = []
    for pc_idx in list(range(n_pcs)) + ['sim_to_val']:
        corrs = [np.mean(layer_pc_corrs[li].get(pc_idx, [0])) for li in layer_indices]
        c = np.array(corrs)
        e = np.mean(c[:n_third])
        m = np.mean(c[n_third:2*n_third])
        l = np.mean(c[2*n_third:])
        name = f'PC{pc_idx+1}' if isinstance(pc_idx, int) else 'Sim to val'
        if max(abs(e), abs(m), abs(l)) > 0.005:
            print(f"  {name:<15s} {e:>+8.4f} {m:>+8.4f} {l:>+8.4f} {abs(e-l):>8.4f}")
            pc_results.append((name, e, m, l, abs(e-l), corrs))

    # Sort by |E-L| to find the most layer-depth-dependent PCs
    pc_results.sort(key=lambda x: -x[4])

    # ══════════════════════════════════════════════════════════════
    # 2. ATTENTION vs MLP WITHIN BLOCKS
    # ══════════════════════════════════════════════════════════════
    print(f"\n--- 2. Attention vs MLP Within Blocks ---")

    # For each block, compute Spearman between attn-layers' mean score and mlp-layers' mean score
    block_attn_mlp_corr = defaultdict(list)

    for sd in steps:
        block_attn_scores = defaultdict(lambda: np.zeros(batch_size))
        block_mlp_scores = defaultdict(lambda: np.zeros(batch_size))
        block_attn_n = defaultdict(int)
        block_mlp_n = defaultdict(int)

        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            scores = layer.get('scores', [])
            if not scores or len(scores) != batch_size:
                continue
            block = sel_to_block.get(li, -1)
            if block < 0:
                continue
            s = np.array(scores)
            lt = layer_type.get(li, 'other')
            if lt == 'attn':
                block_attn_scores[block] += s
                block_attn_n[block] += 1
            elif lt == 'mlp':
                block_mlp_scores[block] += s
                block_mlp_n[block] += 1

        for block in sorted(set(block_attn_n.keys()) & set(block_mlp_n.keys())):
            if block_attn_n[block] > 0 and block_mlp_n[block] > 0:
                attn = block_attn_scores[block] / block_attn_n[block]
                mlp = block_mlp_scores[block] / block_mlp_n[block]
                rho, _ = spearmanr(attn, mlp)
                block_attn_mlp_corr[block].append(rho)

    all_blocks = sorted(block_attn_mlp_corr.keys())
    print(f"  Spearman correlation between Attention and MLP scores (same block):")
    print(f"  {'Block':>6s} {'Attn-MLP rho':>14s}")
    print(f"  {'-'*22}")
    attn_mlp_means = []
    for b in all_blocks:
        m = np.mean(block_attn_mlp_corr[b])
        attn_mlp_means.append(m)
        print(f"  {b:>6d} {m:>14.4f}")

    print(f"\n  Overall mean Attn-MLP within-block correlation: {np.mean(attn_mlp_means):.4f}")
    print(f"  (1.0 = perfect agreement, 0.0 = unrelated)")

    # ══════════════════════════════════════════════════════════════
    # 3. INTRA-BLOCK SELECTION DIVERSITY
    # ══════════════════════════════════════════════════════════════
    print(f"\n--- 3. Intra-Block Selection Diversity ---")

    # For each block: how many unique selection patterns among its ~7 layers?
    block_diversity = defaultdict(list)

    for sd in steps:
        block_patterns = defaultdict(set)
        for layer in sd['layerwise']['layers']:
            li = layer['layer_idx']
            block = sel_to_block.get(li, -1)
            if block < 0:
                continue
            pattern = tuple(sorted(layer['selected_indices']))
            block_patterns[block].add(pattern)

        for block, patterns in block_patterns.items():
            block_diversity[block].append(len(patterns))

    print(f"  Mean unique selection patterns within each block (out of ~7 layers):")
    print(f"  {'Block':>6s} {'Patterns':>10s} {'Max possible':>14s}")
    print(f"  {'-'*32}")
    for b in all_blocks:
        m = np.mean(block_diversity[b])
        n_layers_in_block = sum(1 for li in layer_indices if sel_to_block.get(li) == b)
        print(f"  {b:>6d} {m:>10.2f} {n_layers_in_block:>14d}")

    return {
        'pc_results': pc_results,
        'attn_mlp_means': attn_mlp_means,
        'all_blocks': all_blocks,
        'block_diversity': {b: np.mean(block_diversity[b]) for b in all_blocks},
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
    ds_labels = []
    for path, label in datasets:
        results.append(analyze(path, label))
        ds_labels.append(label)

    # ── Plot: Embedding PC correlation by layer depth ──
    script_dir = Path(__file__).resolve().parent
    save_dir = script_dir / 'figures'
    save_dir.mkdir(exist_ok=True)

    # Pick top 3 PCs with strongest layer-depth gradient for each dataset
    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 2, wspace=0.35)

    for col, (res, label) in enumerate(zip(results, ds_labels)):
        ax = fig.add_subplot(gs[col])
        ax.set_facecolor('white')

        layer_indices = list(range(113))  # approximate
        n_third = 113 // 3
        top_pcs = res['pc_results'][:4]  # top 4 by |E-L|

        colors = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#984EA3']
        for i, (name, e, m, l, diff, corrs) in enumerate(top_pcs):
            # Smooth the per-layer curve
            c = np.array(corrs)
            kernel = np.ones(7) / 7
            if len(c) > 7:
                s = np.convolve(c, kernel, mode='valid')
                x_s = list(range(3, 3 + len(s)))
                ax.plot(x_s, s, color=colors[i % len(colors)], linewidth=2.5,
                        label=f'{name} (E-L={diff:+.3f})')

        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('Layer Index')
        ax.set_ylabel('Spearman $\\rho$ (score, PC)')
        ax.set_title(f'{chr(97+col)}) {label}\nEmbedding concept axes by layer', fontsize=12, pad=8)
        ax.legend(loc='best', frameon=True, edgecolor='black',
                  fancybox=False, framealpha=1.0, facecolor='white', fontsize=9)

        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')

    fig.patch.set_facecolor('white')
    path = save_dir / 'fig_concept_axes.pdf'
    fig.savefig(str(path), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"\nSaved: {path}")
    plt.close(fig)

    # ── Plot: Attention vs MLP within-block correlation ──
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    for res, label, color, marker in zip(results, ds_labels,
                                          ['#2166AC', '#B2182B'], ['o', 's']):
        ax.plot(res['all_blocks'], res['attn_mlp_means'], color=color,
                linewidth=2.5, marker=marker, markersize=6, label=label)

    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.3)
    ax.set_xlabel('Transformer Block')
    ax.set_ylabel('Spearman $\\rho$ (Attention vs MLP scores)')
    ax.set_title('Attention vs MLP agreement within blocks', fontsize=13)
    ax.legend(loc='best', frameon=True, edgecolor='black',
              fancybox=False, framealpha=1.0, facecolor='white')

    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('black')

    path = save_dir / 'fig_attn_vs_mlp.pdf'
    fig.savefig(str(path), format='pdf', facecolor='white', bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
