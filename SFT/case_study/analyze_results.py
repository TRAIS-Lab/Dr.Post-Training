#!/usr/bin/env python3
"""
Analyze case study results: Layerwise vs Subset selection comparison.

Computes:
1. Agreement between methods (Jaccard similarity)
2. Per-layer selection diversity in Layerwise
3. Score distributions and correlations
4. Sample-level analysis: which samples are consistently selected/rejected
5. Layer-group analysis: do early/mid/late layers select differently?
"""

import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path


def load_data(path):
    with open(path) as f:
        return json.load(f)


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def analyze(records_path, output_dir):
    data = load_data(records_path)
    meta = data['metadata']
    steps = data['steps']

    num_steps = len(steps)
    num_layers = meta['num_layers']
    batch_size = meta['train_batch_size']
    frac = meta['selection_frac']
    num_selected = max(1, int(batch_size * frac))
    layer_names = meta['layer_names']

    print(f"=" * 70)
    print(f"Case Study Analysis: Layerwise vs Subset Selection")
    print(f"=" * 70)
    print(f"Steps: {num_steps} | Layers: {num_layers} | Batch: {batch_size} | Select: {num_selected}/{batch_size}")
    print()

    # ================================================================
    # 1. AGREEMENT BETWEEN METHODS
    # ================================================================
    print("=" * 70)
    print("1. AGREEMENT: Layerwise vs Subset")
    print("=" * 70)

    # For Layerwise, compare each layer's selection with Subset's global selection
    # Also compute "Layerwise majority vote" — samples selected by >50% of layers
    jaccard_per_step = []
    jaccard_per_layer_per_step = defaultdict(list)  # layer_idx -> list of jaccards
    majority_vote_jaccard = []

    for step_data in steps:
        subset_sel = set(step_data['subset']['selected_indices'])
        lw_layers = step_data['layerwise']['layers']

        # Per-layer Jaccard with Subset
        layer_selection_counts = Counter()
        for layer in lw_layers:
            lw_sel = set(layer['selected_indices'])
            j = jaccard(lw_sel, subset_sel)
            jaccard_per_layer_per_step[layer['layer_idx']].append(j)
            for idx in lw_sel:
                layer_selection_counts[idx] += 1

        # Majority vote: samples selected by >50% of layers
        majority_threshold = num_layers / 2
        majority_sel = {idx for idx, count in layer_selection_counts.items()
                       if count > majority_threshold}
        mv_j = jaccard(majority_sel, subset_sel)
        majority_vote_jaccard.append(mv_j)

        # Average Jaccard across all layers for this step
        step_jaccards = [jaccard(set(l['selected_indices']), subset_sel) for l in lw_layers]
        jaccard_per_step.append(np.mean(step_jaccards))

    print(f"Avg Jaccard (per-layer vs Subset):  {np.mean(jaccard_per_step):.4f} ± {np.std(jaccard_per_step):.4f}")
    print(f"Avg Jaccard (majority-vote vs Subset): {np.mean(majority_vote_jaccard):.4f} ± {np.std(majority_vote_jaccard):.4f}")
    print()

    # Jaccard by layer group (early / middle / late)
    layer_indices = sorted(jaccard_per_layer_per_step.keys())
    n_layers = len(layer_indices)
    third = n_layers // 3
    early = layer_indices[:third]
    middle = layer_indices[third:2*third]
    late = layer_indices[2*third:]

    for group_name, group_indices in [("Early", early), ("Middle", middle), ("Late", late)]:
        group_jaccards = []
        for idx in group_indices:
            group_jaccards.extend(jaccard_per_layer_per_step[idx])
        print(f"  {group_name} layers ({len(group_indices)} layers): "
              f"Jaccard = {np.mean(group_jaccards):.4f} ± {np.std(group_jaccards):.4f}")
    print()

    # ================================================================
    # 2. LAYERWISE SELECTION DIVERSITY
    # ================================================================
    print("=" * 70)
    print("2. LAYERWISE SELECTION DIVERSITY")
    print("=" * 70)

    unique_patterns_per_step = []
    for step_data in steps:
        lw_layers = step_data['layerwise']['layers']
        patterns = set()
        for layer in lw_layers:
            patterns.add(tuple(sorted(layer['selected_indices'])))
        unique_patterns_per_step.append(len(patterns))

    print(f"Unique selection patterns per step:")
    print(f"  Mean: {np.mean(unique_patterns_per_step):.1f} / {num_layers} layers")
    print(f"  Min:  {np.min(unique_patterns_per_step)} | Max: {np.max(unique_patterns_per_step)}")
    print()

    # How often does a layer agree with its neighbors?
    neighbor_agreements = []
    for step_data in steps:
        lw_layers = step_data['layerwise']['layers']
        # Sort by layer_idx for sequential comparison
        sorted_layers = sorted(lw_layers, key=lambda x: x['layer_idx'])
        for i in range(len(sorted_layers) - 1):
            j = jaccard(
                set(sorted_layers[i]['selected_indices']),
                set(sorted_layers[i+1]['selected_indices'])
            )
            neighbor_agreements.append(j)

    print(f"Adjacent layer agreement (Jaccard): {np.mean(neighbor_agreements):.4f} ± {np.std(neighbor_agreements):.4f}")
    print()

    # ================================================================
    # 3. SCORE DISTRIBUTIONS
    # ================================================================
    print("=" * 70)
    print("3. SCORE DISTRIBUTIONS")
    print("=" * 70)

    # Subset scores across all steps
    all_subset_scores = []
    for step_data in steps:
        scores = step_data['subset']['selection'].get('scores', [])
        if scores:
            all_subset_scores.append(scores)

    if all_subset_scores:
        all_scores_flat = np.array(all_subset_scores)
        print(f"Subset (global) scores across {len(all_scores_flat)} steps:")
        print(f"  Mean: {all_scores_flat.mean():.4f} | Std: {all_scores_flat.std():.4f}")
        print(f"  Range: [{all_scores_flat.min():.4f}, {all_scores_flat.max():.4f}]")
        print()

        # Score spread: how differentiated are samples?
        score_ranges = all_scores_flat.max(axis=1) - all_scores_flat.min(axis=1)
        print(f"  Score spread (max-min per step): {score_ranges.mean():.4f} ± {score_ranges.std():.4f}")

        # Score correlation between consecutive steps
        if len(all_scores_flat) > 1:
            step_corrs = []
            for i in range(len(all_scores_flat) - 1):
                # Different batches each step, so correlation isn't meaningful across steps
                # Instead, compute within-step rank correlation
                pass
        print()

    # Layerwise: score variance across layers for same sample
    print("Layerwise per-sample score variance across layers:")
    per_sample_layer_score_vars = []
    for step_data in steps[:50]:  # Sample first 50 steps
        lw_layers = step_data['layerwise']['layers']
        # Build matrix: [num_layers, batch_size]
        score_matrix = []
        for layer in lw_layers:
            if 'scores' in layer:
                score_matrix.append(layer['scores'])
        if score_matrix:
            score_matrix = np.array(score_matrix)  # [num_layers, batch_size]
            per_sample_vars = score_matrix.var(axis=0)  # variance across layers per sample
            per_sample_layer_score_vars.append(per_sample_vars.mean())

    if per_sample_layer_score_vars:
        print(f"  Mean variance: {np.mean(per_sample_layer_score_vars):.6f}")
        print(f"  This indicates how much a sample's score varies across layers")
    print()

    # ================================================================
    # 4. SAMPLE-LEVEL ANALYSIS
    # ================================================================
    print("=" * 70)
    print("4. SAMPLE-LEVEL ANALYSIS (per batch position)")
    print("=" * 70)

    # How often is each batch position selected by Subset vs Layerwise?
    subset_position_counts = Counter()
    lw_position_counts = Counter()  # across all layers
    lw_majority_position_counts = Counter()

    for step_data in steps:
        # Subset
        for idx in step_data['subset']['selected_indices']:
            subset_position_counts[idx] += 1

        # Layerwise: count how many layers select each position
        lw_layers = step_data['layerwise']['layers']
        layer_counts = Counter()
        for layer in lw_layers:
            for idx in layer['selected_indices']:
                lw_position_counts[idx] += 1
                layer_counts[idx] += 1

        for idx, count in layer_counts.items():
            if count > num_layers / 2:
                lw_majority_position_counts[idx] += 1

    print("Batch position selection frequency (across all steps):")
    print(f"  Position | Subset sel% | LW layer-avg sel% | LW majority%")
    print(f"  " + "-" * 55)
    for pos in range(batch_size):
        ss_pct = subset_position_counts[pos] / num_steps * 100
        lw_pct = lw_position_counts[pos] / (num_steps * num_layers) * 100
        lw_maj_pct = lw_majority_position_counts[pos] / num_steps * 100
        print(f"  {pos:8d} | {ss_pct:10.1f}% | {lw_pct:16.1f}% | {lw_maj_pct:11.1f}%")
    print()

    # ================================================================
    # 5. TEMPORAL ANALYSIS: How do selections change over training?
    # ================================================================
    print("=" * 70)
    print("5. TEMPORAL ANALYSIS")
    print("=" * 70)

    # Split training into phases
    phase_size = num_steps // 4
    phases = [
        ("Phase 1 (steps 1-{})".format(phase_size), steps[:phase_size]),
        ("Phase 2 (steps {}-{})".format(phase_size+1, 2*phase_size), steps[phase_size:2*phase_size]),
        ("Phase 3 (steps {}-{})".format(2*phase_size+1, 3*phase_size), steps[2*phase_size:3*phase_size]),
        ("Phase 4 (steps {}-{})".format(3*phase_size+1, num_steps), steps[3*phase_size:]),
    ]

    print("Method agreement over training phases:")
    for phase_name, phase_steps in phases:
        phase_jaccards = []
        phase_diversity = []
        for step_data in phase_steps:
            subset_sel = set(step_data['subset']['selected_indices'])
            lw_layers = step_data['layerwise']['layers']

            # Average Jaccard for this step
            step_j = [jaccard(set(l['selected_indices']), subset_sel) for l in lw_layers]
            phase_jaccards.append(np.mean(step_j))

            # Diversity
            patterns = set(tuple(sorted(l['selected_indices'])) for l in lw_layers)
            phase_diversity.append(len(patterns))

        if phase_jaccards:
            print(f"  {phase_name}:")
            print(f"    LW-Subset Jaccard: {np.mean(phase_jaccards):.4f}")
            print(f"    LW unique patterns: {np.mean(phase_diversity):.1f}")
    print()

    # ================================================================
    # 6. LAYER-GROUP SCORE ANALYSIS
    # ================================================================
    print("=" * 70)
    print("6. LAYER-GROUP ANALYSIS: Score patterns")
    print("=" * 70)

    # For a few example steps, show how scores differ across layer groups
    example_steps = [0, num_steps // 4, num_steps // 2, 3 * num_steps // 4, num_steps - 1]
    example_steps = [min(i, num_steps - 1) for i in example_steps]

    for step_idx in example_steps:
        step_data = steps[step_idx]
        lw_layers = step_data['layerwise']['layers']
        sorted_layers = sorted(lw_layers, key=lambda x: x['layer_idx'])

        # Get early/mid/late layer scores
        n = len(sorted_layers)
        early_scores = np.array([l['scores'] for l in sorted_layers[:n//3]])
        mid_scores = np.array([l['scores'] for l in sorted_layers[n//3:2*n//3]])
        late_scores = np.array([l['scores'] for l in sorted_layers[2*n//3:]])

        # Mean score per sample across layer groups
        early_mean = early_scores.mean(axis=0)
        mid_mean = mid_scores.mean(axis=0)
        late_mean = late_scores.mean(axis=0)

        # Correlation between layer groups
        early_mid_corr = np.corrcoef(early_mean, mid_mean)[0, 1]
        mid_late_corr = np.corrcoef(mid_mean, late_mean)[0, 1]
        early_late_corr = np.corrcoef(early_mean, late_mean)[0, 1]

        print(f"  Step {step_data['step']}:")
        print(f"    Score correlation: Early-Mid={early_mid_corr:.3f}, "
              f"Mid-Late={mid_late_corr:.3f}, Early-Late={early_late_corr:.3f}")

        # Which samples are consistently preferred/rejected across all layer groups?
        # Rank by mean score in each group
        early_rank = np.argsort(early_mean)[::-1]
        late_rank = np.argsort(late_mean)[::-1]
        subset_sel = step_data['subset']['selected_indices']
        print(f"    Early layers top-{num_selected}: {sorted(early_rank[:num_selected].tolist())}")
        print(f"    Late layers top-{num_selected}:  {sorted(late_rank[:num_selected].tolist())}")
        print(f"    Subset selected:        {sorted(subset_sel)}")
    print()

    # ================================================================
    # 7. QUALITATIVE: Show example batches with divergent selections
    # ================================================================
    print("=" * 70)
    print("7. QUALITATIVE EXAMPLES: High-divergence batches")
    print("=" * 70)

    # Find steps where Layerwise majority vote disagrees most with Subset
    divergent_steps = []
    for i, step_data in enumerate(steps):
        subset_sel = set(step_data['subset']['selected_indices'])
        lw_layers = step_data['layerwise']['layers']

        # Majority vote
        layer_counts = Counter()
        for layer in lw_layers:
            for idx in layer['selected_indices']:
                layer_counts[idx] += 1
        majority_sel = {idx for idx, count in layer_counts.items()
                       if count > num_layers / 2}

        j = jaccard(majority_sel, subset_sel)
        divergent_steps.append((j, i, subset_sel, majority_sel))

    divergent_steps.sort()  # lowest Jaccard first

    print("Top 5 most divergent batches (LW majority vs Subset):")
    for j, step_idx, subset_sel, majority_sel in divergent_steps[:5]:
        step_data = steps[step_idx]
        print(f"\n  Step {step_data['step']} (Jaccard={j:.3f}):")
        print(f"    Subset selected:  {sorted(subset_sel)}")
        print(f"    LW majority vote: {sorted(majority_sel)}")

        # Show subset scores
        ss_scores = step_data['subset']['selection'].get('scores', [])
        if ss_scores:
            print(f"    Subset scores: {['%.2f' % s for s in ss_scores]}")

        # Show val sample snippet
        val_texts = step_data['val_samples']
        for vi, vt in enumerate(val_texts):
            snippet = vt[:200].replace('\n', ' ')
            print(f"    Val[{vi}]: {snippet}...")

        # Show train sample snippets with selection status
        train_texts = step_data['train_samples']
        for ti, tt in enumerate(train_texts):
            in_subset = ti in subset_sel
            in_majority = ti in majority_sel
            marker = ""
            if in_subset and in_majority:
                marker = "[BOTH]  "
            elif in_subset:
                marker = "[SS]    "
            elif in_majority:
                marker = "[LW]    "
            else:
                marker = "[NONE]  "
            snippet = tt[:150].replace('\n', ' ')
            print(f"    Train[{ti}] {marker}: {snippet}...")
    print()

    # ================================================================
    # SUMMARY
    # ================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"1. Layerwise and Subset agree on average {np.mean(jaccard_per_step)*100:.1f}% "
          f"(Jaccard) per layer-step")
    print(f"2. Layerwise majority vote agrees with Subset {np.mean(majority_vote_jaccard)*100:.1f}% "
          f"on average")
    print(f"3. Layerwise produces {np.mean(unique_patterns_per_step):.0f} unique selection patterns "
          f"per step across {num_layers} layers")
    print(f"4. Adjacent layers agree {np.mean(neighbor_agreements)*100:.1f}% (Jaccard) on selections")
    print()

    return data


if __name__ == "__main__":
    _scratch = os.environ.get("SCRATCH_DIR", "/scratch")
    records_path = sys.argv[1] if len(sys.argv) > 1 else (
        f"{_scratch}/Gradient-Streaming/SFT/"
        "case_study_tulu3_tydiqa-Llama-3.2-1B-p0.005-lr3.61e-05-b8-v8-s42/"
        "selection_records.json"
    )
    output_dir = Path(records_path).parent
    analyze(records_path, output_dir)
