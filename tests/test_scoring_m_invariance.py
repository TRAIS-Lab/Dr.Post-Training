#!/usr/bin/env python
"""
Verify that all scoring mechanisms produce identical scores regardless of
how the validation batch is split (m-invariance).

For a fixed val set of M samples, scores computed with:
  - m=M (all at once)
  - m=1 (one at a time, accumulated)
  - m=M//2 (two halves, accumulated)
should be identical. This tests linearity: score_i = Σ_v <g_i, g_v>.
"""

import sys
import torch

sys.path.insert(0, '.')

from drpt.selection.utils import (
    compute_scores_and_similarity,
    compute_scores_ghost_greats,
    compute_scores_direct_materialization,
)


def test_m_invariance():
    torch.manual_seed(42)
    B_train, S, O, I = 8, 16, 32, 64

    train_go = torch.randn(B_train, S, O)
    train_inp = torch.randn(B_train, S, I)

    # Fixed val set of M=4 samples
    M = 4
    val_go_all = torch.randn(M, S, O)
    val_inp_all = torch.randn(M, S, I)

    scoring_fns = {
        "ghost": compute_scores_and_similarity,
        "ghost_greats": compute_scores_ghost_greats,
        "direct": compute_scores_direct_materialization,
    }

    # Split configs: (name, list of (start, end) slices for val)
    splits = [
        ("m=4 (all at once)", [(0, 4)]),
        ("m=1 (one at a time)", [(0, 1), (1, 2), (2, 3), (3, 4)]),
        ("m=2 (two halves)", [(0, 2), (2, 4)]),
        ("m=1,3 (uneven)", [(0, 1), (1, 4)]),
    ]

    all_pass = True

    for fn_name, fn in scoring_fns.items():
        print(f"\n{'='*60}")
        print(f"Scoring method: {fn_name}")
        print(f"{'='*60}")

        # Reference: all val at once
        ref_scores, ref_sim = fn(
            train_go, train_inp, val_go_all, val_inp_all, None, True
        )

        for split_name, slices in splits:
            # Accumulate SCORES across val sub-batches (linear in val: score_i = Σ_v <g_i, g_v>)
            # Similarity is train-train only (<g_i, g_j>), does NOT depend on val,
            # so we check it from any single call, not accumulated.
            acc_scores = torch.zeros(B_train)
            first_sim = None

            for start, end in slices:
                vgo = val_go_all[start:end]
                vinp = val_inp_all[start:end]
                s, sim = fn(train_go, train_inp, vgo, vinp, None, True)
                acc_scores += s
                if first_sim is None and sim is not None:
                    first_sim = sim

            score_match = torch.allclose(ref_scores, acc_scores, atol=1e-3, rtol=1e-3)
            score_diff = (ref_scores - acc_scores).abs().max().item()

            # Similarity should be identical regardless of val split (train-only quantity)
            sim_match = True
            sim_diff = 0.0
            if first_sim is not None:
                sim_match = torch.allclose(ref_sim, first_sim, atol=1e-4, rtol=1e-3)
                sim_diff = (ref_sim - first_sim).abs().max().item()

            status = "OK" if (score_match and sim_match) else "FAIL"
            if not (score_match and sim_match):
                all_pass = False
            print(f"  [{status}] {split_name}: score_diff={score_diff:.2e}, sim_diff={sim_diff:.2e}")

    # Also test with val_grad_total path (precomputed total)
    print(f"\n{'='*60}")
    print("Testing val_grad_total path (precomputed)")
    print(f"{'='*60}")

    val_grad_total = torch.einsum('vso,vsi->oi', val_go_all, val_inp_all)

    for fn_name, fn in scoring_fns.items():
        # Scores from factorized (all at once)
        ref_scores, _ = fn(train_go, train_inp, val_go_all, val_inp_all, None, False)
        # Scores from precomputed total
        total_scores, _ = fn(train_go, train_inp, None, None, val_grad_total, False)

        match = torch.allclose(ref_scores, total_scores, atol=1e-4, rtol=1e-3)
        diff = (ref_scores - total_scores).abs().max().item()
        status = "OK" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  [{status}] {fn_name}: factorized==total, diff={diff:.2e}")

    # 2D case
    print(f"\n{'='*60}")
    print("2D m-invariance")
    print(f"{'='*60}")

    train_go_2d = torch.randn(B_train, O)
    train_inp_2d = torch.randn(B_train, I)
    val_go_2d = torch.randn(M, O)
    val_inp_2d = torch.randn(M, I)

    for fn_name, fn in scoring_fns.items():
        ref, _ = fn(train_go_2d, train_inp_2d, val_go_2d, val_inp_2d, None, False)
        acc = torch.zeros(B_train)
        for j in range(M):
            s, _ = fn(train_go_2d, train_inp_2d,
                       val_go_2d[j:j+1], val_inp_2d[j:j+1], None, False)
            acc += s
        match = torch.allclose(ref, acc, atol=1e-4, rtol=1e-3)
        diff = (ref - acc).abs().max().item()
        status = "OK" if match else "FAIL"
        if not match:
            all_pass = False
        print(f"  [{status}] {fn_name}: m=4 vs m=1×4, diff={diff:.2e}")

    print(f"\n{'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == "__main__":
    passed = test_m_invariance()
    sys.exit(0 if passed else 1)
