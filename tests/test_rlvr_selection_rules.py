"""
Tests for the gather-then-select ("v2") RLVR selection path (2026-09-24):

  * drpt_verl.selection_rules: filtering / top-k rules, prompt-level aggregation + broadcast,
    layer normalization, advantage re-centring, diagnostics
  * GlobalSubsetStateVerl.record_layer_scores: the recorded per-layer score vectors equal the
    exact per-layer inner products of the toy model from tests/test_rlvr_embedding_hook.py
  * LayerWiseSubsetStateVerl.fixed_selections: the hooked backward reproduces the mean gradient
    over the prescribed samples per layer, keeps everything for layers without an entry, and
    yields a zero gradient for an empty entry

Run:  python tests/test_rlvr_selection_rules.py   (or pytest -q tests/test_rlvr_selection_rules.py)
"""
from __future__ import annotations

import os
import sys
import warnings

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RLVR"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
warnings.filterwarnings("ignore")

from drpt_verl.hook import GradientHookVerl  # noqa: E402
from drpt_verl.selection_rules import (  # noqa: E402
    SelectionRule, aggregate_by_group, group_ids_from_uids, keep_mask_rows, normalize_layer_scores,
    recentered_advantage_shift, select_masks, selection_diagnostics,
)
import test_rlvr_embedding_hook as H  # noqa: E402  (toy model, exact per-example reference gradients)

torch.manual_seed(0)


# ----------------------------------------------------------------------------- rules

def test_filtering_frac1_is_sign():
    S = torch.randn(3, 10)
    S[0, 4] = 0.0
    keep = keep_mask_rows(S, "filtering", 1.0)
    assert torch.equal(keep, S >= 0), "frac=1 filtering must keep exactly the non-negative scores (zero kept)"
    print("  filtering frac=1: keep == (score >= 0)")


def test_filtering_partial_drops_most_negative():
    S = torch.tensor([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], [-1.0, -5.0, 3.0, -0.5, 2.0, 0.1]])
    keep = keep_mask_rows(S, "filtering", 0.5)
    # row 0: 3 negatives -> drop int(1.5)=1, the most negative (-3); row 1: 3 negatives -> drop -5
    assert keep[0].tolist() == [False, True, True, True, True, True], keep[0]
    assert keep[1].tolist() == [True, False, True, True, True, True], keep[1]
    keep0 = keep_mask_rows(S, "filtering", 0.0)
    assert keep0.all()
    print("  filtering frac=0.5: drops the most negative floor(0.5*n_neg) per row; frac=0 keeps all")


def test_topk_budget():
    S = torch.randn(4, 10)
    keep = keep_mask_rows(S, "topk", 0.3)
    assert (keep.sum(dim=1) == 3).all(), keep.sum(dim=1)
    for r in range(4):
        thr = S[r][keep[r]].min()
        assert (S[r][~keep[r]] <= thr).all(), "kept set must be the top-k of the row"
    keep1 = keep_mask_rows(S, "topk", 1.0)
    assert keep1.all()
    keep_small = keep_mask_rows(S, "topk", 0.01)
    assert (keep_small.sum(dim=1) == 1).all(), "top-k keeps at least one candidate"
    print("  top-k: exactly max(1, int(frac*n)) best per row")


def test_prompt_level_broadcast_and_sum():
    uids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    gid, G = group_ids_from_uids(uids)
    assert G == 4 and gid.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    S = torch.tensor([[1.0, -3.0, 2.0, 1.0, -1.0, 0.5, 0.0, 0.0],
                      [5.0, -1.0, -2.0, -2.0, 1.0, 1.0, -0.1, 0.05]])
    Gs = aggregate_by_group(S, gid, G)
    assert torch.allclose(Gs, torch.tensor([[-2.0, 3.0, -0.5, 0.0], [4.0, -4.0, 2.0, -0.05]]))
    keep = select_masks(S, gid, G, SelectionRule(mode="filtering", frac=1.0, level="prompt").validate())
    # row 0: groups a(-2) dropped, b(3) kept, c(-0.5) dropped, d(0) kept; row 1: a kept, b dropped, c kept, d dropped
    assert keep[0].tolist() == [False, False, True, True, False, False, True, True], keep[0]
    assert keep[1].tolist() == [True, True, False, False, True, True, False, False], keep[1]
    keep_k = select_masks(S, gid, G, SelectionRule(mode="topk", frac=0.5, level="prompt").validate())
    assert (keep_k.sum(dim=1) == 4).all(), "prompt top-k 0.5 over 4 groups keeps 2 groups = 4 rollouts"
    for r in range(2):
        for g in range(G):
            members = keep_k[r][gid == g]
            assert members.all() or (~members).all(), "all rollouts of a prompt share one decision"
    print("  prompt level: group sums decide, decision broadcast to every rollout of the prompt; top-k counts prompts")


def test_layer_normalization():
    S = torch.randn(5, 12) * torch.tensor([[1.0], [10.0], [100.0], [0.01], [3.0]])
    N = normalize_layer_scores(S, "layer_meanabs")
    assert torch.allclose(N.abs().mean(dim=1), torch.ones(5), atol=1e-6)
    Nstd = normalize_layer_scores(S, "layer_std")
    assert torch.allclose(Nstd.std(dim=1), torch.ones(5), atol=1e-5)
    assert torch.equal(normalize_layer_scores(S, "none"), S)
    # sign of the global sum can flip once the dominant layer is rescaled
    dominant = S.sum(dim=0)
    balanced = N.sum(dim=0)
    assert dominant.shape == balanced.shape
    print("  layer normalization: every row rescaled to mean|s| = 1 (or std 1); none = identity")


def test_recentered_advantage_shift():
    uids = ["p0"] * 4 + ["p1"] * 4 + ["p2"] * 4
    gid, G = group_ids_from_uids(uids)
    adv = torch.tensor([1.0, 1.0, -1.0, -1.0, 2.0, -0.5, -0.5, -1.0, 0.0, 0.0, 0.0, 0.0])
    keep = torch.tensor([True, True, False, True, True, True, False, False, True, True, True, True])
    shift = recentered_advantage_shift(adv, keep, gid, G)
    new = adv - shift
    for g in range(G):
        m = (gid == g) & keep
        assert abs(new[m].sum().item()) < 1e-6, f"kept advantages of group {g} must re-centre to zero"
    assert (shift[~keep] == 0).all(), "dropped rollouts get no shift"
    assert torch.allclose(shift[:4], torch.tensor([1 / 3, 1 / 3, 0.0, 1 / 3]))
    print("  re-centring: kept advantages sum to zero per prompt after the shift; dropped rollouts untouched")


def test_diagnostics_keys_and_values():
    uids = ["a"] * 4 + ["b"] * 4
    gid, G = group_ids_from_uids(uids)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    keep = torch.tensor([[True, False, True, False, True, True, False, False],
                         [True, True, True, True, True, True, True, True]])
    d = selection_diagnostics(keep, adv, gid, G)
    assert abs(d["keep_frac_pos_adv"] - 1.0) < 1e-6
    assert abs(d["keep_frac_neg_adv"] - 0.5) < 1e-6
    assert abs(d["keep_frac_zero_adv"] - 0.75) < 1e-6
    assert abs(d["kept_prompt_frac"] - 1.0) < 1e-6
    assert d["layer_keep_min"] < d["layer_keep_max"] == 1.0
    # row 0 keeps (1, .5, 0, 0) -> mean .375 ; row 1 keeps all -> 0 ; average .1875
    assert abs(d["kept_adv_mean"] - 0.1875) < 1e-6, d["kept_adv_mean"]
    print("  diagnostics: keep fractions by advantage sign, kept-advantage mean, prompt coverage, layer spread")


# ----------------------------------------------------------------------------- hook integration

def _hook_with_target(model):
    hook = GradientHookVerl(model, H.HOOKED, device="cpu", loss_agg_mode=H.MODE)
    hook.start_val_capture(); model.zero_grad()
    (H.seq_losses_standard(model, H.VAL).sum() / H.B_VAL).backward()
    hook.end_val_capture(); model.zero_grad()
    return hook


def test_recorded_layer_scores_match_exact_inner_products():
    for packed in (False, True):
        model = H.make_model(False)
        hook = _hook_with_target(model)
        hook.setup_selection_with_stored_val(train_batch_size=H.B_TRAIN, selection_method="GlobalSubset", frac=1.0,
                                             lr=1.0, compute_scores_only=True, selection_mode="filtering",
                                             record_layer_scores=True)
        hook.enable_hooks(); hook.set_token_counts(H.TRAIN["labels"], H.B_TRAIN, H.TRAIN["attn"])
        model.zero_grad(); H.loss_of(model, H.TRAIN, packed=packed).backward()
        S, scored = hook.selection_state.layer_score_matrix(len(H.HOOKED))
        assert scored.all(), scored
        for l in range(4):
            H.close(S[l], H.REF.layer_scores[l] / H.B_TRAIN, f"recorded scores of {H.HOOKED[l]} (packed={packed})",
                    atol=1e-7, rtol=1e-4)
        H.close(S.sum(dim=0), hook.selection_state.grad_dot_scores, "row sum == accumulated global score")
        hook.remove_hooks()
    print("  record_layer_scores: per-layer vectors == exact <grad l_b, target>/n for emb/fc/proj/out, both layouts")


def _run_fixed(fixed, packed):
    model = H.make_model(False)
    hook = _hook_with_target(model)
    hook.setup_selection_with_stored_val(train_batch_size=H.B_TRAIN, selection_method="LayerWiseSubset", frac=1.0,
                                         lr=1.0, selection_mode="filtering")
    hook.enable_hooks(); hook.set_token_counts(H.TRAIN["labels"], H.B_TRAIN, H.TRAIN["attn"])
    hook.selection_state.fixed_selections = {k: torch.tensor(v, dtype=torch.long) for k, v in fixed.items()}
    model.zero_grad(); H.loss_of(model, H.TRAIN, packed=packed).backward()
    grads = H.grads_of(model)
    sel = list(hook.selection_state._layer_selections)
    hook.remove_hooks()
    return grads, sel


def test_fixed_selections_drive_layerwise_gradient():
    fixed = {H.EMB: [0, 2], H.FC: [1, 3, 5], H.PROJ: [0, 1, 2, 3, 4, 5], H.OUT: [4]}
    for packed in (False, True):
        grads, sel = _run_fixed(fixed, packed)
        for l, kept in fixed.items():
            expected = H.mean_grads([H.REF.train[b] for b in kept])
            for k in H.LAYER_KEYS[l]:
                H.close(grads[k], expected[k], f"fixed layer {H.HOOKED[l]} {k} (packed={packed}, kept={kept})")
        counts = {l: n for l, n in sel}
        assert counts == {l: len(v) for l, v in fixed.items()}, counts
    print("  fixed_selections: each layer's grad == mean over ITS prescribed samples (both layouts), stats report the counts")


def test_fixed_selection_missing_layer_keeps_all_and_empty_gives_zero():
    plain = H.grads_for(lambda m: H.loss_of(m, H.TRAIN))
    for packed in (False, True):
        grads, _ = _run_fixed({H.FC: []}, packed)
        for k in H.LAYER_KEYS[H.FC]:
            assert torch.all(grads[k] == 0), f"empty fixed selection must zero the layer gradient ({k})"
        for l in (H.EMB, H.PROJ, H.OUT):
            for k in H.LAYER_KEYS[l]:
                H.close(grads[k], plain[k], f"layer without fixed entry keeps the full gradient ({k}, packed={packed})")
    print("  fixed_selections: missing layer -> full gradient; empty entry -> zero gradient")


def _run_force_keep(force, packed):
    model = H.make_model(False)
    hook = _hook_with_target(model)
    hook.setup_selection_with_stored_val(train_batch_size=H.B_TRAIN, selection_method="LayerWiseSubset", frac=1.0,
                                         lr=1.0, selection_mode="filtering")
    hook.enable_hooks(); hook.set_token_counts(H.TRAIN["labels"], H.B_TRAIN, H.TRAIN["attn"])
    hook.selection_state.force_keep = torch.tensor(force, dtype=torch.bool)
    model.zero_grad(); H.loss_of(model, H.TRAIN, packed=packed).backward()
    grads = H.grads_of(model); hook.remove_hooks(); return grads


def test_force_keep_unions_with_the_sign_rule():
    """keep_zero_adv path: forced positions are kept in every layer on top of the score rule."""
    force = [False] * H.B_TRAIN
    # force the samples the sign rule would DROP for the embedding and the output layer
    dropped_emb = [b for b in range(H.B_TRAIN) if b not in H.REF.kept(H.EMB)]
    for b in dropped_emb[:1]:
        force[b] = True
    for packed in (False, True):
        grads = _run_force_keep(force, packed)
        for l in (H.EMB, H.OUT):
            kept = sorted(set(H.REF.kept(l)) | {b for b in range(H.B_TRAIN) if force[b]})
            expected = H.mean_grads([H.REF.train[b] for b in kept])
            for k in H.LAYER_KEYS[l]:
                H.close(grads[k], expected[k], f"force_keep layer {H.HOOKED[l]} {k} (packed={packed}, kept={kept})")
    none = _run_force_keep([False] * H.B_TRAIN, False)
    ref = H.run("LayerWiseSubset", frac=1.0, selection_mode="filtering")["grads"]
    for k in ref:
        H.close(none[k], ref[k], f"all-False force_keep == legacy ({k})")
    print(f"  force_keep: forced sample {dropped_emb[:1]} joins the kept set of every layer; all-False == legacy rule")


def test_legacy_sign_rule_unchanged():
    out = H.run("LayerWiseSubset", frac=1.0, selection_mode="filtering")
    for l in (H.EMB, H.OUT):
        expected = H.mean_grads([H.REF.train[b] for b in H.REF.kept(l)])
        for k in H.LAYER_KEYS[l]:
            H.close(out["grads"][k], expected[k], f"legacy filtering {H.HOOKED[l]} {k}")
    print("  legacy in-backward filtering unchanged when no fixed selection is set")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"\n{len(tests)} tests passed")
