"""
Tests for the keep-zero-advantage path of one-pass Layer-Wise selection in RLVR (keep_zero_adv, on by default):
LayerWiseSubsetStateVerl.force_keep joins the forced micro-batch positions to every layer's kept set on top of the
in-backward sign rule, and an all-False mask reproduces the legacy rule exactly. Uses the toy model and exact
per-example reference gradients of tests/test_rlvr_embedding_hook.py.

Run:  python tests/test_rlvr_keep_zero.py   (or pytest -q tests/test_rlvr_keep_zero.py)
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
import test_rlvr_embedding_hook as H  # noqa: E402  (toy model, exact per-example reference gradients)

torch.manual_seed(0)


def _hook_with_target(model):
    hook = GradientHookVerl(model, H.HOOKED, device="cpu", loss_agg_mode=H.MODE)
    hook.start_val_capture(); model.zero_grad()
    (H.seq_losses_standard(model, H.VAL).sum() / H.B_VAL).backward()
    hook.end_val_capture(); model.zero_grad()
    return hook


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
    print("  legacy in-backward filtering unchanged when force_keep is unset")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"[{t.__name__}]")
        t()
    print(f"\n{len(tests)} tests passed")
