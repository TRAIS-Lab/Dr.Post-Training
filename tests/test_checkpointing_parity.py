"""
Parity test: curated training steps with and without activation (gradient) checkpointing.

The Qwen3-4B/8B configs use `gradient_checkpointing: true` (non-reentrant); the 1.7B configs do not.
Under non-reentrant checkpointing the custom autograd Functions in drpt run their *forward* twice
(original pass + recomputation inside backward) and their *backward* once. This test checks, on a tiny
CPU Qwen3, that checkpointing leaves the selection (which samples are kept) and the resulting parameter
gradients unchanged for GlobalSubset one-pass and LayerWiseSubset, in merged-batch and separate-batch mode,
with the RMSNorm train-only wrapping the trainer applies in merged mode.

    python tests/test_checkpointing_parity.py
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import test_groupwise_selection as tg  # noqa: E402  (fixtures + runners)
from drpt import GradientHook  # noqa: E402
from drpt.selection import create_merged_batch_strategy  # noqa: E402

_orig_make_model = tg.make_model
CKPT = {"on": False}


def make_model_maybe_ckpt():
    model = _orig_make_model()
    if CKPT["on"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        assert model.is_gradient_checkpointing
    model.train()
    return model


tg.make_model = make_model_maybe_ckpt


def run_merged_wrapped(method, frac=tg.FRAC):
    """Merged-batch step incl. hook.wrap_nonlinear_layers() (what the trainer does in merged one-pass mode)."""
    model = tg.make_model()
    names = tg.hooked_layer_names(model)
    hook = GradientHook(model, names, device="cpu")
    strategy = create_merged_batch_strategy(
        method=method, grad_hook=hook, frac=frac, selection_mode="topk",
        record_selections=True, scoring_method="pip", subset_mode="one_pass",
    )
    hook.wrap_nonlinear_layers()
    merged = tg.merge(tg.TRAIN, tg.VAL)
    strategy.execute_training_step(
        model=model, merged_batch=merged, train_batch_size=tg.B_TRAIN,
        compute_loss_fn=lambda m, b: tg.loss_of(m, b), lr=tg.LR, batch_train=tg.TRAIN,
    )
    out = dict(grads=tg.grads_of(model), records=strategy.last_selection_record, names=names)
    hook.remove_hooks()
    return out


def selected(records):
    """Normalise a selection record to a comparable structure."""
    if records is None:
        return None
    if isinstance(records, dict):
        return {k: selected(v) for k, v in sorted(records.items())}
    if isinstance(records, (list, tuple)):
        return [selected(v) for v in records]
    if torch.is_tensor(records):
        return records.tolist()
    return records


def both(fn, *a, **kw):
    CKPT["on"] = False
    off = fn(*a, **kw)
    CKPT["on"] = True
    on = fn(*a, **kw)
    CKPT["on"] = False
    return off, on


def check(name, off, on):
    assert selected(off["records"]) == selected(on["records"]), f"{name}: selection differs with checkpointing"
    worst = tg.assert_same_grads(off["grads"], on["grads"], atol=1e-5, rtol=1e-4, what=name)
    print(f"  {name:48s} selection identical, grads max|diff|={worst:.2e}")


def test_plain_ckpt_matches_plain():
    CKPT["on"] = False; ref = tg.plain_train_grads()
    CKPT["on"] = True; ck = tg.plain_train_grads(); CKPT["on"] = False
    worst = tg.assert_same_grads(ref, ck, what="plain autograd ckpt on/off")
    print(f"  {'plain autograd, no hooks':48s} grads max|diff|={worst:.2e}")


def test_merged_global_one_pass():
    check("merged-batch GlobalSubset one_pass (+RMSNorm wrap)", *both(run_merged_wrapped, "GlobalSubset"))


def test_merged_layerwise():
    check("merged-batch LayerWiseSubset (+RMSNorm wrap)", *both(run_merged_wrapped, "LayerWiseSubset"))


def test_merged_frac1_matches_plain_with_ckpt():
    CKPT["on"] = False; ref = tg.plain_train_grads()
    CKPT["on"] = True
    out = run_merged_wrapped("GlobalSubset", frac=1.0)
    CKPT["on"] = False
    worst = tg.assert_same_grads(out["grads"], ref, what="merged frac=1 ckpt vs plain")
    print(f"  {'merged frac=1.0 with ckpt == plain autograd':48s} grads max|diff|={worst:.2e}")


def test_separate_global_one_pass():
    check("separate-batch GlobalSubset one_pass", *both(tg.run_separate, "GlobalSubset", granularity="global"))


def test_separate_layerwise():
    check("separate-batch LayerWiseSubset", *both(tg.run_separate, "LayerWiseSubset"))


def test_merged_groupwise_block():
    check("merged-batch GroupWiseSubset block", *both(tg.run_merged, "GroupWiseSubset", granularity="block"))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAILED: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
