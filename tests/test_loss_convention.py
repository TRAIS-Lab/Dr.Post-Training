"""
Loss-convention regression tests for drpt (what counts as one "item").

drpt reads per-sample gradients off the backward pass of a single batch loss, so
whatever that loss averages over defines the per-sample gradient inside every score
and the weights of the curated update. The paper's convention ("sample_mean"): the
per-sample gradient is the gradient of the per-example (token-mean) loss lbar_b, the
target gradient is the mean over the m target examples, and every update is a plain
mean over samples (1/n over the batch, 1/k over the selected set).

Checked here on a tiny CPU Qwen3 against per-example gradients from plain autograd:

1. Scores. For GlobalSubset (merged- and separate-batch) and LayerWiseSubset (per
   layer) the recorded score of sample b equals <grad lbar_b, (1/m) sum_j grad lbar_j>
   up to the common factor 1/n, and the selection is the top-k of that quantity.
2. Update. The assembled gradient equals grad[(1/k) sum_{b in S} lbar_b], the mean of
   per-example losses on the selected set (per layer for LayerWiseSubset); non-hooked
   RMSNorm weights receive the train-only mean gradient.
3. Invariant. With selection_frac=1 every strategy (GlobalSubset one/two-pass,
   LayerWiseSubset, GroupWiseSubset; merged and separate) reproduces the
   full-training gradient.
4. Second order. In merged-batch mode the similarity is rescaled by the train-side
   factor squared (not the score correction squared), so greedy selection equals the
   separate-batch result and the reference built from per-example gradients.
5. Legacy. GradientHook(loss_reduction="token_mean") with the HF token-mean loss still
   satisfies the frac=1 invariant, and its merged-batch scores are
   <G_b, H> / (T_train * T_val) with G_b the token-SUM gradient of sample b — i.e. the
   paper's score times the sample's token count (the deviation this fixes).

    python tests/test_loss_convention.py
    pytest tests/test_loss_convention.py -q
"""
from __future__ import annotations

import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import test_groupwise_selection as tg  # noqa: E402  (fixtures)
from drpt import GradientHook, causal_lm_loss, item_counts_from_labels, reduce_masked_loss  # noqa: E402
from drpt.selection import (  # noqa: E402
    build_layer_groups,
    create_merged_batch_strategy,
    create_separate_batch_strategy,
)
from drpt.utils import greedy_selection, topk_selection  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)

TRAIN, VAL = tg.TRAIN, tg.VAL
N_TRAIN, N_VAL, LR, FRAC = tg.B_TRAIN, tg.B_VAL, tg.LR, tg.FRAC
K = max(1, int(N_TRAIN * FRAC))
MERGED_TOL = dict(atol=1e-4, rtol=1e-3)
SEP_TOL = dict(atol=1e-5, rtol=1e-4)


# ----------------------------------------------------------------------------- references

def rows(batch, idx):
    return {k: v[idx] for k, v in batch.items()}


def grads_for(loss_fn):
    model = tg.make_model()
    model.zero_grad()
    loss_fn(model).backward()
    return tg.grads_of(model)


def per_example_grads(batch):
    """grad lbar_i for every row i: the sample-mean loss on the single row is lbar_i."""
    bsz = batch["input_ids"].shape[0]
    return [grads_for(lambda m, i=i: tg.loss_of(m, rows(batch, slice(i, i + 1)))) for i in range(bsz)]


def mean_grads(grad_list):
    return {k: sum(g[k] for g in grad_list) / len(grad_list) for k in grad_list[0]}


def dot(a, b, keys):
    return sum((a[k].double() * b[k].double()).sum().item() for k in keys)


def hooked_param_names(names, grads):
    return [p for p in grads if any(p == f"{n}.weight" or p == f"{n}.bias" for n in names)]


def supervised_tokens(batch):
    """Tokens the (shifted) LM loss is computed on, per row."""
    return (batch["labels"][:, 1:] != -100).sum(dim=1)


class Reference:
    """Per-example gradients and paper-convention scores for the fixture batches."""

    def __init__(self):
        self.model = tg.make_model()
        self.names = tg.hooked_layer_names(self.model)
        self.train = per_example_grads(TRAIN)          # [grad lbar_b]
        self.val = per_example_grads(VAL)              # [grad lbar_j]
        self.g_val = mean_grads(self.val)              # (1/m) sum_j grad lbar_j
        self.hooked = hooked_param_names(self.names, self.train[0])
        assert len(self.hooked) == len(self.names), "every hooked layer should own exactly one weight here"
        # score_b = <grad lbar_b, g_val> over the hooked parameters
        self.scores = torch.tensor([dot(g, self.g_val, self.hooked) for g in self.train])
        # sim_bb' = <grad lbar_b, grad lbar_b'>
        self.sim = torch.tensor([[dot(a, b, self.hooked) for b in self.train] for a in self.train])

    def layer_scores(self, layer_idx):
        key = f"{self.names[layer_idx]}.weight"
        return torch.tensor([dot(g, self.g_val, [key]) for g in self.train])

    def selected_mean(self, selected, keys=None):
        """grad[(1/k) sum_{b in S} lbar_b] = mean of the per-example gradients on S."""
        sel = mean_grads([self.train[b] for b in selected])
        return sel if keys is None else {k: sel[k] for k in keys}


REF = None


def ref():
    global REF
    if REF is None:
        REF = Reference()
    return REF


# ----------------------------------------------------------------------------- runners

def _hook(model, method, granularity, loss_reduction):
    names = tg.hooked_layer_names(model)
    hook = GradientHook(model, names, device="cpu", loss_reduction=loss_reduction)
    if method == "GroupWiseSubset":
        hook.set_layer_groups(build_layer_groups(names, granularity))
    return hook, names


def run_merged(method, frac=FRAC, subset_mode="one_pass", use_second_order=False,
               loss_reduction="sample_mean", loss_of=tg.loss_of, granularity="block"):
    model = tg.make_model()
    hook, names = _hook(model, method, granularity, loss_reduction)
    strategy = create_merged_batch_strategy(
        method=method, grad_hook=hook, frac=frac, use_second_order=use_second_order,
        selection_mode="topk", record_selections=True, scoring_method="reduced_ghost",
        subset_mode=subset_mode,
    )
    strategy.execute_training_step(
        model=model, merged_batch=tg.merge(TRAIN, VAL), train_batch_size=N_TRAIN,
        compute_loss_fn=lambda m, b: loss_of(m, b), lr=LR, batch_train=TRAIN,
    )
    out = dict(grads=tg.grads_of(model), records=strategy.last_selection_record, names=names)
    hook.remove_hooks()
    return out


def run_separate(method, frac=FRAC, subset_mode="one_pass", use_second_order=False,
                 loss_reduction="sample_mean", loss_of=tg.loss_of, granularity="block"):
    model = tg.make_model()
    hook, names = _hook(model, method, granularity, loss_reduction)
    strategy = create_separate_batch_strategy(
        method=method, grad_hook=hook, frac=frac, use_second_order=use_second_order,
        selection_mode="topk", record_selections=True, scoring_method="reduced_ghost",
        subset_mode=subset_mode,
    )
    hook.start_val_capture(scoring_method="reduced_ghost")
    model.zero_grad()
    loss_of(model, VAL).backward()
    hook.end_val_capture()
    strategy.execute_training_step(
        model=model, batch_size=N_TRAIN, compute_loss_fn=lambda: (loss_of(model, TRAIN), {}),
        lr=LR, labels=TRAIN["labels"],
        filter_batch_fn=lambda idx: (lambda: (loss_of(model, rows(TRAIN, idx)), {})),
    )
    hook.clear_val_buffer()
    out = dict(grads=tg.grads_of(model), records=strategy.last_selection_record, names=names)
    hook.remove_hooks()
    return out


def global_record(out):
    rec = out["records"]
    assert len(rec) == 1, rec
    return torch.tensor(rec[0]["scores"]), sorted(rec[0]["selected_indices"])


def assert_close(a, b, what, **tol):
    assert torch.allclose(a, b, **tol), f"{what}: max|diff|={(a - b).abs().max().item():.3e}\n  got {a}\n  ref {b}"


# ----------------------------------------------------------------------------- 1. scores

def test_global_scores_merged_are_per_example_scores():
    """Merged-batch GlobalSubset: recorded score_b == <grad lbar_b, (1/m) sum_j grad lbar_j> / n."""
    r = ref()
    scores, selected = global_record(run_merged("GlobalSubset"))
    assert_close(scores, r.scores / N_TRAIN, "merged GlobalSubset scores", atol=1e-6, rtol=1e-3)
    assert selected == sorted(topk_selection(r.scores * LR, K).tolist()), (selected, r.scores)
    print(f"  merged GlobalSubset: scores == paper score / n (n={N_TRAIN}, m={N_VAL}); top-k identical")


def test_global_scores_separate_are_per_example_scores():
    """Separate-batch GlobalSubset (cached target): same scores as merged, same reference."""
    r = ref()
    scores, selected = global_record(run_separate("GlobalSubset"))
    assert_close(scores, r.scores / N_TRAIN, "separate GlobalSubset scores", atol=1e-6, rtol=1e-3)
    assert selected == sorted(topk_selection(r.scores * LR, K).tolist())
    print("  separate GlobalSubset: scores == paper score / n; top-k identical")


def test_layerwise_scores_are_per_layer_per_example_scores():
    """LayerWiseSubset (merged): per-layer recorded scores == <grad_l lbar_b, grad_l L_val> / n."""
    r = ref()
    out = run_merged("LayerWiseSubset")
    assert len(out["records"]) == len(r.names)
    worst = 0.0
    for rec in out["records"]:
        expected = r.layer_scores(rec["layer_idx"])
        got = torch.tensor(rec["scores"])
        scale = expected.abs().max().item()
        assert_close(got, expected / N_TRAIN, f"layer {rec['layer_idx']} scores",
                     atol=1e-3 * scale / N_TRAIN + 1e-9, rtol=1e-3)
        worst = max(worst, ((got - expected / N_TRAIN).abs().max() / (scale / N_TRAIN + 1e-30)).item())
        assert sorted(rec["selected_indices"]) == sorted(topk_selection(expected * LR, K).tolist()), rec["layer_idx"]
    print(f"  LayerWiseSubset: {len(out['records'])} layers, per-layer scores == paper score / n "
          f"(worst rel err {worst:.1e}); per-layer top-k identical")


# ----------------------------------------------------------------------------- 2. update

def test_global_update_is_mean_over_selected():
    """Curated hooked-layer gradient == grad[(1/k) sum_{b in S} lbar_b]; RMSNorm == train-only mean."""
    r = ref()
    plain = grads_for(lambda m: tg.loss_of(m, TRAIN))
    for mode, run, tol in [("merged", run_merged, MERGED_TOL), ("separate", run_separate, SEP_TOL)]:
        for subset_mode in ["one_pass", "two_pass"]:
            out = run("GlobalSubset", subset_mode=subset_mode)
            _, selected = global_record(out)
            assert len(selected) == K
            expected = grads_for(lambda m: tg.loss_of(m, rows(TRAIN, selected)))   # (1/k) sum_S lbar_b
            worst = 0.0
            for name, g in out["grads"].items():
                if name in r.hooked or subset_mode == "two_pass":
                    target = expected[name]          # two-pass: fresh backward on S for every param
                else:
                    target = plain[name]             # one-pass: non-hooked params keep the train-only grad
                assert torch.allclose(g, target, **tol), (
                    f"{mode}/{subset_mode}: {name} max|diff|={(g - target).abs().max().item():.3e}")
                worst = max(worst, (g - target).abs().max().item())
            print(f"  {mode:8s} GlobalSubset {subset_mode}: update == mean of per-example losses on S "
                  f"(k={K}), max|diff|={worst:.2e}")


def test_layerwise_update_is_per_layer_mean_over_selected():
    """Each hooked layer's gradient == grad_l[(1/k) sum_{b in S_l} lbar_b] for its own selection."""
    r = ref()
    for mode, run, tol in [("merged", run_merged, MERGED_TOL), ("separate", run_separate, SEP_TOL)]:
        out = run("LayerWiseSubset")
        worst = 0.0
        for rec in out["records"]:
            key = f"{r.names[rec['layer_idx']]}.weight"
            expected = r.selected_mean(rec["selected_indices"], [key])[key]
            got = out["grads"][key]
            assert torch.allclose(got, expected, **tol), f"{mode}: {key} max|diff|={(got - expected).abs().max().item():.3e}"
            worst = max(worst, (got - expected).abs().max().item())
        print(f"  {mode:8s} LayerWiseSubset: every layer's grad == mean over its own S_l, max|diff|={worst:.2e}")


# ----------------------------------------------------------------------------- 3. frac = 1 invariant

def test_frac1_reproduces_full_training_every_strategy():
    plain = grads_for(lambda m: tg.loss_of(m, TRAIN))
    cases = [
        ("GlobalSubset", "one_pass"), ("GlobalSubset", "two_pass"),
        ("LayerWiseSubset", "one_pass"), ("GroupWiseSubset", "one_pass"),
    ]
    for mode, run, tol in [("merged", run_merged, MERGED_TOL), ("separate", run_separate, SEP_TOL)]:
        for method, subset_mode in cases:
            out = run(method, frac=1.0, subset_mode=subset_mode)
            worst = tg.assert_same_grads(out["grads"], plain, what=f"{mode}/{method}/{subset_mode}/frac=1", **tol)
            print(f"  {mode:8s} {method:16s} {subset_mode:8s} frac=1.0 == full training, max|diff|={worst:.2e}")


# ----------------------------------------------------------------------------- 4. second order

def test_second_order_merged_matches_separate_and_reference():
    """Merged similarity rescale = (N/n)^2 (train side squared), so greedy selection is exact."""
    r = ref()
    expected = sorted(greedy_selection(r.scores / N_TRAIN * LR, r.sim / N_TRAIN ** 2 * LR ** 2, K).tolist())
    sep = run_separate("GlobalSubset", use_second_order=True)
    mer = run_merged("GlobalSubset", use_second_order=True)
    _, sel_sep = global_record(sep)
    _, sel_mer = global_record(mer)
    assert sel_sep == expected, (sel_sep, expected)
    assert sel_mer == expected, (sel_mer, expected)
    # the same selection must give the same curated gradient on the hooked layers
    for k in r.hooked:
        assert torch.allclose(mer["grads"][k], sep["grads"][k], **MERGED_TOL), k
    plain_topk = sorted(topk_selection(r.scores * LR, K).tolist())
    print(f"  greedy (2nd order) selection merged == separate == reference {expected} "
          f"(plain top-k: {plain_topk})")


# ----------------------------------------------------------------------------- 5. legacy token_mean

def test_legacy_token_mean_convention_pinned():
    """
    GradientHook(loss_reduction="token_mean") + HF token-mean loss: frac=1 still reproduces
    full (token-mean) training, and merged scores are <G_b, H>/(T_train*T_val) with G_b the
    token-SUM gradient — the paper's score times T_b, i.e. the length bias this change removes.
    """
    r = ref()
    plain_hf = grads_for(lambda m: tg.hf_loss_of(m, TRAIN))
    for mode, run, tol in [("merged", run_merged, MERGED_TOL), ("separate", run_separate, SEP_TOL)]:
        out = run("GlobalSubset", frac=1.0, loss_reduction="token_mean", loss_of=tg.hf_loss_of)
        worst = tg.assert_same_grads(out["grads"], plain_hf, what=f"legacy/{mode}/frac=1", **tol)
        print(f"  legacy token_mean {mode:8s} frac=1.0 == HF full training, max|diff|={worst:.2e}")

    t_train = supervised_tokens(TRAIN).double()
    t_val = supervised_tokens(VAL).double()
    G = [{k: g[k] * t for k in r.hooked} for g, t in zip(r.train, t_train)]     # token-sum grads
    H = {k: sum(g[k] * t for g, t in zip(r.val, t_val)) for k in r.hooked}     # token-sum over targets
    legacy_ref = torch.tensor([dot(g, H, r.hooked) for g in G]) / (t_train.sum() * t_val.sum())
    scores, selected = global_record(run_merged("GlobalSubset", loss_reduction="token_mean", loss_of=tg.hf_loss_of))
    assert_close(scores, legacy_ref.float(), "legacy merged scores", atol=1e-6, rtol=1e-3)
    # The same identity written with per-example gradients: legacy_b = (T_b / T_train) *
    # <grad lbar_b, H / T_val>. Relative to the per-example score against the same
    # (token-weighted) target, sample b carries the extra factor T_b. With m > 1 the
    # legacy target H / T_val is itself token-weighted, unlike the paper's (1/m) sum_j grad lbar_j.
    h_w = {k: H[k] / t_val.sum() for k in r.hooked}
    per_example_vs_hw = torch.tensor([dot(g, h_w, r.hooked) for g in r.train])
    assert_close(scores.double(), per_example_vs_hw * t_train / t_train.sum(),
                 "legacy score == per-example score x T_b / T_train", atol=1e-9, rtol=1e-3)
    paper_sel = sorted(topk_selection(r.scores * LR, K).tolist())
    print(f"  legacy scores == <G_b, H>/(T_train T_val) == <grad lbar_b, H/T_val> x T_b/T_train; "
          f"tokens/sample {t_train.long().tolist()}; legacy top-k {selected} vs paper top-k {paper_sel}"
          f"{' (DIFFERENT)' if selected != paper_sel else ' (same on this batch)'}")


# ----------------------------------------------------------------------------- helpers

def test_loss_helpers():
    model = tg.make_model()
    logits = model(input_ids=TRAIN["input_ids"], attention_mask=TRAIN["attention_mask"]).logits
    hf = tg.hf_loss_of(model, TRAIN)
    assert torch.allclose(causal_lm_loss(logits, TRAIN["labels"], "token_mean"), hf, atol=1e-6), "token_mean != HF loss"
    per_row = torch.stack([causal_lm_loss(logits[i:i + 1], TRAIN["labels"][i:i + 1], "token_mean") for i in range(N_TRAIN)])
    assert torch.allclose(causal_lm_loss(logits, TRAIN["labels"], "sample_mean"), per_row.mean(), atol=1e-6)
    assert not torch.allclose(per_row.mean(), hf), "fixture should make the two conventions differ"

    assert item_counts_from_labels(TRAIN["labels"], "sample_mean").tolist() == [1] * N_TRAIN
    assert item_counts_from_labels(TRAIN["labels"], "token_mean").tolist() == (TRAIN["labels"] != -100).sum(1).tolist()

    # empty rows: excluded from the sample mean, count as 0 items, never NaN
    loss = torch.tensor([[1.0, 3.0], [5.0, 7.0], [9.0, 9.0]])
    mask = torch.tensor([[1, 1], [1, 0], [0, 0]])
    assert reduce_masked_loss(loss, mask, "sample_mean").item() == (2.0 + 5.0) / 2
    assert reduce_masked_loss(loss, mask, "token_mean").item() == (1.0 + 3.0 + 5.0) / 3
    assert reduce_masked_loss(loss, torch.zeros_like(mask), "sample_mean").item() == 0.0
    labels = torch.full((2, 3), -100); labels[0, 1] = 5
    assert item_counts_from_labels(labels, "sample_mean").tolist() == [1, 0]
    print(f"  helpers OK: token_mean == HF loss; sample_mean == mean of per-row losses "
          f"(HF {hf.item():.4f} vs sample-mean {per_row.mean().item():.4f})")


ALL_TESTS = [
    test_loss_helpers,
    test_global_scores_merged_are_per_example_scores,
    test_global_scores_separate_are_per_example_scores,
    test_layerwise_scores_are_per_layer_per_example_scores,
    test_global_update_is_mean_over_selected,
    test_layerwise_update_is_per_layer_mean_over_selected,
    test_frac1_reproduces_full_training_every_strategy,
    test_second_order_merged_matches_separate_and_reference,
    test_legacy_token_mean_convention_pinned,
]

if __name__ == "__main__":
    failed = 0
    for t in ALL_TESTS:
        print(f"[{t.__name__}]")
        try:
            t()
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
    print(f"\n{len(ALL_TESTS) - failed}/{len(ALL_TESTS)} tests passed")
    sys.exit(1 if failed else 0)
