"""
Loss-convention tests for the RLVR hook (RLVR/drpt_verl: GradientHookVerl + SelectionStateVerl + packed_ops).

The RLVR path reimplements the hook for verl (packed sequences via cu_seqlens), so it is not
covered by tests/test_loss_convention.py. verl itself is not needed here: hook.py,
selection_state.py and packed_ops.py are plain torch. The actor (dp_actor_selection.py) is
verl-dependent and is NOT imported; its validation-target normalisation is replicated verbatim.

Checked on a tiny MLP over token features, in both layouts the actor uses (standard [B, S, I]
and packed [1, total_nnz, I] + cu_seqlens), under the paper convention
(``loss_agg_mode="seq-mean-token-mean"``: every response is one item) against per-example
gradients from plain autograd:

1. item counts are ones (responses) under seq-mean-token-mean and token counts under token-mean;
2. GlobalSubset scores equal <grad lbar_b, (1/m) sum_j grad lbar_j> / n per hooked layer sum,
   and negative filtering keeps exactly the positive-score responses;
3. LayerWiseSubset with negative filtering (the RLVR mode) writes, for every hooked layer,
   the mean of the per-example gradients of the responses with positive layer score;
4. selection_frac=1 top-k reproduces the full seq-mean-token-mean gradient; the legacy
   token-mean mode reproduces the full token-mean gradient;
5. packed and standard layouts agree.

    python tests/test_rlvr_hook_convention.py
"""
from __future__ import annotations

import os
import sys
import warnings

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "RLVR"))
warnings.filterwarnings("ignore")

from drpt_verl.hook import GradientHookVerl  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)

I, H, O = 6, 10, 4
B_TRAIN, B_VAL, S = 6, 3, 9
PROMPT = [2, 3, 2, 4, 3, 2, 3, 2, 4]          # prompt tokens per sample (unsupervised)
RESP = [7, 2, 5, 3, 6, 1, 4, 6, 2]            # response tokens per sample (supervised), uneven
LR = 1.0


def make_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(I, H), nn.Tanh(), nn.Linear(H, O))


HOOKED = ["0", "2"]


def make_batch(bsz, offset, seed):
    """Standard layout: x [B, S, I], targets [B, S, O], attention_mask [B, S] (prompt+response),
    labels [B, S] (-100 except response positions)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(bsz, S, I, generator=g)
    y = torch.randn(bsz, S, O, generator=g)
    attn = torch.zeros(bsz, S, dtype=torch.long)
    labels = torch.full((bsz, S), -100, dtype=torch.long)
    for b in range(bsz):
        p, r = PROMPT[offset + b], RESP[offset + b]
        attn[b, : p + r] = 1
        labels[b, p : p + r] = 1                # any value != -100 marks a supervised token
        x[b, p + r :] = 0; y[b, p + r :] = 0    # padding
    return dict(x=x, y=y, attn=attn, labels=labels)


TRAIN = make_batch(B_TRAIN, 0, 1)
VAL = make_batch(B_VAL, B_TRAIN, 2)


def pack(batch):
    """Packed layout: real tokens of every sample concatenated -> [1, total_nnz, *]; mask over packed tokens."""
    xs, ys, ms = [], [], []
    for b in range(batch["x"].shape[0]):
        n = int(batch["attn"][b].sum())
        xs.append(batch["x"][b, :n]); ys.append(batch["y"][b, :n]); ms.append(batch["labels"][b, :n] != -100)
    return dict(x=torch.cat(xs)[None], y=torch.cat(ys)[None], mask=torch.cat(ms)[None].float(),
                lens=batch["attn"].sum(dim=1))


def per_token_loss(model, x, y):
    return ((model(x) - y) ** 2).mean(-1)      # [B, S] or [1, total]


def seq_losses_standard(model, batch):
    """per-response token-mean loss, [B]"""
    mask = (batch["labels"] != -100).float()
    lt = per_token_loss(model, batch["x"], batch["y"])
    return (lt * mask).sum(-1) / mask.sum(-1)


def seq_losses_packed(model, packed):
    lt = per_token_loss(model, packed["x"], packed["y"])[0]
    m = packed["mask"][0]
    out, start = [], 0
    for n in packed["lens"].tolist():
        seg = slice(start, start + n)
        out.append((lt[seg] * m[seg]).sum() / m[seg].sum()); start += n
    return torch.stack(out)


def loss_of(model, batch, mode, packed=False):
    """The batch loss the actor backpropagates under verl's loss_agg_mode."""
    if mode == "seq-mean-token-mean":
        return (seq_losses_packed(model, pack(batch)) if packed else seq_losses_standard(model, batch)).mean()
    # token-mean over all supervised tokens
    if packed:
        p = pack(batch); lt = per_token_loss(model, p["x"], p["y"]); return (lt * p["mask"]).sum() / p["mask"].sum()
    mask = (batch["labels"] != -100).float()
    return (per_token_loss(model, batch["x"], batch["y"]) * mask).sum() / mask.sum()


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def grads_for(loss_fn):
    model = make_model(); model.zero_grad(); loss_fn(model).backward(); return grads_of(model)


def row(batch, b):
    return {k: v[b : b + 1] for k, v in batch.items()}


def per_example_grads(batch):
    return [grads_for(lambda m, b=b: seq_losses_standard(m, row(batch, b))[0]) for b in range(batch["x"].shape[0])]


def mean_grads(gs):
    return {k: sum(g[k] for g in gs) / len(gs) for k in gs[0]}


def dot(a, b, keys):
    return sum((a[k].double() * b[k].double()).sum().item() for k in keys)


def layer_keys(idx):
    """parameters the curated update writes for hooked layer idx (weight and bias)"""
    return [f"{HOOKED[idx]}.weight", f"{HOOKED[idx]}.bias"]


def score_keys(idx):
    """the RLVR hook scores weight gradients only (no bias augmentation)"""
    return [f"{HOOKED[idx]}.weight"]


ALL_KEYS = layer_keys(0) + layer_keys(1)
SCORE_KEYS = score_keys(0) + score_keys(1)


class Ref:
    def __init__(self):
        self.train = per_example_grads(TRAIN)
        self.val = per_example_grads(VAL)
        self.g_val = mean_grads(self.val)
        self.scores = torch.tensor([dot(g, self.g_val, SCORE_KEYS) for g in self.train])
        self.layer_scores = [torch.tensor([dot(g, self.g_val, score_keys(l)) for g in self.train]) for l in range(2)]


REF = Ref()


# ----------------------------------------------------------------------------- hook driver

def run(method, mode="seq-mean-token-mean", packed=False, frac=1.0, selection_mode="topk"):
    """Mirror dp_actor_selection: capture target (mean over val responses), then one curated step."""
    model = make_model()
    hook = GradientHookVerl(model, HOOKED, device="cpu", loss_agg_mode=mode)

    # --- validation target, normalised the way capture_validation_gradients_external does:
    # per-sequence loss per loss_agg_mode, summed, divided by the GLOBAL item count
    hook.start_val_capture(); model.zero_grad()
    if mode == "seq-mean-token-mean":
        vloss = seq_losses_standard(model, VAL).sum() / B_VAL
    else:
        vmask = (VAL["labels"] != -100).float()
        vloss = (per_token_loss(model, VAL["x"], VAL["y"]) * vmask).sum() / vmask.sum()
    vloss.backward(); hook.end_val_capture(); model.zero_grad()

    # --- curated training step
    hook.setup_selection_with_stored_val(train_batch_size=B_TRAIN, selection_method=method, frac=frac, lr=LR,
                                         selection_mode=selection_mode)
    hook.enable_hooks()
    hook.set_token_counts(TRAIN["labels"], B_TRAIN, TRAIN["attn"])
    state = hook.selection_state
    items = state.tokens_per_sample.clone()
    model.zero_grad()
    loss_of(model, TRAIN, mode, packed=packed).backward()
    out = dict(grads=grads_of(model), items=items, state=state, hook=hook)
    hook.remove_hooks()
    return out


def close(a, b, what, atol=1e-6, rtol=1e-4):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{what}: max|diff|={(a - b).abs().max().item():.3e}\n got {a}\n ref {b}"


# ----------------------------------------------------------------------------- tests

def test_item_counts_follow_loss_agg_mode():
    resp = torch.tensor(RESP[:B_TRAIN])
    for packed in (False, True):
        out = run("LayerWiseSubset", "seq-mean-token-mean", packed=packed)
        assert out["items"].tolist() == [1] * B_TRAIN, out["items"]
        assert torch.equal(out["state"].cu_seqlens[1:].cpu(), torch.cumsum(TRAIN["attn"].sum(1), 0).int()), "cu_seqlens must follow the attention mask"
        out = run("LayerWiseSubset", "token-mean", packed=packed)
        assert out["items"].tolist() == resp.tolist(), out["items"]
    print(f"  seq-mean-token-mean -> ones; token-mean -> response tokens {RESP[:B_TRAIN]}; cu_seqlens from attention mask (both layouts)")


def test_global_scores_and_filtering():
    for packed in (False, True):
        out = run("GlobalSubset", packed=packed, frac=1.0, selection_mode="filtering")
        scores = out["state"].grad_dot_scores.cpu()
        close(scores, REF.scores / B_TRAIN, f"GlobalSubset scores (packed={packed})", atol=1e-7, rtol=1e-4)
        kept = sorted(out["state"].get_final_selection().tolist())
        expected = [b for b in range(B_TRAIN) if REF.scores[b] > 0]
        assert kept == expected, (kept, expected)
    print(f"  GlobalSubset: scores == <grad lbar_b, mean_j grad lbar_j>/n in both layouts; filtering keeps positives {expected} "
          f"(paper scores {[round(v, 4) for v in REF.scores.tolist()]})")


def test_layerwise_filtering_update_is_mean_over_positive():
    for packed in (False, True):
        out = run("LayerWiseSubset", packed=packed, frac=1.0, selection_mode="filtering")
        for l in range(2):
            kept = [b for b in range(B_TRAIN) if REF.layer_scores[l][b] > 0]
            assert kept, "fixture should keep at least one response per layer"
            expected = mean_grads([REF.train[b] for b in kept])
            for k in layer_keys(l):
                close(out["grads"][k], expected[k], f"layer {l} {k} (packed={packed}, kept={kept})", atol=1e-6, rtol=1e-4)
    print(f"  LayerWiseSubset filtering: every hooked layer's grad == mean of per-example grads over its positive-score responses "
          f"(layer 0 keeps {[b for b in range(B_TRAIN) if REF.layer_scores[0][b] > 0]}, layer 1 keeps {[b for b in range(B_TRAIN) if REF.layer_scores[1][b] > 0]})")


def test_frac1_reproduces_full_training_both_conventions():
    for mode in ("seq-mean-token-mean", "token-mean"):
        plain = grads_for(lambda m: loss_of(m, TRAIN, mode))
        for packed in (False, True):
            for method in ("LayerWiseSubset",):
                out = run(method, mode, packed=packed, frac=1.0, selection_mode="topk")
                for k in ALL_KEYS:
                    close(out["grads"][k], plain[k], f"{mode} {method} frac=1 (packed={packed}) {k}", atol=1e-6, rtol=1e-4)
        print(f"  {mode:20s}: LayerWiseSubset frac=1 == full gradient of the {mode} loss (both layouts)")


def test_topk_half_is_mean_over_selected():
    for packed in (False, True):
        out = run("LayerWiseSubset", packed=packed, frac=0.5, selection_mode="topk")
        k = max(1, int(B_TRAIN * 0.5))
        for l in range(2):
            sel = sorted(torch.topk(REF.layer_scores[l], k).indices.tolist())
            expected = mean_grads([REF.train[b] for b in sel])
            for key in layer_keys(l):
                close(out["grads"][key], expected[key], f"layer {l} top-k (packed={packed})", atol=1e-6, rtol=1e-4)
    print(f"  LayerWiseSubset top-k 0.5: layer grad == (1/k) sum over its top-k responses (scale n/k), both layouts")


def test_legacy_target_is_token_weighted():
    """Under token-mean the cached target is the token-weighted mean over val responses, not the plain mean."""
    out = run("GlobalSubset", "token-mean", frac=1.0, selection_mode="filtering")
    t_val = torch.tensor(RESP[B_TRAIN:B_TRAIN + B_VAL], dtype=torch.float64)
    h_w = {k: sum(g[k] * t for g, t in zip(REF.val, t_val)) / t_val.sum() for k in SCORE_KEYS}
    cached = out["hook"].get_val_grad(1).cpu().double()        # layer index 1 = "2" (weight only)
    close(cached.float(), h_w["2.weight"].float(), "legacy token-weighted target", atol=1e-6, rtol=1e-4)
    close_uniform = torch.allclose(cached.float(), REF.g_val["2.weight"].float(), atol=1e-6, rtol=1e-4)
    assert not close_uniform, "fixture lengths should make token-weighted and uniform targets differ"
    print(f"  token-mean target == token-weighted mean over val responses (lengths {t_val.long().tolist()}), != paper's uniform mean")


ALL_TESTS = [
    test_item_counts_follow_loss_agg_mode,
    test_global_scores_and_filtering,
    test_layerwise_filtering_update_is_mean_over_positive,
    test_frac1_reproduces_full_training_both_conventions,
    test_topk_half_is_mean_over_selected,
    test_legacy_target_is_token_weighted,
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
