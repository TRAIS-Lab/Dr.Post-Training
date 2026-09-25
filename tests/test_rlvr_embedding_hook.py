"""
Embedding-hook tests for the RLVR hook (RLVR/drpt_verl): the token embedding and the (optionally
weight-tied) output layer are hooked like Linear layers, with the lookup-based scoring of
packed_ops.compute_embedding_* (the embedding analogue of PIP; no per-sample [V, D] gradient is built).

Tiny model: ids -> Embedding(V, D, padding_idx=0) -> Linear(D, H) -> Tanh -> Linear(H, D) -> out (Linear(D, V)),
where ``out`` is either an independent copy of the embedding matrix (untied twin, used as the reference)
or the embedding weight itself (tied, as in Qwen3). Loss: per-token MSE of the V-dim output against random
targets, seq-mean-token-mean over response tokens, in both layouts the actor uses (standard [B, S] and
packed [1, total_nnz] + cu_seqlens).

Checks against per-example gradients from plain autograd:
1. val capture: the cached embedding target equals the autograd gradient of the val loss (padding row zero);
2. GlobalSubset scores equal the sum over ALL hooked modules, embedding and output layer included,
   of <grad l_b, mean_j grad l_j> / n, and differ from the Linear-only scores;
3. LayerWiseSubset with negative filtering writes, for the embedding and the output layer, the mean of the
   per-example gradients over the responses with non-negative layer score (the filtering rule keeps scores >= 0);
4. selection_frac=1 top-k reproduces the full gradient for every parameter, untied and tied;
5. tied weight: the shared .grad is the curated embedding gradient plus the curated output-layer gradient,
   each with its own kept set (autograd accumulates the two hooked contributions).

    python tests/test_rlvr_embedding_hook.py
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

V, D, H = 11, 6, 8
PAD = 0
B_TRAIN, B_VAL, S = 6, 3, 9
PROMPT = [2, 3, 2, 4, 3, 2, 3, 2, 4]          # prompt tokens per sample (unsupervised)
RESP = [7, 2, 5, 3, 6, 1, 4, 6, 2]            # response tokens per sample (supervised), uneven
LR = 1.0
MODE = "seq-mean-token-mean"


class Toy(nn.Module):
    def __init__(self, tied: bool):
        super().__init__()
        self.emb = nn.Embedding(V, D, padding_idx=PAD)
        self.fc = nn.Linear(D, H)
        self.proj = nn.Linear(H, D)
        self.out = nn.Linear(D, V, bias=False)
        if tied:
            self.out.weight = self.emb.weight
        else:
            with torch.no_grad():
                self.out.weight.copy_(self.emb.weight)

    def forward(self, ids):
        return self.out(self.proj(torch.tanh(self.fc(self.emb(ids)))))


def make_model(tied: bool = False):
    torch.manual_seed(0)
    return Toy(tied)


HOOKED = ["emb", "fc", "proj", "out"]
EMB, FC, PROJ, OUT = range(4)


def make_batch(bsz, offset, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, V, (bsz, S), generator=g)          # real tokens never use the padding id
    y = torch.randn(bsz, S, V, generator=g)
    attn = torch.zeros(bsz, S, dtype=torch.long)
    labels = torch.full((bsz, S), -100, dtype=torch.long)
    for b in range(bsz):
        p, r = PROMPT[offset + b], RESP[offset + b]
        attn[b, : p + r] = 1
        labels[b, p : p + r] = 1
        ids[b, p + r :] = PAD; y[b, p + r :] = 0             # padding
    return dict(ids=ids, y=y, attn=attn, labels=labels)


TRAIN = make_batch(B_TRAIN, 0, 1)
VAL = make_batch(B_VAL, B_TRAIN, 60)    # seed chosen so that every hooked module keeps a strict subset under filtering and no score is near zero


def pack(batch):
    xs, ys, ms = [], [], []
    for b in range(batch["ids"].shape[0]):
        n = int(batch["attn"][b].sum())
        xs.append(batch["ids"][b, :n]); ys.append(batch["y"][b, :n]); ms.append(batch["labels"][b, :n] != -100)
    return dict(ids=torch.cat(xs)[None], y=torch.cat(ys)[None], mask=torch.cat(ms)[None].float(),
                lens=batch["attn"].sum(dim=1))


def per_token_loss(model, ids, y):
    return ((model(ids) - y) ** 2).mean(-1)


def seq_losses_standard(model, batch):
    mask = (batch["labels"] != -100).float()
    lt = per_token_loss(model, batch["ids"], batch["y"])
    return (lt * mask).sum(-1) / mask.sum(-1)


def seq_losses_packed(model, packed):
    lt = per_token_loss(model, packed["ids"], packed["y"])[0]
    m = packed["mask"][0]
    out, start = [], 0
    for n in packed["lens"].tolist():
        seg = slice(start, start + n)
        out.append((lt[seg] * m[seg]).sum() / m[seg].sum()); start += n
    return torch.stack(out)


def loss_of(model, batch, packed=False):
    return (seq_losses_packed(model, pack(batch)) if packed else seq_losses_standard(model, batch)).mean()


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def grads_for(loss_fn, tied=False):
    model = make_model(tied); model.zero_grad(); loss_fn(model).backward(); return grads_of(model)


def row(batch, b):
    return {k: v[b : b + 1] for k, v in batch.items()}


def per_example_grads(batch):
    return [grads_for(lambda m, b=b: seq_losses_standard(m, row(batch, b))[0]) for b in range(batch["ids"].shape[0])]


def mean_grads(gs):
    return {k: sum(g[k] for g in gs) / len(gs) for k in gs[0]}


def dot(a, b, keys):
    return sum((a[k].double() * b[k].double()).sum().item() for k in keys)


LAYER_KEYS = {EMB: ["emb.weight"], FC: ["fc.weight", "fc.bias"], PROJ: ["proj.weight", "proj.bias"], OUT: ["out.weight"]}
SCORE_KEYS = {EMB: ["emb.weight"], FC: ["fc.weight"], PROJ: ["proj.weight"], OUT: ["out.weight"]}   # weights only
ALL_SCORE_KEYS = sum(SCORE_KEYS.values(), [])


class Ref:
    def __init__(self):
        self.train = per_example_grads(TRAIN)
        self.val = per_example_grads(VAL)
        self.g_val = mean_grads(self.val)
        self.scores = torch.tensor([dot(g, self.g_val, ALL_SCORE_KEYS) for g in self.train])
        self.linear_only_scores = torch.tensor([dot(g, self.g_val, SCORE_KEYS[FC] + SCORE_KEYS[PROJ]) for g in self.train])
        self.layer_scores = {l: torch.tensor([dot(g, self.g_val, SCORE_KEYS[l]) for g in self.train]) for l in range(4)}

        # tied-aware treatment: the shared weight is one group whose score is the inner product of the SUMMED
        # site gradients (embedding use + output use), i.e. the two diagonal terms plus the two cross terms
        gv_tied = (self.g_val["emb.weight"] + self.g_val["out.weight"]).double()
        self.tied_group_scores = torch.tensor([((g["emb.weight"] + g["out.weight"]).double() * gv_tied).sum().item() for g in self.train])
        self.tied_exact_scores = self.layer_scores[FC] + self.layer_scores[PROJ] + self.tied_group_scores

    def kept_tied(self):
        return [b for b in range(B_TRAIN) if self.tied_group_scores[b] >= 0]

    def kept(self, l):
        """negative filtering keeps scores >= 0 (drpt.utils.negative_filtering)"""
        return [b for b in range(B_TRAIN) if self.layer_scores[l][b] >= 0]

    def kept_global(self):
        return [b for b in range(B_TRAIN) if self.scores[b] >= 0]


REF = Ref()
# Embedding scores are exactly zero for responses whose tokens never occur in the validation batch;
# the fixture must not sit on that boundary (or on any other), otherwise float rounding decides the selection.
assert torch.cat([REF.layer_scores[l] for l in range(4)] + [REF.scores]).abs().min() > 1e-3, "fixture has a near-zero score"


# ----------------------------------------------------------------------------- hook driver

def run(method, tied=False, packed=False, frac=1.0, selection_mode="topk", tie_embeddings=False):
    """Mirror dp_actor_selection: capture the target (mean over val responses), then one curated step."""
    model = make_model(tied)
    hook = GradientHookVerl(model, HOOKED, device="cpu", loss_agg_mode=MODE, tie_embeddings=tie_embeddings)

    hook.start_val_capture(); model.zero_grad()
    (seq_losses_standard(model, VAL).sum() / B_VAL).backward()
    hook.end_val_capture(); model.zero_grad()

    hook.setup_selection_with_stored_val(train_batch_size=B_TRAIN, selection_method=method, frac=frac, lr=LR,
                                         selection_mode=selection_mode)
    hook.enable_hooks()
    hook.set_token_counts(TRAIN["labels"], B_TRAIN, TRAIN["attn"])
    state = hook.selection_state
    model.zero_grad()
    loss_of(model, TRAIN, packed=packed).backward()
    out = dict(grads=grads_of(model), state=state, hook=hook)
    hook.remove_hooks()
    return out


def close(a, b, what, atol=1e-6, rtol=1e-4):
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{what}: max|diff|={(a - b).abs().max().item():.3e}\n got {a}\n ref {b}"


# ----------------------------------------------------------------------------- tests

def test_val_capture_embedding_and_output():
    out = run("LayerWiseSubset")
    hook = out["hook"]
    assert hook.get_num_val_layers_captured() == 4, hook.get_num_val_layers_captured()
    cached = hook.get_val_grad(EMB)
    close(cached, REF.g_val["emb.weight"], "embedding val target")
    assert torch.all(cached[PAD] == 0), "padding row of the embedding target must be zero"
    close(hook.get_val_grad(OUT), REF.g_val["out.weight"], "output-layer val target")
    print("  val capture: embedding target == autograd grad of the val loss (padding row 0); output layer too")


def test_global_scores_include_embedding_and_output():
    for packed in (False, True):
        out = run("GlobalSubset", packed=packed, frac=1.0, selection_mode="filtering")
        scores = out["state"].grad_dot_scores.cpu()
        close(scores, REF.scores / B_TRAIN, f"GlobalSubset scores (packed={packed})", atol=1e-7, rtol=1e-4)
        assert not torch.allclose(scores, REF.linear_only_scores / B_TRAIN, atol=1e-7, rtol=1e-4), \
            "fixture: embedding/output contributions should change the global scores"
        kept = sorted(out["state"].get_final_selection().tolist())
        expected = REF.kept_global()
        assert kept == expected and 0 < len(kept) < B_TRAIN, (kept, expected)
    print(f"  GlobalSubset: scores == sum over emb/fc/proj/out of <grad l_b, mean_j grad l_j>/n (both layouts); keeps {expected}")


def test_layerwise_filtering_embedding_and_output():
    for packed in (False, True):
        out = run("LayerWiseSubset", packed=packed, frac=1.0, selection_mode="filtering")
        for l in (EMB, OUT):
            kept = REF.kept(l)
            assert kept and len(kept) < B_TRAIN, f"fixture should keep a strict subset for layer {l}: {kept}"
            expected = mean_grads([REF.train[b] for b in kept])
            for k in LAYER_KEYS[l]:
                close(out["grads"][k], expected[k], f"layer {HOOKED[l]} {k} (packed={packed}, kept={kept})")
        assert torch.all(out["grads"]["emb.weight"][PAD] == 0), "curated embedding grad keeps the padding row at zero"
    print(f"  LayerWiseSubset filtering: emb grad == mean over its positive responses {REF.kept(EMB)}, "
          f"out grad == mean over {REF.kept(OUT)} (both layouts)")


def test_frac1_reproduces_full_gradient_untied_and_tied():
    for tied in (False, True):
        plain = grads_for(lambda m: loss_of(m, TRAIN), tied=tied)
        for packed in (False, True):
            out = run("LayerWiseSubset", tied=tied, packed=packed, frac=1.0, selection_mode="topk")
            assert set(out["grads"]) == set(plain), (set(out["grads"]), set(plain))
            for k in plain:
                close(out["grads"][k], plain[k], f"tied={tied} packed={packed} frac=1 {k}")
    print("  frac=1 top-k == plain gradient for every parameter, untied and tied (shared emb/out weight), both layouts")


def test_tied_weight_accumulates_both_hooked_modules():
    for packed in (False, True):
        out = run("LayerWiseSubset", tied=True, packed=packed, frac=1.0, selection_mode="filtering")
        assert "out.weight" not in out["grads"], "tied model exposes the shared weight once"
        exp_emb = mean_grads([REF.train[b] for b in REF.kept(EMB)])["emb.weight"]
        exp_out = mean_grads([REF.train[b] for b in REF.kept(OUT)])["out.weight"]
        close(out["grads"]["emb.weight"], exp_emb + exp_out, f"tied shared grad (packed={packed})")
    print(f"  tied weight: shared grad == curated embedding grad (kept {REF.kept(EMB)}) + curated output grad (kept {REF.kept(OUT)})")


def test_tied_aware_global_scores_exact():
    """tie_embeddings: the two sites share the summed target, so the Global score is the exact tied inner product."""
    assert REF.tied_group_scores.abs().min() > 1e-3 and REF.tied_exact_scores.abs().min() > 1e-3, "fixture near a boundary"
    for packed in (False, True):
        out = run("GlobalSubset", tied=True, packed=packed, frac=1.0, selection_mode="filtering", tie_embeddings=True)
        assert sorted(out["hook"].tied_pairs.items()) == [(EMB, OUT), (OUT, EMB)], out["hook"].tied_pairs
        close(out["hook"].get_val_grad(EMB), REF.g_val["emb.weight"] + REF.g_val["out.weight"], "merged tied target")
        assert out["hook"].get_val_grad(EMB) is out["hook"].get_val_grad(OUT), "both sites must alias one target tensor"
        scores = out["state"].grad_dot_scores.cpu()
        close(scores, REF.tied_exact_scores / B_TRAIN, f"tied-aware GlobalSubset scores (packed={packed})", atol=1e-7, rtol=1e-4)
        assert not torch.allclose(scores, REF.scores / B_TRAIN, atol=1e-7, rtol=1e-4), "fixture: cross terms should change the scores"
    print("  tie_embeddings + GlobalSubset: scores == exact inner product of the tied model (cross terms included), both layouts")


def test_tied_aware_layerwise_one_subset():
    """tie_embeddings + LayerWiseSubset: one subset for the shared weight, chosen from the summed site scores."""
    kept = REF.kept_tied()
    assert 0 < len(kept) < B_TRAIN, f"fixture should keep a strict subset for the tied group: {kept}"
    for packed in (False, True):
        out = run("LayerWiseSubset", tied=True, packed=packed, frac=1.0, selection_mode="filtering", tie_embeddings=True)
        assert "out.weight" not in out["grads"]
        expected = mean_grads([REF.train[b] for b in kept])
        close(out["grads"]["emb.weight"], expected["emb.weight"] + expected["out.weight"],
              f"tied-aware shared grad (packed={packed}, kept={kept})")
        for l in (FC, PROJ):   # the other layers still select on their own
            exp_l = mean_grads([REF.train[b] for b in REF.kept(l)])
            for k in LAYER_KEYS[l]:
                close(out["grads"][k], exp_l[k], f"layer {HOOKED[l]} {k} under tie_embeddings (packed={packed})")
        sel = dict(out["state"]._layer_selections)
        assert sel.get(EMB) == len(kept) and sel.get(OUT) == len(kept), sel
        assert not out["hook"]._tied_stash, "stash must be consumed"
    same = set(kept) == set(REF.kept(EMB)) == set(REF.kept(OUT))
    print(f"  tie_embeddings + LayerWiseSubset: shared grad == mean over the tied group's kept set {kept} of (emb grad + out grad), "
          f"both layouts{' (on this fixture the tied subset coincides with both sites, the scores differ)' if same else ''}")


ALL_TESTS = [
    test_tied_aware_global_scores_exact,
    test_tied_aware_layerwise_one_subset,
    test_val_capture_embedding_and_output,
    test_global_scores_include_embedding_and_output,
    test_layerwise_filtering_embedding_and_output,
    test_frac1_reproduces_full_gradient_untied_and_tied,
    test_tied_weight_accumulates_both_hooked_modules,
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
