"""
Parity tests for GroupWiseSubset curation (drpt.selection).

All runs use the "sample_mean" loss convention (drpt.losses.causal_lm_loss: mean over
examples of per-example token-mean losses), which is the GradientHook default; see
tests/test_loss_convention.py for the convention itself and the legacy "token_mean".

Runs on CPU with a tiny randomly initialised Qwen3 model, so it can be executed
without a GPU allocation:

    python tests/test_groupwise_selection.py            # plain runner
    pytest tests/test_groupwise_selection.py -q         # if pytest is available

What is checked
---------------
1. Layer grouping presets / custom rules on HF and PEFT layer names.
2. GroupWiseSubset with selection_frac=1.0 reproduces plain autograd gradients
   (separate-batch and merged-batch modes).
3. granularity=layer  == LayerWiseSubset      (selections and gradients)
4. granularity=global == GlobalSubset one_pass (selection and gradients)
5. granularity=block / sublayer / custom rules: groups are contiguous in
   backward order, every hooked parameter receives a gradient, and the
   block-wise selection equals top-k of the summed per-layer LayerWiseSubset
   scores.
"""

from __future__ import annotations

import copy
import os
import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from drpt import GradientHook, causal_lm_loss  # noqa: E402
from drpt.selection import (  # noqa: E402
    build_layer_groups,
    create_merged_batch_strategy,
    create_separate_batch_strategy,
    group_members,
    parse_group_rules,
)
from drpt.utils import topk_selection  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(2)

# ----------------------------------------------------------------------------- fixtures

VOCAB, B_TRAIN, B_VAL, SEQ = 101, 6, 2, 12
FRAC = 0.5
LR = 1e-3


TIE_EMBEDDINGS = False   # toggled by test_tied_embeddings_frac1


def make_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(
        vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=64, tie_word_embeddings=TIE_EMBEDDINGS, attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).float()
    if TIE_EMBEDDINGS:
        model.tie_weights()
        assert model.lm_head.weight is model.model.embed_tokens.weight
    return model


def hooked_layer_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))]


# Trailing positions masked per row: uneven response lengths (5..10 supervised tokens),
# so the "sample_mean" and legacy "token_mean" conventions give different scores.
RESPONSE_CUTS = [3, 0, 5, 1, 4, 2]


def make_batch(bsz, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (bsz, SEQ), generator=g)
    labels = ids.clone()
    labels[:, :2] = -100                      # prompt tokens are not trained on
    for i in range(bsz):
        cut = RESPONSE_CUTS[i % len(RESPONSE_CUTS)]
        if cut:
            labels[i, -cut:] = -100           # uneven response lengths
    attn = torch.ones_like(ids)
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


TRAIN = make_batch(B_TRAIN, 1)
VAL = make_batch(B_VAL, 2)


def loss_of(model, batch):
    """Sample-mean loss: mean over examples of per-example token-mean CE (the drpt default)."""
    logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
    return causal_lm_loss(logits, batch["labels"], reduction="sample_mean")


def hf_loss_of(model, batch):
    """Hugging Face's token mean over the batch (legacy "token_mean" convention)."""
    return model(**batch).loss


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def plain_train_grads():
    model = make_model()
    model.zero_grad()
    loss_of(model, TRAIN).backward()
    return grads_of(model)


def merge(train, val):
    return {k: torch.cat([train[k], val[k]], 0) for k in train}


# ----------------------------------------------------------------------------- runners

def run_separate(method, frac=FRAC, granularity="block", rules=None, record=True):
    """One curated training step in separate-batch (cached-val) mode."""
    model = make_model()
    names = hooked_layer_names(model)
    hook = GradientHook(model, names, device="cpu")
    if method == "GroupWiseSubset":
        hook.set_layer_groups(build_layer_groups(names, granularity, rules))
    strategy = create_separate_batch_strategy(
        method=method, grad_hook=hook, frac=frac, selection_mode="topk",
        record_selections=record, scoring_method="reduced_ghost", subset_mode="one_pass",
    )

    hook.start_val_capture(scoring_method="reduced_ghost")
    model.zero_grad()
    loss_of(model, VAL).backward()
    hook.end_val_capture()

    loss, stats = strategy.execute_training_step(
        model=model, batch_size=B_TRAIN,
        compute_loss_fn=lambda: (loss_of(model, TRAIN), {}),
        lr=LR, labels=TRAIN["labels"],
        filter_batch_fn=lambda idx: (lambda: (loss_of(model, {k: v[idx] for k, v in TRAIN.items()}), {})),
    )
    hook.clear_val_buffer()
    out = dict(grads=grads_of(model), records=strategy.last_selection_record, stats=stats,
               names=names, hook=hook, model=model)
    hook.remove_hooks()
    return out


def run_merged(method, frac=FRAC, granularity="block", rules=None):
    """One curated training step in merged-batch mode."""
    model = make_model()
    names = hooked_layer_names(model)
    hook = GradientHook(model, names, device="cpu")
    if method == "GroupWiseSubset":
        hook.set_layer_groups(build_layer_groups(names, granularity, rules))
    strategy = create_merged_batch_strategy(
        method=method, grad_hook=hook, frac=frac, selection_mode="topk",
        record_selections=True, scoring_method="reduced_ghost", subset_mode="one_pass",
    )
    merged = merge(TRAIN, VAL)
    strategy.execute_training_step(
        model=model, merged_batch=merged, train_batch_size=B_TRAIN,
        compute_loss_fn=lambda m, b: loss_of(m, b), lr=LR, batch_train=TRAIN,
    )
    out = dict(grads=grads_of(model), records=strategy.last_selection_record, names=names)
    hook.remove_hooks()
    return out


def assert_same_grads(a, b, atol=1e-5, rtol=1e-4, what=""):
    assert set(a) == set(b), f"{what}: parameter sets differ: {set(a) ^ set(b)}"
    worst = 0.0
    for n in a:
        diff = (a[n] - b[n]).abs().max().item()
        worst = max(worst, diff)
        assert torch.allclose(a[n], b[n], atol=atol, rtol=rtol), f"{what}: grad mismatch on {n}: max|diff|={diff:.3e}"
    return worst


# ----------------------------------------------------------------------------- tests

def test_grouping_presets():
    names = [
        "model.embed_tokens",
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj", "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj",
        "model.layers.1.self_attn.q_proj", "model.layers.1.mlp.down_proj",
        "lm_head",
    ]
    assert build_layer_groups(names, "layer") == names
    assert len(set(build_layer_groups(names, "global"))) == 1

    block = build_layer_groups(names, "block")
    assert block[0] == "model.embed_tokens" and block[-1] == "lm_head"
    assert set(block[1:8]) == {"model.layers.0"} and set(block[8:10]) == {"model.layers.1"}

    sub = build_layer_groups(names, "sublayer")
    assert set(sub[1:5]) == {"model.layers.0.self_attn"} and set(sub[5:8]) == {"model.layers.0.mlp"}

    rules = ("attn.qkv=self_attn.q_proj,self_attn.k_proj,self_attn.v_proj;attn.o=self_attn.o_proj;"
             "mlp.gateup=gate_proj,up_proj;mlp.down=mlp.down_proj")
    assert list(parse_group_rules(rules)) == ["attn.qkv", "attn.o", "mlp.gateup", "mlp.down"]
    cust = build_layer_groups(names, "custom", rules)
    assert cust[1:4] == ["model.layers.0.attn.qkv"] * 3
    assert cust[4] == "model.layers.0.attn.o"
    assert cust[5:7] == ["model.layers.0.mlp.gateup"] * 2 and cust[7] == "model.layers.0.mlp.down"
    assert cust[0] == "model.embed_tokens" and cust[-1] == "lm_head"
    # rules imply custom granularity
    assert build_layer_groups(names, "block", rules) == cust

    # PEFT names keep the block prefix and match leaf members
    peft = ["base_model.model.model.layers.4.self_attn.q_proj.lora_A.default",
            "base_model.model.model.layers.4.self_attn.q_proj.lora_B.default",
            "base_model.model.model.layers.4.mlp.up_proj.lora_A.default"]
    assert build_layer_groups(peft, "block") == ["base_model.model.model.layers.4"] * 3
    assert build_layer_groups(peft, "sublayer")[:2] == ["base_model.model.model.layers.4.self_attn"] * 2
    assert build_layer_groups(peft, "custom", "qkv=q_proj,k_proj,v_proj")[0] == "base_model.model.model.layers.4.qkv"
    assert build_layer_groups(peft, "custom", "qkv=q_proj,k_proj,v_proj")[2] == peft[2]  # unmatched -> singleton

    # partial-match must not fire: 'proj' is not a whole component
    assert build_layer_groups(names, "custom", "x=proj")[1] == names[1]

    for bad in ["nogroup", "a=", "=b", "a=b;a=c"]:
        try:
            parse_group_rules(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rule {bad!r} should be rejected")
    print("  grouping presets OK")


def test_frac1_matches_plain_separate():
    ref = plain_train_grads()
    for gran in ["block", "sublayer", "layer", "global"]:
        out = run_separate("GroupWiseSubset", frac=1.0, granularity=gran)
        worst = assert_same_grads(out["grads"], ref, what=f"separate/{gran}/frac=1")
        print(f"  separate-batch frac=1.0 [{gran}] == plain autograd (max|diff|={worst:.2e})")


def test_frac1_matches_plain_merged():
    ref = plain_train_grads()
    for gran in ["block", "sublayer"]:
        out = run_merged("GroupWiseSubset", frac=1.0, granularity=gran)
        worst = assert_same_grads(out["grads"], ref, atol=1e-4, rtol=1e-3, what=f"merged/{gran}/frac=1")
        print(f"  merged-batch frac=1.0 [{gran}] == plain train-only autograd (max|diff|={worst:.2e})")


def test_layer_granularity_equals_layerwise():
    lw = run_separate("LayerWiseSubset")
    gw = run_separate("GroupWiseSubset", granularity="layer")
    assert len(lw["records"]) == len(gw["records"]) == len(lw["names"])
    for r_lw, r_gw in zip(lw["records"], gw["records"]):
        assert r_gw["group"] == lw["names"][r_lw["layer_idx"]]
        assert r_lw["selected_indices"] == r_gw["selected_indices"], (r_lw, r_gw)
        assert torch.allclose(torch.tensor(r_lw["scores"]), torch.tensor(r_gw["scores"]), atol=1e-6, rtol=1e-5)
    worst = assert_same_grads(gw["grads"], lw["grads"], what="layer vs LayerWiseSubset")
    print(f"  granularity=layer == LayerWiseSubset (selections identical, max|grad diff|={worst:.2e})")


def test_global_granularity_equals_globalsubset():
    gs = run_separate("GlobalSubset")
    gw = run_separate("GroupWiseSubset", granularity="global")
    assert len(gw["records"]) == 1 and len(gs["records"]) == 1
    assert sorted(gs["records"][0]["selected_indices"]) == gw["records"][0]["selected_indices"]
    assert torch.allclose(torch.tensor(gs["records"][0]["scores"]), torch.tensor(gw["records"][0]["scores"]), atol=1e-6, rtol=1e-5)
    worst = assert_same_grads(gw["grads"], gs["grads"], what="global vs GlobalSubset one_pass")
    print(f"  granularity=global == GlobalSubset one_pass (selection identical, max|grad diff|={worst:.2e})")


def test_block_selection_is_topk_of_summed_layer_scores():
    lw = run_separate("LayerWiseSubset")
    bw = run_separate("GroupWiseSubset", granularity="block")
    names = lw["names"]
    groups = build_layer_groups(names, "block")
    members = group_members(names, groups)

    per_layer_scores = {r["layer_idx"]: torch.tensor(r["scores"]) for r in lw["records"]}
    k = max(1, int(B_TRAIN * FRAC))
    by_group = {r["group"]: r for r in bw["records"]}
    assert set(by_group) == set(members), (set(by_group) ^ set(members))
    for key, idxs in members.items():
        summed = sum(per_layer_scores[i] for i in idxs)
        expected = topk_selection(summed * LR, k).sort()[0].tolist()
        assert by_group[key]["selected_indices"] == expected, (key, by_group[key]["selected_indices"], expected)
        assert by_group[key]["layer_indices"] == idxs
    n_blocks = sum(1 for key in members if ".layers." in key)
    assert n_blocks == 3, members.keys()
    # every hooked weight got a curated gradient
    for n in names:
        assert f"{n}.weight" in bw["grads"], f"missing grad for {n}"
    print(f"  granularity=block: {len(members)} groups ({n_blocks} decoder blocks + embed + lm_head); "
          f"selection == top-k of summed per-layer scores; all hooked weights have grads")


def test_sublayer_and_custom_rules_run_contiguously():
    for gran, rules, expected_groups_per_block in [
        ("sublayer", None, 2),
        ("custom", "attn.qkv=q_proj,k_proj,v_proj;attn.o=o_proj;mlp.gateup=gate_proj,up_proj;mlp.down=down_proj", 4),
    ]:
        out = run_separate("GroupWiseSubset", granularity=gran, rules=rules)
        groups = build_layer_groups(out["names"], gran, rules)
        members = group_members(out["names"], groups)
        n_in_blocks = sum(1 for key in members if ".layers." in key)
        assert n_in_blocks == 3 * expected_groups_per_block, (gran, n_in_blocks)
        assert len(out["records"]) == len(members)
        assert out["stats"]["selection/n_selected"] == max(1, int(B_TRAIN * FRAC))
        for n in out["names"]:
            assert f"{n}.weight" in out["grads"], f"missing grad for {n}"
        print(f"  granularity={gran}: {len(members)} groups, all finalized, every hooked weight has a grad")


def test_groups_finalize_inside_backward():
    """All groups complete during loss.backward(): finalize_remaining() finds nothing."""
    import logging
    from drpt.selection import strategies as S
    messages = []

    class H(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())

    h = H()
    S.logger.addHandler(h)
    S.logger.setLevel(logging.INFO)
    try:
        run_separate("GroupWiseSubset", granularity="block")
        run_merged("GroupWiseSubset", granularity="sublayer")
    finally:
        S.logger.removeHandler(h)
    assert not any("did not run backward" in m for m in messages), messages
    assert not any("not contiguous" in m for m in messages), messages
    assert sum("contiguous in backward order" in m for m in messages) == 2, messages
    print("  every group finalized inside backward; all groups contiguous (separate + merged)")


def test_selection_frac_effect_on_grads():
    """With frac<1 the curated gradient differs from plain training (sanity: selection is active)."""
    ref = plain_train_grads()
    out = run_separate("GroupWiseSubset", frac=FRAC, granularity="block")
    changed = sum(1 for n in ref if not torch.allclose(ref[n], out["grads"][n], atol=1e-6))
    assert changed > 0
    print(f"  frac={FRAC}: {changed}/{len(ref)} parameter gradients differ from plain training (selection active)")


def test_tied_embeddings_frac1():
    """Qwen3-1.7B-Base ties lm_head to embed_tokens: two singleton groups write into one .grad."""
    global TIE_EMBEDDINGS
    TIE_EMBEDDINGS = True
    try:
        ref = plain_train_grads()
        assert "lm_head.weight" not in ref and "model.embed_tokens.weight" in ref
        for gran in ["block", "sublayer"]:
            out = run_separate("GroupWiseSubset", frac=1.0, granularity=gran)
            worst = assert_same_grads(out["grads"], ref, what=f"tied/separate/{gran}")
            out_m = run_merged("GroupWiseSubset", frac=1.0, granularity=gran)
            worst_m = assert_same_grads(out_m["grads"], ref, atol=1e-4, rtol=1e-3, what=f"tied/merged/{gran}")
            print(f"  tied embeddings frac=1.0 [{gran}]: separate max|diff|={worst:.2e}, merged max|diff|={worst_m:.2e}")
        # with selection active, the tied weight still receives exactly one accumulated grad (no double count):
        # lm_head + embed groups each add their curated part; check it equals LayerWiseSubset's result.
        lw = run_separate("LayerWiseSubset")
        gw = run_separate("GroupWiseSubset", granularity="layer")
        assert_same_grads(gw["grads"], lw["grads"], what="tied/layer vs LayerWiseSubset")
        print("  tied embeddings frac=0.5 [layer] == LayerWiseSubset")
    finally:
        TIE_EMBEDDINGS = False


ALL_TESTS = [
    test_grouping_presets,
    test_tied_embeddings_frac1,
    test_frac1_matches_plain_separate,
    test_frac1_matches_plain_merged,
    test_layer_granularity_equals_layerwise,
    test_global_granularity_equals_globalsubset,
    test_block_selection_is_topk_of_summed_layer_scores,
    test_sublayer_and_custom_rules_run_contiguously,
    test_groups_finalize_inside_backward,
    test_selection_frac_effect_on_grads,
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
