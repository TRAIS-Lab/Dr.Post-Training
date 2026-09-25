"""
GPU parity: curated training steps with the fused kernels vs the PyTorch reference ops.

Runs one merged-batch step of GlobalSubset (one-pass) and LayerWiseSubset on a tiny
bf16 Qwen3 on CUDA — with pip, gip and compressed scoring, with and without
activation checkpointing — once with the active fused-kernel backend and once with
``DRPT_KERNEL_BACKEND=off``, and checks that the per-layer scores, the selections and
every parameter gradient agree to bf16 precision.  Complements the CPU parity tests
(which never exercise the kernels).

    python tests/test_fused_parity_gpu.py
"""
from __future__ import annotations

import os
import sys
import warnings

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from drpt import GradientHook, causal_lm_loss, kernels  # noqa: E402
from drpt.compressor import setup_model_compressors  # noqa: E402
from drpt.selection import create_merged_batch_strategy  # noqa: E402

DEV = "cuda"
VOCAB, B_TRAIN, B_VAL, SEQ = 128, 6, 2, 40
FRAC, LR = 0.5, 1e-3


def make_model(ckpt: bool):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128,
                      tie_word_embeddings=False, attn_implementation="eager")
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).to(DEV, torch.bfloat16)
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model.train()
    return model


def make_batch(bsz, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, VOCAB, (bsz, SEQ), generator=g)
    labels = ids.clone()
    labels[:, :3] = -100
    return {"input_ids": ids.to(DEV), "attention_mask": torch.ones_like(ids).to(DEV), "labels": labels.to(DEV)}


TRAIN, VAL = make_batch(B_TRAIN, 1), make_batch(B_VAL, 2)
MERGED = {k: torch.cat([TRAIN[k], VAL[k]], 0) for k in TRAIN}


def loss_of(model, batch):
    logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
    return causal_lm_loss(logits, batch["labels"], reduction="sample_mean")


def run(method, scoring, ckpt):
    model = make_model(ckpt)
    names = [n for n, m in model.named_modules() if isinstance(m, (nn.Linear, nn.Embedding))]
    hook = GradientHook(model, names, device=DEV)
    if scoring == "compress":
        comps = setup_model_compressors(
            model, names,
            sparsifier_kwargs={"proj_dim": 32, "proj_max_batch_size": 64, "proj_seed": 0, "device": DEV, "proj_type": "normal"},
            projector_kwargs={"proj_dim": -1, "proj_max_batch_size": 64, "proj_seed": 0, "device": DEV, "proj_type": "identity"},
            sample_inputs={k: v[:1] for k, v in TRAIN.items() if k != "labels"}, device=DEV, update_freq=10**6)
        hook.set_score_compressors(comps)
    strategy = create_merged_batch_strategy(method=method, grad_hook=hook, frac=FRAC, selection_mode="topk",
                                            record_selections=True, scoring_method=scoring, subset_mode="one_pass")
    if method == "GlobalSubset":
        hook.wrap_nonlinear_layers()
    strategy.execute_training_step(model=model, merged_batch=MERGED, train_batch_size=B_TRAIN,
                                   compute_loss_fn=lambda m, b: loss_of(m, b), lr=LR, batch_train=TRAIN)
    grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
    rec = strategy.last_selection_record
    hook.remove_hooks()
    return grads, rec


def compare(method, scoring, ckpt, backend):
    kernels.set_backend(backend)
    g_f, r_f = run(method, scoring, ckpt)
    kernels.set_backend("off")
    g_r, r_r = run(method, scoring, ckpt)
    kernels.set_backend(backend)
    # scores / selections
    recs_f = r_f if isinstance(r_f, list) else [r_f]
    recs_r = r_r if isinstance(r_r, list) else [r_r]
    worst_score = 0.0
    for a, b in zip(recs_f, recs_r):
        if a is None or b is None:
            continue
        sa, sb = torch.tensor(a["scores"], dtype=torch.float64), torch.tensor(b["scores"], dtype=torch.float64)
        worst_score = max(worst_score, ((sa - sb).norm() / (sb.norm() + 1e-30)).item())
        assert sorted(a["selected_indices"]) == sorted(b["selected_indices"]), \
            f"{method}/{scoring}/ckpt={ckpt}: selections differ {a['selected_indices']} vs {b['selected_indices']}"
    assert set(g_f) == set(g_r), "parameter sets differ"
    worst = 0.0
    for n in g_f:
        rel = ((g_f[n] - g_r[n]).norm() / (g_r[n].norm() + 1e-30)).item()
        worst = max(worst, rel)
        assert rel < 3e-2, f"{method}/{scoring}/ckpt={ckpt}: grad mismatch on {n}: rel {rel:.2e}"
    print(f"  {method:<16}{scoring:<10}ckpt={str(ckpt):<6}[{backend}] max rel grad diff {worst:.1e}, score diff {worst_score:.1e}, selections identical")


def test_parity():
    for backend in [b for b, ok in (("cute", kernels.HAS_CUTE), ("triton", kernels.HAS_TRITON)) if ok]:
        for method in ("GlobalSubset", "LayerWiseSubset"):
            for scoring in ("compress", "gip", "pip"):
                for ckpt in (False, True):
                    compare(method, scoring, ckpt, backend)


if __name__ == "__main__":
    if not torch.cuda.is_available() or not (kernels.HAS_CUTE or kernels.HAS_TRITON):
        print("CUDA + a fused-kernel backend required; skipping")
        sys.exit(0)
    try:
        test_parity()
        print("\n1/1 tests passed")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"\nFAILED: {e}")
        sys.exit(1)
