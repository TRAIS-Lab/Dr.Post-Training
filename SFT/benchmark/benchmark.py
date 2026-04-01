#!/usr/bin/env python
"""
Timing Benchmark for Dr. Post-Training.

One subprocess per method (Standard / Layerwise / Subset).
CUDA events at every component boundary — no residual decomposition.

Usage:
  python benchmark.py --batch-size 8 --seq-length 512
  python benchmark.py --method standard  # single method (subprocess mode)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from SFT.benchmark.utils import (
    BenchmarkConfig,
    set_seed,
    setup_model,
    setup_grad_hook,
    create_dataloaders,
    get_batches,
    pad_and_merge_batches,
    CUDAEventTimer,
    EventRecorder,
)
from drpt import GradientHook

METHODS = ["standard", "layerwise", "subset", "subset_one_pass"]


# =============================================================================
# Shared helpers
# =============================================================================

def _warmup_and_measure(step_fn, train_batches, val_batches, config, rec):
    """Run warmup + timed iterations, collecting EventRecorder data."""
    N = config.num_iterations
    comp_accum = {}

    for i in range(config.num_warmup):
        rec.reset()
        if val_batches is not None:
            step_fn(train_batches[i % len(train_batches)],
                    val_batches[i % len(val_batches)])
        else:
            step_fn(train_batches[i % len(train_batches)])

    for i in range(N):
        rec.reset()
        if val_batches is not None:
            step_fn(train_batches[(config.num_warmup + i) % len(train_batches)],
                    val_batches[(config.num_warmup + i) % len(val_batches)], i)
        else:
            step_fn(train_batches[(config.num_warmup + i) % len(train_batches)], i)
        torch.cuda.synchronize()
        rec.accumulate(comp_accum)

    return {k: v / N for k, v in comp_accum.items()}


# =============================================================================
# Measurement: Standard
# =============================================================================

def measure_standard(model, optimizer, grad_hook, train_batches, config):
    """Measure Standard training with per-layer act_grad / w.grad breakdown."""
    N = config.num_iterations
    timer = CUDAEventTimer(["forward", "backward", "optimizer"], N)
    rec = EventRecorder()

    # Monkey-patch Linear layers with timed backward
    class TimedLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight, bias)
            return F.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            if input.dtype != grad_output.dtype:
                input = input.to(grad_output.dtype)
            rec.mark('act_grad')
            grad_input = grad_output @ weight.to(grad_output.dtype)
            rec.mark('act_grad')
            rec.mark('wgrad')
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            in_2d = input.reshape(-1, input.shape[-1])
            grad_weight = go_2d.T @ in_2d
            grad_bias = go_2d.sum(dim=0) if bias is not None else None
            rec.mark('wgrad')
            return grad_input, grad_weight, grad_bias

    patched = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, '_original_forward'):
            orig = module._original_forward
            def _make(mod):
                return lambda input: TimedLinear.apply(input, mod.weight, mod.bias)
            module._original_forward = _make(module)
            patched.append((module, orig))

    grad_hook.disable_hooks()

    def step(batch, i=None):
        optimizer.zero_grad()
        if i is not None: timer.mark("forward", i, True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(**batch).loss
        if i is not None: timer.mark("forward", i, False)
        if i is not None: timer.mark("backward", i, True)
        loss.backward()
        if i is not None: timer.mark("backward", i, False)
        if i is not None: timer.mark("optimizer", i, True)
        optimizer.step()
        if i is not None: timer.mark("optimizer", i, False)

    comp = _warmup_and_measure(step, train_batches, None, config, rec)

    for module, orig in patched:
        module._original_forward = orig
    grad_hook.enable_hooks()

    result = timer.mean_elapsed()
    result.update(comp)
    return result


# =============================================================================
# Measurement: Layerwise
# =============================================================================

def measure_layerwise(model, optimizer, grad_hook, train_batches, val_batches, config, tokenizer):
    """Measure Layerwise with full backward breakdown: act_grad, compress, score, select, w.grad."""
    import drpt.selection.backward as _bwd
    from drpt.selection.backward import (
        augment_input_for_bias, split_train_val_batch,
        _do_selection, _store_update_grad, _produce_gradient_update,
    )

    N = config.num_iterations
    timer = CUDAEventTimer(["forward", "backward", "optimizer"], N)
    rec = EventRecorder()
    pad_token_id = tokenizer.pad_token_id or 0

    # Save originals
    _orig_backward = _bwd.LayerwiseLinearBackward.backward
    _orig_compressed = _bwd.LayerwiseLinearBackward._backward_compressed

    # Patched backward: times act_grad
    @staticmethod
    def timed_backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hm = ctx.hook_manager_ref()
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)
        rec.mark('act_grad')
        grad_input = grad_output @ weight.to(grad_output.dtype)
        rec.mark('act_grad')
        sc = hm.score_compressors[layer_idx]
        uc = hm.update_compressors[layer_idx]
        state = hm.selection_state
        cvm = hm.capture_val_mode
        usv = state is not None and getattr(state, '_use_stored_val', False)
        with torch.no_grad():
            if sc is not None:
                gw, gb = _bwd.LayerwiseLinearBackward._backward_compressed(
                    hm, sc, uc, state, layer_idx, input, grad_output, bias, cvm, usv)
            else:
                gw, gb = _bwd.LayerwiseLinearBackward._backward_full(
                    hm, uc, state, layer_idx, input, bias, grad_output, cvm, usv)
        return grad_input, gw, gb, None, None

    # Patched _backward_compressed: times compress, score, select, wgrad
    @staticmethod
    def timed_compressed(hm, sc, uc, state, lidx, inp, go, bias, cvm, usv):
        hb = bias is not None
        ia = augment_input_for_bias(inp, hb)
        rec.mark('compress'); scg = sc.forward((go, ia)); rec.mark('compress')
        if cvm:
            tg = scg.sum(dim=0); vc = hm.val_cache
            vc._compressed[lidx] = tg if vc._compressed[lidx] is None else vc._compressed[lidx] + tg
            return None, None
        if state is None:
            _store_update_grad(hm, uc, sc, lidx, go, ia, hb, scg.sum(dim=0, keepdim=True))
            return None, None
        rec.mark('score')
        if usv:
            tg2, vg, scorr = scg, hm._val_cache.get_compressed(lidx), None
        else:
            tg2, vgs = split_train_val_batch(scg, state.train_batch_size)
            vg = vgs.sum(dim=0); scorr = state.score_correction
        if vg is None:
            rec.mark('score')
            _store_update_grad(hm, uc, sc, lidx, go, ia, hb, scg.mean(dim=0, keepdim=True))
            return None, None
        scores = tg2 @ vg
        if scorr is not None: scores = scores * scorr
        sim = None
        if state.use_second_order:
            sim = tg2 @ tg2.T
            if scorr is not None: sim = sim * (scorr ** 2)
        rec.mark('score')
        if uc is not None and uc is sc:
            # Unfuse process_layer_gradients for fair timing: select, then reduce
            rec.mark('select')
            si = _do_selection(state, lidx, scores, sim)
            rec.mark('select')
            rec.mark('wgrad')
            selected_grads = tg2[si]
            scale_factor = state._compute_scale_factor(si)
            rg = selected_grads.sum(dim=0, keepdim=True) * scale_factor
            hm._store_compressed_grad(lidx, rg)
            state.num_selected = si.shape[0]
            rec.mark('wgrad')
            return None, None
        # select: top-k only (apple-to-apple with Subset selection)
        rec.mark('select')
        si = _do_selection(state, lidx, scores, sim)
        rec.mark('select')

        # wgrad: split + indexing + matmul
        rec.mark('wgrad')
        if usv: tgo, ti = go, inp
        else:
            tgo, _ = split_train_val_batch(go, state.train_batch_size)
            ti, _ = split_train_val_batch(inp, state.train_batch_size)
        scale_factor = state._compute_scale_factor(si)
        sel_go = tgo[si]
        sel_inp = ti[si]
        if uc is not None:
            sel_inp_aug = augment_input_for_bias(sel_inp, hb)
            uc_compressed = uc.forward((sel_go, sel_inp_aug))
            reduced = uc_compressed.mean(dim=0, keepdim=True) * scale_factor
            hm._store_compressed_grad(lidx, reduced)
            rec.mark('wgrad')
            return None, None
        else:
            if sel_go.dim() == 3:
                gw = torch.einsum('kso,ksi->oi', sel_go, sel_inp) * scale_factor
                gb = sel_go.sum(dim=(0,1)) * scale_factor if hb else None
            else:
                gw = torch.einsum('ko,ki->oi', sel_go, sel_inp) * scale_factor
                gb = sel_go.sum(dim=0) * scale_factor if hb else None
            rec.mark('wgrad')
            return gw, gb

    # Patched _backward_full: times score, select, wgrad (no compression step)
    _orig_full = _bwd.LayerwiseLinearBackward._backward_full

    @staticmethod
    def timed_full(hm, uc, state, lidx, inp, bias, go, cvm, usv):
        from drpt.selection.backward import (
            _get_val_components, _dispatch_scoring, _do_selection as _do_sel,
            _produce_gradient_update as _produce_gu,
            _produce_gradient_update_with_val as _produce_gu_val,
        )
        hb = bias is not None
        if cvm:
            hm.val_cache.store_layer(layer_idx=lidx, grad_output=go.detach(),
                                     input=inp.detach(), compressor=None)
            return None, None
        if state is None:
            if uc is not None:
                ia = augment_input_for_bias(inp, hb)
                uc_c = uc.forward((go, ia))
                hm._store_compressed_grad(lidx, uc_c.sum(dim=0, keepdim=True))
            return None, None
        # Score
        rec.mark('score')
        if usv:
            tgo, ti = go, inp
            vgo, vinp, vgt = _get_val_components(hm, lidx)
            if vgo is None and vgt is None:
                rec.mark('score')
                return None, None
            scorr = None
        else:
            tgo, vgo = split_train_val_batch(go, state.train_batch_size)
            ti, vinp = split_train_val_batch(inp, state.train_batch_size)
            vgt = None
            scorr = state.score_correction
        sm = getattr(state, 'scoring_method', 'reduced_ghost')
        scores, sim = _dispatch_scoring(sm, tgo, ti, vgo, vinp, vgt, state.use_second_order)
        if scorr is not None:
            scores = scores * scorr
            if sim is not None: sim = sim * (scorr ** 2)
        rec.mark('score')
        # Select
        rec.mark('select')
        si = _do_sel(state, lidx, scores, sim)
        rec.mark('select')
        # Wgrad
        rec.mark('wgrad')
        if (getattr(state, 'include_val_in_update', False) and not usv):
            gw, gb = _produce_gu_val(state, si, tgo, ti, vgo, vinp, hb)
        else:
            gw, gb = _produce_gu(hm, uc, state, lidx, tgo, ti, si, hb)
        rec.mark('wgrad')
        return gw, gb

    _bwd.LayerwiseLinearBackward.backward = timed_backward
    _bwd.LayerwiseLinearBackward._backward_compressed = timed_compressed
    _bwd.LayerwiseLinearBackward._backward_full = timed_full

    # --- Embedding timing patch ---
    _orig_emb_backward = _bwd.LayerwiseEmbeddingBackward.backward

    @staticmethod
    def timed_emb_backward(ctx, grad_output):
        from drpt.selection.utils import (
            compute_embedding_scores,
            compute_embedding_val_gradient,
            compute_embedding_selected_gradients,
            split_train_val_batch,
        )
        from drpt.selection.backward import _do_selection

        input_ids, weight = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        padding_idx = ctx.padding_idx
        V, D = weight.shape

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            return None, None, None, None, None

        state = hook_manager.selection_state
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            getattr(state, '_use_stored_val', False)
        )

        with torch.no_grad():
            if capture_val_mode:
                val_grad = compute_embedding_val_gradient(grad_output, input_ids, V, D)
                hook_manager.val_cache.store_precomputed(layer_idx, val_grad)
                return None, None, None, None, None

            if state is None:
                return None, None, None, None, None

            # Score (val gradient + scoring)
            rec.mark('emb_score')
            if use_stored_val:
                train_go, train_ids = grad_output, input_ids
                val_grad_weight = hook_manager.val_cache.get_full(layer_idx)
                if val_grad_weight is None:
                    rec.mark('emb_score')
                    return None, None, None, None, None
                score_correction = None
            else:
                train_go, val_go = split_train_val_batch(grad_output, state.train_batch_size)
                train_ids, val_ids = split_train_val_batch(input_ids, state.train_batch_size)
                val_grad_weight = compute_embedding_val_gradient(val_go, val_ids, V, D)
                score_correction = state.score_correction

            scores = compute_embedding_scores(train_go, train_ids, val_grad_weight)
            if score_correction is not None:
                scores = scores * score_correction
            rec.mark('emb_score')

            # Select
            rec.mark('emb_select')
            selected_indices = _do_selection(state, layer_idx, scores, None)
            rec.mark('emb_select')

            # Wgrad
            rec.mark('emb_wgrad')
            scale_factor = state._compute_scale_factor(selected_indices)
            grad_weight = compute_embedding_selected_gradients(
                train_go, train_ids, selected_indices, scale_factor,
                V, D, padding_idx,
            )
            grad_weight = grad_weight.to(weight.dtype)
            rec.mark('emb_wgrad')

        return None, grad_weight, None, None, None

    _bwd.LayerwiseEmbeddingBackward.backward = timed_emb_backward

    def step(batch, val_batch, i=None):
        train_bs = batch['input_ids'].shape[0]
        merged = pad_and_merge_batches(batch, val_batch, pad_token_id=pad_token_id)
        grad_hook.setup_selection(
            train_batch_size=train_bs, selection_method="Layerwise",
            frac=0.5, lr=optimizer.param_groups[0].get("lr", 5e-5),
            selection_mode="topk", use_second_order=config.use_second_order,
            scoring_method=getattr(config, 'scoring_method', 'reduced_ghost'),
            direct_batch_size=getattr(config, 'direct_batch_size', 0),
        )
        if 'labels' in merged:
            grad_hook.set_token_counts(merged['labels'], train_bs)
        optimizer.zero_grad()
        if i is not None: timer.mark("forward", i, True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(**merged).loss
        if i is not None: timer.mark("forward", i, False)
        if i is not None: timer.mark("backward", i, True)
        loss.backward()
        if i is not None: timer.mark("backward", i, False)
        grad_hook.clear_selection()
        grad_hook.clear_token_counts()
        if i is not None: timer.mark("optimizer", i, True)
        optimizer.step()
        if i is not None: timer.mark("optimizer", i, False)

    comp = _warmup_and_measure(step, train_batches, val_batches, config, rec)

    _bwd.LayerwiseLinearBackward.backward = _orig_backward
    _bwd.LayerwiseLinearBackward._backward_compressed = _orig_compressed
    _bwd.LayerwiseLinearBackward._backward_full = _orig_full
    _bwd.LayerwiseEmbeddingBackward.backward = _orig_emb_backward

    result = timer.mean_elapsed()
    result.update(comp)
    return result


# =============================================================================
# Measurement: Subset
# =============================================================================

def measure_subset(model, optimizer, grad_hook, train_batches, val_batches, config, tokenizer):
    """Measure Subset with pass-1 breakdown (act_grad, compress, score) and pass-2 phases."""
    import drpt.selection.backward as _bwd
    from drpt.selection.backward import augment_input_for_bias, split_train_val_batch
    from drpt.selection.state import SubsetState

    N = config.num_iterations
    phases = ["pass1_forward", "pass1_backward", "selection",
              "pass2_forward", "pass2_backward", "optimizer"]
    timer = CUDAEventTimer(phases, N)
    rec = EventRecorder()      # pass 1 component events
    p2_rec = EventRecorder()   # pass 2 component events
    pad_token_id = tokenizer.pad_token_id or 0

    # TimedLinear for pass 2 backward breakdown (act_grad vs w.grad)
    class TimedLinearP2(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight, bias)
            return F.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            if input.dtype != grad_output.dtype:
                input = input.to(grad_output.dtype)
            p2_rec.mark('p2_act_grad')
            grad_input = grad_output @ weight.to(grad_output.dtype)
            p2_rec.mark('p2_act_grad')
            p2_rec.mark('p2_wgrad')
            go_2d = grad_output.reshape(-1, grad_output.shape[-1])
            in_2d = input.reshape(-1, input.shape[-1])
            grad_weight = go_2d.T @ in_2d
            grad_bias = go_2d.sum(dim=0) if bias is not None else None
            p2_rec.mark('p2_wgrad')
            return grad_input, grad_weight, grad_bias

    # Pre-build patched forwards for pass 2
    _p2_patches = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) and hasattr(module, '_original_forward'):
            def _make(mod):
                return lambda input: TimedLinearP2.apply(input, mod.weight, mod.bias)
            _p2_patches.append((module, module._original_forward, _make(module)))

    # Save originals
    _orig_backward = _bwd.SubsetLinearBackward.backward
    _orig_accum = _bwd.SubsetLinearBackward._accumulate_compressed
    _orig_accum_full = _bwd.SubsetLinearBackward._accumulate_full

    # Patched backward: times act_grad
    @staticmethod
    def timed_backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hm = ctx.hook_manager_ref()
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)
        rec.mark('act_grad')
        grad_input = grad_output @ weight.to(grad_output.dtype)
        rec.mark('act_grad')
        sc = hm.score_compressors[layer_idx]
        state = hm.selection_state
        if state is None:
            return grad_input, None, None, None, None
        usv = getattr(state, '_use_stored_val', False)
        with torch.no_grad():
            if sc is not None:
                _bwd.SubsetLinearBackward._accumulate_compressed(
                    hm, sc, state, layer_idx, input, grad_output, bias, usv)
            else:
                _bwd.SubsetLinearBackward._accumulate_full(
                    hm, state, layer_idx, input, grad_output, bias, usv)
        return grad_input, None, None, None, None

    # Patched _accumulate_compressed: times compress, score
    @staticmethod
    def timed_accum(hm, compressor, state, lidx, inp, go, bias, usv):
        ia = augment_input_for_bias(inp, bias is not None)
        rec.mark('compress'); cg = compressor.forward((go, ia)); rec.mark('compress')
        rec.mark('score')
        if usv:
            tg, vg, scorr = cg, hm._val_cache.get_compressed(lidx), None
        else:
            tg, vgs = split_train_val_batch(cg, state.train_batch_size)
            vg = vgs.sum(dim=0); scorr = state.score_correction
        if vg is not None:
            state.process_layer_gradients(tg, vg, lidx, scorr)
        rec.mark('score')

    # Patched _accumulate_full: times score (no compression step)
    @staticmethod
    def timed_accum_full(hm, state, lidx, inp, go, bias, usv):
        rec.mark('score')
        _orig_accum_full(hm, state, lidx, inp, go, bias, usv)
        rec.mark('score')

    _bwd.SubsetLinearBackward.backward = timed_backward
    _bwd.SubsetLinearBackward._accumulate_compressed = timed_accum
    _bwd.SubsetLinearBackward._accumulate_full = timed_accum_full

    # --- Embedding timing patch (two-pass subset: score only, no wgrad in pass 1) ---
    _orig_emb_backward = _bwd.SubsetEmbeddingBackward.backward

    @staticmethod
    def timed_emb_backward(ctx, grad_output):
        from drpt.selection.utils import (
            compute_embedding_scores,
            compute_embedding_val_gradient,
            split_train_val_batch,
        )

        input_ids, weight = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            return None, None, None, None, None

        state = hook_manager.selection_state
        if state is None:
            return None, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        with torch.no_grad():
            rec.mark('emb_score')
            if use_stored_val:
                train_go, train_ids = grad_output, input_ids
                val_grad_weight = hook_manager.val_cache.get_full(layer_idx)
                if val_grad_weight is None:
                    rec.mark('emb_score')
                    return None, None, None, None, None
                score_correction = None
            else:
                train_go, val_go = split_train_val_batch(grad_output, state.train_batch_size)
                train_ids, val_ids = split_train_val_batch(input_ids, state.train_batch_size)
                V, D = weight.shape
                val_grad_weight = compute_embedding_val_gradient(val_go, val_ids, V, D)
                score_correction = state.score_correction

            scores = compute_embedding_scores(train_go, train_ids, val_grad_weight)
            if score_correction is not None:
                scores = scores * score_correction
            state.accumulate_precomputed_scores(scores, None, None)
            rec.mark('emb_score')

        return None, None, None, None, None

    _bwd.SubsetEmbeddingBackward.backward = timed_emb_backward

    # Disable score compressors unless scoring_method is "compress"
    scoring_method = getattr(config, 'scoring_method', 'reduced_ghost')
    saved_score_compressors = grad_hook.score_compressors
    if scoring_method != "compress":
        grad_hook.score_compressors = [None] * len(saved_score_compressors)

    def step(batch, val_batch, i=None):
        train_bs = batch['input_ids'].shape[0]
        merged = pad_and_merge_batches(batch, val_batch, pad_token_id=pad_token_id)

        # Pass 1
        grad_hook.setup_selection(
            train_batch_size=train_bs, selection_method="Subset",
            frac=0.5, lr=optimizer.param_groups[0].get("lr", 5e-5),
            selection_mode="topk", use_second_order=config.use_second_order,
            scoring_method=scoring_method,
            direct_batch_size=getattr(config, 'direct_batch_size', 0),
        )
        if 'labels' in merged:
            grad_hook.set_token_counts(merged['labels'], train_bs)
        optimizer.zero_grad()
        if i is not None: timer.mark("pass1_forward", i, True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(**merged).loss
        if i is not None: timer.mark("pass1_forward", i, False)
        if i is not None: timer.mark("pass1_backward", i, True)
        loss.backward()
        if i is not None: timer.mark("pass1_backward", i, False)

        if i is not None: timer.mark("selection", i, True)
        selected = grad_hook.selection_state.get_final_selection().sort()[0]
        if i is not None: timer.mark("selection", i, False)
        grad_hook.clear_selection()
        grad_hook.clear_token_counts()

        # Pass 2
        if len(selected) == 0:
            if i is not None:
                for p in ["pass2_forward", "pass2_backward", "optimizer"]:
                    timer.mark(p, i, True); timer.mark(p, i, False)
            return
        filtered = {k: batch[k][selected] for k in ['input_ids', 'attention_mask', 'labels']}
        has_update = grad_hook.compression_mode.uses_compressed_updates
        if not has_update:
            grad_hook.disable_hooks()
            # Apply TimedLinearP2 patches for pass 2 breakdown
            for mod, orig, timed in _p2_patches:
                mod._original_forward = timed
        optimizer.zero_grad()
        if i is not None: timer.mark("pass2_forward", i, True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss2 = model(**filtered).loss
        if i is not None: timer.mark("pass2_forward", i, False)
        if i is not None: timer.mark("pass2_backward", i, True)
        loss2.backward()
        if i is not None: timer.mark("pass2_backward", i, False)
        if not has_update:
            # Restore original forwards
            for mod, orig, timed in _p2_patches:
                mod._original_forward = orig
            grad_hook.enable_hooks()
        if i is not None: timer.mark("optimizer", i, True)
        optimizer.step()
        if i is not None: timer.mark("optimizer", i, False)

    # Warmup + timed (need to handle both rec and p2_rec)
    N = config.num_iterations
    p1_accum = {}
    p2_accum = {}

    for i in range(config.num_warmup):
        rec.reset(); p2_rec.reset()
        step(train_batches[i % len(train_batches)],
             val_batches[i % len(val_batches)])

    for i in range(N):
        rec.reset(); p2_rec.reset()
        step(train_batches[(config.num_warmup + i) % len(train_batches)],
             val_batches[(config.num_warmup + i) % len(val_batches)], i)
        torch.cuda.synchronize()
        rec.accumulate(p1_accum)
        p2_rec.accumulate(p2_accum)

    _bwd.SubsetLinearBackward.backward = _orig_backward
    _bwd.SubsetLinearBackward._accumulate_compressed = _orig_accum
    _bwd.SubsetLinearBackward._accumulate_full = _orig_accum_full
    _bwd.SubsetEmbeddingBackward.backward = _orig_emb_backward
    grad_hook.score_compressors = saved_score_compressors

    result = timer.mean_elapsed()
    for k, v in p1_accum.items():
        result[f"p1_{k}"] = v / N
    for k, v in p2_accum.items():
        result[k] = v / N
    return result


# =============================================================================
# Measurement: Subset One-Pass
# =============================================================================

def measure_subset_one_pass(model, optimizer, grad_hook, train_batches, val_batches, config, tokenizer):
    """Measure one-pass Subset (Algorithm 4.2): score + retain during backward, post-hoc assembly."""
    import drpt.selection.backward as _bwd
    from drpt.selection.backward import augment_input_for_bias, split_train_val_batch
    from drpt.selection.state import SubsetState

    N = config.num_iterations
    phases = ["forward", "backward", "selection", "wgrad", "optimizer"]
    timer = CUDAEventTimer(phases, N)
    rec = EventRecorder()
    pad_token_id = tokenizer.pad_token_id or 0

    scoring_method = getattr(config, 'scoring_method', 'reduced_ghost')

    # Save originals
    _orig_backward = _bwd.SubsetLinearBackward.backward
    _orig_accum_full = _bwd.SubsetLinearBackward._accumulate_full
    _orig_accum_compressed = _bwd.SubsetLinearBackward._accumulate_compressed

    # Patched backward: times act_grad + retain
    @staticmethod
    def timed_backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx
        hm = ctx.hook_manager_ref()
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)
        rec.mark('act_grad')
        grad_input = grad_output @ weight.to(grad_output.dtype)
        rec.mark('act_grad')
        sc = hm.score_compressors[layer_idx]
        state = hm.selection_state
        if state is None:
            return grad_input, None, None, None, None
        usv = getattr(state, '_use_stored_val', False)
        with torch.no_grad():
            if sc is not None:
                _bwd.SubsetLinearBackward._accumulate_compressed(
                    hm, sc, state, layer_idx, input, grad_output, bias, usv)
            else:
                _bwd.SubsetLinearBackward._accumulate_full(
                    hm, state, layer_idx, input, grad_output, bias, usv)
            # One-pass: retain data
            if state.one_pass:
                rec.mark('retain')
                if usv:
                    hm.retain_layer_data(layer_idx, grad_output, input)
                else:
                    tgo, _ = split_train_val_batch(grad_output, state.train_batch_size)
                    tinp, _ = split_train_val_batch(input, state.train_batch_size)
                    hm.retain_layer_data(layer_idx, tgo, tinp)
                rec.mark('retain')
        return grad_input, None, None, None, None

    # Patched _accumulate_full: times score
    @staticmethod
    def timed_accum_full(hm, state, lidx, inp, go, bias, usv):
        rec.mark('score')
        _orig_accum_full(hm, state, lidx, inp, go, bias, usv)
        rec.mark('score')

    # Patched _accumulate_compressed: times compress + score
    @staticmethod
    def timed_accum_compressed(hm, compressor, state, lidx, inp, go, bias, usv):
        ia = augment_input_for_bias(inp, bias is not None)
        rec.mark('compress'); cg = compressor.forward((go, ia)); rec.mark('compress')
        rec.mark('score')
        if usv:
            tg, vg, scorr = cg, hm._val_cache.get_compressed(lidx), None
        else:
            tg, vgs = split_train_val_batch(cg, state.train_batch_size)
            vg = vgs.sum(dim=0); scorr = state.score_correction
        if vg is not None:
            state.process_layer_gradients(tg, vg, lidx, scorr)
        rec.mark('score')

    _bwd.SubsetLinearBackward.backward = timed_backward
    _bwd.SubsetLinearBackward._accumulate_full = timed_accum_full
    _bwd.SubsetLinearBackward._accumulate_compressed = timed_accum_compressed

    # --- Embedding timing patch (one-pass: score + retain) ---
    _orig_emb_backward = _bwd.SubsetEmbeddingBackward.backward

    @staticmethod
    def timed_emb_backward(ctx, grad_output):
        from drpt.selection.utils import (
            compute_embedding_scores,
            compute_embedding_val_gradient,
            split_train_val_batch,
        )

        input_ids, weight = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            return None, None, None, None, None

        state = hook_manager.selection_state
        if state is None:
            return None, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        with torch.no_grad():
            rec.mark('emb_score')
            if use_stored_val:
                train_go, train_ids = grad_output, input_ids
                val_grad_weight = hook_manager.val_cache.get_full(layer_idx)
                if val_grad_weight is None:
                    rec.mark('emb_score')
                    if state.one_pass:
                        rec.mark('emb_retain')
                        hook_manager.retain_layer_data(layer_idx, grad_output, input_ids)
                        rec.mark('emb_retain')
                    return None, None, None, None, None
                score_correction = None
            else:
                train_go, val_go = split_train_val_batch(grad_output, state.train_batch_size)
                train_ids, val_ids = split_train_val_batch(input_ids, state.train_batch_size)
                V, D = weight.shape
                val_grad_weight = compute_embedding_val_gradient(val_go, val_ids, V, D)
                score_correction = state.score_correction

            scores = compute_embedding_scores(train_go, train_ids, val_grad_weight)
            if score_correction is not None:
                scores = scores * score_correction
            state.accumulate_precomputed_scores(scores, None, None)
            rec.mark('emb_score')

            if state.one_pass:
                rec.mark('emb_retain')
                if use_stored_val:
                    hook_manager.retain_layer_data(layer_idx, grad_output, input_ids)
                else:
                    hook_manager.retain_layer_data(layer_idx, train_go, train_ids)
                rec.mark('emb_retain')

        return None, None, None, None, None

    _bwd.SubsetEmbeddingBackward.backward = timed_emb_backward

    # Disable score compressors for one-pass — use exact scoring
    saved_score_compressors = grad_hook.score_compressors
    if scoring_method != "compress":
        grad_hook.score_compressors = [None] * len(saved_score_compressors)

    def step(batch, val_batch, i=None):
        train_bs = batch['input_ids'].shape[0]
        merged = pad_and_merge_batches(batch, val_batch, pad_token_id=pad_token_id)

        grad_hook.setup_selection(
            train_batch_size=train_bs, selection_method="Subset",
            frac=0.5, lr=optimizer.param_groups[0].get("lr", 5e-5),
            selection_mode="topk", use_second_order=config.use_second_order,
            scoring_method=scoring_method,
            one_pass=True,
            direct_batch_size=getattr(config, 'direct_batch_size', 0),
        )
        if 'labels' in merged:
            grad_hook.set_token_counts(merged['labels'], train_bs)

        optimizer.zero_grad()
        if i is not None: timer.mark("forward", i, True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss = model(**merged).loss
        if i is not None: timer.mark("forward", i, False)

        if i is not None: timer.mark("backward", i, True)
        loss.backward()
        if i is not None: timer.mark("backward", i, False)

        if i is not None: timer.mark("selection", i, True)
        state = grad_hook.selection_state
        selected = state.get_final_selection().sort()[0]
        if i is not None: timer.mark("selection", i, False)

        if i is not None: timer.mark("wgrad", i, True)
        if len(selected) > 0:
            scale_factor = state._compute_scale_factor_for_assembly(selected)
            # No model.zero_grad() here — non-linear layers retain their autograd
            # gradients from backward, and hooked linear layers have None grad
            # (SubsetLinearBackward suppresses weight/bias grads).
            grad_hook.assemble_gradients_from_retained(selected, scale_factor)
        else:
            grad_hook.clear_retained_data()
        if i is not None: timer.mark("wgrad", i, False)

        grad_hook.clear_selection()
        grad_hook.clear_token_counts()

        if i is not None: timer.mark("optimizer", i, True)
        optimizer.step()
        if i is not None: timer.mark("optimizer", i, False)

    # Warmup + timed
    comp_accum = {}
    for j in range(config.num_warmup):
        rec.reset()
        step(train_batches[j % len(train_batches)],
             val_batches[j % len(val_batches)])

    for j in range(N):
        rec.reset()
        step(train_batches[(config.num_warmup + j) % len(train_batches)],
             val_batches[(config.num_warmup + j) % len(val_batches)], j)
        torch.cuda.synchronize()
        rec.accumulate(comp_accum)

    _bwd.SubsetLinearBackward.backward = _orig_backward
    _bwd.SubsetLinearBackward._accumulate_full = _orig_accum_full
    _bwd.SubsetLinearBackward._accumulate_compressed = _orig_accum_compressed
    _bwd.SubsetEmbeddingBackward.backward = _orig_emb_backward
    grad_hook.score_compressors = saved_score_compressors

    result = timer.mean_elapsed()
    for k, v in comp_accum.items():
        result[k] = v / N
    return result


# =============================================================================
# Display
# =============================================================================

def print_results(standard, layerwise, subset, config, peak_mem, subset_one_pass=None):
    has_op = subset_one_pass is not None
    W = 130 if has_op else 110
    print()
    print("=" * W)
    print(f"BENCHMARK RESULTS (ms per step, averaged over {config.num_iterations} iterations)")
    print(f"Model: {config.model_name} | Batch: {config.batch_size} | "
          f"Seq: {config.seq_length} | Dtype: {config.dtype}")
    scoring = getattr(config, 'scoring_method', 'reduced_ghost')
    print(f"Dataset: {config.dataset} → {config.val_dataset} | "
          f"Score compression: {config.score_compression} | Scoring: {scoring}")
    print("=" * W)

    def _v(v):
        return "—" if v < 0.01 else f"{v:.1f}"

    if has_op:
        def _row(label, s, l, sub, op):
            print(f"  {label:<30} {_v(s):>12} {_v(l):>12} {_v(sub):>12} {_v(op):>12}")
        print(f"  {'Component':<30} {'Standard':>12} {'Layerwise':>12} {'Subset 2P':>12} {'Subset 1P':>12}")
    else:
        def _row(label, s, l, sub, op=0):
            print(f"  {label:<30} {_v(s):>12} {_v(l):>12} {_v(sub):>12}")
        print(f"  {'Component':<30} {'Standard':>12} {'Layerwise':>12} {'Subset':>12}")
    print("-" * W)

    s_fwd, s_bwd, s_opt = standard['forward'], standard['backward'], standard['optimizer']
    l_fwd, l_bwd, l_opt = layerwise['forward'], layerwise['backward'], layerwise['optimizer']
    sub_p1f = subset['pass1_forward']
    sub_p1b = subset['pass1_backward']
    sub_sel = subset['selection']
    sub_p2f = subset['pass2_forward']
    sub_p2b = subset['pass2_backward']
    sub_opt = subset['optimizer']
    s_total = s_fwd + s_bwd + s_opt
    l_total = l_fwd + l_bwd + l_opt
    sub_total = sub_p1f + sub_p1b + sub_sel + sub_p2f + sub_p2b + sub_opt

    if has_op:
        op_fwd = subset_one_pass['forward']
        op_bwd = subset_one_pass['backward']
        op_sel = subset_one_pass['selection']
        op_asm = subset_one_pass['wgrad']
        op_opt = subset_one_pass['optimizer']
        op_total = op_fwd + op_bwd + op_sel + op_asm + op_opt
    else:
        op_fwd = op_bwd = op_sel = op_asm = op_opt = op_total = 0

    _row("Forward", s_fwd, l_fwd, sub_p1f, op_fwd)
    _row("Backward (score+retain)", s_bwd, l_bwd, sub_p1b + sub_sel, op_bwd + op_sel)
    _row("Pass-2 Fwd+Bwd / Assembly", 0, 0, sub_p2f + sub_p2b, op_asm)
    _row("Optimizer", s_opt, l_opt, sub_opt, op_opt)
    print("-" * W)

    totals = f"  {'TOTAL':<30} {s_total:>11.1f}  {l_total:>11.1f}  {sub_total:>11.1f}"
    if has_op: totals += f"  {op_total:>11.1f}"
    print(totals)

    mem_line = (f"  {'Peak Memory':<30} {peak_mem.get('standard',0):>10.2f} GB"
                f" {peak_mem.get('layerwise',0):>10.2f} GB"
                f" {peak_mem.get('subset',0):>10.2f} GB")
    if has_op: mem_line += f" {peak_mem.get('subset_one_pass',0):>10.2f} GB"
    print(mem_line)

    ovh = f"  {'Overhead vs Standard':<30} {'':>12} {(l_total/s_total-1)*100:>10.1f}%  {(sub_total/s_total-1)*100:>10.1f}%"
    if has_op: ovh += f"  {(op_total/s_total-1)*100:>10.1f}%"
    print(ovh)

    if has_op:
        savings = (1 - op_total/sub_total) * 100 if sub_total > 0 else 0
        print(f"  {'1P savings vs 2P':<30} {'':>38} {savings:>10.1f}%")

    # Detailed breakdowns
    print()
    print("-" * W)
    print("BACKWARD BREAKDOWNS (directly measured, ms per step)")
    print("-" * W)

    def _bkd(label, items, total):
        print(f"  {label}:")
        for name, val in items:
            if name.startswith("---"):
                print(f"    {name}")
            else:
                print(f"    {name:<33} {val:>8.1f} ms")
        print(f"    {'total backward':<33} {total:>8.1f} ms")

    # Standard
    s_act = standard.get('act_grad', 0)
    s_wg = standard.get('wgrad', 0)
    s_items = []
    if s_act > 0.01:
        s_items = [("act_grad (chain rule)", s_act), ("w.grad", s_wg),
                   ("autograd overhead", s_bwd - s_act - s_wg)]
    _bkd("Standard", s_items, s_bwd)

    # Layerwise (fold embedding into corresponding phases)
    l_act = layerwise.get('act_grad', 0)
    l_comp = layerwise.get('compress', 0)
    l_score = layerwise.get('score', 0) + layerwise.get('emb_score', 0)
    l_sel = layerwise.get('select', 0) + layerwise.get('emb_select', 0)
    l_wg = layerwise.get('wgrad', 0) + layerwise.get('emb_wgrad', 0)
    l_sw = layerwise.get('select_wgrad', 0)
    l_measured = l_act + l_comp + l_score + l_sel + l_wg + l_sw
    l_items = [("act_grad (chain rule)", l_act),
               ("compress (sparsifier.forward)", l_comp),
               ("score (matmul + correction)", l_score)]
    if l_sw > 0.01:
        l_items.append(("select + w.grad (shared compr.)", l_sw))
    else:
        l_items += [("select (top-k)", l_sel), ("w.grad (selected gradients)", l_wg)]
    l_items.append(("autograd overhead", l_bwd - l_measured))
    _bkd("Layerwise", l_items, l_bwd)

    # Subset two-pass (fold embedding into score)
    p1_act = subset.get('p1_act_grad', 0)
    p1_comp = subset.get('p1_compress', 0)
    p1_score = subset.get('p1_score', 0) + subset.get('p1_emb_score', 0)
    p1_measured = p1_act + p1_comp + p1_score
    p1_total = sub_p1b + sub_sel
    p1_items = [("act_grad (chain rule)", p1_act),
                ("compress (sparsifier.forward)", p1_comp),
                ("score (accumulate)", p1_score),
                ("selection (top-k)", sub_sel),
                ("autograd overhead", sub_p1b - p1_measured)]
    _bkd("Subset two-pass: pass 1 (scoring)", p1_items, p1_total)
    p2_act = subset.get('p2_act_grad', 0)
    p2_wg = subset.get('p2_wgrad', 0)
    p2_measured = p2_act + p2_wg
    p2_overhead = sub_p2b - p2_measured if p2_measured > 0.01 else 0
    print(f"  Subset two-pass: pass 2 (w.grad on selected):")
    print(f"    {'forward':<33} {sub_p2f:>8.1f} ms")
    if p2_act > 0.01:
        print(f"    {'act_grad (chain rule)':<33} {p2_act:>8.1f} ms")
        print(f"    {'w.grad':<33} {p2_wg:>8.1f} ms")
        print(f"    {'autograd overhead':<33} {p2_overhead:>8.1f} ms")
    print(f"    {'backward':<33} {sub_p2b:>8.1f} ms")

    # Subset one-pass (fold embedding into score/retain)
    if has_op:
        op_act = subset_one_pass.get('act_grad', 0)
        op_comp = subset_one_pass.get('compress', 0)
        op_score = subset_one_pass.get('score', 0) + subset_one_pass.get('emb_score', 0)
        op_retain = subset_one_pass.get('retain', 0) + subset_one_pass.get('emb_retain', 0)
        op_measured = op_act + op_comp + op_score + op_retain
        op_items = [("act_grad (chain rule)", op_act),
                    ("compress (sparsifier.forward)", op_comp),
                    ("score (accumulate)", op_score),
                    ("retain (save per-layer data)", op_retain),
                    ("selection (top-k)", op_sel),
                    ("assembly (w.grad from retained)", op_asm),
                    ("autograd overhead", op_bwd - op_measured)]
        _bkd("Subset one-pass (score+retain+assemble)", op_items,
             op_bwd + op_sel + op_asm)

    print("=" * W)


# =============================================================================
# Runner & CLI
# =============================================================================

def run_method(method, config):
    set_seed(config.seed)
    total_needed = config.num_warmup + config.num_iterations

    model, tokenizer = setup_model(config)
    grad_hook = setup_grad_hook(model, config, tokenizer, config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    train_loader, val_loader = create_dataloaders(config, tokenizer)
    train_batches, val_batches = get_batches(train_loader, val_loader, total_needed, config.device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if method == "standard":
        result = measure_standard(model, optimizer, grad_hook, train_batches, config)
    elif method == "layerwise":
        result = measure_layerwise(model, optimizer, grad_hook,
                                   train_batches, val_batches, config, tokenizer)
    elif method == "subset":
        result = measure_subset(model, optimizer, grad_hook,
                                train_batches, val_batches, config, tokenizer)
    elif method == "subset_one_pass":
        result = measure_subset_one_pass(model, optimizer, grad_hook,
                                         train_batches, val_batches, config, tokenizer)
    else:
        raise ValueError(f"Unknown method: {method}")

    result["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    return result


def _build_config(args):
    kwargs = {}
    for attr, arg in [('model_name','model'), ('dtype','dtype'), ('batch_size','batch_size'),
                      ('seq_length','seq_length'), ('val_batch_size','val_batch_size'),
                      ('dataset','dataset'), ('val_dataset','val_dataset'),
                      ('num_warmup','num_warmup'), ('num_iterations','num_iterations'),
                      ('seed','seed'), ('score_compression','score_compression'),
                      ('scoring_method','scoring_method'),
                      ('direct_batch_size','direct_batch_size')]:
        val = getattr(args, arg, None)
        if val is not None:
            kwargs[attr] = val
    if getattr(args, 'no_flash_attention', False):
        kwargs['use_flash_attention'] = False
    if getattr(args, 'use_second_order', False):
        kwargs['use_second_order'] = True
    if getattr(args, 'gradient_checkpointing', False):
        kwargs['gradient_checkpointing'] = True
    return BenchmarkConfig(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='Timing Benchmark for Dr. Post-Training')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--dtype', type=str, default=None, choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--no-flash-attention', action='store_true')
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--seq-length', type=int, default=None)
    parser.add_argument('--val-batch-size', type=int, default=None)
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['dummy', 'alpaca', 'gsm8k', 'dolly', 'tulu3'])
    parser.add_argument('--val-dataset', type=str, default=None,
                        choices=['samsum', 'gsm8k', 'tydiqa', 'mmlu', 'bbh', 'math500'])
    parser.add_argument('--score-compression', type=str, default=None)
    parser.add_argument('--scoring-method', type=str, default=None,
                        choices=['reduced_ghost', 'full_ghost', 'direct', 'compress'])
    parser.add_argument('--direct-batch-size', type=int, default=None,
                        help='Chunk size for batched direct scoring. 0=all at once, 1=min memory.')
    parser.add_argument('--num-warmup', type=int, default=None)
    parser.add_argument('--num-iterations', type=int, default=None)
    parser.add_argument('--use-second-order', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient (activation) checkpointing to trade compute for memory.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--method', type=str, default=None, choices=METHODS)

    args = parser.parse_args()
    config = _build_config(args)

    if args.method is not None:
        result = run_method(args.method, config)
        print(f"Peak memory: {result['peak_memory_gb']:.2f} GB")
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        return

    # Run all methods in subprocesses
    print(f"Config: {config}\n")

    script = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(script), "..", ".."))
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root + ":" + env.get("PYTHONPATH", "")

    # Forward config as CLI args
    cli_args = []
    d = BenchmarkConfig()
    for attr, flag in [('model_name','--model'), ('dtype','--dtype'), ('batch_size','--batch-size'),
                       ('seq_length','--seq-length'), ('val_batch_size','--val-batch-size'),
                       ('dataset','--dataset'), ('val_dataset','--val-dataset'),
                       ('score_compression','--score-compression'), ('scoring_method','--scoring-method'),
                       ('num_warmup','--num-warmup'),
                       ('num_iterations','--num-iterations'), ('seed','--seed')]:
        val = getattr(config, attr)
        if val != getattr(d, attr):
            cli_args += [flag, str(val)]
    if not config.use_flash_attention:
        cli_args += ["--no-flash-attention"]
    if config.use_second_order:
        cli_args += ["--use-second-order"]
    if config.gradient_checkpointing:
        cli_args += ["--gradient-checkpointing"]

    all_results = {}
    peak_mem = {}

    for method in METHODS:
        print(f"{'='*60}\n  Running: {method}\n{'='*60}")
        tmp = tempfile.mktemp(suffix='.json')
        cmd = [sys.executable, script, "--method", method, "--output", tmp] + cli_args
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            print(f"  FAILED (exit code {proc.returncode})")
            continue
        with open(tmp) as f:
            result = json.load(f)
        os.remove(tmp)
        all_results[method] = result
        peak_mem[method] = result.get("peak_memory_gb", 0)
        print(f"  Done. Peak memory: {peak_mem[method]:.2f} GB")

    required = ["standard", "layerwise", "subset"]
    if all(m in all_results for m in required):
        print_results(
            all_results["standard"], all_results["layerwise"],
            all_results["subset"], config, peak_mem,
            subset_one_pass=all_results.get("subset_one_pass"),
        )
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"config": asdict(config), "results": all_results}, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
