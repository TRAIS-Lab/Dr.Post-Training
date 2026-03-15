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

METHODS = ["standard", "layerwise", "subset"]


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
            rec.mark('select_wgrad')
            rg, _ = state.process_layer_gradients(tg2, vg, lidx, scorr)
            hm._store_compressed_grad(lidx, rg)
            rec.mark('select_wgrad')
            return None, None
        rec.mark('select'); si = _do_selection(state, lidx, scores, sim); rec.mark('select')
        rec.mark('wgrad')
        if usv: tgo, ti = go, inp
        else:
            tgo, _ = split_train_val_batch(go, state.train_batch_size)
            ti, _ = split_train_val_batch(inp, state.train_batch_size)
        result = _produce_gradient_update(hm, uc, state, lidx, tgo, ti, si, hb)
        rec.mark('wgrad')
        return result

    _bwd.LayerwiseLinearBackward.backward = timed_backward
    _bwd.LayerwiseLinearBackward._backward_compressed = timed_compressed

    def step(batch, val_batch, i=None):
        train_bs = batch['input_ids'].shape[0]
        merged = pad_and_merge_batches(batch, val_batch, pad_token_id=pad_token_id)
        grad_hook.setup_selection(
            train_batch_size=train_bs, selection_method="Layerwise",
            frac=0.5, lr=optimizer.param_groups[0].get("lr", 5e-5),
            selection_mode="topk", use_second_order=config.use_second_order,
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
    rec = EventRecorder()
    pad_token_id = tokenizer.pad_token_id or 0

    # Save originals
    _orig_backward = _bwd.SubsetLinearBackward.backward
    _orig_accum = _bwd.SubsetLinearBackward._accumulate_compressed

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
                    hm, state, layer_idx, input, grad_output, usv)
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

    _bwd.SubsetLinearBackward.backward = timed_backward
    _bwd.SubsetLinearBackward._accumulate_compressed = timed_accum

    def step(batch, val_batch, i=None):
        train_bs = batch['input_ids'].shape[0]
        merged = pad_and_merge_batches(batch, val_batch, pad_token_id=pad_token_id)

        # Pass 1
        grad_hook.setup_selection(
            train_batch_size=train_bs, selection_method="Subset",
            frac=0.5, lr=optimizer.param_groups[0].get("lr", 5e-5),
            selection_mode="topk", use_second_order=config.use_second_order,
            scoring_method="ghost",
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
        optimizer.zero_grad()
        if i is not None: timer.mark("pass2_forward", i, True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            loss2 = model(**filtered).loss
        if i is not None: timer.mark("pass2_forward", i, False)
        if i is not None: timer.mark("pass2_backward", i, True)
        loss2.backward()
        if i is not None: timer.mark("pass2_backward", i, False)
        if not has_update:
            grad_hook.enable_hooks()
        if i is not None: timer.mark("optimizer", i, True)
        optimizer.step()
        if i is not None: timer.mark("optimizer", i, False)

    comp = _warmup_and_measure(step, train_batches, val_batches, config, rec)

    _bwd.SubsetLinearBackward.backward = _orig_backward
    _bwd.SubsetLinearBackward._accumulate_compressed = _orig_accum

    result = timer.mean_elapsed()
    for k, v in comp.items():
        result[f"p1_{k}"] = v
    return result


# =============================================================================
# Display
# =============================================================================

def print_results(standard, layerwise, subset, config, peak_mem):
    W = 110
    print()
    print("=" * W)
    print(f"BENCHMARK RESULTS (ms per step, averaged over {config.num_iterations} iterations)")
    print(f"Model: {config.model_name} | Batch: {config.batch_size} | "
          f"Seq: {config.seq_length} | Dtype: {config.dtype}")
    print(f"Dataset: {config.dataset} → {config.val_dataset} | "
          f"Score compression: {config.score_compression}")
    print("=" * W)

    def _v(v):
        return "—" if v < 0.01 else f"{v:.1f}"

    def _row(label, s, l, sub):
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

    _row("Forward", s_fwd, l_fwd, sub_p1f)
    _row("Backward", s_bwd, l_bwd, sub_p1b)
    _row("Selection", 0, 0, sub_sel)
    _row("Pass-2 Forward", 0, 0, sub_p2f)
    _row("Pass-2 Backward", 0, 0, sub_p2b)
    _row("Optimizer", s_opt, l_opt, sub_opt)
    print("-" * W)
    print(f"  {'TOTAL':<30} {s_total:>11.1f}  {l_total:>11.1f}  {sub_total:>11.1f}")
    print(f"  {'Peak Memory':<30} {peak_mem.get('standard',0):>10.2f} GB"
          f" {peak_mem.get('layerwise',0):>10.2f} GB"
          f" {peak_mem.get('subset',0):>10.2f} GB")
    print(f"  {'Overhead vs Standard':<30} {'':>12}"
          f" {(l_total/s_total-1)*100:>10.1f}%  {(sub_total/s_total-1)*100:>10.1f}%")

    # Detailed breakdowns
    print()
    print("-" * W)
    print("BACKWARD BREAKDOWNS (directly measured, ms per step)")
    print("-" * W)

    def _bkd(label, items, total):
        print(f"  {label}:")
        for name, val in items:
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

    # Layerwise
    l_act = layerwise.get('act_grad', 0)
    l_comp = layerwise.get('compress', 0)
    l_score = layerwise.get('score', 0)
    l_sel = layerwise.get('select', 0)
    l_wg = layerwise.get('wgrad', 0)
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

    # Subset
    p1_act = subset.get('p1_act_grad', 0)
    p1_comp = subset.get('p1_compress', 0)
    p1_score = subset.get('p1_score', 0)
    p1_measured = p1_act + p1_comp + p1_score
    p1_items = [("act_grad (chain rule)", p1_act),
                ("compress (sparsifier.forward)", p1_comp),
                ("score (accumulate)", p1_score),
                ("autograd overhead", sub_p1b - p1_measured)]
    _bkd("Subset pass 1 (scoring)", p1_items, sub_p1b)
    print(f"  Subset pass 2 (w.grad on selected):")
    print(f"    {'forward':<33} {sub_p2f:>8.1f} ms")
    print(f"    {'backward':<33} {sub_p2b:>8.1f} ms")

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
                      ('seed','seed'), ('score_compression','score_compression')]:
        val = getattr(args, arg, None)
        if val is not None:
            kwargs[attr] = val
    if getattr(args, 'no_flash_attention', False):
        kwargs['use_flash_attention'] = False
    if getattr(args, 'use_second_order', False):
        kwargs['use_second_order'] = True
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
    parser.add_argument('--num-warmup', type=int, default=None)
    parser.add_argument('--num-iterations', type=int, default=None)
    parser.add_argument('--use-second-order', action='store_true')
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
                       ('score_compression','--score-compression'), ('num_warmup','--num-warmup'),
                       ('num_iterations','--num-iterations'), ('seed','--seed')]:
        val = getattr(config, attr)
        if val != getattr(d, attr):
            cli_args += [flag, str(val)]
    if not config.use_flash_attention:
        cli_args += ["--no-flash-attention"]
    if config.use_second_order:
        cli_args += ["--use-second-order"]

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

    if all(m in all_results for m in METHODS):
        print_results(all_results["standard"], all_results["layerwise"],
                      all_results["subset"], config, peak_mem)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({"config": asdict(config), "results": all_results}, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
