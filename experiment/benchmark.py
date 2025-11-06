#!/usr/bin/env python
"""
Max Memory Benchmark

Tracks torch.cuda.max_memory_allocated() throughout entire test
WITHOUT resetting between iterations. This shows the true peak
memory ever reached during training.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import sys
import json
import argparse
import os
import time
import warnings

# Suppress dynamo warnings about custom CUDA kernels (sjlt)
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

from core.train.compress_gradient.hook import GradientHook
from core.train.compress_gradient.compressor import setup_model_compressors
from core.train.compress_gradient.optimizer import MeSOAdamW

from GaLore.galore_torch import GaLoreAdamW

device = 'cuda'
batch_size = 10
seq_length = 512  # Back to 512 to see the real bottleneck
num_iterations = 10

def get_mem():
    return torch.cuda.memory_allocated() / 1024**3

def get_max_mem():
    return torch.cuda.max_memory_allocated() / 1024**3

def reset():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def run_method(method_name, setup_fn):
    """Run a method for multiple iterations, tracking max memory WITHOUT resetting"""
    print("="*80)
    print(f"Testing: {method_name}")
    print("="*80)

    reset()

    # Setup model and optimizer
    print("Setting up model and optimizer...")
    torch.cuda.synchronize()
    setup_start = time.time()
    model, optimizer, tokenizer = setup_fn()
    torch.cuda.synchronize()
    setup_time = time.time() - setup_start
    print(f"Setup time: {setup_time:.2f}s")

    mem_after_setup = get_mem()
    max_after_setup = get_max_mem()
    print(f"After setup - Current: {mem_after_setup:.3f} GB, Max so far: {max_after_setup:.3f} GB")

    # Track memory per iteration and timing
    iteration_stats = []

    # Start timing for training iterations
    torch.cuda.synchronize()
    total_start = time.time()

    for i in range(num_iterations):
        print(f"\n--- Iteration {i+1}/{num_iterations} ---")

        # Create batch
        batch = tokenizer(
            ["This is a test sentence for memory profiling."] * batch_size,
            return_tensors='pt',
            padding='max_length',
            max_length=seq_length,
            truncation=True
        ).to(device)
        batch['labels'] = batch['input_ids'].clone()

        mem_before_forward = get_mem()
        max_before_forward = get_max_mem()

        # Forward
        torch.cuda.synchronize()
        forward_start = time.time()
        outputs = model(**batch)
        loss = outputs.loss
        torch.cuda.synchronize()
        forward_time = time.time() - forward_start

        mem_after_forward = get_mem()
        max_after_forward = get_max_mem()
        print(f"  Forward:   Current = {mem_after_forward:6.3f} GB, Max = {max_after_forward:6.3f} GB (peak: +{max_after_forward - max_before_forward:.3f} GB, time: {forward_time:.3f}s)")

        # Backward
        torch.cuda.synchronize()
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - backward_start

        mem_after_backward = get_mem()
        max_after_backward = get_max_mem()
        print(f"  Backward:  Current = {mem_after_backward:6.3f} GB, Max = {max_after_backward:6.3f} GB (peak: +{max_after_backward - max_after_forward:.3f} GB, time: {backward_time:.3f}s)")

        # Optimizer step
        torch.cuda.synchronize()
        optimizer_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        optimizer_time = time.time() - optimizer_start

        mem_after_optimizer = get_mem()
        max_after_optimizer = get_max_mem()
        print(f"  Optimizer: Current = {mem_after_optimizer:6.3f} GB, Max = {max_after_optimizer:6.3f} GB (peak: +{max_after_optimizer - max_after_backward:.3f} GB, time: {optimizer_time:.3f}s)")

        iteration_stats.append({
            'iteration': i + 1,
            'current_after_optimizer': mem_after_optimizer,
            'max_ever': max_after_optimizer,
            'max_during_forward': max_after_forward - max_before_forward,
            'max_during_backward': max_after_backward - max_after_forward,
            'max_during_optimizer': max_after_optimizer - max_after_backward,
            'forward_time': forward_time,
            'backward_time': backward_time,
            'optimizer_time': optimizer_time,
        })

    # Total training time
    torch.cuda.synchronize()
    total_time = time.time() - total_start

    # Final summary
    absolute_max = get_max_mem()
    final_current = get_mem()

    print(f"\n{'='*80}")
    print(f"{method_name} - FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Absolute maximum memory EVER reached: {absolute_max:.3f} GB")
    print(f"Current memory at end:                {final_current:.3f} GB")
    print(f"Peak occurred during:                 ", end="")

    # Find when peak occurred
    max_forward = max(s['max_during_forward'] for s in iteration_stats)
    max_backward = max(s['max_during_backward'] for s in iteration_stats)
    max_optimizer = max(s['max_during_optimizer'] for s in iteration_stats)

    if max_forward > max_backward and max_forward > max_optimizer:
        print("FORWARD pass")
    elif max_backward > max_optimizer:
        print("BACKWARD pass")
    else:
        print("OPTIMIZER step")

    # Check memory growth
    first_iter_max = iteration_stats[0]['max_ever']
    last_iter_max = iteration_stats[-1]['max_ever']
    growth = last_iter_max - first_iter_max

    print(f"\nMemory growth from iter 1 to iter {num_iterations}:")
    print(f"  Iter 1 max:  {first_iter_max:.3f} GB")
    print(f"  Iter {num_iterations} max:  {last_iter_max:.3f} GB")
    print(f"  Growth:      {growth:+.3f} GB")

    if abs(growth) < 0.1:
        print(f"  Status:      ✓ STABLE")
    elif growth > 0:
        print(f"  Status:      ⚠ GROWING")
    else:
        print(f"  Status:      ✓ DECREASING")

    # Timing statistics
    avg_forward = sum(s['forward_time'] for s in iteration_stats) / len(iteration_stats)
    avg_backward = sum(s['backward_time'] for s in iteration_stats) / len(iteration_stats)
    avg_optimizer = sum(s['optimizer_time'] for s in iteration_stats) / len(iteration_stats)
    avg_iteration = avg_forward + avg_backward + avg_optimizer

    print(f"\nTiming (average per iteration):")
    print(f"  Setup time:     {setup_time:6.3f}s")
    print(f"  Forward:        {avg_forward:6.3f}s")
    print(f"  Backward:       {avg_backward:6.3f}s")
    print(f"  Optimizer:      {avg_optimizer:6.3f}s")
    print(f"  Total/iter:     {avg_iteration:6.3f}s")
    print(f"  Total ({num_iterations} iters): {total_time:6.3f}s")
    print(f"  Throughput:     {num_iterations * batch_size / total_time:.2f} samples/s")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        'method': method_name,
        'absolute_max': absolute_max,
        'final_current': final_current,
        'iterations': iteration_stats,
        'growth': growth,
        'setup_time': setup_time,
        'total_time': total_time,
        'avg_iteration_time': avg_iteration,
        'avg_forward_time': avg_forward,
        'avg_backward_time': avg_backward,
        'avg_optimizer_time': avg_optimizer,
        'throughput': num_iterations * batch_size / total_time,
    }

# =============================================================================
# Setup functions for each method
# =============================================================================

def setup_full_adamw():
    """Full fine-tuning with standard AdamW"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_meso():
    """Full fine-tuning with MeSO (MeSOAdamW)"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )
    model.train()

    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token
    sample_inputs = {k: v.to(device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression
    sparsifier_kwargs = {
        "proj_dim": 1024,
        "proj_max_batch_size": 64,
        "proj_seed": 42,
        "device": str(device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": 262144,
        "proj_max_batch_size": 64,
        "proj_seed": 42,
        "device": str(device),
        "proj_type": "sjlt",
    }

    sparsifiers, projectors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_compressor_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_sparsifiers(sparsifiers)
    grad_hook.set_projectors(projectors)

    optimizer = MeSOAdamW(
        model.parameters(),
        grad_hook=grad_hook,
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    return model, optimizer, tokenizer

def setup_full_galore():
    """Full fine-tuning with GaLore"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )
    model.train()

    # Separate parameters for GaLore
    galore_params = []
    regular_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim == 2 and min(param.shape) >= 128:
                galore_params.append(param)
            else:
                regular_params.append(param)

    param_groups = [
        {'params': regular_params},
        {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
    ]

    optimizer = GaLoreAdamW(param_groups, lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_sgd():
    """Full fine-tuning with vanilla SGD (no momentum)"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_sgd_momentum():
    """Full fine-tuning with SGD + momentum"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_lora_sgd():
    """LoRA with SGD (no momentum)"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256,
        lora_alpha=1,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_lora_sgd_momentum():
    """LoRA with SGD + momentum"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256,
        lora_alpha=1,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_lora_adamw():
    """LoRA with AdamW"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256,
        lora_alpha=1,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_adamw_gradient_checkpointing():
    """Full fine-tuning with AdamW + gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_sgd_gradient_checkpointing():
    """Full fine-tuning with vanilla SGD (no momentum) + gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_sgd_momentum_gradient_checkpointing():
    """Full fine-tuning with SGD + momentum + gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_meso_gradient_checkpointing():
    """
    Full fine-tuning with MeSO (MeSOAdamW) + gradient checkpointing.

    UPDATED: Now uses the optimized hook.py with Solution 1 built-in!
    - Eliminates pre_activations storage (never used in gradient computation)
    - ~50% reduction in activation memory
    - No special configuration needed - optimization is automatic
    """
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.train()

    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token
    sample_inputs = {k: v.to(device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression
    sparsifier_kwargs = {
        "proj_dim": 1024,
        "proj_max_batch_size": 64,
        "proj_seed": 42,
        "device": str(device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": 262144,
        "proj_max_batch_size": 64,
        "proj_seed": 42,
        "device": str(device),
        "proj_type": "sjlt",
    }

    sparsifiers, projectors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_compressor_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_sparsifiers(sparsifiers)
    grad_hook.set_projectors(projectors)

    optimizer = MeSOAdamW(
        model.parameters(),
        grad_hook=grad_hook,
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    return model, optimizer, tokenizer

def setup_full_galore_gradient_checkpointing():
    """Full fine-tuning with GaLore + gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    model.train()

    # Separate parameters for GaLore
    galore_params = []
    regular_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.ndim == 2 and min(param.shape) >= 128:
                galore_params.append(param)
            else:
                regular_params.append(param)

    param_groups = [
        {'params': regular_params},
        {'params': galore_params, 'rank': 128, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
    ]

    optimizer = GaLoreAdamW(param_groups, lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_lora_sgd_gradient_checkpointing():
    """LoRA with SGD (no momentum) + gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256,
        lora_alpha=1,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_lora_sgd_momentum_gradient_checkpointing():
    """LoRA with SGD + momentum + gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256,
        lora_alpha=1,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_lora_adamw_gradient_checkpointing():
    """LoRA with AdamW + gradient checkpointing"""
    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        dtype=torch.float32,
        device_map=device
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=256,
        lora_alpha=1,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

# =============================================================================
# Main execution
# =============================================================================

def get_all_methods():
    """Return list of all available test methods"""
    return [
        ("Full + SGD (no momentum)", setup_full_sgd),
        ("Full + SGD (no momentum) + GC", setup_full_sgd_gradient_checkpointing),
        ("Full + SGD (momentum=0.9)", setup_full_sgd_momentum),
        ("Full + SGD (momentum=0.9) + GC", setup_full_sgd_momentum_gradient_checkpointing),
        ("Full + AdamW", setup_full_adamw),
        ("Full + AdamW + GC", setup_full_adamw_gradient_checkpointing),
        ("Full + MeSO (MeSOAdamW)", setup_full_meso),
        ("Full + MeSO + GC", setup_full_meso_gradient_checkpointing),
        ("Full + GaLore", setup_full_galore),
        ("Full + GaLore + GC", setup_full_galore_gradient_checkpointing),
        ("LoRA + SGD (no momentum)", setup_lora_sgd),
        ("LoRA + SGD (no momentum) + GC", setup_lora_sgd_gradient_checkpointing),
        ("LoRA + SGD (momentum=0.9)", setup_lora_sgd_momentum),
        ("LoRA + SGD (momentum=0.9) + GC", setup_lora_sgd_momentum_gradient_checkpointing),
        ("LoRA + AdamW", setup_lora_adamw),
        ("LoRA + AdamW + GC", setup_lora_adamw_gradient_checkpointing),
    ]

def aggregate_results(results_dir='benchmark_results'):
    """Aggregate results from individual test runs - simplified output"""
    results = []

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found")
        return

    # Load all individual result files
    for filename in sorted(os.listdir(results_dir)):
        if filename.startswith('result_') and filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                result = json.load(f)
                results.append(result)

    if not results:
        print(f"No result files found in '{results_dir}'")
        return

    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(f"Total methods benchmarked: {len(results)}")
    print("="*80 + "\n")

    # Single comparison table with Absolute Max and Throughput
    print(f"{'Method':<40} {'Absolute Max':<15} {'Throughput':<15}")
    print("-" * 80)
    for r in results:
        throughput_str = f"{r.get('throughput', 0):.2f} samp/s" if 'throughput' in r else "N/A"
        print(f"{r['method']:<40} {r['absolute_max']:>8.3f} GB     {throughput_str:>13}")

    # Winner analysis
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)

    min_max = min(r['absolute_max'] for r in results)
    for r in results:
        if r['absolute_max'] == min_max:
            print(f"LOWEST MEMORY:       {r['method']} ({r['absolute_max']:.3f} GB)")
            break

    if any('total_time' in r for r in results):
        results_with_timing = [r for r in results if 'total_time' in r]
        if results_with_timing:
            highest_throughput = max(results_with_timing, key=lambda x: x.get('throughput', 0))
            print(f"HIGHEST THROUGHPUT:  {highest_throughput['method']} ({highest_throughput.get('throughput', 0):.2f} samples/s)")

    print("="*80 + "\n")

    # Save aggregated results
    with open('benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("Aggregated results saved to: benchmark.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Max Memory Benchmark with Timing')
    parser.add_argument('--method', type=int, default=None,
                        help='Run specific method by index (0-20). If not specified, runs all methods.')
    parser.add_argument('--list', action='store_true',
                        help='List all available methods with their indices')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate results from individual runs')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                        help='Directory to save individual results (default: benchmark_results)')

    args = parser.parse_args()

    methods = get_all_methods()

    if args.list:
        print("\nAvailable methods:")
        print("="*80)
        for i, (name, _) in enumerate(methods):
            print(f"{i:2d}: {name}")
        print("="*80)
        sys.exit(0)

    if args.aggregate:
        aggregate_results(args.output_dir)
        sys.exit(0)

    print("\n" + "="*80)
    print("MAX MEMORY BENCHMARK")
    print("="*80)
    print(f"Model: Llama-3.2-1B")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Iterations: {num_iterations}")
    print(f"Precision: float32")
    print("="*80 + "\n")

    # Create output directory if running individual tests
    if args.method is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.method is not None:
        # Run single method
        if args.method < 0 or args.method >= len(methods):
            print(f"Error: Method index {args.method} is out of range (0-{len(methods)-1})")
            print("Use --list to see available methods")
            sys.exit(1)

        method_name, setup_fn = methods[args.method]
        print(f"Running method {args.method}: {method_name}\n")
        result = run_method(method_name, setup_fn)

        # Save individual result
        output_file = os.path.join(args.output_dir, f'result_{args.method:02d}.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {output_file}")

    else:
        # Run all methods (original behavior)
        results = []
        for method_name, setup_fn in methods:
            result = run_method(method_name, setup_fn)
            results.append(result)
            print()

        # Final comparison - simplified
        print("\n" + "="*80)
        print("BENCHMARK RESULTS")
        print("="*80)
        print(f"{'Method':<40} {'Absolute Max':<15} {'Throughput':<15}")
        print("-" * 80)
        for r in results:
            throughput_str = f"{r.get('throughput', 0):.2f} samp/s" if 'throughput' in r else "N/A"
            print(f"{r['method']:<40} {r['absolute_max']:>8.3f} GB     {throughput_str:>13}")

        # Winner analysis
        print("\n" + "="*80)
        print("WINNER ANALYSIS")
        print("="*80)

        min_max = min(r['absolute_max'] for r in results)
        for r in results:
            if r['absolute_max'] == min_max:
                print(f"LOWEST MEMORY:       {r['method']} ({r['absolute_max']:.3f} GB)")
                break

        if any('total_time' in r for r in results):
            results_with_timing = [r for r in results if 'total_time' in r]
            if results_with_timing:
                highest_throughput = max(results_with_timing, key=lambda x: x.get('throughput', 0))
                print(f"HIGHEST THROUGHPUT:  {highest_throughput['method']} ({highest_throughput.get('throughput', 0):.2f} samples/s)")

        print("="*80 + "\n")

        # Save detailed results
        with open('benchmark_max_memory_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        print("Detailed results saved to: benchmark_max_memory_results.json")
