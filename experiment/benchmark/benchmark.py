#!/usr/bin/env python
"""
Enhanced Memory and Performance Benchmark

Comprehensive benchmarking with:
- Memory tracking (peak, current, growth)
- Detailed timing (forward, backward, optimizer, data selection)
- Component-level profiling (compressor operations, gradient computation)
- Support for data selection methods (GREATS, GradNorm, MaxLoss)

Usage:
    # Basic benchmark (existing behavior)
    python benchmark.py --method 7  # Test MeSO

    # List all methods
    python benchmark.py --list

    # Enable detailed profiling
    # Edit the file and set: ENABLE_DETAILED_PROFILING = True

    # Enable data selection profiling
    # Edit the file and set: ENABLE_DATA_SELECTION = True
    # This will test with GREATS method for MeSO configurations

    # Aggregate results
    python benchmark.py --aggregate

Configuration:
    USE_FLASH_ATTENTION: Enable Flash Attention 2 (default: False)
    MODEL_DTYPE: Model precision - torch.float32, torch.bfloat16, or torch.float16 (default: float32)
                 Note: Flash Attention requires bfloat16 or float16
    ENABLE_DETAILED_PROFILING: Show millisecond-level timing and detailed breakdowns
    ENABLE_DATA_SELECTION: Test with GREATS data selection for applicable methods
    batch_size: Training batch size (default: 8)
    seq_length: Sequence length (default: 512)
    num_warmup_iterations: Warmup iterations (not timed, default: 10)
    num_timed_iterations: Timed iterations for throughput (default: 10)
    num_iterations: Total iterations (default: 20 = 10 warmup + 10 timed)

Output:
    - Per-iteration memory and timing stats
    - Final summary with memory growth analysis
    - Timing breakdown (forward, backward, optimizer)
    - Data selection breakdown (if enabled):
        * Dot product computation time (our optimized version!)
        * Greedy selection time
        * Total selection overhead
    - JSON output for further analysis
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Suppress dynamo warnings about custom CUDA kernels (sjlt)
warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

from compress_gradient.hook import GradientHook
from compress_gradient.compressor import setup_model_compressors
from compress_gradient.optimizer import MeSOAdamW
from compress_gradient.utils import greedy_selection

from experiment.benchmark.GaLore.galore_torch import GaLoreAdamW

# Configuration
device = 'cuda'
batch_size = 8
val_batch_size = 1
seq_length = 512
num_warmup_iterations = 10  # Warmup iterations (not timed)
num_timed_iterations = 10   # Timed iterations for throughput measurement
num_iterations = num_warmup_iterations + num_timed_iterations  # Total: 20

# Model configuration
USE_FLASH_ATTENTION = True  # Set to True to enable Flash Attention 2
MODEL_DTYPE = torch.bfloat16  # Options: torch.float32, torch.bfloat16, torch.float16
# Note: Flash Attention requires bfloat16 or float16 (not float32)

# Profiling flags
ENABLE_DETAILED_PROFILING = True  # Set to False for basic benchmark only
ENABLE_DATA_SELECTION = False  # Set to True to test with GREATS/GradNorm
ENABLE_MEMORY_SNAPSHOT = False  # Set to True to capture memory snapshots for visualization
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000  # Max memory events to capture

@dataclass
class IterationProfile:
    """Detailed profiling data for a single iteration."""
    iteration: int

    # Memory (GB)
    current_after_optimizer: float
    max_ever: float
    max_during_forward: float
    max_during_backward: float
    max_during_optimizer: float

    # Basic timing (seconds)
    forward_time: float
    backward_time: float
    optimizer_time: float

    # Detailed timing (seconds) - only populated if ENABLE_DETAILED_PROFILING
    selection_total_time: float = 0.0
    selection_val_forward_time: float = 0.0
    selection_val_backward_time: float = 0.0
    selection_train_forward_time: float = 0.0
    selection_train_backward_time: float = 0.0
    selection_grad_comp_time: float = 0.0  # Total gradient computation (val+train fwd/bwd + dot product)
    selection_dotprod_compute_time: float = 0.0  # Just the actual dot product calculation
    selection_greedy_time: float = 0.0
    compressor_refresh_time: float = 0.0

    # Additional timing breakdowns for throughput analysis
    batch_prep_time: float = 0.0  # Time to prepare batch data
    data_transfer_time: float = 0.0  # Time to move data to GPU
    model_forward_compute_time: float = 0.0  # Pure forward computation
    loss_computation_time: float = 0.0  # Loss calculation
    backward_compute_time: float = 0.0  # Pure backward computation
    gradient_sync_time: float = 0.0  # Gradient synchronization (if applicable)
    optimizer_step_time: float = 0.0  # Pure optimizer step
    optimizer_zero_grad_time: float = 0.0  # Zeroing gradients
    cuda_sync_overhead: float = 0.0  # CUDA synchronization overhead

    # Enhanced memory profiling
    mem_before_forward: float = 0.0
    mem_after_forward: float = 0.0
    mem_after_backward: float = 0.0
    mem_after_optimizer: float = 0.0
    mem_delta_forward: float = 0.0
    mem_delta_backward: float = 0.0
    mem_delta_optimizer: float = 0.0

    # Additional metadata
    selected_samples: int = 0
    total_samples: int = 0


def get_mem():
    """Get current GPU memory in GB."""
    return torch.cuda.memory_allocated() / 1024**3


def get_max_mem():
    """Get peak GPU memory in GB."""
    return torch.cuda.max_memory_allocated() / 1024**3


def reset():
    """Reset CUDA cache and peak memory stats."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def cuda_timer_sync():
    """Synchronize CUDA for accurate timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _profile_data_selection(model, train_batch, val_batch, grad_hook, profile: IterationProfile,
                            return_compressed_grads: bool = False):
    """
    Profile data selection (GREATS/GradNorm) with detailed timing breakdown.

    Uses separate forward/backward passes for validation and training batches
    to ensure correct per-sample gradient computation.

    Args:
        model: The model
        train_batch: Training batch
        val_batch: Validation batch
        grad_hook: GradientHook instance
        profile: IterationProfile to record timing
        return_compressed_grads: Whether to return compressed gradients for reuse (True for MeSO)

    Returns:
        Tuple of (grad_dot_scores, similarity_matrix, compressed_grads)
        compressed_grads will be None if return_compressed_grads=False
    """
    device = train_batch['input_ids'].device
    val_batch = {k: v.to(device) for k, v in val_batch.items()}

    train_batch_size = train_batch['input_ids'].shape[0]

    # === STEP 1: Validation Forward/Backward Pass ===
    cuda_timer_sync()
    val_forward_start = time.time()

    model.zero_grad()
    val_outputs = model(**val_batch)
    val_loss = val_outputs.loss

    cuda_timer_sync()
    val_backward_start = time.time()

    val_loss.backward()

    cuda_timer_sync()
    val_extract_start = time.time()

    # Get validation gradients from hooks (per-sample)
    val_grads = grad_hook.get_compressed_grads()

    cuda_timer_sync()
    profile.selection_val_forward_time = val_backward_start - val_forward_start
    profile.selection_val_backward_time = val_extract_start - val_backward_start

    # === STEP 2: Training Forward/Backward Pass ===
    cuda_timer_sync()
    train_forward_start = time.time()

    model.zero_grad()
    train_outputs = model(**train_batch)
    train_loss = train_outputs.loss

    cuda_timer_sync()
    train_backward_start = time.time()

    train_loss.backward()

    cuda_timer_sync()
    train_extract_start = time.time()

    # Get training gradients from hooks (per-sample)
    train_grads = grad_hook.get_compressed_grads()

    cuda_timer_sync()
    profile.selection_train_forward_time = train_backward_start - train_forward_start
    profile.selection_train_backward_time = train_extract_start - train_backward_start

    # === STEP 3: Gradient Dot Product Computation ===
    cuda_timer_sync()
    dotprod_start = time.time()

    # Infer dtype from gradients (important for bfloat16/float16 support)
    grad_dtype = None
    for g in train_grads:
        if g is not None:
            grad_dtype = g.dtype
            break
    if grad_dtype is None:
        grad_dtype = torch.float32  # Fallback to float32 if no gradients found

    # Initialize accumulators on device with correct dtype
    grad_dot_scores = torch.zeros(train_batch_size, device=device, dtype=grad_dtype)
    similarity_matrix = torch.zeros(train_batch_size, train_batch_size, device=device, dtype=grad_dtype)

    # Accumulate dot products layer by layer
    for train_g, val_g in zip(train_grads, val_grads):
        if train_g is None or val_g is None:
            continue

        # Average validation gradient for this layer: [layer_dim]
        val_g_avg = val_g.mean(dim=0)

        # Gradient dot product for this layer
        torch.addmv(grad_dot_scores, train_g, val_g_avg, out=grad_dot_scores)

        # Similarity matrix for this layer
        torch.addmm(similarity_matrix, train_g, train_g.t(), out=similarity_matrix)

    # Detach from computation graph
    grad_dot_scores = grad_dot_scores.detach()
    similarity_matrix = similarity_matrix.detach()

    cuda_timer_sync()
    profile.selection_dotprod_compute_time = time.time() - dotprod_start

    # === STEP 4: Return compressed gradients if requested ===
    compressed_grads = None
    if return_compressed_grads:
        # Deep copy to avoid being overwritten by subsequent operations
        compressed_grads = [g.clone() if g is not None else None for g in train_grads]

    # Total time for gradient computation phase (includes all sub-steps)
    profile.selection_grad_comp_time = (
        profile.selection_val_forward_time +
        profile.selection_val_backward_time +
        profile.selection_train_forward_time +
        profile.selection_train_backward_time +
        profile.selection_dotprod_compute_time
    )

    return grad_dot_scores, similarity_matrix, compressed_grads


def _select_compressed_grads(compressed_grads, selected_indices):
    """
    Select compressed gradients for chosen samples.

    This keeps the selection logic consistent with trainer.py - the optimizer doesn't
    need to know about data selection at all. The optimizer will then aggregate
    these selected gradients.

    Args:
        compressed_grads: List of per-sample compressed gradients [batch_size, k_l] for each layer
        selected_indices: Indices of selected samples (list or tensor)

    Returns:
        List of selected compressed gradients [num_selected, k_l] for each layer
    """
    selected_grads = []
    for grad in compressed_grads:
        if grad is None:
            selected_grads.append(None)
        else:
            # Select samples only - optimizer will average them
            selected_grad = grad[selected_indices]  # [num_selected, k_l]
            selected_grads.append(selected_grad)
    return selected_grads


def _run_single_iteration(model, optimizer, tokenizer, grad_hook, val_batch, selection_method,
                          enable_profiling=False, iter_num=0, is_warmup=False):
    """
    Run a single training iteration with optional profiling.

    This function is used by both warmup and timed phases to ensure they execute identical operations.

    Args:
        model: The model
        optimizer: The optimizer
        tokenizer: The tokenizer
        grad_hook: GradientHook instance (or None)
        val_batch: Validation batch for data selection (or None)
        selection_method: Data selection method ('GREATS', 'GradNorm', 'MaxLoss', None)
        enable_profiling: If True, collect detailed timing and memory stats
        iter_num: Iteration number for display
        is_warmup: If True, this is a warmup iteration (for display only)

    Returns:
        IterationProfile if enable_profiling=True, None otherwise
    """
    # Initialize iteration profile
    profile = None
    if enable_profiling:
        profile = IterationProfile(
            iteration=iter_num,
            current_after_optimizer=0.0,
            max_ever=0.0,
            max_during_forward=0.0,
            max_during_backward=0.0,
            max_during_optimizer=0.0,
            forward_time=0.0,
            backward_time=0.0,
            optimizer_time=0.0,
            total_samples=batch_size
        )

    # BATCH PREPARATION (detailed profiling)
    if enable_profiling and ENABLE_DETAILED_PROFILING:
        cuda_timer_sync()
        batch_prep_start = time.time()

    batch = tokenizer(
        ["This is a test sentence for memory profiling."] * batch_size,
        return_tensors='pt',
        padding='max_length',
        max_length=seq_length,
        truncation=True
    )

    if enable_profiling and ENABLE_DETAILED_PROFILING:
        profile.batch_prep_time = time.time() - batch_prep_start

        # Data transfer to GPU
        cuda_timer_sync()
        transfer_start = time.time()

    batch = batch.to(device)
    batch['labels'] = batch['input_ids'].clone()

    if enable_profiling and ENABLE_DETAILED_PROFILING:
        cuda_timer_sync()
        profile.data_transfer_time = time.time() - transfer_start

    # DATA SELECTION PHASE (if selection method specified)
    selected_indices = None
    saved_compressed_grads = None

    # Detect if using compressed optimizer (MeSO)
    using_compressed_optimizer = isinstance(optimizer, MeSOAdamW)

    if selection_method and grad_hook and val_batch:
        selection_start = time.time()

        if selection_method == 'GREATS':
            scores, similarity_matrix, saved_compressed_grads = _profile_data_selection(
                model, batch, val_batch, grad_hook, profile if enable_profiling else IterationProfile(
                    iteration=0, current_after_optimizer=0.0, max_ever=0.0,
                    max_during_forward=0.0, max_during_backward=0.0, max_during_optimizer=0.0,
                    forward_time=0.0, backward_time=0.0, optimizer_time=0.0
                ),
                return_compressed_grads=using_compressed_optimizer
            )
            lr = optimizer.param_groups[0].get("lr", 5e-5)

            if enable_profiling:
                cuda_timer_sync()
            greedy_start = time.time()
            selected_indices = greedy_selection(
                scores * lr,
                similarity_matrix * (lr ** 2),
                int(len(scores) * 0.5)  # 50% selection
            )
            if enable_profiling:
                cuda_timer_sync()
                profile.selection_greedy_time = time.time() - greedy_start

        elif selection_method == 'GradNorm':
            _, similarity_matrix, saved_compressed_grads = _profile_data_selection(
                model, batch, val_batch, grad_hook, profile if enable_profiling else IterationProfile(
                    iteration=0, current_after_optimizer=0.0, max_ever=0.0,
                    max_during_forward=0.0, max_during_backward=0.0, max_during_optimizer=0.0,
                    forward_time=0.0, backward_time=0.0, optimizer_time=0.0
                ),
                return_compressed_grads=using_compressed_optimizer
            )

            if enable_profiling:
                cuda_timer_sync()
            greedy_start = time.time()
            scores = torch.diag(similarity_matrix)
            selected_indices = greedy_selection(
                scores,
                similarity_matrix * 0,
                int(len(scores) * 0.5)
            )
            if enable_profiling:
                cuda_timer_sync()
                profile.selection_greedy_time = time.time() - greedy_start

        elif selection_method == 'MaxLoss':
            if enable_profiling:
                cuda_timer_sync()
            selection_forward_start = time.time()
            with torch.no_grad():
                losses = []
                for idx in range(batch_size):
                    single_batch = {k: v[[idx]] for k, v in batch.items()}
                    outputs = model(**single_batch)
                    losses.append(outputs.loss.item())
            if enable_profiling:
                cuda_timer_sync()
                profile.selection_train_forward_time = time.time() - selection_forward_start

                cuda_timer_sync()
            greedy_start = time.time()
            selected_indices = greedy_selection(
                torch.tensor(losses),
                torch.zeros((len(losses), len(losses))),
                int(len(losses) * 0.5)
            )
            if enable_profiling:
                cuda_timer_sync()
                profile.selection_greedy_time = time.time() - greedy_start

        if enable_profiling:
            profile.selection_total_time = time.time() - selection_start
            profile.selected_samples = len(selected_indices) if selected_indices is not None else batch_size

        # Filter batch if selection was performed
        if selected_indices is not None:
            batch = {k: v[selected_indices] for k, v in batch.items()}

    # Check if we can reuse gradients from selection phase (MeSO + GREATS optimization)
    can_reuse_grads = (
        using_compressed_optimizer and
        saved_compressed_grads is not None and
        selected_indices is not None
    )

    mem_before_forward = get_mem() if enable_profiling else 0.0
    max_before_forward = get_max_mem() if enable_profiling else 0.0
    if enable_profiling:
        profile.mem_before_forward = mem_before_forward

    if can_reuse_grads:
        # === OPTIMIZED PATH: Reuse compressed gradients from selection ===
        if not is_warmup and enable_profiling:
            print(f"  → Reusing compressed gradients from selection (MeSO optimization)")

        # Select compressed gradients for chosen samples BEFORE injecting
        # This matches the trainer.py design where selection happens before optimizer
        grad_hook.compressed_grads = _select_compressed_grads(
            saved_compressed_grads,
            selected_indices
        )

        # Compute loss for logging/display only (no backward pass needed!)
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss

        # Forward and backward time are essentially zero (just loss computation for display)
        if enable_profiling:
            profile.forward_time = 0.0
            profile.backward_time = 0.0
            profile.model_forward_compute_time = 0.0
            profile.loss_computation_time = 0.0
            profile.backward_compute_time = 0.0

            # Memory doesn't change (no actual computation)
            mem_after_forward = mem_before_forward
            max_after_forward = max_before_forward
            profile.max_during_forward = 0.0
            profile.mem_after_forward = mem_after_forward
            profile.mem_delta_forward = 0.0

            mem_after_backward = mem_after_forward
            max_after_backward = max_after_forward
            profile.max_during_backward = 0.0
            profile.mem_after_backward = mem_after_backward
            profile.mem_delta_backward = 0.0

            if ENABLE_DETAILED_PROFILING:
                print(f"  Forward:   SKIPPED (reusing gradients from selection)")
                print(f"  Backward:  SKIPPED (reusing gradients from selection)")
            else:
                print(f"  Forward:   SKIPPED (using cached gradients)")
                print(f"  Backward:  SKIPPED (using cached gradients)")

    else:
        # === STANDARD PATH: Regular forward/backward pass ===

        # FORWARD PASS WITH DETAILED BREAKDOWN
        if enable_profiling:
            cuda_timer_sync()
        forward_start = time.time()

        # Model forward computation
        if enable_profiling and ENABLE_DETAILED_PROFILING:
            cuda_timer_sync()
            model_forward_start = time.time()

        outputs = model(**batch)

        if enable_profiling and ENABLE_DETAILED_PROFILING:
            cuda_timer_sync()
            profile.model_forward_compute_time = time.time() - model_forward_start

            # Loss computation
            cuda_timer_sync()
            loss_start = time.time()

        loss = outputs.loss

        if enable_profiling and ENABLE_DETAILED_PROFILING:
            cuda_timer_sync()
            profile.loss_computation_time = time.time() - loss_start

        if enable_profiling:
            cuda_timer_sync()
            profile.forward_time = time.time() - forward_start

            mem_after_forward = get_mem()
            max_after_forward = get_max_mem()
            profile.max_during_forward = max_after_forward - max_before_forward
            profile.mem_after_forward = mem_after_forward
            profile.mem_delta_forward = mem_after_forward - mem_before_forward

            if ENABLE_DETAILED_PROFILING:
                breakdown = f"model: {profile.model_forward_compute_time*1000:.1f}ms, loss: {profile.loss_computation_time*1000:.1f}ms"
                print(f"  Forward:   Current = {mem_after_forward:6.3f} GB, Max = {max_after_forward:6.3f} GB (peak: +{profile.max_during_forward:.3f} GB, delta: {profile.mem_delta_forward:+.3f} GB, time: {profile.forward_time*1000:.1f}ms, {breakdown})")
            else:
                print(f"  Forward:   Current = {mem_after_forward:6.3f} GB, Max = {max_after_forward:6.3f} GB (time: {profile.forward_time:.3f}s)")

        # BACKWARD PASS WITH DETAILED BREAKDOWN
        if enable_profiling:
            cuda_timer_sync()
        backward_start = time.time()

        # Backward computation
        if enable_profiling and ENABLE_DETAILED_PROFILING:
            cuda_timer_sync()
            backward_compute_start = time.time()

        loss.backward()

        if enable_profiling and ENABLE_DETAILED_PROFILING:
            cuda_timer_sync()
            profile.backward_compute_time = time.time() - backward_compute_start

        if enable_profiling:
            cuda_timer_sync()
            profile.backward_time = time.time() - backward_start

            mem_after_backward = get_mem()
            max_after_backward = get_max_mem()
            profile.max_during_backward = max_after_backward - max_after_forward
            profile.mem_after_backward = mem_after_backward
            profile.mem_delta_backward = mem_after_backward - mem_after_forward

            if ENABLE_DETAILED_PROFILING:
                print(f"  Backward:  Current = {mem_after_backward:6.3f} GB, Max = {max_after_backward:6.3f} GB (peak: +{profile.max_during_backward:.3f} GB, delta: {profile.mem_delta_backward:+.3f} GB, time: {profile.backward_time*1000:.1f}ms, compute: {profile.backward_compute_time*1000:.1f}ms)")
            else:
                print(f"  Backward:  Current = {mem_after_backward:6.3f} GB, Max = {max_after_backward:6.3f} GB (time: {profile.backward_time:.3f}s)")

    # OPTIMIZER STEP WITH DETAILED BREAKDOWN
    if enable_profiling:
        cuda_timer_sync()
    optimizer_start = time.time()

    # Optimizer step
    if enable_profiling and ENABLE_DETAILED_PROFILING:
        cuda_timer_sync()
        opt_step_start = time.time()

    # Optimizer step (gradients already filtered if reusing from selection)
    optimizer.step()

    if enable_profiling and ENABLE_DETAILED_PROFILING:
        cuda_timer_sync()
        profile.optimizer_step_time = time.time() - opt_step_start

        # Zero grad
        cuda_timer_sync()
        zero_grad_start = time.time()

    optimizer.zero_grad()

    if enable_profiling and ENABLE_DETAILED_PROFILING:
        cuda_timer_sync()
        profile.optimizer_zero_grad_time = time.time() - zero_grad_start

    if enable_profiling:
        cuda_timer_sync()
        profile.optimizer_time = time.time() - optimizer_start

        mem_after_optimizer = get_mem()
        max_after_optimizer = get_max_mem()
        profile.max_during_optimizer = max_after_optimizer - max_after_backward
        profile.current_after_optimizer = mem_after_optimizer
        profile.mem_after_optimizer = mem_after_optimizer
        profile.mem_delta_optimizer = mem_after_optimizer - mem_after_backward
        profile.max_ever = max_after_optimizer

        if ENABLE_DETAILED_PROFILING:
            opt_breakdown = f"step: {profile.optimizer_step_time*1000:.1f}ms, zero_grad: {profile.optimizer_zero_grad_time*1000:.1f}ms"
            print(f"  Optimizer: Current = {mem_after_optimizer:6.3f} GB, Max = {max_after_optimizer:6.3f} GB (peak: +{profile.max_during_optimizer:.3f} GB, delta: {profile.mem_delta_optimizer:+.3f} GB, time: {profile.optimizer_time*1000:.1f}ms, {opt_breakdown})")
            if profile.selection_total_time > 0:
                print(f"  Selection: Time = {profile.selection_total_time*1000:.1f}ms (grad_comp: {profile.selection_grad_comp_time*1000:.1f}ms, greedy: {profile.selection_greedy_time*1000:.1f}ms, selected: {profile.selected_samples}/{profile.total_samples})")
            if profile.batch_prep_time > 0 or profile.data_transfer_time > 0:
                print(f"  Overhead:  Batch prep: {profile.batch_prep_time*1000:.1f}ms, Data transfer: {profile.data_transfer_time*1000:.1f}ms")
        else:
            print(f"  Optimizer: Current = {mem_after_optimizer:6.3f} GB, Max = {max_after_optimizer:6.3f} GB (time: {profile.optimizer_time:.3f}s)")

    return profile


def run_method(method_name, setup_fn, selection_method: Optional[str] = None):
    """
    Run a method for multiple iterations with comprehensive profiling.

    Args:
        method_name: Name of the method being tested
        setup_fn: Function that returns (model, optimizer, tokenizer, grad_hook)
        selection_method: Optional data selection method ('GREATS', 'GradNorm', 'MaxLoss', None)
    """
    print("="*80)
    print(f"Testing: {method_name}")
    if selection_method:
        print(f"Data Selection: {selection_method}")
    print("="*80)

    reset()

    # Setup model and optimizer
    print("Setting up model and optimizer...")
    cuda_timer_sync()
    setup_start = time.time()
    setup_result = setup_fn()

    # Handle different return types
    if len(setup_result) == 3:
        model, optimizer, tokenizer = setup_result
        grad_hook = None
    elif len(setup_result) == 4:
        model, optimizer, tokenizer, grad_hook = setup_result
    else:
        raise ValueError(f"setup_fn must return (model, optimizer, tokenizer) or (model, optimizer, tokenizer, grad_hook)")

    cuda_timer_sync()
    setup_time = time.time() - setup_start
    print(f"Setup time: {setup_time:.2f}s")

    mem_after_setup = get_mem()
    max_after_setup = get_max_mem()
    print(f"After setup - Current: {mem_after_setup:.3f} GB, Max so far: {max_after_setup:.3f} GB")

    # Prepare validation batch for data selection (if selection method specified)
    val_batch = None
    if selection_method and grad_hook:
        val_batch = tokenizer(
            ["Validation sentence for data selection."] * val_batch_size,  # Small val batch
            return_tensors='pt',
            padding='max_length',
            max_length=seq_length,
            truncation=True
        ).to(device)
        val_batch['labels'] = val_batch['input_ids'].clone()

    # Track memory per iteration and timing
    iteration_stats = []

    # WARMUP PHASE (not timed, not profiled)
    print(f"\n{'='*80}")
    print(f"WARMUP PHASE - {num_warmup_iterations} iterations (not timed)")
    print(f"{'='*80}")
    for i in range(num_warmup_iterations):
        print(f"Warmup {i+1}/{num_warmup_iterations}...", end=' ')

        # Run the same iteration logic as timed phase, but without profiling
        _run_single_iteration(
            model, optimizer, tokenizer, grad_hook, val_batch, selection_method,
            enable_profiling=False, iter_num=i+1, is_warmup=True
        )

        print("done")

    print(f"\n{'='*80}")
    print(f"TIMED PHASE - {num_timed_iterations} iterations (measured for throughput)")
    print(f"{'='*80}")

    # Start memory snapshot recording if enabled
    snapshot_file = None
    if ENABLE_MEMORY_SNAPSHOT:
        print(f"\n📸 Memory snapshot recording ENABLED")
        print(f"   Max events: {MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT:,}")
        torch.cuda.memory._record_memory_history(
            max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
        )

    # Start timing for timed iterations only
    cuda_timer_sync()
    total_start = time.time()

    for i in range(num_timed_iterations):
        iter_num = num_warmup_iterations + i + 1
        print(f"\n--- Iteration {iter_num}/{num_iterations} (Timed {i+1}/{num_timed_iterations}) ---")

        # Run the same iteration logic as warmup, but WITH profiling enabled
        profile = _run_single_iteration(
            model, optimizer, tokenizer, grad_hook, val_batch, selection_method,
            enable_profiling=True, iter_num=i+1, is_warmup=False
        )

        iteration_stats.append(asdict(profile))

    # Total training time
    cuda_timer_sync()
    total_time = time.time() - total_start

    # Save memory snapshot if enabled
    if ENABLE_MEMORY_SNAPSHOT:
        import os
        os.makedirs('memory_snapshots', exist_ok=True)
        snapshot_file = f"memory_snapshots/{method_name.replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')}.pickle"

        print(f"\n📸 Saving memory snapshot...")
        try:
            torch.cuda.memory._dump_snapshot(snapshot_file)
            print(f"   ✓ Snapshot saved to: {snapshot_file}")
            print(f"   💡 To visualize: python -m torch.utils.viz_memory {snapshot_file}")
        except Exception as e:
            print(f"   ✗ Failed to capture memory snapshot: {e}")
            snapshot_file = None
        finally:
            # Stop recording memory snapshot history
            torch.cuda.memory._record_memory_history(enabled=None)

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

    # Check memory growth (only looking at timed iterations)
    first_iter_max = iteration_stats[0]['max_ever']
    last_iter_max = iteration_stats[-1]['max_ever']
    growth = last_iter_max - first_iter_max

    print(f"\nMemory growth during timed iterations (iter 1 to {num_timed_iterations}):")
    print(f"  Iter 1 max:  {first_iter_max:.3f} GB")
    print(f"  Iter {num_timed_iterations} max:  {last_iter_max:.3f} GB")
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

    # Calculate average iteration time including ALL components
    # This ensures Total(ms) matches what's actually included in throughput calculation
    avg_selection = sum(s.get('selection_total_time', 0) for s in iteration_stats) / len(iteration_stats)
    avg_batch_prep = sum(s.get('batch_prep_time', 0) for s in iteration_stats) / len(iteration_stats)
    avg_data_transfer = sum(s.get('data_transfer_time', 0) for s in iteration_stats) / len(iteration_stats)

    # Total iteration time = sum of all components
    avg_iteration = avg_forward + avg_backward + avg_optimizer + avg_selection + avg_batch_prep + avg_data_transfer

    print(f"\nTiming (average per iteration):")
    print(f"  Setup time:     {setup_time:6.3f}s")
    if avg_selection > 0:
        print(f"  Selection:      {avg_selection:6.3f}s ({avg_selection*1000:6.1f}ms)")
    print(f"  Forward:        {avg_forward:6.3f}s ({avg_forward*1000:6.1f}ms)")
    print(f"  Backward:       {avg_backward:6.3f}s ({avg_backward*1000:6.1f}ms)")
    print(f"  Optimizer:      {avg_optimizer:6.3f}s ({avg_optimizer*1000:6.1f}ms)")
    if avg_batch_prep > 0 or avg_data_transfer > 0:
        print(f"  Batch prep:     {avg_batch_prep:6.3f}s ({avg_batch_prep*1000:6.1f}ms)")
        print(f"  Data transfer:  {avg_data_transfer:6.3f}s ({avg_data_transfer*1000:6.1f}ms)")

    # Detailed profiling breakdown (if enabled)
    if ENABLE_DETAILED_PROFILING:
        # Check if we have detailed timing data
        if any(s.get('batch_prep_time', 0) > 0 for s in iteration_stats):
            avg_batch_prep = sum(s.get('batch_prep_time', 0) for s in iteration_stats) / len(iteration_stats)
            avg_data_transfer = sum(s.get('data_transfer_time', 0) for s in iteration_stats) / len(iteration_stats)
            avg_model_forward = sum(s.get('model_forward_compute_time', 0) for s in iteration_stats) / len(iteration_stats)
            avg_loss_comp = sum(s.get('loss_computation_time', 0) for s in iteration_stats) / len(iteration_stats)
            avg_backward_comp = sum(s.get('backward_compute_time', 0) for s in iteration_stats) / len(iteration_stats)
            avg_opt_step = sum(s.get('optimizer_step_time', 0) for s in iteration_stats) / len(iteration_stats)
            avg_opt_zero = sum(s.get('optimizer_zero_grad_time', 0) for s in iteration_stats) / len(iteration_stats)

            print(f"\n{'='*80}")
            print(f"DETAILED BREAKDOWN - Bottleneck Analysis")
            print(f"{'='*80}")
            print(f"\nOverhead (non-compute):")
            print(f"  Batch preparation:     {avg_batch_prep*1000:6.1f}ms ({avg_batch_prep/avg_iteration*100:5.1f}%)")
            print(f"  Data transfer to GPU:  {avg_data_transfer*1000:6.1f}ms ({avg_data_transfer/avg_iteration*100:5.1f}%)")

            print(f"\nForward Pass Breakdown:")
            print(f"  Model computation:     {avg_model_forward*1000:6.1f}ms ({avg_model_forward/avg_iteration*100:5.1f}%)")
            print(f"  Loss computation:      {avg_loss_comp*1000:6.1f}ms ({avg_loss_comp/avg_iteration*100:5.1f}%)")

            print(f"\nBackward Pass Breakdown:")
            print(f"  Gradient computation:  {avg_backward_comp*1000:6.1f}ms ({avg_backward_comp/avg_iteration*100:5.1f}%)")

            print(f"\nOptimizer Breakdown:")
            print(f"  Parameter update:      {avg_opt_step*1000:6.1f}ms ({avg_opt_step/avg_iteration*100:5.1f}%)")
            print(f"  Zeroing gradients:     {avg_opt_zero*1000:6.1f}ms ({avg_opt_zero/avg_iteration*100:5.1f}%)")

            # Find the biggest bottleneck
            components = {
                "Batch preparation": avg_batch_prep,
                "Data transfer": avg_data_transfer,
                "Model forward": avg_model_forward,
                "Loss computation": avg_loss_comp,
                "Backward computation": avg_backward_comp,
                "Optimizer step": avg_opt_step,
                "Optimizer zero_grad": avg_opt_zero
            }
            max_component = max(components.items(), key=lambda x: x[1])
            print(f"\n🔍 BIGGEST BOTTLENECK: {max_component[0]} ({max_component[1]*1000:.1f}ms, {max_component[1]/avg_iteration*100:.1f}%)")

        # Data selection breakdown
        if any(s.get('selection_total_time', 0) > 0 for s in iteration_stats):
            stats_with_selection = [s for s in iteration_stats if s.get('selection_total_time', 0) > 0]
            if stats_with_selection:
                avg_selection_total = sum(s['selection_total_time'] for s in stats_with_selection) / len(stats_with_selection)
                avg_selection_grad_comp = sum(s.get('selection_grad_comp_time', 0) for s in stats_with_selection) / len(stats_with_selection)
                avg_selection_greedy = sum(s.get('selection_greedy_time', 0) for s in stats_with_selection) / len(stats_with_selection)

                # Detailed sub-components
                avg_val_fwd = sum(s.get('selection_val_forward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
                avg_val_bwd = sum(s.get('selection_val_backward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
                avg_train_fwd = sum(s.get('selection_train_forward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
                avg_train_bwd = sum(s.get('selection_train_backward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
                avg_dotprod_compute = sum(s.get('selection_dotprod_compute_time', 0) for s in stats_with_selection) / len(stats_with_selection)

                print(f"\nData Selection Breakdown (GREATS/GradNorm):")
                print(f"  Total selection:           {avg_selection_total:6.3f}s ({avg_selection_total*1000:6.1f}ms, {avg_selection_total/avg_iteration*100:5.1f}%)")
                print(f"    1) Gradient computation: {avg_selection_grad_comp:6.3f}s ({avg_selection_grad_comp*1000:6.1f}ms, {avg_selection_grad_comp/avg_iteration*100:5.1f}%)")

                if avg_val_fwd > 0:
                    print(f"       ├─ Val forward:       {avg_val_fwd:6.3f}s ({avg_val_fwd*1000:6.1f}ms, {avg_val_fwd/avg_iteration*100:5.1f}%)")
                    print(f"       ├─ Val backward:      {avg_val_bwd:6.3f}s ({avg_val_bwd*1000:6.1f}ms, {avg_val_bwd/avg_iteration*100:5.1f}%)")
                    print(f"       ├─ Train forward:     {avg_train_fwd:6.3f}s ({avg_train_fwd*1000:6.1f}ms, {avg_train_fwd/avg_iteration*100:5.1f}%)")
                    print(f"       ├─ Train backward:    {avg_train_bwd:6.3f}s ({avg_train_bwd*1000:6.1f}ms, {avg_train_bwd/avg_iteration*100:5.1f}%)")
                    print(f"       └─ Dot product calc:  {avg_dotprod_compute:6.3f}s ({avg_dotprod_compute*1000:6.1f}ms, {avg_dotprod_compute/avg_iteration*100:5.1f}%)")

                print(f"    2) Greedy selection:     {avg_selection_greedy:6.3f}s ({avg_selection_greedy*1000:6.1f}ms, {avg_selection_greedy/avg_iteration*100:5.1f}%)")

        # MeSO-specific operations note
        if 'MeSO' in method_name or 'meso' in method_name.lower():
            print(f"\nMeSO-Specific Operations:")
            if selection_method in ['GREATS', 'GradNorm']:
                print(f"  Optimization:           ✓ GRADIENT REUSE ENABLED")
                print(f"    - Compressed gradients computed during selection")
                print(f"    - Forward/backward SKIPPED for selected samples")
                print(f"    - Optimizer aggregates cached compressed gradients")
                print(f"    - This matches real trainer.py behavior (33% speedup!)")
            else:
                print(f"  Gradient compression:   Included in backward time (via hooks)")
                print(f"  Gradient decompression: Included in optimizer time (during step)")
            print(f"  Compressor refresh:     Not triggered (update_freq=200, iterations={num_timed_iterations})")

    print(f"\n{'='*80}")
    print(f"Total/iter:     {avg_iteration:6.3f}s ({avg_iteration*1000:.1f}ms)")
    print(f"Total ({num_timed_iterations} timed iters): {total_time:6.3f}s")
    print(f"Throughput:     {num_timed_iterations * batch_size / total_time:.2f} samples/s")

    # Sanity check: verify our component sum matches wall-clock time
    expected_total = avg_iteration * num_timed_iterations
    actual_total = total_time
    timing_discrepancy = abs(expected_total - actual_total)
    timing_discrepancy_pct = (timing_discrepancy / actual_total) * 100

    if timing_discrepancy_pct > 5.0:  # More than 5% discrepancy
        print(f"\n⚠ TIMING DISCREPANCY DETECTED:")
        print(f"  Expected total (sum of components): {expected_total:.3f}s")
        print(f"  Actual wall-clock total:            {actual_total:.3f}s")
        print(f"  Unaccounted time:                   {timing_discrepancy:.3f}s ({timing_discrepancy_pct:.1f}%)")
        print(f"  This suggests some operations are not being timed!")
    else:
        print(f"✓ Timing verification: Components sum to {expected_total:.3f}s vs wall-clock {actual_total:.3f}s (discrepancy: {timing_discrepancy_pct:.1f}%)")

    print(f"\nNote: Throughput measured from {num_timed_iterations} iterations after {num_warmup_iterations} warmup iterations")
    print(f"{'='*80}")

    del model, optimizer
    if grad_hook:
        del grad_hook
    torch.cuda.empty_cache()

    result = {
        'method': method_name,
        'absolute_max': absolute_max,
        'final_current': final_current,
        'iterations': iteration_stats,
        'growth': growth,
        'setup_time': setup_time,
        'total_time': total_time,
        'num_warmup_iterations': num_warmup_iterations,
        'num_timed_iterations': num_timed_iterations,
        'avg_iteration_time': avg_iteration,
        'avg_forward_time': avg_forward,
        'avg_backward_time': avg_backward,
        'avg_optimizer_time': avg_optimizer,
        'throughput': num_timed_iterations * batch_size / total_time,
        'memory_snapshot_file': snapshot_file if ENABLE_MEMORY_SNAPSHOT else None,
    }

    # Add detailed profiling stats if available
    if ENABLE_DETAILED_PROFILING:
        # Data selection stats
        stats_with_selection = [s for s in iteration_stats if s.get('selection_total_time', 0) > 0]
        if stats_with_selection:
            result['avg_selection_total_time'] = sum(s['selection_total_time'] for s in stats_with_selection) / len(stats_with_selection)
            result['avg_selection_grad_comp_time'] = sum(s.get('selection_grad_comp_time', 0) for s in stats_with_selection) / len(stats_with_selection)
            result['avg_selection_greedy_time'] = sum(s.get('selection_greedy_time', 0) for s in stats_with_selection) / len(stats_with_selection)

            # Detailed selection breakdown
            result['avg_selection_val_forward_time'] = sum(s.get('selection_val_forward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
            result['avg_selection_val_backward_time'] = sum(s.get('selection_val_backward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
            result['avg_selection_train_forward_time'] = sum(s.get('selection_train_forward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
            result['avg_selection_train_backward_time'] = sum(s.get('selection_train_backward_time', 0) for s in stats_with_selection) / len(stats_with_selection)
            result['avg_selection_dotprod_compute_time'] = sum(s.get('selection_dotprod_compute_time', 0) for s in stats_with_selection) / len(stats_with_selection)

        # Detailed component timings
        if any(s.get('batch_prep_time', 0) > 0 for s in iteration_stats):
            result['avg_batch_prep_time'] = sum(s.get('batch_prep_time', 0) for s in iteration_stats) / len(iteration_stats)
            result['avg_data_transfer_time'] = sum(s.get('data_transfer_time', 0) for s in iteration_stats) / len(iteration_stats)
            result['avg_model_forward_compute_time'] = sum(s.get('model_forward_compute_time', 0) for s in iteration_stats) / len(iteration_stats)
            result['avg_loss_computation_time'] = sum(s.get('loss_computation_time', 0) for s in iteration_stats) / len(iteration_stats)
            result['avg_backward_compute_time'] = sum(s.get('backward_compute_time', 0) for s in iteration_stats) / len(iteration_stats)
            result['avg_optimizer_step_time'] = sum(s.get('optimizer_step_time', 0) for s in iteration_stats) / len(iteration_stats)
            result['avg_optimizer_zero_grad_time'] = sum(s.get('optimizer_zero_grad_time', 0) for s in iteration_stats) / len(iteration_stats)

    return result

# =============================================================================
# Setup functions for each method
# =============================================================================

def setup_full_adamw():
    """Full fine-tuning with standard AdamW"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_meso():
    """Full fine-tuning with MeSO (MeSOAdamW)"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    optimizer = MeSOAdamW(
        model.parameters(),
        grad_hook=grad_hook,
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    return model, optimizer, tokenizer, grad_hook

def setup_full_galore():
    """Full fine-tuning with GaLore"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
    )
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_full_sgd_momentum():
    """Full fine-tuning with SGD + momentum"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
    )
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer

def setup_lora_sgd():
    """LoRA with SGD (no momentum)"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    optimizer = MeSOAdamW(
        model.parameters(),
        grad_hook=grad_hook,
        lr=5e-5,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    return model, optimizer, tokenizer, grad_hook

def setup_full_galore_gradient_checkpointing():
    """Full fine-tuning with GaLore + gradient checkpointing"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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

def setup_full_greats():
    """Full fine-tuning with GREATS data selection (requires compression for gradient computation)"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
    )
    model.train()

    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token
    sample_inputs = {k: v.to(device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression (needed for GREATS gradient computation)
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

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    # Use regular AdamW (not MeSOAdamW) since this is GREATS, not MeSO
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook

def setup_lora_greats():
    """LoRA fine-tuning with GREATS data selection (requires compression for gradient computation)"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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

    # Get layer names - for LoRA we need to target the base model's linear layers
    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token
    sample_inputs = {k: v.to(device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression (needed for GREATS gradient computation)
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

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    # Use regular AdamW (not MeSOAdamW) since this is GREATS, not MeSO
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook

def setup_full_greats_gradient_checkpointing():
    """Full fine-tuning with GREATS data selection + gradient checkpointing"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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

    # Setup compression (needed for GREATS gradient computation)
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

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    # Use regular AdamW (not MeSOAdamW) since this is GREATS, not MeSO
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook

def setup_lora_greats_gradient_checkpointing():
    """LoRA fine-tuning with GREATS data selection + gradient checkpointing"""
    model_kwargs = {
        'dtype': MODEL_DTYPE,
        'device_map': device
    }
    if USE_FLASH_ATTENTION:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        **model_kwargs
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

    # Get layer names - for LoRA we need to target the base model's linear layers
    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    tokenizer.pad_token = tokenizer.eos_token
    sample_inputs = {k: v.to(device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression (needed for GREATS gradient computation)
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

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(device),
        update_freq=200
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    # Use regular AdamW (not MeSOAdamW) since this is GREATS, not MeSO
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook

# =============================================================================
# Configuration and utility functions
# =============================================================================

def get_config_subfolder_name():
    """
    Generate a subfolder name based on current configuration.

    Returns:
        str: Subfolder name describing the configuration (e.g., "fp32", "bf16_flashattn")
    """
    dtype_map = {
        torch.float32: "fp32",
        torch.bfloat16: "bf16",
        torch.float16: "fp16"
    }
    dtype_str = dtype_map.get(MODEL_DTYPE, "unknown")

    if USE_FLASH_ATTENTION:
        return f"{dtype_str}_flashattn"
    else:
        return dtype_str

# =============================================================================
# Main execution
# =============================================================================

def get_all_methods():
    """
    Return list of all available test methods.
    Each entry is a tuple: (name, setup_fn, selection_method)
    where selection_method can be None (no data selection) or 'GREATS'/'GradNorm'/'MaxLoss'
    """
    return [
        ("Full + SGD", setup_full_sgd, None),
        ("Full + SGD + GC", setup_full_sgd_gradient_checkpointing, None),
        ("Full + SGD-momentum", setup_full_sgd_momentum, None),
        ("Full + SGD-momentum + GC", setup_full_sgd_momentum_gradient_checkpointing, None),
        ("Full + AdamW", setup_full_adamw, None),
        ("Full + AdamW + GC", setup_full_adamw_gradient_checkpointing, None),
        ("Full + MeSO", setup_full_meso, None),
        ("Full + MeSO + GC", setup_full_meso_gradient_checkpointing, None),
        ("Full + GaLore", setup_full_galore, None),
        ("Full + GaLore + GC", setup_full_galore_gradient_checkpointing, None),
        ("LoRA + SGD", setup_lora_sgd, None),
        ("LoRA + SGD + GC", setup_lora_sgd_gradient_checkpointing, None),
        ("LoRA + SGD-momentum", setup_lora_sgd_momentum, None),
        ("LoRA + SGD-momentum + GC", setup_lora_sgd_momentum_gradient_checkpointing, None),
        ("LoRA + AdamW", setup_lora_adamw, None),
        ("LoRA + AdamW + GC", setup_lora_adamw_gradient_checkpointing, None),
        # GREATS-enabled methods (require compression for gradient computation)
        ("Full + GREATS", setup_full_greats, 'GREATS'),
        ("Full + GREATS + GC", setup_full_greats_gradient_checkpointing, 'GREATS'),
        ("LoRA + GREATS", setup_lora_greats, 'GREATS'),
        ("LoRA + GREATS + GC", setup_lora_greats_gradient_checkpointing, 'GREATS'),
        ("MeSO + GREATS", setup_full_meso, 'GREATS'),
        ("MeSO + GREATS + GC", setup_full_meso_gradient_checkpointing, 'GREATS'),
    ]

def aggregate_results(results_dir='results'):
    """
    Aggregate results from individual test runs - simplified output.

    Looks for results in subfolders (e.g., results/fp32/, results/bf16_flashattn/)
    or directly in the specified directory for backwards compatibility.
    """
    results = []
    configs_found = []

    # Name mapping for cleaner display
    name_mapping = {
        'Full + SGD (no momentum)': 'Full + SGD',
        'Full + SGD (no momentum) + GC': 'Full + SGD + GC',
        'Full + SGD (momentum=0.9)': 'Full + SGD-momentum',
        'Full + SGD (momentum=0.9) + GC': 'Full + SGD-momentum + GC',
        'LoRA + SGD (no momentum)': 'LoRA + SGD',
        'LoRA + SGD (no momentum) + GC': 'LoRA + SGD + GC',
        'LoRA + SGD (momentum=0.9)': 'LoRA + SGD-momentum',
        'LoRA + SGD (momentum=0.9) + GC': 'LoRA + SGD-momentum + GC',
    }

    if not os.path.exists(results_dir):
        print(f"Error: Results directory '{results_dir}' not found")
        return

    # Look for config subfolders (fp32, bf16, bf16_flashattn, etc.)
    subfolders = [d for d in os.listdir(results_dir)
                  if os.path.isdir(os.path.join(results_dir, d))]

    if subfolders:
        # New structure: results/<config>/*.json
        print(f"Found configuration folders: {', '.join(subfolders)}")
        for subfolder in sorted(subfolders):
            subfolder_path = os.path.join(results_dir, subfolder)
            subfolder_results = []

            for filename in sorted(os.listdir(subfolder_path)):
                if filename.startswith('result_') and filename.endswith('.json'):
                    filepath = os.path.join(subfolder_path, filename)
                    with open(filepath, 'r') as f:
                        result = json.load(f)
                        # Update method name if it's in the mapping
                        if result['method'] in name_mapping:
                            result['method'] = name_mapping[result['method']]
                        # Add config information
                        result['config'] = subfolder
                        results.append(result)
                        subfolder_results.append(result)

            if subfolder_results:
                configs_found.append((subfolder, len(subfolder_results)))
    else:
        # Old structure or direct files: results/*.json (backwards compatibility)
        for filename in sorted(os.listdir(results_dir)):
            if filename.startswith('result_') and filename.endswith('.json'):
                filepath = os.path.join(results_dir, filename)
                with open(filepath, 'r') as f:
                    result = json.load(f)
                    # Update method name if it's in the mapping
                    if result['method'] in name_mapping:
                        result['method'] = name_mapping[result['method']]
                    result['config'] = 'default'
                    results.append(result)

        if results:
            configs_found.append(('default', len(results)))

    if not results:
        print(f"No result files found in '{results_dir}' or its subfolders")
        return

    # Display found configurations
    if configs_found:
        print(f"\nConfigurations found:")
        for config, count in configs_found:
            print(f"  {config}: {count} methods")

    print("\n" + "="*135)
    print(f"Benchmark Results (Total methods: {len(results)} | Results directory: {results_dir})")
    print("="*135)

    # Check if any results have selection time
    has_selection = any('avg_selection_total_time' in r for r in results)
    # Check if we have multiple configs
    has_multiple_configs = len(set(r.get('config', 'default') for r in results)) > 1

    # Single comprehensive table
    if has_selection:
        if has_multiple_configs:
            print(f"{'Config':<16} {'Method':<28} {'Peak Mem':<10} {'Sel(ms)':<9} {'Fwd(ms)':<9} {'Bwd(ms)':<9} {'Opt(ms)':<9} {'Total(ms)':<10} {'Throughput':<12}")
            print("-" * 135)
        else:
            print(f"{'Method':<28} {'Peak Mem':<10} {'Sel(ms)':<9} {'Fwd(ms)':<9} {'Bwd(ms)':<9} {'Opt(ms)':<9} {'Total(ms)':<10} {'Throughput':<12}")
            print("-" * 119)
    else:
        if has_multiple_configs:
            print(f"{'Config':<16} {'Method':<28} {'Peak Mem':<10} {'Fwd(ms)':<9} {'Bwd(ms)':<9} {'Opt(ms)':<9} {'Total(ms)':<10} {'Throughput':<12}")
            print("-" * 111)
        else:
            print(f"{'Method':<28} {'Peak Mem':<10} {'Fwd(ms)':<9} {'Bwd(ms)':<9} {'Opt(ms)':<9} {'Total(ms)':<10} {'Throughput':<12}")
            print("-" * 95)

    for r in results:
        config = r.get('config', 'default')
        peak_mem = f"{r['absolute_max']:.2f} GB"
        sel_ms = f"{r.get('avg_selection_total_time', 0)*1000:.1f}" if 'avg_selection_total_time' in r and r.get('avg_selection_total_time', 0) > 0 else "-"
        fwd_ms = f"{r.get('avg_forward_time', 0)*1000:.1f}" if 'avg_forward_time' in r else "N/A"
        bwd_ms = f"{r.get('avg_backward_time', 0)*1000:.1f}" if 'avg_backward_time' in r else "N/A"
        opt_ms = f"{r.get('avg_optimizer_time', 0)*1000:.1f}" if 'avg_optimizer_time' in r else "N/A"
        total_ms = f"{r.get('avg_iteration_time', 0)*1000:.1f}" if 'avg_iteration_time' in r else "N/A"
        throughput = f"{r.get('throughput', 0):.2f} samp/s" if 'throughput' in r else "N/A"

        if has_selection:
            if has_multiple_configs:
                print(f"{config:<16} {r['method']:<28} {peak_mem:<10} {sel_ms:<9} {fwd_ms:<9} {bwd_ms:<9} {opt_ms:<9} {total_ms:<10} {throughput:<12}")
            else:
                print(f"{r['method']:<28} {peak_mem:<10} {sel_ms:<9} {fwd_ms:<9} {bwd_ms:<9} {opt_ms:<9} {total_ms:<10} {throughput:<12}")
        else:
            if has_multiple_configs:
                print(f"{config:<16} {r['method']:<28} {peak_mem:<10} {fwd_ms:<9} {bwd_ms:<9} {opt_ms:<9} {total_ms:<10} {throughput:<12}")
            else:
                print(f"{r['method']:<28} {peak_mem:<10} {fwd_ms:<9} {bwd_ms:<9} {opt_ms:<9} {total_ms:<10} {throughput:<12}")

    if has_selection:
        if has_multiple_configs:
            print("="*135)
        else:
            print("="*119)
        print("Note: Sel(ms) = Data selection time (GREATS/GradNorm), includes val/train fwd/bwd + dot product")
    else:
        if has_multiple_configs:
            print("="*111)
        else:
            print("="*95)

    # Detailed selection breakdown for methods with selection
    if has_selection:
        # Check if any results have detailed selection breakdown
        has_detailed_selection = any(r.get('avg_selection_val_forward_time', 0) > 0 for r in results)

        if has_detailed_selection:
            print("\n" + "="*119)
            print("DETAILED SELECTION BREAKDOWN")
            print("="*119)
            print(f"{'Method':<28} {'Val Fwd':<10} {'Val Bwd':<10} {'Train Fwd':<11} {'Train Bwd':<11} {'Dot Prod':<11} {'Greedy':<10}")
            print("-" * 119)

            for r in results:
                if r.get('avg_selection_total_time', 0) > 0:
                    val_fwd = f"{r.get('avg_selection_val_forward_time', 0)*1000:.1f}ms" if 'avg_selection_val_forward_time' in r else "-"
                    val_bwd = f"{r.get('avg_selection_val_backward_time', 0)*1000:.1f}ms" if 'avg_selection_val_backward_time' in r else "-"
                    train_fwd = f"{r.get('avg_selection_train_forward_time', 0)*1000:.1f}ms" if 'avg_selection_train_forward_time' in r else "-"
                    train_bwd = f"{r.get('avg_selection_train_backward_time', 0)*1000:.1f}ms" if 'avg_selection_train_backward_time' in r else "-"
                    dotprod = f"{r.get('avg_selection_dotprod_compute_time', 0)*1000:.1f}ms" if 'avg_selection_dotprod_compute_time' in r else "-"
                    greedy = f"{r.get('avg_selection_greedy_time', 0)*1000:.1f}ms" if 'avg_selection_greedy_time' in r else "-"

                    print(f"{r['method']:<28} {val_fwd:<10} {val_bwd:<10} {train_fwd:<11} {train_bwd:<11} {dotprod:<11} {greedy:<10}")

            print("="*119)
            print("Selection Components:")
            print("  Val Fwd/Bwd:    Validation forward/backward pass (small batch)")
            print("  Train Fwd/Bwd:  Training forward/backward pass (per-sample gradients)")
            print("  Dot Prod:       Gradient similarity computation (layer-by-layer)")
            print("  Greedy:         Greedy selection algorithm (choose top samples)")
            print("="*119)

    # Save aggregated results
    with open('benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Max Memory Benchmark with Timing')
    parser.add_argument('--method', type=int, default=None,
                        help='Run specific method by index (0-20). If not specified, runs all methods.')
    parser.add_argument('--list', action='store_true',
                        help='List all available methods with their indices')
    parser.add_argument('--aggregate', action='store_true',
                        help='Aggregate results from individual runs')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Base directory to save results (default: results). Results will be saved in a subfolder based on configuration.')
    parser.add_argument('--memory-snapshot', action='store_true',
                        help='Enable memory snapshot recording for detailed memory profiling')

    args = parser.parse_args()

    # Override ENABLE_MEMORY_SNAPSHOT if --memory-snapshot flag is provided
    if args.memory_snapshot:
        globals()['ENABLE_MEMORY_SNAPSHOT'] = True

    methods = get_all_methods()

    if args.list:
        print("\nAvailable methods:")
        print("="*80)
        for i, (name, _, _) in enumerate(methods):
            print(f"{i:2d}: {name}")
        print("="*80)
        sys.exit(0)

    if args.aggregate:
        aggregate_results(args.output_dir)
        sys.exit(0)

    # Validate Flash Attention configuration
    if USE_FLASH_ATTENTION and MODEL_DTYPE == torch.float32:
        print("\n" + "="*80)
        print("⚠ WARNING: Flash Attention requires bfloat16 or float16!")
        print("="*80)
        print("Flash Attention is enabled but MODEL_DTYPE is set to float32.")
        print("Please set MODEL_DTYPE to torch.bfloat16 or torch.float16")
        print("Example: MODEL_DTYPE = torch.bfloat16")
        print("="*80 + "\n")
        sys.exit(1)

    print("\n" + "="*80)
    print("MAX MEMORY BENCHMARK")
    print("="*80)
    print(f"Model: Llama-3.2-1B")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_length}")
    print(f"Warmup iterations: {num_warmup_iterations} (not timed)")
    print(f"Timed iterations: {num_timed_iterations} (for throughput)")
    print(f"Total iterations: {num_iterations}")
    print(f"Precision: {MODEL_DTYPE}")
    print(f"Flash Attention: {'ENABLED' if USE_FLASH_ATTENTION else 'DISABLED'}")
    print("="*80 + "\n")

    # Create output directory with configuration subfolder if running individual tests
    if args.method is not None:
        config_subfolder = get_config_subfolder_name()
        output_dir = os.path.join(args.output_dir, config_subfolder)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}\n")

    if args.method is not None:
        # Run single method
        if args.method < 0 or args.method >= len(methods):
            print(f"Error: Method index {args.method} is out of range (0-{len(methods)-1})")
            print("Use --list to see available methods")
            sys.exit(1)

        method_name, setup_fn, selection_method = methods[args.method]
        print(f"Running method {args.method}: {method_name}\n")
        if selection_method:
            print(f"Data selection enabled: {selection_method}\n")
        result = run_method(method_name, setup_fn, selection_method)

        # Save individual result to config subfolder
        result_file = os.path.join(output_dir, f'result_{args.method:02d}.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResult saved to: {result_file}")

    else:
        # Run all methods (original behavior)
        results = []
        for method_name, setup_fn, selection_method in methods:
            result = run_method(method_name, setup_fn, selection_method)
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
