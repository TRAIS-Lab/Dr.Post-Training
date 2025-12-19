#!/usr/bin/env python
"""
Simplified Memory and Performance Benchmark

Key metrics:
1. Peak GPU memory (GB)
2. Throughput (samples/sec)
3. Avg time per iteration (ms)

Design principles:
- Uses actual training code (no re-implementation)
- Black-box timing (just wraps training step)
- Warm-up phase before measurement
- Minimal custom code
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import sys
import json
import argparse
import os
import time
import warnings
from typing import Optional, Callable, Any, List, Dict, Tuple
from dataclasses import dataclass, asdict

warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

from gradstream.hook import GradientHook
from gradstream.compressor import setup_model_compressors
from gradstream.optimizer import MeSOAdamW
from gradstream.utils import greedy_selection

from SFT.benchmark.GaLore.galore_torch import GaLoreAdamW


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # Model
    model_name: str = 'meta-llama/Llama-3.2-1B'
    dtype: str = 'bfloat16'
    use_flash_attention: bool = True

    # Training
    batch_size: int = 128
    seq_length: int = 256
    val_batch_size: int = 1

    # Benchmark
    num_warmup: int = 10
    num_iterations: int = 10

    # MeSO config
    meso_sparsifier_dim: int = 860
    meso_projector_dim: int = 739600
    meso_projector_type: str = "identity"
    meso_update_freq: int = 200

    # LoRA config
    lora_rank: int = 256
    lora_alpha: int = 1

    # GaLore config
    galore_rank: int = 128

    # GREATS config
    use_second_order: bool = False  # If True, use greedy selection with O(k*n) complexity

    # Device
    device: str = 'cuda'

    def get_torch_dtype(self):
        return {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    method_name: str
    peak_memory_gb: float
    avg_iteration_time_ms: float
    throughput_samples_per_sec: float
    total_time_sec: float
    num_iterations: int
    batch_size: int
    seq_length: int
    model_name: str
    # Memory breakdown
    memory_after_setup_gb: float = 0.0


# =============================================================================
# Dummy Dataset for Benchmarking
# =============================================================================

class DummyDataset(Dataset):
    """Dummy dataset that generates random tokens for benchmarking."""

    def __init__(self, tokenizer, seq_length: int, size: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.size = size
        # Pre-tokenize a dummy sentence
        self.dummy_text = "This is a test sentence for memory and performance benchmarking. " * 20

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.dummy_text,
            return_tensors='pt',
            padding='max_length',
            max_length=self.seq_length,
            truncation=True
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': tokens['input_ids'].squeeze(0).clone(),
        }


# =============================================================================
# Benchmark Class
# =============================================================================

class Benchmark:
    """
    Simple benchmark wrapper for training methods.

    Usage:
        config = BenchmarkConfig(batch_size=64, num_warmup=10, num_iterations=20)
        bench = Benchmark(config)

        result = bench.run(
            method_name="Full+AdamW",
            setup_fn=setup_full_adamw,
            step_fn=step_standard,
        )
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def _reset_memory(self):
        """Reset CUDA cache and peak memory stats."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _get_peak_memory_gb(self) -> float:
        """Get peak GPU memory in GB."""
        return torch.cuda.max_memory_allocated() / 1024**3

    def _get_current_memory_gb(self) -> float:
        """Get current GPU memory in GB."""
        return torch.cuda.memory_allocated() / 1024**3

    def _sync(self):
        """Synchronize CUDA for accurate timing."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def run(
        self,
        method_name: str,
        setup_fn: Callable[['BenchmarkConfig'], Tuple],
        step_fn: Callable,
    ) -> BenchmarkResult:
        """
        Run the benchmark for a single method.

        Args:
            method_name: Name of the method being benchmarked
            setup_fn: Function that takes config and returns (model, optimizer, tokenizer, *extras)
            step_fn: Function that performs one training step: step_fn(model, optimizer, batch, *extras)

        Returns:
            BenchmarkResult with timing and memory metrics
        """
        print("=" * 80)
        print(f"Benchmarking: {method_name}")
        print("=" * 80)

        self._reset_memory()

        # Setup
        print("Setting up model and optimizer...")
        self._sync()
        setup_start = time.time()

        setup_result = setup_fn(self.config)
        model, optimizer, tokenizer = setup_result[:3]
        extras = setup_result[3:] if len(setup_result) > 3 else ()

        self._sync()
        setup_time = time.time() - setup_start
        print(f"Setup time: {setup_time:.2f}s")

        memory_after_setup = self._get_current_memory_gb()
        print(f"Memory after setup: {memory_after_setup:.3f} GB")

        # Create dataloader
        dataset = DummyDataset(tokenizer, self.config.seq_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        data_iter = iter(dataloader)

        # Reset memory stats before training
        self._reset_memory()

        # Warmup phase
        print(f"\nWarmup: {self.config.num_warmup} iterations...")
        for i in range(self.config.num_warmup):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            step_fn(model, optimizer, batch, *extras)

            if (i + 1) % 5 == 0:
                print(f"  Warmup {i + 1}/{self.config.num_warmup} done")

        # Reset memory stats after warmup
        self._sync()
        self._reset_memory()

        # Timed phase
        print(f"\nTiming: {self.config.num_iterations} iterations...")
        self._sync()
        start_time = time.time()

        for i in range(self.config.num_iterations):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            step_fn(model, optimizer, batch, *extras)

            if (i + 1) % 5 == 0:
                print(f"  Iteration {i + 1}/{self.config.num_iterations} done")

        self._sync()
        total_time = time.time() - start_time
        peak_memory = self._get_peak_memory_gb()

        # Calculate metrics
        avg_iteration_time_ms = (total_time / self.config.num_iterations) * 1000
        throughput = (self.config.num_iterations * self.config.batch_size) / total_time

        # Print results
        print(f"\n{'=' * 80}")
        print(f"Results: {method_name}")
        print(f"{'=' * 80}")
        print(f"Peak Memory:     {peak_memory:.3f} GB")
        print(f"Avg Time/Iter:   {avg_iteration_time_ms:.1f} ms")
        print(f"Throughput:      {throughput:.2f} samples/sec")
        print(f"Total Time:      {total_time:.2f}s ({self.config.num_iterations} iterations)")
        print(f"{'=' * 80}\n")

        # Cleanup
        del model, optimizer
        torch.cuda.empty_cache()

        return BenchmarkResult(
            method_name=method_name,
            peak_memory_gb=peak_memory,
            avg_iteration_time_ms=avg_iteration_time_ms,
            throughput_samples_per_sec=throughput,
            total_time_sec=total_time,
            num_iterations=self.config.num_iterations,
            batch_size=self.config.batch_size,
            seq_length=self.config.seq_length,
            model_name=self.config.model_name,
            memory_after_setup_gb=memory_after_setup,
        )


# =============================================================================
# Step Functions (Training Logic)
# =============================================================================

def step_standard(model, optimizer, batch):
    """Standard training step for AdamW, SGD, GaLore, etc."""
    model.train()
    optimizer.zero_grad()
    # Use autocast to keep loss computation in bfloat16 (avoids OOM from logits.float())
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item()


def step_meso(model, optimizer, batch, grad_hook):
    """Training step for MeSO (without data selection)."""
    model.train()
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss
    loss.backward()
    optimizer.step()
    return loss.item()


class SelectionStepHelper:
    """
    Helper class for step functions that need data selection.

    This creates and manages a validation dataloader for selection.
    """

    def __init__(self, tokenizer, seq_length: int, batch_size: int, device: str):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.device = device
        self._val_iter = None
        self._val_dataloader = None

    def _create_val_dataloader(self):
        """Create validation dataloader for selection."""
        dataset = DummyDataset(self.tokenizer, self.seq_length, size=1000)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_val_batch(self) -> Dict[str, torch.Tensor]:
        """Get a validation batch for selection."""
        if self._val_dataloader is None:
            self._val_dataloader = self._create_val_dataloader()
            self._val_iter = iter(self._val_dataloader)

        try:
            batch = next(self._val_iter)
        except StopIteration:
            self._val_iter = iter(self._val_dataloader)
            batch = next(self._val_iter)

        return {k: v.to(self.device) for k, v in batch.items()}


def make_step_greats(selection_helper: SelectionStepHelper, selection_frac: float = 0.5, use_second_order: bool = False):
    """
    Create a step function for pure GREATS selection (with standard optimizer).

    This is a two-pass approach:
    1. First pass: Compute per-sample gradient scores using compression
    2. Second pass: Standard training on selected samples

    Args:
        selection_helper: SelectionStepHelper instance for validation batches
        selection_frac: Fraction of samples to select
        use_second_order: If True, use greedy selection with second-order interactions.

    Returns:
        Step function for GREATS + standard optimizer
    """
    def step_fn(model, optimizer, batch, grad_hook):
        model.train()

        # Get validation batch
        val_batch = selection_helper.get_val_batch()
        train_batch_size = batch['input_ids'].shape[0]

        # Merge batches: [train_samples, val_samples]
        merged_batch = {}
        for key in batch.keys():
            if key in val_batch:
                merged_batch[key] = torch.cat([batch[key], val_batch[key]], dim=0)
            else:
                merged_batch[key] = batch[key]

        lr = optimizer.param_groups[0].get("lr", 5e-5)

        # Pass 1: Compute selection scores using compressed gradients
        grad_hook.setup_selection(
            train_batch_size=train_batch_size,
            selection_method='GREATS',
            selection_frac=selection_frac,
            lr=lr,
            compute_scores_only=True,
            use_second_order=use_second_order
        )

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(**merged_batch)
            loss_for_scoring = outputs.loss
        loss_for_scoring.backward()

        # Get selected indices
        selected_indices = grad_hook.selection_state.get_selected_indices()
        grad_hook.clear_selection()

        # Pass 2: Standard training on selected samples
        filtered_batch = {
            'input_ids': batch['input_ids'][selected_indices],
            'attention_mask': batch['attention_mask'][selected_indices],
            'labels': batch['labels'][selected_indices]
        }

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(**filtered_batch)
            loss = outputs.loss
        loss.backward()
        optimizer.step()

        return loss.item()

    return step_fn


def make_step_meso_greats(selection_helper: SelectionStepHelper, selection_frac: float = 0.5, use_second_order: bool = False, selection_mode: str = 'per_layer'):
    """
    Create a step function for MeSO + GREATS selection.

    Args:
        selection_helper: SelectionStepHelper instance for validation batches
        selection_frac: Fraction of samples to select
        use_second_order: If True, use greedy selection with second-order interactions.
                         If False (default), use simple top-k selection (~200x faster).
        selection_mode: 'per_layer' (default) or 'full'.
                       per_layer: Each layer independently selects samples (single pass).
                       full: Accumulates scores across all layers, selects globally,
                             then does second pass for compressed gradient computation.

    Returns:
        Step function for MeSO + GREATS
    """
    def step_fn(model, optimizer, batch, grad_hook):
        model.train()

        # Get validation batch
        val_batch = selection_helper.get_val_batch()
        train_batch_size = batch['input_ids'].shape[0]

        # Merge batches: [train_samples, val_samples]
        merged_batch = {}
        for key in batch.keys():
            if key in val_batch:
                merged_batch[key] = torch.cat([batch[key], val_batch[key]], dim=0)
            else:
                merged_batch[key] = batch[key]

        # Get learning rate
        lr = optimizer.param_groups[0].get("lr", 5e-5)

        if selection_mode == 'per_layer':
            # Per-layer selection: single pass with on-the-fly selection and aggregation
            # Each layer independently selects samples based on that layer's gradient alignment
            grad_hook.setup_selection(
                train_batch_size=train_batch_size,
                selection_method='GREATS',
                selection_frac=selection_frac,
                lr=lr,
                compute_scores_only=False,
                use_second_order=use_second_order
            )

            # Forward + backward on merged batch
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**merged_batch)
                loss = outputs.loss
            loss.backward()

            # Clear selection state
            grad_hook.clear_selection()

        else:
            # Full sample-level selection: two passes
            # This approach doesn't use the block-diagonal approximation

            # Pass 1: Accumulate scores across all layers
            grad_hook.setup_selection(
                train_batch_size=train_batch_size,
                selection_method='GREATS',
                selection_frac=selection_frac,
                lr=lr,
                compute_scores_only=True,  # Only accumulate scores
                use_second_order=use_second_order
            )

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**merged_batch)
                loss_for_scoring = outputs.loss
            loss_for_scoring.backward()

            # Get globally selected indices based on accumulated scores
            selected_indices = grad_hook.selection_state.get_selected_indices()
            grad_hook.clear_selection()

            # Pass 2: Compute compressed gradients on selected samples
            filtered_batch = {
                'input_ids': batch['input_ids'][selected_indices],
                'attention_mask': batch['attention_mask'][selected_indices],
                'labels': batch['labels'][selected_indices]
            }

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(**filtered_batch)
                loss = outputs.loss
            loss.backward()

        # Optimizer step
        optimizer.step()

        return loss.item()

    return step_fn


# =============================================================================
# Setup Functions
# =============================================================================

def setup_full_adamw(config: BenchmarkConfig):
    """Full fine-tuning with AdamW."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_sgd(config: BenchmarkConfig):
    """Full fine-tuning with SGD (no momentum)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_sgd_momentum(config: BenchmarkConfig):
    """Full fine-tuning with SGD + momentum."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_sgd_gc(config: BenchmarkConfig):
    """Full fine-tuning with SGD (no momentum) + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_sgd_momentum_gc(config: BenchmarkConfig):
    """Full fine-tuning with SGD + momentum + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_galore(config: BenchmarkConfig):
    """Full fine-tuning with GaLore."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
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
        {'params': galore_params, 'rank': config.galore_rank, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
    ]

    optimizer = GaLoreAdamW(param_groups, lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_lora_adamw(config: BenchmarkConfig):
    """LoRA with AdamW."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_lora_sgd(config: BenchmarkConfig):
    """LoRA with SGD (no momentum)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_lora_sgd_momentum(config: BenchmarkConfig):
    """LoRA with SGD + momentum."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_lora_adamw_gc(config: BenchmarkConfig):
    """LoRA with AdamW + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_lora_sgd_gc(config: BenchmarkConfig):
    """LoRA with SGD (no momentum) + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_lora_sgd_momentum_gc(config: BenchmarkConfig):
    """LoRA with SGD + momentum + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_meso(config: BenchmarkConfig):
    """Full fine-tuning with MeSO."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Sample input for compressor setup
    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression
    sparsifier_kwargs = {
        "proj_dim": config.meso_sparsifier_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": config.meso_projector_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": config.meso_projector_type,
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=config.meso_update_freq
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
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


def setup_full_meso_gc(config: BenchmarkConfig):
    """Full fine-tuning with MeSO + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    sparsifier_kwargs = {
        "proj_dim": config.meso_sparsifier_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": config.meso_projector_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": config.meso_projector_type,
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=config.meso_update_freq
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
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


def setup_full_adamw_gc(config: BenchmarkConfig):
    """Full fine-tuning with AdamW + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_galore_gc(config: BenchmarkConfig):
    """Full fine-tuning with GaLore + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

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
        {'params': galore_params, 'rank': config.galore_rank, 'update_proj_gap': 200, 'scale': 0.25, 'proj_type': 'std'}
    ]

    optimizer = GaLoreAdamW(param_groups, lr=5e-5)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, optimizer, tokenizer


def setup_full_greats(config: BenchmarkConfig):
    """Full fine-tuning with GREATS data selection (uses compression for gradient computation, AdamW optimizer)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression (needed for GREATS gradient computation)
    sparsifier_kwargs = {
        "proj_dim": config.meso_sparsifier_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": config.meso_projector_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": config.meso_projector_type,
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=config.meso_update_freq
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    # Use regular AdamW (not MeSOAdamW) - GREATS is just for data selection
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook


def setup_full_greats_gc(config: BenchmarkConfig):
    """Full fine-tuning with GREATS data selection + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    sparsifier_kwargs = {
        "proj_dim": config.meso_sparsifier_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": config.meso_projector_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": config.meso_projector_type,
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=config.meso_update_freq
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook


def setup_lora_greats(config: BenchmarkConfig):
    """LoRA with GREATS data selection."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model = model.to(config.get_torch_dtype())
    model.train()

    # Get layer names for LoRA layers only
    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and ('lora_A' in n or 'lora_B' in n)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    sparsifier_kwargs = {
        "proj_dim": config.meso_sparsifier_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": config.meso_projector_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": config.meso_projector_type,
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=config.meso_update_freq
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook


def setup_lora_greats_gc(config: BenchmarkConfig):
    """LoRA with GREATS data selection + gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_config)
    model = model.to(config.get_torch_dtype())
    model.train()

    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and ('lora_A' in n or 'lora_B' in n)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    sparsifier_kwargs = {
        "proj_dim": config.meso_sparsifier_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": config.meso_projector_dim,
        "proj_max_batch_size": config.batch_size,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": config.meso_projector_type,
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=config.meso_update_freq
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, optimizer, tokenizer, grad_hook


# =============================================================================
# Method Registry
# =============================================================================

# Methods that use standard step function
STANDARD_METHODS = {
    'full_adamw': setup_full_adamw,
    'full_sgd': setup_full_sgd,
    'full_sgd_momentum': setup_full_sgd_momentum,
    'full_galore': setup_full_galore,
    'lora_adamw': setup_lora_adamw,
    'lora_sgd': setup_lora_sgd,
    'lora_sgd_momentum': setup_lora_sgd_momentum,
    'full_adamw_gc': setup_full_adamw_gc,
    'full_sgd_gc': setup_full_sgd_gc,
    'full_sgd_momentum_gc': setup_full_sgd_momentum_gc,
    'full_galore_gc': setup_full_galore_gc,
    'lora_adamw_gc': setup_lora_adamw_gc,
    'lora_sgd_gc': setup_lora_sgd_gc,
    'lora_sgd_momentum_gc': setup_lora_sgd_momentum_gc,
}

# Methods that use MeSO (without selection)
MESO_METHODS = {
    'full_meso': setup_full_meso,
    'full_meso_gc': setup_full_meso_gc,
}

# Methods that use MeSO with data selection
# Format: method_name -> (setup_fn, selection_frac, selection_mode)
MESO_SELECTION_METHODS = {
    'full_meso_greats': (setup_full_meso, 0.5, 'per_layer'),  # Per-layer selection (default)
    'full_meso_greats_gc': (setup_full_meso_gc, 0.5, 'per_layer'),
    # Full sample-level selection variants (two-pass approach)
    'full_meso_greats_full': (setup_full_meso, 0.5, 'full'),  # Full selection
    'full_meso_greats_full_gc': (setup_full_meso_gc, 0.5, 'full'),
}

# Methods that use GREATS selection with standard optimizer (AdamW)
# These use compression for gradient computation but regular AdamW for optimization
# Format: method_name -> (setup_fn, selection_frac)
GREATS_METHODS = {
    'full_greats': (setup_full_greats, 0.5),
    'full_greats_gc': (setup_full_greats_gc, 0.5),
    'lora_greats': (setup_lora_greats, 0.5),
    'lora_greats_gc': (setup_lora_greats_gc, 0.5),
}

# Combined list for CLI help
ALL_METHODS = (list(STANDARD_METHODS.keys()) + list(MESO_METHODS.keys()) +
               list(MESO_SELECTION_METHODS.keys()) + list(GREATS_METHODS.keys()))


# =============================================================================
# CLI Interface
# =============================================================================

def run_benchmark(methods: List[str], config: BenchmarkConfig, output_file: Optional[str] = None) -> List[BenchmarkResult]:
    """Run benchmarks for specified methods."""
    bench = Benchmark(config)
    results = []

    for method_name in methods:
        # Check which category this method belongs to
        if method_name in STANDARD_METHODS:
            setup_fn = STANDARD_METHODS[method_name]
            step_fn = step_standard
        elif method_name in MESO_METHODS:
            setup_fn = MESO_METHODS[method_name]
            step_fn = step_meso
        elif method_name in MESO_SELECTION_METHODS:
            # Handle MeSO with GREATS selection - needs special setup
            base_setup_fn, selection_frac, sel_mode = MESO_SELECTION_METHODS[method_name]

            # We need to wrap this to create the selection helper after setup
            def make_selection_setup_and_step(base_setup, sel_frac, mode):
                def wrapped_setup(cfg):
                    model, optimizer, tokenizer, grad_hook = base_setup(cfg)
                    # Create selection helper with the tokenizer
                    helper = SelectionStepHelper(tokenizer, cfg.seq_length, cfg.val_batch_size, cfg.device)
                    # Create the step function with selection_mode
                    step = make_step_meso_greats(helper, sel_frac, use_second_order=cfg.use_second_order, selection_mode=mode)
                    return model, optimizer, tokenizer, grad_hook, step
                return wrapped_setup

            setup_fn = make_selection_setup_and_step(base_setup_fn, selection_frac, sel_mode)

            # For selection methods, the step function is returned as part of setup
            def selection_step_wrapper(model, optimizer, batch, grad_hook, step_fn):
                return step_fn(model, optimizer, batch, grad_hook)

            step_fn = selection_step_wrapper
        elif method_name in GREATS_METHODS:
            # Handle pure GREATS with standard optimizer (AdamW)
            base_setup_fn, selection_frac = GREATS_METHODS[method_name]

            def make_greats_setup_and_step(base_setup, sel_frac):
                def wrapped_setup(cfg):
                    model, optimizer, tokenizer, grad_hook = base_setup(cfg)
                    helper = SelectionStepHelper(tokenizer, cfg.seq_length, cfg.val_batch_size, cfg.device)
                    step = make_step_greats(helper, sel_frac, use_second_order=cfg.use_second_order)
                    return model, optimizer, tokenizer, grad_hook, step
                return wrapped_setup

            setup_fn = make_greats_setup_and_step(base_setup_fn, selection_frac)

            def greats_step_wrapper(model, optimizer, batch, grad_hook, step_fn):
                return step_fn(model, optimizer, batch, grad_hook)

            step_fn = greats_step_wrapper
        else:
            print(f"Unknown method: {method_name}. Available: {ALL_METHODS}")
            continue

        try:
            result = bench.run(
                method_name=method_name,
                setup_fn=setup_fn,
                step_fn=step_fn,
            )
            results.append(result)
        except Exception as e:
            print(f"Error running {method_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print summary table
    if results:
        print("\n" + "=" * 110)
        print("BENCHMARK SUMMARY")
        print(f"Model: {config.model_name} | Batch: {config.batch_size} | Val Batch: {config.val_batch_size} | Seq: {config.seq_length} | Dtype: {config.dtype}")
        print("=" * 110)
        print(f"{'Method':<28} {'Peak Mem':<12} {'Setup Mem':<12} {'Time/Iter':<14} {'Throughput':<16} {'Total Time':<12}")
        print("-" * 110)
        for r in results:
            peak_mem = f"{r.peak_memory_gb:.2f} GB"
            setup_mem = f"{r.memory_after_setup_gb:.2f} GB"
            time_iter = f"{r.avg_iteration_time_ms:.1f} ms"
            throughput = f"{r.throughput_samples_per_sec:.2f} samp/s"
            total_time = f"{r.total_time_sec:.1f} s"
            print(f"{r.method_name:<28} {peak_mem:<12} {setup_mem:<12} {time_iter:<14} {throughput:<16} {total_time:<12}")
        print("=" * 110)

    # Save results
    if output_file:
        results_dict = {
            'config': asdict(config),
            'results': [asdict(r) for r in results],
        }
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return results


def append_results(results: List[BenchmarkResult], config: BenchmarkConfig, results_file: str):
    """Append results to a JSONL file (one JSON object per line)."""
    with open(results_file, 'a') as f:
        for r in results:
            entry = {
                'config': asdict(config),
                'result': asdict(r),
            }
            f.write(json.dumps(entry) + '\n')


def print_summary_from_file(results_file: str):
    """Read results from JSONL file and print a summary table."""
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return

    results = []
    config = None
    with open(results_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                results.append(entry['result'])
                if config is None:
                    config = entry['config']

    if not results:
        print("No results found in file.")
        return

    print("\n" + "=" * 110)
    print("BENCHMARK SUMMARY (Aggregated)")
    print(f"Model: {config['model_name']} | Batch: {config['batch_size']} | Val Batch: {config['val_batch_size']} | Seq: {config['seq_length']} | Dtype: {config['dtype']}")
    print("=" * 110)
    print(f"{'Method':<28} {'Peak Mem':<12} {'Setup Mem':<12} {'Time/Iter':<14} {'Throughput':<16} {'Total Time':<12}")
    print("-" * 110)
    for r in results:
        peak_mem = f"{r['peak_memory_gb']:.2f} GB"
        setup_mem = f"{r['memory_after_setup_gb']:.2f} GB"
        time_iter = f"{r['avg_iteration_time_ms']:.1f} ms"
        throughput = f"{r['throughput_samples_per_sec']:.2f} samp/s"
        total_time = f"{r['total_time_sec']:.1f} s"
        print(f"{r['method_name']:<28} {peak_mem:<12} {setup_mem:<12} {time_iter:<14} {throughput:<16} {total_time:<12}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description='Simplified Memory and Performance Benchmark')

    # Method selection
    parser.add_argument('--methods', nargs='+', default=['full_adamw', 'full_meso'],
                        help=f'Methods to benchmark. Available: {ALL_METHODS}')
    parser.add_argument('--all', action='store_true', help='Run all methods')
    parser.add_argument('--list', action='store_true', help='List all available methods and exit')

    # Model config
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Model name')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--no-flash-attention', action='store_true', help='Disable flash attention')

    # Training config
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Validation batch size for GREATS selection (default: 1)')

    # Benchmark config
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-iterations', type=int, default=10)

    # GREATS config
    parser.add_argument('--use-second-order', action='store_true',
                        help='Use second-order interactions in GREATS (greedy selection, slower)')

    # Output
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--results-file', type=str, default=None,
                        help='JSONL file to append results (for aggregating across runs)')
    parser.add_argument('--print-summary', type=str, metavar='FILE',
                        help='Print summary table from results file and exit')

    args = parser.parse_args()

    # If --print-summary is specified, just print and exit
    if args.print_summary:
        print_summary_from_file(args.print_summary)
        return

    # If --list is specified, list all methods and exit
    if args.list:
        print("\nAvailable methods:")
        print("=" * 70)
        print(f"{'Index':<6} {'Method Name':<35} {'Category'}")
        print("-" * 70)
        idx = 0
        for name in STANDARD_METHODS:
            print(f"{idx:<6} {name:<35} Standard")
            idx += 1
        for name in MESO_METHODS:
            print(f"{idx:<6} {name:<35} MeSO")
            idx += 1
        for name in MESO_SELECTION_METHODS:
            print(f"{idx:<6} {name:<35} MeSO+Selection")
            idx += 1
        for name in GREATS_METHODS:
            print(f"{idx:<6} {name:<35} GREATS")
            idx += 1
        print("=" * 70)
        print(f"Total: {len(ALL_METHODS)} methods")
        return

    config = BenchmarkConfig(
        model_name=args.model,
        dtype=args.dtype,
        use_flash_attention=not args.no_flash_attention,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        val_batch_size=args.val_batch_size,
        num_warmup=args.num_warmup,
        num_iterations=args.num_iterations,
        use_second_order=args.use_second_order,
    )

    methods = ALL_METHODS if args.all else args.methods

    print("Benchmark Configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Dtype: {config.dtype}")
    print(f"  Flash Attention: {config.use_flash_attention}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Val Batch Size: {config.val_batch_size}")
    print(f"  Seq Length: {config.seq_length}")
    print(f"  Warmup Iterations: {config.num_warmup}")
    print(f"  Timed Iterations: {config.num_iterations}")
    print(f"  Use Second Order: {config.use_second_order}")
    print(f"  Methods: {methods}")
    print()

    results = run_benchmark(methods, config, args.output)

    # Append to results file if specified (for aggregating across shell script runs)
    if args.results_file and results:
        append_results(results, config, args.results_file)
        print(f"Results appended to: {args.results_file}")


if __name__ == '__main__':
    main()
