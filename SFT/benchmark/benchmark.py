#!/usr/bin/env python
"""
Memory and Performance Benchmark for Gradient Streaming

Key metrics:
1. Peak GPU memory (GB)
2. Throughput (samples/sec)
3. Avg time per iteration (ms)

Naming convention follows the experimental configurations:
  {selection}-{compression}-{training_type}

  selection: NA (baseline), Streaming (per-layer), GREATS (global)
  compression: NA (standard optimizer), LoGra (MeSO optimizer)
  training_type: full, lora

Note: GraSS compression is also available but LoGra is used in default experiments.

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

    # LoRA config
    lora_rank: int = 256
    lora_alpha: int = 1

    # Selection config
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


def make_step_streaming(selection_helper: SelectionStepHelper, selection_frac: float = 0.5, use_second_order: bool = False, has_compression: bool = True):
    """
    Create a step function for Streaming (per-layer) selection.

    Streaming mode: Single-pass approach with per-layer selection.
    Each layer independently selects samples based on that layer's gradient alignment.

    Args:
        selection_helper: SelectionStepHelper instance for validation batches
        selection_frac: Fraction of samples to select
        use_second_order: If True, use greedy selection with second-order interactions.
        has_compression: If True, uses MeSO optimizer (compressed gradients stored in hook).
                        If False, uses standard optimizer (full gradients returned).

    Returns:
        Step function for Streaming selection
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

        # Streaming mode: single pass with per-layer selection and aggregation
        grad_hook.setup_selection(
            train_batch_size=train_batch_size,
            selection_method='Streaming',
            selection_frac=selection_frac,
            lr=lr,
            compute_scores_only=False,  # Per-layer selection and aggregation
            use_second_order=use_second_order
        )

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(**merged_batch)
            loss = outputs.loss
        loss.backward()

        grad_hook.clear_selection()
        optimizer.step()

        return loss.item()

    return step_fn


def make_step_greats(selection_helper: SelectionStepHelper, selection_frac: float = 0.5, use_second_order: bool = False, has_compression: bool = True):
    """
    Create a step function for GREATS (global) selection.

    GREATS mode: Two-pass approach with global selection.
    1. First pass: Accumulate scores across all layers
    2. Second pass: Compute gradients on globally selected samples

    Args:
        selection_helper: SelectionStepHelper instance for validation batches
        selection_frac: Fraction of samples to select
        use_second_order: If True, use greedy selection with second-order interactions.
        has_compression: If True, uses MeSO optimizer (compressed gradients stored in hook).
                        If False, uses standard optimizer (full gradients returned).

    Returns:
        Step function for GREATS selection
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

        # Pass 1: Compute selection scores (accumulate across all layers)
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

        # Get globally selected indices
        selected_indices = grad_hook.selection_state.get_selected_indices()
        grad_hook.clear_selection()

        # Pass 2: Compute gradients on selected samples
        filtered_batch = {
            'input_ids': batch['input_ids'][selected_indices],
            'attention_mask': batch['attention_mask'][selected_indices],
            'labels': batch['labels'][selected_indices]
        }

        optimizer.zero_grad()

        if not has_compression:
            # GREATS-NA: Disable hooks for full gradient computation
            grad_hook.disable_hooks()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(**filtered_batch)
            loss = outputs.loss
        loss.backward()

        if not has_compression:
            grad_hook.enable_hooks()

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


def _setup_compression(model, config: BenchmarkConfig, use_meso_optimizer: bool = True):
    """
    Helper to set up compression (GraSS/LoGra) for a model.

    Args:
        model: The model to set up compression for
        config: Benchmark configuration
        use_meso_optimizer: If True, return MeSOAdamW optimizer. If False, return standard AdamW.

    Returns:
        Tuple of (grad_hook, optimizer)
    """
    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    # Setup compression (GraSS uses random_mask sparsifier + SJLT projection)
    # Note: proj_dim is the factorized dimension (e.g., 1024 for random_mask-1024*1024)
    sparsifier_kwargs = {
        "proj_dim": 1024,  # Factorized: 1024*1024, so each dimension is 1024
        "proj_max_batch_size": 64,  # Match train.py default
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": 256,  # SJLT projection dimension
        "proj_max_batch_size": 64,  # Match train.py default
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "sjlt",
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=200  # Default update frequency
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    if use_meso_optimizer:
        optimizer = MeSOAdamW(
            model.parameters(),
            grad_hook=grad_hook,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return grad_hook, optimizer, tokenizer


def setup_NA_GraSS_full(config: BenchmarkConfig):
    """NA-GraSS-full: MeSO only (no selection, compressed optimizer)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def setup_NA_GraSS_full_gc(config: BenchmarkConfig):
    """NA-GraSS-full with gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def setup_Streaming_NA_full(config: BenchmarkConfig):
    """Streaming-NA-full: Per-layer selection with full gradients (standard optimizer)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    # Need grad_hook for selection scoring but no compression, use standard optimizer
    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=False)
    # Clear compressors to use full gradients for the actual update
    grad_hook.compressors = [None] * len(grad_hook.layer_names)

    return model, optimizer, tokenizer, grad_hook


def setup_Streaming_NA_full_gc(config: BenchmarkConfig):
    """Streaming-NA-full with gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=False)
    grad_hook.compressors = [None] * len(grad_hook.layer_names)

    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_NA_full(config: BenchmarkConfig):
    """GREATS-NA-full: Global selection with full gradients (standard optimizer)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    # Need grad_hook with compressors for scoring, but use standard optimizer
    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=False)

    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_NA_full_gc(config: BenchmarkConfig):
    """GREATS-NA-full with gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=False)

    return model, optimizer, tokenizer, grad_hook


def setup_Streaming_GraSS_full(config: BenchmarkConfig):
    """Streaming-GraSS-full: Per-layer selection + MeSO (GraSS)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_GraSS_full(config: BenchmarkConfig):
    """GREATS-GraSS-full: Global selection + MeSO (GraSS)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    grad_hook, optimizer, tokenizer = _setup_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def _setup_logra_compression(model, config: BenchmarkConfig, use_meso_optimizer: bool = True):
    """
    Helper to set up LoGra compression for a model.

    LoGra uses Gaussian random projection (normal) without additional projection.
    This is different from GraSS which uses random_mask + SJLT projection.

    Args:
        model: The model to set up compression for
        config: Benchmark configuration
        use_meso_optimizer: If True, return MeSOAdamW optimizer. If False, return standard AdamW.

    Returns:
        Tuple of (grad_hook, optimizer, tokenizer)
    """
    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    # LoGra uses Gaussian projection (normal) without additional projection
    # train.sh: sparsification="normal-512*512" means proj_dim=512 (factorized, applied to both dimensions)
    # The actual dimension is 512, not 512*512 - see train.py parsing logic
    sparsifier_kwargs = {
        "proj_dim": 512,  # Factorized: 512*512, so each dimension is 512
        "proj_max_batch_size": 64,  # Match train.py default
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "normal",  # Gaussian random projection
    }

    # No projector for LoGra (identity)
    projector_kwargs = {
        "proj_dim": -1,  # Identity projection
        "proj_max_batch_size": 64,
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "identity",
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=200  # Default update frequency
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    if use_meso_optimizer:
        optimizer = MeSOAdamW(
            model.parameters(),
            grad_hook=grad_hook,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return grad_hook, optimizer, tokenizer


def setup_NA_LoGra_full(config: BenchmarkConfig):
    """NA-LoGra-full: MeSO only with LoGra compression (no selection)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    grad_hook, optimizer, tokenizer = _setup_logra_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def setup_Streaming_LoGra_full(config: BenchmarkConfig):
    """Streaming-LoGra-full: Per-layer selection + MeSO (LoGra)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    grad_hook, optimizer, tokenizer = _setup_logra_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def setup_Streaming_LoGra_full_gc(config: BenchmarkConfig):
    """Streaming-LoGra-full with gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    grad_hook, optimizer, tokenizer = _setup_logra_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_LoGra_full(config: BenchmarkConfig):
    """GREATS-LoGra-full: Global selection + MeSO (LoGra)."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    grad_hook, optimizer, tokenizer = _setup_logra_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_LoGra_full_gc(config: BenchmarkConfig):
    """GREATS-LoGra-full with gradient checkpointing."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.gradient_checkpointing_enable()
    model.train()

    grad_hook, optimizer, tokenizer = _setup_logra_compression(model, config, use_meso_optimizer=True)

    return model, optimizer, tokenizer, grad_hook


def _setup_lora_with_grad_hook(config: BenchmarkConfig, use_meso_optimizer: bool = True, use_gc: bool = False):
    """
    Helper to set up LoRA model with gradient hook for data selection.

    Sets up the grad_hook infrastructure needed for selection scoring.
    Note: For NA (no compression) variants, the compressors should be cleared
    after calling this function to use full gradients.

    Args:
        config: Benchmark configuration
        use_meso_optimizer: If True, return MeSOAdamW optimizer. If False, return standard AdamW.
        use_gc: If True, enable gradient checkpointing.

    Returns:
        Tuple of (model, grad_hook, optimizer, tokenizer)
    """
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if use_gc:
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

    # Get layer names for LoRA layers only
    layer_names = [n for n, m in model.named_modules()
                   if isinstance(m, nn.Linear) and ('lora_A' in n or 'lora_B' in n)]

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    sample_inputs = {k: v.to(config.device) for k, v in
                     tokenizer('test', return_tensors='pt', max_length=config.seq_length, truncation=True).items()
                     if k != 'labels'}

    # GraSS compression: random_mask sparsifier + SJLT projection
    sparsifier_kwargs = {
        "proj_dim": 1024,  # Factorized: 1024*1024, so each dimension is 1024
        "proj_max_batch_size": 64,  # Match train.py default
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "random_mask",
    }

    projector_kwargs = {
        "proj_dim": 256,  # SJLT projection dimension
        "proj_max_batch_size": 64,  # Match train.py default
        "proj_seed": 42,
        "device": str(config.device),
        "proj_type": "sjlt",
    }

    compressors = setup_model_compressors(
        model=model,
        layer_names=layer_names,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        sample_inputs=sample_inputs,
        device=str(config.device),
        update_freq=200  # Default update frequency
    )

    grad_hook = GradientHook(
        model=model,
        layer_names=layer_names,
        device=str(config.device),
        register_hooks=True
    )
    grad_hook.set_compressors(compressors)

    if use_meso_optimizer:
        optimizer = MeSOAdamW(
            model.parameters(),
            grad_hook=grad_hook,
            lr=5e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    return model, grad_hook, optimizer, tokenizer


def setup_Streaming_NA_lora(config: BenchmarkConfig):
    """Streaming-NA-lora: Per-layer selection with full gradients, LoRA."""
    model, grad_hook, optimizer, tokenizer = _setup_lora_with_grad_hook(config, use_meso_optimizer=False)
    # Clear compressors to use full gradients for the actual update
    grad_hook.compressors = [None] * len(grad_hook.layer_names)
    return model, optimizer, tokenizer, grad_hook


def setup_Streaming_NA_lora_gc(config: BenchmarkConfig):
    """Streaming-NA-lora with gradient checkpointing."""
    model, grad_hook, optimizer, tokenizer = _setup_lora_with_grad_hook(config, use_meso_optimizer=False, use_gc=True)
    grad_hook.compressors = [None] * len(grad_hook.layer_names)
    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_NA_lora(config: BenchmarkConfig):
    """GREATS-NA-lora: Global selection with full gradients, LoRA."""
    model, grad_hook, optimizer, tokenizer = _setup_lora_with_grad_hook(config, use_meso_optimizer=False)
    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_NA_lora_gc(config: BenchmarkConfig):
    """GREATS-NA-lora with gradient checkpointing."""
    model, grad_hook, optimizer, tokenizer = _setup_lora_with_grad_hook(config, use_meso_optimizer=False, use_gc=True)
    return model, optimizer, tokenizer, grad_hook


def setup_Streaming_GraSS_lora(config: BenchmarkConfig):
    """Streaming-GraSS-lora: Per-layer selection + MeSO, LoRA."""
    model, grad_hook, optimizer, tokenizer = _setup_lora_with_grad_hook(config, use_meso_optimizer=True)
    return model, optimizer, tokenizer, grad_hook


def setup_GREATS_GraSS_lora(config: BenchmarkConfig):
    """GREATS-GraSS-lora: Global selection + MeSO, LoRA."""
    model, grad_hook, optimizer, tokenizer = _setup_lora_with_grad_hook(config, use_meso_optimizer=True)
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


# =============================================================================
# Method Registry
# =============================================================================

# Naming convention: {selection}_{compression}_{training}
#   selection: NA (baseline), Streaming (per-layer), GREATS (global)
#   compression: NA (standard optimizer), GraSS, LoGra (MeSO optimizer)
#   training: full, lora

# === Baseline Methods (NA-NA-*) ===
# No selection, no compression, standard optimizer
BASELINE_METHODS = {
    'NA_NA_full': setup_full_adamw,              # Standard full fine-tuning
    'NA_NA_lora': setup_lora_adamw,              # Standard LoRA fine-tuning
    'NA_NA_full_gc': setup_full_adamw_gc,        # With gradient checkpointing
    'NA_NA_lora_gc': setup_lora_adamw_gc,
}

# === MeSO Only Methods (NA-{compression}-*) ===
# Note: Compression without selection doesn't provide meaningful benefit,
# so we don't include NA-LoGra-full or NA-GraSS-full configurations.
# These would just add compression overhead without the selection benefit.
MESO_ONLY_METHODS = {
}

# === Streaming Selection Methods (Streaming-*-*) ===
# Per-layer selection (single-pass)
# Format: method_name -> (setup_fn, selection_frac, has_compression)
# Note: LoRA doesn't need compression (already low-rank), so no Streaming_LoGra_lora
# Note: We use LoGra (not GraSS) as the compression method
STREAMING_METHODS = {
    # Streaming-NA: Per-layer selection with full gradients
    'Streaming_NA_full': (setup_Streaming_NA_full, 0.5, False),
    'Streaming_NA_lora': (setup_Streaming_NA_lora, 0.5, False),
    'Streaming_NA_full_gc': (setup_Streaming_NA_full_gc, 0.5, False),
    'Streaming_NA_lora_gc': (setup_Streaming_NA_lora_gc, 0.5, False),
    # Streaming-LoGra: Per-layer selection with MeSO (full fine-tuning only)
    'Streaming_LoGra_full': (setup_Streaming_LoGra_full, 0.5, True),
    'Streaming_LoGra_full_gc': (setup_Streaming_LoGra_full_gc, 0.5, True),
}

# === GREATS Selection Methods (GREATS-*-*) ===
# Global selection (two-pass)
# Format: method_name -> (setup_fn, selection_frac, has_compression)
# Note: LoRA doesn't need compression (already low-rank), so no GREATS_LoGra_lora
# Note: We use LoGra (not GraSS) as the compression method
GREATS_METHODS = {
    # GREATS-NA: Global selection with full gradients
    'GREATS_NA_full': (setup_GREATS_NA_full, 0.5, False),
    'GREATS_NA_lora': (setup_GREATS_NA_lora, 0.5, False),
    'GREATS_NA_full_gc': (setup_GREATS_NA_full_gc, 0.5, False),
    'GREATS_NA_lora_gc': (setup_GREATS_NA_lora_gc, 0.5, False),
    # GREATS-LoGra: Global selection with MeSO (full fine-tuning only)
    'GREATS_LoGra_full': (setup_GREATS_LoGra_full, 0.5, True),
    'GREATS_LoGra_full_gc': (setup_GREATS_LoGra_full_gc, 0.5, True),
}

# === External Baselines ===
# Other methods for comparison (SGD variants)
EXTERNAL_BASELINES = {
    'full_sgd': setup_full_sgd,
    'full_sgd_momentum': setup_full_sgd_momentum,
    'lora_sgd': setup_lora_sgd,
    'lora_sgd_momentum': setup_lora_sgd_momentum,
    'full_sgd_gc': setup_full_sgd_gc,
    'full_sgd_momentum_gc': setup_full_sgd_momentum_gc,
    'lora_sgd_gc': setup_lora_sgd_gc,
    'lora_sgd_momentum_gc': setup_lora_sgd_momentum_gc,
}

# Combined list for CLI help
ALL_METHODS = (
    list(BASELINE_METHODS.keys()) +
    list(MESO_ONLY_METHODS.keys()) +
    list(STREAMING_METHODS.keys()) +
    list(GREATS_METHODS.keys()) +
    list(EXTERNAL_BASELINES.keys())
)


# =============================================================================
# CLI Interface
# =============================================================================

def run_benchmark(methods: List[str], config: BenchmarkConfig, output_file: Optional[str] = None) -> List[BenchmarkResult]:
    """Run benchmarks for specified methods."""
    bench = Benchmark(config)
    results = []

    for method_name in methods:
        setup_fn = None
        step_fn = None

        # === Baseline Methods (NA-NA-*) ===
        if method_name in BASELINE_METHODS:
            setup_fn = BASELINE_METHODS[method_name]
            step_fn = step_standard

        # === MeSO Only Methods (NA-{compression}-*) ===
        elif method_name in MESO_ONLY_METHODS:
            setup_fn = MESO_ONLY_METHODS[method_name]
            step_fn = step_meso

        # === Streaming Selection Methods ===
        elif method_name in STREAMING_METHODS:
            base_setup_fn, selection_frac, has_compression = STREAMING_METHODS[method_name]

            def make_streaming_setup(base_setup, sel_frac, has_comp):
                def wrapped_setup(cfg):
                    model, optimizer, tokenizer, grad_hook = base_setup(cfg)
                    helper = SelectionStepHelper(tokenizer, cfg.seq_length, cfg.val_batch_size, cfg.device)
                    step = make_step_streaming(helper, sel_frac, use_second_order=cfg.use_second_order, has_compression=has_comp)
                    return model, optimizer, tokenizer, grad_hook, step
                return wrapped_setup

            setup_fn = make_streaming_setup(base_setup_fn, selection_frac, has_compression)

            def streaming_step_wrapper(model, optimizer, batch, grad_hook, step_fn):
                return step_fn(model, optimizer, batch, grad_hook)

            step_fn = streaming_step_wrapper

        # === GREATS Selection Methods ===
        elif method_name in GREATS_METHODS:
            base_setup_fn, selection_frac, has_compression = GREATS_METHODS[method_name]

            def make_greats_setup(base_setup, sel_frac, has_comp):
                def wrapped_setup(cfg):
                    model, optimizer, tokenizer, grad_hook = base_setup(cfg)
                    helper = SelectionStepHelper(tokenizer, cfg.seq_length, cfg.val_batch_size, cfg.device)
                    step = make_step_greats(helper, sel_frac, use_second_order=cfg.use_second_order, has_compression=has_comp)
                    return model, optimizer, tokenizer, grad_hook, step
                return wrapped_setup

            setup_fn = make_greats_setup(base_setup_fn, selection_frac, has_compression)

            def greats_step_wrapper(model, optimizer, batch, grad_hook, step_fn):
                return step_fn(model, optimizer, batch, grad_hook)

            step_fn = greats_step_wrapper

        # === External Baselines ===
        elif method_name in EXTERNAL_BASELINES:
            setup_fn = EXTERNAL_BASELINES[method_name]
            step_fn = step_standard

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
    parser = argparse.ArgumentParser(
        description='Memory and Performance Benchmark for Gradient Streaming',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Naming Convention: {selection}_{compression}_{training}
  selection:   NA (baseline), Streaming (per-layer), GREATS (global)
  compression: NA (standard optimizer), LoGra (MeSO optimizer)
  training:    full, lora

Note: GraSS compression is also available but LoGra is used in default experiments.

Examples:
  NA_NA_full           - Baseline full fine-tuning
  Streaming_LoGra_full - Per-layer selection with MeSO
  GREATS_NA_full       - Global selection with standard optimizer
        """
    )

    # Method selection
    parser.add_argument('--methods', nargs='+', default=['NA_NA_full', 'Streaming_LoGra_full'],
                        help=f'Methods to benchmark. Use --list to see all available methods.')
    parser.add_argument('--all', action='store_true', help='Run all methods')
    parser.add_argument('--list', action='store_true', help='List all available methods and exit')

    # Model config
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B',
                        help='Model name')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--no-flash-attention', action='store_true', help='Disable flash attention')

    # Training config
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-length', type=int, default=512)
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='Validation batch size for data selection (Streaming/GREATS)')

    # Benchmark config
    parser.add_argument('--num-warmup', type=int, default=10)
    parser.add_argument('--num-iterations', type=int, default=10)

    # Selection config
    parser.add_argument('--use-second-order', action='store_true',
                        help='Use second-order interactions for selection (greedy, slower but more accurate)')

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
        print("Naming convention: {selection}_{compression}_{training}")
        print("  selection: NA (baseline), Streaming (per-layer), GREATS (global)")
        print("  compression: NA (standard optimizer), LoGra (MeSO optimizer)")
        print("  training: full, lora")
        print("")
        print("Note: GraSS compression is also available but LoGra is used in default experiments.")
        print("")
        print("=" * 80)
        print(f"{'Index':<6} {'Method Name':<30} {'Category':<25} {'Description'}")
        print("-" * 80)
        idx = 0
        for name in BASELINE_METHODS:
            print(f"{idx:<6} {name:<30} {'Baseline (NA-NA)':<25} No selection, standard optimizer")
            idx += 1
        for name in MESO_ONLY_METHODS:
            print(f"{idx:<6} {name:<30} {'MeSO Only':<25} No selection, compressed optimizer")
            idx += 1
        for name in STREAMING_METHODS:
            _, _, has_comp = STREAMING_METHODS[name]
            desc = "Per-layer selection" + (" + MeSO" if has_comp else " + AdamW")
            print(f"{idx:<6} {name:<30} {'Streaming':<25} {desc}")
            idx += 1
        for name in GREATS_METHODS:
            _, _, has_comp = GREATS_METHODS[name]
            desc = "Global selection" + (" + MeSO" if has_comp else " + AdamW")
            print(f"{idx:<6} {name:<30} {'GREATS':<25} {desc}")
            idx += 1
        for name in EXTERNAL_BASELINES:
            print(f"{idx:<6} {name:<30} {'External Baseline':<25} Comparison methods")
            idx += 1
        print("=" * 80)
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
