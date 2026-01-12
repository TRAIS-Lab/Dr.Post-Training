#!/usr/bin/env python3
"""
Minimal streaming vs baseline benchmark using dummy tokens.

Methods:
  - NA_NA_Full: standard AdamW training
  - Streaming_NA_Full: per-layer selection with standard AdamW
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root is on sys.path for local imports when run from standalone/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _nvtx_push(msg: str) -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(msg)


def _nvtx_pop() -> None:
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop()


@dataclass
class Config:
    model: str
    batch_size: int
    seq_length: int
    val_batch_size: int
    val_seq_length: int
    iterations: int
    warmup: int
    method: str
    val_strategy: str
    dtype: str
    device: str
    use_flash_attention: bool
    selection_frac: float

    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping[self.dtype]


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Minimal streaming vs baseline benchmark.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-length", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--val-seq-length", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--method", type=str, default="Streaming_NA_Full",
                        choices=["NA_NA_Full", "Streaming_NA_Full"])
    parser.add_argument("--val-strategy", type=str, default="separate",
                        choices=["separate", "merged"])
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-flash-attention", action="store_true")
    parser.add_argument("--selection-frac", type=float, default=0.5)

    args = parser.parse_args()
    val_seq_length = args.val_seq_length if args.val_seq_length is not None else args.seq_length

    return Config(
        model=args.model,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        val_batch_size=args.val_batch_size,
        val_seq_length=val_seq_length,
        iterations=args.iterations,
        warmup=args.warmup,
        method=args.method,
        val_strategy=args.val_strategy,
        dtype=args.dtype,
        device=args.device,
        use_flash_attention=not args.no_flash_attention,
        selection_frac=args.selection_frac,
    )


def build_model_and_tokenizer(cfg: Config):
    model_kwargs = {"dtype": cfg.torch_dtype(), "device_map": cfg.device}
    if cfg.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(cfg.model, **model_kwargs)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def split_train_val_batch(tensor: torch.Tensor, train_batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return tensor[:train_batch_size], tensor[train_batch_size:]


def compute_total_gradient(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    if grad_output.dim() == 3:
        return torch.einsum("bso,bsi->oi", grad_output, input)
    return torch.einsum("bo,bi->oi", grad_output, input)


def compute_scores_and_similarity(
    train_grad_output: torch.Tensor,
    train_input: torch.Tensor,
    val_grad_output: Optional[torch.Tensor],
    val_input: Optional[torch.Tensor],
    val_grad_total: Optional[torch.Tensor],
) -> torch.Tensor:
    if val_grad_output is not None and val_input is not None:
        if train_grad_output.dim() == 3:
            val_grad_total = torch.einsum("vso,vsi->oi", val_grad_output, val_input)
            temp = train_input @ val_grad_total.T
            return (train_grad_output * temp).sum(dim=(1, 2))
        val_grad_total = torch.einsum("vo,vi->oi", val_grad_output, val_input)
        temp = train_input @ val_grad_total.T
        return (train_grad_output * temp).sum(dim=1)

    if val_grad_total is not None:
        if train_grad_output.dim() == 3:
            temp = train_input @ val_grad_total.T
            return (train_grad_output * temp).sum(dim=(1, 2))
        temp = train_input @ val_grad_total.T
        return (train_grad_output * temp).sum(dim=1)

    raise ValueError("Missing validation gradients for scoring")


def compute_selected_gradients(
    train_grad_output: torch.Tensor,
    train_input: torch.Tensor,
    selected_indices: torch.Tensor,
    has_bias: bool,
    scale_factor: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _nvtx_push("selected_slice")
    selected_grad_output = train_grad_output[selected_indices]
    selected_input = train_input[selected_indices]
    _nvtx_pop()

    if selected_grad_output.dim() == 3:
        _nvtx_push("w_grad")
        grad_weight = torch.einsum("kso,ksi->oi", selected_grad_output, selected_input) * scale_factor
        _nvtx_pop()
        if has_bias:
            _nvtx_push("b_grad")
            grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor
            _nvtx_pop()
        else:
            grad_bias = None
    else:
        _nvtx_push("w_grad")
        grad_weight = torch.einsum("ko,ki->oi", selected_grad_output, selected_input) * scale_factor
        _nvtx_pop()
        if has_bias:
            _nvtx_push("b_grad")
            grad_bias = selected_grad_output.sum(dim=0) * scale_factor
            _nvtx_pop()
        else:
            grad_bias = None

    return grad_weight, grad_bias


def topk_selection(scores: torch.Tensor, num_selected: int) -> torch.Tensor:
    if num_selected <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    num_selected = min(num_selected, scores.numel())
    return torch.topk(scores, k=num_selected, largest=True).indices


class MinimalStreamingState:
    def __init__(self, train_batch_size: int, frac: float, lr: float):
        self.train_batch_size = train_batch_size
        self.frac = frac
        self.lr = lr
        self.num_selected = max(1, int(train_batch_size * frac))
        self.tokens_per_sample: Optional[torch.Tensor] = None
        self.train_total_tokens: int = 0
        self.train_total_tokens_tensor: Optional[torch.Tensor] = None
        self.score_correction: float = 1.0

    def set_token_counts(self, labels: torch.Tensor, train_batch_size: Optional[int] = None) -> None:
        valid_mask = (labels != -100)
        tokens_per_sample = valid_mask.sum(dim=1)
        self.tokens_per_sample = tokens_per_sample

        if train_batch_size is None:
            train_batch_size = tokens_per_sample.shape[0]
        train_tokens = tokens_per_sample[:train_batch_size]
        total_train_tokens = train_tokens.sum().item()
        total_tokens = tokens_per_sample.sum().item()

        self.train_total_tokens = total_train_tokens
        self.train_total_tokens_tensor = torch.tensor(
            float(total_train_tokens),
            device=labels.device,
            dtype=torch.float32,
        )
        val_tokens = total_tokens - total_train_tokens
        if val_tokens > 0 and total_train_tokens > 0:
            self.score_correction = float(total_tokens ** 2) / float(total_train_tokens * val_tokens)
        else:
            self.score_correction = 1.0

    def compute_scale_factor(self, selected_indices: torch.Tensor) -> torch.Tensor:
        if self.tokens_per_sample is None or self.train_total_tokens_tensor is None:
            return torch.tensor(1.0, device=selected_indices.device, dtype=torch.float32)
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=selected_indices.device, dtype=torch.float32)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.train_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens > 0, scale, torch.ones_like(scale))


class MinimalGradientHook:
    def __init__(self, model: nn.Module):
        self.model = model
        self.layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        self.layer_name_to_idx = {name: idx for idx, name in enumerate(self.layer_names)}
        self.layer_name_to_module = {}
        self.selection_state: Optional[MinimalStreamingState] = None
        self.capture_val_mode: bool = False
        self.use_factorized_val: bool = True
        self.val_grad_output_buffer = [None] * len(self.layer_names)
        self.val_input_buffer = [None] * len(self.layer_names)
        self.val_grad_buffer = [None] * len(self.layer_names)
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                idx = self.layer_name_to_idx[name]
                self.layer_name_to_module[name] = module
                if not isinstance(module, nn.Linear):
                    continue
                module._original_forward = module.forward
                module.forward = lambda input, m=module, i=idx: MinimalLinearBackward.apply(
                    input, m.weight, m.bias, self, i
                )

    def start_val_capture(self, use_factorized: bool = True) -> None:
        self.capture_val_mode = True
        self.use_factorized_val = use_factorized
        self.val_grad_output_buffer = [None] * len(self.layer_names)
        self.val_input_buffer = [None] * len(self.layer_names)
        self.val_grad_buffer = [None] * len(self.layer_names)

    def end_val_capture(self) -> None:
        self.capture_val_mode = False

    def clear_val_buffer(self) -> None:
        self.val_grad_output_buffer = [None] * len(self.layer_names)
        self.val_input_buffer = [None] * len(self.layer_names)
        self.val_grad_buffer = [None] * len(self.layer_names)
        self.capture_val_mode = False

    def setup_streaming(self, train_batch_size: int, frac: float, lr: float) -> None:
        self.selection_state = MinimalStreamingState(
            train_batch_size=train_batch_size,
            frac=frac,
            lr=lr,
        )

    def clear_selection(self) -> None:
        self.selection_state = None


class MinimalLinearBackward(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, hook: MinimalGradientHook, layer_idx: int):
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook = hook
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        hook = ctx.hook
        layer_idx = ctx.layer_idx

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        _nvtx_push(f"layer_{layer_idx}/grad_input")
        grad_input = grad_output @ weight.to(grad_output.dtype)
        _nvtx_pop()

        if hook.capture_val_mode:
            _nvtx_push(f"layer_{layer_idx}/val_capture")
            if hook.use_factorized_val:
                hook.val_grad_output_buffer[layer_idx] = grad_output.detach()
                hook.val_input_buffer[layer_idx] = input.detach()
            else:
                total = compute_total_gradient(grad_output, input)
                if hook.val_grad_buffer[layer_idx] is None:
                    hook.val_grad_buffer[layer_idx] = total
                else:
                    hook.val_grad_buffer[layer_idx] += total
            _nvtx_pop()
            return grad_input, None, None, None, None

        state = hook.selection_state
        if state is None:
            _nvtx_push(f"layer_{layer_idx}/full_grad")
            grad_weight = compute_total_gradient(grad_output, input)
            if bias is not None:
                if grad_output.dim() == 3:
                    grad_bias = grad_output.sum(dim=(0, 1))
                else:
                    grad_bias = grad_output.sum(dim=0)
            else:
                grad_bias = None
            _nvtx_pop()
            return grad_input, grad_weight, grad_bias, None, None

        val_grad_output = hook.val_grad_output_buffer[layer_idx]
        val_input = hook.val_input_buffer[layer_idx]
        val_grad_total = hook.val_grad_buffer[layer_idx]

        train_grad_output = grad_output
        train_input = input

        _nvtx_push(f"layer_{layer_idx}/val_resolve")
        if val_grad_output is None and val_grad_total is None:
            if grad_output.shape[0] > state.train_batch_size:
                train_grad_output, val_grad_output = split_train_val_batch(
                    grad_output, state.train_batch_size
                )
                train_input, val_input = split_train_val_batch(
                    input, state.train_batch_size
                )
                val_grad_total = compute_total_gradient(val_grad_output, val_input)
                val_grad_output = None
                val_input = None
        _nvtx_pop()

        _nvtx_push(f"layer_{layer_idx}/score_compute")
        scores = compute_scores_and_similarity(
            train_grad_output,
            train_input,
            val_grad_output,
            val_input,
            val_grad_total,
        )
        _nvtx_pop()
        _nvtx_push(f"layer_{layer_idx}/select_reduce")
        scores = scores * state.score_correction
        selected_indices = topk_selection(scores * state.lr, state.num_selected).sort()[0]
        scale_factor = state.compute_scale_factor(selected_indices)
        grad_weight, grad_bias = compute_selected_gradients(
            train_grad_output,
            train_input,
            selected_indices,
            bias is not None,
            scale_factor,
        )
        _nvtx_pop()
        return grad_input, grad_weight, grad_bias, None, None


def random_batch(
    tokenizer: AutoTokenizer,
    batch_size: int,
    seq_length: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    vocab_size = tokenizer.vocab_size
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def pad_to_length(tensor: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
    if tensor.shape[1] >= target_len:
        return tensor
    pad_len = target_len - tensor.shape[1]
    pad = torch.full(
        (tensor.shape[0], pad_len),
        pad_value,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.cat([tensor, pad], dim=1)


def pad_and_merge_batches(
    train_batch: Dict[str, torch.Tensor],
    val_batch: Dict[str, torch.Tensor],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    max_len = max(train_batch["input_ids"].shape[1], val_batch["input_ids"].shape[1])
    merged = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        if key == "input_ids":
            pad_value = pad_token_id
        elif key == "labels":
            pad_value = -100
        else:
            pad_value = 0
        train = pad_to_length(train_batch[key], max_len, pad_value)
        val = pad_to_length(val_batch[key], max_len, pad_value)
        merged[key] = torch.cat([train, val], dim=0)
    return merged


def compute_loss(model: torch.nn.Module, batch: Dict[str, torch.Tensor], device_type: str) -> torch.Tensor:
    with torch.autocast(device_type=device_type, dtype=next(model.parameters()).dtype):
        outputs = model(**batch)
        return outputs.loss


def step_baseline(model, optimizer, batch, device_type: str) -> float:
    optimizer.zero_grad()
    loss = compute_loss(model, batch, device_type)
    _nvtx_push("baseline/backward")
    loss.backward()
    _nvtx_pop()
    optimizer.step()
    return loss.item()


def step_streaming_separate(
    model,
    optimizer,
    grad_hook: MinimalGradientHook,
    train_batch,
    val_batch,
    device_type: str,
    lr: float,
    selection_frac: float,
) -> Tuple[float, float, float]:
    val_start = time.perf_counter()
    grad_hook.start_val_capture(use_factorized=True)
    optimizer.zero_grad()
    val_loss = compute_loss(model, val_batch, device_type)
    _nvtx_push("streaming/val_backward")
    val_loss.backward()
    _nvtx_pop()
    grad_hook.end_val_capture()
    val_end = time.perf_counter()

    optimizer.zero_grad()
    grad_hook.setup_streaming(
        train_batch_size=train_batch["input_ids"].shape[0],
        frac=selection_frac,
        lr=lr,
    )
    grad_hook.selection_state.set_token_counts(train_batch["labels"])
    train_start = time.perf_counter()
    loss = compute_loss(model, train_batch, device_type)
    _nvtx_push("streaming/train_backward")
    loss.backward()
    _nvtx_pop()
    train_end = time.perf_counter()
    optimizer.step()
    grad_hook.clear_selection()
    grad_hook.clear_val_buffer()
    return loss.item(), val_end - val_start, train_end - train_start


def step_streaming_merged(
    model,
    optimizer,
    grad_hook: MinimalGradientHook,
    train_batch,
    val_batch,
    device_type: str,
    lr: float,
    pad_token_id: int,
    selection_frac: float,
) -> Tuple[float, float]:
    merged_batch = pad_and_merge_batches(train_batch, val_batch, pad_token_id)
    optimizer.zero_grad()
    grad_hook.setup_streaming(
        train_batch_size=train_batch["input_ids"].shape[0],
        frac=selection_frac,
        lr=lr,
    )
    grad_hook.selection_state.set_token_counts(merged_batch["labels"], train_batch["input_ids"].shape[0])
    train_start = time.perf_counter()
    loss = compute_loss(model, merged_batch, device_type)
    _nvtx_push("streaming/merged_backward")
    loss.backward()
    _nvtx_pop()
    train_end = time.perf_counter()
    optimizer.step()
    grad_hook.clear_selection()
    return loss.item(), train_end - train_start


def main() -> None:
    cfg = parse_args()
    device_type = "cuda" if "cuda" in cfg.device else "cpu"

    model, tokenizer = build_model_and_tokenizer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    grad_hook = None
    if cfg.method == "Streaming_NA_Full":
        grad_hook = MinimalGradientHook(model)

    timings = []
    val_timings = []
    train_timings = []
    peak_allocated = 0
    peak_reserved = 0

    memory_after_setup = None
    if device_type == "cuda":
        torch.cuda.synchronize()
        memory_after_setup = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    total_iters = cfg.warmup + cfg.iterations
    for step in range(total_iters):
        train_batch = random_batch(tokenizer, cfg.batch_size, cfg.seq_length, cfg.device)
        val_batch = random_batch(tokenizer, cfg.val_batch_size, cfg.val_seq_length, cfg.device)

        if device_type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()

        if cfg.method == "NA_NA_Full":
            loss_val = step_baseline(model, optimizer, train_batch, device_type)
        else:
            lr = optimizer.param_groups[0]["lr"]
            if cfg.val_strategy == "separate":
                loss_val, val_time, train_time = step_streaming_separate(
                    model,
                    optimizer,
                    grad_hook,
                    train_batch,
                    val_batch,
                    device_type,
                    lr,
                    cfg.selection_frac,
                )
                if step >= cfg.warmup:
                    val_timings.append(val_time)
                    train_timings.append(train_time)
            else:
                loss_val, train_time = step_streaming_merged(
                    model,
                    optimizer,
                    grad_hook,
                    train_batch,
                    val_batch,
                    device_type,
                    lr,
                    pad_token_id=tokenizer.pad_token_id or 0,
                    selection_frac=cfg.selection_frac,
                )
                if step >= cfg.warmup:
                    train_timings.append(train_time)

        if device_type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        if step == cfg.warmup - 1 and device_type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        if step >= cfg.warmup:
            timings.append(end - start)
            if device_type == "cuda":
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())

        if (step + 1) % 10 == 0:
            print(f"step {step + 1}/{total_iters} loss={loss_val:.4f}")

    if timings:
        avg_time = sum(timings) / len(timings)
        throughput = cfg.batch_size / avg_time
        print("")
        print("Results:")
        print(f"  method: {cfg.method}")
        print(f"  val_strategy: {cfg.val_strategy}")
        print(f"  avg_time_per_iter_s: {avg_time:.4f}")
        print(f"  throughput_samples_per_sec: {throughput:.2f}")
        if cfg.method == "Streaming_NA_Full" and val_timings:
            avg_val = sum(val_timings) / len(val_timings)
            print(f"  avg_val_pass_s: {avg_val:.4f}")
        if cfg.method == "Streaming_NA_Full" and train_timings:
            avg_train = sum(train_timings) / len(train_timings)
            label = "avg_train_pass_s" if cfg.val_strategy == "separate" else "avg_merged_pass_s"
            print(f"  {label}: {avg_train:.4f}")
        if device_type == "cuda":
            if memory_after_setup is not None:
                print(f"  memory_after_setup_gb: {memory_after_setup / (1024 ** 3):.3f}")
            print(f"  peak_allocated_gb: {peak_allocated / (1024 ** 3):.3f}")
            print(f"  peak_reserved_gb: {peak_reserved / (1024 ** 3):.3f}")


if __name__ == "__main__":
    main()
