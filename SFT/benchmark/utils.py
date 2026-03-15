#!/usr/bin/env python
"""
Detailed Timing Breakdown Benchmark for Dr. Post-Training.

Measures per-component runtime for each method:
  1. Forward pass
  2. Backward (activation gradients, excluding w.grad)
  3. Score computation (selection methods only)
  4. w.grad computation
  5. Optimizer step

Methods benchmarked:
  - Standard: baseline training (forward + backward + optimizer)
  - Layerwise: per-layer selection (single-pass backward with scores + w.grad)
  - Subset: global selection (two-pass: scoring pass + materialization pass)

Methodology for decomposing backward into components:
  - Activation-grad time: Backward with requires_grad=False on all weight params
    (except embedding). This makes PyTorch skip grad_weight computation per layer,
    measuring only the chain-rule grad_input propagation.
  - Score time: Subset pass-1 backward (activation grads + score accumulation,
    no w.grad) minus activation-grad-only time.
  - w.grad time (Standard): Full backward minus activation-grad-only backward.
  - w.grad time (Layerwise): Total train backward minus activation-grad minus score.
  - w.grad time (Subset): Pass-2 forward + pass-2 backward (direct measurement).

All timing uses CUDA events for accurate GPU measurement.
"""

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='torch._dynamo')

from datasets import load_dataset

from drpt.hook import GradientHook
from drpt.compression_mode import CompressionMode


# =============================================================================
# Seed Utility
# =============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility across all random sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class BenchmarkConfig:
    """
    Benchmark configuration.

    This is the single source of truth for all configuration defaults.
    CLI arguments override these defaults when specified.
    """
    # Model
    model_name: str = 'meta-llama/Llama-3.2-1B'
    dtype: str = 'bfloat16'
    use_flash_attention: bool = True

    # Training
    batch_size: int = 16
    seq_length: int = 512
    val_batch_size: int = 1

    # Dataset config
    # Options: 'dummy', 'alpaca', 'gsm8k', 'dolly', 'openhermes'
    dataset: str = 'tulu3'

    # Benchmark
    num_warmup: int = 10
    num_iterations: int = 20

    # LoRA config
    lora_rank: int = 32
    lora_alpha: int = 1

    # Selection config
    use_second_order: bool = False  # If True, use greedy selection with O(k*n) complexity
    val_strategy: str = 'merged'  # 'separate' or 'merged' - how to handle validation gradients
    score_compression: str = "normal-64*64"  # Score-only compression for influence scoring (e.g., "normal-64*64")
    val_dataset: str = 'tydiqa'  # Validation dataset for selection. Options: 'samsum', 'gsm8k', 'bbh', etc. If None, uses same as training dataset
    data_dir: str = 'data'  # Data directory for validation datasets (used when val_dataset is set)

    # Reproducibility
    seed: int = 42

    # Device
    device: str = 'cuda'

    def get_torch_dtype(self):
        return {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]


# =============================================================================
# Dataset Classes for Benchmarking
# =============================================================================

def concat_messages(messages, tokenizer):
    """
    Concatenate messages into a single string with role delimiters.
    Matches the format used in SFT/data/get_train_dataset.py
    """
    message_text = ""
    for message in messages:
        if message["role"] == "system":
            message_text += "<|system|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "user":
            message_text += "<|user|>\n" + message["content"].strip() + "\n"
        elif message["role"] == "assistant":
            message_text += "<|assistant|>\n" + \
                message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Encode an example with messages format.
    Matches the encoding used in SFT/data/get_train_dataset.py
    """
    messages = example['messages']
    if len(messages) == 0:
        return None

    example_text = concat_messages(messages, tokenizer)
    tokenized_example = tokenizer(
        example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    concat_messages(messages[:message_idx], tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer) + "<|assistant|>\n"
            else:
                messages_so_far = concat_messages(
                    messages[:message_idx+1], tokenizer)
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


class DummyDataset(Dataset):
    """Dummy dataset that generates random tokens for benchmarking."""

    def __init__(self, tokenizer, seq_length: int, size: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.size = size
        # Pre-tokenize a dummy sentence
        self.dummy_text = "This is a test sentence for memory and performance benchmarking." * 128

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


class RealDataset(Dataset):
    """
    Dataset that loads real training data from HuggingFace.
    Uses the same encoding format as the actual training code.
    """

    # Mapping of dataset names to HuggingFace dataset info
    DATASET_INFO = {
        'alpaca': {
            'path': 'tatsu-lab/alpaca',
            'split': 'train',
            'format': 'alpaca',  # instruction, input, output format
        },
        'gsm8k': {
            'path': 'openai/gsm8k',
            'name': 'main',
            'split': 'train',
            'format': 'gsm8k',  # question, answer format
        },
        'dolly': {
            'path': 'databricks/databricks-dolly-15k',
            'split': 'train',
            'format': 'dolly',  # instruction, context, response format
        },
        'openhermes': {
            'path': 'teknium/OpenHermes-2.5',
            'split': 'train',
            'format': 'sharegpt',  # conversations format
        },
        'samsum': {
            'path': 'knkarthick/samsum',
            'split': 'train',
            'format': 'samsum',  # dialogue, summary format
        },
        'vicuna': {
            'path': 'Aeala/ShareGPT_Vicuna_unfiltered',
            'split': 'train',
            'format': 'sharegpt',  # conversations format
        },
        'wizardlm': {
            'path': 'WizardLMTeam/WizardLM_evol_instruct_V2_196k',
            'split': 'train',
            'format': 'sharegpt',  # conversations format
        },
        'tulu3': {
            'path': 'allenai/tulu-3-sft-mixture',
            'split': 'train',
            'format': 'tulu',  # messages format
        },
        'oasst1': {
            'path': 'OpenAssistant/oasst1',
            'split': 'train',
            'format': 'oasst',  # tree structure, prompter/assistant roles
        },
        'cot': {
            'path': 'kaist-ai/CoT-Collection',
            'split': 'train',
            'format': 'cot',  # source, rationale format
        },
    }

    def __init__(self, dataset_name: str, tokenizer, seq_length: int, size: int = 10000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.size = size

        if dataset_name not in self.DATASET_INFO:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASET_INFO.keys())}")

        info = self.DATASET_INFO[dataset_name]
        print(f"Loading dataset: {info['path']}...")

        # Load dataset from HuggingFace
        if 'name' in info:
            raw_dataset = load_dataset(info['path'], info['name'], split=info['split'])
        else:
            raw_dataset = load_dataset(info['path'], split=info['split'])

        # Convert to messages format and encode
        self.encoded_examples = []
        for i, example in enumerate(raw_dataset):
            if len(self.encoded_examples) >= size:
                break

            messages = self._convert_to_messages(example, info['format'])
            if messages is None:
                continue

            encoded = encode_with_messages_format(
                {'messages': messages}, tokenizer, seq_length)
            if encoded is not None:
                self.encoded_examples.append(encoded)

        # Compute and print sequence length statistics
        seq_lengths = [ex['input_ids'].shape[0] for ex in self.encoded_examples]
        avg_len = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
        print(f"Loaded {len(self.encoded_examples)} examples from {dataset_name} "
              f"(avg seq len: {avg_len:.1f}, min: {min(seq_lengths)}, max: {max(seq_lengths)})")

    def _convert_to_messages(self, example, format_type):
        """Convert different dataset formats to unified messages format."""
        try:
            if format_type == 'alpaca':
                # Alpaca format: instruction, input, output
                instruction = example.get('instruction', '')
                input_text = example.get('input', '')
                output = example.get('output', '')

                if input_text:
                    user_content = f"{instruction}\n\n{input_text}"
                else:
                    user_content = instruction

                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output}
                ]

            elif format_type == 'gsm8k':
                # GSM8K format: question, answer
                question = example.get('question', '')
                answer = example.get('answer', '')

                user_content = f"Solve the following math problem step by step.\n\n{question}"

                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer}
                ]

            elif format_type == 'dolly':
                # Dolly format: instruction, context, response
                instruction = example.get('instruction', '')
                context = example.get('context', '')
                response = example.get('response', '')

                if context:
                    user_content = f"{instruction}\n\nContext: {context}"
                else:
                    user_content = instruction

                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response}
                ]

            elif format_type == 'sharegpt':
                # ShareGPT/OpenHermes/Vicuna/WizardLM format: conversations list
                conversations = example.get('conversations', [])
                messages = []
                for conv in conversations:
                    role = conv.get('from', conv.get('role', ''))
                    content = conv.get('value', conv.get('content', ''))
                    if role in ('human', 'user'):
                        messages.append({"role": "user", "content": content})
                    elif role in ('gpt', 'assistant'):
                        messages.append({"role": "assistant", "content": content})
                    elif role == 'system':
                        messages.append({"role": "system", "content": content})
                return messages if len(messages) >= 2 else None

            elif format_type == 'samsum':
                # SamSUM format: dialogue, summary
                dialogue = example.get('dialogue', '')
                summary = example.get('summary', '')
                return [
                    {"role": "user", "content": f"Summarize the following dialogue:\n\n{dialogue}"},
                    {"role": "assistant", "content": summary}
                ]

            elif format_type == 'tulu':
                # Tulu3 format: messages list with role/content
                messages_raw = example.get('messages', [])
                messages = []
                for msg in messages_raw:
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role == 'user':
                        messages.append({"role": "user", "content": content})
                    elif role == 'assistant':
                        messages.append({"role": "assistant", "content": content})
                    # Skip system messages
                return messages if len(messages) >= 2 else None

            elif format_type == 'oasst':
                # OpenAssistant format: role is 'prompter' or 'assistant'
                role = example.get('role', '')
                text = example.get('text', '')
                # OASST has tree structure; for simplicity, we skip standalone messages
                # and only use examples that have clear user/assistant pairs
                # This is a simplified approach - full OASST needs tree reconstruction
                if role == 'prompter':
                    return [{"role": "user", "content": text}]
                elif role == 'assistant':
                    return [{"role": "assistant", "content": text}]
                return None

            elif format_type == 'cot':
                # Chain-of-Thought format: source, rationale
                source = example.get('source', '')
                rationale = example.get('rationale', '')
                return [
                    {"role": "user", "content": source},
                    {"role": "assistant", "content": rationale}
                ]

            else:
                return None
        except Exception:
            return None

    def __len__(self):
        return len(self.encoded_examples)

    def __getitem__(self, idx):
        return self.encoded_examples[idx % len(self.encoded_examples)]


class ValidationDataset(Dataset):
    """
    Dataset that loads validation data directly from HuggingFace for selection methods.
    This avoids the need to prepare JSONL files for benchmark usage.
    """

    # Mapping of dataset names to HuggingFace dataset info
    DATASET_INFO = {
        'samsum': {
            'path': 'knkarthick/samsum',
            'split': 'validation',
            'format': 'samsum',
        },
        'gsm8k': {
            'path': 'openai/gsm8k',
            'name': 'main',
            'split': 'test',  # GSM8K uses test split for validation
            'format': 'gsm8k',
        },
        'tydiqa': {
            'path': 'tydiqa',
            'name': 'secondary_task',
            'split': 'validation',
            'format': 'tydiqa',
        },
        'mmlu': {
            'path': 'cais/mmlu',
            'name': 'sociology',  # Use one subject for benchmark (full MMLU has 57 subjects)
            'split': 'validation',
            'format': 'mmlu',
        },
        'bbh': {
            'path': 'lukaemon/bbh',
            'name': 'boolean_expressions',  # Use one task for benchmark (full BBH has 27 tasks)
            'split': 'test',
            'format': 'bbh',
        },
        'math500': {
            'path': 'HuggingFaceH4/MATH-500',
            'split': 'test',
            'format': 'math500',
        },
    }

    def __init__(self, dataset_name: str, tokenizer, seq_length: int, size: int = 1000):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.size = size

        if dataset_name not in self.DATASET_INFO:
            raise ValueError(f"Unknown validation dataset: {dataset_name}. Available: {list(self.DATASET_INFO.keys())}")

        info = self.DATASET_INFO[dataset_name]
        print(f"Loading validation dataset: {info['path']} (split: {info['split']})...")

        # Load dataset from HuggingFace
        if 'name' in info:
            raw_dataset = load_dataset(info['path'], info['name'], split=info['split'])
        else:
            raw_dataset = load_dataset(info['path'], split=info['split'])

        # Convert to messages format and encode
        self.encoded_examples = []
        for i, example in enumerate(raw_dataset):
            if len(self.encoded_examples) >= size:
                break

            messages = self._convert_to_messages(example, info['format'])
            if messages is None:
                continue

            encoded = encode_with_messages_format(
                {'messages': messages}, tokenizer, seq_length)
            if encoded is not None:
                self.encoded_examples.append(encoded)

        # Compute and print sequence length statistics
        seq_lengths = [ex['input_ids'].shape[0] for ex in self.encoded_examples]
        avg_len = sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0
        print(f"Loaded {len(self.encoded_examples)} validation examples from {dataset_name} "
              f"(avg seq len: {avg_len:.1f}, min: {min(seq_lengths)}, max: {max(seq_lengths)})")

    def _convert_to_messages(self, example, format_type):
        """Convert different dataset formats to unified messages format."""
        try:
            if format_type == 'samsum':
                # SamSUM format: dialogue, summary
                dialogue = example.get('dialogue', '')
                summary = example.get('summary', '')

                return [
                    {"role": "user", "content": f"Summarize the following dialogue:\n\n{dialogue}"},
                    {"role": "assistant", "content": summary}
                ]

            elif format_type == 'gsm8k':
                # GSM8K format: question, answer
                question = example.get('question', '')
                answer = example.get('answer', '')

                return [
                    {"role": "user", "content": f"Solve the following math problem step by step.\n\n{question}"},
                    {"role": "assistant", "content": answer}
                ]

            elif format_type == 'tydiqa':
                # TyDiQA format: context, question, answers
                context = example.get('context', '')
                question = example.get('question', '')
                answers = example.get('answers', {})
                answer_texts = answers.get('text', [])
                answer = answer_texts[0] if answer_texts else ''

                user_content = f"Answer the question based on the given context.\n\nContext: {context}\n\nQuestion: {question}"
                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer}
                ]

            elif format_type == 'mmlu':
                # MMLU format: question, choices, answer
                question = example.get('question', '')
                choices = example.get('choices', [])
                answer_idx = example.get('answer', 0)
                choices_letters = ["A", "B", "C", "D"]

                # Format question with choices
                prompt = f"The following is a multiple choice question. Answer with A, B, C, or D.\n\n{question}\n"
                for i, choice in enumerate(choices):
                    prompt += f"{choices_letters[i]}. {choice}\n"
                prompt += "\nAnswer:"

                # Convert answer index to letter
                answer = choices_letters[answer_idx] if isinstance(answer_idx, int) else answer_idx

                return [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer}
                ]

            elif format_type == 'bbh':
                # BBH format: input, target
                input_text = example.get('input', '')
                target = example.get('target', '')

                return [
                    {"role": "user", "content": input_text},
                    {"role": "assistant", "content": target}
                ]

            elif format_type == 'math500':
                # MATH500 format: problem, solution
                problem = example.get('problem', '')
                solution = example.get('solution', '')
                level = example.get('level', 'unknown')
                prob_type = example.get('type', 'unknown')

                user_content = f"Solve the following Level {level} {prob_type} problem:\n{problem}"

                return [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": solution}
                ]

            else:
                return None
        except Exception:
            return None

    def __len__(self):
        return len(self.encoded_examples)

    def __getitem__(self, idx):
        return self.encoded_examples[idx % len(self.encoded_examples)]


def get_benchmark_dataset(config: BenchmarkConfig, tokenizer, size: int = 10000) -> Dataset:
    """Get the appropriate dataset based on config."""
    if config.dataset == 'dummy':
        return DummyDataset(tokenizer, config.seq_length, size)
    else:
        return RealDataset(config.dataset, tokenizer, config.seq_length, size)


def get_data_collator(tokenizer, config: BenchmarkConfig):
    """Get the appropriate data collator based on config."""
    if config.dataset == 'dummy':
        # Dummy dataset already pads to max_length, use default collator
        return None
    else:
        # Real datasets have variable lengths, need padding collator
        return DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")


def pad_and_merge_batches(batch1: Dict[str, torch.Tensor], batch2: Dict[str, torch.Tensor], pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    """
    Pad two batches to the same sequence length and concatenate them.

    Args:
        batch1: First batch (train batch)
        batch2: Second batch (validation batch)
        pad_token_id: Token ID to use for padding (default: 0)

    Returns:
        Merged batch with both batches padded to the same length
    """
    merged_batch = {}

    for key in batch1.keys():
        if key not in batch2:
            merged_batch[key] = batch1[key]
            continue

        t1, t2 = batch1[key], batch2[key]

        # Get sequence lengths
        seq_len1 = t1.shape[1] if t1.dim() > 1 else t1.shape[0]
        seq_len2 = t2.shape[1] if t2.dim() > 1 else t2.shape[0]
        max_len = max(seq_len1, seq_len2)

        # Determine padding value based on key
        if key == 'labels':
            pad_value = -100  # Ignore index for loss
        elif key == 'attention_mask':
            pad_value = 0  # No attention to padding
        else:
            pad_value = pad_token_id

        # Pad if needed
        if t1.dim() > 1:
            # 2D tensor (batch_size, seq_len)
            if seq_len1 < max_len:
                padding = torch.full((t1.shape[0], max_len - seq_len1), pad_value, dtype=t1.dtype, device=t1.device)
                t1 = torch.cat([t1, padding], dim=1)
            if seq_len2 < max_len:
                padding = torch.full((t2.shape[0], max_len - seq_len2), pad_value, dtype=t2.dtype, device=t2.device)
                t2 = torch.cat([t2, padding], dim=1)

        merged_batch[key] = torch.cat([t1, t2], dim=0)

    return merged_batch


# =============================================================================
# CUDA Timing Utilities
# =============================================================================

class CUDAEventTimer:
    """
    Batched CUDA event timer that avoids synchronization inside measurement loops.

    Usage:
        timer = CUDAEventTimer(phases=["forward", "backward"], num_iters=20)
        for i in range(20):
            timer.mark("forward", i, is_start=True)
            ... forward pass ...
            timer.mark("forward", i, is_start=False)
            timer.mark("backward", i, is_start=True)
            ... backward pass ...
            timer.mark("backward", i, is_start=False)
        results = timer.elapsed()  # single sync, returns dict of phase -> list[ms]
    """

    def __init__(self, phases: List[str], num_iters: int):
        self.phases = phases
        self.num_iters = num_iters
        # Pre-allocate all CUDA events (no sync needed during recording)
        self._starts: Dict[str, List[torch.cuda.Event]] = {}
        self._ends: Dict[str, List[torch.cuda.Event]] = {}
        for phase in phases:
            self._starts[phase] = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_iters)
            ]
            self._ends[phase] = [
                torch.cuda.Event(enable_timing=True) for _ in range(num_iters)
            ]

    def mark(self, phase: str, iteration: int, is_start: bool):
        """Record a CUDA event. No synchronization — just enqueues the event."""
        if is_start:
            self._starts[phase][iteration].record()
        else:
            self._ends[phase][iteration].record()

    def elapsed(self) -> Dict[str, List[float]]:
        """Synchronize once and compute all elapsed times (ms)."""
        torch.cuda.synchronize()
        results = {}
        for phase in self.phases:
            results[phase] = [
                self._starts[phase][i].elapsed_time(self._ends[phase][i])
                for i in range(self.num_iters)
            ]
        return results

    def mean_elapsed(self) -> Dict[str, float]:
        """Return mean elapsed time per phase (ms)."""
        all_times = self.elapsed()
        return {
            phase: sum(times) / len(times) if times else 0.0
            for phase, times in all_times.items()
        }


class EventRecorder:
    """
    Records paired CUDA events for per-component timing inside backward passes.

    Usage:
        rec = EventRecorder()
        rec.mark('compress')   # start
        ... compress ...
        rec.mark('compress')   # end
        rec.mark('score')      # start
        ... score ...
        rec.mark('score')      # end

        # After torch.cuda.synchronize():
        rec.accumulate(accum_dict)  # adds elapsed times to accum_dict
        rec.reset()                 # clear for next iteration
    """

    def __init__(self):
        self._events = []

    def mark(self, phase: str):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events.append((phase, ev))

    def accumulate(self, accum: Dict[str, float]):
        """Parse event pairs and add elapsed times to accum dict."""
        i = 0
        while i < len(self._events) - 1:
            p1, e1 = self._events[i]
            p2, e2 = self._events[i + 1]
            if p1 == p2:
                accum[p1] = accum.get(p1, 0.0) + e1.elapsed_time(e2)
                i += 2
            else:
                i += 1

    def reset(self):
        self._events.clear()


# =============================================================================
# Data Result Structures
# =============================================================================

@dataclass
class BreakdownResult:
    """Timing breakdown for one method."""
    method: str

    # Measured phase times (ms)
    val_forward_ms: float = 0.0
    val_backward_ms: float = 0.0
    train_forward_ms: float = 0.0
    train_backward_ms: float = 0.0  # Total backward (method-specific)
    selection_decision_ms: float = 0.0  # Subset only
    pass2_forward_ms: float = 0.0  # Subset only
    pass2_backward_ms: float = 0.0  # Subset only
    optimizer_step_ms: float = 0.0

    # Decomposed components (ms)
    activation_grad_ms: float = 0.0
    score_computation_ms: float = 0.0
    wgrad_ms: float = 0.0

    # Totals
    total_step_ms: float = 0.0
    peak_memory_gb: float = 0.0

    # Instrumented backward components (directly measured, Layerwise only)
    bwd_compress_ms: float = 0.0
    bwd_score_ms: float = 0.0
    bwd_select_ms: float = 0.0
    bwd_wgrad_ms: float = 0.0
    bwd_select_wgrad_ms: float = 0.0  # Combined (for shared compressor path)


# =============================================================================
# Model & Hook Setup
# =============================================================================

def setup_model(config: BenchmarkConfig):
    """Create model and tokenizer."""
    model_kwargs = {'dtype': config.get_torch_dtype(), 'device_map': config.device}
    if config.use_flash_attention:
        model_kwargs['attn_implementation'] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    model.train()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def setup_grad_hook(
    model: nn.Module,
    config: BenchmarkConfig,
    tokenizer,
    device: str = 'cuda',
) -> GradientHook:
    """Create GradientHook with optional score compression."""
    layer_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    grad_hook = GradientHook(model=model, layer_names=layer_names, device=device)

    # Set up score compressors if configured
    if config.score_compression and config.score_compression != "none":
        from drpt.compressor import setup_model_compressors
        from drpt.utils import create_sample_inputs

        method, dim_str = config.score_compression.split("-")
        dim = int(dim_str.split("*")[0])

        sparsifier_kwargs = {
            "proj_dim": dim, "proj_max_batch_size": 64,
            "proj_seed": config.seed, "device": device, "proj_type": method,
        }
        projector_kwargs = {
            "proj_dim": -1, "proj_max_batch_size": 64,
            "proj_seed": config.seed, "device": device, "proj_type": "identity",
        }

        sample_inputs = create_sample_inputs(
            tokenizer=tokenizer, max_seq_length=config.seq_length, device=device,
        )

        score_compressors = setup_model_compressors(
            model=model, layer_names=layer_names,
            sparsifier_kwargs=sparsifier_kwargs,
            projector_kwargs=projector_kwargs,
            sample_inputs=sample_inputs,
            device=device, update_freq=1000000,
        )
        grad_hook.set_score_compressors(score_compressors)
        print(f"  Score compression: {config.score_compression} ({len(score_compressors)} layers)")
    else:
        print("  Score compression: none (exact scoring)")

    return grad_hook


def create_dataloaders(
    config: BenchmarkConfig,
    tokenizer,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders."""
    train_dataset = get_benchmark_dataset(config, tokenizer)
    train_collator = get_data_collator(tokenizer, config)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=train_collator,
    )

    val_dataset = ValidationDataset(
        config.val_dataset, tokenizer, config.seq_length, size=1000,
    )
    val_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest")
    val_loader = DataLoader(
        val_dataset, batch_size=config.val_batch_size,
        shuffle=True, collate_fn=val_collator,
    )

    return train_loader, val_loader


def get_batches(
    train_loader, val_loader, num_batches: int, device: str,
) -> Tuple[List[Dict], List[Dict]]:
    """Pre-fetch batches for deterministic benchmarking."""
    train_batches = []
    val_batches = []
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)

    for _ in range(num_batches):
        try:
            tb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            tb = next(train_iter)
        train_batches.append({k: v.to(device) for k, v in tb.items()})

        try:
            vb = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            vb = next(val_iter)
        val_batches.append({k: v.to(device) for k, v in vb.items()})

    return train_batches, val_batches
