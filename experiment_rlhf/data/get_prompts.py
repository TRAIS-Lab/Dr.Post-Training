"""
Dataset utilities for RLHF prompt data.

This module provides functions to load and prepare prompts for RLHF training,
compatible with TRL's PPOTrainer format.

The key function `build_toxicity_promptdata` mirrors the reference repo's
implementation and returns datasets in the format expected by TRL.
"""

import os
import logging
from typing import Optional, Tuple
from datasets import load_dataset, Dataset
import torch

logger = logging.getLogger(__name__)


def build_toxicity_promptdata(
    tokenizer,
    input_min_text_length: int = 10,
    input_max_text_length: int = 15,
    num_samples: Optional[int] = None,
    seed: int = 42,
    val_strategy: str = 'random',
    toxicity_threshold: float = 0.3,
) -> Tuple[Dataset, Dataset]:
    """
    Build toxicity prompt dataset for PPO training.

    This function loads RealToxicityPrompts, filters by toxicity, tokenizes,
    and returns train/validation splits in TRL-compatible format.

    Args:
        tokenizer: HuggingFace tokenizer
        input_min_text_length: Minimum number of tokens in prompt
        input_max_text_length: Maximum number of tokens in prompt
        num_samples: Number of validation samples
        seed: Random seed for splitting
        val_strategy: 'random' or 'top' (most toxic prompts for validation)
        toxicity_threshold: Minimum toxicity score to include prompt

    Returns:
        Tuple of (train_dataset, valid_dataset) in TRL format with columns:
        - input_ids: Tokenized prompt
        - query: Text prompt
    """
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading RealToxicityPrompts dataset...")
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    # Filter by toxicity threshold
    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > toxicity_threshold

    ds = ds.filter(filter_fn, batched=False)
    logger.info(f"After toxicity filtering: {len(ds)} samples")

    # Import LengthSampler from TRL for variable-length tokenization
    try:
        from trl.core import LengthSampler
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
    except ImportError:
        # Fallback if LengthSampler not available
        import random
        def input_size():
            return random.randint(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        """Tokenize prompt and continuation, truncate to sampled length."""
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        # Tokenize full text and truncate to random length
        full_text = prompt + continuation
        sample["input_ids"] = tokenizer.encode(full_text)[:input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    # Split into train and validation
    ds = ds.train_test_split(test_size=0.2, shuffle=False, seed=seed)
    ds_train = ds['train']

    # Select validation samples based on strategy
    if num_samples is None:
        num_samples = min(1024, len(ds['test']))

    if val_strategy == 'random':
        ds_valid = ds['test'].select(range(min(num_samples, len(ds['test']))))
    elif val_strategy == 'top':
        # Select the most toxic prompts for validation
        logger.info('Selecting most toxic prompts for validation')
        test_split = ds["test"].map(lambda x: {"toxicity": x["prompt"]["toxicity"]})
        ds_valid = test_split.sort("toxicity", reverse=True).select(range(num_samples))
    else:
        raise ValueError(f"Unknown val_strategy: {val_strategy}")

    logger.info(f"Train samples: {len(ds_train)}, Validation samples: {len(ds_valid)}")
    return ds_train, ds_valid


def build_imdb_promptdata(
    tokenizer,
    split: str = "train",
    num_samples: Optional[int] = None,
    seed: int = 42,
    input_min_text_length: int = 2,
    input_max_text_length: int = 8,
) -> Dataset:
    """
    Build IMDB prompt dataset for sentiment control experiments.

    Args:
        tokenizer: HuggingFace tokenizer
        split: Dataset split ('train' or 'test')
        num_samples: Number of samples (for test set)
        seed: Random seed
        input_min_text_length: Minimum tokens
        input_max_text_length: Maximum tokens

    Returns:
        Dataset in TRL format
    """
    ds = load_dataset("stanfordnlp/imdb", split="train")
    ds = ds.rename_columns({"text": "review"})

    # Filter for minimum length
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    if num_samples is not None and split == 'test':
        ds = ds.shuffle(seed=seed).select(range(num_samples))

    try:
        from trl.core import LengthSampler
        input_size = LengthSampler(input_min_text_length, input_max_text_length)
    except ImportError:
        import random
        def input_size():
            return random.randint(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    return ds


def get_prompt_dataset(
    dataset_name: str,
    tokenizer,
    **kwargs,
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load prompt dataset by name.

    Args:
        dataset_name: Name of dataset ('toxicity', 'imdb', etc.)
        tokenizer: HuggingFace tokenizer
        **kwargs: Additional arguments passed to specific loader

    Returns:
        Tuple of (train_dataset, valid_dataset) or (dataset, None)
    """
    if "toxicity" in dataset_name.lower():
        return build_toxicity_promptdata(tokenizer, **kwargs)
    elif "imdb" in dataset_name.lower():
        ds = build_imdb_promptdata(tokenizer, **kwargs)
        return ds, None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collator(data):
    """
    Data collator for PPO training.

    Pads input_ids to the same length and stacks as tensors.
    Other fields are returned as lists.
    """
    result = {}

    for key in data[0]:
        values = [d[key] for d in data]
        if key == "input_ids":
            # Pad input_ids to same length and stack
            from torch.nn.utils.rnn import pad_sequence
            # Convert to tensors if needed
            tensors = [torch.tensor(v) if not isinstance(v, torch.Tensor) else v for v in values]
            # Pad to max length in batch (pad_sequence expects batch_first=True for left padding)
            result[key] = pad_sequence(tensors, batch_first=True, padding_value=0)
        else:
            # Keep other fields as lists
            result[key] = values

    return result
