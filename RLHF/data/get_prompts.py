"""
Prompt dataset loading for RLHF experiments.

This module provides utilities for loading prompt datasets for PPO training.
Currently supports:
- toxicity: Real Toxicity Prompts dataset (allenai/real-toxicity-prompts)
"""

from typing import Optional, Tuple

from datasets import Dataset, load_dataset


def get_prompt_dataset(
    task: str,
    tokenizer,
    input_min_length: int = 5,
    input_max_length: int = 15,
    seed: int = 42,
) -> Dataset:
    """
    Load prompt dataset for RLHF training.

    Args:
        task: Task name ('toxicity', etc.)
        tokenizer: Tokenizer for encoding prompts
        input_min_length: Minimum prompt length in tokens
        input_max_length: Maximum prompt length in tokens
        seed: Random seed for shuffling

    Returns:
        Training dataset with 'input_ids' and 'query' columns
    """
    if task == "toxicity":
        return _load_toxicity_prompts(
            tokenizer=tokenizer,
            input_min_length=input_min_length,
            input_max_length=input_max_length,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown task: {task}. Supported: ['toxicity']")


def _load_toxicity_prompts(
    tokenizer,
    input_min_length: int = 5,
    input_max_length: int = 15,
    seed: int = 42,
    toxicity_threshold: float = 0.3,
) -> Dataset:
    """
    Load Real Toxicity Prompts dataset for detoxification task.

    The goal is to train the model to generate less toxic continuations
    given potentially toxic prompts.

    Args:
        tokenizer: Tokenizer for encoding
        input_min_length: Minimum prompt length
        input_max_length: Maximum prompt length
        seed: Random seed
        toxicity_threshold: Minimum toxicity score for filtering

    Returns:
        Dataset with 'input_ids' and 'query' columns
    """
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")

    # Filter for prompts with sufficient toxicity
    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > toxicity_threshold

    ds = ds.filter(filter_fn, batched=False)

    # Length sampler for variable input lengths
    import random
    random.seed(seed)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        # Sample input length
        input_size = random.randint(input_min_length, input_max_length)

        # Encode prompt + continuation, truncate to sampled length
        sample["input_ids"] = tokenizer.encode(prompt + continuation)[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    # Shuffle and return
    ds = ds.shuffle(seed=seed)

    return ds


def collator(data):
    """
    Collate function for prompt datasets.

    Handles variable-length sequences by returning lists.
    """
    return {key: [d[key] for d in data] for key in data[0]}
