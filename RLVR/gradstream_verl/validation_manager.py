"""
Validation Data Manager for Online Gradient-Based Selection.

This module provides utilities for managing validation data with online rollout
generation, following the RLHF implementation design:
1. Large pool of validation prompts (no pre-generated responses)
2. Rotating batches - different validation batch each step
3. Online rollout generation using current policy
4. Online reward computation on generated responses

Usage:
    val_manager = ValidationDataManager(
        val_prompts_path="data/gsm8k/val_prompts.parquet",
        tokenizer=tokenizer,
        val_batch_size=32,
        val_pool_size=500,
    )

    # Each training step:
    val_batch = val_manager.get_next_batch()  # Rotates through pool
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Optional, List, Dict, Any

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation data management."""

    # Path to validation prompts (parquet file with prompts only)
    val_prompts_path: Optional[str] = None

    # Size of validation prompt pool
    val_pool_size: int = 500

    # Batch size for validation gradient capture
    val_batch_size: int = 32

    # Whether to shuffle validation batches
    shuffle: bool = True

    # Random seed for reproducibility
    seed: int = 42

    # Max prompt length (for filtering)
    max_prompt_length: int = 1024

    # Max response length for generation
    max_response_length: int = 1024

    # Temperature for validation rollout generation
    temperature: float = 1.0

    # Number of responses to generate per prompt (for averaging)
    n_responses: int = 1


class ValidationPromptDataset(Dataset):
    """
    Dataset of validation prompts (without responses).

    This dataset loads prompts from a parquet file and prepares them
    for rollout generation.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: "PreTrainedTokenizer",
        max_prompt_length: int = 1024,
        max_samples: int = -1,
    ):
        """
        Initialize validation prompt dataset.

        Args:
            data_path: Path to parquet file containing prompts
            tokenizer: Tokenizer for encoding prompts
            max_prompt_length: Maximum prompt length (longer prompts are filtered)
            max_samples: Maximum number of samples to load (-1 for all)
        """
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

        # Load data
        self.data = self._load_data(data_path, max_samples)
        logger.info(f"Loaded {len(self.data)} validation prompts from {data_path}")

    def _load_data(self, data_path: str, max_samples: int) -> List[Dict[str, Any]]:
        """Load and preprocess data from parquet file."""
        import pyarrow.parquet as pq

        table = pq.read_table(data_path)
        df_dict = table.to_pydict()

        # Determine the prompt column name
        prompt_col = None
        for col in ['prompt', 'question', 'input', 'query', 'text']:
            if col in df_dict:
                prompt_col = col
                break

        if prompt_col is None:
            raise ValueError(f"Could not find prompt column. Available: {list(df_dict.keys())}")

        # Check for data_source column (for multi-dataset support)
        has_data_source = 'data_source' in df_dict

        # Build dataset
        data = []
        num_samples = len(df_dict[prompt_col])

        for i in range(num_samples):
            prompt = df_dict[prompt_col][i]

            # Handle chat message format (list of dicts) vs plain string
            if isinstance(prompt, list):
                # Chat message format: use apply_chat_template
                tokens = self.tokenizer.apply_chat_template(
                    prompt, add_generation_prompt=True, tokenize=True
                )
            else:
                # Plain string format
                tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

            if len(tokens) > self.max_prompt_length:
                continue

            item = {
                'prompt': prompt,
                'input_ids': tokens,
            }

            # Add data_source if available (for reward computation)
            if has_data_source:
                item['data_source'] = df_dict['data_source'][i]

            # Add any extra metadata
            if 'extra_info' in df_dict:
                item['extra_info'] = df_dict['extra_info'][i]
            elif 'reward_model' in df_dict:
                item['reward_model'] = df_dict['reward_model'][i]

            data.append(item)

            if max_samples > 0 and len(data) >= max_samples:
                break

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def collate_validation_prompts(
    batch: List[Dict[str, Any]],
    tokenizer: "PreTrainedTokenizer",
    max_length: int = 1024,
) -> Dict[str, torch.Tensor]:
    """
    Collate validation prompts into a batch.

    Args:
        batch: List of prompt dictionaries
        tokenizer: Tokenizer for padding
        max_length: Maximum sequence length

    Returns:
        Dictionary with batched tensors
    """
    prompts = [item['prompt'] for item in batch]

    # Handle chat message format (list of dicts) vs plain string
    if prompts and isinstance(prompts[0], list):
        # Chat message format: apply chat template to get strings, then tokenize
        prompt_strings = [
            tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]
    else:
        # Plain string format
        prompt_strings = prompts

    # Set left padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    # Tokenize with padding
    encoded = tokenizer(
        prompt_strings,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )

    # Restore original padding side
    tokenizer.padding_side = original_padding_side

    result = {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'prompts': prompts,  # Keep original format for downstream use
    }

    # Add data_source if available
    if 'data_source' in batch[0]:
        result['data_source'] = [item.get('data_source', 'unknown') for item in batch]

    # Add extra_info if available
    if 'extra_info' in batch[0]:
        result['extra_info'] = [item.get('extra_info') for item in batch]

    return result


class ValidationDataManager:
    """
    Manager for validation data with rotating batches.

    This class provides:
    1. A pool of validation prompts
    2. An iterator that rotates through the pool
    3. Methods to prepare batches for rollout generation

    Unlike the fixed validation batch approach, this manager samples
    different validation batches each time, providing more diverse
    validation signal.
    """

    def __init__(
        self,
        config: ValidationConfig,
        tokenizer: "PreTrainedTokenizer",
    ):
        """
        Initialize validation data manager.

        Args:
            config: Validation configuration
            tokenizer: Tokenizer for encoding prompts
        """
        self.config = config
        self.tokenizer = tokenizer

        # Load validation prompts
        if config.val_prompts_path is None:
            raise ValueError("val_prompts_path must be specified in config")

        self.dataset = ValidationPromptDataset(
            data_path=config.val_prompts_path,
            tokenizer=tokenizer,
            max_prompt_length=config.max_prompt_length,
            max_samples=config.val_pool_size,
        )

        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.val_batch_size,
            shuffle=config.shuffle,
            collate_fn=lambda batch: collate_validation_prompts(
                batch, tokenizer, config.max_prompt_length
            ),
            drop_last=True,  # Ensure consistent batch size
        )

        # Create iterator
        self._iterator: Optional[Iterator] = None
        self._reset_iterator()

        # Statistics
        self._batches_served = 0
        self._epochs_completed = 0

        logger.info(
            f"ValidationDataManager initialized: "
            f"pool_size={len(self.dataset)}, "
            f"batch_size={config.val_batch_size}, "
            f"num_batches={len(self.dataloader)}"
        )

    def _reset_iterator(self) -> None:
        """Reset the dataloader iterator."""
        self._iterator = iter(self.dataloader)

    def get_next_batch(self) -> Dict[str, Any]:
        """
        Get the next validation batch.

        Automatically cycles through the validation pool, reshuffling
        when the iterator is exhausted.

        Returns:
            Dictionary containing:
                - input_ids: [batch_size, seq_len]
                - attention_mask: [batch_size, seq_len]
                - prompts: List of prompt strings
        """
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._epochs_completed += 1
            self._reset_iterator()
            batch = next(self._iterator)

        self._batches_served += 1
        return batch

    def get_stats(self) -> Dict[str, Any]:
        """Get validation data manager statistics."""
        return {
            'val/pool_size': len(self.dataset),
            'val/batch_size': self.config.val_batch_size,
            'val/batches_served': self._batches_served,
            'val/epochs_completed': self._epochs_completed,
        }

    @property
    def pool_size(self) -> int:
        """Get the size of the validation pool."""
        return len(self.dataset)

    @property
    def batch_size(self) -> int:
        """Get the validation batch size."""
        return self.config.val_batch_size


def prepare_validation_batch_for_verl(
    val_batch: Dict[str, Any],
    device: torch.device,
) -> "DataProto":
    """
    Prepare validation batch in VERL DataProto format.

    Args:
        val_batch: Validation batch from ValidationDataManager
        device: Target device for tensors

    Returns:
        DataProto object ready for rollout generation
    """
    from verl import DataProto

    batch_dict = {
        'input_ids': val_batch['input_ids'].to(device),
        'attention_mask': val_batch['attention_mask'].to(device),
    }

    meta_info = {
        'prompts': val_batch['prompts'],
    }

    if 'data_source' in val_batch:
        meta_info['data_source'] = val_batch['data_source']

    if 'extra_info' in val_batch:
        meta_info['extra_info'] = val_batch['extra_info']

    return DataProto(batch=batch_dict, meta_info=meta_info)


def create_validation_prompts_from_train(
    train_data_path: str,
    output_path: str,
    num_samples: int = 500,
    seed: int = 42,
) -> None:
    """
    Create validation prompts file from training data.

    This extracts prompts from the training data to create a validation
    prompt pool. The responses are NOT included - they will be generated
    online during training.

    Args:
        train_data_path: Path to training parquet file
        output_path: Path to save validation prompts parquet
        num_samples: Number of prompts to extract
        seed: Random seed for sampling
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Load training data
    table = pq.read_table(train_data_path)
    df_dict = table.to_pydict()

    # Find prompt column
    prompt_col = None
    for col in ['prompt', 'question', 'input', 'query', 'text']:
        if col in df_dict:
            prompt_col = col
            break

    if prompt_col is None:
        raise ValueError(f"Could not find prompt column. Available: {list(df_dict.keys())}")

    # Sample indices
    random.seed(seed)
    num_total = len(df_dict[prompt_col])
    indices = random.sample(range(num_total), min(num_samples, num_total))

    # Extract prompts
    output_dict = {
        'prompt': [df_dict[prompt_col][i] for i in indices],
    }

    # Copy data_source if available
    if 'data_source' in df_dict:
        output_dict['data_source'] = [df_dict['data_source'][i] for i in indices]

    # Copy extra_info/reward_model if available
    for col in ['extra_info', 'reward_model']:
        if col in df_dict:
            output_dict[col] = [df_dict[col][i] for i in indices]

    # Save to parquet
    output_table = pa.table(output_dict)
    pq.write_table(output_table, output_path)

    logger.info(f"Created validation prompts file: {output_path} ({num_samples} prompts)")
