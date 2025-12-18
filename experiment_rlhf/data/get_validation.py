"""
Validation dataset utilities for RLHF experiments.

This module provides functions to load high-quality (prompt, response) pairs
for use as validation data in layer-wise selection.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Tuple
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


def get_validation_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    split: str = "train",
    max_samples: int = 256,
    use_chosen_only: bool = True,
    max_prompt_length: int = 128,
    max_response_length: int = 256,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load validation dataset with high-quality (prompt, response) pairs.
    
    For RLHF with layer-wise selection, we need "gold standard" responses
    that represent the direction we want to optimize towards.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        max_samples: Maximum number of validation samples
        use_chosen_only: If True, only use chosen responses (for preference datasets)
        max_prompt_length: Maximum prompt length in tokens
        max_response_length: Maximum response length in tokens
        cache_dir: Cache directory
        
    Returns:
        Dataset with 'prompt' and 'response' columns
    """
    logger.info(f"Loading validation dataset: {dataset_name}")
    
    if dataset_name == "Anthropic/hh-rlhf":
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        def extract_chosen(example):
            text = example.get("chosen", "")
            
            # Parse the conversation format
            if "\n\nHuman:" in text and "\n\nAssistant:" in text:
                parts = text.split("\n\nAssistant:")
                human_part = parts[0]
                assistant_part = parts[1] if len(parts) > 1 else ""
                
                # Extract the human prompt
                if "\n\nHuman:" in human_part:
                    prompt = human_part.split("\n\nHuman:")[-1].strip()
                else:
                    prompt = human_part.strip()
                
                # Clean assistant response
                response = assistant_part.strip()
                if "\n\nHuman:" in response:
                    response = response.split("\n\nHuman:")[0].strip()
            else:
                prompt = text[:max_prompt_length]
                response = ""
            
            return {
                "prompt": prompt[:max_prompt_length * 4],  # Rough char limit
                "response": response[:max_response_length * 4],
            }
        
        dataset = dataset.map(extract_chosen)
        
        # Filter out empty responses
        dataset = dataset.filter(lambda x: len(x["response"]) > 10)
        
    elif dataset_name == "stanfordnlp/SHP":
        # Stanford Human Preferences dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        def extract_preferred(example):
            # Use the preferred response (score_A > score_B means A is preferred)
            if example.get("score_A", 0) > example.get("score_B", 0):
                response = example.get("human_ref_A", "")
            else:
                response = example.get("human_ref_B", "")
            
            return {
                "prompt": example.get("history", "")[:max_prompt_length * 4],
                "response": response[:max_response_length * 4],
            }
        
        dataset = dataset.map(extract_preferred)
        
    elif "oasst" in dataset_name.lower():
        # OpenAssistant dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        def extract_qa(example):
            # This depends on the specific format
            return {
                "prompt": example.get("instruction", example.get("text", ""))[:max_prompt_length * 4],
                "response": example.get("output", example.get("response", ""))[:max_response_length * 4],
            }
        
        dataset = dataset.map(extract_qa)
        
    else:
        # Generic loading
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        # Try to find appropriate columns
        prompt_cols = ["prompt", "instruction", "input", "question", "text"]
        response_cols = ["response", "output", "answer", "completion", "chosen"]
        
        prompt_col = None
        response_col = None
        
        for col in prompt_cols:
            if col in dataset.column_names:
                prompt_col = col
                break
        
        for col in response_cols:
            if col in dataset.column_names:
                response_col = col
                break
        
        if prompt_col is None or response_col is None:
            raise ValueError(
                f"Cannot find prompt/response columns in {dataset_name}. "
                f"Available columns: {dataset.column_names}"
            )
        
        def extract_generic(example):
            return {
                "prompt": str(example.get(prompt_col, ""))[:max_prompt_length * 4],
                "response": str(example.get(response_col, ""))[:max_response_length * 4],
            }
        
        dataset = dataset.map(extract_generic)
    
    # Keep only prompt and response columns
    columns_to_keep = ["prompt", "response"]
    columns_to_remove = [c for c in dataset.column_names if c not in columns_to_keep]
    dataset = dataset.remove_columns(columns_to_remove)
    
    # Limit samples
    if len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=42).select(range(max_samples))
    
    logger.info(f"Loaded {len(dataset)} validation samples")
    return dataset


def get_hh_rlhf_validation(
    max_samples: int = 256,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Convenience function for loading HH-RLHF validation data.

    Args:
        max_samples: Maximum validation samples
        cache_dir: Cache directory

    Returns:
        Dataset with chosen responses
    """
    return get_validation_dataset(
        dataset_name="Anthropic/hh-rlhf",
        max_samples=max_samples,
        cache_dir=cache_dir,
    )


def get_high_quality_samples(
    dataset: Dataset,
    quality_column: str = "reward",
    top_k: Optional[int] = None,
    top_fraction: float = 0.1,
) -> Dataset:
    """
    Select high-quality samples from a dataset based on a quality metric.

    This is useful for selecting validation data that represents the
    "good direction" for layer-wise selection in RLHF.

    Args:
        dataset: Input dataset with quality scores
        quality_column: Column name containing quality scores
        top_k: Select top k samples (mutually exclusive with top_fraction)
        top_fraction: Select top fraction of samples

    Returns:
        Dataset with high-quality samples
    """
    if quality_column not in dataset.column_names:
        raise ValueError(f"Quality column '{quality_column}' not found in dataset")

    # Sort by quality (descending)
    sorted_dataset = dataset.sort(quality_column, reverse=True)

    # Select top samples
    if top_k is not None:
        n_select = min(top_k, len(sorted_dataset))
    else:
        n_select = max(1, int(len(sorted_dataset) * top_fraction))

    return sorted_dataset.select(range(n_select))
