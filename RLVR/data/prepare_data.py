#!/usr/bin/env python3
"""
Prepare validation prompts for gradient-based data selection.

This script creates a validation prompts file from training data.
The prompts will be used for online rollout generation during training.

Usage:
    python data/prepare_data.py \
        --train_data data/gsm8k/train.parquet \
        --output data/gsm8k/val_prompts.parquet \
        --num_samples 500 \
        --seed 42
"""

import argparse
import logging
import os
import random
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_parquet(path: str) -> dict:
    """Load parquet file and return as dict."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    return table.to_pydict()


def save_parquet(data: dict, path: str) -> None:
    """Save dict as parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table(data)
    pq.write_table(table, path)


def find_prompt_column(df_dict: dict) -> str:
    """Find the prompt column in the data."""
    for col in ['prompt', 'question', 'input', 'query', 'text']:
        if col in df_dict:
            return col
    raise ValueError(f"Could not find prompt column. Available: {list(df_dict.keys())}")


def prepare_validation_prompts(
    train_data_paths: list,
    output_path: str,
    num_samples: int = 500,
    seed: int = 42,
    samples_per_source: dict = None,
) -> None:
    """
    Prepare validation prompts from training data.

    Args:
        train_data_paths: List of paths to training parquet files
        output_path: Path to save validation prompts
        num_samples: Total number of prompts to extract
        seed: Random seed for sampling
        samples_per_source: Optional dict mapping source name to sample count
    """
    random.seed(seed)

    all_prompts = []
    all_data_sources = []
    all_extra_info = []
    all_reward_model = []

    for data_path in train_data_paths:
        logger.info(f"Loading {data_path}...")
        df_dict = load_parquet(data_path)

        prompt_col = find_prompt_column(df_dict)
        num_total = len(df_dict[prompt_col])

        # Determine data source name - use the data_source field from the data if available
        # This is important because reward functions expect specific data_source names
        # (e.g., 'openai/gsm8k' not 'gsm8k', 'DigitalLearningGmbH/MATH-lighteval' not 'math')
        if 'data_source' in df_dict and len(df_dict['data_source']) > 0:
            # Get the first data_source value (assuming all rows have the same source)
            data_source_default = df_dict['data_source'][0]
            logger.info(f"  Using data_source from file: {data_source_default}")
        else:
            # Fall back to folder name if no data_source column
            data_source_default = os.path.basename(os.path.dirname(data_path))  # e.g., 'gsm8k', 'math'
            logger.info(f"  Using folder name as data_source: {data_source_default}")

        logger.info(f"  Found {num_total} prompts")

        for i in range(num_total):
            all_prompts.append(df_dict[prompt_col][i])
            # Use per-row data_source if available, otherwise use default
            if 'data_source' in df_dict:
                all_data_sources.append(df_dict['data_source'][i])
            else:
                all_data_sources.append(data_source_default)

            # Store extra_info if available
            if 'extra_info' in df_dict:
                all_extra_info.append(df_dict['extra_info'][i])
            else:
                all_extra_info.append(None)

            # Store reward_model if available (contains ground_truth for reward computation)
            if 'reward_model' in df_dict:
                all_reward_model.append(df_dict['reward_model'][i])
            else:
                all_reward_model.append(None)

    # Sample from combined data
    total_available = len(all_prompts)
    num_to_sample = min(num_samples, total_available)

    logger.info(f"Sampling {num_to_sample} prompts from {total_available} total...")

    indices = random.sample(range(total_available), num_to_sample)

    # Build output data
    output_dict = {
        'prompt': [all_prompts[i] for i in indices],
        'data_source': [all_data_sources[i] for i in indices],
    }

    # Add extra_info if any were found
    if any(info is not None for info in all_extra_info):
        output_dict['extra_info'] = [all_extra_info[i] for i in indices]

    # Add reward_model if any were found (needed for ground_truth in reward computation)
    if any(rm is not None for rm in all_reward_model):
        output_dict['reward_model'] = [all_reward_model[i] for i in indices]

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_parquet(output_dict, output_path)

    # Print summary
    source_counts = {}
    for source in output_dict['data_source']:
        source_counts[source] = source_counts.get(source, 0) + 1

    logger.info(f"Saved {num_to_sample} validation prompts to {output_path}")
    logger.info(f"Distribution by source: {source_counts}")


def main():
    parser = argparse.ArgumentParser(description="Prepare validation prompts for gradient-based selection")
    parser.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to training parquet file(s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for validation prompts parquet",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of prompts to extract (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    prepare_validation_prompts(
        train_data_paths=args.train_data,
        output_path=args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
