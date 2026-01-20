#!/usr/bin/env python3
"""
Prepare validation prompts for gradient-based data selection.

This script creates a validation prompts file from either:
1. Training data (legacy mode) - samples prompts from training set
2. Test data (recommended) - splits test set into validation and cleaned test

Mode 1: Sample from training data (legacy)
    python data/prepare_data.py \
        --train_data data/gsm8k/train.parquet \
        --output data/gsm8k/val_prompts.parquet \
        --num_samples 500 \
        --seed 42

Mode 2: Split test set (recommended for reproducibility)
    python data/prepare_data.py \
        --test_data data/math/test.parquet \
        --output data/math/val_prompts.parquet \
        --output_test data/math/test_cleaned.parquet \
        --num_samples 1000 \
        --seed 42

    This splits test.parquet into:
    - val_prompts.parquet: 1000 prompts for validation gradient computation
    - test_cleaned.parquet: remaining samples for final evaluation
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


def split_test_data(
    test_data_path: str,
    val_output_path: str,
    test_output_path: str,
    num_val_samples: int = 1000,
    seed: int = 42,
) -> None:
    """
    Split test data into validation prompts and cleaned test set.

    This is the recommended approach for gradient-based selection as it:
    - Ensures no overlap between validation and training data
    - Provides a true generalization signal for selection
    - Maintains reproducibility

    Args:
        test_data_path: Path to test parquet file
        val_output_path: Path to save validation prompts (prompts only)
        test_output_path: Path to save cleaned test set (full data)
        num_val_samples: Number of samples for validation
        seed: Random seed for sampling
    """
    random.seed(seed)

    logger.info(f"Loading test data from {test_data_path}...")
    df_dict = load_parquet(test_data_path)

    prompt_col = find_prompt_column(df_dict)
    num_total = len(df_dict[prompt_col])

    if num_val_samples >= num_total:
        raise ValueError(
            f"num_val_samples ({num_val_samples}) must be less than total samples ({num_total})"
        )

    logger.info(f"  Found {num_total} test samples")
    logger.info(f"  Splitting into {num_val_samples} validation + {num_total - num_val_samples} test")

    # Randomly select indices for validation
    all_indices = list(range(num_total))
    random.shuffle(all_indices)
    val_indices = set(all_indices[:num_val_samples])
    test_indices = [i for i in range(num_total) if i not in val_indices]

    # Get data source info
    if 'data_source' in df_dict and len(df_dict['data_source']) > 0:
        data_source_default = df_dict['data_source'][0]
        logger.info(f"  Using data_source from file: {data_source_default}")
    else:
        data_source_default = os.path.basename(os.path.dirname(test_data_path))
        logger.info(f"  Using folder name as data_source: {data_source_default}")

    # Build validation output (prompts only, for online generation)
    val_output_dict = {
        'prompt': [df_dict[prompt_col][i] for i in val_indices],
        'data_source': [
            df_dict['data_source'][i] if 'data_source' in df_dict else data_source_default
            for i in val_indices
        ],
    }

    # Add extra_info if available
    if 'extra_info' in df_dict:
        val_output_dict['extra_info'] = [df_dict['extra_info'][i] for i in val_indices]

    # Add reward_model if available (needed for ground_truth in reward computation)
    if 'reward_model' in df_dict:
        val_output_dict['reward_model'] = [df_dict['reward_model'][i] for i in val_indices]

    # Build cleaned test output (full data, preserving all columns)
    test_output_dict = {}
    for col in df_dict.keys():
        test_output_dict[col] = [df_dict[col][i] for i in test_indices]

    # Save both outputs
    os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
    save_parquet(val_output_dict, val_output_path)
    logger.info(f"Saved {len(val_output_dict['prompt'])} validation prompts to {val_output_path}")

    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    save_parquet(test_output_dict, test_output_path)
    logger.info(f"Saved {len(test_output_dict[prompt_col])} cleaned test samples to {test_output_path}")

    # Print summary
    if 'data_source' in val_output_dict:
        source_counts = {}
        for source in val_output_dict['data_source']:
            source_counts[source] = source_counts.get(source, 0) + 1
        logger.info(f"Validation distribution by source: {source_counts}")


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
    parser = argparse.ArgumentParser(
        description="Prepare validation prompts for gradient-based selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Sample from training data (legacy)
  python data/prepare_data.py --train_data data/gsm8k/train.parquet --output data/gsm8k/val_prompts.parquet

  # Mode 2: Split test set (recommended)
  python data/prepare_data.py --test_data data/math/test.parquet --output data/math/val_prompts.parquet --output_test data/math/test_cleaned.parquet --num_samples 1000
        """,
    )

    # Mode 1: Sample from training data
    parser.add_argument(
        "--train_data",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to training parquet file(s) (legacy mode)",
    )

    # Mode 2: Split test data
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test parquet file (recommended mode - splits into val + cleaned test)",
    )
    parser.add_argument(
        "--output_test",
        type=str,
        default=None,
        help="Output path for cleaned test parquet (required when using --test_data)",
    )

    # Common arguments
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
        help="Number of validation prompts to extract (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.test_data and args.train_data:
        parser.error("Cannot use both --test_data and --train_data. Choose one mode.")

    if args.test_data:
        # Mode 2: Split test data
        if not args.output_test:
            parser.error("--output_test is required when using --test_data")
        split_test_data(
            test_data_path=args.test_data,
            val_output_path=args.output,
            test_output_path=args.output_test,
            num_val_samples=args.num_samples,
            seed=args.seed,
        )
    elif args.train_data:
        # Mode 1: Sample from training data (legacy)
        prepare_validation_prompts(
            train_data_paths=args.train_data,
            output_path=args.output,
            num_samples=args.num_samples,
            seed=args.seed,
        )
    else:
        parser.error("Must specify either --train_data or --test_data")


if __name__ == "__main__":
    main()
