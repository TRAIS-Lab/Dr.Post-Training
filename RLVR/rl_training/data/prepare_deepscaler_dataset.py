#!/usr/bin/env python3
"""
Download and prepare the full DeepScaleR dataset from HuggingFace.

This script:
1. Downloads the full DeepScaleR-Preview-Dataset (40,300 samples)
2. Converts it to the required verl parquet format
3. Creates train/validation splits

Usage:
    # Full dataset with 10% validation
    python prepare_deepscaler_dataset.py --val_ratio 0.1 --seed 42

    # Limited samples for quick experiments
    python prepare_deepscaler_dataset.py --train_samples 1000 --val_samples 200

    # Mixed: use ratio but limit total samples
    python prepare_deepscaler_dataset.py --val_ratio 0.1 --train_samples 5000
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def prepare_deepscaler_data(
    val_ratio: float = 0.1,
    seed: int = 42,
    output_dir: str = None,
    train_samples: int = -1,
    val_samples: int = -1,
):
    """
    Download DeepScaleR dataset and prepare train/val splits in verl format.

    Args:
        val_ratio: Fraction of data to use for validation (default: 0.1 = 10%)
        seed: Random seed for reproducibility
        output_dir: Directory to save parquet files
        train_samples: Maximum number of training samples (-1 for all)
        val_samples: Maximum number of validation samples (-1 for all)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DeepScaleR Dataset Preparation")
    print("=" * 60)

    # Download the dataset from HuggingFace
    print("\n[1/4] Downloading DeepScaleR dataset from HuggingFace...")
    ds = load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    print(f"  Downloaded {len(ds)} samples")
    print(f"  Features: {list(ds.features.keys())}")

    # Convert to pandas for easier manipulation
    df = ds.to_pandas()

    # Show sample data
    print("\n[2/4] Sample data preview:")
    sample = df.iloc[0]
    print(f"  problem: {sample['problem'][:100]}...")
    print(f"  answer: {sample['answer']}")
    print(f"  solution: {sample['solution'][:100] if sample['solution'] else 'N/A'}...")

    # Convert to verl format
    print("\n[3/4] Converting to verl parquet format...")

    # The verl format expects:
    # - prompt: list of chat messages [{"role": "user", "content": "..."}]
    # - reward_model: {"style": "rule", "ground_truth": "..."}
    # - extra_info: optional metadata

    # Create the instruction template for math problems
    instruction_suffix = (
        "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    )

    def convert_row(row, idx):
        """Convert a row from HuggingFace format to verl format."""
        problem_text = row['problem']
        answer = row['answer']

        # Create chat format prompt
        prompt = [
            {
                "role": "user",
                "content": problem_text + instruction_suffix
            }
        ]

        return {
            "prompt": prompt,
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "index": idx,
                "question": problem_text,
                "solution": row.get('solution', ''),
            },
            "data_source": "agentica-org/DeepScaleR-Preview-Dataset",
            "ability": "math",
        }

    # Convert all rows
    converted_data = [convert_row(row, idx) for idx, row in df.iterrows()]
    converted_df = pd.DataFrame(converted_data)

    print(f"  Converted {len(converted_df)} samples to verl format")

    # Shuffle and split
    print("\n[4/4] Creating train/validation splits...")

    # Shuffle with seed for reproducibility
    converted_df = converted_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Determine split sizes
    total_samples = len(converted_df)

    if train_samples > 0 and val_samples > 0:
        # Both specified explicitly
        n_train = min(train_samples, total_samples)
        n_val = min(val_samples, total_samples - n_train)
        print(f"  Using explicit sample counts: train={train_samples}, val={val_samples}")
    elif train_samples > 0:
        # Train specified, use ratio for val from remaining
        n_train = min(train_samples, total_samples)
        remaining = total_samples - n_train
        n_val = int(remaining * val_ratio / (1 - val_ratio)) if val_ratio < 1 else remaining
        n_val = min(n_val, remaining)
        print(f"  Using train_samples={train_samples}, val from ratio ({val_ratio})")
    elif val_samples > 0:
        # Val specified, use ratio for train from remaining
        n_val = min(val_samples, total_samples)
        remaining = total_samples - n_val
        n_train = remaining
        print(f"  Using val_samples={val_samples}, train from remaining")
    else:
        # Use ratio for both (default behavior)
        n_val = int(total_samples * val_ratio)
        n_train = total_samples - n_val
        print(f"  Using val_ratio={val_ratio} for split")

    # Create splits
    train_df = converted_df.iloc[:n_train].reset_index(drop=True)
    val_df = converted_df.iloc[n_train:n_train + n_val].reset_index(drop=True)

    # Update indices after split
    for i in range(len(train_df)):
        train_df.at[i, 'extra_info'] = {
            **train_df.iloc[i]['extra_info'],
            'index': i
        }
    for i in range(len(val_df)):
        val_df.at[i, 'extra_info'] = {
            **val_df.iloc[i]['extra_info'],
            'index': i
        }

    print(f"  Train samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")

    # Save to parquet
    train_path = output_dir / "deepscaler_train.parquet"
    val_path = output_dir / "deepscaler_val.parquet"
    full_path = output_dir / "deepscaler_full.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    converted_df.to_parquet(full_path, index=False)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Train:      {train_path} ({len(train_df)} samples)")
    print(f"  Validation: {val_path} ({len(val_df)} samples)")
    print(f"  Full:       {full_path} ({len(converted_df)} samples)")

    # Print file sizes
    for p in [train_path, val_path, full_path]:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name}: {size_mb:.2f} MB")

    print("\nUsage in training scripts:")
    print(f"  data.train_files=data/deepscaler_train.parquet")
    print(f"  data.val_files=data/deepscaler_val.parquet")

    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DeepScaleR dataset for RLVR training"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1 = 10%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for parquet files (default: same as script)"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=-1,
        help="Maximum number of training samples (-1 for all, default: -1)"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=-1,
        help="Maximum number of validation samples (-1 for all, default: -1)"
    )

    args = parser.parse_args()

    prepare_deepscaler_data(
        val_ratio=args.val_ratio,
        seed=args.seed,
        output_dir=args.output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
    )


if __name__ == "__main__":
    main()
