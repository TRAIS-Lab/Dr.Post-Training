# RLVR Experiment

This directory contains the implementation for online gradient-based data curation during RLHF/GRPO training, built on top of [verl](https://github.com/volcengine/verl).

## Quick Start

```bash
# 1. Prepare data
python verl/examples/data_preprocess/math_dataset.py --local_save_dir $DATA_DIR/math

# 2. Run training with Layerwise curation
bash train.sh -c configs/math -m Layerwise

# 3. Or run baseline (no curation)
bash train.sh -c configs/math -m Standard

# 4. Run all methods
bash train.sh -c configs/math -m all

# 5. Dry-run to verify commands
bash train.sh -c configs/math -m Layerwise --dry-run
```

## Data Preparation

### Step 1: Download and Preprocess MATH Dataset

```bash
python verl/examples/data_preprocess/math_dataset.py --local_save_dir $DATA_DIR/math
```

This creates:
- `$DATA_DIR/math/train.parquet` (~7,500 problems)
- `$DATA_DIR/math/test.parquet` (~5,000 problems)

### Step 2: Prepare Validation Prompts

For gradient-based curation, we need validation prompts. There are two options:

#### Option A: Validation from Test Set (Default)

Split the test set into validation prompts and cleaned test set:

```bash
python data/prepare_data.py \
    --test_data $DATA_DIR/math/test.parquet \
    --output $DATA_DIR/math/val_from_test.parquet \
    --output_test $DATA_DIR/math/test_cleaned.parquet \
    --num_samples 1000 \
    --seed 42
```

This creates:
- `val_from_test.parquet`: Validation prompts (disjoint from test_cleaned)
- `test_cleaned.parquet`: Remaining samples for final evaluation

**Important:** When evaluating, use `test_cleaned.parquet` instead of `test.parquet` to avoid evaluating on validation data.

#### Option B: Validation from Training Set (For Ablation)

Sample validation prompts from the training set:

```bash
python data/prepare_data.py \
    --train_data $DATA_DIR/math/train.parquet \
    --output $DATA_DIR/math/val_from_train.parquet \
    --num_samples 1000 \
    --seed 42
```

This creates:
- `val_from_train.parquet`: Validation prompts sampled from training data

Note: `val_from_train` can overlap with the training set (it's just a random sample, not a split).

#### Preparing Both (Recommended for Experiments)

To run ablation studies comparing validation sources, prepare both:

```bash
# 1. Split test → val_from_test + test_cleaned
python data/prepare_data.py \
    --test_data $DATA_DIR/math/test.parquet \
    --output $DATA_DIR/math/val_from_test.parquet \
    --output_test $DATA_DIR/math/test_cleaned.parquet \
    --num_samples 1000 \
    --seed 42

# 2. Sample train → val_from_train
python data/prepare_data.py \
    --train_data $DATA_DIR/math/train.parquet \
    --output $DATA_DIR/math/val_from_train.parquet \
    --num_samples 1000 \
    --seed 42
```

This creates 4 files:
| File | Source | Description |
|------|--------|-------------|
| `train.parquet` | original | Training data (unchanged) |
| `test_cleaned.parquet` | test | Test data minus validation samples |
| `val_from_test.parquet` | test | Validation prompts (disjoint from test_cleaned) |
| `val_from_train.parquet` | train | Validation prompts (can overlap with train) |

To switch which validation set is used, set `val_source` in the config:
- `from_train` (default): validation from training set
- `from_test`: validation from test set

## Running Experiments

```bash
# Layerwise curation
bash train.sh -c configs/math -m Layerwise

# Baseline (no curation)
bash train.sh -c configs/math -m Standard

# Subset curation
bash train.sh -c configs/math -m Subset

# All methods
bash train.sh -c configs/math -m all

# Override seed or learning rate
bash train.sh -c configs/math -m Layerwise --seed 123 --lr 5e-7

# Dry-run to inspect generated commands
bash train.sh -c configs/math -m all --dry-run

# List available methods
bash train.sh -c configs/math --list
```

## Configuration

Settings live in YAML config files under `configs/`. Each config directory has:
- `defaults.yaml`: shared settings inherited by all methods
- Per-method configs (e.g., `Standard.yaml`, `Layerwise.yaml`, `Subset.yaml`)

### Config Options

| Option | Default | Description |
| --- | --- | --- |
| `model` | `Qwen/Qwen3-1.7B-Base` | Model path |
| `seed` | `42` | Random seed |
| `train_batch_size` | `128` | Training batch size |
| `learning_rate` | `1e-6` | Learning rate |
| `total_epochs` | `5` | Number of epochs |
| `selection_frac` | `1.0` | Fraction of samples to curate |
| `val_pool_size` | `512` | Number of validation prompts |
| `val_batch_size` | `64` | Batch size for validation gradient capture |
| `val_loss_type` | `reward` | Validation loss: `reward` or `train-loss` |
| `val_source` | `from_train` | Validation source: `from_train` or `from_test` |
| `refresh_freq` | `1` | How often to refresh validation gradients |

### Validation Loss Types

| Name | Formula | Description |
| --- | --- | --- |
| `reward` | `-E[normalized_reward * log_prob]` | Batch-normalized reward weighting |
| `train-loss` | `-E[advantages * log_prob]` | GRPO-normalized, matches training objective |

### Legacy Script

The original `scripts/run_qwen1.7b_math_grpo.sh` is preserved for backward compatibility but delegates to `train.py`. For new experiments, use `train.sh` with config files.

## MATH Dataset Info

| Aspect              | Value                                |
| ------------------- | ------------------------------------ |
| Difficulty          | Competition level                    |
| Train samples       | ~7,500                               |
| Test samples        | ~5,000 (split: 1,000 val + 4,000 test) |
| Answer format       | `\boxed{<answer>}`                   |
| Max response length | 4096                                 |
| Data source         | `DigitalLearningGmbH/MATH-lighteval` |
