# RLVR Experiment

This directory contains the implementation for online gradient-based data selection during RLHF/GRPO training, built on top of [verl](https://github.com/volcengine/verl).

## Quick Start

```bash
# 1. Prepare data
python verl/examples/data_preprocess/math_dataset.py --local_save_dir $DATA_DIR/math

# 2. Split test set into validation + cleaned test
python data/prepare_data.py \
    --test_data $DATA_DIR/math/test.parquet \
    --output $DATA_DIR/math/val_prompts.parquet \
    --output_test $DATA_DIR/math/test_cleaned.parquet \
    --num_samples 1000 \
    --seed 42

# 3. Run training with Streaming selection (default)
bash scripts/run_qwen1.7b_math_grpo.sh

# 4. Or disable selection for baseline
bash scripts/run_qwen1.7b_math_grpo.sh ++selection.enable=False
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

For gradient-based selection, we need validation prompts. There are two options:

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

To switch which validation set is used, set `VAL_PROMPTS_PATH` before running:

```bash
# Use validation from test (default)
VAL_PROMPTS_PATH=$DATA_DIR/math/val_from_test.parquet bash scripts/run_qwen1.7b_math_grpo.sh

# Use validation from train (ablation)
VAL_PROMPTS_PATH=$DATA_DIR/math/val_from_train.parquet bash scripts/run_qwen1.7b_math_grpo.sh
```

## Running Experiments

**Important:** Always run scripts from the `RLVR/` directory to ensure all output folders are created in consistent locations.

```bash
# With Streaming selection (default)
bash scripts/run_qwen1.7b_math_grpo.sh

# Baseline (no selection)
bash scripts/run_qwen1.7b_math_grpo.sh ++selection.enable=False

# With GREATS selection
SELECTION_METHOD=GREATS bash scripts/run_qwen1.7b_math_grpo.sh
```

## Configuration

### Environment Variables

| Variable            | Default   | Description                                |
| ------------------- | --------- | ------------------------------------------ |
| `N_GPUS`            | auto      | Number of GPUs                             |
| `SEED`              | 42        | Random seed for reproducibility            |
| `SELECTION_ENABLED` | True      | Enable/disable selection                   |
| `SELECTION_METHOD`  | Streaming | Selection method: `Streaming` or `GREATS`  |
| `SELECTION_FRAC`    | 1.0       | Fraction of samples to select              |
| `VAL_POOL_SIZE`     | 500       | Number of validation prompts               |
| `VAL_BATCH_SIZE`    | 32        | Batch size for validation gradient capture |
| `REFRESH_FREQ`      | 1         | How often to refresh validation gradients  |

### Hydra Overrides

Pass additional config via command line:

```bash
# Disable selection
bash scripts/run_qwen1.7b_math_grpo.sh ++selection.enable=False

# Change selection fraction
bash scripts/run_qwen1.7b_math_grpo.sh ++selection.frac=0.5

# Change model
bash scripts/run_qwen1.7b_math_grpo.sh actor_rollout_ref.model.path=Qwen/Qwen3-4B

# Set random seed for different runs
bash scripts/run_qwen1.7b_math_grpo.sh seed=123

# Multiple overrides
bash scripts/run_qwen1.7b_math_grpo.sh ++selection.method=GREATS ++selection.frac=0.8
```

## MATH Dataset Info

| Aspect              | Value                                |
| ------------------- | ------------------------------------ |
| Difficulty          | Competition level                    |
| Train samples       | ~7,500                               |
| Test samples        | ~5,000 (split: 1,000 val + 4,000 test) |
| Answer format       | `\boxed{<answer>}`                   |
| Max response length | 4096                                 |
| Data source         | `DigitalLearningGmbH/MATH-lighteval` |
