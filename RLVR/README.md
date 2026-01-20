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

### Step 2: Split Test Set for Validation

For gradient-based selection, we need validation prompts. We split the test set into:
- **Validation prompts** (1,000 samples): Used to compute validation gradients
- **Cleaned test set** (4,000 samples): Used for final evaluation

```bash
python data/prepare_data.py \
    --test_data $DATA_DIR/math/test.parquet \
    --output $DATA_DIR/math/val_prompts.parquet \
    --output_test $DATA_DIR/math/test_cleaned.parquet \
    --num_samples 1000 \
    --seed 42
```

This creates:
- `val_prompts.parquet`: Prompts only (responses generated online during training)
- `test_cleaned.parquet`: Remaining samples for final evaluation

**Important:** When evaluating, use `test_cleaned.parquet` instead of `test.parquet` to avoid evaluating on validation data.

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
