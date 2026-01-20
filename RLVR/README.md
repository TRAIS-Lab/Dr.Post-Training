# RLVR Experiment

This directory contains the implementation for online gradient-based data selection during RLHF/GRPO training, built on top of [verl](https://github.com/volcengine/verl).

## Quick Start

```bash
# 1. Prepare data (GSM8K or MATH)
python verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
python verl/examples/data_preprocess/math_dataset.py --local_save_dir data/math

# 2. Run training with Streaming selection (default)
bash scripts/run_qwen1.7b_gsm8k_grpo.sh

# 3. Or disable selection for baseline
bash scripts/run_qwen1.7b_gsm8k_grpo.sh ++selection.enable=False
```

## Data Preparation

### GSM8K (Grade School Math)

```bash
python verl/examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
```

This creates:
- `data/gsm8k/train.parquet` (~7,473 problems)
- `data/gsm8k/test.parquet` (~1,319 problems)

### MATH (Competition Mathematics)

```bash
python verl/examples/data_preprocess/math_dataset.py --local_save_dir data/math
```

This creates:
- `data/math/train.parquet` (~7,500 problems)
- `data/math/test.parquet` (~5,000 problems)

### Validation Prompts (Auto-Created)

Validation prompts are used for online gradient-based selection. They are **auto-created** when you run training with selection enabled. To create manually:

```bash
python data/prepare_data.py \
    --train_data data/gsm8k/train.parquet \
    --output data/gsm8k/val_prompts.parquet \
    --num_samples 500
```

## Running Experiments

**Important:** Always run scripts from the `RLVR/` directory to ensure all output folders are created in consistent locations. This ensures:
- `output/` - checkpoints and model outputs
- `outputs/` - Hydra configuration logs
- `wandb/` - Weights & Biases run logs

### GSM8K

```bash
# With Streaming selection (default)
bash scripts/run_qwen1.7b_gsm8k_grpo.sh

# Baseline (no selection)
bash scripts/run_qwen1.7b_gsm8k_grpo.sh ++selection.enable=False

# With GREATS selection
SELECTION_METHOD=GREATS bash scripts/run_qwen1.7b_gsm8k_grpo.sh
```

### MATH

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
bash scripts/run_qwen1.7b_gsm8k_grpo.sh ++selection.enable=False

# Change selection fraction
bash scripts/run_qwen1.7b_gsm8k_grpo.sh ++selection.frac=0.5

# Change model
bash scripts/run_qwen1.7b_gsm8k_grpo.sh actor_rollout_ref.model.path=Qwen/Qwen3-4B

# Set random seed for different runs
bash scripts/run_qwen1.7b_gsm8k_grpo.sh seed=123

# Multiple overrides
bash scripts/run_qwen1.7b_gsm8k_grpo.sh ++selection.method=GREATS ++selection.frac=0.8
```

## Dataset Comparison

| Aspect              | GSM8K           | MATH                                 |
| ------------------- | --------------- | ------------------------------------ |
| Difficulty          | Grade school    | Competition level                    |
| Answer format       | `#### <number>` | `\boxed{<answer>}`                   |
| Max response length | 1024            | 4096                                 |
| Data source         | `openai/gsm8k`  | `DigitalLearningGmbH/MATH-lighteval` |
