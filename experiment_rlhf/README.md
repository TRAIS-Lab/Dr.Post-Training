# RLHF Experiments with Layer-wise Selection

This directory contains code for Reinforcement Learning from Human Feedback (RLHF) experiments using the layer-wise selection streaming paradigm.

## Overview

We integrate the layer-wise selection method into PPO training for RLHF. The key insight is that different layers may benefit from different data subsets, and we can perform this selection efficiently during the gradient computation stream.

### Two-Pass Approach

Since RLHF uses different losses for training (PPO) and validation (likelihood), we use a two-pass approach:

1. **Phase 1 (capture_val)**: Forward/backward on validation data to capture compressed gradients representing the "good direction"
2. **Phase 2 (train_select)**: Forward/backward on training data with PPO loss, selecting samples layer-by-layer based on alignment with validation gradients

## Directory Structure

```
experiment_rlhf/
├── __init__.py
├── README.md
├── train.sh                 # Shell script for single experiments
├── launch_experiments.sh    # Launch all configurations
├── train_rlhf.py           # Main training script
├── ppo_trainer.py          # PPO trainer with selection
├── selection_mixin.py      # Layer-wise selection mixin
├── data/                   # Dataset utilities
│   ├── get_prompts.py      # Prompt datasets (RealToxicityPrompts, etc.)
│   └── get_validation.py   # Validation datasets (HH-RLHF, etc.)
├── eval/                   # Evaluation utilities
│   └── toxicity_eval.py    # Toxicity evaluation
└── outputs/                # Training outputs
```

## Quick Start

### Launch All Experiments

```bash
# Dry run - see all commands
bash experiment_rlhf/launch_experiments.sh --dry-run

# Run all experiments locally
bash experiment_rlhf/launch_experiments.sh

# Submit to SLURM
bash experiment_rlhf/launch_experiments.sh --sbatch

# Customize hyperparameters
bash experiment_rlhf/launch_experiments.sh \
    --model llama3-1b \
    --steps 500 \
    --per_device_train_batch_size 64 \
    --val_size 512 \
    --epochs 2 \
    --lr 1e-5 \
    --lr_lora 2e-4 \
    --selection_mode per_layer \
    --val_loss_type reward_weighted \
    --dry-run
```

This launches **10 configurations**:
- 2 Baseline PPO (LoRA + Full Fine-tuning)
- 4 GREATS Layer-wise Selection (LoRA/Full × GraSS/LoGra)
- 2 MeSO + GREATS (Full × GraSS/LoGra)
- 2 Second-Order Selection variants

### Single Experiment

```bash
# Baseline PPO with LoRA (no selection)
bash experiment_rlhf/train.sh \
    --model llama3-1b \
    --lora \
    --selection_method NA

# GREATS Layer-wise Selection with GraSS (per-layer mode)
bash experiment_rlhf/train.sh \
    --model llama3-1b \
    --lora \
    --selection_method GREATS \
    --selection_mode per_layer \
    --compression GraSS

# GREATS Global Selection with GraSS
bash experiment_rlhf/train.sh \
    --model llama3-1b \
    --lora \
    --selection_method GREATS \
    --selection_mode global \
    --compression GraSS

# MeSO + GREATS (Full Fine-tuning)
bash experiment_rlhf/train.sh \
    --model llama3-1b \
    --no_lora \
    --selection_method GREATS \
    --compression GraSS \
    --MeSO

# Second-order selection (more accurate, slower)
bash experiment_rlhf/train.sh \
    --model llama3-1b \
    --lora \
    --selection_method GREATS \
    --compression GraSS \
    --use_second_order

# Custom validation batch size (independent of val_size)
bash experiment_rlhf/train.sh \
    --model llama3-1b \
    --lora \
    --selection_method GREATS \
    --val_size 1024 \
    --val_batch_size 128 \
    --compression GraSS
```

## Parameters

### Model Options

| Argument  | Description      | Default                                |
| --------- | ---------------- | -------------------------------------- |
| `--model` | Model identifier | `llama3-1b`, `llama2-7b`, `mistral-7b` |

### Selection Options

| Argument             | Description                                    | Default                                            |
| -------------------- | ---------------------------------------------- | -------------------------------------------------- |
| `--selection_method` | Selection method                               | `NA`, `GREATS`, `GradNorm`                         |
| `--selection_frac`   | Fraction of samples to select                  | `0.5`                                              |
| `--selection_mode`   | Selection granularity: `global` or `per_layer` | `per_layer`                                        |
| `--val_loss_type`    | Validation loss type                           | `logprob`, `reward_weighted`, `advantage_weighted` |
| `--use_second_order` | Use second-order selection                     | flag                                               |

### Compression Options

| Argument        | Description                   | Default          |
| --------------- | ----------------------------- | ---------------- |
| `--compression` | Compression method            | `GraSS`, `LoGra` |
| `--MeSO`        | Use MeSO compressed optimizer | flag             |

### Training Options (consistent with SFT naming)

| Argument                        | Description                             | Default       |
| ------------------------------- | --------------------------------------- | ------------- |
| `--lr`                          | Learning rate (for full fine-tuning)    | `5e-6`        |
| `--lr_lora`                     | Learning rate (for LoRA)                | `1.41e-5`     |
| `--per_device_train_batch_size` | Training batch size (per device)        | `32`          |
| `--mini_batch_size`             | PPO mini-batch size                     | `4`           |
| `--val_size`                    | Total validation pool size              | `256`         |
| `--val_batch_size`              | Validation batch size for selection     | `val_size`    |
| `--epochs`                      | Epochs per PPO step                     | `1`           |
| `--steps`                       | Total training steps                    | `1000`        |
| `--seed`                        | Random seed                             | `42`          |
| `--log_interval`                | Log metrics every N steps               | `steps // 10` |

### LoRA Options

| Argument       | Description      | Default |
| -------------- | ---------------- | ------- |
| `--lora`       | Enable LoRA      | on      |
| `--no_lora`    | Full fine-tuning | -       |
| `--lora_r`     | LoRA rank        | `16`    |
| `--lora_alpha` | LoRA alpha       | `32`    |

### PPO Options

| Argument    | Description            | Default |
| ----------- | ---------------------- | ------- |
| `--kl_coef` | KL penalty coefficient | `0.1`   |

## Algorithm

The layer-wise selection for RLHF works as follows:

```
For each PPO step:
    1. Generate rollouts (prompts → responses → rewards)
    2. Compute PPO quantities (logprobs, values, advantages)

    3. === Phase 1: Capture Validation Gradients ===
       hook.set_mode("capture_val")
       Forward/backward on high-quality validation samples
       → Compressed val gradients stored per layer

    4. === Phase 2: PPO Update with Selection ===
       hook.set_mode("train_select")
       For each minibatch:
           Forward on training samples
           Backward with PPO loss
           → At each layer:
              - Compute compressed train gradients
              - Score = train_grad · val_grad
              - Select top-k samples
              - Reduce selected gradients
           optimizer.step()
```

### Selection Modes

| Mode        | Description                                                                   | Pros                                      | Cons                                                                         |
| ----------- | ----------------------------------------------------------------------------- | ----------------------------------------- | ---------------------------------------------------------------------------- |
| `per_layer` | Each layer independently selects samples based on that layer's gradients      | Single-pass, memory-efficient             | Different layers may select different samples (block-diagonal approximation) |
| `global`    | Accumulates scores across all layers, selects globally, then does second pass | Consistent sample selection across layers | Requires two passes, more computation                                        |

### First-Order vs Second-Order Selection

| Method                                 | Description                                           | Speed        | Accuracy                          |
| -------------------------------------- | ----------------------------------------------------- | ------------ | --------------------------------- |
| First-order (`--use_second_order` off) | Simple top-k selection based on scores                | ~200x faster | Good                              |
| Second-order (`--use_second_order` on) | Greedy selection considering sample-sample similarity | Slower       | Better (avoids redundant samples) |

## Naming Consistency with SFT

The RLHF codebase uses consistent naming with the SFT (`experiment/`) codebase:

| Purpose                        | SFT (`experiment/`)            | RLHF (`experiment_rlhf/`)     |
| ------------------------------ | ------------------------------ | ----------------------------- |
| Train batch size               | `per_device_train_batch_size`  | `per_device_train_batch_size` |
| Validation pool size           | `n_val`                        | `val_size`                    |
| Validation batch for selection | `val_batch_size_for_selection` | `val_batch_size`              |
| Training epochs                | `num_train_epochs`             | `epochs`                      |
| Selection mode                 | `selection_mode`               | `selection_mode`              |
| Second-order selection         | `use_second_order`             | `use_second_order`            |
| Selection fraction             | `selection_frac`               | `selection_frac`              |
| Selection method               | `method`                       | `selection_method`            |

## Key Components

### `PPOTrainerWithSelection`

Wrapper class that adds layer-wise selection to any PPO trainer:

```python
from experiment_rlhf import PPOTrainerWithSelection
from compress_gradient import GradientHook

# Setup gradient hook
grad_hook = GradientHook(model, layer_names, device)

# Wrap PPO trainer
selection_trainer = PPOTrainerWithSelection(
    ppo_trainer=ppo_trainer,
    grad_hook=grad_hook,
    selection_frac=0.5,
    selection_method="GREATS",
    val_loss_type="logprob",
)

# Training step
stats = selection_trainer.step_with_layerwise_selection(
    queries, responses, scores,
    val_queries, val_responses, val_scores,
    timing, output_dir,
)
```

### Validation Loss Types

The validation loss type determines how the "good direction" is computed for GREATS selection:

- **`logprob`**: Simple negative log-likelihood on validation prompts (default). Fast but may not align well with RLHF rewards.
- **`reward_weighted`**: Generates responses for validation prompts, weights log-likelihood by reward. **Recommended for RLHF** as it aligns the selection direction with the reward objective.
- **`advantage_weighted`**: Log-likelihood weighted by advantage (similar to PPO policy gradient direction).

**Note**: Investigation shows that `logprob` has essentially zero correlation with rewards (~0.06), meaning GREATS selection with `logprob` validation selects samples based on a direction unrelated to reward improvement. Use `reward_weighted` for better alignment with RLHF objectives.

## Getting Started

### 1. Install Dependencies

```bash
# From the project root
pip install -r requirements.txt

# Additional dependencies for RLHF
pip install trl accelerate
```

### 2. Data Preparation

**No manual download required!** The datasets are automatically downloaded from HuggingFace:

| Dataset             | Source                          | Purpose                                  |
| ------------------- | ------------------------------- | ---------------------------------------- |
| RealToxicityPrompts | `allenai/real-toxicity-prompts` | Prompt dataset for toxicity reduction    |
| HH-RLHF             | `Anthropic/hh-rlhf`             | Validation data (high-quality responses) |

The first run will download and cache the datasets. Subsequent runs will use the cached version.

```python
# Test data loading
from experiment_rlhf.data import build_toxicity_promptdata
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
train_ds, val_ds = build_toxicity_promptdata(tokenizer, num_samples=1024)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
```

### 3. Run a Quick Test

```bash
# Quick test with small model (few minutes)
cd experiment_rlhf
python train_rlhf.py \
    --model_name EleutherAI/gpt-neo-125m \
    --per_device_train_batch_size 32 \
    --steps 10 \
    --no_selection \
    --output_dir outputs/test_run

# With layer-wise selection
python train_rlhf.py \
    --model_name EleutherAI/gpt-neo-125m \
    --per_device_train_batch_size 32 \
    --steps 10 \
    --selection_method GREATS \
    --selection_frac 0.5 \
    --output_dir outputs/test_selection
```

### 4. Full Training

```bash
# Standard PPO baseline (reference toxicity experiment)
accelerate launch train_rlhf.py \
    --model_name EleutherAI/gpt-neo-2.7B \
    --per_device_train_batch_size 256 \
    --mini_batch_size 1 \
    --epochs 4 \
    --steps 200 \
    --init_kl_coef 0.04 \
    --learning_rate 1e-5 \
    --no_selection \
    --output_dir outputs/ppo_baseline

# With layer-wise selection
accelerate launch train_rlhf.py \
    --model_name EleutherAI/gpt-neo-2.7B \
    --per_device_train_batch_size 256 \
    --mini_batch_size 1 \
    --epochs 4 \
    --steps 200 \
    --init_kl_coef 0.04 \
    --selection_method GREATS \
    --selection_frac 0.5 \
    --output_dir outputs/ppo_selection
```

## Datasets

### Prompt Datasets
- `allenai/real-toxicity-prompts` (default for toxicity reduction)
- Custom prompt datasets via `--dataset_name`

### Validation Datasets
- `Anthropic/hh-rlhf` (default, uses chosen responses)
- `stanfordnlp/SHP` (Stanford Human Preferences)
- Custom datasets via data module

## Evaluation

Toxicity evaluation using local classifier or Perspective API:

```python
from experiment_rlhf.eval import evaluate_toxicity

results = evaluate_toxicity(
    model=model,
    tokenizer=tokenizer,
    prompts=test_prompts,
    max_new_tokens=48,
)

print(f"Mean toxicity: {results['mean_toxicity']:.4f}")
print(f"Toxicity rate: {results['toxicity_rate']:.2%}")
```

## Reference

This implementation is based on the layer-wise selection streaming paradigm. See the main README for the theoretical background.
