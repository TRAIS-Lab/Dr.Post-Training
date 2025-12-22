# RLHF Experiments

This folder contains the training code for Reinforcement Learning from Human Feedback with **Gradient Streaming**.

## Naming Convention

Experiments follow the pattern: `{task}-{selection}-{compression}-{model}-{training_type}-lr{lr}-b{batch}-v{nval}-s{seed}`

| Component       | Options                     | Description                                   |
| --------------- | --------------------------- | --------------------------------------------- |
| `selection`     | `NA`, `Streaming`, `GREATS` | Data selection method                         |
| `compression`   | `NA`, `LoGra`               | Gradient compression (implies MeSO optimizer) |
| `training_type` | `full`, `lora`              | Full fine-tuning or LoRA                      |

> **Note:** GraSS compression is also available (`--compression GraSS`) but not used in default experiments.

### Selection Methods
- **NA**: No data selection (baseline PPO)
- **Streaming**: Per-layer selection - each layer independently selects samples (single-pass)
- **GREATS**: Global selection - accumulates scores across all layers (two-pass)

### Compression Methods
- **NA**: No compression - uses full gradients and standard AdamW optimizer
- **LoGra**: Low-rank Gradient compression (Gaussian projection) - uses MeSO optimizer
- **GraSS**: Gradient Sparsification with Sketching (available but not used in default experiments)

## Experiment Configurations

### Full Configuration Matrix (8 experiments)

| #   | Selection | Compression | Training | Description                               |
| --- | --------- | ----------- | -------- | ----------------------------------------- |
| 1a  | NA        | NA          | full     | Baseline PPO full fine-tuning             |
| 1b  | NA        | NA          | lora     | Baseline PPO LoRA fine-tuning             |
| 2a  | Streaming | NA          | full     | Per-layer selection, full gradients       |
| 2b  | Streaming | NA          | lora     | Per-layer selection, full gradients, LoRA |
| 3a  | GREATS    | NA          | full     | Global selection, full gradients          |
| 3b  | GREATS    | NA          | lora     | Global selection, full gradients, LoRA    |
| 4   | Streaming | LoGra       | full     | Per-layer selection + MeSO                |
| 5   | GREATS    | LoGra       | full     | Global selection + MeSO                   |

> [!Note]
> Second-order interaction is enabled by default for all data selection methods (Streaming and GREATS).

## Learning Rate Configuration

The `RLHF/train/lr/` folder contains tools for finding optimal learning rates, where learning rates are managed via `RLHF/train/lr/config.json`.

### Recommended Workflow

1. **Run LR sweep** using either Grid Search or Binary Search:

```bash
# Grid Search (default) - tests multiple discrete LRs
bash RLHF/train/lr/lr_sweep.sh --mode grid --experiments all --task toxicity

# Binary Search - efficient search using golden section method
bash RLHF/train/lr/lr_sweep.sh --mode binary --experiments all --task toxicity
```

2. **Run training** with optimal LRs (loaded automatically from config.json):

```bash
bash RLHF/train/train.sh --experiments all --task toxicity
```

## Running Experiments

All experiments are launched using the unified `train.sh` script.

### Multi-Experiment Mode

Run multiple experiments with the `--experiments` flag:

```bash
# Run all 8 experiments
bash RLHF/train/train.sh --experiments all --task toxicity

# Run by category
bash RLHF/train/train.sh --experiments baseline --task toxicity      # NA-NA-full, NA-NA-lora
bash RLHF/train/train.sh --experiments streaming --task toxicity     # All Streaming-* variants
bash RLHF/train/train.sh --experiments greats --task toxicity        # All GREATS-* variants
bash RLHF/train/train.sh --experiments compression --task toxicity   # *-LoGra-* variants
bash RLHF/train/train.sh --experiments lora --task toxicity          # All *-lora variants
bash RLHF/train/train.sh --experiments full --task toxicity          # All *-full variants

# Run specific experiments
bash RLHF/train/train.sh --experiments "NA-NA-full,Streaming-LoGra-full" --task toxicity

# Combine categories
bash RLHF/train/train.sh --experiments "baseline,streaming" --task toxicity

# Dry run - preview commands without executing
bash RLHF/train/train.sh --experiments all --task toxicity --dry-run

# Submit to SLURM
bash RLHF/train/train.sh --experiments all --task toxicity --sbatch
```

#### Available Categories

| Category         | Experiments                                                |
| ---------------- | ---------------------------------------------------------- |
| `all`            | All 8 experiments                                          |
| `baseline`       | NA-NA-full, NA-NA-lora                                     |
| `streaming`      | Streaming-NA-full, Streaming-NA-lora, Streaming-LoGra-full |
| `greats`         | GREATS-NA-full, GREATS-NA-lora, GREATS-LoGra-full          |
| `full`           | All *-full experiments (5 total)                           |
| `lora`           | All *-lora experiments (3 total)                           |
| `compression`    | Streaming-LoGra-full, GREATS-LoGra-full                    |
| `no-compression` | All experiments without compression (6 total)              |

### Single Experiment Mode

Run a single experiment by specifying individual options:

```bash
# From project root
bash RLHF/train/train.sh [options]

# Or with SLURM
sbatch RLHF/train/train.sh [options]
```

#### Examples

```bash
# NA-NA-full: Baseline PPO full fine-tuning
bash RLHF/train/train.sh --task toxicity --model EleutherAI/gpt-neo-125m

# NA-NA-lora: Baseline PPO LoRA fine-tuning
bash RLHF/train/train.sh --task toxicity --model EleutherAI/gpt-neo-125m --lora

# Streaming-NA-full: Per-layer selection with full gradients (second-order enabled)
bash RLHF/train/train.sh --task toxicity --model EleutherAI/gpt-neo-125m \
    --method Streaming --use_second_order

# GREATS-NA-full: Global selection with full gradients (second-order enabled)
bash RLHF/train/train.sh --task toxicity --model EleutherAI/gpt-neo-125m \
    --method GREATS --use_second_order

# Streaming-LoGra-full: Per-layer selection + MeSO (second-order enabled)
bash RLHF/train/train.sh --task toxicity --model EleutherAI/gpt-neo-125m \
    --method Streaming --compression LoGra --use_second_order

# GREATS-LoGra-full: Global selection + MeSO (second-order enabled)
bash RLHF/train/train.sh --task toxicity --model EleutherAI/gpt-neo-125m \
    --method GREATS --compression LoGra --use_second_order
```

### Parameters

The unified training script accepts the following arguments:

#### Task Arguments

- `--task <task>` - Task: `toxicity`, `imdb` (default: `toxicity`)
- `--model <model>` - Policy model (default: `EleutherAI/gpt-neo-2.7B`)
- `--reward_model <model>` - Reward model (default: `facebook/roberta-hate-speech-dynabench-r4-target`)

#### Data Selection Arguments

- `--method <method>` - Data selection method:
  - `NA` - No selection (baseline, default)
  - `Streaming` - Per-layer selection (single-pass)
  - `GREATS` - Global selection (two-pass)
- `--use_second_order` - Enable greedy selection with second-order interactions (enabled by default for all selection methods)

#### Compression Arguments

- `--compression <method>` - Gradient compression method (implies MeSO optimizer):
  - `LoGra` - Low-rank Gradient compression (Gaussian projection, default)
  - `GraSS` - Gradient Sparsification with Sketching (available but not used in default experiments)
  - If not specified, uses full gradients and standard AdamW optimizer
- `--update_compressor_freq <steps>` - Projector refresh interval (default: `200`)

#### Core Training Arguments

- `--lr <lr>` - Learning rate override (if not specified, looked up from `lr_config.json`; fallback: `1e-5`)
- `--lr_config <path>` - LR config file path (default: `RLHF/train/lr/config.json`)
- `--batch_size <size>` - Batch size (default: `64`)
- `--max_steps <steps>` - Maximum training steps (default: `200`)
- `--seed <seed>` - Random seed (default: `42`)

#### Data Selection Arguments

- `--n_val <n>` - Validation examples for data selection (default: `16`)
- `--val_batch_size <size>` - Validation batch size for data selection (default: `32`)
- `--selection_frac <frac>` - Fraction of samples to select (default: `0.5`)

#### PPO Arguments

- `--ppo_epochs <n>` - PPO epochs per batch (default: `4`)
- `--kl_coef <coef>` - KL penalty coefficient (default: `0.04`)
- `--max_new_tokens <n>` - Maximum new tokens to generate (default: `30`)
- `--min_new_tokens <n>` - Minimum new tokens to generate (default: `20`)

#### LoRA Arguments

- `--lora` - Enable LoRA fine-tuning (flag, omit for full fine-tuning)
- `--lora_r <r>` - LoRA rank (default: `16`)
- `--lora_alpha <alpha>` - LoRA alpha (default: `32`)

#### Training Configuration

- `--max_grad_norm` - Gradient clipping (default: `0.0` = disabled)
- Learning rate: `1e-5` (fallback if not in config.json)

> **Note:** Gradient clipping is disabled by default (`max_grad_norm=0.0`). With gradient clipping enabled (e.g., `max_grad_norm=1.0`), the effective updates become too small for PPO to learn properly.

## Key Differences from SFT

| Aspect           | SFT                             | RLHF (PPO)                                |
| ---------------- | ------------------------------- | ----------------------------------------- |
| Training Loss    | Cross-entropy                   | PPO objective (policy + value + KL)       |
| Data             | Static dataset                  | Dynamic rollouts (generate responses)     |
| Validation Loss  | NLL on target task              | Sequence-level reward-weighted likelihood |
| Selection Target | Improve target task performance | Generate high-reward responses            |

> **Note:** The validation gradient is computed using sequence-level attribution: `f^seq(θ) = -E[log π_θ(y|x) * Â(x,y)]`, which points toward higher-reward sequences.

## Experiment Summary

The following experiments can be run with the commands below.

| Task     | Model        | Batch | Val Size | Max Steps | LoRA Rank |
| -------- | ------------ | ----- | -------- | --------- | --------- |
| Toxicity | gpt-neo-2.7B | 256   | 16       | 200       | 16        |

> **Note:** Learning rates are managed via `RLHF/train/lr/config.json`. Run `lr_sweep.sh` to find optimal LRs before training.

### LR Sweep Commands

Run LR sweep before full training to find optimal learning rates:

```bash
# Toxicity task
bash RLHF/train/lr/lr_sweep.sh --mode binary --experiments all --task toxicity \
    --batch_size 256 --n_val 4 --sweep_max_steps 50 --seed 2
```

### Training Commands

Training commands (LRs loaded from lr_config.json):

```bash
# Toxicity task - all experiments
bash RLHF/train/train.sh --experiments all --task toxicity \
    --batch_size 64 --n_val 128 --seed 42

# Toxicity task - baseline only
bash RLHF/train/train.sh --experiments baseline --task toxicity \
    --batch_size 64 --n_val 128 --seed 42

# Toxicity task - streaming methods
bash RLHF/train/train.sh --experiments streaming --task toxicity \
    --batch_size 64 --n_val 128 --seed 42
```

### Evaluation Commands

Evaluate trained models for toxicity:

```bash
# Evaluate all models
bash RLHF/eval/eval.sh --task toxicity --batch_size 256 --seed 82

# Evaluate specific model
python -m RLHF.eval.eval --model_path /path/to/model --n_samples 400
```

## Evaluation

The evaluation script measures model toxicity on toxic prompt completions.

### Dataset

- **Source**: `OxAISH-AL-LLM/wiki_toxic` (test split)
- **Filtering**: Only toxic prompts (label == 1)
- **Default samples**: 400

### Metrics

| Metric        | Description                                 |
| ------------- | ------------------------------------------- |
| Mean Toxicity | Average toxicity score across generations   |
| Std Toxicity  | Standard deviation of toxicity scores       |
| Max Toxicity  | Maximum toxicity score in batch             |
| Toxicity Rate | Fraction of generations with toxicity > 0.5 |

### Toxicity Classifier

Uses `facebook/roberta-hate-speech-dynabench-r4-target` (same as training reward model):
- Class 0: nothate
- Class 1: hate (toxicity score)

### Usage

```bash
# Evaluate all models in directory
bash RLHF/eval/eval.sh

# Filter by task
bash RLHF/eval/eval.sh --task toxicity

# Custom settings
bash RLHF/eval/eval.sh --n_samples 1000 --batch_size 32 --max_new_tokens 50

# Submit to SLURM
bash RLHF/eval/eval.sh --sbatch

# Dry run
bash RLHF/eval/eval.sh --dry-run
```
