# RLHF Experiments

This folder contains the training code for Reinforcement Learning from Human Feedback with gradient streaming data selection.

## Naming Convention

Experiments follow the pattern: `{task}-{method}-{compression}-{model}-{training_type}-lr{lr}-b{batch}-v{nval}-s{seed}`

| Component       | Options                     | Description                                   |
| --------------- | --------------------------- | --------------------------------------------- |
| `method`        | `NA`, `Streaming`, `GREATS` | Data selection method                         |
| `compression`   | `NA`, `GraSS`, `LoGra`      | Gradient compression (implies MeSO optimizer) |
| `training_type` | `full`, `lora`              | Full fine-tuning or LoRA                      |

### Selection Methods
- **NA**: Baseline PPO (no data selection)
- **Streaming**: Per-layer selection - each layer independently selects samples (single-pass)
- **GREATS**: Global selection - accumulates scores across all layers (two-pass)

### Compression Methods
- **NA**: No compression - uses full gradients and standard AdamW optimizer
- **GraSS**: Gradient Sparsification with Sketching - uses MeSO optimizer
- **LoGra**: Low-rank Gradient compression - uses MeSO optimizer

## Experiment Configurations

### Full Configuration Matrix

| #   | Method    | Compression | Training | Description                               |
| --- | --------- | ----------- | -------- | ----------------------------------------- |
| 1a  | NA        | NA          | lora     | Baseline PPO with LoRA                    |
| 1b  | NA        | NA          | full     | Baseline PPO full fine-tuning             |
| 2a  | Streaming | NA          | lora     | Per-layer selection, full gradients, LoRA |
| 2b  | Streaming | NA          | full     | Per-layer selection, full gradients       |
| 3a  | GREATS    | NA          | lora     | Global selection, full gradients, LoRA    |
| 3b  | GREATS    | NA          | full     | Global selection, full gradients          |
| 4a  | Streaming | GraSS       | lora     | Per-layer selection + MeSO (GraSS)        |
| 4b  | GREATS    | GraSS       | lora     | Global selection + MeSO (GraSS)           |

## Running Experiments

### Launch All Configurations

```bash
# Dry run (see all commands without executing)
bash RLHF/train/launch_experiments.sh --task toxicity --dry-run

# Run all experiments
bash RLHF/train/launch_experiments.sh --task toxicity \
    --lr 1e-5 --batch_size 64 --n_val 16 --max_steps 200

# Submit to SLURM
bash RLHF/train/launch_experiments.sh --task toxicity --sbatch
```

### Launch Single Configuration

```bash
# Baseline PPO (no selection)
bash RLHF/train/train.sh --task toxicity --method NA

# Streaming (per-layer selection)
bash RLHF/train/train.sh --task toxicity --method Streaming --selection_frac 0.5

# GREATS (global selection)
bash RLHF/train/train.sh --task toxicity --method GREATS --selection_frac 0.5

# With compression
bash RLHF/train/train.sh --task toxicity --method Streaming --sparsification Rademacher-64*64
```

### Full Example

```bash
bash RLHF/train/train.sh \
    --task toxicity \
    --method Streaming \
    --model EleutherAI/gpt-neo-125m \
    --batch_size 64 \
    --lr 1e-5 \
    --n_val 16 \
    --selection_frac 0.5 \
    --max_steps 200 \
    --lora \
    --sparsification Rademacher-64*64 \
    --seed 42
```

### Command Line Arguments

| Argument               | Default                                          | Description                                    |
| ---------------------- | ------------------------------------------------ | ---------------------------------------------- |
| `--task`               | `toxicity`                                       | Task name                                      |
| `--method`             | `NA`                                             | Selection method: NA, Streaming, GREATS        |
| `--model`              | `EleutherAI/gpt-neo-125m`                        | Policy model                                   |
| `--reward_model`       | `facebook/roberta-hate-speech-dynabench-r4-target` | Reward model                                 |
| `--batch_size`         | `64`                                             | Training batch size                            |
| `--lr`                 | `1e-5`                                           | Learning rate                                  |
| `--n_val`              | `16`                                             | Number of validation samples                   |
| `--selection_frac`     | `0.5`                                            | Fraction of samples to select                  |
| `--max_steps`          | `200`                                            | Maximum training steps                         |
| `--lora` / `--no_lora` | `--lora`                                         | Enable/disable LoRA                            |
| `--sparsification`     | (none)                                           | Sparsification: e.g., `Rademacher-64*64`       |
| `--projection`         | (none)                                           | Projection: e.g., `Gaussian-256`               |
| `--val_strategy`       | `random`                                         | Validation strategy: random or top             |
| `--val_loss_type`      | `logprob`                                        | Validation loss type                           |
| `--use_second_order`   | (flag)                                           | Enable second-order selection                  |

## Key Differences from SFT

| Aspect                  | SFT                              | RLHF (PPO)                              |
| ----------------------- | -------------------------------- | --------------------------------------- |
| Training Loss           | Cross-entropy                    | PPO objective (policy + value + KL)     |
| Data                    | Static dataset                   | Dynamic rollouts (generate responses)   |
| Validation Loss         | NLL on target task               | Reward-weighted likelihood              |
| Selection Target        | Improve target task performance  | Generate high-reward responses          |

## Validation Loss Types

For data selection, we need gradients that represent the "good direction":

- **logprob**: Simple NLL on validation prompts (like SFT)
- **reward_weighted**: Generate responses, weight by rewards (RLHF-aligned)
- **advantage_weighted**: Weight by GAE advantages

## Output Directory

Results are saved to: `/scratch/pbb/Project/Gradient-Streaming/RLHF/`

Each experiment creates a folder with:
- `final/` - Saved model checkpoint
- Training logs (via console output)

## Architecture

```
RLHF/
├── __init__.py
├── README.md
├── data/
│   ├── __init__.py
│   ├── get_prompts.py       # Training prompt datasets
│   └── get_validation.py    # Validation data for selection
├── train/
│   ├── __init__.py
│   ├── train.py             # Main entry point
│   ├── trainer.py           # StreamingPPOTrainer
│   ├── training_arguments.py
│   ├── model_arguments.py
│   ├── rewards.py           # Reward model utilities
│   └── train.sh             # Launch script
└── eval/
    └── __init__.py
```
