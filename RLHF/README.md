# RLHF Experiments

This folder contains the training and evaluation code and method configurations for Reinforcement Learning from Human Feedback.

## Experiment Summary

| Task     | Model        | Batch | Val Size | Epochs | LoRA Rank |
| -------- | ------------ | ----- | -------- | ------ | --------- |
| Toxicity | gpt-neo-2.7B | 256   | 1024     | 1      | 16        |

### Method Configurations

All methods use **LoRA training** (no MeSO compression). Each method has a YAML config in `RLHF/train/configs/`:

| Config                | Selection | Description                                    |
| --------------------- | --------- | ---------------------------------------------- |
| `Standard-LoRA.yaml`    | NA        | Baseline PPO (no data selection)               |
| `IIF-LoRA.yaml`         | IIF       | Pre-filter rollout before PPO epochs           |
| `Layerwise-LoRA.yaml`   | Layerwise | Per-layer selection with projected scoring |
| `Subset-LoRA.yaml`      | Subset    | Global selection with exact scoring        |

**Selection Methods:**
- **NA**: No data selection (baseline)
- **IIF**: Influence Function-based Filtering — pre-filter entire rollout *before* PPO epochs
- **Layerwise**: Per-layer, per-mini-batch selection during PPO training
- **Subset**: Global selection across all layers, per-mini-batch during PPO training

### Training Commands

```bash
# Run all 4 methods
bash RLHF/train/train.sh --methods all --task toxicity

# Run by category
bash RLHF/train/train.sh --methods baseline --task toxicity
bash RLHF/train/train.sh --methods layerwise --task toxicity
bash RLHF/train/train.sh --methods subset --task toxicity

# Run specific methods
bash RLHF/train/train.sh --methods "Layerwise-LoRA,Subset-LoRA" --task toxicity

# List available methods
bash RLHF/train/train.sh --list

# Dry run
bash RLHF/train/train.sh --methods all --dry-run
```

<details>
  <summary>Detailed Training Script Configuration</summary>

#### Parameters

Method-specific settings (selection method, score compression) are in YAML config files.
Common settings are CLI arguments:

1. Task Arguments
   - `--task <task>` — Task: `toxicity` (default: `toxicity`)
   - `--model <model>` — Policy model (default: `EleutherAI/gpt-neo-2.7B`)
   - `--reward_model <model>` — Reward model (default: auto-selected per task)
2. Training Arguments
   - `--lr <lr>` — Learning rate override (default: from `config.json`)
   - `--lr_vhead <lr>` — Value head LR override (default: from `config.json`)
   - `--lr_config <path>` — Config file (default: `RLHF/train/config.json`)
   - `--batch_size <size>` — Batch size (default: `256`)
   - `--max_steps <steps>` — Max training steps (`-1` = use epochs, default)
   - `--epochs <n>` — Training epochs (default: `1`)
   - `--seed <seed>` — Random seed (default: `42`)
   - `--filter_frac <frac>` — Fraction of negative samples to drop (default: `1.0`)
3. PPO Arguments
   - `--ppo_epochs <n>` — PPO epochs per batch (default: `4`)
   - `--mini_batch_size <n>` — Mini-batch size (default: `4`)
   - `--init_kl_coef <coef>` — Initial KL coefficient override (default: from `config.json`)
   - `--kl_estimator <mode>` — KL estimator: `k1`, `k2`, `k3` (default: `k1`)
   - `--target <val>` — Target KL for adaptive controller (default: `70.0`)
   - `--target_kl <kl>` — Early stopping threshold (default: `0.3`)
   - `--max_new_tokens <n>` — Max new tokens (default: `30`)
4. Validation (for data selection)
   - `--n_val <n>` — Validation samples (default: `1024`, `0` = self-ref)
   - `--val_batch_size <n>` — Val batch size (default: `256`)
   - `--val_loss_type <type>` — `seqloss-reward` (default), `seqloss-lastadv`, or `tokenpg`
5. LoRA
   - `--lora_r <r>` — LoRA rank (default: `16`)
   - `--lora_alpha <alpha>` — LoRA alpha (default: `32`)

#### LR/KL Resolution

1. If `--lr` / `--lr_vhead` / `--init_kl_coef` is specified, use that value
2. Otherwise, look up from `config.json` based on task + method name
3. If not found, use fallback defaults

</details>

### Evaluation Commands

Evaluate trained models for toxicity:

```bash
# Evaluate all models
bash RLHF/eval/eval.sh --task toxicity --batch_size 256 --seed 82

# Evaluate specific model
python -m RLHF.eval.eval --model_path /path/to/model --n_samples 400
```

<details>
  <summary>Detailed Evaluation Configuration</summary>

#### Two-Classifier Approach

To ensure genuine toxicity reduction (not reward hacking), we use **different classifiers** for training and evaluation:

| Purpose    | Classifier                                         | Library    |
| ---------- | -------------------------------------------------- | ---------- |
| Training   | `facebook/roberta-hate-speech-dynabench-r4-target` | Direct     |
| Evaluation | `DaNLP/da-electra-hatespeech-detection`            | `evaluate` |

#### Dataset

- **Source**: `allenai/real-toxicity-prompts` (train split)
- **Filtering**: Prompts with toxicity score > 0.5
- **Default samples**: 100 (training), 400 (post-training eval)

#### Metrics

| Metric        | Description                                 |
| ------------- | ------------------------------------------- |
| Mean Toxicity | Average toxicity score across generations   |
| Std Toxicity  | Standard deviation of toxicity scores       |
| Max Toxicity  | Maximum toxicity score in batch             |
| Toxicity Rate | Fraction of generations with toxicity > 0.5 |

#### Usage

```bash
# Evaluate all models in directory
bash RLHF/eval/eval.sh

# Filter by task
bash RLHF/eval/eval.sh --task toxicity

# Custom settings
bash RLHF/eval/eval.sh --n_samples 1000 --batch_size 32 --max_new_tokens 50
```

</details>
