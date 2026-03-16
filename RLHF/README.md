# RLHF Experiments

This folder contains the training and evaluation code and method configurations for Reinforcement Learning from Human Feedback.

## Experiment Summary

| Task     | Model        | Batch | Val Size | Epochs | LoRA Rank |
| -------- | ------------ | ----- | -------- | ------ | --------- |
| Toxicity | gpt-neo-2.7B | 256   | 1024     | 1      | 16        |

### Method Configurations

All methods use **LoRA training** (no MeSO compression). Each method has a YAML config in `RLHF/train/configs/`:

| Config                | Curation  | Description                                    |
| --------------------- | --------- | ---------------------------------------------- |
| `Standard-LoRA.yaml`    | NA        | Baseline PPO (no data curation)               |
| `IIF-LoRA.yaml`         | IIF       | Pre-filter rollout before PPO epochs           |
| `Layerwise-LoRA.yaml`   | Layerwise | Per-layer curation with projected scoring |
| `Subset-LoRA.yaml`      | Subset    | Global curation with exact scoring        |

**Curation Methods:**
- **NA**: No data curation (baseline)
- **IIF**: Influence Function-based Filtering — pre-filter entire rollout *before* PPO epochs
- **Layerwise**: Per-layer, per-mini-batch curation during PPO training
- **Subset**: Global curation across all layers, per-mini-batch during PPO training

### Training Commands

All methods are launched using `train.sh` with a config directory. Each config directory is self-contained: a `defaults.yaml` for shared experiment settings (model, reward model, PPO params, LR, etc.) and one YAML per method.

```bash
# Run all 4 methods
bash RLHF/train/train.sh -c configs/toxicity -m all

# Run by category
bash RLHF/train/train.sh -c configs/toxicity -m baseline
bash RLHF/train/train.sh -c configs/toxicity -m layerwise
bash RLHF/train/train.sh -c configs/toxicity -m subset

# Run specific methods
bash RLHF/train/train.sh -c configs/toxicity -m "Layerwise-LoRA,Subset-LoRA"

# Dry run / list
bash RLHF/train/train.sh -c configs/toxicity -m all --dry-run
bash RLHF/train/train.sh -c configs/toxicity --list
```

#### Seed Sweeps and CLI Overrides

The `--seed`, `--lr`, `--lr_vhead`, and `--init_kl_coef` flags override config values:

```bash
# Run all methods with 3 different seeds
for s in 42 123 456; do
  bash RLHF/train/train.sh -c configs/toxicity -m all --seed $s
done

# Quick LR test
bash RLHF/train/train.sh -c configs/toxicity -m Layerwise-LoRA --lr 5e-6
```

| Category    | Matches            |
| ----------- | ------------------ |
| `all`       | All methods        |
| `baseline`  | `Standard-*`       |
| `iif`       | `IIF-*`            |
| `layerwise` | `Layerwise-*`      |
| `subset`    | `Subset-*`         |
| `lora`      | `*-LoRA`           |

<details>
  <summary>Config Directory Structure</summary>

#### Layout

Each config directory contains a `defaults.yaml` and one YAML per method:

```
configs/toxicity/
  defaults.yaml          # shared: model, reward_model, PPO params, LR, LoRA
  Standard-LoRA.yaml     # method only (everything else from defaults)
  IIF-LoRA.yaml          # method + compression
  Layerwise-LoRA.yaml    # method + compression
  Subset-LoRA.yaml       # method + compression
```

#### defaults.yaml (shared experiment settings)

```yaml
model: EleutherAI/gpt-neo-2.7B
reward_model: facebook/roberta-hate-speech-dynabench-r4-target
task: toxicity
seed: 42
batch_size: 256
epochs: 1
learning_rate: 1e-5
lr_vhead: 5e-4
init_kl_coef: 0.02
ppo_epochs: 4
mini_batch_size: 4
lora_r: 16
lora_alpha: 32
# ... (see file for full list)
```

#### Method config (method-specific settings)

Method configs only specify what differs from defaults. Example (`Layerwise-LoRA.yaml`):

```yaml
method: Layerwise
finetuning: LoRA

score_grad_compression:
  sparsifier: none
  projector: none
```

Values load in order: `reset_config()` defaults → `defaults.yaml` → method config → CLI overrides (`--seed`, `--lr`, etc.).

#### Creating a New Experiment

To set up a new task:

1. Create a new folder under `configs/` (e.g., `configs/sentiment/`)
2. Copy a `defaults.yaml` and update model, reward_model, task, LRs, etc.
3. Copy method configs (usually unchanged for method-specific settings)

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
