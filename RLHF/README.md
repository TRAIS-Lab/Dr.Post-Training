# RLHF Experiments

This folder contains the training and evaluation code and method configurations for Reinforcement Learning from Human Feedback.

## Experiment Summary

The following methods have been run and can be rerun with the commands below.

| Task     | Model        | Batch | Val Size | Epochs | LoRA Rank |
| -------- | ------------ | ----- | -------- | ------ | --------- |
| Toxicity | gpt-neo-2.7B | 256   | 1024     | 1      | 16        |

### Experiment Configurations

We consider the following 8 methods for the toxicity task:

| #   | Selection | Compression | Training | Description                               |
| --- | --------- | ----------- | -------- | ----------------------------------------- |
| 1a  | NA        | NA          | Full     | Baseline PPO full fine-tuning             |
| 1b  | NA        | NA          | LoRA     | Baseline PPO LoRA fine-tuning             |
| 2a  | Streaming | NA          | Full     | Per-layer selection, full gradients       |
| 2b  | Streaming | NA          | LoRA     | Per-layer selection, full gradients, LoRA |
| 3a  | GREATS    | NA          | Full     | Global selection, full gradients          |
| 3b  | GREATS    | NA          | LoRA     | Global selection, full gradients, LoRA    |
| 4   | Streaming | LoGra       | Full     | Per-layer selection + MeSO                |
| 5   | GREATS    | LoGra       | Full     | Global selection + MeSO                   |

> Experiments follow the pattern: `{task}-{model}-{method_str}-{training_type}-lr{lr}-b{batch}-v{nval}b{val_batch}-pe{ppo_epochs}-mb{mini_batch}-kl{kl_coef}-s{seed}`

### LR Sweep Commands

The `RLHF/train/lr/` folder contains tools for finding optimal learning rates, where learning rates are managed via `RLHF/train/lr/config.json`. Run LR sweep before full training to find optimal learning rates:

```bash
# Toxicity task
bash RLHF/train/lr/lr_sweep.sh --mode binary --methods all --task toxicity \
    --batch_size 256 --n_val 4 --sweep_max_steps 50 --seed 2
```

You can also run grid search via `--mode grid`.

### Training Commands

All methods are launched using the unified `train.sh` script. Training commands (LRs loaded from lr_config.json):

```bash
# Toxicity task - all methods
bash RLHF/train/train.sh --methods all --task toxicity

# Toxicity task - baseline only
bash RLHF/train/train.sh --methods baseline --task toxicity

# Toxicity task - streaming methods
bash RLHF/train/train.sh --methods streaming --task toxicity
```

<details>
  <summary>Detailed Training Script Configuration</summary>

#### Run Multiple Experiments

Run multiple methods with the `--methods` flag:

```bash
# Run all 8 methods
bash RLHF/train/train.sh --methods all --task toxicity

# Run by category
bash RLHF/train/train.sh --methods baseline --task toxicity      # NA-NA-Full, NA-NA-LoRA
bash RLHF/train/train.sh --methods streaming --task toxicity     # All Streaming-* variants
bash RLHF/train/train.sh --methods greats --task toxicity        # All GREATS-* variants
bash RLHF/train/train.sh --methods compression --task toxicity   # *-LoGra-* variants
bash RLHF/train/train.sh --methods lora --task toxicity          # All *-LoRA variants
bash RLHF/train/train.sh --methods full --task toxicity          # All *-Full variants

# Run specific methods
bash RLHF/train/train.sh --methods "NA-NA-Full,Streaming-LoGra-Full" --task toxicity

# Combine categories
bash RLHF/train/train.sh --methods "baseline,streaming" --task toxicity

# Dry run - preview commands without executing
bash RLHF/train/train.sh --methods all --task toxicity --dry-run

# Submit to SLURM
bash RLHF/train/train.sh --methods all --task toxicity --sbatch
```

Available Categories:

| Category         | Experiments                                                |
| ---------------- | ---------------------------------------------------------- |
| `all`            | All 8 methods                                              |
| `baseline`       | NA-NA-Full, NA-NA-LoRA                                     |
| `streaming`      | Streaming-NA-Full, Streaming-NA-LoRA, Streaming-LoGra-Full |
| `greats`         | GREATS-NA-Full, GREATS-NA-LoRA, GREATS-LoGra-Full          |
| `full`           | All *-Full methods (5 total)                               |
| `lora`           | All *-LoRA methods (3 total)                               |
| `compression`    | Streaming-LoGra-Full, GREATS-LoGra-Full                    |
| `no-compression` | All methods without compression (6 total)                  |

#### Parameters

The unified training script accepts the following arguments:

1. Task Arguments
   - `--task <task>` - Task: `toxicity` (default: `toxicity`)
   - `--model <model>` - Policy model (default: `EleutherAI/gpt-neo-2.7B`)
   - `--reward_model <model>` - Reward model (default: `facebook/roberta-hate-speech-dynabench-r4-target`)
2. Data Selection Arguments
   - `--data_selection <method>` - Data selection method:
     - `NA` - No selection (baseline, default)
     - `Streaming` - Per-layer selection (single-pass)
     - `GREATS` - Global selection (two-pass)
   - `--use_second_order` - Enable greedy selection with second-order interactions (default: disabled)
3. Compression Arguments
   - `--compression <method>` - Gradient compression method (implies MeSO optimizer):
     - `LoGra` - Low-rank Gradient compression (Gaussian projection, default)
     - `GraSS` - Gradient Sparsification with Sketching (available but not used in default methods)
     - If not specified, uses full gradients and standard AdamW optimizer
   - `--update_compressor_freq <steps>` - Projector refresh interval (default: `200`)
4. Core Training Arguments
   - `--lr <lr>` - Learning rate override (if not specified, looked up from `lr_config.json`; fallback: `1e-5`)
   - `--lr_config <path>` - LR config file path (default: `RLHF/train/lr/config.json`)
   - `--batch_size <size>` - Batch size (default: `256`)
   - `--max_steps <steps>` - Maximum training steps (default: `-1`, meaning use epochs instead)
   - `--epochs <n>` - Number of training epochs (default: `1`, used when max_steps <= 0)
   - `--seed <seed>` - Random seed (default: `42`)
   - `--filter_frac <frac>` - Fraction of negative-influence samples to drop (default: `1.0`)
5. PPO Arguments
   - `--ppo_epochs <n>` - PPO epochs per batch (default: `4`)
   - `--mini_batch_size <n>` - Mini-batch size for PPO updates (default: `8`)
   - `--init_kl_coef <coef>` - Initial KL penalty coefficient (default: `0.1`)
   - `--kl_estimator <mode>` - KL estimator: `k1`, `k2`, `k3` (default: `k1`)
   - `--target <val>` - Target KL for adaptive KL controller (default: `1.0`)
   - `--target_kl <kl>` - Early stopping threshold (default: `0.1`)
   - `--max_new_tokens <n>` - Maximum new tokens to generate (default: `30`)
   - `--min_new_tokens <n>` - Minimum new tokens for evaluation only (default: `0`, not used in training)
6. LoRA Arguments
   - `--lora` - Enable LoRA fine-tuning (flag, omit for full fine-tuning)
   - `--lora_r <r>` - LoRA rank (default: `16`)
   - `--lora_alpha <alpha>` - LoRA alpha (default: `32`)
   - `--lora_target_modules <modules>` - Target modules for LoRA (default: auto-detect)
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

#### Classifier Selection

```bash
# Use independent classifier (default, recommended)
python -m RLHF.eval.eval --model_path /path/to/model --classifier independent

# Use reward model classifier (for comparison)
python -m RLHF.eval.eval --model_path /path/to/model --classifier reward
```

#### Usage

```bash
# Evaluate all models in directory (uses independent classifier by default)
bash RLHF/eval/eval.sh

# Filter by task
bash RLHF/eval/eval.sh --task toxicity

# Custom settings
bash RLHF/eval/eval.sh --n_samples 1000 --batch_size 32 --max_new_tokens 50

# Use reward model classifier for comparison
bash RLHF/eval/eval.sh --classifier reward

# Submit to SLURM
bash RLHF/eval/eval.sh --sbatch

# Dry run
bash RLHF/eval/eval.sh --dry-run
```

#### Evaluation Arguments

Training-time toxicity evaluation uses a **different classifier** than the reward model to prevent reward hacking and provide unbiased measurement.

- `--eval_interval <n>` - Evaluate every N steps; 0 = epoch end only (default: `1`)
- `--n_eval <n>` - Number of samples for full evaluation (default: `500`)
- `--eval_batch_size <n>` - Batch size for generation during evaluation (default: `256`)

Training metrics include:
- `eval/toxicity_prob` - Mean toxicity probability of step generations
- `eval/toxicity_rate` - Fraction of toxic generations per step
</details>

## Key Differences from SFT

| Aspect           | SFT                             | RLHF (PPO)                                |
| ---------------- | ------------------------------- | ----------------------------------------- |
| Training Loss    | Cross-entropy                   | PPO objective (policy + value + KL)       |
| Data             | Static dataset                  | Dynamic rollouts (generate responses)     |
| Validation Loss  | NLL on target task              | Sequence-level reward-weighted likelihood |
| Selection Target | Improve target task performance | Generate high-reward responses            |
