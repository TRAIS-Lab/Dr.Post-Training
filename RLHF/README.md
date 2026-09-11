# RLHF Experiments

This folder contains the training and evaluation code and method configurations for Reinforcement Learning from Human Feedback.

## Experiment Summary

| Task     | Model        | Batch | Val Size | Epochs | LoRA Rank |
| -------- | ------------ | ----- | -------- | ------ | --------- |
| Toxicity | gpt-neo-2.7B | 256   | 0 (self-ref) or 1024 (held-out) | 1 | 16 |

### Method Configurations

All methods use **LoRA training** (no MeSO compression). Each method has a YAML config in `RLHF/train/configs/`:

| Config                | Curation  | Description                                    |
| --------------------- | --------- | ---------------------------------------------- |
| `FullTraining-LoRA.yaml`    | NA        | Baseline PPO (no data curation)               |
| `TargetOnly-LoRA.yaml`      | NA        | Baseline PPO trained only on the `n_val` held-out target prompts (28 epochs, capped at 110 steps to match FullTraining's update budget); compares against held-out curation runs |
| `IIF-LoRA.yaml`         | IIF       | Pre-filter rollout before PPO epochs           |
| `LayerWiseSubset-LoRA.yaml`   | LayerWiseSubset | Per-layer curation with projected scoring |
| `GlobalSubset-LoRA.yaml`      | GlobalSubset    | Global curation with exact scoring (`subset_mode: one_pass`, single backward like LayerWiseSubset; the un-hooked value head then trains on the full mini-batch) |
| `BlockWiseSubset-LoRA.yaml`   | GroupWiseSubset (block)    | Per-transformer-block curation: all hooked LoRA layers of a block select jointly (single backward) |
| `SublayerWiseSubset-LoRA.yaml`| GroupWiseSubset (sublayer) | Per attention / MLP sub-block curation (single backward) |

**Curation Methods:**
- **NA**: No data curation (baseline)
- **IIF**: Influence Function-based Filtering — pre-filter entire rollout *before* PPO epochs
- **LayerWiseSubset**: Per-layer, per-mini-batch curation during PPO training
- **GlobalSubset**: Global curation across all layers, per-mini-batch during PPO training
- **GroupWiseSubset**: Per-layer-group curation (`selection_granularity: layer | sublayer | block | global`, or
  custom per-block `selection_groups` rules — see `SFT/README.md`, "Group-wise curation");
  `BlockWiseSubset` / `SublayerWiseSubset` are the block / sublayer aliases

### Training Commands

All methods are launched using `train.sh` with a config directory. Each config directory is self-contained: a `defaults.yaml` for shared experiment settings (model, reward model, PPO params, LR, etc.) and one YAML per method.

```bash
# Run all 4 methods
bash RLHF/train/train.sh -c configs/toxicity -m all

# Run by category
bash RLHF/train/train.sh -c configs/toxicity -m baseline
bash RLHF/train/train.sh -c configs/toxicity -m layer-wise-subset
bash RLHF/train/train.sh -c configs/toxicity -m global-subset

# Run specific methods
bash RLHF/train/train.sh -c configs/toxicity -m "LayerWiseSubset-LoRA,GlobalSubset-LoRA"

# Dry run / list
bash RLHF/train/train.sh -c configs/toxicity -m all --dry-run
bash RLHF/train/train.sh -c configs/toxicity --list
```

#### Seed Sweeps and CLI Overrides

The `--seed`, `--lr`, `--lr_vhead`, `--init_kl_coef`, `--n_val`, `--val_loss_type`,
`--val_batch_size` and `--max_steps` flags override config values:

```bash
# Run all methods with 3 different seeds
for s in 42 123 456; do
  bash RLHF/train/train.sh -c configs/toxicity -m all --seed $s
done

# Quick LR test
bash RLHF/train/train.sh -c configs/toxicity -m LayerWiseSubset-LoRA --lr 5e-6

# Held-out target with the PPO training loss as validation objective
bash RLHF/train/train.sh -c configs/toxicity -m GlobalSubset-LoRA --n_val 1024 --val_loss_type train-loss
```

#### Validation Target (`val_loss_type`) and Run Naming

The curation methods score each training sample by the alignment of its gradient with a
*target* gradient captured once per rollout batch (before the PPO epochs, at the same
parameters that produced the rollouts). The target is defined by two knobs:

| Knob | Values | Meaning |
| --- | --- | --- |
| `n_val` | `0` | Self-referencing: the target is computed on the training rollouts themselves |
| | `>0` | Held-out: `n_val` prompts from the RTP test split; each step regenerates one `val_batch_size` batch from the current policy |
| `val_loss_type` | `reward` (dir: `rew`) | $-\mathbb{E}_i[\text{normalize}(R_i)\,\overline{\log\pi_\theta}(y_i\mid x_i)]$, sequence-level reward weighting |
| | `token-pg` (dir: `tpg`) | $-\mathbb{E}_i[\overline{A_t \log\pi_\theta(y_t\mid\cdot)}]$, token-level policy gradient with GAE advantages |
| | `train-loss` (dir: `tloss`) | the PPO training objective: clipped surrogate + `vf_coef` × clipped value loss |

Here $\overline{\cdot}$ is the per-response token mean (`loss_reduction=sample_mean`, the default
since 2026-09-11): every response is one item, the target is the mean over validation
responses, and the PPO training loss is likewise the mean over responses of per-response
token-mean policy/value losses, so per-sample gradients are gradients of per-response losses
and the curated update is the plain mean over the kept responses (see `drpt.losses`).
`loss_reduction=token_mean` restores the legacy behaviour (token mean over the micro-batch
for training, per-response token *sums* for `reward`/`token-pg`), where responses are
weighted by length in scores and updates.

Because the target is captured at the rollout parameters, the PPO ratio is 1 and the clipping
is inactive, so `train-loss` equals `token-pg` plus the value-loss term.
Clipping only affects the *training-side* per-sample gradients inside the PPO epochs.

Run directories are named
`{task}-{model}-{Method}-{finetuning}-lr{lr}-b{batch}-v{n_val}-{rew|tpg|tloss}[-b{val_batch_size}]-pe{ppo_epochs}-mb{mini_batch}-kl{init_kl_coef}-s{seed}`
under `$SCRATCH_DIR/Dr.Post-Training/RLHF/`. `FullTraining` never uses a target and is always
filed under `v0-rew`; `result.ipynb` reuses that single run for every scenario. The same names
(`SCENARIOS`, `VAL_TYPE_SHORT`) are used in `result.ipynb` and `train/submit_all.sh`.


#### Full Sweep on Slurm

`RLHF/train/submit_all.sh` writes one manifest line per (method, scenario, seed) and submits a
single job array (never a loop of `sbatch` calls):

```bash
bash RLHF/train/submit_all.sh --dry-run                 # manifest + sbatch line only
bash RLHF/train/submit_all.sh --smoke                   # one 2-step job per val_loss_type
bash RLHF/train/submit_all.sh                           # 5 seeds x 6 scenarios x 3 methods + 5 FullTraining
bash RLHF/train/submit_all.sh --seeds "42" --scenarios "self-ref/train-loss,held-out/train-loss"
```

| Category    | Matches            |
| ----------- | ------------------ |
| `all`       | All methods        |
| `baseline`  | `FullTraining-*`       |
| `iif`       | `IIF-*`            |
| `layer-wise-subset` | `LayerWiseSubset-*`      |
| `global-subset`    | `GlobalSubset-*`         |
| `block-wise-subset` | `BlockWiseSubset-*`     |
| `sublayer-wise-subset` | `SublayerWiseSubset-*` |
| `group-wise-subset` | `GroupWiseSubset-*`, `BlockWiseSubset-*`, `SublayerWiseSubset-*` |
| `lora`      | `*-LoRA`           |

<details>
  <summary>Config Directory Structure</summary>

#### Layout

Each config directory contains a `defaults.yaml` and one YAML per method:

```
configs/toxicity/
  defaults.yaml          # shared: model, reward_model, PPO params, LR, LoRA
  FullTraining-LoRA.yaml     # method only (everything else from defaults)
  IIF-LoRA.yaml          # method + compression
  LayerWiseSubset-LoRA.yaml    # method + compression
  GlobalSubset-LoRA.yaml       # method + compression
  BlockWiseSubset-LoRA.yaml    # group-wise curation (per block)
  SublayerWiseSubset-LoRA.yaml # group-wise curation (per attention / MLP sub-block)
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

Method configs only specify what differs from defaults. Example (`LayerWiseSubset-LoRA.yaml`):

```yaml
method: LayerWiseSubset
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

- **Training prompts**: `allenai/real-toxicity-prompts`, prompt toxicity > 0.3, first 80% of the
  filtered rows (unshuffled split), 5–15-token prefixes of prompt+continuation.
- **Held-out 20%** of the filtered rows is split in two halves: the first half supplies the fixed
  validation prompts for curation (`n_val`), the second half supplies the in-training evaluation
  prompts (filtered again to toxicity > 0.5, first `n_eval`, default 500). Training, validation and
  evaluation prompts are therefore disjoint (previously the in-training eval prompts were the
  first rows of the *whole* dataset, i.e. inside the training set).
- **Post-training eval** (`RLHF/eval/eval.py`): `OxAISH-AL-LLM/wiki_toxic` test split, toxic label.
- In-training eval samples inside a forked RNG state (fixed seed), so evaluation does not perturb the
  training trajectory and eval noise is identical across steps and methods.

#### Metrics

| Metric        | Description                                 |
| ------------- | ------------------------------------------- |
| Mean Toxicity | Average toxicity score across generations   |
| Std Toxicity  | Standard deviation of toxicity scores       |
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
