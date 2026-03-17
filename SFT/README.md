# SFT Experiments

This folder contains the training and evaluation code and method configurations for Supervised Fine-Tuning.

## Data Preparation

Download and prepare datasets using the unified data preparation script:

```bash
# See available datasets and options
python SFT/data/prepare_datasets.py -h

# Download specific datasets
python SFT/data/prepare_datasets.py --datasets mmlu bbh tydiqa

# Download all evaluation datasets
python SFT/data/prepare_datasets.py --datasets mmlu bbh tydiqa gsm8k math500 samsum

# Download training datasets
python SFT/data/prepare_datasets.py --datasets alpaca dolly flan_v2 cot oasst1
```

<details>
  <summary>Available Datasets</summary>

### Evaluation Datasets

| Dataset   | Task Type          | Description                                              |
| --------- | ------------------ | -------------------------------------------------------- |
| `samsum`  | Summarization      | SamSUM dialogue summarization                            |
| `tydiqa`  | Question Answering | Typologically Diverse QA (9 languages)                   |
| `mmlu`    | Multiple Choice    | Massive Multitask Language Understanding (57 subjects)   |
| `bbh`     | Reasoning          | BIG-Bench Hard (23 challenging reasoning tasks with CoT) |
| `gsm8k`   | Math               | Grade School Math (8K problems)                          |
| `math500` | Math               | MATH benchmark (500 competition problems)                |

### Training Datasets

| Dataset      | Size | Description                           |
| ------------ | ---- | ------------------------------------- |
| `less`       | 1M   | LESS-selected instruction data        |
| `alpaca`     | 52K  | Stanford Alpaca instruction-following |
| `tulu3`      | 939K | Tulu-3 SFT mixture                    |
| `dolly`      | 15K  | Databricks Dolly 2.0                  |
| `flan_v2`    | 100K | FLAN v2 instruction tuning mixture    |
| `cot`        | 100K | Chain-of-Thought reasoning examples   |
| `oasst1`     | 88K  | OpenAssistant conversations           |
| `vicuna`     | 125K | ShareGPT-based conversations          |
| `wizardlm`   | 196K | WizardLM evolved instructions         |
| `openhermes` | 1M   | OpenHermes 2.5 diverse instructions   |
</details>

## Experiment Summary

The following methods have been run and can be rerun with the commands below.

| Train Dataset | Eval Task | Percentage | Batch | Val Size | LoRA Rank |
| ------------- | --------- | ---------- | ----- | -------- | --------- |
| Alpaca        | SamSUM    | 0.4        | 8     | 32       | 32        |
| Tulu3         | TydiQA    | 0.01       | 8     | 32       | 32        |
| LESS          | MMLU      | 0.05       | 8     | 32       | 128       |
| LESS          | BBH       | 0.05       | 8     | 32       | 128       |

### Experiment Configurations

We consider the following 9 methods for each of the training datasets above. Each method has a YAML config in `SFT/train/configs/`:

| Config           | Curation  | Description                          |
| ---------------- | --------- | ------------------------------------ |
| `Standard-Full`  | NA        | Baseline full fine-tuning            |
| `Standard-LoRA`  | NA        | Baseline LoRA fine-tuning            |
| `Standard-MeSO`  | NA        | Baseline MeSO fine-tuning            |
| `Layerwise-Full` | Layerwise | Per-layer curation, full fine-tuning |
| `Layerwise-LoRA` | Layerwise | Per-layer curation, LoRA fine-tuning |
| `Layerwise-MeSO` | Layerwise | Per-layer curation + MeSO            |
| `Subset-Full`    | Subset    | Global curation, full fine-tuning    |
| `Subset-LoRA`    | Subset    | Global curation, LoRA fine-tuning    |
| `Subset-MeSO`    | Subset    | Global curation + MeSO               |

> Experiments follow the pattern: `{train}_{task}-{model}-{Method}-{FinetuningMethod}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

### LR Sweep Commands

The `SFT/train/lr/` folder contains tools for finding optimal learning rates, where learning rates are managed via `SFT/train/lr/config.json`. Run LR sweep before full training to find optimal learning rates:

```bash
# Alpaca -> SamSUM
bash SFT/train/lr/lr_sweep.sh --mode binary --methods all --train alpaca --task samsum --batch_size 8 --n_val 8 --sweep_percentage 0.04 --seed 2

# Tulu3 -> TydiQA
bash SFT/train/lr/lr_sweep.sh --mode binary --methods all --train tulu3 --task tydiqa --batch_size 8 --n_val 8 --sweep_percentage 0.001 --seed 2

# LESS -> MMLU/BBH
bash SFT/train/lr/lr_sweep.sh --mode binary --methods all --train less --task mmlu --subject sociology --batch_size 8 --n_val 8 --sweep_percentage 0.005 --seed 2 --lora_r 128
```

You can also run grid search via `--mode grid`.

### Training Commands

All methods are launched using `train.sh` with a config directory. Each config directory is self-contained: a `defaults.yaml` for shared experiment settings (model, dataset, batch size, LR scheduler, etc.) and one YAML per method (curation type, LR, compression).

```bash
# Alpaca -> SamSUM (all 9 methods)
bash SFT/train/train.sh -c configs/alpaca_samsum -m all

# Tulu3 -> TydiQA (all 9 methods)
bash SFT/train/train.sh -c configs/tulu3_tydiqa -m all
```

#### Seed Sweeps and CLI Overrides

The `--seed` and `--lr` flags override the corresponding config values, useful for sweeps:

```bash
# Run all methods with 3 different seeds
for s in 42 123 456; do
  bash SFT/train/train.sh -c configs/tulu3_tydiqa -m all --seed $s
done

# Quick LR test on a single method
bash SFT/train/train.sh -c configs/tulu3_tydiqa -m Layerwise-Full --lr 1e-04
```

#### Running by Category

```bash
# Run by category
bash SFT/train/train.sh -c configs/tulu3_tydiqa -m standard    # All Standard-* variants
bash SFT/train/train.sh -c configs/tulu3_tydiqa -m layerwise   # All Layerwise-* variants
bash SFT/train/train.sh -c configs/tulu3_tydiqa -m subset      # All Subset-* variants

# Run specific methods
bash SFT/train/train.sh -c configs/tulu3_tydiqa -m "Layerwise-Full,Subset-Full"

# Dry run (print commands without executing)
bash SFT/train/train.sh -c configs/tulu3_tydiqa -m all --dry-run

# List available methods in a config directory
bash SFT/train/train.sh -c configs/tulu3_tydiqa --list
```

| Category    | Matches                            |
| ----------- | ---------------------------------- |
| `all`       | All methods in the config directory|
| `standard`  | `Standard-*`                       |
| `layerwise` | `Layerwise-*`                      |
| `subset`    | `Subset-*`                         |
| `full`      | `*-Full`                           |
| `lora`      | `*-LoRA`                           |
| `meso`      | `*-MeSO`                           |

<details>
  <summary>Config Directory Structure</summary>

#### Layout

Each config directory contains a `defaults.yaml` and one YAML per method:

```
configs/tulu3_tydiqa/
  defaults.yaml          # shared: model, dataset, training hyperparams
  Standard-Full.yaml     # method + learning_rate
  Layerwise-Full.yaml    # method + learning_rate + compression
  ...
```

#### defaults.yaml (shared experiment settings)

```yaml
model: meta-llama/Llama-3.2-1B
train_dataset: tulu3
target_task: tydiqa
percentage: 0.01
seed: 42
batch_size: 8
gradient_accumulation_steps: 1
optim: adamw_torch
max_seq_length: 512
lr_scheduler_type: linear
warmup_ratio: 0.03
weight_decay: 0.0
num_train_epochs: 1
eval_steps: 50
use_flash_attention: true
n_eval: 500
selection_frac: 0.5
n_val: 8
val_batch_size: 1
val_strategy: merged_batch
```

#### Method config (method-specific settings)

Method configs only need to specify what differs from defaults. Example (`Layerwise-Full.yaml`):

```yaml
method: Layerwise
finetuning: Full
learning_rate: 4.96e-05

score_grad_compression:
  sparsifier: normal-64*64
  projector: none
```

Values in the method config override `defaults.yaml`. The load order is: `reset_config()` defaults → `defaults.yaml` → method config → CLI overrides (`--seed`, `--lr`).

#### Creating a New Experiment

To set up a new dataset combination:

1. Create a new folder under `configs/` (e.g., `configs/less_mmlu_sociology/`)
2. Copy a `defaults.yaml` from an existing experiment and update dataset, percentage, subject, etc.
3. Copy method configs and update learning rates (from LR sweep results)

</details>

### Evaluation Commands
Evaluation commands for each experiment:

```bash
# Alpaca -> SamSUM
bash SFT/eval/eval.sh --train alpaca --task samsum --batch_size 64

# Tulu3 -> TyDiQA
bash SFT/eval/eval.sh --train tulu3 --task tydiqa --batch_size 64 --n_test 500

# LESS -> MMLU/BBH
bash SFT/eval/eval.sh --train less --task mmlu --subject sociology --batch_size 64
```