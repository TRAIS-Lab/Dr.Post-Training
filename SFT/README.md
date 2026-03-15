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

| Config              | Curation  | Description                               |
| ------------------- | --------- | ----------------------------------------- |
| `Standard-Full`     | NA        | Baseline full fine-tuning                 |
| `Standard-LoRA`     | NA        | Baseline LoRA fine-tuning                 |
| `Standard-MeSO`     | NA        | Baseline MeSO fine-tuning                 |
| `Layerwise-Full`    | Layerwise | Per-layer curation, full fine-tuning     |
| `Layerwise-LoRA`    | Layerwise | Per-layer curation, LoRA fine-tuning     |
| `Layerwise-MeSO`    | Layerwise | Per-layer curation + MeSO                |
| `Subset-Full`       | Subset    | Global curation, full fine-tuning        |
| `Subset-LoRA`       | Subset    | Global curation, LoRA fine-tuning        |
| `Subset-MeSO`       | Subset    | Global curation + MeSO                   |

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

All methods are launched using the unified `train.sh` script. Training commands for each experiment (LRs loaded from lr_config.json):

```bash
# Alpaca -> SamSUM
bash SFT/train/train.sh --methods all --train alpaca --task samsum --batch_size 8 --n_val 32 --percentage 0.4 --seed 42

# Tulu3 -> TydiQA
bash SFT/train/train.sh --methods all --train tulu3 --task tydiqa --batch_size 8 --n_val 32 --percentage 0.01 --seed 42

# LESS -> MMLU/BBH
bash SFT/train/train.sh --methods all --train less --task mmlu --subject sociology --batch_size 8 --n_val 32 --percentage 0.05 --seed 42 --lora_r 128
```


<details>
  <summary>Detailed Training Script Configuration</summary>

#### Method Configs

Each method is defined by a YAML config in `SFT/train/configs/`. Method-specific settings (curation method, compression, LoRA, score compression) live in the config file — no need to pass them via CLI.

Example config (`Layerwise-Full.yaml`):
```yaml
method: Layerwise
finetuning: Full

score_grad_compression:
  sparsifier: normal-64*64
  projector: none

opt_grad_compression:
  sparsifier: none
  projector: none
```

To customize a method, edit its config file directly.

#### Running Experiments

```bash
# List available methods
bash SFT/train/train.sh --list

# Run all methods
bash SFT/train/train.sh --methods all --task mmlu --subject sociology

# Run by category
bash SFT/train/train.sh --methods baseline --task mmlu      # Standard-Full, Standard-LoRA
bash SFT/train/train.sh --methods layerwise --task mmlu     # All Layerwise-* variants
bash SFT/train/train.sh --methods subset --task mmlu        # All Subset-* variants

# Run specific methods
bash SFT/train/train.sh --methods "Standard-Full,Layerwise-MeSO" --task mmlu

# Dry run
bash SFT/train/train.sh --methods all --task mmlu --dry-run
```

Available Categories:

| Category         | Experiments                                                |
| ---------------- | ---------------------------------------------------------- |
| `all`            | All 9 methods                                          |
| `baseline`       | Standard-Full, Standard-LoRA                               |
| `layerwise`      | Layerwise-Full, Layerwise-LoRA, Layerwise-MeSO             |
| `subset`         | Subset-Full, Subset-LoRA, Subset-MeSO                      |
| `full`           | All *-Full methods                                         |
| `lora`           | All *-LoRA methods                                         |
| `compression`    | All *-MeSO* methods                                        |
| `no-compression` | All methods without compression                            |

#### CLI Parameters

These are experiment-level settings shared across methods (passed via CLI):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--task` | `mmlu` | Eval task: mmlu, bbh, tydiqa, gsm8k, math500, samsum |
| `--subject` | `sociology` | Subject for MMLU/BBH |
| `--train` | task default | Training dataset |
| `--model` | `meta-llama/Llama-3.2-1B` | Model path |
| `--lr` | from config.json | Learning rate override |
| `--batch_size` | `8` | Training batch size |
| `--val_batch_size` | `1` | Val batch size for curation |
| `--percentage` | `0.05` | Data sampling fraction |
| `--n_val` | `8` | Validation examples |
| `--n_eval` | `500` | Evaluation examples |
| `--seed` | `42` | Random seed |
| `--selection_frac` | `0.5` | Curation fraction |
| `--use_second_order` | disabled | Enable greedy curation |

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