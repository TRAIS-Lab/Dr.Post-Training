# SFT Experiments

This folder contains the training and evaluation code and experiment configurations for Supervised Fine-Tuning with **Gradient Streaming**.

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

### Available Datasets

#### Evaluation Datasets

| Dataset   | Task Type          | Description                                              |
| --------- | ------------------ | -------------------------------------------------------- |
| `mmlu`    | Multiple Choice    | Massive Multitask Language Understanding (57 subjects)   |
| `bbh`     | Reasoning          | BIG-Bench Hard (23 challenging reasoning tasks with CoT) |
| `tydiqa`  | Question Answering | Typologically Diverse QA (9 languages)                   |
| `gsm8k`   | Math               | Grade School Math (8K problems)                          |
| `math500` | Math               | MATH benchmark (500 competition problems)                |
| `samsum`  | Summarization      | SAMSum dialogue summarization                            |

#### Training Datasets

| Dataset      | Size | Description                           |
| ------------ | ---- | ------------------------------------- |
| `alpaca`     | 52K  | Stanford Alpaca instruction-following |
| `dolly`      | 15K  | Databricks Dolly 2.0                  |
| `flan_v2`    | 100K | FLAN v2 instruction tuning mixture    |
| `cot`        | 100K | Chain-of-Thought reasoning examples   |
| `oasst1`     | 88K  | OpenAssistant conversations           |
| `vicuna`     | 125K | ShareGPT-based conversations          |
| `wizardlm`   | 196K | WizardLM evolved instructions         |
| `openhermes` | 1M   | OpenHermes 2.5 diverse instructions   |
| `tulu3`      | 939K | Tulu-3 SFT mixture                    |
| `less`       | 1M   | LESS-selected instruction data        |

## Naming Convention

Experiments follow the pattern: `{train}_{task}-{selection}-{compression}-{model}-{training_type}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

| Component       | Options                     | Description                                   |
| --------------- | --------------------------- | --------------------------------------------- |
| `selection`     | `NA`, `Streaming`, `GREATS` | Data selection method                         |
| `compression`   | `NA`, `LoGra`               | Gradient compression (implies MeSO optimizer) |
| `training_type` | `full`, `lora`              | Full fine-tuning or LoRA                      |

> **Note:** GraSS compression is also available (`--compression GraSS`) but not used in default experiments.

### Selection Modes
- **NA**: No data selection (baseline)
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
| 1a  | NA        | NA          | full     | Baseline full fine-tuning                 |
| 1b  | NA        | NA          | lora     | Baseline LoRA fine-tuning                 |
| 2a  | Streaming | NA          | full     | Per-layer selection, full gradients       |
| 2b  | Streaming | NA          | lora     | Per-layer selection, full gradients, LoRA |
| 3a  | GREATS    | NA          | full     | Global selection, full gradients          |
| 3b  | GREATS    | NA          | lora     | Global selection, full gradients, LoRA    |
| 4   | Streaming | LoGra       | full     | Per-layer selection + MeSO                |
| 5   | GREATS    | LoGra       | full     | Global selection + MeSO                   |

> [!Note]
> Second-order interaction is enabled by default for all data selection methods (Streaming and GREATS).

## Learning Rate Configuration

Learning rates are managed via `SFT/train/lr_config.json`. Each experiment configuration (selection × compression × training type) can have its own optimal LR for each task/dataset combination.

### Recommended Workflow

1. **Run LR sweep** (uses 5% of data by default):
```bash
# Sweep all experiments for a specific task/dataset
bash SFT/train/lr_sweep.sh --experiments all --task samsum --train alpaca

# Sweep specific experiments
bash SFT/train/lr_sweep.sh --experiments baseline --task tydiqa --train less

# Custom LR grid
bash SFT/train/lr_sweep.sh --experiments all --task samsum --train alpaca \
    --lr_grid "1e-6,5e-6,1e-5,5e-5" \
    --lr_grid_lora "5e-5,1e-4,2e-4,5e-4"
```

2. **Review results**: Check `SFT/lr_sweep_results/` for detailed logs and `SFT/train/lr_config.json` for best LRs.

3. **Run full training**: LRs are automatically loaded from the config:
```bash
bash SFT/train/train.sh --experiments all --task samsum --train alpaca
```

### LR Resolution Order

1. If `--lr` is specified on command line, use that (override)
2. Look up from `lr_config.json` based on `{train}_{task}` + experiment name
3. Fall back to defaults (5e-05 for full, 2e-04 for LoRA)

### lr_config.json Format

```json
{
  "alpaca_samsum": {
    "NA-NA-full": {"lr": 5e-6, "val_loss": 1.23},
    "Streaming-NA-full": {"lr": 1e-5, "val_loss": 1.18},
    ...
  }
}
```

> **Important:** LR selection uses `val_loss` (validation set), not `eval_loss` (test set), to prevent test set leakage into hyperparameter tuning.

## Running Experiments

All experiments are launched using the unified `train.sh` script.

### Multi-Experiment Mode

Run multiple experiments with the `--experiments` flag:

```bash
# Run all 8 experiments
bash SFT/train/train.sh --experiments all --task mmlu --subject sociology

# Run by category
bash SFT/train/train.sh --experiments baseline --task mmlu      # NA-NA-full, NA-NA-lora
bash SFT/train/train.sh --experiments streaming --task mmlu     # All Streaming-* variants
bash SFT/train/train.sh --experiments greats --task mmlu        # All GREATS-* variants
bash SFT/train/train.sh --experiments compression --task mmlu   # *-LoGra-* variants
bash SFT/train/train.sh --experiments lora --task mmlu          # All *-lora variants
bash SFT/train/train.sh --experiments full --task mmlu          # All *-full variants

# Run specific experiments
bash SFT/train/train.sh --experiments "NA-NA-full,Streaming-LoGra-full" --task mmlu

# Combine categories
bash SFT/train/train.sh --experiments "baseline,streaming" --task mmlu

# Dry run - preview commands without executing
bash SFT/train/train.sh --experiments all --task mmlu --dry-run

# Submit to SLURM
bash SFT/train/train.sh --experiments all --task mmlu --sbatch
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
bash SFT/train/train.sh [options]

# Or with SLURM
sbatch SFT/train/train.sh [options]
```

#### Examples

```bash
# NA-NA-full: Baseline full fine-tuning
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b

# NA-NA-lora: Baseline LoRA fine-tuning
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b --lora

# Streaming-NA-full: Per-layer selection with full gradients (second-order enabled)
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection Streaming --use_second_order

# GREATS-NA-full: Global selection with full gradients (second-order enabled)
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection GREATS --use_second_order

# Streaming-LoGra-full: Per-layer selection + MeSO (second-order enabled by default)
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection Streaming --compression LoGra --use_second_order

# GREATS-LoGra-full: Global selection + MeSO (second-order enabled by default)
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection GREATS --compression LoGra --use_second_order

# Custom training dataset (with compression, second-order enabled)
bash SFT/train/train.sh --task samsum --model llama3-1b \
    --train openhermes --data_selection Streaming --compression LoGra --use_second_order
```

### Parameters

The unified training script accepts the following arguments:

#### Task Arguments

- `--task <task>` - Evaluation task: `mmlu`, `bbh`, `tydiqa`, `gsm8k`, `math500`, `samsum`
- `--subject <subject>` - Subject for MMLU/BBH (default: `world_religions`)
- `--train <dataset>` - Training dataset (optional, overrides task-based default):
  - `alpaca`, `dolly`, `flan_v2`, `cot`, `oasst1` - Instruction tuning
  - `gsm8k` - Math training data
  - `vicuna`, `wizardlm`, `openhermes`, `tulu3`, `less` - Large-scale instruction data

#### Data Selection Arguments

- `--data_selection <method>` - Data selection method:
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

- `--model <model>` - Model: `llama3-1b`, `llama2-7b`, `llama2-13b`, `mistral-7b` (default: `llama3-1b`)
- `--lr <lr>` - Learning rate (default: `5e-05`)
- `--batch_size <size>` - Batch size (default: `4`)
- `--seed <seed>` - Random seed (default: `42`)
- `--gradient_accumulation_steps <steps>` - Gradient accumulation (default: `1`)

#### Data Arguments

- `--percentage <pct>` - Data sampling, e.g., `0.05` for 5% (default: `0.05`)
- `--n_val <n>` - Validation examples for data selection (default: `5`)
- `--n_eval <n>` - Evaluation examples (default: `500`)
- `--val_batch_size <size>` - Validation batch size for data selection (default: same as `batch_size`)
- `--data_dir <dir>` - Data directory (default: `SFT/data`)

#### LoRA Arguments

- `--lora` - Enable LoRA fine-tuning (flag, omit for full fine-tuning)
- `--lora_alpha <alpha>` - LoRA alpha (default: `1`)
- `--lora_r <r>` - LoRA rank (default: `8`)
- `--lora_dropout <dropout>` - LoRA dropout (default: `0.1`)

## Evaluation

Trained models are automatically evaluated after training. Results are saved in the output directory.

### Manual Evaluation

For standalone evaluation of trained models:

```bash
# MMLU evaluation
python SFT/eval/eval.py --task mmlu --model_path <path_to_model> \
    --subject sociology --n_val 5 --n_eval 500

# TyDiQA evaluation
python SFT/eval/eval.py --task tydiqa --model_path <path_to_model> \
    --n_test 100
```

## Experiment Summary

The following experiments have been run and can be rerun with the commands below.

| Train Dataset | Eval Task | Percentage | Batch | N_val |
| ------------- | --------- | ---------- | ----- | ----- |
| Alpaca        | SAMSum    | 0.4        | 8     | 8     |
| LESS          | MMLU      | 0.05       | 8     | 8     |
| WizardLM      | BBH       | 0.1        | 8     | 8     |
| Tulu3         | TydiQA    | 0.01       | 8     | 8     |

> **Note:** Learning rates are managed via `SFT/train/lr_config.json`. Run `lr_sweep.sh` to find optimal LRs before training.

### LR Sweep Commands

Run LR sweep before full training to find optimal learning rates:

```bash
# Alpaca -> SAMSum
bash SFT/train/lr_sweep.sh --experiments all --task samsum --train alpaca \
    --batch_size 8 --n_val 8 --sweep_percentage 0.04

# LESS -> MMLU
bash SFT/train/lr_sweep.sh --experiments all --task mmlu --subject sociology --train less \
    --batch_size 8 --n_val 8 --sweep_percentage 0.005

# WizardLM -> BBH
bash SFT/train/lr_sweep.sh --experiments all --task bbh --subject navigate --train wizardlm \
    --batch_size 8 --n_val 8 --sweep_percentage 0.01

# Tulu3 -> TydiQA
bash SFT/train/lr_sweep.sh --experiments all --task tydiqa --train tulu3 \
    --batch_size 8 --n_val 8 --sweep_percentage 0.001
```

### Training Commands

Training commands for each experiment (LRs loaded from lr_config.json):

```bash
# Alpaca -> SAMSum
bash SFT/train/train.sh --experiments all --task samsum --train alpaca \
    --batch_size 8 --n_val 8 --percentage 0.4 --seed 42

# LESS -> MMLU
bash SFT/train/train.sh --experiments all --task mmlu --subject sociology --train less \
    --batch_size 8 --n_val 8 --percentage 0.05 --seed 42

# WizardLM -> BBH
bash SFT/train/train.sh --experiments all --task bbh --subject navigate --train wizardlm \
    --batch_size 8 --n_val 8 --percentage 0.1 --seed 42

# Tulu3 -> TydiQA
bash SFT/train/train.sh --experiments all --task tydiqa --train tulu3 \
    --batch_size 8 --n_val 8 --percentage 0.01 --seed 42
```

Evaluation commands for each experiment:

```bash
# Alpaca -> SAMSum
bash SFT/eval/eval.sh --train alpaca --task samsum --batch_size 64

# LESS -> MMLU
bash SFT/eval/eval.sh --train less --task mmlu --subject sociology --batch_size 64

# WizardLM -> BBH
bash SFT/eval/eval.sh --train wizardlm --task bbh --subject navigate --batch_size 64

# Tulu3 -> TyDiQA
bash SFT/eval/eval.sh --train tulu3 --task tydiqa --batch_size 64
```

## Output Directory

Results are saved to: `/scratch/pbb/Project/Gradient-Streaming/SFT/`

Each experiment creates a folder with the naming pattern above, containing:
- `train.log` - Training logs
- Model checkpoints (if save enabled)
- Evaluation results
