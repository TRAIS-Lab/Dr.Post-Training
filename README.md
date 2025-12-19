# Gradient Streaming

This repository implements **Gradient Streaming** for memory-efficient fine-tuning with compressed gradient scoring and data selection.

## Overview

The codebase supports three main features that can be combined:
1. **Data Selection** - Streaming (per-layer) or GREATS (global) selection
2. **Gradient Compression** - GraSS or LoGra methods (implies MeSO optimizer)
3. **Training Type** - Full fine-tuning or LoRA

### Naming Convention

Experiments follow the pattern: `{selection}-{compression}-{training_type}`

| Component    | Options                                      | Description                                    |
| ------------ | -------------------------------------------- | ---------------------------------------------- |
| `selection`  | `NA`, `Streaming`, `GREATS`                  | Data selection method                          |
| `compression`| `NA`, `GraSS`, `LoGra`                       | Gradient compression (implies MeSO optimizer)  |
| `training`   | `full`, `lora`                               | Full fine-tuning or LoRA                       |

**Examples:**
- `NA-NA-full` - Baseline full fine-tuning (no selection, no compression)
- `Streaming-NA-full` - Per-layer selection with full gradients
- `GREATS-GraSS-full` - Global selection with GraSS compression (MeSO optimizer)
- `Streaming-LoGra-lora` - Per-layer selection with LoGra compression and LoRA

### Key Design Principle

**Compression controls both scoring AND optimizer:**
- If compression is specified (`GraSS` or `LoGra`) → Use compressed gradients for scoring AND MeSO optimizer
- If no compression (`NA`) → Use full gradients for scoring AND standard AdamW optimizer

## Quick Start

```bash
pip install -r requirements.txt
```

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

### Recommended Environment Setup

It's **not** required to follow the exact same steps in this section. But this is a verified environment setup flow that may help users to avoid most of the issues during the installation.

```bash
conda create -n GradStream python=3.10
conda activate GradStream

conda install -c "nvidia/label/cuda-12.4.0" cudatoolkit
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip3 install packaging ninja

pip3 install sjlt --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

pip3 install -r requirements.txt
```

> Before installing `flash-attn`, you might need to install `psutil` first.

## Running Experiments

### Launch All Experiments

To run all experimental configurations for a given task:

```bash
# Dry run - see all commands without executing
bash SFT/train/launch_experiments.sh --task mmlu --subject sociology --dry-run

# Run all experiments locally
bash SFT/train/launch_experiments.sh --task mmlu --subject sociology

# Submit all experiments to SLURM
bash SFT/train/launch_experiments.sh --task mmlu --subject sociology --sbatch
```

This launches **16 configurations**:

| # | Configuration | Description |
|---|---------------|-------------|
| 1a | `NA-NA-full` | Baseline full fine-tuning |
| 1b | `NA-NA-lora` | Baseline LoRA fine-tuning |
| 2a | `Streaming-NA-full` | Per-layer selection, full gradients |
| 2b | `Streaming-NA-lora` | Per-layer selection, full gradients, LoRA |
| 3a | `GREATS-NA-full` | Global selection, full gradients |
| 3b | `GREATS-NA-lora` | Global selection, full gradients, LoRA |
| 4a | `NA-GraSS-full` | MeSO only with GraSS (no selection) |
| 4b | `NA-LoGra-full` | MeSO only with LoGra (no selection) |
| 5a | `Streaming-GraSS-full` | Per-layer selection + MeSO (GraSS) |
| 5b | `Streaming-LoGra-full` | Per-layer selection + MeSO (LoGra) |
| 6a | `GREATS-GraSS-full` | Global selection + MeSO (GraSS) |
| 6b | `GREATS-LoGra-full` | Global selection + MeSO (LoGra) |
| 7a | `Streaming-GraSS-2nd-full` | Per-layer + MeSO + second-order selection |
| 7b | `GREATS-GraSS-2nd-full` | Global + MeSO + second-order selection |

### Single Experiment

For running individual experiments with specific configurations:

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

# Streaming-NA-full: Per-layer selection with full gradients
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection Streaming --selection_mode per_layer

# GREATS-NA-full: Global selection with full gradients
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection Streaming --selection_mode global

# NA-GraSS-full: MeSO only (no selection, compressed optimizer)
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --compression GraSS

# Streaming-GraSS-full: Per-layer selection + MeSO
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection Streaming --selection_mode per_layer --compression GraSS

# GREATS-LoGra-full: Global selection + MeSO with LoGra
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection Streaming --selection_mode global --compression LoGra

# With second-order selection
bash SFT/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection Streaming --selection_mode global --compression GraSS --use_second_order

# Custom training dataset
bash SFT/train/train.sh --task samsum --model llama3-1b \
    --train openhermes --data_selection Streaming --compression GraSS
```

#### Parameters

The unified training script accepts the following arguments:

##### Task Arguments

- `--task <task>` - Evaluation task: `mmlu`, `bbh`, `tydiqa`, `gsm8k`, `math500`, `samsum`
- `--subject <subject>` - Subject for MMLU/BBH (default: `world_religions`)
- `--train <dataset>` - Training dataset (optional, overrides task-based default):
  - `alpaca`, `dolly`, `flan_v2`, `cot`, `oasst1` - Instruction tuning
  - `gsm8k` - Math training data
  - `vicuna`, `wizardlm`, `openhermes`, `tulu3`, `less` - Large-scale instruction data

##### Data Selection Arguments

- `--data_selection <method>` - Enable data selection: `NA` (none), `Streaming` (default: `NA`)
- `--selection_mode <mode>` - Selection granularity (only when `--data_selection Streaming`):
  - `per_layer` - Streaming: each layer independently selects (single-pass, default)
  - `global` - GREATS: accumulate scores across layers (two-pass)
- `--use_second_order` - Enable greedy selection with second-order interactions (flag)

##### Compression Arguments

- `--compression <method>` - Gradient compression method (implies MeSO optimizer):
  - `GraSS` - Gradient Sparsification with Sketching
  - `LoGra` - Low-rank Gradient compression
  - If not specified, uses full gradients and standard AdamW optimizer
- `--update_compressor_freq <steps>` - Projector refresh interval (default: `200`)

##### Core Training Arguments

- `--model <model>` - Model: `llama3-1b`, `llama2-7b`, `llama2-13b`, `mistral-7b` (default: `llama3-1b`)
- `--lr <lr>` - Learning rate (default: `5e-05`)
- `--batch_size <size>` - Batch size (default: `4`)
- `--seed <seed>` - Random seed (default: `42`)
- `--gradient_accumulation_steps <steps>` - Gradient accumulation (default: `1`)

##### Data Arguments

- `--percentage <pct>` - Data sampling, e.g., `0.05` for 5% (default: `0.05`)
- `--n_val <n>` - Validation examples for data selection (default: `5`)
- `--n_eval <n>` - Evaluation examples (default: `500`)
- `--val_batch_size <size>` - Validation batch size for data selection (default: same as `batch_size`)
- `--data_dir <dir>` - Data directory (default: `SFT/data`)

##### LoRA Arguments

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

## Experiment Results

See [SFT/README.md](SFT/README.md) for detailed experiment configurations and results.

### Archived Results

Previous experiment results (using old naming convention) are archived in `/scratch/pbb/Project/Gradient-Streaming/SFT_archive_v1/`. See the archive README for details on those experiments.