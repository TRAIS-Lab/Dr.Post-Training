# Supervised Fine-Tuning (SFT) Experiments

This directory contains code for supervised fine-tuning experiments with layer-wise data selection and memory-efficient optimization.

## Directory Structure

```
experiment/
├── train/              # Training scripts and arguments
│   ├── train.py        # Main training script
│   ├── train.sh        # Shell script for single experiments
│   ├── launch_experiments.sh  # Launch all configurations
│   ├── data_arguments.py
│   ├── model_arguments.py
│   └── training_arguments.py
├── data/               # Dataset utilities
│   ├── get_train_dataset.py
│   ├── get_val_dataset.py
│   ├── get_test_dataset.py
│   └── prepare_datasets.py
├── eval/               # Evaluation utilities
│   ├── eval.py
│   ├── mmlu.py
│   └── tydiqa.py
└── benchmark/          # Benchmark experiments
```

## Quick Start

### Launch All Experiments

```bash
# Dry run - see all commands without executing
bash experiment/train/launch_experiments.sh --task mmlu --subject sociology --dry-run

# Run all experiments locally
bash experiment/train/launch_experiments.sh --task mmlu --subject sociology

# Submit all experiments to SLURM
bash experiment/train/launch_experiments.sh --task mmlu --subject sociology --sbatch
```

This launches **16 configurations**:
- 2 Baselines (Full + LoRA fine-tuning)
- 4 GREATS Data Selection (Full/LoRA × GraSS/LoGra)
- 2 MeSO Compressed Optimizer (Full × GraSS/LoGra)
- 4 MeSO + GREATS (global/per-layer × GraSS/LoGra)
- 4 MeSO + GREATS + Second-Order (global/per-layer × GraSS/LoGra)

### Single Experiment

```bash
# Baseline: Full fine-tuning
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b

# Baseline: LoRA fine-tuning
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b --lora

# Data Selection: GREATS with GraSS compression
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection GREATS --compression GraSS

# MeSO + Data Selection: Both optimizations combined
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --MeSO --data_selection GREATS --compression GraSS
```

## Parameters

### Task Arguments

| Argument | Description | Options |
|----------|-------------|---------|
| `--task` | Evaluation task | `mmlu`, `bbh`, `tydiqa`, `gsm8k`, `math500`, `samsum` |
| `--subject` | Subject for MMLU/BBH | e.g., `world_religions`, `sociology` |
| `--train` | Training dataset | `alpaca`, `dolly`, `flan_v2`, `cot`, `oasst1`, `gsm8k`, `vicuna`, `wizardlm`, `openhermes`, `tulu3` |

### Core Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_selection` | Data selection method | `NA`, `GREATS`, `GradNorm` |
| `--MeSO` | Enable MeSO compressed optimizer | flag |
| `--model` | Model identifier | `llama3-1b`, `llama2-7b`, `llama2-13b`, `mistral-7b` |
| `--lr` | Learning rate | `5e-05` |
| `--batch_size` | Batch size | `4` |
| `--seed` | Random seed | `42` |

### Data Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--percentage` | Data sampling percentage | `0.05` |
| `--n_val` | Validation examples for selection | `5` |
| `--n_eval` | Evaluation examples | `500` |
| `--val_batch_size` | Validation batch size | same as `batch_size` |

### Compression Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--compression` | Gradient compression method | `GraSS`, `LoGra` |
| `--update_compressor_freq` | Projector refresh interval | `200` |
| `--selection_mode` | Selection mode for MeSO+GREATS | `global`, `per_layer` |
| `--use_second_order` | Use second-order selection | flag |

### LoRA Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--lora` | Enable LoRA fine-tuning | flag |
| `--lora_alpha` | LoRA alpha | `1` |
| `--lora_r` | LoRA rank | `8` |
| `--lora_dropout` | LoRA dropout | `0.1` |

## Evaluation

Trained models are automatically evaluated after training. For manual evaluation:

```bash
# MMLU evaluation
python experiment/eval/eval.py --task mmlu --model_path <path_to_model> \
    --subject sociology --n_val 5 --n_eval 500

# TyDiQA evaluation
python experiment/eval/eval.py --task tydiqa --model_path <path_to_model> \
    --n_test 100
```

## Example Experiments

```bash
# SAMSum with OpenHermes training data
CUDA_VISIBLE_DEVICES=0 ./launch_experiments.sh \
    --task samsum --train openhermes \
    --lr 5e-06 --lr_lora 1e-04 \
    --batch_size 8 --val_batch_size 1 --n_val 16 \
    --percentage 0.05 --seed 42

# TyDiQA with Vicuna training data
CUDA_VISIBLE_DEVICES=0 ./launch_experiments.sh \
    --task tydiqa --train vicuna \
    --lr 5e-06 --lr_lora 1e-04 \
    --batch_size 8 --val_batch_size 1 --n_val 16 \
    --percentage 0.5 --seed 42
```
