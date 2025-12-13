# Efficient Fine-Tuning

This repository implements memory-efficient fine-tuning with **compressed optimizer** support, enabling full model fine-tuning on limited GPU memory.

## Quick Start

```bash
pip install -r requirements.txt
```

## Data Preparation

Download and prepare datasets using the unified data preparation script:

```bash
# See available datasets and options
python experiment/data/prepare_datasets.py -h

# Download specific datasets
python experiment/data/prepare_datasets.py --datasets mmlu bbh tydiqa

# Download all evaluation datasets
python experiment/data/prepare_datasets.py --datasets mmlu bbh tydiqa gsm8k math500 samsum

# Download training datasets
python experiment/data/prepare_datasets.py --datasets alpaca dolly flan_v2 cot oasst1
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

### Recommended Environment Setup

It's **not** required to follow the exact same steps in this section. But this is a verified environment setup flow that may help users to avoid most of the issues during the installation.

```bash
conda create -n IF python=3.10
conda activate IF

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

To run all experimental configurations (baselines + data selection + MeSO) for a given task:

```bash
# Dry run - see all commands without executing
bash experiment/train/launch_experiments.sh --task mmlu --subject sociology --dry-run

# Run all experiments locally
bash experiment/train/launch_experiments.sh --task mmlu --subject sociology

# Submit all experiments to SLURM
bash experiment/train/launch_experiments.sh --task mmlu --subject sociology --sbatch
```

This launches **10 configurations**:
- 2 Baselines (Full + LoRA fine-tuning)
- 4 GREATS Data Selection (Full/LoRA × GraSS/LoGra)
- 2 MeSO Compressed Optimizer (Full × GraSS/LoGra)
- 2 MeSO + GREATS (Full × GraSS/LoGra)

### Single Experiment

For running individual experiments with specific configurations:

```bash
# From project root
bash experiment/train/train.sh [options]

# Or with SLURM
sbatch experiment/train/train.sh [options]
```

#### Examples

```bash
# Baseline: Full fine-tuning
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b

# Baseline: LoRA fine-tuning
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b --lora

# Data Selection: GREATS with GraSS compression
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --data_selection GREATS --compression GraSS

# Data Selection: GREATS + LoRA with LoGra compression
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b --lora \
    --data_selection GREATS --compression LoGra

# MeSO: Compressed optimizer without data selection
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --MeSO --compression GraSS

# MeSO + Data Selection: Both optimizations combined
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --MeSO --data_selection GREATS --compression GraSS

# Custom training dataset (instead of task-based default)
bash experiment/train/train.sh --task mmlu --subject sociology --model llama3-1b \
    --train wizardlm --data_selection GREATS --compression GraSS
```

#### Parameters

The unified training script accepts the following arguments:

##### Task Arguments

- `--task <task>` - Evaluation task: `mmlu`, `bbh`, `tydiqa`, `gsm8k`, `math500`, `samsum`
- `--subject <subject>` - Subject for MMLU/BBH (default: `world_religions`)
- `--train <dataset>` - Training dataset (optional, overrides task-based default):
  - `alpaca`, `dolly`, `flan_v2`, `cot`, `oasst1` - Instruction tuning
  - `gsm8k` - Math training data
  - `vicuna`, `wizardlm`, `openhermes`, `tulu3` - Large-scale instruction data

##### Core Training Arguments

- `--data_selection <method>` - Data selection: `NA`, `GREATS`, `GradNorm` (default: `NA`)
- `--MeSO` - Enable MeSO compressed optimizer for memory efficiency
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
- `--data_dir <dir>` - Data directory (default: `experiment/data`)

##### Compression Arguments

- `--compression <method>` - Gradient compression method (auto-enabled when needed)
  - `GraSS` - Gradient Sparsification with Sketching [default]
  - `LoGra` - Low-rank Gradient compression
  - Auto-enabled when `--data_selection` is not `NA` or `--MeSO` is set
- `--update_compressor_freq <steps>` - Projector refresh interval (default: `200`)

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
python experiment/eval/eval.py --task mmlu --model_path <path_to_model> \
    --subject sociology --n_val 5 --n_eval 500

# TyDiQA evaluation
python experiment/eval/eval.py --task tydiqa --model_path <path_to_model> \
    --n_test 100
```

## Current Experiments

```bash
CUDA_VISIBLE_DEVICES=0 ./launch_experiments.sh --task samsum --train openhermes --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --val_batch_size 1 --n_val 16 --percentage 0.05 --seed 42

CUDA_VISIBLE_DEVICES=0 ./launch_experiments.sh --task tydiqa --train vicuna --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --val_batch_size 1 --n_val 16 --percentage 0.5 --seed 42

CUDA_VISIBLE_DEVICES=0 ./launch_experiments.sh --task math500 --train less --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --val_batch_size 1 --n_val 16 --percentage 0.05 --seed 42

CUDA_VISIBLE_DEVICES=0 ./launch_experiments.sh --task gsm8k --train wizardlm --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --val_batch_size 1 --n_val 16 --percentage 0.3 --seed 42
```