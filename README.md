# Efficient Fine-Tuning with Layer-wise Data Selection

This repository implements **memory-efficient fine-tuning** with layer-wise data selection and compressed optimizer support, enabling full model fine-tuning on limited GPU memory.

## Key Features

- **Layer-wise Data Selection**: Select beneficial training samples independently at each layer during backward pass
- **Streaming Paradigm**: Joint data selection and model update using a single gradient computation stream
- **Memory Efficient**: MeSO optimizer maintains optimizer states in compressed space
- **Flexible Compression**: Support for GraSS (gradient sparsification + sketching) and LoGra (low-rank gradients)

## Experiments

| Experiment | Description | README |
|------------|-------------|--------|
| **SFT** | Supervised Fine-Tuning | [experiment/README.md](experiment/README.md) |
| **RLHF** | Reinforcement Learning from Human Feedback | [experiment_rlhf/README.md](experiment_rlhf/README.md) |

## Quick Start

```bash
pip install -r requirements.txt
```

### SFT Quick Start

```bash
# Run all SFT experiments
bash experiment/train/launch_experiments.sh --task mmlu --subject sociology

# Single SFT experiment with layer-wise selection
bash experiment/train/train.sh --task mmlu --model llama3-1b \
    --data_selection GREATS --compression GraSS --MeSO
```

### RLHF Quick Start

```bash
# Run all RLHF experiments
bash experiment_rlhf/launch_experiments.sh

# Single RLHF experiment with layer-wise selection
bash experiment_rlhf/train.sh --model llama3-1b --lora \
    --selection_method GREATS --compression GraSS
```

## Data Preparation

Download and prepare datasets using the unified data preparation script:

```bash
# See available datasets and options
python experiment/data/prepare_datasets.py -h

# Download evaluation datasets
python experiment/data/prepare_datasets.py --datasets mmlu bbh tydiqa gsm8k math500 samsum

# Download training datasets
python experiment/data/prepare_datasets.py --datasets alpaca dolly flan_v2 cot oasst1
```

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

## Core Modules

### `compress_gradient/`

The core library for gradient compression and layer-wise selection:

- **`hook.py`**: `GradientHook` - Hooks for capturing compressed gradients with selection support
- **`compressor.py`**: `Compressor` - Two-stage compression (sparsification + projection)
- **`optimizer.py`**: `MeSOAdamW` - Memory-efficient optimizer in compressed space
- **`trainer.py`**: `CompGradTrainer` - SFT trainer with data selection

### Key Concepts

1. **Layer-wise Selection**: Each layer independently selects beneficial samples based on gradient alignment with validation gradients
2. **Two-Stage Compression**:
   - Stage 1: Sparsification (random mask or low-rank)
   - Stage 2: Projection (SJLT sketching or identity)
3. **Streaming Update**: Selection and model update happen during the same backward pass

## Available Datasets

### Evaluation Datasets

| Dataset   | Task Type          | Description                                              |
| --------- | ------------------ | -------------------------------------------------------- |
| `mmlu`    | Multiple Choice    | Massive Multitask Language Understanding (57 subjects)   |
| `bbh`     | Reasoning          | BIG-Bench Hard (23 challenging reasoning tasks with CoT) |
| `tydiqa`  | Question Answering | Typologically Diverse QA (9 languages)                   |
| `gsm8k`   | Math               | Grade School Math (8K problems)                          |
| `math500` | Math               | MATH benchmark (500 competition problems)                |
| `samsum`  | Summarization      | SAMSum dialogue summarization                            |

### Training Datasets

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

## Detailed Documentation

For detailed experiment-specific documentation, see:

- **SFT Experiments**: [experiment/README.md](experiment/README.md)
- **RLHF Experiments**: [experiment_rlhf/README.md](experiment_rlhf/README.md)