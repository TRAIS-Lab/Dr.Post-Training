# Gradient Streaming

This repository implements **Gradient Streaming** for memory-efficient fine-tuning with compressed gradient scoring and data selection.

## Overview

Gradient Streaming enables on-the-fly data selection during training by computing gradient-based scores layer-by-layer, avoiding the need to store full gradients. The codebase supports:

- **Data Selection** - Streaming (per-layer) or GREATS (global) selection
- **Gradient Compression** - GraSS or LoGra methods for reduced memory
- **Training Type** - Full fine-tuning or LoRA

## Installation

```bash
pip install -r requirements.txt
```

### Recommended Environment Setup

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

## Core Library: `gradstream`

The `gradstream/` module provides the core components for gradient streaming:

| Component                 | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `GradientHook`            | Captures and processes gradients layer-by-layer       |
| `setup_model_compressors` | Sets up gradient compressors (sparsifiers/projectors) |
| `MeSOAdamW`               | Memory-efficient Subspace Optimizer                   |

```python
from gradstream import GradientHook, MeSOAdamW, setup_model_compressors
```

Task-specific trainers (e.g., `StreamingTrainer` for SFT) are in their respective experiment directories.

## Experiments

| Experiment | Description                                                        | Documentation                    |
| ---------- | ------------------------------------------------------------------ | -------------------------------- |
| **SFT**    | Supervised Fine-Tuning with gradient streaming data selection      | [SFT/README.md](SFT/README.md)   |
| **RLHF**   | Reinforcement Learning from Human Feedback with gradient streaming | [RLHF/README.md](RLHF/README.md) |
