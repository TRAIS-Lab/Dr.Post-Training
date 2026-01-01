# One Stream to Rule Them All

This repository implements **Gradient Streaming** for fine-grained data selection for modern model training.

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

## Experiments

| Experiment | Description                                                        | Documentation                    |
| ---------- | ------------------------------------------------------------------ | -------------------------------- |
| **SFT**    | Supervised Fine-Tuning with gradient streaming data selection      | [SFT/README.md](SFT/README.md)   |
| **RLHF**   | Reinforcement Learning from Human Feedback with gradient streaming | [RLHF/README.md](RLHF/README.md) |

Generally speaking, we support the following:

1. Selection Modes
   - **NA**: No data selection (baseline PPO)
   - **Streaming**: Per-layer selection - each layer independently selects samples (single-pass)
   - **GREATS**: Global selection - accumulates scores across all layers (two-pass)
2. Compression Methods
   - **NA**: No compression - uses full gradients and standard AdamW optimizer
   - **LoGra**: Low-rank Gradient compression (Gaussian projection) - uses MeSO optimizer
   - **GraSS**: Gradient Sparsification with Sketching (available but not used in default methods)
3. Training Types
   - **Full**: Full fine-tuning of all model parameters
   - **LoRA**: LoRA fine-tuning of low-rank adapters only

>[!Note]
>This library is implemented in plain PyTorch without advanced distributed training frameworks (e.g., DeepSpeed, FairScale, or Hugging Face Accelerator) to maximize clarity and ease of understanding. For large-scale training, integrating with such frameworks may be necessary.