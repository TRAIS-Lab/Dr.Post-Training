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

>[!Note]
>This library is implemented in plain PyTorch without advanced distributed training frameworks (e.g., DeepSpeed, FairScale, or Hugging Face Accelerator) to maximize clarity and ease of understanding. For large-scale training, integrating with such frameworks may be necessary.