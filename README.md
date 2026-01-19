# One Stream to Rule Them All

This repository implements **Gradient Streaming** for fine-grained data selection for modern model training.

The trainer of SFT and RLHF is implemented in plain PyTorch without advanced distributed training frameworks (e.g., DeepSpeed, FairScale, or Hugging Face Accelerator) to maximize clarity and ease of understanding. For large-scale training, we provide our implementation in the RLVR experiment with [VERL](https://github.com/volcengine/verl) (Ray-based distributed RL) with vLLM for fast generation.

## Getting Started

```bash
# Clone with submodules
git clone --recursive https://github.com/TRAIS-Lab/Gradient-Streaming.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

## Environment Setup

> [!IMPORTANT]
> **SFT/RLHF** and **RLVR** require **different conda environments** due to incompatible dependencies (e.g., `transformers` version conflicts). Choose the appropriate setup below.

### For SFT and RLHF

```bash
conda create -n GradStream python=3.10
conda activate GradStream

conda install -c "nvidia/label/cuda-12.4.0" cudatoolkit
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip3 install packaging ninja psutil
pip3 install sjlt --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

pip3 install -r requirements.txt
```

>[!Note]
>It is only required to install `cudatoolkit` with appropriate `torch` in order to build `sjlt`. Without `sjlt` installation, you can still run the experiment with other gradient compression methods (e.g., LoGra), which is the default. For instance, the following should also work as long as you don't use `GraSS` compression (which requires `sjlt`):
> ```bash
> conda create -n GradStream python=3.10
> conda activate GradStream
> pip3 install -r requirements.txt
> pip3 install flash-attn --no-build-isolation --no-cache-dir
> ```

### For RLVR

To set up the environment for RLVR experiments, use the following commands:

```bash
conda create -n GradStream_RLVR python=3.12
conda activate GradStream_RLVR

# Install VERL (submodule)
cd RLVR/verl
pip install -e ".[vllm,math]"
pip install flash-attn --no-build-isolation --no-cache-dir
```

Note that due to the complicated dependencies of VERL (which is included as a git submodule) and vLLM, we recommend using a separate conda environment for RLVR experiments and let the VERL installation handle all the dependencies.

## Experiments

| Experiment | Environment       | Description                                                        | Documentation                    |
| ---------- | ----------------- | ------------------------------------------------------------------ | -------------------------------- |
| **SFT**    | `GradStream`      | Supervised Fine-Tuning with gradient streaming data selection      | [SFT/README.md](SFT/README.md)   |
| **RLHF**   | `GradStream`      | Reinforcement Learning from Human Feedback with gradient streaming | [RLHF/README.md](RLHF/README.md) |
| **RLVR**   | `GradStream-RLVR` | Reinforcement Learning with Verifiable Rewards (VERL + vLLM)       | [RLVR/README.md](RLVR/README.md) |

## Methods

Generally speaking, we support the following:

1. Selection Modes
   - **NA**: No data selection
   - **Baseline**: Data selection in the existing literature
   - **GREATS**: Global online selection
   - **Streaming**: Per-layer selection
2. Compression Methods
   - **NA**: No compression - uses full gradients and standard AdamW optimizer
   - **LoGra**: Low-rank Gradient compression (Gaussian projection) - uses MeSO optimizer
   - **GraSS**: Gradient Sparsification with Sketching (available but not used in default methods)
3. Training Types
   - **Full**: Full fine-tuning of all model parameters
   - **LoRA**: LoRA fine-tuning of low-rank adapters only
