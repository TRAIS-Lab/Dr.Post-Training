# Dr. Post-Training

This repository implements **Dr. Post-Training** for fine-grained data curation for modern model training.

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
conda create -n drpt python=3.10
conda activate drpt

conda install -c "nvidia/label/cuda-12.4.0" cudatoolkit
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip3 install packaging ninja psutil
pip3 install sjlt --no-build-isolation
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

pip3 install -r requirements.txt
```

>[!Note]
>It is only required to install `cudatoolkit` with appropriate `torch` in order to build `sjlt`. Without `sjlt` installation, you can still run the experiment with other gradient compression methods, such as the default. For instance, the following should also work as long as you don't use `GraSS` compression (which requires `sjlt`):
> ```bash
> conda create -n drpt python=3.10
> conda activate drpt
> pip3 install -r requirements.txt
> pip3 install flash-attn --no-build-isolation --no-cache-dir
> ```

### For RLVR

To set up the environment for RLVR experiments, use the following commands:

```bash
conda create -n drpt_rlvr python=3.12
conda activate drpt_rlvr

# Install VERL (submodule)
cd RLVR/verl
pip install -e ".[vllm,math]"
pip install flash-attn --no-build-isolation --no-cache-dir
```

Note that due to the complicated dependencies of VERL (which is included as a git submodule) and vLLM, we recommend using a separate conda environment for RLVR experiments and let the VERL installation handle all the dependencies.

## Experiments

| Experiment | Environment | Description                                                              | Documentation                    |
| ---------- | ----------- | ------------------------------------------------------------------------ | -------------------------------- |
| **SFT**    | `drpt`      | Supervised Fine-Tuning with layerwise data curation                     | [SFT/README.md](SFT/README.md)   |
| **RLHF**   | `drpt`      | Reinforcement Learning from Human Feedback with layerwise data curation | [RLHF/README.md](RLHF/README.md) |
| **RLVR**   | `drpt_rlvr` | Reinforcement Learning with Verifiable Rewards (VERL + vLLM)             | [RLVR/README.md](RLVR/README.md) |

## Methods

Generally speaking, we support the following:

1. Curation Modes
   - **NA**: No data curation
   - **Baseline**: Data curation in the existing literature
   - **Subset**: Global online curation
   - **Layerwise**: Per-layer curation
2. Compression Methods
   - **NA**: No compression - uses full gradients and standard AdamW optimizer
   - **LoGra**: Low-rank Gradient compression (Gaussian projection) - uses MeSO optimizer
   - **GraSS**: Gradient Sparsification with Sketching (available but not used in default methods)
3. Training Types
   - **Full**: Full fine-tuning of all model parameters
   - **LoRA**: LoRA fine-tuning of low-rank adapters only
