# Dr. Post-Training

This is the official implementation of [Dr. Post-Training: A Data Regularization Perspective on LLM Post-Training](https://arxiv.org/abs/2605.07063).

## Getting Started

The trainer of SFT and RLHF is implemented in plain PyTorch without advanced distributed training frameworks (e.g., DeepSpeed, FairScale, or Hugging Face Accelerator) to maximize clarity and ease of understanding. For large-scale training, we provide our implementation in the RLVR experiment with [Verl](https://github.com/volcengine/verl) (Ray-based distributed RL) with vLLM for fast generation.

```bash
# Clone with submodules
git clone --recursive https://github.com/TRAIS-Lab/Dr.Post-Training.git

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

## Cluster Setup

Both `cluster_env.sh` and `submit.sh` are **gitignored** — each user creates their own at the repo root. Every script sources `cluster_env.sh` for paths and conda activation, while `submit.sh` wraps `sbatch` with cluster-specific SLURM defaults.

### 1. Create `cluster_env.sh`

```bash
cat > cluster_env.sh << 'EOF'
# Cluster-specific configuration
# This file is gitignored — safe to edit without merge conflicts.

# Directory paths
export SCRATCH_DIR="/scratch/$USER/Project"   # where data/checkpoints live
export CODE_DIR="$HOME/Project"               # where this repo is cloned

# Conda environment activation
activate_env() { conda activate drpt; }
# Or if conda activate doesn't work in non-interactive shells:
# activate_env() { export PATH="$HOME/.conda/envs/drpt/bin:$PATH"; }

# SLURM defaults
export SLURM_ACCOUNT="my-account"
export SLURM_PARTITION="gpuA40x4"
export SLURM_MAIL_USER="user@example.edu"
EOF
```

### 2. Create `submit.sh`

```bash
cat > submit.sh << 'SCRIPT'
#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }

SCRIPT="$1"; shift
JOB_NAME="${JOB_NAME:-$(basename "$SCRIPT" .sh)}"
GPUS="${GPUS:-4}"
MEM="${MEM:-128g}"
CPUS="${CPUS:-16}"
TIME="${TIME:-12:00:00}"
CONSTRAINT="${CONSTRAINT:-scratch}"
LOG_DIR="${CODE_DIR}/Dr.Post-Training/log"
mkdir -p "$LOG_DIR"

sbatch --job-name="$JOB_NAME" --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
    --mail-user="$SLURM_MAIL_USER" --mail-type="END" --mem="$MEM" --nodes=1 \
    --ntasks-per-node=1 --cpus-per-task="$CPUS" --gpus-per-node="$GPUS" \
    --gpu-bind=none --time="$TIME" --constraint="$CONSTRAINT" \
    --output="$LOG_DIR/%x-%j.log" "$SCRIPT" "$@"
SCRIPT
chmod +x submit.sh
```

### 3. Submit jobs

```bash
# SFT training
./submit.sh SFT/train/train.sh --methods baseline

# RLHF training
./submit.sh RLHF/train/train.sh --methods all

# RLVR training
./submit.sh RLVR/scripts/run_qwen1.7b_math_grpo.sh

# Override SLURM defaults per-invocation
GPUS=1 TIME=1:00:00 MEM=64g ./submit.sh SFT/eval/eval.sh --task samsum
```

## Experiments

| Experiment | Environment | Description                                                             | Documentation                    |
| ---------- | ----------- | ----------------------------------------------------------------------- | -------------------------------- |
| **SFT**    | `drpt`      | Supervised Fine-Tuning with layer-wise-subset data curation                     | [SFT/README.md](SFT/README.md)   |
| **RLHF**   | `drpt`      | Reinforcement Learning from Human Feedback with layer-wise-subset data curation | [RLHF/README.md](RLHF/README.md) |
| **RLVR**   | `drpt_rlvr` | Reinforcement Learning with Verifiable Rewards (VERL + vLLM)            | [RLVR/README.md](RLVR/README.md) |

## TODOs

1. Gradient Accumulation
2. Adaptive Exact Scoring
