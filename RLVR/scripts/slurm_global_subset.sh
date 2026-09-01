#!/bin/bash
#SBATCH --job-name=rlvr-global-subset
#SBATCH --partition=general,overflow
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#
# Submit: sbatch RLVR/scripts/slurm_global_subset.sh

set -x

# SLURM copies scripts to /var/spool, so use SLURM_SUBMIT_DIR instead of BASH_SOURCE
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
activate_env

cd "$REPO_ROOT"

# Avoid permission errors on shared /tmp for tiktoken/vllm cache
export TIKTOKEN_CACHE_DIR="$SCRATCH_DIR/.cache/tiktoken"
export TMPDIR="$SCRATCH_DIR/.cache/tmp"
mkdir -p "$TIKTOKEN_CACHE_DIR" "$TMPDIR"

SEED="${1:-42}"
exec bash RLVR/train.sh -c configs/math -m GlobalSubset-Full --seed "$SEED"
