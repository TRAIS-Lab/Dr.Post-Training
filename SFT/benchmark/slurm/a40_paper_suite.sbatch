#!/bin/bash
# A40 Slurm cluster (one node, 4x A40): regenerate the paper's A40 system tables from the current code state.
# Runs the full run_benchmarks.sh suite (breakdown, checkpointing, scoring) into a fresh
# RESULTS_ROOT, adds the Llama-3.2-3B cuBLASLt Standard baselines, and regenerates the
# compress/pip/gip table sets. The existing results/paper/ is left untouched; compare and
# swap by hand afterwards.
#
#   sbatch [--dependency=afterany:<jobid>] SFT/benchmark/slurm/a40_paper_suite.sbatch
#   RESULTS_ROOT=SFT/benchmark/results/paper_v3 (default); override by exporting RESULTS_ROOT before sbatch
#SBATCH --job-name=paper-suite --partition=standard --gres=gpu:4
#SBATCH --cpus-per-task=32 --mem=160G --time=6:00:00
#SBATCH --output=logs/benchmark/%x_%j.out
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source cluster_env.sh; activate_env
export PYTHON="$(command -v python)"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export RESULTS_ROOT="${RESULTS_ROOT:-SFT/benchmark/results/paper_v3}"
mkdir -p "$RESULTS_ROOT"
echo "[suite] $(date)  RESULTS_ROOT=$RESULTS_ROOT  GPUs=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

# 1) main suite: 3 models in parallel on GPUs 0-2, no ckpt -> ckpt -> standalone scoring
bash SFT/benchmark/run_benchmarks.sh breakdown
bash SFT/benchmark/run_benchmarks.sh checkpointing
bash SFT/benchmark/run_benchmarks.sh scoring

# 2) Llama-3.2-3B Standard baseline with cuBLASLt (works around the torch/cuBLAS down_proj
#    w.grad kernel choice on A40); system_efficiency.py reads *_full_training_cublaslt.json
LLAMA="meta-llama/Llama-3.2-3B"
for ckpt in "" "--gradient-checkpointing"; do
    sub=breakdown; [[ -n "$ckpt" ]] && sub=breakdown_checkpointing
    for cfg in "8,512,1" "2,1024,1"; do
        IFS=',' read -r n T m <<< "$cfg"
        out="$RESULTS_ROOT/$sub/llama-3.2-3b_n${n}_T${T}_m${m}_full_training_cublaslt.json"
        [[ -f "$out" ]] && { echo "[suite] SKIP $out"; continue; }
        echo "[suite] $(date) cuBLASLt Standard n=$n T=$T $ckpt"
        CUDA_VISIBLE_DEVICES=2 $PYTHON SFT/benchmark/benchmark.py --model "$LLAMA" --method full_training --dataset dummy --num-warmup 10 --num-iterations 20 \
            --batch-size "$n" --seq-length "$T" --val-batch-size "$m" --cublaslt $ckpt --output "$out" 2>&1 | tail -3
    done
done

# 3) paper tables (exact GIP curated columns, written into Paper/<venue>/Tables/ by the registered generator;
#    `--scoring compress|pip` prints the same grids for the other backends to stdout)
$PYTHON SFT/tables/system_efficiency.py --benchmark-dir "$RESULTS_ROOT"
echo "[suite] done $(date)"
