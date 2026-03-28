#!/bin/bash
# Run all benchmark experiments directly (no Slurm).
# 3 models x 3 configs x 13 combos = 117 runs across 3 GPUs in parallel.
#
# Crossover design (V* = sum(O*I) / (S * sum(O+I)), all at T=512):
#   Qwen3-0.6B    n=12  V*=1.36  → crossover between m=1 and m=2
#   Qwen3-1.7B    n=8   V*=2.53  → crossover between m=2 and m=4
#   Llama-3.2-3B  n=4   V*=3.62  → crossover at m≈4
#
# Memory verified on A40 (46GB) — all non-direct combos fit.
# Direct scoring may OOM at larger m; this is expected.
#
# Usage:
#   bash SFT/benchmark/run_all.sh
#   bash SFT/benchmark/run_all.sh 2>&1 | tee SFT/benchmark/results/run_all.log

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHON="/home/pbb/miniconda3/envs/GradStream/bin/python"
RUNNER="SFT/benchmark/benchmark_run.py"
RESULTS="SFT/benchmark/results"
mkdir -p "$RESULTS"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

run_model() {
    local gpu="$1" model="$2" tag="$3" n="$4" T="$5"
    shift 5
    local m_values=("$@")

    echo "[GPU $gpu] Starting $tag (n=$n T=$T m=${m_values[*]})"
    for m in "${m_values[@]}"; do
        local out="$RESULTS/${tag}_n${n}_T${T}_m${m}.json"
        if [[ -f "$out" ]]; then
            echo "[GPU $gpu] SKIP $tag n=$n T=$T m=$m (exists)"
            continue
        fi
        echo "[GPU $gpu] Running $tag n=$n T=$T m=$m ..."
        $PYTHON "$RUNNER" \
            --gpu "$gpu" \
            --model "$model" \
            --batch-size "$n" \
            --seq-length "$T" \
            --val-batch-size "$m" \
            --output "$out" \
            2>&1 | tail -5
        echo "[GPU $gpu] Done $tag m=$m"
    done
    echo "[GPU $gpu] Finished $tag"
}

echo "========================================"
echo "  Dr.Post-Training Benchmark (A40)"
echo "  3 models x 3 configs x 13 combos = 117 runs"
echo "  Started: $(date)"
echo "========================================"

# Run 3 models in parallel on separate GPUs
run_model 0  "Qwen/Qwen3-0.6B"         "qwen3-0.6b"   12  512  1 2 4 &
run_model 1  "Qwen/Qwen3-1.7B"         "qwen3-1.7b"    8  512  1 2 4 &
run_model 2  "meta-llama/Llama-3.2-3B"  "llama-3.2-3b"  4  512  1 2 4 &

wait

echo "========================================"
echo "  All done: $(date)"
echo "  Results: $RESULTS/"
echo "========================================"
