#!/bin/bash
# =============================================================================
# Dr. Post-Training — Timing Benchmarks
# =============================================================================
#
# Two benchmarks:
#   1. Breakdown: Per-component timing for all 4 methods (Standard, Layerwise,
#      Subset 2P, Subset 1P) with all scoring variants. Uses real models.
#   2. Scoring: Standalone scoring comparison with synthetic tensors.
#      Shows regime-dependent optimal scoring method.
#
# Models: Qwen3-0.6B, Qwen3-1.7B, Llama-3.2-3B
# Configs: Same (n, T, m) across all models for fair comparison.
#   Config A: n=8  T=512  m=1
#   Config B: n=2  T=1024 m=1
#
# Hardware: 3x NVIDIA A40 (46GB), one model per GPU.
#
# Usage:
#   bash SFT/benchmark/run_benchmarks.sh              # Run both
#   bash SFT/benchmark/run_benchmarks.sh breakdown     # Breakdown only
#   bash SFT/benchmark/run_benchmarks.sh scoring       # Scoring only
# =============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export PYTHON="/home/pbb/miniconda3/envs/GradStream/bin/python"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

MODE="${1:-both}"

# ── Benchmark 1: Breakdown ──────────────────────────────────────────────────

run_breakdown() {
    local RESULTS="SFT/benchmark/results/breakdown"
    mkdir -p "$RESULTS"

    run_model() {
        local gpu="$1" model="$2" tag="$3"
        shift 3
        local configs=("$@")
        echo "[GPU $gpu] $tag: ${#configs[@]} configs"
        for cfg in "${configs[@]}"; do
            IFS=',' read -r n T m <<< "$cfg"
            local out="$RESULTS/${tag}_n${n}_T${T}_m${m}.json"
            if [[ -f "$out" ]]; then
                echo "[GPU $gpu] SKIP $tag n=$n T=$T m=$m"
                continue
            fi
            echo "[GPU $gpu] Running $tag n=$n T=$T m=$m ..."
            $PYTHON SFT/benchmark/benchmark_run.py \
                --gpu "$gpu" --model "$model" \
                --batch-size "$n" --seq-length "$T" --val-batch-size "$m" \
                --direct-batch-size 1 \
                --output "$out" 2>&1 | tail -5
        done
        echo "[GPU $gpu] $tag done"
    }

    echo "  Benchmark 1: Breakdown (3 models x 2 configs x 13 combos)"
    run_model 0 "Qwen/Qwen3-0.6B"         "qwen3-0.6b"    "8,512,1" "2,1024,1" &
    run_model 1 "Qwen/Qwen3-1.7B"         "qwen3-1.7b"    "8,512,1" "2,1024,1" &
    run_model 2 "meta-llama/Llama-3.2-3B"  "llama-3.2-3b"  "8,512,1" "2,1024,1" &
    wait

    echo "  Generating tables..."
    $PYTHON SFT/benchmark/aggregate_breakdown.py \
        --results-dir "$RESULTS" --output "$RESULTS/breakdown_tables.txt" 2>/dev/null
}

# ── Benchmark 2: Scoring ────────────────────────────────────────────────────

run_scoring() {
    local RESULTS="SFT/benchmark/results/scoring"
    mkdir -p "$RESULTS"

    echo "  Benchmark 2: Scoring (3 models x 10 configs x 4 methods)"
    $PYTHON SFT/benchmark/benchmark_scoring.py \
        --gpu 0 --model-tag qwen3-0.6b \
        --output "$RESULTS/scoring_qwen3-0.6b.json" &

    $PYTHON SFT/benchmark/benchmark_scoring.py \
        --gpu 1 --model-tag qwen3-1.7b \
        --output "$RESULTS/scoring_qwen3-1.7b.json" &

    $PYTHON SFT/benchmark/benchmark_scoring.py \
        --gpu 2 --model-tag llama-3.2-3b \
        --output "$RESULTS/scoring_llama-3.2-3b.json" &
    wait
}

# ── Main ────────────────────────────────────────────────────────────────────

echo "========================================"
echo "  Dr. Post-Training Benchmarks"
echo "  Started: $(date)"
echo "========================================"

if [[ "$MODE" == "breakdown" || "$MODE" == "both" ]]; then
    run_breakdown
fi

if [[ "$MODE" == "scoring" || "$MODE" == "both" ]]; then
    run_scoring
fi

echo "========================================"
echo "  Done: $(date)"
echo "  Results: SFT/benchmark/results/"
echo "========================================"
