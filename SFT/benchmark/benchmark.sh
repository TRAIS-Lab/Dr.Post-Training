#!/bin/bash
# Timing Breakdown Benchmark for Dr. Post-Training
#
# Runs the detailed timing breakdown benchmark comparing:
#   Standard vs Layerwise (per-layer) vs Subset (global)
#
# Usage:
#   bash benchmark.sh [OPTIONS]
#
# Examples:
#   bash benchmark.sh
#   bash benchmark.sh --batch-size 8 --seq-length 256
#   bash benchmark.sh --model meta-llama/Llama-3.2-1B --num-iterations 30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source cluster config
source "$PROJECT_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found. See README.md for cluster setup."; exit 1; }
activate_env

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Python interpreter (override with PYTHON env var if needed)
PYTHON="${PYTHON:-python}"

echo ""
echo "========================================================"
echo "  Timing Breakdown Benchmark"
echo "  Standard vs Layerwise vs Subset"
echo "========================================================"
echo ""

# Clear GPU memory before run
echo "Clearing GPU memory..."
$PYTHON -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); print(f'GPU cleared: {torch.cuda.memory_allocated()/1024**3:.3f} GB allocated')" 2>/dev/null || true
echo ""

# Run the breakdown benchmark, passing all CLI args through
$PYTHON "$SCRIPT_DIR/benchmark.py" "$@"

echo ""
echo "========================================================"
echo "  Benchmark Complete!"
echo "========================================================"
