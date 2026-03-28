#!/bin/bash
# Clean-process wrapper for benchmark.py.
# Ensures fresh GPU state by clearing memory before each run.
# Called by benchmark_run.py for each method x scoring combo.
#
# Usage:
#   bash benchmark.sh [OPTIONS]
#   bash benchmark.sh --method standard --batch-size 8 --seq-length 512

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source cluster config (skip if already activated, e.g. from Slurm)
if [[ -z "$PYTHONPATH" ]] || ! python -c "import drpt" 2>/dev/null; then
    if [[ -f "$PROJECT_ROOT/cluster_env.sh" ]]; then
        source "$PROJECT_ROOT/cluster_env.sh"
        activate_env 2>/dev/null
    fi
fi

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

PYTHON="${PYTHON:-python}"

# Clear GPU memory — ensures clean measurement for this run
$PYTHON -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
" 2>/dev/null || true

# Run benchmark
exec $PYTHON "$SCRIPT_DIR/benchmark.py" "$@"
