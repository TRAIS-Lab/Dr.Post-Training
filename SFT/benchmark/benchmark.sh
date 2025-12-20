#!/bin/bash
# Benchmark launcher with proper memory cleanup between runs
# Each method runs in a separate Python process with GPU memory cleared between runs
#
# IMPORTANT: Always use this script instead of running benchmark.py directly
# to ensure GPU memory is properly cleaned up between method runs.
#
# Usage:
#   bash benchmark.sh [OPTIONS]
#
# Examples:
#   bash benchmark.sh --methods NA_NA_full Streaming_LoGra_full
#   bash benchmark.sh --all
#   bash benchmark.sh --methods GREATS_LoGra_full --batch-size 32 --num-iterations 20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Python interpreter (override with PYTHON env var if needed)
PYTHON="${PYTHON:-$HOME/miniconda3/envs/IF/bin/python}"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_info() {
    echo -e "${YELLOW}$1${NC}"
}

# Parse arguments to extract methods
METHODS=()
OTHER_ARGS=()
CAPTURE_METHODS=false

for arg in "$@"; do
    if [ "$arg" = "--methods" ]; then
        CAPTURE_METHODS=true
    elif [ "$arg" = "--all" ]; then
        # Get all methods from --list output (column 2, skip header lines)
        METHODS=($($PYTHON "$SCRIPT_DIR/benchmark.py" --list 2>&1 | awk 'NR>3 && /^[0-9]/ {print $2}'))
        CAPTURE_METHODS=false
    elif [ "$arg" = "--list" ]; then
        # Just show the list and exit
        $PYTHON "$SCRIPT_DIR/benchmark.py" --list
        exit 0
    elif [ "$CAPTURE_METHODS" = true ] && [[ "$arg" != --* ]]; then
        METHODS+=("$arg")
    else
        CAPTURE_METHODS=false
        OTHER_ARGS+=("$arg")
    fi
done

# If no methods specified, use defaults
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=("NA_NA_full" "Streaming_LoGra_full")
fi

# Create a temp file for aggregating results
RESULTS_FILE=$(mktemp /tmp/benchmark_results_XXXXXX.jsonl)
trap "rm -f $RESULTS_FILE" EXIT

print_header "Benchmark Configuration"
echo "Methods to run: ${METHODS[*]}"
echo "Other args: ${OTHER_ARGS[*]}"
echo "Results file: $RESULTS_FILE"
echo ""

# Run each method in a separate Python process
TOTAL=${#METHODS[@]}
CURRENT=0

for METHOD in "${METHODS[@]}"; do
    CURRENT=$((CURRENT + 1))

    print_header "[$CURRENT/$TOTAL] Running: $METHOD"

    # Clear GPU memory before each run
    print_info "Clearing GPU memory..."
    $PYTHON -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); print(f'GPU cleared: {torch.cuda.memory_allocated()/1024**3:.3f} GB allocated')" 2>/dev/null || true

    # Run benchmark for this single method, appending to results file
    $PYTHON "$SCRIPT_DIR/benchmark.py" --methods "$METHOD" --results-file "$RESULTS_FILE" "${OTHER_ARGS[@]}"

    echo ""
done

# Print aggregated summary table
print_header "ALL BENCHMARKS COMPLETE"
$PYTHON "$SCRIPT_DIR/benchmark.py" --print-summary "$RESULTS_FILE"
