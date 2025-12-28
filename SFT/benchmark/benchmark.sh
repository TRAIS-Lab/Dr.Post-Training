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
#   bash benchmark.sh --methods NA-NA-Full,Streaming-LoGra-Full
#   bash benchmark.sh --methods all
#   bash benchmark.sh --methods baseline --batch-size 32 --num-iterations 20

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Python interpreter (override with PYTHON env var if needed)
PYTHON="${PYTHON:-$HOME/miniconda3/envs/IF/bin/python}"

# Default values
methods=""
dry_run=false

# Method definitions (matching train.sh naming convention)
# Maps method names to benchmark internal names (uses underscore internally)
declare -A METHOD_TO_INTERNAL=(
    ["NA-NA-Full"]="NA_NA_Full"
    ["NA-NA-LoRA"]="NA_NA_LoRA"
    ["Streaming-NA-Full"]="Streaming_NA_Full"
    ["Streaming-NA-LoRA"]="Streaming_NA_LoRA"
    ["GREATS-NA-Full"]="GREATS_NA_Full"
    ["GREATS-NA-LoRA"]="GREATS_NA_LoRA"
    ["Streaming-LoGra-Full"]="Streaming_LoGra_Full"
    ["GREATS-LoGra-Full"]="GREATS_LoGra_Full"
)

# Category mappings (matching train.sh)
declare -A CATEGORY_METHODS=(
    ["all"]="NA-NA-Full,NA-NA-LoRA,Streaming-NA-Full,Streaming-NA-LoRA,GREATS-NA-Full,GREATS-NA-LoRA,Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["baseline"]="NA-NA-Full,NA-NA-LoRA"
    ["streaming"]="Streaming-NA-Full,Streaming-NA-LoRA,Streaming-LoGra-Full"
    ["greats"]="GREATS-NA-Full,GREATS-NA-LoRA,GREATS-LoGra-Full"
    ["full"]="NA-NA-Full,Streaming-NA-Full,GREATS-NA-Full,Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["lora"]="NA-NA-LoRA,Streaming-NA-LoRA,GREATS-NA-LoRA"
    ["compression"]="Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["no-compression"]="NA-NA-Full,NA-NA-LoRA,Streaming-NA-Full,Streaming-NA-LoRA,GREATS-NA-Full,GREATS-NA-LoRA"
)

# Collect other arguments to pass to benchmark.py
OTHER_ARGS=()

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)
            methods="$2"
            shift 2
            ;;
        --list)
            # Just show the list and exit
            $PYTHON "$SCRIPT_DIR/benchmark.py" --list
            exit 0
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run benchmarks for SFT methods with proper GPU memory cleanup."
            echo ""
            echo "Options:"
            echo "  --methods <list>           Methods to benchmark (comma-separated or category)"
            echo "  --list                     List available methods and exit"
            echo "  --dry-run                  Print commands without executing"
            echo ""
            echo "  Method names:"
            echo "    NA-NA-Full, NA-NA-LoRA, Streaming-NA-Full, Streaming-NA-LoRA,"
            echo "    GREATS-NA-Full, GREATS-NA-LoRA, Streaming-LoGra-Full, GREATS-LoGra-Full"
            echo ""
            echo "  Categories: all, baseline, streaming, greats, full, lora, compression, no-compression"
            echo ""
            echo "  Examples:"
            echo "    --methods all                              Run all 8 methods"
            echo "    --methods baseline                         Run NA-NA-Full and NA-NA-LoRA"
            echo "    --methods \"NA-NA-Full,Streaming-LoGra-Full\" Run specific methods"
            echo ""
            echo "Benchmark Options (passed to benchmark.py):"
            echo "  --batch-size N             Batch size for benchmarking"
            echo "  --num-iterations N         Number of iterations"
            echo "  --model PATH               Model path"
            echo ""
            echo "Note: Each method runs in a separate Python process with GPU memory"
            echo "      cleared between runs to ensure accurate memory measurements."
            exit 0
            ;;
        *)
            # Pass through other arguments to benchmark.py
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

# ========================================
# Resolve method names from categories
# ========================================
resolve_methods() {
    local input="$1"
    local resolved=""

    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)  # trim whitespace

        # Check if it's a category
        if [[ -n "${CATEGORY_METHODS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,${CATEGORY_METHODS[$item]}"
            else
                resolved="${CATEGORY_METHODS[$item]}"
            fi
        # Check if it's a valid method name
        elif [[ -n "${METHOD_TO_INTERNAL[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,$item"
            else
                resolved="$item"
            fi
        else
            echo "ERROR: Unknown method or category: $item"
            echo "Valid methods: ${!METHOD_TO_INTERNAL[*]}"
            echo "Valid categories: ${!CATEGORY_METHODS[*]}"
            exit 1
        fi
    done

    # Remove duplicates while preserving order
    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# ========================================
# Main execution
# ========================================

# If no methods specified, use defaults
if [[ -z "$methods" ]]; then
    methods="NA-NA-Full,Streaming-LoGra-Full"
fi

# Resolve methods
resolved_methods=$(resolve_methods "$methods")

# Convert to array
IFS=',' read -ra method_list <<< "$resolved_methods"
TOTAL=${#method_list[@]}

# Create a temp file for aggregating results
RESULTS_FILE=$(mktemp /tmp/benchmark_results_XXXXXX.jsonl)
trap "rm -f $RESULTS_FILE" EXIT

echo ""
echo "========================================================"
echo "  SFT Benchmark"
echo "========================================================"
echo "Methods:       $resolved_methods"
echo "Total:         $TOTAL methods"
echo "Results file:  $RESULTS_FILE"
if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
    echo ""
    echo "Benchmark args: ${OTHER_ARGS[*]}"
fi
echo "========================================================"

# Run each method in a separate Python process
current=0

for method_name in "${method_list[@]}"; do
    current=$((current + 1))

    # Convert method name to internal name (dash to underscore)
    internal_name="${METHOD_TO_INTERNAL[$method_name]}"

    echo ""
    echo "========================================================"
    echo "  [$current/$TOTAL] $method_name"
    echo "========================================================"

    # Clear GPU memory before each run
    echo "Clearing GPU memory..."
    $PYTHON -c "import torch; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); print(f'GPU cleared: {torch.cuda.memory_allocated()/1024**3:.3f} GB allocated')" 2>/dev/null || true

    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] Would execute: $PYTHON $SCRIPT_DIR/benchmark.py --methods $internal_name --results-file $RESULTS_FILE ${OTHER_ARGS[*]}"
    else
        # Run benchmark for this single method, appending to results file
        $PYTHON "$SCRIPT_DIR/benchmark.py" --methods "$internal_name" --results-file "$RESULTS_FILE" "${OTHER_ARGS[@]}"
    fi

    echo ""
done

# Print aggregated summary table
echo ""
echo "========================================================"
echo "  Benchmark Complete!"
echo "========================================================"
echo "Total: $TOTAL methods"

if [ "$dry_run" = false ]; then
    echo ""
    $PYTHON "$SCRIPT_DIR/benchmark.py" --print-summary "$RESULTS_FILE"
fi
echo "========================================================"
