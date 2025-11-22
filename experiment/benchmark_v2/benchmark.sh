#!/bin/bash
# Benchmark runner script for LLM fine-tuning methods
#
# Usage:
#   bash benchmark.sh [OPTIONS] [METHOD_INDICES...]
#
# Options:
#   --list              List all available methods and exit
#   --compare           Run MeSO vs GaLore comparison (methods 6, 7, 8, 9)
#   --meso-galore       Alias for --compare
#   --aggregate         Only aggregate existing results (skip running)
#
# Examples:
#   bash benchmark.sh --list                  # List all methods
#   bash benchmark.sh --compare               # Run MeSO vs GaLore comparison
#   bash benchmark.sh 6 8                     # Run specific methods (6 and 8)
#   bash benchmark.sh                         # Run ALL methods
#   bash benchmark.sh --aggregate             # Aggregate existing results only
#
# Environment variables:
#   OUTPUT_DIR          Directory for results (default: experiment/benchmark_v2/results)
#   PYTHON              Python interpreter (default: $HOME/miniconda3/envs/IF/bin/python)

cd $HOME/Project/Efficient-Fine-Tuning

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$HOME/Project/Efficient-Fine-Tuning:$PYTHONPATH"

set -e  # Exit on error

# Check for special flags
AGGREGATE_ONLY=false
LIST_ONLY=false
COMPARE_MODE=false

if [ "$1" = "--aggregate" ]; then
    AGGREGATE_ONLY=true
    shift  # Remove --aggregate from arguments
elif [ "$1" = "--list" ]; then
    LIST_ONLY=true
    shift  # Remove --list from arguments
elif [ "$1" = "--compare" ] || [ "$1" = "--meso-galore" ]; then
    COMPARE_MODE=true
    shift  # Remove flag from arguments
fi

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-experiment/benchmark_v2/results}"
PYTHON="${PYTHON:-$HOME/miniconda3/envs/IF/bin/python}"
BENCHMARK_SCRIPT="experiment/benchmark_v2/benchmark.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}================================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check if benchmark.py exists
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
    print_error "benchmark.py not found at $BENCHMARK_SCRIPT"
    exit 1
fi

# Handle --list flag: just show available methods and exit
if [ "$LIST_ONLY" = true ]; then
    print_header "AVAILABLE BENCHMARK METHODS"
    $PYTHON $BENCHMARK_SCRIPT --list
    exit 0
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

# If aggregate-only mode, skip to aggregation
if [ "$AGGREGATE_ONLY" = true ]; then
    print_header "AGGREGATE-ONLY MODE"
    print_info "Skipping benchmark runs, will only aggregate existing results"
    SUCCESSFUL_RUNS=1  # Set to 1 to allow aggregation to proceed
else
    # Get list of available methods
    print_info "Fetching available methods..."

    # Temporarily disable 'set -e' so we can catch the error instead of crashing
    set +e
    METHODS_OUTPUT=$($PYTHON $BENCHMARK_SCRIPT --list 2>&1)
    EXIT_CODE=$?
    set -e

    # Check if it failed
    if [ $EXIT_CODE -ne 0 ]; then
        print_error "Failed to list methods (Exit code: $EXIT_CODE)"
        echo -e "${RED}Python Error Traceback:${NC}"
        echo "$METHODS_OUTPUT"
        exit 1
    fi

    echo "$METHODS_OUTPUT"

    # Determine which methods to run
    if [ "$COMPARE_MODE" = true ]; then
        # MeSO vs GaLore comparison: run only these 4 methods
        METHODS_TO_RUN="16 17 20 21"
        print_info "COMPARISON MODE: Running MeSO vs GaLore (methods $METHODS_TO_RUN)"
    elif [ $# -eq 0 ]; then
        print_info "No method indices specified, will run ALL methods"
        # Extract method indices from --list output (lines like " 0: Method Name" or "10: Method Name")
        METHODS_TO_RUN=$(echo "$METHODS_OUTPUT" | grep -E "^ *[0-9]+:" | sed 's/^[[:space:]]*//' | cut -d: -f1)
    else
        METHODS_TO_RUN="$@"
        print_info "Running specified methods: $METHODS_TO_RUN"
    fi
fi

# Only run benchmarks if not in aggregate-only mode
if [ "$AGGREGATE_ONLY" = false ]; then
    # Count total methods
    TOTAL_METHODS=$(echo "$METHODS_TO_RUN" | wc -w)
    print_header "Starting benchmark of $TOTAL_METHODS method(s)"

    # Track successful and failed runs
    SUCCESSFUL_RUNS=0
    FAILED_RUNS=0
    FAILED_METHODS=""

    # Run each method
    CURRENT=0
    for METHOD_IDX in $METHODS_TO_RUN; do
    CURRENT=$((CURRENT + 1))

    print_header "[$CURRENT/$TOTAL_METHODS] Running method $METHOD_IDX"

    # Get method name for display
    METHOD_NAME=$($PYTHON $BENCHMARK_SCRIPT --list 2>&1 | grep -E "^ *${METHOD_IDX}:" | cut -d: -f2- | xargs)
    print_info "Method: $METHOD_NAME"

    # Clean GPU memory before each run
    print_info "Cleaning GPU memory..."
    $PYTHON -c "import torch; torch.cuda.empty_cache(); print(f'GPU memory cleared: {torch.cuda.memory_allocated()/1024**3:.3f} GB allocated')" 2>/dev/null || true

    # Run the benchmark
    print_info "Running benchmark..."

    # Run Python and capture exit code
    # Temporarily disable set -e to capture exit code properly
    set +e
    $PYTHON $BENCHMARK_SCRIPT --method "$METHOD_IDX" --output-dir "$OUTPUT_DIR"
    PYTHON_EXIT_CODE=$?
    set -e

    if [ $PYTHON_EXIT_CODE -eq 0 ]; then
        # Python finished successfully. Verify the result file exists in the output dir
        # Look for any JSON file starting with the method index (e.g., 01_Full_SGD_GC.json)
        FOUND_FILE=$(find "$OUTPUT_DIR" -name "$(printf '%02d' ${METHOD_IDX})_*.json" -print -quit)

        if [ -n "$FOUND_FILE" ]; then
            print_success "Method $METHOD_IDX completed successfully"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        else
            print_error "Method $METHOD_IDX finished but output file could not be verified"
            print_info "Expected pattern: $(printf '%02d' ${METHOD_IDX})_*.json"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            FAILED_METHODS="$FAILED_METHODS $METHOD_IDX"
        fi
    else
        print_error "Method $METHOD_IDX failed (Exit Code: $PYTHON_EXIT_CODE)"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_METHODS="$FAILED_METHODS $METHOD_IDX"
    fi

    echo ""
    done

    # Summary
    print_header "BENCHMARK RUN SUMMARY"
    echo -e "Total methods:    ${TOTAL_METHODS}"
    echo -e "Successful:       ${GREEN}${SUCCESSFUL_RUNS}${NC}"
    if [ $FAILED_RUNS -gt 0 ]; then
        echo -e "Failed:           ${RED}${FAILED_RUNS}${NC}"
        echo -e "Failed methods:   ${RED}${FAILED_METHODS}${NC}"
    else
        echo -e "Failed:           ${GREEN}0${NC}"
    fi
    echo ""
else
    # In aggregate-only mode, set defaults for summary
    TOTAL_METHODS=0
    FAILED_RUNS=0
    FAILED_METHODS=""
fi

# Aggregate results
# In aggregate-only mode, always try to aggregate
# Otherwise, only aggregate if there were successful runs
if [ "$AGGREGATE_ONLY" = true ] || [ $SUCCESSFUL_RUNS -gt 0 ]; then
    print_header "AGGREGATING RESULTS"
    print_info "Running aggregation with detailed breakdown..."

    if $PYTHON $BENCHMARK_SCRIPT --aggregate --output-dir "$OUTPUT_DIR"; then
        print_success "Results aggregated successfully"

        # Show output files
        echo ""
        print_info "Output files:"
        echo "  - Aggregated JSON: benchmark.json"
        echo "  - Individual results: $OUTPUT_DIR/<config>/result_*.json"
        if [ "$AGGREGATE_ONLY" = false ]; then
            echo "  - Logs: $OUTPUT_DIR/<config>/log_*.txt"
        fi
    else
        print_error "Aggregation failed"
        exit 1
    fi
else
    print_error "No successful runs to aggregate"
    exit 1
fi

print_header "ALL DONE!"
exit 0
