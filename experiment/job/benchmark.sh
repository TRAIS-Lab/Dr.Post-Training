#!/bin/bash
#
# Benchmark Runner Script
#
# Runs benchmarks sequentially with clean state between each run.
# Automatically aggregates results at the end with detailed breakdown.
#
# Usage:
#     # Run all methods
#     bash run_benchmark.sh
#
#     # Run specific methods (by index)
#     bash run_benchmark.sh 0 7 8
#
#     # Run with custom output directory
#     OUTPUT_DIR=my_results bash run_benchmark.sh
#

cd ~/Project/Efficient-Fine-Tuning/experiment

set -e  # Exit on error

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results}"
PYTHON="${PYTHON:-python}"
BENCHMARK_SCRIPT="benchmark/benchmark.py"

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
    print_error "benchmark.py not found in current directory"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_success "Output directory: $OUTPUT_DIR"

# Get list of available methods
print_info "Fetching available methods..."
METHODS_OUTPUT=$($PYTHON $BENCHMARK_SCRIPT --list 2>&1)
echo "$METHODS_OUTPUT"

# Determine which methods to run
if [ $# -eq 0 ]; then
    print_info "No method indices specified, will run ALL methods"
    # Extract method indices from --list output (lines like " 0: Method Name" or "10: Method Name")
    METHODS_TO_RUN=$(echo "$METHODS_OUTPUT" | grep -E "^ *[0-9]+:" | sed 's/^[[:space:]]*//' | cut -d: -f1)
else
    METHODS_TO_RUN="$@"
    print_info "Running specified methods: $METHODS_TO_RUN"
fi

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
    # Use zero-padded format to match benchmark.py output (result_00.json, result_01.json, etc.)
    OUTPUT_FILE="$OUTPUT_DIR/result_$(printf '%02d' ${METHOD_IDX}).json"
    LOG_FILE="$OUTPUT_DIR/log_$(printf '%02d' ${METHOD_IDX}).txt"

    print_info "Running benchmark (output: $LOG_FILE)..."

    if $PYTHON $BENCHMARK_SCRIPT --method "$METHOD_IDX" --output-dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"; then
        if [ -f "$OUTPUT_FILE" ]; then
            print_success "Method $METHOD_IDX completed successfully"
            SUCCESSFUL_RUNS=$((SUCCESSFUL_RUNS + 1))
        else
            print_error "Method $METHOD_IDX completed but no output file generated"
            FAILED_RUNS=$((FAILED_RUNS + 1))
            FAILED_METHODS="$FAILED_METHODS $METHOD_IDX"
        fi
    else
        print_error "Method $METHOD_IDX failed"
        FAILED_RUNS=$((FAILED_RUNS + 1))
        FAILED_METHODS="$FAILED_METHODS $METHOD_IDX"
    fi

    # Wait a bit between runs
    if [ $CURRENT -lt $TOTAL_METHODS ]; then
        print_info "Waiting 2 seconds before next run..."
        sleep 2
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

# Aggregate results
if [ $SUCCESSFUL_RUNS -gt 0 ]; then
    print_header "AGGREGATING RESULTS"
    print_info "Running aggregation with detailed breakdown..."

    if $PYTHON $BENCHMARK_SCRIPT --aggregate --output-dir "$OUTPUT_DIR"; then
        print_success "Results aggregated successfully"

        # Show output files
        echo ""
        print_info "Output files:"
        echo "  - Aggregated JSON: benchmark.json"
        echo "  - Individual results: $OUTPUT_DIR/result_*.json"
        echo "  - Logs: $OUTPUT_DIR/log_*.txt"
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
