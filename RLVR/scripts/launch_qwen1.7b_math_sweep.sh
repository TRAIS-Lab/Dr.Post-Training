#!/bin/bash
# Launch script for running Qwen 1.7B GRPO on MATH with multiple seeds and settings
# Settings: Baseline, Streaming (default), GREATS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/run_qwen1.7b_math_grpo.sh"

# Seeds to run
SEEDS=(2 22 42 62 82)

# Submit jobs for each setting and seed
echo "Submitting jobs for 3 settings × 5 seeds = 15 total jobs"
echo "============================================================"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "=== Seed $SEED ==="

    # 1. Baseline (selection disabled)
    # echo "Submitting: Baseline (seed=$SEED)"
    # sbatch --job-name="RLVR-MATH-baseline-s${SEED}" \
    #     --export=ALL,SEED=$SEED,SELECTION_ENABLED=False \
    #     "$MAIN_SCRIPT"

    # 2. Streaming (default selection method)
    echo "Submitting: Streaming (seed=$SEED)"
    sbatch --job-name="RLVR-MATH-streaming-s${SEED}" \
        --export=ALL,SEED=$SEED,SELECTION_ENABLED=True,SELECTION_METHOD=Streaming \
        "$MAIN_SCRIPT"

    # # 3. GREATS selection
    # echo "Submitting: GREATS (seed=$SEED)"
    # sbatch --job-name="RLVR-MATH-greats-s${SEED}" \
    #     --export=ALL,SEED=$SEED,SELECTION_ENABLED=True,SELECTION_METHOD=GREATS \
    #     "$MAIN_SCRIPT"
done

echo ""
echo "============================================================"
echo "All 15 jobs submitted. Use 'squeue -u \$USER' to monitor."
