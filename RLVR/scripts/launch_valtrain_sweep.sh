#!/bin/bash
# Launch script for val_from_train ablation experiments
# Submits: Baseline, Streaming, GREATS across 5 seeds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_SCRIPT="$SCRIPT_DIR/run_qwen1.7b_math_grpo.sh"

SEEDS=(82)

echo "Submitting 15 jobs (3 methods × 5 seeds)"
echo "========================================="

for SEED in "${SEEDS[@]}"; do
    # Baseline
    sbatch --job-name="baseline-s${SEED}" \
        --export=ALL,SEED=$SEED,SELECTION_ENABLED=False \
        "$MAIN_SCRIPT"

    # Streaming with val_from_train
    sbatch --job-name="streaming-s${SEED}-valtrain" \
        --export=ALL,SEED=$SEED,SELECTION_ENABLED=True,SELECTION_METHOD=Streaming \
        "$MAIN_SCRIPT"

    # GREATS with val_from_train
    sbatch --job-name="greats-s${SEED}-valtrain" \
        --export=ALL,SEED=$SEED,SELECTION_ENABLED=True,SELECTION_METHOD=GREATS \
        "$MAIN_SCRIPT"
done

echo ""
echo "Done. Check with: squeue -u \$USER -o '%.10i %.25j %.8T' --sort=j"
