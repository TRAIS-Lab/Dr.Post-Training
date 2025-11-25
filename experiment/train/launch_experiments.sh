#!/bin/bash
#
# Launch script for running all experimental configurations
# Usage: bash experiment/train/launch_experiments.sh [options]
#
# Options:
#   --task <task>       Evaluation task (default: mmlu)
#   --train <dataset>   Training dataset (default: task-based)
#   --subject <subject> Subject for MMLU/BBH (default: sociology)
#   --model <model>     Model name (default: llama3-1b)
#   --dry-run           Print commands without executing
#   --sbatch            Use sbatch instead of bash
#

cd $HOME/Project/Efficient-Fine-Tuning

set -e

# Default values
TASK="mmlu"
TRAIN=""
SUBJECT="sociology"
MODEL="llama3-1b"
DRY_RUN=false
USE_SBATCH=false
PERCENTAGE="0.05"
SEED="42"

# Training hyperparameters
BATCH_SIZE="4"
VAL_BATCH_SIZE=""
LR="5e-05"           # Learning rate for full fine-tuning
LR_LORA="2e-04"      # Learning rate for LoRA (typically higher)
N_VAL="5"

# Compression settings
COMPRESSION_GRASS="GraSS"
COMPRESSION_LOGRA="LoGra"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --train)
            TRAIN="$2"
            shift 2
            ;;
        --subject)
            SUBJECT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --percentage)
            PERCENTAGE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --val_batch_size)
            VAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --lr_lora)
            LR_LORA="$2"
            shift 2
            ;;
        --n_val)
            N_VAL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --sbatch)
            USE_SBATCH=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Launch all experimental configurations for a given task"
            echo ""
            echo "Options:"
            echo "  --task <task>           Evaluation task: mmlu, bbh, tydiqa, samsum, gsm8k, math500 (default: mmlu)"
            echo "  --train <dataset>       Training dataset: alpaca, dolly, flan_v2, cot, oasst1, gsm8k, vicuna, wizardlm, openhermes, tulu3 (default: task-based)"
            echo "  --subject <subject>     Subject for MMLU/BBH (default: sociology)"
            echo "  --model <model>         Model name (default: llama3-1b)"
            echo "  --percentage <pct>      Data sampling percentage (default: 0.05)"
            echo "  --seed <seed>           Random seed (default: 42)"
            echo ""
            echo "Training Hyperparameters:"
            echo "  --batch_size <size>     Training batch size (default: 4)"
            echo "  --val_batch_size <size> Validation batch size for data selection (default: same as batch_size)"
            echo "  --lr <lr>               Learning rate for full fine-tuning (default: 5e-05)"
            echo "  --lr_lora <lr>          Learning rate for LoRA fine-tuning (default: 2e-04)"
            echo "  --n_val <n>             Number of validation examples (default: 5)"
            echo ""
            echo "Execution:"
            echo "  --dry-run               Print commands without executing"
            echo "  --sbatch                Use sbatch instead of bash"
            echo ""
            echo "Experimental Configurations:"
            echo "  1. Baseline Full Fine-tuning"
            echo "  2. Baseline LoRA Fine-tuning"
            echo "  3. GREATS Data Selection (Full + LoRA) with GraSS"
            echo "  4. GREATS Data Selection (Full + LoRA) with LoGra"
            echo "  5. MeSO Compressed Optimizer (Full) with GraSS/LoGra"
            echo "  6. MeSO + GREATS (Full) with GraSS/LoGra"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Build base command
if [ "$USE_SBATCH" = true ]; then
    LAUNCHER="sbatch"
else
    LAUNCHER="bash"
fi

SCRIPT="experiment/train/train.sh"

# Build base args (shared by all)
BASE_ARGS="--task $TASK --model $MODEL --percentage $PERCENTAGE --seed $SEED --batch_size $BATCH_SIZE --n_val $N_VAL"

# Add val_batch_size if specified
if [[ -n "$VAL_BATCH_SIZE" ]]; then
    BASE_ARGS="$BASE_ARGS --val_batch_size $VAL_BATCH_SIZE"
fi

# Add subject for MMLU/BBH
if [[ "$TASK" == "mmlu" ]] || [[ "$TASK" == "bbh" ]]; then
    BASE_ARGS="$BASE_ARGS --subject $SUBJECT"
fi

# Add training dataset if specified
if [[ -n "$TRAIN" ]]; then
    BASE_ARGS="$BASE_ARGS --train $TRAIN"
fi

# Build args for full fine-tuning (uses LR)
FULL_ARGS="$BASE_ARGS --lr $LR"

# Build args for LoRA fine-tuning (uses LR_LORA)
LORA_ARGS="$BASE_ARGS --lr $LR_LORA --lora"

# Function to run or print command
run_cmd() {
    local desc="$1"
    local args="$2"

    echo ""
    echo "=============================================="
    echo "  $desc"
    echo "=============================================="
    echo "Command: $LAUNCHER $SCRIPT $args"
    echo ""

    if [ "$DRY_RUN" = false ]; then
        $LAUNCHER $SCRIPT $args
        echo "Submitted."
    else
        echo "[DRY-RUN] Would execute above command"
    fi
}

echo ""
echo "========================================================"
echo "  Launching Experiments"
echo "========================================================"
echo "Task: $TASK"
echo "Training data: ${TRAIN:-default}"
echo "Model: $MODEL"
echo "Subject: $SUBJECT"
echo "Percentage: $PERCENTAGE"
echo "Seed: $SEED"
echo ""
echo "Training Settings:"
echo "  Batch size: $BATCH_SIZE"
echo "  Val batch size: ${VAL_BATCH_SIZE:-same as batch_size}"
echo "  Learning rate (full): $LR"
echo "  Learning rate (LoRA): $LR_LORA"
echo "  N_val: $N_VAL"
echo ""
echo "Launcher: $LAUNCHER"
echo "========================================================"

# ============================================
# 1. Baseline Experiments (No compression)
# ============================================

# 1a. Full Fine-tuning Baseline
run_cmd "1a. Baseline - Full Fine-tuning" \
    "$FULL_ARGS"

# 1b. LoRA Fine-tuning Baseline
run_cmd "1b. Baseline - LoRA Fine-tuning" \
    "$LORA_ARGS"

# ============================================
# 2. GREATS Data Selection with GraSS
# ============================================

# 2a. GREATS + Full Fine-tuning + GraSS
run_cmd "2a. GREATS Data Selection - Full Fine-tuning (GraSS)" \
    "$FULL_ARGS --data_selection GREATS --compression $COMPRESSION_GRASS"

# 2b. GREATS + LoRA + GraSS
run_cmd "2b. GREATS Data Selection - LoRA Fine-tuning (GraSS)" \
    "$LORA_ARGS --data_selection GREATS --compression $COMPRESSION_GRASS"

# ============================================
# 3. GREATS Data Selection with LoGra
# ============================================

# 3a. GREATS + Full Fine-tuning + LoGra
run_cmd "3a. GREATS Data Selection - Full Fine-tuning (LoGra)" \
    "$FULL_ARGS --data_selection GREATS --compression $COMPRESSION_LOGRA"

# 3b. GREATS + LoRA + LoGra
run_cmd "3b. GREATS Data Selection - LoRA Fine-tuning (LoGra)" \
    "$LORA_ARGS --data_selection GREATS --compression $COMPRESSION_LOGRA"

# ============================================
# 4. MeSO Compressed Optimizer
# ============================================

# 4a. MeSO + Full Fine-tuning + GraSS (no data selection)
run_cmd "4a. MeSO Compressed Optimizer - Full Fine-tuning (GraSS)" \
    "$FULL_ARGS --MeSO --compression $COMPRESSION_GRASS"

# 4b. MeSO + Full Fine-tuning + LoGra (no data selection)
run_cmd "4b. MeSO Compressed Optimizer - Full Fine-tuning (LoGra)" \
    "$FULL_ARGS --MeSO --compression $COMPRESSION_LOGRA"

# ============================================
# 5. MeSO + GREATS (Both optimizations)
# ============================================

# 5a. MeSO + GREATS + Full Fine-tuning + GraSS
run_cmd "5a. MeSO + GREATS - Full Fine-tuning (GraSS)" \
    "$FULL_ARGS --MeSO --data_selection GREATS --compression $COMPRESSION_GRASS"

# 5b. MeSO + GREATS + Full Fine-tuning + LoGra
run_cmd "5b. MeSO + GREATS - Full Fine-tuning (LoGra)" \
    "$FULL_ARGS --MeSO --data_selection GREATS --compression $COMPRESSION_LOGRA"

echo ""
echo "========================================================"
echo "  All experiments launched!"
echo "========================================================"
echo ""
echo "Total configurations: 10"
echo "  - 2 Baseline (Full + LoRA)"
echo "  - 4 GREATS Data Selection (Full/LoRA × GraSS/LoGra)"
echo "  - 2 MeSO Optimizer (Full × GraSS/LoGra)"
echo "  - 2 MeSO + GREATS (Full × GraSS/LoGra)"
echo ""
