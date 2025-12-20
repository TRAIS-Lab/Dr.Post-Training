#!/bin/bash
#
# Launch script for running all experimental configurations
# Usage: bash SFT/train/launch_experiments.sh [options]
#
# Naming convention: {selection}-{compression}-{training_type}
#   selection: NA, Streaming (per-layer), GREATS (global)
#   compression: NA (standard optimizer), GraSS, LoGra (MeSO optimizer)
#   training_type: full, lora
#

cd $HOME/Project/Gradient-Streaming

set -e

# Default values
TASK="mmlu"
TRAIN="less"
SUBJECT="sociology"
MODEL="llama3-1b"
DRY_RUN=false
USE_SBATCH=false
PERCENTAGE="0.05"
SEED="42"

# Training hyperparameters
BATCH_SIZE="8"
N_VAL="8"
VAL_BATCH_SIZE="1"
LR="5e-05"           # Learning rate for full fine-tuning
LR_LORA="2e-04"      # Learning rate for LoRA (typically higher)

# Compression settings (using LoGra only)
COMPRESSION="LoGra"

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
            echo "Naming convention: {selection}-{compression}-{training_type}"
            echo "  selection: NA (baseline), Streaming (per-layer), GREATS (global)"
            echo "  compression: NA (standard optimizer), GraSS, LoGra (MeSO optimizer)"
            echo "  training_type: full, lora"
            echo ""
            echo "Options:"
            echo "  --task <task>           Evaluation task: mmlu, bbh, tydiqa, samsum, gsm8k (default: mmlu)"
            echo "  --train <dataset>       Training dataset (default: task-based)"
            echo "  --subject <subject>     Subject for MMLU/BBH (default: sociology)"
            echo "  --model <model>         Model name (default: llama3-1b)"
            echo "  --percentage <pct>      Data sampling percentage (default: 0.05)"
            echo "  --seed <seed>           Random seed (default: 42)"
            echo ""
            echo "Training Hyperparameters:"
            echo "  --batch_size <size>     Training batch size (default: 4)"
            echo "  --val_batch_size <size> Validation batch size for selection (default: same as batch_size)"
            echo "  --lr <lr>               Learning rate for full fine-tuning (default: 5e-05)"
            echo "  --lr_lora <lr>          Learning rate for LoRA fine-tuning (default: 2e-04)"
            echo "  --n_val <n>             Number of validation examples (default: 5)"
            echo ""
            echo "Execution:"
            echo "  --dry-run               Print commands without executing"
            echo "  --sbatch                Use sbatch instead of bash"
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

SCRIPT="SFT/train/train.sh"

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
# 1. Baseline Experiments (NA-NA)
#    No selection, no compression
# ============================================

run_cmd "1a. NA-NA-full (Baseline Full Fine-tuning)" \
    "$FULL_ARGS"

run_cmd "1b. NA-NA-lora (Baseline LoRA Fine-tuning)" \
    "$LORA_ARGS"

# ============================================
# 2. Streaming without compression (Streaming-NA)
#    Per-layer selection, full gradients
# ============================================

run_cmd "2a. Streaming-NA-full (Per-layer selection, full gradients)" \
    "$FULL_ARGS --data_selection Streaming"

run_cmd "2b. Streaming-NA-lora (Per-layer selection, full gradients, LoRA)" \
    "$LORA_ARGS --data_selection Streaming"

# ============================================
# 3. GREATS without compression (GREATS-NA)
#    Global selection, full gradients
# ============================================

run_cmd "3a. GREATS-NA-full (Global selection, full gradients)" \
    "$FULL_ARGS --data_selection GREATS"

run_cmd "3b. GREATS-NA-lora (Global selection, full gradients, LoRA)" \
    "$LORA_ARGS --data_selection GREATS"

# ============================================
# 4. Streaming with compression (Streaming-LoGra)
#    Per-layer selection, MeSO
# ============================================

run_cmd "4. Streaming-LoGra-full (Per-layer selection, MeSO)" \
    "$FULL_ARGS --data_selection Streaming --compression $COMPRESSION"

# ============================================
# 5. GREATS with compression (GREATS-LoGra)
#    Global selection, MeSO
# ============================================

run_cmd "5. GREATS-LoGra-full (Global selection, MeSO)" \
    "$FULL_ARGS --data_selection GREATS --compression $COMPRESSION"

# ============================================
# 6. Second-order variants
# ============================================

run_cmd "6a. Streaming-LoGra-2nd-full (Per-layer, MeSO, second-order)" \
    "$FULL_ARGS --data_selection Streaming --compression $COMPRESSION --use_second_order"

run_cmd "6b. GREATS-LoGra-2nd-full (Global, MeSO, second-order)" \
    "$FULL_ARGS --data_selection GREATS --compression $COMPRESSION --use_second_order"

echo ""
echo "========================================================"
echo "  All experiments launched!"
echo "========================================================"
echo ""
echo "Naming convention: {selection}-{compression}-{training_type}"
echo "  selection: NA (baseline), Streaming (per-layer), GREATS (global)"
echo "  compression: NA (standard optimizer), LoGra (MeSO optimizer)"
echo "  training_type: full, lora"
echo ""
echo "Total configurations: 10"
echo "  - 2 Baseline (NA-NA-full/lora)"
echo "  - 4 Without compression (Streaming-NA, GREATS-NA) x (full, lora)"
echo "  - 2 With compression (Streaming-LoGra, GREATS-LoGra) x full"
echo "  - 2 Second-order (Streaming-LoGra-2nd, GREATS-LoGra-2nd) x full"
echo ""
