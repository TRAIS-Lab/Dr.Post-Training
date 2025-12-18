#!/bin/bash
#
# Launch script for RLHF experiments with layer-wise selection
#
# Usage: bash experiment_rlhf/launch_experiments.sh [options]
#

cd $HOME/Project/Efficient-Fine-Tuning

set -e

# ========================================
# Default Values
# ========================================
MODEL="llama3-1b"
STEPS="1000"
SEED="42"
DRY_RUN=false
USE_SBATCH=false

# Training hyperparameters (consistent naming with SFT)
PER_DEVICE_TRAIN_BATCH_SIZE="32"
MINI_BATCH_SIZE="4"
VAL_SIZE="256"
VAL_BATCH_SIZE=""  # Validation batch size for selection (defaults to VAL_SIZE if empty)
EPOCHS="1"
LR="2e-6"            # Learning rate for full fine-tuning
LR_LORA="2e-5"    # Learning rate for LoRA (typically higher)

# Selection
SELECTION_MODE="per_layer"  # global or per_layer
VAL_LOSS_TYPE="logprob"  # logprob, reward_weighted, advantage_weighted

# Logging
LOG_INTERVAL=""  # Log every N steps (defaults to steps // 10 if empty)

# Compression
COMPRESSION_GRASS="GraSS"
COMPRESSION_LOGRA="LoGra"

# ========================================
# Parse Arguments
# ========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --per_device_train_batch_size|--batch_size)
            PER_DEVICE_TRAIN_BATCH_SIZE="$2"
            shift 2
            ;;
        --mini_batch_size)
            MINI_BATCH_SIZE="$2"
            shift 2
            ;;
        --val_size)
            VAL_SIZE="$2"
            shift 2
            ;;
        --val_batch_size)
            VAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --selection_mode)
            SELECTION_MODE="$2"
            shift 2
            ;;
        --val_loss_type)
            VAL_LOSS_TYPE="$2"
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
        --log_interval)
            LOG_INTERVAL="$2"
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
            echo "Launch RLHF experiments with layer-wise selection"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --model <model>                    Model name (default: llama3-1b)"
            echo "  --steps <n>                        Training steps (default: 1000)"
            echo "  --seed <seed>                      Random seed (default: 42)"
            echo "  --per_device_train_batch_size <n>  Training batch size (default: 32)"
            echo "  --mini_batch_size <size>           PPO mini-batch size (default: 4)"
            echo "  --val_size <size>                  Total validation pool size (default: 256)"
            echo "  --val_batch_size <size>            Validation batch size for selection (default: val_size)"
            echo "  --epochs <n>                       Epochs per PPO step (default: 1)"
            echo "  --selection_mode <mode>            Selection mode: global, per_layer (default: per_layer)"
            echo "  --val_loss_type <type>             Validation loss: logprob, reward_weighted (default: logprob)"
            echo "  --lr <lr>                          Learning rate for full fine-tuning (default: 5e-6)"
            echo "  --lr_lora <lr>                     Learning rate for LoRA (default: 1.41e-5)"
            echo "  --log_interval <n>                 Log every N steps (default: steps // 10)"
            echo "  --dry-run                          Print commands without executing"
            echo "  --sbatch                           Submit to SLURM instead of running locally"
            echo ""
            echo "Experimental Configurations:"
            echo "  1. Baseline PPO (LoRA)"
            echo "  2. Baseline PPO (Full Fine-tuning)"
            echo "  3. GREATS Layer-wise Selection (LoRA) + GraSS"
            echo "  4. GREATS Layer-wise Selection (Full) + GraSS"
            echo "  5. GREATS Layer-wise Selection (LoRA) + LoGra"
            echo "  6. MeSO + GREATS (Full) + GraSS"
            echo "  7. MeSO + GREATS (Full) + LoGra"
            echo "  8-10. Second-Order Selection variants"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ========================================
# Build Launcher
# ========================================
if [ "$USE_SBATCH" = true ]; then
    LAUNCHER="sbatch"
else
    LAUNCHER="bash"
fi

SCRIPT="experiment_rlhf/train.sh"

# Base args shared by all
BASE_ARGS="--model $MODEL --steps $STEPS --seed $SEED --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE --mini_batch_size $MINI_BATCH_SIZE --val_size $VAL_SIZE --epochs $EPOCHS --selection_mode $SELECTION_MODE --val_loss_type $VAL_LOSS_TYPE"

# Add val_batch_size if specified
if [[ -n "$VAL_BATCH_SIZE" ]]; then
    BASE_ARGS="$BASE_ARGS --val_batch_size $VAL_BATCH_SIZE"
fi

# Add log_interval if specified
if [[ -n "$LOG_INTERVAL" ]]; then
    BASE_ARGS="$BASE_ARGS --log_interval $LOG_INTERVAL"
fi

# LoRA args
LORA_ARGS="$BASE_ARGS --lora --lr $LR_LORA"

# Full fine-tuning args
FULL_ARGS="$BASE_ARGS --no_lora --lr $LR"

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
echo "  RLHF Experiments with Layer-wise Selection"
echo "========================================================"
echo "Model: $MODEL"
echo "Steps: $STEPS"
echo "Seed: $SEED"
echo ""
echo "Training Settings:"
echo "  per_device_train_batch_size: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  Mini-batch size: $MINI_BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Val size (total pool): $VAL_SIZE"
echo "  Val batch size: ${VAL_BATCH_SIZE:-$VAL_SIZE (same as val_size)}"
echo "  Selection mode: $SELECTION_MODE"
echo "  Val loss type: $VAL_LOSS_TYPE"
echo "  Learning rate (Full): $LR"
echo "  Learning rate (LoRA): $LR_LORA"
echo "  Log interval: ${LOG_INTERVAL:-$(($STEPS / 10)) (steps // 10)}"
echo ""
echo "Launcher: $LAUNCHER"
echo "========================================================"

# ============================================
# 1. Baseline Experiments
# ============================================

# 1a. Baseline PPO with LoRA
run_cmd "1a. Baseline PPO - LoRA" \
    "$LORA_ARGS --selection_method NA"

# 1b. Baseline PPO with Full Fine-tuning
run_cmd "1b. Baseline PPO - Full Fine-tuning" \
    "$FULL_ARGS --selection_method NA"

# ============================================
# 2. GREATS Layer-wise Selection with GraSS
# ============================================

# 2a. GREATS + LoRA + GraSS
run_cmd "2a. GREATS Layer-wise Selection - LoRA (GraSS)" \
    "$LORA_ARGS --selection_method GREATS --compression $COMPRESSION_GRASS"

# 2b. GREATS + Full + GraSS
run_cmd "2b. GREATS Layer-wise Selection - Full Fine-tuning (GraSS)" \
    "$FULL_ARGS --selection_method GREATS --compression $COMPRESSION_GRASS"

# ============================================
# 3. GREATS Layer-wise Selection with LoGra
# ============================================

# 3a. GREATS + LoRA + LoGra
run_cmd "3a. GREATS Layer-wise Selection - LoRA (LoGra)" \
    "$LORA_ARGS --selection_method GREATS --compression $COMPRESSION_LOGRA"

# 3b. GREATS + Full + LoGra
run_cmd "3b. GREATS Layer-wise Selection - Full Fine-tuning (LoGra)" \
    "$FULL_ARGS --selection_method GREATS --compression $COMPRESSION_LOGRA"

# ============================================
# 4. MeSO + GREATS (Full Fine-tuning only)
# ============================================

# 4a. MeSO + GREATS + Full + GraSS
run_cmd "4a. MeSO + GREATS - Full Fine-tuning (GraSS)" \
    "$FULL_ARGS --selection_method GREATS --compression $COMPRESSION_GRASS --MeSO"

# 4b. MeSO + GREATS + Full + LoGra
run_cmd "4b. MeSO + GREATS - Full Fine-tuning (LoGra)" \
    "$FULL_ARGS --selection_method GREATS --compression $COMPRESSION_LOGRA --MeSO"

# ============================================
# 5. Second-Order Selection Experiments
# ============================================

# 5a. GREATS + Second-Order + LoRA + GraSS
run_cmd "5a. GREATS + Second-Order - LoRA (GraSS)" \
    "$LORA_ARGS --selection_method GREATS --compression $COMPRESSION_GRASS --use_second_order"

# 5b. MeSO + GREATS + Second-Order + Full + GraSS
run_cmd "5b. MeSO + GREATS + Second-Order - Full Fine-tuning (GraSS)" \
    "$FULL_ARGS --selection_method GREATS --compression $COMPRESSION_GRASS --MeSO --use_second_order"

echo ""
echo "========================================================"
echo "  All experiments launched!"
echo "========================================================"
echo ""
echo "Total configurations: 10"
echo "  - 2 Baseline PPO (LoRA + Full)"
echo "  - 4 GREATS Layer-wise Selection (LoRA/Full × GraSS/LoGra)"
echo "  - 2 MeSO + GREATS (Full × GraSS/LoGra)"
echo "  - 2 Second-Order Selection variants"
echo ""
