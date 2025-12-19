#!/bin/bash
#
# Launch script for running all RLHF experimental configurations
# Usage: bash RLHF/train/launch_experiments.sh [options]
#
# Naming convention: {task}-{method}-{compression}-{model}-{training_type}
#   method: NA, Streaming (per-layer), GREATS (global)
#   compression: NA (standard optimizer), GraSS, LoGra (MeSO optimizer)
#   training_type: full, lora
#

cd $HOME/Project/Gradient-Streaming

set -e

# Default values
TASK="toxicity"
MODEL="EleutherAI/gpt-neo-125m"
DRY_RUN=false
USE_SBATCH=false
SEED="42"

# Training hyperparameters
BATCH_SIZE="64"
VAL_BATCH_SIZE=""
LR="1e-5"            # Learning rate for full fine-tuning
LR_LORA="1e-5"       # Learning rate for LoRA (same for RLHF typically)
N_VAL="16"
MAX_STEPS="200"
SELECTION_FRAC="0.5"

# PPO settings
PPO_EPOCHS="4"
KL_COEF="0.04"

# Compression settings
SPARSIFICATION_GRASS="Rademacher-64*64"
PROJECTION_LOGRA="Gaussian-256"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
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
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --selection_frac)
            SELECTION_FRAC="$2"
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
            echo "Launch all RLHF experimental configurations"
            echo ""
            echo "Naming convention: {task}-{method}-{compression}-{model}-{training_type}"
            echo "  method: NA (baseline), Streaming (per-layer), GREATS (global)"
            echo "  compression: NA (standard optimizer), GraSS, LoGra (MeSO optimizer)"
            echo "  training_type: full, lora"
            echo ""
            echo "Options:"
            echo "  --task <task>           Task: toxicity (default: toxicity)"
            echo "  --model <model>         Model name (default: EleutherAI/gpt-neo-125m)"
            echo "  --seed <seed>           Random seed (default: 42)"
            echo ""
            echo "Training Hyperparameters:"
            echo "  --batch_size <size>     Training batch size (default: 64)"
            echo "  --val_batch_size <size> Validation batch size (default: same as n_val)"
            echo "  --lr <lr>               Learning rate for full fine-tuning (default: 1e-5)"
            echo "  --lr_lora <lr>          Learning rate for LoRA fine-tuning (default: 1e-5)"
            echo "  --n_val <n>             Number of validation examples (default: 16)"
            echo "  --max_steps <steps>     Maximum training steps (default: 200)"
            echo "  --selection_frac <frac> Fraction of samples to select (default: 0.5)"
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

SCRIPT="RLHF/train/train.sh"

# Build base args (shared by all)
BASE_ARGS="--task $TASK --model $MODEL --seed $SEED --batch_size $BATCH_SIZE --n_val $N_VAL --max_steps $MAX_STEPS --selection_frac $SELECTION_FRAC --ppo_epochs $PPO_EPOCHS --kl_coef $KL_COEF"

# Add val_batch_size if specified
if [[ -n "$VAL_BATCH_SIZE" ]]; then
    BASE_ARGS="$BASE_ARGS --val_batch_size $VAL_BATCH_SIZE"
fi

# Build args for full fine-tuning
FULL_ARGS="$BASE_ARGS --lr $LR --no_lora"

# Build args for LoRA fine-tuning
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
echo "  Launching RLHF Experiments"
echo "========================================================"
echo "Task: $TASK"
echo "Model: $MODEL"
echo "Seed: $SEED"
echo ""
echo "Training Settings:"
echo "  Batch size: $BATCH_SIZE"
echo "  Val batch size: ${VAL_BATCH_SIZE:-same as n_val}"
echo "  Learning rate (full): $LR"
echo "  Learning rate (LoRA): $LR_LORA"
echo "  N_val: $N_VAL"
echo "  Max steps: $MAX_STEPS"
echo "  Selection fraction: $SELECTION_FRAC"
echo ""
echo "PPO Settings:"
echo "  PPO epochs: $PPO_EPOCHS"
echo "  KL coefficient: $KL_COEF"
echo ""
echo "Launcher: $LAUNCHER"
echo "========================================================"

# ============================================
# 1. Baseline Experiments (NA-NA)
#    No selection, no compression
# ============================================

run_cmd "1a. NA-NA-lora (Baseline PPO with LoRA)" \
    "$LORA_ARGS --method NA"

run_cmd "1b. NA-NA-full (Baseline PPO Full Fine-tuning)" \
    "$FULL_ARGS --method NA"

# ============================================
# 2. Streaming without compression (Streaming-NA)
#    Per-layer selection, full gradients
# ============================================

run_cmd "2a. Streaming-NA-lora (Per-layer selection, full gradients, LoRA)" \
    "$LORA_ARGS --method Streaming"

run_cmd "2b. Streaming-NA-full (Per-layer selection, full gradients)" \
    "$FULL_ARGS --method Streaming"

# ============================================
# 3. GREATS without compression (GREATS-NA)
#    Global selection, full gradients
# ============================================

run_cmd "3a. GREATS-NA-lora (Global selection, full gradients, LoRA)" \
    "$LORA_ARGS --method GREATS"

run_cmd "3b. GREATS-NA-full (Global selection, full gradients)" \
    "$FULL_ARGS --method GREATS"

# ============================================
# 4. MeSO only (NA-{compression})
#    No selection, compressed gradients
# ============================================

run_cmd "4a. NA-GraSS-lora (MeSO only, GraSS, LoRA)" \
    "$LORA_ARGS --method NA --sparsification $SPARSIFICATION_GRASS"

run_cmd "4b. NA-LoGra-lora (MeSO only, LoGra, LoRA)" \
    "$LORA_ARGS --method NA --projection $PROJECTION_LOGRA"

# ============================================
# 5. Streaming with compression (Streaming-{compression})
#    Per-layer selection, MeSO
# ============================================

run_cmd "5a. Streaming-GraSS-lora (Per-layer selection, MeSO GraSS, LoRA)" \
    "$LORA_ARGS --method Streaming --sparsification $SPARSIFICATION_GRASS"

run_cmd "5b. Streaming-LoGra-lora (Per-layer selection, MeSO LoGra, LoRA)" \
    "$LORA_ARGS --method Streaming --projection $PROJECTION_LOGRA"

# ============================================
# 6. GREATS with compression (GREATS-{compression})
#    Global selection, MeSO
# ============================================

run_cmd "6a. GREATS-GraSS-lora (Global selection, MeSO GraSS, LoRA)" \
    "$LORA_ARGS --method GREATS --sparsification $SPARSIFICATION_GRASS"

run_cmd "6b. GREATS-LoGra-lora (Global selection, MeSO LoGra, LoRA)" \
    "$LORA_ARGS --method GREATS --projection $PROJECTION_LOGRA"

# ============================================
# 7. Second-order variants (optional)
# ============================================

run_cmd "7a. Streaming-GraSS-2nd-lora (Per-layer, MeSO, second-order, LoRA)" \
    "$LORA_ARGS --method Streaming --sparsification $SPARSIFICATION_GRASS --use_second_order"

run_cmd "7b. GREATS-GraSS-2nd-lora (Global, MeSO, second-order, LoRA)" \
    "$LORA_ARGS --method GREATS --sparsification $SPARSIFICATION_GRASS --use_second_order"

echo ""
echo "========================================================"
echo "  All experiments launched!"
echo "========================================================"
echo ""
echo "Naming convention: {task}-{method}-{compression}-{model}-{training_type}"
echo "  method: NA (baseline), Streaming (per-layer), GREATS (global)"
echo "  compression: NA (standard optimizer), GraSS, LoGra (MeSO optimizer)"
echo "  training_type: full, lora"
echo ""
echo "Total configurations: 14"
echo "  - 2 Baseline (NA-NA) x (full, lora)"
echo "  - 4 Without compression (Streaming-NA, GREATS-NA) x (full, lora)"
echo "  - 2 MeSO only (NA-GraSS, NA-LoGra) x lora"
echo "  - 4 With compression (Streaming, GREATS) x (GraSS, LoGra) x lora"
echo "  - 2 Second-order variants (Streaming-GraSS-2nd, GREATS-GraSS-2nd) x lora"
echo ""
