#!/bin/bash

#SBATCH --job-name=SFT-Eval
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfwm-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Gradient-Streaming/SFT/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

cd $HOME/Project/Gradient-Streaming

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$HOME/Project/Gradient-Streaming:$PYTHONPATH"

set -e

# Default values
MODELS_DIR="/scratch/pbb/Project/Gradient-Streaming/SFT"
TRAIN=""
TASK=""
SUBJECT=""
N_TEST=-1
BATCH_SIZE=1
MAX_NEW_TOKENS=128
SEED=42
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --train)
            TRAIN="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --subject)
            SUBJECT="$2"
            shift 2
            ;;
        --n_test)
            N_TEST="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --models_dir DIR     Models directory (default: /scratch/pbb/Project/Gradient-Streaming/SFT)"
            echo "  --train NAME         Filter by training dataset (alpaca, less, tulu3, wizardlm)"
            echo "  --task NAME          Override task (samsum, tydiqa, mmlu, bbh, gsm8k, math500)"
            echo "  --subject NAME       MMLU subject or BBH task to evaluate on (default: all)"
            echo "  --n_test N           Number of test examples (-1 for all)"
            echo "  --batch_size N       Batch size for generation (default: 1)"
            echo "  --max_new_tokens N   Max tokens to generate (default: 128)"
            echo "  --seed N             Random seed for reproducibility (default: 42)"
            echo "  --dry-run            Print command without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$SFT_DIR/data"

echo ""
echo "========================================================"
echo "  SFT Evaluation"
echo "========================================================"
echo "Models dir:      $MODELS_DIR"
echo "Data dir:        $DATA_DIR"
echo ""
echo "Filters:"
echo "  Train:         ${TRAIN:-all}"
echo "  Task:          ${TASK:-auto-detect}"
echo "  Subject:       ${SUBJECT:-all}"
echo ""
echo "Generation:"
echo "  Batch size:    $BATCH_SIZE"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  N test:        $N_TEST (-1 = all)"
echo "  Seed:          $SEED"
echo "========================================================"

# Build command
CMD="python -m SFT.eval.eval"
CMD="$CMD --models_dir $MODELS_DIR"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --n_test $N_TEST"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --seed $SEED"

if [[ -n "$TRAIN" ]]; then
    CMD="$CMD --train $TRAIN"
fi

if [[ -n "$TASK" ]]; then
    CMD="$CMD --task $TASK"
fi

if [[ -n "$SUBJECT" ]]; then
    CMD="$CMD --subject $SUBJECT"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run: $CMD"
else
    eval "$CMD"
fi
