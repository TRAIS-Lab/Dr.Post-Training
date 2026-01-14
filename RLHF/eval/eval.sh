#!/bin/bash

#SBATCH --job-name=RLHF-Eval
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfwm-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Gradient-Streaming/RLHF/log/%x-%j.log

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
MODELS_DIR="/scratch/pbb/Project/Gradient-Streaming/RLHF"
TASK=""
N_SAMPLES=400
BATCH_SIZE=16
MAX_NEW_TOKENS=30
SEED=42
CLASSIFIER="independent"  # Use independent classifier by default for unbiased evaluation
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --n_samples)
            N_SAMPLES="$2"
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
        --classifier)
            CLASSIFIER="$2"
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
            echo "  --models_dir DIR       Models directory (default: /scratch/pbb/Project/Gradient-Streaming/RLHF)"
            echo "  --task NAME            Filter by task (e.g., toxicity)"
            echo "  --n_samples N          Number of test samples (default: 400, -1 for all)"
            echo "  --batch_size N         Batch size for generation (default: 16)"
            echo "  --max_new_tokens N     Max tokens to generate (default: 30)"
            echo "  --seed N               Random seed for reproducibility (default: 42)"
            echo "  --classifier TYPE      Toxicity classifier: independent (default) or reward"
            echo "                         'independent' uses DaNLP/da-electra-hatespeech-detection"
            echo "                         'reward' uses facebook/roberta-hate-speech-dynabench-r4-target"
            echo "  --dry-run              Print command without executing"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================================"
echo "  RLHF Toxicity Evaluation"
echo "========================================================"
echo "Models dir:      $MODELS_DIR"
echo "Task:            ${TASK:-all}"
echo ""
echo "Generation:"
echo "  N samples:     $N_SAMPLES (-1 = all)"
echo "  Batch size:    $BATCH_SIZE"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Seed:          $SEED"
echo ""
echo "Evaluation:"
echo "  Classifier:    $CLASSIFIER"
echo "========================================================"

# Build command
CMD="python -m RLHF.eval.eval"
CMD="$CMD --models_dir $MODELS_DIR"
CMD="$CMD --n_samples $N_SAMPLES"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"
CMD="$CMD --seed $SEED"
CMD="$CMD --classifier $CLASSIFIER"

if [[ -n "$TASK" ]]; then
    CMD="$CMD --task $TASK"
fi

if [[ "$DRY_RUN" == true ]]; then
    echo "Dry run: $CMD"
else
    eval "$CMD"
fi
