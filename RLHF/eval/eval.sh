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

SCRATCH_DIR=/work/hdd/bfwm/phu1/Project
CODE_DIR=/u/phu1/Project

cd $CODE_DIR/Gradient-Streaming

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$CODE_DIR/Gradient-Streaming:$PYTHONPATH"

set -e

# Default values
models_dir="$SCRATCH_DIR/Gradient-Streaming/RLHF"
task=""
n_samples=400
batch_size=16
max_new_tokens=30
seed=42
classifier="independent"  # Use independent classifier by default for unbiased evaluation
dry_run=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)
            models_dir="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --n_samples)
            n_samples="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --max_new_tokens)
            max_new_tokens="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --classifier)
            classifier="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
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
echo "Models dir:      $models_dir"
echo "Task:            ${task:-all}"
echo ""
echo "Generation:"
echo "  N samples:     $n_samples (-1 = all)"
echo "  Batch size:    $batch_size"
echo "  Max new tokens: $max_new_tokens"
echo "  Seed:          $seed"
echo ""
echo "Evaluation:"
echo "  Classifier:    $classifier"
echo "========================================================"

# Build command
cmd="python -m RLHF.eval.eval"
cmd="$cmd --models_dir $models_dir"
cmd="$cmd --n_samples $n_samples"
cmd="$cmd --batch_size $batch_size"
cmd="$cmd --max_new_tokens $max_new_tokens"
cmd="$cmd --seed $seed"
cmd="$cmd --classifier $classifier"

if [[ -n "$task" ]]; then
    cmd="$cmd --task $task"
fi

if [[ "$dry_run" == true ]]; then
    echo "Dry run: $cmd"
else
    eval "$cmd"
fi
