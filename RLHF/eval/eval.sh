#!/bin/bash
#
# Evaluation script for RLHF experiments
#
# Evaluates toxicity of trained models using toxic prompts from wiki_toxic dataset.
#
# Usage:
#   bash RLHF/eval/eval.sh                       # Evaluate all models
#   bash RLHF/eval/eval.sh --task toxicity       # Filter by task
#   bash RLHF/eval/eval.sh --batch_size 16       # Set batch size
#   bash RLHF/eval/eval.sh --sbatch              # Submit to SLURM
#

set -e

# Default values
MODELS_DIR="/scratch/pbb/Project/Gradient-Streaming/RLHF"
TASK=""
N_SAMPLES=400
BATCH_SIZE=16
MAX_NEW_TOKENS=30
SEED=42
CLASSIFIER="independent"  # Use independent classifier by default for unbiased evaluation
SBATCH=false
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
        --sbatch)
            SBATCH=true
            shift
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
            echo "  --sbatch               Submit as SLURM job"
            echo "  --dry-run              Print command without executing"
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
RLHF_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$RLHF_DIR")"

echo "======================================"
echo "RLHF Toxicity Evaluation"
echo "======================================"
echo "Models:     $MODELS_DIR"
echo "Task:       ${TASK:-all}"
echo "Samples:    $N_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Seed:       $SEED"
echo "Classifier: $CLASSIFIER"
echo "======================================"

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

if [[ "$SBATCH" == true ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="rlhf_eval_${TIMESTAMP}"
    LOG_DIR="$MODELS_DIR/logs"
    mkdir -p "$LOG_DIR"

    SLURM_SCRIPT="$LOG_DIR/${JOB_NAME}.sbatch"
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdzy-delta-gpu

cd $PROJECT_DIR
export PYTHONPATH="$PROJECT_DIR:\$PYTHONPATH"
$CMD
EOF

    echo "SLURM script: $SLURM_SCRIPT"
    if [[ "$DRY_RUN" == true ]]; then
        echo "Dry run - would submit: sbatch $SLURM_SCRIPT"
    else
        sbatch "$SLURM_SCRIPT"
    fi
else
    cd "$PROJECT_DIR"
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
    if [[ "$DRY_RUN" == true ]]; then
        echo "Dry run: $CMD"
    else
        eval "$CMD"
    fi
fi
