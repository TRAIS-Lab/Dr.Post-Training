#!/bin/bash
#
# Evaluation script for SFT experiments
#
# Task is auto-detected from model name (e.g., alpaca_samsum -> samsum)
#
# Usage:
#   bash SFT/eval/eval.sh                       # Evaluate all models
#   bash SFT/eval/eval.sh --train alpaca        # Only alpaca training set
#   bash SFT/eval/eval.sh --batch_size 4        # Set batch size
#   bash SFT/eval/eval.sh --sbatch              # Submit to SLURM
#

set -e

# Default values
MODELS_DIR="/scratch/pbb/Project/Gradient-Streaming/SFT"
TRAIN=""
TASK=""
SUBJECT=""
N_TEST=-1
BATCH_SIZE=1
MAX_NEW_TOKENS=128
SBATCH=false
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
            echo "  --models_dir DIR     Models directory (default: /scratch/pbb/Project/Gradient-Streaming/SFT)"
            echo "  --train NAME         Filter by training dataset (alpaca, less, tulu3, wizardlm)"
            echo "  --task NAME          Override task (samsum, tydiqa, mmlu, bbh, gsm8k, math500)"
            echo "  --subject NAME       MMLU subject or BBH task to evaluate on (default: all)"
            echo "  --n_test N           Number of test examples (-1 for all)"
            echo "  --batch_size N       Batch size for generation (default: 1)"
            echo "  --max_new_tokens N   Max tokens to generate (default: 128)"
            echo "  --sbatch             Submit as SLURM job"
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
PROJECT_DIR="$(dirname "$SFT_DIR")"
DATA_DIR="$SFT_DIR/data"

echo "======================================"
echo "SFT Evaluation"
echo "======================================"
echo "Models:     $MODELS_DIR"
echo "Train:      ${TRAIN:-all}"
echo "Task:       ${TASK:-auto-detect}"
echo "Subject:    ${SUBJECT:-all}"
echo "Batch size: $BATCH_SIZE"
echo "======================================"

# Build command
CMD="python -m SFT.eval.eval"
CMD="$CMD --models_dir $MODELS_DIR"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --n_test $N_TEST"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_new_tokens $MAX_NEW_TOKENS"

if [[ -n "$TRAIN" ]]; then
    CMD="$CMD --train $TRAIN"
fi

if [[ -n "$TASK" ]]; then
    CMD="$CMD --task $TASK"
fi

if [[ -n "$SUBJECT" ]]; then
    CMD="$CMD --subject $SUBJECT"
fi

if [[ "$SBATCH" == true ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="eval_${TIMESTAMP}"
    LOG_DIR="$MODELS_DIR/logs"
    mkdir -p "$LOG_DIR"

    SLURM_SCRIPT="$LOG_DIR/${JOB_NAME}.sbatch"
    cat > "$SLURM_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_NAME}_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

cd $PROJECT_DIR
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
    if [[ "$DRY_RUN" == true ]]; then
        echo "Dry run: $CMD"
    else
        eval "$CMD"
    fi
fi
