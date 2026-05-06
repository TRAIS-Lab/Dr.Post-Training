#!/bin/bash

# Hardcoded path to cluster_env.sh — see SFT/train/train.sh for rationale.
source /workspace-vast/pbb/Dr.Post-Training/cluster_env.sh \
    || { echo "ERROR: /workspace-vast/pbb/Dr.Post-Training/cluster_env.sh not found."; exit 1; }
activate_env

cd "$CODE_DIR/Dr.Post-Training"

export PYTHONPATH="$CODE_DIR/Dr.Post-Training:$PYTHONPATH"

set -e

# Default values
models_dir="$SCRATCH_DIR/Dr.Post-Training/SFT/runs"
data_dir="$SCRATCH_DIR/Dr.Post-Training/SFT/data"
model_path=""
train=""
task=""
subject=""
method=""
n_test=-1
batch_size=1
max_new_tokens=128
seed=42
dry_run=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)
            models_dir="$2"
            shift 2
            ;;
        --data_dir)
            data_dir="$2"
            shift 2
            ;;
        --model_path)
            model_path="$2"
            shift 2
            ;;
        --train)
            train="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --subject)
            subject="$2"
            shift 2
            ;;
        --method)
            method="$2"
            shift 2
            ;;
        --n_test)
            n_test="$2"
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
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --models_dir DIR     Models directory (default: \$SCRATCH_DIR/Dr.Post-Training/SFT)"
            echo "  --data_dir DIR       Data directory (default: \$SCRATCH_DIR/Dr.Post-Training/SFT/data)"
            echo "  --train NAME         Filter by training dataset (alpaca, less, tulu3, wizardlm)"
            echo "  --task NAME          Override task (samsum, tydiqa, mmlu, bbh, gsm8k, math500)"
            echo "  --subject NAME       MMLU subject or BBH task to evaluate on (default: all)"
            echo "  --method NAME        Filter by method (e.g., FullTraining-MeSO, LayerWiseSubset-Full)"
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

echo ""
echo "========================================================"
echo "  SFT Evaluation"
echo "========================================================"
echo "Models dir:      $models_dir"
echo "Data dir:        $data_dir"
echo ""
echo "Filters:"
echo "  Train:         ${train:-all}"
echo "  Task:          ${task:-auto-detect}"
echo "  Subject:       ${subject:-all}"
echo "  Method:        ${method:-all}"
echo ""
echo "Generation:"
echo "  Batch size:    $batch_size"
echo "  Max new tokens: $max_new_tokens"
echo "  N test:        $n_test (-1 = all)"
echo "  Seed:          $seed"
echo "========================================================"

# Build command
cmd="python -m SFT.eval.eval"
if [[ -n "$model_path" ]]; then
    cmd="$cmd --model_path $model_path"
else
    cmd="$cmd --models_dir $models_dir"
fi
cmd="$cmd --data_dir $data_dir"
cmd="$cmd --n_test $n_test"
cmd="$cmd --batch_size $batch_size"
cmd="$cmd --max_new_tokens $max_new_tokens"
cmd="$cmd --seed $seed"

if [[ -n "$train" ]]; then
    cmd="$cmd --train $train"
fi

if [[ -n "$task" ]]; then
    cmd="$cmd --task $task"
fi

if [[ -n "$subject" ]]; then
    cmd="$cmd --subject $subject"
fi

if [[ -n "$method" ]]; then
    cmd="$cmd --method $method"
fi

if [[ "$dry_run" == true ]]; then
    echo "Dry run: $cmd"
else
    eval "$cmd"
fi
