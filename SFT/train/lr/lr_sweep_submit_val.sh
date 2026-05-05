#!/bin/bash

# =============================================================================
# LR Sweep Submit — Target-Only Ablation
# =============================================================================
# Submits parallel SLURM jobs for dense LR grid search on target-only training
# (train_val_ablation.sh). Each job trains on n_val examples for max_steps and
# evaluates on the lr split (no test contamination).
#
# Usage:
#   bash lr_sweep_submit_val.sh --task triviaqa --model meta-llama/Llama-3.2-1B
#   bash lr_sweep_submit_val.sh --task triviaqa --model Qwen/Qwen3-1.7B-Base --n_lrs 20
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Dr.Post-Training
export PYTHONPATH="$CODE_DIR/Dr.Post-Training:$PYTHONPATH"

SUBMIT_SCRIPT="$CODE_DIR/Dr.Post-Training/submit.sh"

# CLI defaults
task=""
methods="FullTraining-Full"
model="meta-llama/Llama-3.2-1B"
seed=42
n_val=16
batch_size=8
n_lrs=20
lr_min=""
lr_max=""
# Matched-budget max_steps. Default tracks FullTraining-Full's 1 epoch on the
# subsampled training set at the same batch_size (e.g. for triviaqa: nq_open
# pct=0.1 with bs=8 -> 1100 steps).
max_steps=1100
dry_run=false

sweep_gpus="${SWEEP_GPUS:-1}"
sweep_mem="${SWEEP_MEM:-64g}"
sweep_time="${SWEEP_TIME:-1:00:00}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)        task="$2"; shift 2 ;;
        --methods)     methods="$2"; shift 2 ;;
        --model)       model="$2"; shift 2 ;;
        --seed)        seed="$2"; shift 2 ;;
        --n_val)       n_val="$2"; shift 2 ;;
        --batch_size)  batch_size="$2"; shift 2 ;;
        --n_lrs)       n_lrs="$2"; shift 2 ;;
        --lr_min)      lr_min="$2"; shift 2 ;;
        --lr_max)      lr_max="$2"; shift 2 ;;
        --max_steps)   max_steps="$2"; shift 2 ;;
        --dry-run)     dry_run=true; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash lr_sweep_submit_val.sh --task <task> [options]

Submits parallel SLURM jobs (1 GPU each) for target-only LR sweep.
Each job runs train_val_ablation.sh with --eval_split lr at one LR.

Required:
  --task <task>          Task: triviaqa, tydiqa, samsum, mmlu, etc.

Options:
  --model <model>        HF model name (default: meta-llama/Llama-3.2-1B)
  --seed <seed>          Seed (default: 42)
  --n_val <n>            Val samples to train on (default: 16)
  --batch_size <bs>      Batch size (default: 8)
  --n_lrs <n>            Number of log-spaced LRs (default: 20)
  --lr_min <lr>          Min LR (default: 1e-7)
  --lr_max <lr>          Max LR (default: 1e-3)
  --max_steps <n>        Matched-budget max_steps (default: 1100; pass main
                         experiment's step count at matching batch_size)
  --dry-run              Print without submitting

Env: SWEEP_GPUS=1 SWEEP_MEM=64g SWEEP_TIME=1:00:00
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done

if [[ -z "$task" ]]; then
    echo "ERROR: --task is required"
    exit 1
fi

# Default LR range depends on finetuning method (LoRA needs higher LRs).
if [[ -z "$lr_min" || -z "$lr_max" ]]; then
    if [[ "$methods" == *"LoRA"* ]]; then
        : "${lr_min:=1e-5}"; : "${lr_max:=1e-1}"
    else
        : "${lr_min:=1e-7}"; : "${lr_max:=1e-3}"
    fi
fi

generate_lr_grid() {
    local lr_min="$1" lr_max="$2" n="$3"
    python3 -c "
import numpy as np
for lr in np.logspace(np.log10($lr_min), np.log10($lr_max), $n):
    print(f'{lr:.2e}')
"
}

lr_values=$(generate_lr_grid "$lr_min" "$lr_max" "$n_lrs")
model_name=$(basename "$model")

# Method tag for unique job names when sweeping multiple finetuning variants in
# parallel (avoids collisions on the same task+model+lr).
method_tag=""
case "$methods" in
    *LoRA*) method_tag="_lora" ;;
    *MeSO*) method_tag="_meso" ;;
esac

echo ""
echo "========================================================"
echo "  Target-Only LR Sweep Submit"
echo "========================================================"
echo "Task:          $task"
echo "Methods:       $methods"
echo "Model:         $model"
echo "Seed:          $seed | n_val: $n_val | batch_size: $batch_size"
echo "Max steps:     $max_steps (matched-budget at bs=$batch_size)"
echo "LR range:      $lr_min -> $lr_max ($n_lrs values)"
echo "SLURM:         GPUs=$sweep_gpus, MEM=$sweep_mem, TIME=$sweep_time"
echo "========================================================"

# Suppress email notifications for sweep jobs
unset SLURM_MAIL_USER

total=0
for lr in $lr_values; do
    job_name="lr_sweep_val_${task}${method_tag}_${model_name}_${lr}"
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] GPUS=$sweep_gpus MEM=$sweep_mem TIME=$sweep_time JOB_NAME=$job_name" \
             "$SUBMIT_SCRIPT SFT/train/train_val_ablation.sh" \
             "--task $task --methods $methods --model $model --seed $seed" \
             "--n_val $n_val --batch_size $batch_size --max_steps $max_steps --lr $lr --eval_split lr"
    else
        GPUS=$sweep_gpus MEM=$sweep_mem TIME=$sweep_time JOB_NAME="$job_name" \
            "$SUBMIT_SCRIPT" SFT/train/train_val_ablation.sh \
            --task "$task" --methods "$methods" --model "$model" --seed "$seed" \
            --n_val "$n_val" --batch_size "$batch_size" --max_steps "$max_steps" \
            --lr "$lr" --eval_split lr
    fi
    total=$((total + 1))
done

echo ""
echo "========================================================"
echo "  Submitted $total target-only sweep jobs"
echo "========================================================"
echo ""
echo "After all jobs complete, collect results:"
echo "  bash SFT/train/lr/lr_sweep_collect_val.sh --task $task --model $model"
echo "========================================================"
