#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Source cluster config (skip if already set by submit.sh)
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Dr.Post-Training

export PYTHONPATH="$CODE_DIR/Dr.Post-Training:$PYTHONPATH"

# =============================================================================
# Case study: Standard training with dual LayerWiseSubset + GlobalSubset scoring
# =============================================================================

# Defaults — match `less_tydiqa` from the main 4-setting matrix.
# Override via CLI to run any of the other 3 settings, or use --sweep
# to run all 4 settings × 5 seeds.
model="meta-llama/Llama-3.2-1B"
data_dir="$SCRATCH_DIR/Dr.Post-Training/SFT/data"
task="tydiqa"
train_dataset="less"
percentage=0.005
seed=42
batch_size=8
n_val=16
n_eval=500
lr="1e-05"  # FullTraining-Full LR (matches main experiment)
selection_frac="0.5"
val_batch_size="1"
val_strategy="separate_batch_factorized"
record_freq=1
max_steps=-1  # -1 = full epoch
dry_run=false
sweep=false

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)          model="$2"; shift 2 ;;
        --task)           task="$2"; shift 2 ;;
        --train)          train_dataset="$2"; shift 2 ;;
        --batch_size)     batch_size="$2"; shift 2 ;;
        --percentage)     percentage="$2"; shift 2 ;;
        --lr)             lr="$2"; shift 2 ;;
        --seed)           seed="$2"; shift 2 ;;
        --n_val)          n_val="$2"; shift 2 ;;
        --n_eval)         n_eval="$2"; shift 2 ;;
        --selection_frac) selection_frac="$2"; shift 2 ;;
        --val_batch_size) val_batch_size="$2"; shift 2 ;;
        --record_freq)    record_freq="$2"; shift 2 ;;
        --max_steps)      max_steps="$2"; shift 2 ;;
        --sweep)          sweep=true; shift ;;
        --dry-run)        dry_run=true; shift ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# --sweep: run all 4 settings × 5 seeds (matches main 4-setting matrix).
# Re-invokes this script per (setting, seed) so each run gets its own
# JOB_NAME / output_dir.
if [[ "$sweep" == "true" ]]; then
    SETTINGS=(
        # train:task:percentage
        "alpaca:samsum:0.4"
        "less:tydiqa:0.005"
        "triviaqa:nq_open:0.05"
        "less:squad:0.005"
    )
    SEEDS=(2 22 42 62 82)
    SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
    dry_flag=""
    [[ "$dry_run" == "true" ]] && dry_flag="--dry-run"

    for entry in "${SETTINGS[@]}"; do
        IFS=':' read -r tr tk pct <<< "$entry"
        for s in "${SEEDS[@]}"; do
            echo ""
            echo "############################################################"
            echo "  Sweep: ${tr} → ${tk} (p=${pct}) seed=${s}"
            echo "############################################################"
            bash "$SCRIPT" --train "$tr" --task "$tk" --percentage "$pct" \
                --seed "$s" --lr "$lr" $dry_flag
        done
    done
    exit 0
fi

model_name=$(basename "$model")
DATA_SEED=$((seed + 1))
ID=$RANDOM
PORT=$((29400 + RANDOM % 10000))

# Case-study dirs live under runs/case_study/ (sibling of main + target-only dirs).
# Inner dir name has no "case_study_" prefix since the parent dir already encodes that,
# and we don't want collision with main runs (which share the same {train}_{task}-... prefix).
JOB_NAME="${train_dataset}_${task}-${model_name}-p${percentage}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"
output_dir=$SCRATCH_DIR/Dr.Post-Training/SFT/runs/case_study/${JOB_NAME}
mkdir -p "$output_dir"

echo ""
echo "=============================================="
echo "  Case Study: Selection Analysis"
echo "=============================================="
echo "Model: $model"
echo "Task: $train_dataset → $task"
echo "LR: $lr | Batch: $batch_size | Percentage: $percentage"
echo "Selection frac: $selection_frac | Val batch: $val_batch_size"
echo "Record freq: every $record_freq steps"
echo "Output: $output_dir"
echo "=============================================="

header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id=$ID --rdzv_backend c10d --rdzv_endpoint=localhost:$PORT \
-m SFT.case_study.analyze_selection"

training_args="--do_train=True \
--do_eval=True \
--max_seq_length=512 \
--use_fast_tokenizer=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--eval_steps=50 \
--eval_strategy=steps \
--save_strategy=no \
--num_train_epochs=1 \
--bf16=True \
--tf32=False \
--fp16=False \
--overwrite_output_dir=True \
--report_to=none \
--model_name_or_path $model \
--output_dir $output_dir \
--data_dir $data_dir \
--percentage $percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $batch_size \
--method NA \
--n_val $n_val \
--n_eval $n_eval \
--analysis_dataset $task \
--learning_rate $lr \
--gradient_accumulation_steps 1 \
--seed $seed \
--optim adamw_torch \
--selection_frac $selection_frac \
--val_strategy $val_strategy \
--use_flash_attention true \
--train_dataset_names $train_dataset \
--val_batch_size_for_selection $val_batch_size \
--lora False \
--record_selections True \
--record_selections_freq $record_freq"

# Add max_steps if specified
if [[ "$max_steps" -gt 0 ]]; then
    training_args="$training_args --max_steps $max_steps"
fi

training_args="$training_args 2>&1 | tee $output_dir/train.log"

if [ "$dry_run" = true ]; then
    echo "[DRY-RUN] $header $training_args"
else
    eval "$header" "$training_args"
fi
