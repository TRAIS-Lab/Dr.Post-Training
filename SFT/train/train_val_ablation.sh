#!/bin/bash
#
# Ablation study: Train on ~n_val validation samples (standard training only).
#
# Instead of tulu3->tydiqa or alpaca->samsum, we train on a small subset of the
# task's validation split and evaluate on the test split. Uses percentage-based
# sampling to select ~n_val samples. Optimization steps match the original experiment.
#
# Only Standard methods (no curation): FullTraining-Full, FullTraining-LoRA, FullTraining-MeSO.
#
# Usage:
#   bash SFT/train/train_val_ablation.sh --task tydiqa --methods all --seed 42
#   bash SFT/train/train_val_ablation.sh --task samsum --methods all --seed 42
#   bash SFT/train/train_val_ablation.sh --task truthfulqa --methods all --seed 42
#   bash SFT/train/train_val_ablation.sh --task tydiqa --methods all --lr 5e-05 --dry-run

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Dr.Post-Training

export PYTHONPATH="$CODE_DIR/Dr.Post-Training:$PYTHONPATH"

SCRIPT_DIR="$CODE_DIR/Dr.Post-Training/SFT/train"
CONFIG_DIR="$SCRIPT_DIR/configs"

# =============================================================================
# Defaults (base_training_args is built after CLI parsing so --eval_steps takes effect)
# =============================================================================

model="meta-llama/Llama-3.2-1B"
data_dir="$SCRATCH_DIR/Dr.Post-Training/SFT/data"
task=""
seed=42

optim="adamw_torch"
batch_size=8
gradient_accumulation_steps=1
use_flash_attention=true

n_val=16
n_eval=500
eval_steps=400

# LR configuration
lr_config_file="SFT/train/lr/config.json"
lr_override=""
default_lr_full="2e-05"
default_lr_lora="5e-04"

# LoRA defaults
lora_r=32
lora_alpha=1
lora_dropout=0.1

# Compressor update frequency
update_compressor_freq=200

# Multi-method mode
methods=""
dry_run=false
max_steps_override=""

# Optional override: config subdirectory (relative to SFT/train/configs).
# If unset, derived from --task via LR_CONFIG_KEYS.
config_dir_override=""

# Optional eval_split override (for LR sweeps: --eval_split lr).
eval_split_override=""

# Target optimization steps: match total sample-passes of main experiments
# main_steps * main_batch_size / val_ablation_batch_size = main_steps * 8
declare -A TARGET_STEPS=(
    ["tydiqa"]=1174       # tulu3_tydiqa main: 1174 steps at bs=8
    ["samsum"]=2600       # alpaca_samsum main: 2600 steps at bs=8
    ["nq_open"]=1100      # triviaqa_nq main: 1100 steps at bs=8
    ["squad"]=1100        # nq_squad main: 1100 steps at bs=8
)
declare -A LR_CONFIG_KEYS=(
    ["tydiqa"]="tulu3_tydiqa"
    ["samsum"]="alpaca_samsum"
    ["nq_open"]="triviaqa_nq"
    ["squad"]="nq_squad"
)

# =============================================================================
# Category mappings (Standard methods only)
# =============================================================================
declare -A CATEGORY_METHODS=(
    ["all"]="FullTraining-Full,FullTraining-LoRA,FullTraining-MeSO"
    ["baseline"]="FullTraining-Full,FullTraining-LoRA"
    ["full"]="FullTraining-Full"
    ["lora"]="FullTraining-LoRA"
    ["compression"]="FullTraining-MeSO"
)

# =============================================================================
# Parse CLI arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)           task="$2"; shift 2 ;;
        --config_dir|-c)  config_dir_override="$2"; shift 2 ;;
        --methods)        methods="$2"; shift 2 ;;
        --max_steps)      max_steps_override="$2"; shift 2 ;;
        --model)          model="$2"; shift 2 ;;
        --batch_size)     batch_size="$2"; shift 2 ;;
        --n_val)          n_val="$2"; shift 2 ;;
        --n_eval)         n_eval="$2"; shift 2 ;;
        --eval_steps)     eval_steps="$2"; shift 2 ;;
        --lr)             lr_override="$2"; shift 2 ;;
        --lr_config)      lr_config_file="$2"; shift 2 ;;
        --eval_split)     eval_split_override="$2"; shift 2 ;;
        --seed)           seed="$2"; shift 2 ;;
        --data_dir)       data_dir="$2"; shift 2 ;;
        --gradient_accumulation_steps) gradient_accumulation_steps="$2"; shift 2 ;;
        --dry-run)        dry_run=true; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash train_val_ablation.sh --task <task> --methods <methods> [options]

Ablation: Standard training on ~n_val validation samples.
Matches optimization steps to the original tulu3->tydiqa / alpaca->samsum / less->truthfulqa experiments.

Required:
  --task <task>          Task: tydiqa, samsum, or truthfulqa
  --methods <list>       Methods: all, baseline, full, lora, compression,
                         or specific names (FullTraining-Full, FullTraining-LoRA, FullTraining-MeSO)

Optional:
  --max_steps <n>        Override target optimization steps
  --batch_size <n>       Batch size (default: 8)
  --n_val <n>            Number of val samples to train on (default: 16)
  --n_eval <n>           Evaluation examples (default: 500)
  --eval_steps <n>       Evaluate every N steps (default: 400; set to ~max_steps/100 for ~100 ppl points)
  --lr <lr>              Learning rate override
  --seed <seed>          Random seed (default: 42)
  --dry-run              Print commands without executing
HELP
            exit 0
            ;;
        *)
            echo "Unknown argument: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validate inputs
# =============================================================================
if [[ -z "$task" ]]; then
    echo "ERROR: --task is required (tydiqa or samsum)"
    exit 1
fi

if [[ -z "$methods" ]]; then
    echo "ERROR: --methods is required"
    exit 1
fi

# Build base_training_args after CLI parsing so --eval_steps takes effect
export base_training_args="--do_train=True \
--do_eval=True \
--max_seq_length=512 \
--use_fast_tokenizer=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--eval_steps=$eval_steps \
--eval_strategy=steps \
--save_strategy=no \
--bf16=True \
--tf32=False \
--fp16=False \
--overwrite_output_dir=True \
--report_to=none"

val_file="${data_dir}/eval/${task}/${task}_validation_data.jsonl"
if [[ ! -f "$val_file" ]]; then
    echo "ERROR: Validation file not found: $val_file"
    exit 1
fi

# Determine max_steps
if [[ -n "$max_steps_override" ]]; then
    max_steps="$max_steps_override"
elif [[ -n "${TARGET_STEPS[$task]}" ]]; then
    max_steps="${TARGET_STEPS[$task]}"
else
    echo "ERROR: No target steps defined for task '$task'. Use --max_steps to specify."
    exit 1
fi

# Resolve task-specific config directory.
# Priority: --config_dir override > LR_CONFIG_KEYS[task] > error
if [[ -n "$config_dir_override" ]]; then
    if [[ "$config_dir_override" = /* ]]; then
        task_config_dir="$config_dir_override"
    else
        task_config_dir="$CONFIG_DIR/$config_dir_override"
        # Strip leading "configs/" if user passed "configs/foo"
        task_config_dir="${task_config_dir/configs\/configs\//configs/}"
    fi
elif [[ -n "${LR_CONFIG_KEYS[$task]}" ]]; then
    task_config_dir="$CONFIG_DIR/${LR_CONFIG_KEYS[$task]}"
else
    echo "ERROR: No config directory mapped for task '$task'. Use --config_dir to specify."
    exit 1
fi

if [[ ! -d "$task_config_dir" ]]; then
    echo "ERROR: Config directory not found: $task_config_dir"
    exit 1
fi

model_name=$(basename "$model")

# Compute percentage to get exactly n_val samples from the validation file
# Use (n_val + 0.5) / n_lines to avoid int() truncation from float rounding
n_file_lines=$(wc -l < "$val_file")
percentage=$(python3 -c "print(($n_val + 0.5) / $n_file_lines)")
n_actual=$(python3 -c "print(int($n_file_lines * $percentage))")

steps_per_epoch=$((n_actual / batch_size))
if [[ $((n_actual % batch_size)) -ne 0 ]]; then
    steps_per_epoch=$((steps_per_epoch + 1))
fi
n_epochs=$(( (max_steps + steps_per_epoch - 1) / steps_per_epoch ))

# =============================================================================
# Helper: Read YAML config (same as train.sh)
# =============================================================================
read_yaml() {
    local config_file="$1"

    cfg_method="FullTraining"
    cfg_finetuning="Full"
    cfg_score_sparsifier=""
    cfg_score_projector=""
    cfg_opt_sparsifier=""
    cfg_opt_projector=""
    cfg_lora_r=""
    cfg_lora_alpha=""
    cfg_lora_dropout=""

    local section=""
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue

        local full_key="" val=""
        if [[ "$line" =~ ^[[:space:]] ]]; then
            val=$(echo "$line" | cut -d: -f2- | xargs | sed 's/^"//;s/"$//' | sed "s/^'//;s/'$//")
            full_key="${section}.$(echo "$line" | cut -d: -f1 | xargs)"
        else
            local top_key top_val
            top_key=$(echo "$line" | cut -d: -f1 | xargs)
            top_val=$(echo "$line" | cut -d: -f2- | xargs | sed 's/^"//;s/"$//' | sed "s/^'//;s/'$//")
            if [[ -z "$top_val" ]]; then
                section="$top_key"; continue
            fi
            section=""
            full_key="$top_key"
            val="$top_val"
        fi

        case "$full_key" in
            method)                              cfg_method="$val" ;;
            finetuning)                          cfg_finetuning="$val" ;;
            lora_r)                              cfg_lora_r="$val" ;;
            lora_alpha)                          cfg_lora_alpha="$val" ;;
            lora_dropout)                        cfg_lora_dropout="$val" ;;
            opt_grad_compression.sparsifier)     cfg_opt_sparsifier="$val" ;;
            opt_grad_compression.projector)      cfg_opt_projector="$val" ;;
        esac
    done < "$config_file"

    case "$cfg_method" in
        Standard) cfg_internal_method="NA" ;;
        *)        cfg_internal_method="$cfg_method" ;;
    esac

    case "$cfg_finetuning" in
        LoRA|MeSO-LoRA) cfg_lora="true" ;;
        *)              cfg_lora="false" ;;
    esac
}

# =============================================================================
# Helper: Look up LR from config file
# =============================================================================
lookup_lr() {
    local config_key="$1"
    local exp_name="$2"
    local is_lora="$3"

    if [[ -n "$lr_override" ]]; then
        echo "$lr_override"
        return
    fi

    if [[ -f "$lr_config_file" ]]; then
        local looked_up_lr
        looked_up_lr=$(python3 -c "
import json, sys
try:
    with open('$lr_config_file', 'r') as f:
        config = json.load(f)
    lr_val = config.get('$config_key', {}).get('$exp_name', {}).get('lr')
    if lr_val is not None:
        print(f'{lr_val:.0e}' if lr_val < 0.001 else f'{lr_val}')
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)
        if [[ $? -eq 0 ]] && [[ -n "$looked_up_lr" ]]; then
            echo "$looked_up_lr"
            return
        fi
    fi

    if [ "$is_lora" = true ]; then
        echo "$default_lr_lora"
    else
        echo "$default_lr_full"
    fi
}

# =============================================================================
# Helper: Resolve method names from categories
# =============================================================================
resolve_methods() {
    local input="$1"
    local resolved=""

    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)
        if [[ -n "${CATEGORY_METHODS[$item]}" ]]; then
            resolved="${resolved:+$resolved,}${CATEGORY_METHODS[$item]}"
        elif [[ -f "$task_config_dir/${item}.yaml" ]]; then
            resolved="${resolved:+$resolved,}$item"
        else
            echo "ERROR: Unknown method or category: $item"
            exit 1
        fi
    done

    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# =============================================================================
# Run a single method
# =============================================================================
run_method() {
    local exp_name="$1"
    local config_file="$task_config_dir/${exp_name}.yaml"

    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config not found: $config_file"
        return 1
    fi

    read_yaml "$config_file"

    # LR lookup: reuse LRs from the original experiment (tulu3_tydiqa / alpaca_samsum)
    local config_key="${LR_CONFIG_KEYS[$task]}"
    local exp_lr=$(lookup_lr "$config_key" "$exp_name" "$cfg_lora")

    local JOB_NAME="${task}_val_${task}-${model_name}-${method_str:-$exp_name}-ms${max_steps}-lr${exp_lr}-b${batch_size}-v${n_val}-s${seed}"

    local output_dir=$SCRATCH_DIR/Dr.Post-Training/SFT/${JOB_NAME}
    mkdir -p "$output_dir"

    echo ""
    echo "=============================================="
    echo "  [Val Ablation] Running: $exp_name"
    echo "=============================================="
    echo "Job: $JOB_NAME"
    echo "Model: $model | Task: $task | LR: $exp_lr"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    echo "Train: ~${n_actual} val samples (pct=${percentage}) | Batch: $batch_size"
    echo "Max steps: $max_steps (~${n_epochs} epochs)"
    echo "Output: $output_dir"
    echo "=============================================="

    local exp_base_training_args="$base_training_args"

    # Model-specific FSDP config
    case "$model" in
        *Llama-2-13b*|*llama-2-13b*)
            exp_base_training_args="$exp_base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune" ;;
        *Mistral-7B*|*mistral-7b*)
            exp_base_training_args="$exp_base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune" ;;
    esac

    local DATA_SEED=$((seed + 1))
    local ID=$RANDOM
    local PORT=$((29400 + RANDOM % 10000))

    local header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id=$ID --rdzv_backend c10d --rdzv_endpoint=localhost:$PORT \
-m SFT.train.train"

    # For tasks whose test split has no gold responses (e.g. hhrlhf →
    # CategoricalHarmfulQA), point the trainer's eval_dataset to the lr split.
    # Explicit --eval_split overrides this default.
    local eval_split_arg=""
    if [[ -n "$eval_split_override" ]]; then
        eval_split_arg="--eval_split $eval_split_override"
    else
        case "$task" in
            hhrlhf) eval_split_arg="--eval_split lr" ;;
        esac
    fi

    local training_args="$exp_base_training_args \
--model_name_or_path $model \
--output_dir $output_dir \
--data_dir $data_dir \
--train_files $val_file \
--percentage $percentage \
--max_steps $max_steps \
--num_train_epochs 99999 \
--data_seed $DATA_SEED \
--per_device_train_batch_size $batch_size \
--method NA \
--n_val $n_val \
--n_eval $n_eval \
--analysis_dataset $task \
--learning_rate $exp_lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed \
--optim $optim $eval_split_arg \
--use_flash_attention $use_flash_attention"

    # LoRA
    if [ "$cfg_lora" = true ]; then
        local eff_lora_r="${cfg_lora_r:-$lora_r}"
        local eff_lora_alpha="${cfg_lora_alpha:-$lora_alpha}"
        local eff_lora_dropout="${cfg_lora_dropout:-$lora_dropout}"
        training_args="$training_args --lora True --lora_r $eff_lora_r --lora_alpha $eff_lora_alpha --lora_dropout $eff_lora_dropout"
    else
        training_args="$training_args --lora False"
    fi

    # Compression (for MeSO)
    [[ -n "$cfg_opt_sparsifier" && "$cfg_opt_sparsifier" != "none" ]] && training_args="$training_args --sparsification $cfg_opt_sparsifier --update_compressor_freq $update_compressor_freq"
    [[ -n "$cfg_opt_projector" && "$cfg_opt_projector" != "none" ]] && training_args="$training_args --projection $cfg_opt_projector"

    training_args="$training_args 2>&1 | tee $output_dir/train.log"

    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] $header $training_args"
    else
        eval "$header" "$training_args"
    fi
}

# =============================================================================
# Main
# =============================================================================
resolved_methods=$(resolve_methods "$methods")
IFS=',' read -ra method_list <<< "$resolved_methods"
TOTAL=${#method_list[@]}

echo ""
echo "========================================================"
echo "  SFT Val-Ablation Training"
echo "========================================================"
echo "Task: $task | Train on ~${n_actual} val samples (pct=${percentage}, batch=$batch_size)"
echo "Val file: $val_file ($n_file_lines total, sampling $n_val)"
echo "Methods: $resolved_methods ($TOTAL total)"
echo "Max steps: $max_steps (~${n_epochs} epochs) matching original experiment"
echo "Model: $model | Seed: $seed"
echo "========================================================"

current=0
for method_name in "${method_list[@]}"; do
    current=$((current + 1))
    echo ""
    echo "[$current/$TOTAL] $method_name"
    run_method "$method_name"
done

echo ""
echo "========================================================"
echo "  All $TOTAL methods completed!"
echo "========================================================"
