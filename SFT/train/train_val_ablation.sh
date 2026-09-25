#!/bin/bash
# Target-Only launcher.
# Trains the standard (non-curated) arm directly on the n_val target examples and evaluates on the target's test split.
# When n_val < the D* file size the run uses exactly the rows a curation run with the same --seed / --n_val sees
# (train.py: random.Random(seed).shuffle(all rows), first n_val), written to <run_dir>/dstar_subset.jsonl.
# Forwards the yaml keys init_eot_from_eos / loss_reduction; run dirs land under --runs_root (default
# $SCRATCH_DIR/Dr.Post-Training/SFT/runs_v2). The yaml `learning_rate` is used unless --lr is given (MeSO 5e-5, LoRA 1e-4,
# Full 1e-5 as in the FullTraining-*.yaml of the setting) and the nested MeSO keys `optimizer.compression` / `optimizer.refresh_freq`
# are parsed. Used by SFT/train/qa_target_only_task.sh for the Target-Only rows of the question-answering tables (adxtab:sft-prelim-ppl);
# the Qwen Target-Only arms are not part of the paper. Never edit this file while jobs run it: copy, edit, launch the copy.
#
# Ablation study: Train on ~n_val validation samples (standard training only).
#
# Instead of tulu3->tydiqa or alpaca->samsum, we train on a small subset of the
# task's validation split and evaluate on the test split. Uses percentage-based
# sampling to select ~n_val samples. Optimization steps match the main experiment.
#
# Only Standard methods (no curation): FullTraining-Full, FullTraining-LoRA, FullTraining-MeSO.
#
# Usage:
#   bash SFT/train/train_val_ablation.sh --task tydiqa --methods all --seed 42
#   bash SFT/train/train_val_ablation.sh --task samsum --methods all --seed 42
#   bash SFT/train/train_val_ablation.sh --task truthfulqa --methods all --seed 42
#   bash SFT/train/train_val_ablation.sh --task tydiqa --methods all --lr 5e-05 --dry-run

# cluster_env.sh: $DRPT_CLUSTER_ENV if set, else the one at the root of this checkout.
_drpt_env=""
for _c in "${DRPT_CLUSTER_ENV:-}" \
          "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)/cluster_env.sh"; do
    [[ -n "$_c" && -f "$_c" ]] && { _drpt_env="$_c"; break; }
done
[[ -n "$_drpt_env" ]] || { echo "ERROR: cluster_env.sh not found (set DRPT_CLUSTER_ENV or create it at the repo root)."; exit 1; }
source "$_drpt_env"
unset _drpt_env _c
activate_env

cd "$CODE_DIR/Dr.Post-Training"

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

# Fixed LRs (override per-call with --lr if needed).
lr_override=""
default_lr_full="1e-05"
default_lr_lora="1e-04"

# LoRA defaults
lora_r=8
lora_alpha=16
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
runs_root_override=""
init_eot_override=""

# Optional eval_split override (e.g., --eval_split lr to evaluate on the
# extra held-out dev split instead of test).
eval_split_override=""

# Model/sequence knobs for the Dolci (Qwen3) settings.
max_seq_length=512
gradient_checkpointing=false
val_seq_length_multiplier=""     # empty -> train.py default (1.2x avg train length); 0 disables

# Optional schedule overrides (for blow-up dynamics studies).
lr_scheduler_override=""
warmup_ratio_override=""
weight_decay_override=""

# Target optimization steps: match total sample-passes of main experiments
# main_steps * main_batch_size / val_ablation_batch_size = main_steps * 8
declare -A TARGET_STEPS=(
    ["tydiqa"]=1174       # less_tydiqa main: ~1225 steps at bs=8
    ["samsum"]=2600       # alpaca_samsum main: 2600 steps at bs=8
    ["nq_open"]=1100      # triviaqa_nq main: ~1107 steps at bs=8
    ["squad"]=1100        # less_squad main: ~1225 steps at bs=8
    ["precise_if"]=4000   # dolci_inst_if main
    ["math_persona"]=4000          # dolci_reason_mathpersona main (32K rows / bs 8, 1 epoch)
    ["math_ref128_gen32b"]=4000    # dolci_reason_mathref128_gen32b main
)
declare -A LR_CONFIG_KEYS=(
    ["tydiqa"]="less_tydiqa"
    ["samsum"]="alpaca_samsum"
    ["nq_open"]="triviaqa_nq"
    ["squad"]="less_squad"
    ["precise_if"]="dolci_inst_if"
    ["math_persona"]="dolci_reason_mathpersona"
    ["math_ref128_gen32b"]="dolci_reason_mathref128_gen32b"
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
        --runs_root)      runs_root_override="$2"; shift 2 ;;
        --init_eot_from_eos) init_eot_override="$2"; shift 2 ;;
        --methods)        methods="$2"; shift 2 ;;
        --max_steps)      max_steps_override="$2"; shift 2 ;;
        --model)          model="$2"; shift 2 ;;
        --batch_size)     batch_size="$2"; shift 2 ;;
        --n_val)          n_val="$2"; shift 2 ;;
        --n_eval)         n_eval="$2"; shift 2 ;;
        --eval_steps)     eval_steps="$2"; shift 2 ;;
        --lr)             lr_override="$2"; shift 2 ;;
        --eval_split)     eval_split_override="$2"; shift 2 ;;
        --lr_scheduler_type) lr_scheduler_override="$2"; shift 2 ;;
        --warmup_ratio)   warmup_ratio_override="$2"; shift 2 ;;
        --weight_decay)   weight_decay_override="$2"; shift 2 ;;
        --seed)           seed="$2"; shift 2 ;;
        --data_dir)       data_dir="$2"; shift 2 ;;
        --gradient_accumulation_steps) gradient_accumulation_steps="$2"; shift 2 ;;
        --max_seq_length) max_seq_length="$2"; shift 2 ;;
        --gradient_checkpointing) gradient_checkpointing="$2"; shift 2 ;;
        --val_seq_length_multiplier) val_seq_length_multiplier="$2"; shift 2 ;;
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
  --model <name>         Base model (default: meta-llama/Llama-3.2-1B)
  --max_seq_length <n>   Sequence length (default: 512; Dolci settings use 2048)
  --gradient_checkpointing <bool>      Non-reentrant activation checkpointing (default: false)
  --val_seq_length_multiplier <x>      D* length-rejection multiplier; 0 disables (default: train.py's 1.2)
  --eval_steps <n>       Evaluate every N steps (default: 400; set to ~max_steps/100 for ~100 ppl points)
  --lr <lr>              Learning rate override
  --seed <seed>          Random seed (default: 42)
  --runs_root <dir>      Parent directory for run dirs (default $SCRATCH_DIR/Dr.Post-Training/SFT/runs_v2)
  --init_eot_from_eos <bool>  Initialise the chat end-of-turn row from <|endoftext|> (overrides the yaml key)
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

# Build base_training_args after CLI parsing so --eval_steps takes effect.
# Schedule defaults (overridable via --lr_scheduler_type / --warmup_ratio / --weight_decay).
sched="${lr_scheduler_override:-linear}"
warmup="${warmup_ratio_override:-0.03}"
wdecay="${weight_decay_override:-0.0}"

export base_training_args="--do_train=True \
--do_eval=True \
--max_seq_length=$max_seq_length \
--use_fast_tokenizer=True \
--lr_scheduler_type=$sched \
--warmup_ratio=$warmup \
--weight_decay=$wdecay \
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
if [[ "$n_val" -ge "$n_file_lines" ]]; then
    # D* smaller than the requested budget (e.g. regenerated targets that dropped unsolved problems): train on every row.
    percentage=1.0
    n_actual=$n_file_lines
    aligned_subset=0
    echo "NOTE: n_val=$n_val >= $n_file_lines rows in $val_file; training on all $n_file_lines rows (run name keeps v${n_val})."
else
    # Aligned subset: same rows as a curation run with this seed/n_val (train.py -> get_val_dataset.load_unified_jsonl).
    percentage=1.0
    n_actual=$n_val
    aligned_subset=1
fi

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
    cfg_learning_rate=""
    cfg_lora_r=""
    cfg_lora_alpha=""
    cfg_lora_dropout=""
    cfg_init_eot_from_eos=""
    cfg_loss_reduction=""

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
            optimizer.compression)               cfg_opt_sparsifier="$val" ;;
            optimizer.refresh_freq)              update_compressor_freq="$val" ;;
            learning_rate)                       cfg_learning_rate="$val" ;;
            init_eot_from_eos)                   cfg_init_eot_from_eos="$val" ;;
            loss_reduction)                      cfg_loss_reduction="$val" ;;
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
# Helper: Pick fixed LR (CLI override > LoRA default > Full default)
# =============================================================================
lookup_lr() {
    local _config_key="$1"   # legacy unused arg
    local _exp_name="$2"     # legacy unused arg
    local is_lora="$3"

    if [[ -n "$lr_override" ]]; then
        echo "$lr_override"
        return
    fi
    if [[ -n "$cfg_learning_rate" ]]; then
        echo "$cfg_learning_rate"
        return
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

    # LR lookup: reuse LRs from the matching main-experiment config
    local config_key="${LR_CONFIG_KEYS[$task]}"
    local exp_lr=$(lookup_lr "$config_key" "$exp_name" "$cfg_lora")

    # Append schedule suffix when overrides differ from cosine+0.1 default — keeps
    # blow-up-dynamics ablation runs in separate dirs from the canonical cosine ones.
    local sched_suffix=""
    if [[ -n "$lr_scheduler_override" || -n "$warmup_ratio_override" ]]; then
        sched_suffix="-${sched}-w${warmup}"
    fi
    local JOB_NAME="${task}_val_${task}-${model_name}-${method_str:-$exp_name}-ms${max_steps}-lr${exp_lr}-b${batch_size}-v${n_val}-s${seed}${sched_suffix}"

    local runs_root="${runs_root_override:-$SCRATCH_DIR/Dr.Post-Training/SFT/runs_v2}"
    local output_dir=$runs_root/${JOB_NAME}
    mkdir -p "$output_dir"

    local train_file="$val_file"
    if [[ "${aligned_subset:-0}" == "1" ]]; then
        train_file="$output_dir/dstar_subset.jsonl"
        python3 - "$val_file" "$train_file" "$seed" "$n_val" <<'PY'
import random, sys
src, dst, seed, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = [l for l in open(src, encoding="utf-8") if l.strip()]
random.Random(seed).shuffle(rows)          # identical to get_val_dataset.load_unified_jsonl(seed=training seed)
open(dst, "w", encoding="utf-8").writelines(rows[:k])
print(f"aligned D* subset: {k} of {len(rows)} rows (seed {seed}) -> {dst}")
PY
    fi

    echo ""
    echo "=============================================="
    echo "  [Val Ablation] Running: $exp_name"
    echo "=============================================="
    echo "Job: $JOB_NAME"
    echo "Model: $model | Task: $task | LR: $exp_lr"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    echo "Train: ${n_actual} D* rows (pct=${percentage}, aligned_subset=${aligned_subset:-0}) | Batch: $batch_size"
    echo "Max steps: $max_steps (~${n_epochs} epochs)"
    echo "Stop-token init: ${init_eot_override:-${cfg_init_eot_from_eos:-<train.py default>}} | Loss reduction: ${cfg_loss_reduction:-sample_mean (default)}"
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
    # Deterministic port from SLURM_JOB_ID (or PID fallback) — see train.sh.
    local PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 40000)))

    local header="torchrun --standalone --nproc_per_node 1 --nnodes 1 \
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
--train_files $train_file \
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
--use_flash_attention $use_flash_attention \
--gradient_checkpointing $gradient_checkpointing"
    [[ -n "$val_seq_length_multiplier" ]] && training_args="$training_args --val_seq_length_multiplier $val_seq_length_multiplier"
    local eff_init_eot="${init_eot_override:-$cfg_init_eot_from_eos}"
    [[ -n "$eff_init_eot" ]] && training_args="$training_args --init_eot_from_eos $eff_init_eot"
    [[ -n "$cfg_loss_reduction" ]] && training_args="$training_args --loss_reduction $cfg_loss_reduction"

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
