#!/bin/bash

# =============================================================================
# LR Sweep Submit — Parallel Dense-Grid LR Search via SLURM
# =============================================================================
# Submits all LR trials as independent SLURM jobs (1 GPU each).
# Each job trains 1 epoch with the config's settings and evaluates on the LR split.
#
# Usage:
#   bash lr_sweep_submit.sh -c configs/tulu3_tydiqa -m all
#   bash lr_sweep_submit.sh -c configs/tulu3_tydiqa -m Standard-Full --n_lrs 10
#   bash lr_sweep_submit.sh -c configs/tulu3_tydiqa -m lora --dry-run
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Dr.Post-Training
export PYTHONPATH="$CODE_DIR/Dr.Post-Training:$PYTHONPATH"

SCRIPT_DIR="$CODE_DIR/Dr.Post-Training/SFT/train"
SUBMIT_SCRIPT="$CODE_DIR/Dr.Post-Training/submit.sh"

# =============================================================================
# CLI
# =============================================================================
config_dir=""
methods=""
dry_run=false
n_lrs=20

# SLURM defaults for sweep jobs
sweep_gpus="${SWEEP_GPUS:-1}"
sweep_mem="${SWEEP_MEM:-64g}"
sweep_time="${SWEEP_TIME:-1:00:00}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)  config_dir="$2"; shift 2 ;;
        --methods|-m)     methods="$2"; shift 2 ;;
        --n_lrs)          n_lrs="$2"; shift 2 ;;
        --dry-run)        dry_run=true; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash lr_sweep_submit.sh -c <config_dir> -m <methods> [options]

Submits parallel SLURM jobs for dense LR grid search.
Each job trains 1 epoch and evaluates on the LR split (not test).

Required:
  -c, --config_dir <dir>  Config directory (relative to SFT/train/ or absolute)
  -m, --methods <list>    Methods or categories (all, baseline, lora, etc.)

Options:
  --n_lrs <n>             Number of log-spaced LR values (default: 20)
  --dry-run               Print commands without submitting

Environment overrides:
  SWEEP_GPUS=1            GPUs per job (default: 1)
  SWEEP_MEM=64g           Memory per job (default: 64g)
  SWEEP_TIME=2:00:00      Time limit per job (default: 2:00:00)

Examples:
  bash lr_sweep_submit.sh -c configs/tulu3_tydiqa -m all
  bash lr_sweep_submit.sh -c configs/tulu3_tydiqa -m Standard-Full --n_lrs 10
  bash lr_sweep_submit.sh -c configs/alpaca_samsum -m lora --dry-run
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done

# Validate
if [[ -z "$config_dir" ]] || [[ -z "$methods" ]]; then
    echo "Usage: bash lr_sweep_submit.sh -c <config_dir> -m <methods> [options]"
    exit 1
fi

[[ "$config_dir" != /* ]] && config_dir="$SCRIPT_DIR/$config_dir"
if [[ ! -d "$config_dir" ]]; then
    echo "ERROR: Config directory not found: $config_dir"
    exit 1
fi

# =============================================================================
# Config parser (same as train.sh)
# =============================================================================
reset_config() {
    cfg_method="Standard"; cfg_finetuning="Full"
    cfg_score_sparsifier=""; cfg_score_projector=""
    cfg_opt_sparsifier=""; cfg_opt_projector=""
    cfg_selection_frac="0.5"; cfg_n_val="8"
    cfg_val_batch_size="1"; cfg_val_strategy="merged_batch"
    cfg_use_second_order="false"
    cfg_lora_r="32"; cfg_lora_alpha="1"; cfg_lora_dropout="0.1"
    cfg_model="meta-llama/Llama-3.2-1B"; cfg_seed="42"
    cfg_batch_size="8"; cfg_gradient_accumulation_steps="1"
    cfg_n_eval="500"; cfg_optim="adamw_torch"
    cfg_use_flash_attention="true"; cfg_learning_rate=""
    cfg_max_seq_length="512"; cfg_lr_scheduler_type="linear"
    cfg_warmup_ratio="0.03"; cfg_weight_decay="0.0"
    cfg_num_train_epochs="1"; cfg_eval_steps="50"
    cfg_train_dataset=""; cfg_target_task=""; cfg_subject=""
    cfg_percentage=""; cfg_update_compressor_freq="200"
}

parse_yaml() {
    local file="$1"
    local section=""
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue
        local key="" val=""
        if [[ "$line" =~ ^[[:space:]] ]]; then
            val=$(echo "$line" | cut -d: -f2- | xargs | sed 's/^"//;s/"$//' | sed "s/^'//;s/'$//")
            key="${section}.$(echo "$line" | cut -d: -f1 | xargs)"
        else
            local top_key top_val
            top_key=$(echo "$line" | cut -d: -f1 | xargs)
            top_val=$(echo "$line" | cut -d: -f2- | xargs | sed 's/^"//;s/"$//' | sed "s/^'//;s/'$//")
            if [[ -z "$top_val" ]]; then section="$top_key"; continue; fi
            section=""; key="$top_key"; val="$top_val"
        fi
        case "$key" in
            method)                              cfg_method="$val" ;;
            finetuning)                          cfg_finetuning="$val" ;;
            score_grad_compression.sparsifier)   cfg_score_sparsifier="$val" ;;
            score_grad_compression.projector)    cfg_score_projector="$val" ;;
            opt_grad_compression.sparsifier)     cfg_opt_sparsifier="$val" ;;
            opt_grad_compression.projector)      cfg_opt_projector="$val" ;;
            selection_frac)   cfg_selection_frac="$val" ;;
            n_val)            cfg_n_val="$val" ;;
            val_batch_size)   cfg_val_batch_size="$val" ;;
            val_strategy)     cfg_val_strategy="$val" ;;
            use_second_order) cfg_use_second_order="$val" ;;
            lora_r)           cfg_lora_r="$val" ;;
            lora_alpha)       cfg_lora_alpha="$val" ;;
            lora_dropout)     cfg_lora_dropout="$val" ;;
            model)            cfg_model="$val" ;;
            seed)             cfg_seed="$val" ;;
            batch_size)       cfg_batch_size="$val" ;;
            gradient_accumulation_steps) cfg_gradient_accumulation_steps="$val" ;;
            n_eval)           cfg_n_eval="$val" ;;
            optim)            cfg_optim="$val" ;;
            use_flash_attention) cfg_use_flash_attention="$val" ;;
            learning_rate)    cfg_learning_rate="$val" ;;
            max_seq_length)   cfg_max_seq_length="$val" ;;
            lr_scheduler_type) cfg_lr_scheduler_type="$val" ;;
            warmup_ratio)     cfg_warmup_ratio="$val" ;;
            weight_decay)     cfg_weight_decay="$val" ;;
            num_train_epochs) cfg_num_train_epochs="$val" ;;
            eval_steps)       cfg_eval_steps="$val" ;;
            train_dataset)    cfg_train_dataset="$val" ;;
            target_task)      cfg_target_task="$val" ;;
            subject)          cfg_subject="$val" ;;
            percentage)       cfg_percentage="$val" ;;
            update_compressor_freq) cfg_update_compressor_freq="$val" ;;
        esac
    done < "$file"
}

# =============================================================================
# Method resolution (auto-discover from config dir)
# =============================================================================
resolve_methods() {
    local input="$1"
    local available=()
    for f in "$config_dir"/*.yaml; do
        [[ ! -f "$f" ]] && continue
        local name=$(basename "$f" .yaml)
        [[ "$name" == "defaults" ]] && continue
        available+=("$name")
    done
    local resolved=""
    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)
        case "$item" in
            all)            for m in "${available[@]}"; do resolved="${resolved:+$resolved,}$m"; done ;;
            baseline)       for m in "${available[@]}"; do [[ "$m" == Standard-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            layerwise)      for m in "${available[@]}"; do [[ "$m" == Layerwise-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            subset)         for m in "${available[@]}"; do [[ "$m" == Subset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            lora)           for m in "${available[@]}"; do [[ "$m" == *-LoRA ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            compression)    for m in "${available[@]}"; do [[ "$m" == *-MeSO ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            *)
                if [[ -f "$config_dir/${item}.yaml" ]]; then
                    resolved="${resolved:+$resolved,}$item"
                else
                    echo "ERROR: Unknown method: $item"; exit 1
                fi ;;
        esac
    done
    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# =============================================================================
# Generate log-spaced LR grid
# =============================================================================
generate_lr_grid() {
    local lr_min="$1"
    local lr_max="$2"
    local n="$3"
    python3 -c "
import numpy as np
lrs = np.logspace(np.log10($lr_min), np.log10($lr_max), $n)
for lr in lrs:
    print(f'{lr:.2e}')
"
}

# =============================================================================
# Main
# =============================================================================
resolved_methods=$(resolve_methods "$methods")
IFS=',' read -ra method_list <<< "$resolved_methods"
TOTAL=${#method_list[@]}

echo ""
echo "========================================================"
echo "  LR Sweep Submit (Parallel SLURM)"
echo "========================================================"
echo "Config dir:    $config_dir"
echo "Methods:       $resolved_methods ($TOTAL total)"
echo "N LRs:         $n_lrs"
echo "SLURM:         GPUs=$sweep_gpus, MEM=$sweep_mem, TIME=$sweep_time"
echo "========================================================"

total_jobs=0
for exp_name in "${method_list[@]}"; do
    local_config="$config_dir/${exp_name}.yaml"

    if [[ ! -f "$local_config" ]]; then
        echo "ERROR: Config not found: $local_config"
        continue
    fi

    # Load config: reset -> defaults -> method
    reset_config
    [[ -f "$config_dir/defaults.yaml" ]] && parse_yaml "$config_dir/defaults.yaml"
    parse_yaml "$local_config"

    use_lora="false"
    [[ "$cfg_finetuning" == "LoRA" || "$cfg_finetuning" == "MeSO-LoRA" ]] && use_lora="true"

    # Determine LR range based on finetuning type
    if [[ "$use_lora" == "true" ]]; then
        lr_min="1e-5"; lr_max="1e-1"
    else
        lr_min="1e-7"; lr_max="1e-3"
    fi

    # Generate LR grid
    lr_values=$(generate_lr_grid "$lr_min" "$lr_max" "$n_lrs")

    echo ""
    echo "========================================================"
    echo "  $exp_name ($cfg_method-$cfg_finetuning)"
    echo "========================================================"
    echo "LR range: $lr_min -> $lr_max ($n_lrs values)"
    echo "Task: $cfg_target_task | Data: ${cfg_train_dataset:-default} ($cfg_percentage)"

    config_rel="${config_dir#$SCRIPT_DIR/}"

    # Suppress email notifications for sweep jobs
    unset SLURM_MAIL_USER

    for lr in $lr_values; do
        job_name="lr_sweep_${exp_name}_${lr}"

        if [[ "$dry_run" == "true" ]]; then
            echo "[DRY-RUN] GPUS=$sweep_gpus MEM=$sweep_mem TIME=$sweep_time JOB_NAME=$job_name" \
                 "$SUBMIT_SCRIPT SFT/train/train.sh" \
                 "-c $config_rel -m $exp_name --lr $lr --eval_split lr"
        else
            GPUS=$sweep_gpus MEM=$sweep_mem TIME=$sweep_time JOB_NAME="$job_name" \
                "$SUBMIT_SCRIPT" SFT/train/train.sh \
                -c "$config_rel" -m "$exp_name" --lr "$lr" --eval_split lr
        fi
        total_jobs=$((total_jobs + 1))
    done
done

echo ""
echo "========================================================"
echo "  Submitted $total_jobs jobs ($TOTAL methods x $n_lrs LRs)"
echo "========================================================"
echo ""
echo "After all jobs complete, collect results:"
echo "  bash SFT/train/lr/lr_sweep_collect.sh -c ${config_dir#$SCRIPT_DIR/} -m $methods"
echo "========================================================"
