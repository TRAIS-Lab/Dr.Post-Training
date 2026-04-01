#!/bin/bash

# =============================================================================
# LR Sweep Collect — Gather results and pick best LR per method
# =============================================================================
# Scans output directories from lr_sweep_submit.sh, extracts eval_loss,
# picks the best LR (smallest within margin of best loss), and writes
# the learning_rate back to each method's YAML config.
#
# Usage:
#   bash lr_sweep_collect.sh -c configs/tulu3_tydiqa -m all
#   bash lr_sweep_collect.sh -c configs/tulu3_tydiqa -m Standard-Full --lr_margin 0.02
#   bash lr_sweep_collect.sh -c configs/tulu3_tydiqa -m all --dry-run
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Dr.Post-Training
export PYTHONPATH="$CODE_DIR/Dr.Post-Training:$PYTHONPATH"

SCRIPT_DIR="$CODE_DIR/Dr.Post-Training/SFT/train"

# =============================================================================
# CLI
# =============================================================================
config_dir=""
methods=""
lr_margin=0.01
dry_run=false
cleanup=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)  config_dir="$2"; shift 2 ;;
        --methods|-m)     methods="$2"; shift 2 ;;
        --lr_margin)      lr_margin="$2"; shift 2 ;;
        --no-cleanup)     cleanup=false; shift ;;
        --dry-run)        dry_run=true; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash lr_sweep_collect.sh -c <config_dir> -m <methods> [options]

Collects eval_loss from LR sweep output dirs, picks best LR per method,
and writes learning_rate back to the method YAML config.

Required:
  -c, --config_dir <dir>  Config directory (relative to SFT/train/ or absolute)
  -m, --methods <list>    Methods or categories (all, baseline, lora, etc.)

Options:
  --lr_margin <pct>       Stability margin — prefer smallest LR within this
                          fraction of the best loss (default: 0.01 = 1%)
  --no-cleanup            Keep model weights in sweep dirs (default: clean up)
  --dry-run               Print best LRs without updating YAML configs

Examples:
  bash lr_sweep_collect.sh -c configs/tulu3_tydiqa -m all
  bash lr_sweep_collect.sh -c configs/tulu3_tydiqa -m all --lr_margin 0.02
  bash lr_sweep_collect.sh -c configs/tulu3_tydiqa -m Standard-Full --dry-run
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done

# Validate
if [[ -z "$config_dir" ]] || [[ -z "$methods" ]]; then
    echo "Usage: bash lr_sweep_collect.sh -c <config_dir> -m <methods> [options]"
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
# Method resolution
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
# Helpers
# =============================================================================
cleanup_model_weights() {
    local d="$1"
    [[ ! -d "$d" ]] && return
    rm -f "$d"/model*.safetensors "$d"/pytorch_model.bin \
          "$d"/adapter_model.safetensors "$d"/adapter_model.bin "$d"/adapter_config.json \
          "$d"/tokenizer.json "$d"/tokenizer_config.json "$d"/special_tokens_map.json \
          "$d"/config.json "$d"/generation_config.json "$d"/README.md 2>/dev/null
    rm -rf "$d"/runs 2>/dev/null
}

update_yaml_lr() {
    local config_file="$1"
    local best_lr="$2"

    python3 << EOF
import re

config_file = "$config_file"
best_lr = float("$best_lr")

with open(config_file, 'r') as f:
    content = f.read()

# Format LR
lr_str = f"{best_lr:.2e}" if best_lr < 0.001 else f"{best_lr}"

# Replace or insert learning_rate line
if re.search(r'^learning_rate:', content, re.MULTILINE):
    content = re.sub(r'^learning_rate:.*$', f'learning_rate: {lr_str}', content, flags=re.MULTILINE)
else:
    # Insert after finetuning line
    content = re.sub(r'(^finetuning:.*$)', f'\\1\nlearning_rate: {lr_str}', content, count=1, flags=re.MULTILINE)

with open(config_file, 'w') as f:
    f.write(content)

print(f"Updated {config_file}: learning_rate={lr_str}")
EOF
}

# =============================================================================
# Main
# =============================================================================
resolved_methods=$(resolve_methods "$methods")
IFS=',' read -ra method_list <<< "$resolved_methods"
TOTAL=${#method_list[@]}

echo ""
echo "========================================================"
echo "  LR Sweep Collect"
echo "========================================================"
echo "Config dir:    $config_dir"
echo "Methods:       $resolved_methods ($TOTAL total)"
echo "LR margin:     $lr_margin"
echo "Cleanup:       $cleanup"
echo "========================================================"

for exp_name in "${method_list[@]}"; do
    local_config="$config_dir/${exp_name}.yaml"

    if [[ ! -f "$local_config" ]]; then
        echo "ERROR: Config not found: $local_config"
        continue
    fi

    # Load config to build the output dir pattern
    reset_config
    [[ -f "$config_dir/defaults.yaml" ]] && parse_yaml "$config_dir/defaults.yaml"
    parse_yaml "$local_config"

    model_name=$(basename "$cfg_model")
    train_str="${cfg_train_dataset:-default}"

    echo ""
    echo "========================================================"
    echo "  $exp_name"
    echo "========================================================"

    # Find all sweep output dirs for this method
    # Pattern: {train}_{task}-{model}-{method}-p{pct}-lr{LR}-b{batch}-v{nval}-s{seed}
    # The LR varies, everything else is fixed from config
    pattern="${train_str}_${cfg_target_task}"
    [[ -n "$cfg_subject" ]] && pattern="${pattern}_${cfg_subject}"
    pattern="${pattern}-${model_name}-${exp_name}-p${cfg_percentage}-lr*-b${cfg_batch_size}-v${cfg_n_val}-s${cfg_seed}"

    sweep_dirs=$(find "$SCRATCH_DIR/Dr.Post-Training/SFT/" -maxdepth 1 -type d -name "$pattern" 2>/dev/null | sort)

    if [[ -z "$sweep_dirs" ]]; then
        echo "  No sweep directories found matching: $pattern"
        echo "  (looked in $SCRATCH_DIR/Dr.Post-Training/SFT/)"
        continue
    fi

    # Collect eval_loss from each directory
    declare -A lr_losses=()
    n_found=0
    n_missing=0

    for dir in $sweep_dirs; do
        # Extract LR from directory name
        dir_name=$(basename "$dir")
        lr=$(echo "$dir_name" | grep -oP '(?<=-lr).*?(?=-b)')
        if [[ -z "$lr" ]]; then
            continue
        fi

        eval_json="$dir/evaluation_results.json"
        if [[ ! -f "$eval_json" ]]; then
            echo "  WARNING: No evaluation_results.json in $dir_name"
            n_missing=$((n_missing + 1))
            continue
        fi

        # Extract eval_loss (last entry)
        eval_loss=$(python3 -c "
import json, sys
try:
    with open('$eval_json') as f:
        results = json.load(f)
    if results:
        loss = results[-1].get('eval_loss')
        if loss is not None:
            print(f'{loss:.10e}')
            sys.exit(0)
except Exception: pass
sys.exit(1)
" 2>/dev/null)

        if [[ $? -eq 0 && -n "$eval_loss" ]]; then
            lr_losses[$lr]="$eval_loss"
            n_found=$((n_found + 1))
            echo "  LR=$lr  eval_loss=$eval_loss"
        else
            echo "  WARNING: Could not extract eval_loss from $dir_name"
            n_missing=$((n_missing + 1))
        fi
    done

    echo "  Found: $n_found results, Missing: $n_missing"

    if [[ $n_found -eq 0 ]]; then
        echo "  SKIP: No valid results found"
        continue
    fi

    # Pick best LR: smallest LR within margin of best loss
    best_result=$(python3 -c "
import sys
inf = float('inf')

lr_loss_pairs = []
$(for lr in "${!lr_losses[@]}"; do echo "lr_loss_pairs.append(($lr, ${lr_losses[$lr]}))"; done)

margin = $lr_margin

# Sort by LR (ascending)
lr_loss_pairs.sort(key=lambda x: x[0])

# Find the absolute best loss
best_loss = min(loss for _, loss in lr_loss_pairs)

# Pick smallest LR within margin of best
for lr, loss in lr_loss_pairs:
    if loss <= best_loss * (1 + margin):
        print(f'{lr:.2e} {loss:.10e}')
        sys.exit(0)

# Fallback: absolute best
for lr, loss in lr_loss_pairs:
    if loss == best_loss:
        print(f'{lr:.2e} {loss:.10e}')
        sys.exit(0)
" 2>/dev/null)

    best_lr=$(echo "$best_result" | awk '{print $1}')
    best_loss=$(echo "$best_result" | awk '{print $2}')

    echo ""
    echo "  >>> Best LR: $best_lr (eval_loss: $best_loss, margin: $lr_margin)"

    # Update YAML config
    if [[ -n "$best_lr" && "$dry_run" != "true" ]]; then
        update_yaml_lr "$local_config" "$best_lr" "$best_loss"
    elif [[ "$dry_run" == "true" ]]; then
        echo "  [DRY-RUN] Would update $local_config with learning_rate: $best_lr"
    fi

    # Clean up model weights from sweep dirs
    if [[ "$cleanup" == "true" && "$dry_run" != "true" ]]; then
        for dir in $sweep_dirs; do
            cleanup_model_weights "$dir"
        done
        echo "  Cleaned up model weights from sweep dirs"
    fi

    unset lr_losses
done

echo ""
echo "========================================================"
echo "  LR Sweep Collection Complete! ($TOTAL methods)"
echo "========================================================"
echo ""
echo "To train with swept LRs:"
echo "  bash SFT/train/train.sh -c ${config_dir#$SCRIPT_DIR/} -m all"
echo "========================================================"
