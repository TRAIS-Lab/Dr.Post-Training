#!/bin/bash

# =============================================================================
# LR Sweep Script
# =============================================================================
# Performs learning rate search for SFT methods.
# Reads ALL settings from config files (same as train.sh).
# Uses val_loss for LR selection (avoids test set contamination).
# Writes best LR back to the method YAML config.
#
# Usage:
#   bash SFT/train/lr/lr_sweep.sh -c configs/tulu3_tydiqa -m all
#   bash SFT/train/lr/lr_sweep.sh -c configs/tulu3_tydiqa -m Standard-Full --mode grid
#   bash SFT/train/lr/lr_sweep.sh -c configs/alpaca_samsum -m all --sweep_percentage 0.1
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Gradient-Streaming
export PYTHONPATH="$CODE_DIR/Gradient-Streaming:$PYTHONPATH"

SCRIPT_DIR="$CODE_DIR/Gradient-Streaming/SFT/train"

# Fixed training args (same as train.sh)
FIXED_ARGS="--do_train=True \
--do_eval=True \
--use_fast_tokenizer=True \
--logging_steps=1 \
--eval_strategy=steps \
--save_strategy=no \
--bf16=True \
--tf32=False \
--fp16=False \
--overwrite_output_dir=True \
--report_to=none"

# =============================================================================
# CLI
# =============================================================================
config_dir=""
methods=""
dry_run=false

# Sweep-specific overrides
sweep_percentage=""  # Override percentage for faster sweep (empty = use config)
sweep_n_eval=100     # Fewer eval examples for speed
sweep_mode="binary"  # "grid" or "binary"
lr_margin=0.01       # Prefer smaller LR unless larger is this much better

# Grid search defaults
lr_grid="1e-6,5e-6,1e-5,5e-5,1e-4"
lr_grid_lora="5e-5,1e-4,2e-4,5e-4,1e-3"

# Binary search defaults
binary_lr_min="1e-7"
binary_lr_max="1e-3"
binary_lr_min_lora="1e-6"
binary_lr_max_lora="1e-2"
binary_max_iters=8

# Results dir
sweep_results_dir="SFT/train/lr/results"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)      config_dir="$2"; shift 2 ;;
        --methods|-m)         methods="$2"; shift 2 ;;
        --sweep_percentage)   sweep_percentage="$2"; shift 2 ;;
        --sweep_n_eval)       sweep_n_eval="$2"; shift 2 ;;
        --mode)               sweep_mode="$2"; shift 2 ;;
        --lr_margin)          lr_margin="$2"; shift 2 ;;
        --lr_grid)            lr_grid="$2"; shift 2 ;;
        --lr_grid_lora)       lr_grid_lora="$2"; shift 2 ;;
        --binary_lr_min)      binary_lr_min="$2"; shift 2 ;;
        --binary_lr_max)      binary_lr_max="$2"; shift 2 ;;
        --binary_lr_min_lora) binary_lr_min_lora="$2"; shift 2 ;;
        --binary_lr_max_lora) binary_lr_max_lora="$2"; shift 2 ;;
        --binary_max_iters)   binary_max_iters="$2"; shift 2 ;;
        --dry-run)            dry_run=true; shift ;;
        --help|-h)
            cat <<'HELP'
Usage: bash lr_sweep.sh -c <config_dir> -m <methods> [options]

Reads ALL settings from config files (same as train.sh).
Uses val_loss for LR selection. Writes best LR back to method YAML.

Required:
  -c, --config_dir <dir>  Config directory (relative to SFT/train/ or absolute)
  -m, --methods <list>    Methods or categories

Sweep Options:
  --mode <mode>             "grid" or "binary" (default: binary)
  --sweep_percentage <pct>  Override data percentage for faster sweep
  --sweep_n_eval <n>        Eval examples during sweep (default: 100)
  --lr_margin <pct>         Stability margin (default: 0.01)

Grid Search:
  --lr_grid <lrs>           Comma-separated LRs for full (default: 1e-6,5e-6,1e-5,5e-5,1e-4)
  --lr_grid_lora <lrs>      Comma-separated LRs for LoRA (default: 5e-5,1e-4,2e-4,5e-4,1e-3)

Binary Search:
  --binary_lr_min/max       Range for full (default: 1e-7 to 1e-3)
  --binary_lr_min/max_lora  Range for LoRA (default: 1e-6 to 1e-2)
  --binary_max_iters <n>    Max iterations (default: 8)

Examples:
  bash lr_sweep.sh -c configs/tulu3_tydiqa -m all
  bash lr_sweep.sh -c configs/alpaca_samsum -m Standard-Full --mode grid
  bash lr_sweep.sh -c configs/tulu3_tydiqa -m lora --sweep_percentage 0.005
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done

# Validate
if [[ -z "$config_dir" ]] || [[ -z "$methods" ]]; then
    echo "Usage: bash lr_sweep.sh -c <config_dir> -m <methods> [options]"
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
# Extract val_loss from training output (NOT eval_loss — avoids contamination)
# =============================================================================
extract_val_loss() {
    local output_dir="$1"
    local eval_json="$output_dir/evaluation_results.json"
    if [[ -f "$eval_json" ]]; then
        python3 -c "
import json, sys
try:
    with open('$eval_json') as f:
        results = json.load(f)
    if results:
        val_loss = results[-1].get('val_loss')
        if val_loss is not None:
            print(f'{val_loss:.10e}')
            sys.exit(0)
except: pass
sys.exit(1)
" 2>/dev/null && return
    fi
    # Fallback
    echo "999.0"
}

is_valid_number() {
    [[ "$1" =~ ^[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]]
}

cleanup_model_weights() {
    local d="$1"
    [[ ! -d "$d" ]] && return
    rm -f "$d"/model*.safetensors "$d"/pytorch_model.bin \
          "$d"/adapter_model.safetensors "$d"/adapter_model.bin "$d"/adapter_config.json \
          "$d"/tokenizer.json "$d"/tokenizer_config.json "$d"/special_tokens_map.json \
          "$d"/config.json "$d"/generation_config.json "$d"/README.md 2>/dev/null
    rm -rf "$d"/runs 2>/dev/null
}

# =============================================================================
# Run single LR trial (uses full config settings)
# =============================================================================
run_lr_trial() {
    local trial_lr="$1"
    local trial_output_dir="$2"

    local internal_method="NA"
    [[ "$cfg_method" != "Standard" ]] && internal_method="$cfg_method"
    local use_lora="false"
    [[ "$cfg_finetuning" == "LoRA" || "$cfg_finetuning" == "MeSO-LoRA" ]] && use_lora="true"

    # Use sweep overrides where applicable
    local eff_percentage="${sweep_percentage:-$cfg_percentage}"
    local eff_n_eval="$sweep_n_eval"

    local data_dir="$SCRATCH_DIR/Gradient-Streaming/SFT/data"
    local DATA_SEED=$((cfg_seed + 1))
    local PORT=$((29400 + RANDOM % 10000))

    # FSDP for large models
    local fsdp_args=""
    case "$cfg_model" in
        *Llama-2-13b*|*llama-2-13b*) fsdp_args="--fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune" ;;
        *Mistral-7B*|*mistral-7b*)   fsdp_args="--fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune" ;;
    esac

    local cmd="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id=$RANDOM --rdzv_backend c10d --rdzv_endpoint=localhost:$PORT \
-m SFT.train.train \
$FIXED_ARGS $fsdp_args \
--max_seq_length $cfg_max_seq_length \
--lr_scheduler_type $cfg_lr_scheduler_type \
--warmup_ratio $cfg_warmup_ratio \
--weight_decay $cfg_weight_decay \
--num_train_epochs $cfg_num_train_epochs \
--eval_steps 99999 \
--model_name_or_path $cfg_model \
--output_dir $trial_output_dir \
--data_dir $data_dir \
--percentage $eff_percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $cfg_batch_size \
--method $internal_method \
--n_val $cfg_n_val \
--n_eval $eff_n_eval \
--analysis_dataset $cfg_target_task \
--learning_rate $trial_lr \
--gradient_accumulation_steps $cfg_gradient_accumulation_steps \
--seed $cfg_seed \
--optim $cfg_optim \
--selection_frac $cfg_selection_frac \
--val_strategy $cfg_val_strategy \
--use_flash_attention $cfg_use_flash_attention"

    [[ -n "$cfg_subject" ]] && cmd="$cmd --subject $cfg_subject"
    [[ -n "$cfg_train_dataset" ]] && cmd="$cmd --train_dataset_names $cfg_train_dataset"
    [[ -n "$cfg_val_batch_size" ]] && cmd="$cmd --val_batch_size_for_selection $cfg_val_batch_size"

    if [[ "$use_lora" == "true" ]]; then
        cmd="$cmd --lora True --lora_r $cfg_lora_r --lora_alpha $cfg_lora_alpha --lora_dropout $cfg_lora_dropout"
    else
        cmd="$cmd --lora False"
    fi

    [[ -n "$cfg_opt_sparsifier" && "$cfg_opt_sparsifier" != "none" ]] && \
        cmd="$cmd --sparsification $cfg_opt_sparsifier --update_compressor_freq $cfg_update_compressor_freq"
    [[ -n "$cfg_opt_projector" && "$cfg_opt_projector" != "none" ]] && \
        cmd="$cmd --projection $cfg_opt_projector"
    [[ -n "$cfg_score_sparsifier" && "$cfg_score_sparsifier" != "none" ]] && \
        cmd="$cmd --score_compression $cfg_score_sparsifier"
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && \
        cmd="$cmd --use_second_order True"

    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        echo "0.0"
    else
        mkdir -p "$trial_output_dir"
        eval $cmd > "$trial_output_dir/train.log" 2>&1
        extract_val_loss "$trial_output_dir"
    fi
}

# =============================================================================
# Write best LR back to method YAML config
# =============================================================================
update_yaml_lr() {
    local config_file="$1"
    local best_lr="$2"
    local best_val_loss="$3"

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
echo "  LR Sweep"
echo "========================================================"
echo "Config dir:    $config_dir"
echo "Methods:       $resolved_methods ($TOTAL total)"
echo "Mode:          $sweep_mode"
echo "Sweep %:       ${sweep_percentage:-from config}"
echo "N eval:        $sweep_n_eval"
echo "LR margin:     $lr_margin"
echo "Results dir:   $sweep_results_dir"
echo "========================================================"

mkdir -p "$sweep_results_dir"

current=0
for exp_name in "${method_list[@]}"; do
    current=$((current + 1))
    local_config="$config_dir/${exp_name}.yaml"

    if [[ ! -f "$local_config" ]]; then
        echo "ERROR: Config not found: $local_config"
        continue
    fi

    # Load config: reset → defaults → method
    reset_config
    [[ -f "$config_dir/defaults.yaml" ]] && parse_yaml "$config_dir/defaults.yaml"
    parse_yaml "$local_config"

    local use_lora="false"
    [[ "$cfg_finetuning" == "LoRA" || "$cfg_finetuning" == "MeSO-LoRA" ]] && use_lora="true"

    echo ""
    echo "========================================================"
    echo "  [$current/$TOTAL] $exp_name"
    echo "========================================================"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    echo "Task: $cfg_target_task | Data: ${cfg_train_dataset:-default} (${sweep_percentage:-$cfg_percentage})"
    echo "Batch: $cfg_batch_size | n_val: $cfg_n_val"

    best_lr=""
    best_val_loss="999.0"

    # Build config key for results dir
    local config_key="${cfg_train_dataset:-default}_${cfg_target_task}"

    if [[ "$sweep_mode" == "binary" ]]; then
        # === Binary Search (Golden Section) ===
        if [[ "$use_lora" == "true" ]]; then
            cur_min="$binary_lr_min_lora"; cur_max="$binary_lr_max_lora"
        else
            cur_min="$binary_lr_min"; cur_max="$binary_lr_max"
        fi
        echo "Binary search: $cur_min -> $cur_max ($binary_max_iters iters)"

        inv_phi="0.6180339887"
        log_a=$(python3 -c "import math; print(math.log10($cur_min))")
        log_b=$(python3 -c "import math; print(math.log10($cur_max))")
        declare -A lr_losses=()

        for ((iter=1; iter<=binary_max_iters; iter++)); do
            echo ""
            echo "--- Iteration $iter/$binary_max_iters [10^$log_a, 10^$log_b] ---"

            log_c=$(python3 -c "print($log_b - $inv_phi * ($log_b - $log_a))")
            log_d=$(python3 -c "print($log_a + $inv_phi * ($log_b - $log_a))")
            lr_c=$(python3 -c "print(f'{10**$log_c:.2e}')")
            lr_d=$(python3 -c "print(f'{10**$log_d:.2e}')")

            # Evaluate lr_c
            if [[ -z "${lr_losses[$lr_c]}" ]]; then
                trial_dir="$sweep_results_dir/${config_key}/${exp_name}/iter${iter}_lr_${lr_c}"
                loss_c=$(run_lr_trial "$lr_c" "$trial_dir")
                lr_losses[$lr_c]="$loss_c"
                echo "  LR $lr_c -> val_loss: $loss_c"
                cleanup_model_weights "$trial_dir"
            else
                loss_c="${lr_losses[$lr_c]}"
                echo "  LR $lr_c -> val_loss: $loss_c (cached)"
            fi

            # Evaluate lr_d
            if [[ -z "${lr_losses[$lr_d]}" ]]; then
                trial_dir="$sweep_results_dir/${config_key}/${exp_name}/iter${iter}_lr_${lr_d}"
                loss_d=$(run_lr_trial "$lr_d" "$trial_dir")
                lr_losses[$lr_d]="$loss_d"
                echo "  LR $lr_d -> val_loss: $loss_d"
                cleanup_model_weights "$trial_dir"
            else
                loss_d="${lr_losses[$lr_d]}"
                echo "  LR $lr_d -> val_loss: $loss_d (cached)"
            fi

            # Narrow range
            prefer_lower=$(python3 -c "
loss_c, loss_d, margin = float('$loss_c'), float('$loss_d'), float('$lr_margin')
print('0' if loss_d < loss_c * (1 - margin) else '1')
" 2>/dev/null)
            if [[ "$prefer_lower" == "1" ]]; then
                log_b="$log_d"
                echo "  -> Lower half [10^$log_a, 10^$log_b]"
            else
                log_a="$log_c"
                echo "  -> Upper half [10^$log_a, 10^$log_b]"
            fi
        done

        # Find best from all evaluated points
        sorted_lrs=$(python3 -c "
lrs = '${!lr_losses[*]}'.split()
for lr in sorted(lrs, key=float): print(lr)
" 2>/dev/null)
        for lr in $sorted_lrs; do
            loss="${lr_losses[$lr]}"
            is_valid_number "$loss" || continue
            should_update=$(python3 -c "
loss, best, margin = float('$loss'), float('$best_val_loss'), float('$lr_margin')
blr = float('${best_lr:-0}') if '${best_lr:-}' else 0
clr = float('$lr')
if blr == 0: print('1')
elif clr < blr: print('1' if loss <= best * (1 + margin) else '0')
else: print('1' if loss < best * (1 - margin) else '0')
" 2>/dev/null)
            [[ "$should_update" == "1" ]] && best_val_loss="$loss" && best_lr="$lr"
        done

    elif [[ "$sweep_mode" == "grid" ]]; then
        # === Grid Search ===
        if [[ "$use_lora" == "true" ]]; then
            current_grid="$lr_grid_lora"
        else
            current_grid="$lr_grid"
        fi
        echo "Grid search: $current_grid"

        IFS=',' read -ra lr_values <<< "$current_grid"
        for trial_lr in "${lr_values[@]}"; do
            trial_lr=$(echo "$trial_lr" | xargs)
            echo ""
            echo "--- LR: $trial_lr ---"
            trial_dir="$sweep_results_dir/${config_key}/${exp_name}/lr_${trial_lr}"
            val_loss=$(run_lr_trial "$trial_lr" "$trial_dir")
            echo "  val_loss: $val_loss"

            if is_valid_number "$val_loss"; then
                should_update=$(python3 -c "
loss, best, margin = float('$val_loss'), float('$best_val_loss'), float('$lr_margin')
blr = float('${best_lr:-0}') if '${best_lr:-}' else 0
clr = float('$trial_lr')
if blr == 0: print('1')
elif clr < blr: print('1' if loss <= best * (1 + margin) else '0')
else: print('1' if loss < best * (1 - margin) else '0')
" 2>/dev/null)
                [[ "$should_update" == "1" ]] && best_val_loss="$val_loss" && best_lr="$trial_lr"
            fi
            cleanup_model_weights "$trial_dir"
        done
    fi

    echo ""
    echo "Best LR for $exp_name: $best_lr (val_loss: $best_val_loss)"

    # Write back to YAML config
    if [[ -n "$best_lr" && "$dry_run" != "true" ]]; then
        update_yaml_lr "$local_config" "$best_lr" "$best_val_loss"
    fi
done

echo ""
echo "========================================================"
echo "  LR Sweep Complete! ($TOTAL methods)"
echo "========================================================"
echo "Results: $sweep_results_dir"
echo ""
echo "To train with swept LRs:"
echo "  bash SFT/train/train.sh -c ${config_dir#$SCRIPT_DIR/} -m all"
echo "========================================================"
