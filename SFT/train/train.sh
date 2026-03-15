#!/bin/bash

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Source cluster config (skip if already set by submit.sh)
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Gradient-Streaming

export PYTHONPATH="$CODE_DIR/Gradient-Streaming:$PYTHONPATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"

# =============================================================================
# Common settings (shared across all methods, overridden via CLI)
# =============================================================================

export base_training_args="--do_train=True \
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
--report_to=none"

model="meta-llama/Llama-3.2-1B"
data_dir="$SCRATCH_DIR/Gradient-Streaming/SFT/data"
train_dataset=""
percentage=0.05
task="mmlu"
subject="sociology"
seed=42

optim="adamw_torch"
batch_size=8
gradient_accumulation_steps=1
use_flash_attention=true

selection_frac="0.5"
use_second_order=false
n_val=8
val_batch_size="1"
val_strategy="merged_batch"

# LR configuration
lr_config_file="SFT/train/lr/config.json"
lr_override=""
default_lr_full="5e-05"
default_lr_lora="2e-04"

# Evaluation
n_eval=500

# LoRA defaults (can be overridden per config)
lora_r=32
lora_alpha=1
lora_dropout=0.1

# Compressor update frequency (for compression methods)
update_compressor_freq=200

# Curation recording
record_selections=false
record_selections_freq=1

# Multi-method mode
methods=""
dry_run=false

# =============================================================================
# Category mappings (for --methods shorthand)
# =============================================================================
declare -A CATEGORY_METHODS=(
    ["all"]="Standard-Full,Standard-LoRA,Standard-MeSO,Layerwise-Full,Layerwise-LoRA,Subset-Full,Subset-LoRA,Layerwise-MeSO,Subset-MeSO"
    ["baseline"]="Standard-Full,Standard-LoRA"
    ["layerwise"]="Layerwise-Full,Layerwise-LoRA,Layerwise-MeSO"
    ["subset"]="Subset-Full,Subset-LoRA,Subset-MeSO"
    ["full"]="Standard-Full,Standard-MeSO,Layerwise-Full,Subset-Full,Layerwise-MeSO,Subset-MeSO"
    ["lora"]="Standard-LoRA,Layerwise-LoRA,Subset-LoRA"
    ["compression"]="Standard-MeSO,Layerwise-MeSO,Subset-MeSO"
    ["no-compression"]="Standard-Full,Standard-LoRA,Layerwise-Full,Layerwise-LoRA,Subset-Full,Subset-LoRA"
)

# =============================================================================
# Parse CLI arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)        methods="$2"; shift 2 ;;
        --model)          model="$2"; shift 2 ;;
        --task)           task="$2"; shift 2 ;;
        --train)          train_dataset="$2"; shift 2 ;;
        --subject)        subject="$2"; shift 2 ;;
        --batch_size)     batch_size="$2"; shift 2 ;;
        --val_batch_size) val_batch_size="$2"; shift 2 ;;
        --percentage)     percentage="$2"; shift 2 ;;
        --n_val)          n_val="$2"; shift 2 ;;
        --n_eval)         n_eval="$2"; shift 2 ;;
        --lr)             lr_override="$2"; shift 2 ;;
        --lr_config)      lr_config_file="$2"; shift 2 ;;
        --seed)           seed="$2"; shift 2 ;;
        --selection_frac) selection_frac="$2"; shift 2 ;;
        --val_strategy)   val_strategy="$2"; shift 2 ;;
        --use_second_order) use_second_order=true; shift 1 ;;
        --data_dir)       data_dir="$2"; shift 2 ;;
        --gradient_accumulation_steps) gradient_accumulation_steps="$2"; shift 2 ;;
        --record_selections) record_selections=true; shift 1 ;;
        --record_selections_freq) record_selections_freq="$2"; shift 2 ;;
        --dry-run)        dry_run=true; shift ;;
        --list)
            echo "Available methods (config files in $CONFIG_DIR):"
            for f in "$CONFIG_DIR"/*.yaml; do
                basename "$f" .yaml
            done
            echo ""
            echo "Categories: ${!CATEGORY_METHODS[*]}"
            exit 0
            ;;
        --help|-h)
            cat <<'HELP'
Usage: bash train.sh --methods <methods> [options]

Run training with method configs from SFT/train/configs/*.yaml.

Method Curation:
  --methods <list>         Methods or categories (comma-separated)
  --list                   List available methods and exit
  --dry-run                Print commands without executing

  Categories: all, baseline, layerwise, subset, full, lora, compression, no-compression

  Examples:
    --methods all                              Run all methods
    --methods baseline                         Run Standard-Full and Standard-LoRA
    --methods "Layerwise-Full,Subset-Full"       Run specific methods

Experiment Settings:
  --model <path>           Model path (default: meta-llama/Llama-3.2-1B)
  --task <task>            Task: mmlu, bbh, tydiqa, samsum, gsm8k
  --train <dataset>        Training dataset (default: task-based)
  --subject <subject>      MMLU/BBH subject (default: sociology)
  --batch_size <n>         Batch size (default: 8)
  --val_batch_size <n>     Val batch size for curation (default: 1)
  --lr <lr>                Learning rate override
  --lr_config <path>       LR config file (default: SFT/train/lr/config.json)
  --percentage <pct>       Data percentage (default: 0.05)
  --n_val <n>              Validation examples (default: 8)
  --n_eval <n>             Evaluation examples (default: 500)
  --seed <seed>            Random seed (default: 42)
  --selection_frac <frac>  Curation fraction (default: 0.5)
  --val_strategy <strat>   Val strategy (default: merged_batch)
  --use_second_order       Enable second-order curation
HELP
            exit 0
            ;;
        *)
            echo "Unknown argument: $1 (use --help for usage)"
            exit 1
            ;;
    esac
done

model_name=$(basename "$model")

# =============================================================================
# Helper: Read YAML config (lightweight, no pyyaml dependency)
# =============================================================================
read_yaml() {
    # Reads a config YAML and sets cfg_* variables.
    # The config is the source of truth for method hyperparameters.
    local config_file="$1"

    # Reset all config fields
    cfg_method="Standard"
    cfg_finetuning="Full"
    cfg_score_sparsifier=""
    cfg_score_projector=""
    cfg_opt_sparsifier=""
    cfg_opt_projector=""
    cfg_lora_r=""
    cfg_lora_alpha=""
    cfg_lora_dropout=""
    cfg_selection_frac=""
    cfg_n_val=""
    cfg_val_batch_size=""
    cfg_val_strategy=""

    # Parse YAML with one level of nesting support (section.key)
    local section=""
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${line// }" ]] && continue

        local full_key="" val=""
        if [[ "$line" =~ ^[[:space:]] ]]; then
            # Indented: child of current section
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
            score_grad_compression.sparsifier)   cfg_score_sparsifier="$val" ;;
            score_grad_compression.projector)    cfg_score_projector="$val" ;;
            opt_grad_compression.sparsifier)     cfg_opt_sparsifier="$val" ;;
            opt_grad_compression.projector)      cfg_opt_projector="$val" ;;
            selection_frac)                      cfg_selection_frac="$val" ;;
            n_val)                               cfg_n_val="$val" ;;
            val_batch_size)                      cfg_val_batch_size="$val" ;;
            val_strategy)                        cfg_val_strategy="$val" ;;
        esac
    done < "$config_file"

    # Map method name to internal curation name
    case "$cfg_method" in
        Standard) cfg_internal_method="NA" ;;
        *)        cfg_internal_method="$cfg_method" ;;
    esac

    # Derive whether LoRA is enabled from finetuning type
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
        elif [[ -f "$CONFIG_DIR/${item}.yaml" ]]; then
            resolved="${resolved:+$resolved,}$item"
        else
            echo "ERROR: Unknown method or category: $item"
            echo "Use --list to see available methods."
            exit 1
        fi
    done

    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# =============================================================================
# Run a single method from its config file
# =============================================================================
run_method() {
    local exp_name="$1"
    local config_file="$CONFIG_DIR/${exp_name}.yaml"

    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config not found: $config_file"
        return 1
    fi

    # Read method config
    read_yaml "$config_file"

    # Resolve: config yaml > shell default (for method-specific params)
    local eff_selection_frac="${cfg_selection_frac:-$selection_frac}"
    local eff_n_val="${cfg_n_val:-$n_val}"
    local eff_val_batch_size="${cfg_val_batch_size:-$val_batch_size}"
    local eff_val_strategy="${cfg_val_strategy:-$val_strategy}"

    # Look up LR
    local train_str="${train_dataset:-default}"
    local config_key
    if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
        config_key="${train_str}_${task}_${subject}"
    else
        config_key="${train_str}_${task}"
    fi
    local exp_lr=$(lookup_lr "$config_key" "$exp_name" "$cfg_lora")

    # Build job name: use config name directly (e.g., Layerwise-Full)
    local method_str="$exp_name"
    if [[ "$cfg_internal_method" != "NA" ]] && [ "$use_second_order" = true ]; then
        method_str="${method_str}-2nd"
    fi

    local JOB_NAME
    if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
        JOB_NAME="${train_str}_${task}_${subject}-${model_name}-${method_str}-p${percentage}-lr${exp_lr}-b${batch_size}-v${n_val}-s${seed}"
    else
        JOB_NAME="${train_str}_${task}-${model_name}-${method_str}-p${percentage}-lr${exp_lr}-b${batch_size}-v${n_val}-s${seed}"
    fi

    local output_dir=$SCRATCH_DIR/Gradient-Streaming/SFT/${JOB_NAME}
    mkdir -p "$output_dir"

    echo ""
    echo "=============================================="
    echo "  Running: $exp_name"
    echo "=============================================="
    echo "Config: $config_file"
    echo "Job: $JOB_NAME"
    echo "Model: $model | Task: $task | LR: $exp_lr"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    echo "Batch: $batch_size | Val: ${eff_val_batch_size} | Curation: $eff_selection_frac"
    echo "Output: $output_dir"
    echo "=============================================="

    # Build training arguments
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

    local training_args="$exp_base_training_args \
--model_name_or_path $model \
--output_dir $output_dir \
--data_dir $data_dir \
--percentage $percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $batch_size \
--method $cfg_internal_method \
--subject $subject \
--n_val $eff_n_val \
--n_eval $n_eval \
--analysis_dataset $task \
--learning_rate $exp_lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed \
--optim $optim \
--selection_frac $eff_selection_frac \
--val_strategy $eff_val_strategy \
--use_flash_attention $use_flash_attention"

    # Optional: train dataset
    [[ -n "$train_dataset" ]] && training_args="$training_args --train_dataset_names $train_dataset"
    [[ -n "$eff_val_batch_size" ]] && training_args="$training_args --val_batch_size_for_selection $eff_val_batch_size"

    # LoRA (from config, with shell defaults as fallback)
    if [ "$cfg_lora" = true ]; then
        local eff_lora_r="${cfg_lora_r:-$lora_r}"
        local eff_lora_alpha="${cfg_lora_alpha:-$lora_alpha}"
        local eff_lora_dropout="${cfg_lora_dropout:-$lora_dropout}"
        training_args="$training_args --lora True --lora_r $eff_lora_r --lora_alpha $eff_lora_alpha --lora_dropout $eff_lora_dropout"
    else
        training_args="$training_args --lora False"
    fi

    # Update compression (opt_grad_compression → --sparsification for MeSO)
    [[ -n "$cfg_opt_sparsifier" && "$cfg_opt_sparsifier" != "none" ]] && training_args="$training_args --sparsification $cfg_opt_sparsifier --update_compressor_freq $update_compressor_freq"
    [[ -n "$cfg_opt_projector" && "$cfg_opt_projector" != "none" ]] && training_args="$training_args --projection $cfg_opt_projector"

    # Score compression (score_grad_compression → --score_compression for influence scoring)
    [[ -n "$cfg_score_sparsifier" && "$cfg_score_sparsifier" != "none" ]] && training_args="$training_args --score_compression $cfg_score_sparsifier"

    # Second-order curation
    if [[ "$cfg_internal_method" != "NA" ]] && [ "$use_second_order" = true ]; then
        training_args="$training_args --use_second_order True"
    fi

    # Curation recording
    if [ "$record_selections" = true ]; then
        training_args="$training_args --record_selections True --record_selections_freq $record_selections_freq"
    fi

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
if [[ -z "$methods" ]]; then
    echo "Usage: bash train.sh --methods <methods> [options]"
    echo "       bash train.sh --help"
    exit 1
fi

resolved_methods=$(resolve_methods "$methods")
IFS=',' read -ra method_list <<< "$resolved_methods"
TOTAL=${#method_list[@]}

echo ""
echo "========================================================"
echo "  SFT Training"
echo "========================================================"
echo "Methods: $resolved_methods ($TOTAL total)"
echo "Model: $model | Task: $task | Seed: $seed"
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
