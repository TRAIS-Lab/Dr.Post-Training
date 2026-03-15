#!/bin/bash

# Source cluster config
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found. See README.md for cluster setup."; exit 1; }

cd $CODE_DIR/Gradient-Streaming

export PYTHONPATH="$CODE_DIR/Gradient-Streaming:$PYTHONPATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/configs"

# =============================================================================
# Common settings (shared across all methods, overridden via CLI)
# =============================================================================

export base_training_args="--bf16=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--report_to=none \
--enable_eval=True \
--eval_on_step_generations=True"

model="EleutherAI/gpt-neo-2.7B"
reward_model=""  # Will be set based on task if not specified
task="toxicity"
seed=42

# Training settings
epochs=1
batch_size=256
ppo_epochs=4
mini_batch_size=4
kl_estimator="k1"
adap_kl_ctrl=true
target=70.0
target_kl=0.3
early_stopping=true
max_new_tokens="30"
min_new_tokens=0
max_steps=-1
use_flash_attention=true

# Data curation
filter_frac=1.0
use_second_order=false
n_val=1024
val_batch_size=256
val_loss_type="seqloss-reward"
update_compressor_freq=200

# LR/KL configuration
lr_config_file="RLHF/train/config.json"
lr_override=""
lr_vhead_override=""
init_kl_coef_override=""
default_lr="1e-5"
default_lr_vhead="5e-4"
default_init_kl_coef="0.02"

# Evaluation settings
eval_interval=1
n_eval=500
eval_batch_size=256

# LoRA defaults
lora_r=16
lora_alpha=32
lora_target_modules=""

# Multi-method mode
methods=""
dry_run=false

# =============================================================================
# Category mappings (for --methods shorthand)
# =============================================================================
declare -A CATEGORY_METHODS=(
    ["all"]="Standard-LoRA,IIF-LoRA,Layerwise-LoRA,Subset-LoRA"
    ["baseline"]="Standard-LoRA"
    ["iif"]="IIF-LoRA"
    ["layerwise"]="Layerwise-LoRA"
    ["subset"]="Subset-LoRA"
)

# =============================================================================
# Parse CLI arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --methods)          methods="$2"; shift 2 ;;
        --task)             task="$2"; shift 2 ;;
        --model)            model="$2"; shift 2 ;;
        --reward_model)     reward_model="$2"; shift 2 ;;
        --batch_size)       batch_size="$2"; shift 2 ;;
        --lr)               lr_override="$2"; shift 2 ;;
        --lr_vhead)         lr_vhead_override="$2"; shift 2 ;;
        --lr_config)        lr_config_file="$2"; shift 2 ;;
        --init_kl_coef)     init_kl_coef_override="$2"; shift 2 ;;
        --filter_frac)      filter_frac="$2"; shift 2 ;;
        --max_steps)        max_steps="$2"; shift 2 ;;
        --epochs)           epochs="$2"; shift 2 ;;
        --seed)             seed="$2"; shift 2 ;;
        --ppo_epochs)       ppo_epochs="$2"; shift 2 ;;
        --mini_batch_size)  mini_batch_size="$2"; shift 2 ;;
        --kl_estimator)     kl_estimator="$2"; shift 2 ;;
        --target)           target="$2"; shift 2 ;;
        --target_kl)        target_kl="$2"; shift 2 ;;
        --early_stopping)   early_stopping=true; shift 1 ;;
        --max_new_tokens)   max_new_tokens="$2"; shift 2 ;;
        --use_second_order) use_second_order=true; shift 1 ;;
        --n_val)            n_val="$2"; shift 2 ;;
        --val_batch_size)   val_batch_size="$2"; shift 2 ;;
        --val_loss_type)    val_loss_type="$2"; shift 2 ;;
        --eval_interval)    eval_interval="$2"; shift 2 ;;
        --n_eval)           n_eval="$2"; shift 2 ;;
        --eval_batch_size)  eval_batch_size="$2"; shift 2 ;;
        --flash_attention)  use_flash_attention=true; shift 1 ;;
        --no_flash_attention) use_flash_attention=false; shift 1 ;;
        --lora_r)           lora_r="$2"; shift 2 ;;
        --lora_alpha)       lora_alpha="$2"; shift 2 ;;
        --lora_target_modules) lora_target_modules="$2"; shift 2 ;;
        --dry-run)          dry_run=true; shift ;;
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

Run RLHF training with method configs from RLHF/train/configs/*.yaml.

Method Curation:
  --methods <list>         Methods or categories (comma-separated)
  --list                   List available methods and exit
  --dry-run                Print commands without executing

  Categories: all, baseline, iif, layerwise, subset

  Methods: Standard-LoRA, IIF-LoRA, Layerwise-LoRA, Subset-LoRA

  Examples:
    --methods all                                  Run all 4 methods
    --methods baseline                             Run Standard-LoRA
    --methods "Layerwise-LoRA,Subset-LoRA"         Run specific methods

Experiment Settings:
  --model <path>           Policy model (default: EleutherAI/gpt-neo-2.7B)
  --task <task>            Task: toxicity (default: toxicity)
  --reward_model <model>   Reward model (auto-selected if not specified)
  --batch_size <n>         Batch size (default: 256)
  --lr <lr>                Learning rate override
  --lr_vhead <lr>          Value head learning rate override
  --lr_config <path>       LR/KL config file (default: RLHF/train/config.json)
  --init_kl_coef <coef>    Initial KL coefficient override
  --seed <seed>            Random seed (default: 42)
  --max_steps <n>          Max training steps (-1 = use epochs, default: -1)
  --epochs <n>             Training epochs (default: 1)
  --filter_frac <frac>     Fraction of negative samples to drop (default: 1.0)

PPO Settings:
  --ppo_epochs <n>         PPO epochs per batch (default: 4)
  --mini_batch_size <n>    Mini-batch size (default: 4)
  --kl_estimator <mode>    KL estimator: k1, k2, k3 (default: k1)
  --target <val>           Target KL for adaptive controller (default: 70.0)
  --target_kl <val>        Early stopping threshold (default: 0.3)
  --early_stopping         Enable early stopping
  --max_new_tokens <n>     Max new tokens to generate (default: 30)

Validation (for data curation):
  --n_val <n>              Validation samples (default: 1024, 0=self-ref)
  --val_batch_size <n>     Val batch size (default: 256)
  --val_loss_type <type>   Val loss: seqloss-lastadv, seqloss-reward, tokenpg

Evaluation:
  --eval_interval <n>      Evaluate every N steps (0=epoch end only)
  --n_eval <n>             Eval samples (default: 500)
  --eval_batch_size <n>    Eval batch size (default: 256)
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
    cfg_finetuning="LoRA"
    cfg_score_sparsifier=""
    cfg_score_projector=""
    cfg_opt_sparsifier=""
    cfg_opt_projector=""
    cfg_lora_r=""
    cfg_lora_alpha=""

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
            score_grad_compression.sparsifier)   cfg_score_sparsifier="$val" ;;
            score_grad_compression.projector)    cfg_score_projector="$val" ;;
            opt_grad_compression.sparsifier)     cfg_opt_sparsifier="$val" ;;
            opt_grad_compression.projector)      cfg_opt_projector="$val" ;;
        esac
    done < "$config_file"

    # Map method name to internal curation name
    case "$cfg_method" in
        Standard) cfg_internal_method="NA" ;;
        *)        cfg_internal_method="$cfg_method" ;;
    esac
}

# =============================================================================
# Helper: Look up value from config JSON
# =============================================================================
lookup_config_value() {
    local config_key="$1"
    local exp_name="$2"
    local field="$3"
    local override="$4"
    local default="$5"

    if [[ -n "$override" ]]; then
        echo "$override"
        return
    fi

    if [[ -f "$lr_config_file" ]]; then
        local looked_up
        looked_up=$(python3 -c "
import json, sys
try:
    with open('$lr_config_file', 'r') as f:
        config = json.load(f)
    val = config.get('$config_key', {}).get('$exp_name', {}).get('$field')
    if val is not None:
        print(f'{val:.0e}' if isinstance(val, float) and val < 0.001 else val)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)
        if [[ $? -eq 0 ]] && [[ -n "$looked_up" ]]; then
            echo "$looked_up"
            return
        fi
    fi

    echo "$default"
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
# Set task-specific defaults
# =============================================================================
if [[ -z "$reward_model" ]]; then
    case "$task" in
        toxicity)
            reward_model="facebook/roberta-hate-speech-dynabench-r4-target"
            ;;
        *)
            echo "ERROR: Unknown task: $task. Valid: toxicity"
            exit 1
            ;;
    esac
fi

if [[ -z "$max_new_tokens" ]]; then
    max_new_tokens=30
fi

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

    # Look up LR, lr_vhead, init_kl_coef from config JSON
    local config_key="${task}"
    local exp_lr=$(lookup_config_value "$config_key" "$exp_name" "lr" "$lr_override" "$default_lr")
    local exp_lr_vhead=$(lookup_config_value "$config_key" "$exp_name" "lr_vhead" "$lr_vhead_override" "$default_lr_vhead")
    local exp_init_kl_coef=$(lookup_config_value "$config_key" "$exp_name" "init_kl_coef" "$init_kl_coef_override" "$default_init_kl_coef")

    # Build job name
    local method_str="${cfg_method}"
    if [[ "$cfg_internal_method" != "NA" ]] && [ "$use_second_order" = true ]; then
        method_str="${method_str}-2nd"
    fi

    local val_type_short
    case "$val_loss_type" in
        seqloss-lastadv) val_type_short="adv" ;;
        seqloss-reward) val_type_short="rew" ;;
        tokenpg) val_type_short="tpg" ;;
        *) val_type_short="$val_loss_type" ;;
    esac
    local val_str="v${n_val}-${val_type_short}"
    if [[ "$n_val" -gt 0 ]]; then
        val_str="${val_str}-b${val_batch_size}"
    fi

    local JOB_NAME="${task}-${model_name}-${method_str}-${cfg_finetuning}-lr${exp_lr}-b${batch_size}-${val_str}-pe${ppo_epochs}-mb${mini_batch_size}-kl${exp_init_kl_coef}-s${seed}"

    local output_dir=$SCRATCH_DIR/Gradient-Streaming/RLHF/${JOB_NAME}
    mkdir -p "$output_dir"

    echo ""
    echo "=============================================="
    echo "  Running: $exp_name"
    echo "=============================================="
    echo "Config: $config_file"
    echo "Job: $JOB_NAME"
    echo "Model: $model | Task: $task"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    echo "LR: $exp_lr | LR vhead: $exp_lr_vhead | KL: $exp_init_kl_coef"
    echo "Batch: $batch_size | PPO epochs: $ppo_epochs | Mini-batch: $mini_batch_size"
    echo "Output: $output_dir"
    echo "=============================================="

    # Build training arguments
    local training_args="$base_training_args \
--task=$task \
--method=$cfg_internal_method \
--model_name_or_path=$model \
--reward_model_name=$reward_model \
--per_device_train_batch_size=$batch_size \
--learning_rate=$exp_lr \
--learning_rate_vhead=$exp_lr_vhead \
--filter_frac=$filter_frac \
--max_steps=$max_steps \
--num_train_epochs=$epochs \
--seed=$seed \
--ppo_epochs=$ppo_epochs \
--mini_batch_size=$mini_batch_size \
--init_kl_coef=$exp_init_kl_coef \
--kl_estimator=$kl_estimator \
--adap_kl_ctrl=$adap_kl_ctrl \
--target=$target \
--target_kl=$target_kl \
--early_stopping=$early_stopping \
--max_new_tokens=$max_new_tokens \
--min_new_tokens=$min_new_tokens \
--output_dir=$output_dir"

    # LoRA settings (from config, with shell defaults as fallback)
    local eff_lora_r="${cfg_lora_r:-$lora_r}"
    local eff_lora_alpha="${cfg_lora_alpha:-$lora_alpha}"
    case "$cfg_finetuning" in
        LoRA|MeSO-LoRA)
            training_args="$training_args --lora=True --lora_r=$eff_lora_r --lora_alpha=$eff_lora_alpha"
            [[ -n "$lora_target_modules" ]] && training_args="$training_args --lora_target_modules $lora_target_modules"
            ;;
        *)
            training_args="$training_args --lora=False"
            ;;
    esac

    # Flash attention
    if [[ "$use_flash_attention" == "true" ]]; then
        training_args="$training_args --use_flash_attention=True"
    else
        training_args="$training_args --use_flash_attention=False"
    fi

    # MeSO optimizer compression (opt_grad_compression)
    [[ -n "$cfg_opt_sparsifier" && "$cfg_opt_sparsifier" != "none" ]] && training_args="$training_args --sparsification=$cfg_opt_sparsifier --update_compressor_freq=$update_compressor_freq"
    [[ -n "$cfg_opt_projector" && "$cfg_opt_projector" != "none" ]] && training_args="$training_args --projection=$cfg_opt_projector"

    # Score-only compression (score_grad_compression)
    [[ -n "$cfg_score_sparsifier" && "$cfg_score_sparsifier" != "none" ]] && training_args="$training_args --score_compression=$cfg_score_sparsifier"

    # Second-order curation
    if [[ "$cfg_internal_method" != "NA" ]] && [ "$use_second_order" = true ]; then
        training_args="$training_args --use_second_order=True"
    fi

    # Validation settings (for data curation)
    training_args="$training_args --n_val=$n_val --val_batch_size=$val_batch_size --val_loss_type=$val_loss_type"

    # Evaluation settings
    training_args="$training_args --eval_interval=$eval_interval --n_eval=$n_eval --eval_batch_size=$eval_batch_size"

    # Log file
    training_args="$training_args 2>&1 | tee $output_dir/train.log"

    local cmd="python RLHF/train/train.py $training_args"

    echo "Running command:"
    echo "$cmd"
    echo ""

    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] Would execute above command"
    else
        eval "$cmd"
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
echo "  RLHF Training"
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
