#!/bin/bash
#
# RLVR Training Runner
#
# All experiment settings live in config files.
# Each config directory has defaults.yaml (shared settings) + per-method configs.
#
# Usage: bash train.sh -c <config_dir> -m <methods> [options]
#

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add project root to PYTHONPATH
export PYTHONPATH=$SCRIPT_DIR:$REPO_ROOT:${PYTHONPATH}

# Unset ROCR_VISIBLE_DEVICES to avoid conflict with CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES

# =============================================================================
# GPU Configuration
# =============================================================================
if [ -z "$N_GPUS" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        N_GPUS=1
    fi
fi

# =============================================================================
# Data paths
# =============================================================================
DATA_DIR=$SCRATCH_DIR/Dr.Post-Training/RLVR/data
OUTPUT_BASE=$SCRATCH_DIR/Dr.Post-Training/RLVR/output

# =============================================================================
# CLI
# =============================================================================
config_dir=""
methods=""
seed_override=""
lr_override=""
dry_run=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)      config_dir="$2"; shift 2 ;;
        --methods|-m)         methods="$2"; shift 2 ;;
        --seed)               seed_override="$2"; shift 2 ;;
        --lr)                 lr_override="$2"; shift 2 ;;
        --dry-run)            dry_run=true; shift ;;
        --list)
            dir="${config_dir:-configs/math}"
            [[ "$dir" != /* ]] && dir="$SCRIPT_DIR/$dir"
            echo "Available methods in $dir:"
            for f in "$dir"/*.yaml; do
                [[ ! -f "$f" ]] && continue
                name=$(basename "$f" .yaml)
                [[ "$name" != "defaults" ]] && echo "  $name"
            done
            echo ""
            echo "Categories: all, standard, layerwise, subset, full"
            exit 0
            ;;
        --help|-h)
            cat <<'HELP'
Usage: bash train.sh -c <config_dir> -m <methods> [options]

All experiment settings live in config files.
Each config directory has a defaults.yaml for shared settings, plus per-method configs.
Config naming convention: Method-Finetuning.yaml (e.g., Layerwise-Full.yaml)

Required:
  -c, --config_dir <dir>  Config directory (relative to RLVR/ or absolute)
  -m, --methods <list>    Methods or categories (comma-separated)

Optional:
  --seed <seed>           Override seed from config
  --lr <lr>               Override learning rate from config
  --dry-run               Print commands without executing
  --list                  List available methods and exit

Categories: all, standard, layerwise, subset, full

Examples:
  bash train.sh -c configs/math -m all
  bash train.sh -c configs/math -m Layerwise-Full --seed 123
  bash train.sh -c configs/math -m "Standard-Full,Layerwise-Full" --dry-run
  bash train.sh -c configs/math -m layerwise --dry-run
HELP
            exit 0
            ;;
        *) echo "Unknown argument: $1 (use --help)"; exit 1 ;;
    esac
done

# Validate required args
if [[ -z "$config_dir" ]] || [[ -z "$methods" ]]; then
    echo "Usage: bash train.sh -c <config_dir> -m <methods> [options]"
    echo "       bash train.sh --help"
    exit 1
fi

# Resolve config dir to absolute path
[[ "$config_dir" != /* ]] && config_dir="$SCRIPT_DIR/$config_dir"

if [[ ! -d "$config_dir" ]]; then
    echo "ERROR: Config directory not found: $config_dir"
    exit 1
fi

# =============================================================================
# Config parser (reuses SFT/RLHF pattern)
# =============================================================================
reset_config() {
    # Method
    cfg_method="Standard"
    cfg_finetuning="Full"

    # Model
    cfg_model="Qwen/Qwen3-1.7B-Base"

    # Training
    cfg_seed="42"
    cfg_train_batch_size="128"
    cfg_max_prompt_length="1024"
    cfg_max_response_length="2048"
    cfg_learning_rate="1e-6"
    cfg_total_epochs="3"
    cfg_ppo_mini_batch_size="32"
    cfg_ppo_micro_batch_size_per_gpu="2"
    cfg_kl_loss_coef="0.001"
    cfg_rollout_n="8"
    cfg_gpu_memory_utilization="0.6"

    # Evaluation & checkpointing
    cfg_test_freq="3"
    cfg_save_freq="1000"

    # Selection
    cfg_selection_frac="1.0"
    cfg_val_pool_size="512"
    cfg_val_batch_size="64"
    cfg_val_loss_type="reward"
    cfg_val_source="from_train"
    cfg_refresh_freq="1"
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
            if [[ -z "$top_val" ]]; then
                section="$top_key"
                continue
            fi
            key="$top_key"
            val="$top_val"
        fi

        case "$key" in
            method)                          cfg_method="$val" ;;
            finetuning)                      cfg_finetuning="$val" ;;
            model)                           cfg_model="$val" ;;
            seed)                            cfg_seed="$val" ;;
            train_batch_size)                cfg_train_batch_size="$val" ;;
            max_prompt_length)               cfg_max_prompt_length="$val" ;;
            max_response_length)             cfg_max_response_length="$val" ;;
            learning_rate)                   cfg_learning_rate="$val" ;;
            total_epochs)                    cfg_total_epochs="$val" ;;
            ppo_mini_batch_size)             cfg_ppo_mini_batch_size="$val" ;;
            ppo_micro_batch_size_per_gpu)    cfg_ppo_micro_batch_size_per_gpu="$val" ;;
            kl_loss_coef)                    cfg_kl_loss_coef="$val" ;;
            rollout_n)                       cfg_rollout_n="$val" ;;
            gpu_memory_utilization)          cfg_gpu_memory_utilization="$val" ;;
            test_freq)                       cfg_test_freq="$val" ;;
            save_freq)                       cfg_save_freq="$val" ;;
            selection_frac)                  cfg_selection_frac="$val" ;;
            val_pool_size)                   cfg_val_pool_size="$val" ;;
            val_batch_size)                  cfg_val_batch_size="$val" ;;
            val_loss_type)                   cfg_val_loss_type="$val" ;;
            val_source)                      cfg_val_source="$val" ;;
            refresh_freq)                    cfg_refresh_freq="$val" ;;
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
            all)       for m in "${available[@]}"; do resolved="${resolved:+$resolved,}$m"; done ;;
            standard)  for m in "${available[@]}"; do [[ "$m" == Standard-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            layerwise) for m in "${available[@]}"; do [[ "$m" == Layerwise-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            subset)    for m in "${available[@]}"; do [[ "$m" == Subset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            full)      for m in "${available[@]}"; do [[ "$m" == *-Full ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            *)
                if [[ -f "$config_dir/${item}.yaml" ]]; then
                    resolved="${resolved:+$resolved,}$item"
                else
                    echo "ERROR: Unknown method or category: $item"
                    echo "Available: ${available[*]}"
                    exit 1
                fi ;;
        esac
    done

    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# =============================================================================
# Run a single method
# =============================================================================
run_method() {
    local exp_name="$1"
    local config_file="$config_dir/${exp_name}.yaml"

    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config not found: $config_file"
        return 1
    fi

    # Load config: reset → defaults → method
    reset_config
    [[ -f "$config_dir/defaults.yaml" ]] && parse_yaml "$config_dir/defaults.yaml"
    parse_yaml "$config_file"

    # CLI overrides
    [[ -n "$seed_override" ]] && cfg_seed="$seed_override"
    [[ -n "$lr_override" ]] && cfg_learning_rate="$lr_override"

    # Derived: selection.enable based on method
    local selection_enabled="False"
    [[ "$cfg_method" != "Standard" ]] && selection_enabled="True"

    # Derived: val_prompts_path from val_source
    local val_prompts_path
    case "$cfg_val_source" in
        from_train) val_prompts_path="$DATA_DIR/math/val_from_train.parquet" ;;
        from_test)  val_prompts_path="$DATA_DIR/math/val_from_test.parquet" ;;
        *)          val_prompts_path="$cfg_val_source" ;;  # Treat as literal path
    esac

    # Data paths
    local math_train_path=$DATA_DIR/math/train.parquet
    local math_test_path=$DATA_DIR/math/test.parquet
    local math_test_cleaned_path=$DATA_DIR/math/test_cleaned.parquet
    local val_from_test_path=$DATA_DIR/math/val_from_test.parquet
    local val_from_train_path=$DATA_DIR/math/val_from_train.parquet

    # Build experiment name: Qwen3-1.7B_{method}_s{seed}_{val_loss_type}
    # Strip "-Base"/"-Instruct" suffix from model name for cleaner wandb names
    local model_name=$(basename "$cfg_model" | sed 's/-Base$//' | sed 's/-Instruct$//')
    local EXP_NAME
    if [[ "$selection_enabled" == "True" ]]; then
        EXP_NAME="${model_name}_${cfg_method}_s${cfg_seed}_${cfg_val_loss_type}"
    else
        EXP_NAME="${model_name}_s${cfg_seed}"
    fi
    local OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
    local HYDRA_DIR="${OUTPUT_BASE}/hydra/${EXP_NAME}"

    echo ""
    echo "=============================================="
    echo "  Running: $exp_name"
    echo "=============================================="
    echo "Model: $cfg_model"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning (selection=$selection_enabled)"
    echo "LR: $cfg_learning_rate | Seed: $cfg_seed"
    echo "Batch: $cfg_train_batch_size | Epochs: $cfg_total_epochs"
    echo "Val loss type: $cfg_val_loss_type | Val source: $cfg_val_source"
    echo "Output: $OUTPUT_DIR"
    echo "=============================================="

    # Auto-prepare data if missing (only for selection methods)
    if [[ "$selection_enabled" == "True" ]]; then
        if [ ! -f "$val_from_test_path" ]; then
            echo "Generating val_from_test.parquet and test_cleaned.parquet..."
            python3 $REPO_ROOT/RLVR/data/prepare_data.py \
                --test_data "$math_test_path" \
                --output "$val_from_test_path" \
                --output_test "$math_test_cleaned_path" \
                --num_samples "$cfg_val_pool_size" \
                --seed "$cfg_seed"
        fi
        if [ ! -f "$val_from_train_path" ]; then
            echo "Generating val_from_train.parquet..."
            python3 $REPO_ROOT/RLVR/data/prepare_data.py \
                --train_data "$math_train_path" \
                --output "$val_from_train_path" \
                --num_samples "$cfg_val_pool_size" \
                --seed "$cfg_seed"
        fi
    fi

    # Use cleaned test set for evaluation when selection is enabled
    local test_files="$math_test_path"
    [[ "$selection_enabled" == "True" && -f "$math_test_cleaned_path" ]] && test_files="$math_test_cleaned_path"

    # Build Hydra command
    local cmd="python3 $REPO_ROOT/RLVR/train.py \
    hydra.run.dir=$HYDRA_DIR \
    algorithm.adv_estimator=grpo \
    data.train_files=$math_train_path \
    data.val_files=$test_files \
    data.seed=$cfg_seed \
    data.train_batch_size=$cfg_train_batch_size \
    data.max_prompt_length=$cfg_max_prompt_length \
    data.max_response_length=$cfg_max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=$cfg_model \
    actor_rollout_ref.actor.optim.lr=$cfg_learning_rate \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$cfg_ppo_mini_batch_size \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$cfg_kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.seed=$cfg_seed \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$cfg_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$cfg_gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$cfg_rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.seed=$cfg_seed \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='[\"console\",\"wandb\"]' \
    trainer.project_name=verl_grpo_math \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=$cfg_save_freq \
    trainer.test_freq=$cfg_test_freq \
    trainer.total_epochs=$cfg_total_epochs \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.resume_mode=disable \
    trainer.balance_batch=True \
    trainer.log_val_generations=5 \
    +selection.enable=$selection_enabled \
    +selection.method=$cfg_method \
    +selection.frac=$cfg_selection_frac \
    +selection.use_second_order=False \
    +selection.val_prompts_path=$val_prompts_path \
    +selection.val_pool_size=$cfg_val_pool_size \
    +selection.val_batch_size=$cfg_val_batch_size \
    +selection.val_max_prompt_length=$cfg_max_prompt_length \
    +selection.val_max_response_length=$cfg_max_response_length \
    +selection.val_seed=$cfg_seed \
    +selection.refresh_freq=$cfg_refresh_freq \
    +selection.val_loss_type=$cfg_val_loss_type"

    echo ""
    echo "Running command:"
    echo "$cmd"
    echo ""

    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] Would execute above command"
    else
        eval "$cmd"
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
echo "  RLVR Training"
echo "========================================================"
echo "Config dir: $config_dir"
echo "Methods: $resolved_methods ($TOTAL total)"
echo "GPUs: $N_GPUS"
echo "========================================================"

current=0
for method_name in "${method_list[@]}"; do
    current=$((current + 1))
    echo ""
    echo "[$current/$TOTAL] $method_name"
    run_method "$method_name"
done
