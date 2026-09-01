#!/bin/bash
#
# RLHF Training Runner
#
# All experiment settings live in config files.
# Each config directory has defaults.yaml (shared settings) + per-method configs.
#
# Usage: bash train.sh -c <config_dir> -m <methods> [options]
#

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

cd $CODE_DIR/Dr.Post-Training
export PYTHONPATH="$CODE_DIR/Dr.Post-Training:$PYTHONPATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fixed training args (infra-level, always the same)
FIXED_ARGS="--bf16=True \
--logging_steps=1 \
--report_to=none \
--enable_eval=True \
--eval_on_step_generations=True"

# =============================================================================
# CLI
# =============================================================================
config_dir=""
methods=""
seed_override=""
lr_override=""
lr_vhead_override=""
init_kl_coef_override=""
dry_run=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)      config_dir="$2"; shift 2 ;;
        --methods|-m)         methods="$2"; shift 2 ;;
        --seed)               seed_override="$2"; shift 2 ;;
        --lr)                 lr_override="$2"; shift 2 ;;
        --lr_vhead)           lr_vhead_override="$2"; shift 2 ;;
        --init_kl_coef)       init_kl_coef_override="$2"; shift 2 ;;
        --dry-run)            dry_run=true; shift ;;
        --list)
            dir="${config_dir:-configs}"
            [[ "$dir" != /* ]] && dir="$SCRIPT_DIR/$dir"
            echo "Available methods in $dir:"
            for f in "$dir"/*.yaml; do
                [[ ! -f "$f" ]] && continue
                name=$(basename "$f" .yaml)
                [[ "$name" != "defaults" ]] && echo "  $name"
            done
            echo ""
            echo "Categories: all, baseline, iif, layer-wise-subset, global-subset"
            exit 0
            ;;
        --help|-h)
            cat <<'HELP'
Usage: bash train.sh -c <config_dir> -m <methods> [options]

All experiment settings (model, batch_size, PPO params, LR, etc.) live in config files.
Each config directory has a defaults.yaml for shared settings, plus per-method configs.

Required:
  -c, --config_dir <dir>  Config directory (relative to RLHF/train/ or absolute)
  -m, --methods <list>    Methods or categories (comma-separated)

Optional:
  --seed <seed>           Override seed from config
  --lr <lr>               Override learning rate from config
  --lr_vhead <lr>         Override value head LR from config
  --init_kl_coef <coef>   Override initial KL coefficient from config
  --dry-run               Print commands without executing
  --list                  List available methods and exit

Categories: all, baseline, iif, layer-wise-subset, global-subset

Examples:
  bash train.sh -c configs/toxicity -m all
  bash train.sh -c configs/toxicity -m "LayerWiseSubset-LoRA,GlobalSubset-LoRA" --seed 123
  bash train.sh -c configs/toxicity -m baseline --dry-run
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
# Config parser
# =============================================================================
reset_config() {
    # Method
    cfg_method="FullTraining"
    cfg_finetuning="LoRA"

    # Scoring
    cfg_scoring_method="reduced_ghost"
    cfg_score_compression=""
    cfg_subset_mode="two_pass"

    # Optimizer compression
    cfg_opt_compression=""

    # Model
    cfg_model="EleutherAI/gpt-neo-2.7B"
    cfg_reward_model=""
    cfg_task=""

    # Training
    cfg_seed="42"
    cfg_batch_size="256"
    cfg_epochs="1"
    cfg_max_steps="-1"
    cfg_use_flash_attention="true"
    cfg_lr_scheduler_type="linear"
    cfg_warmup_ratio="0.03"
    cfg_weight_decay="0.0"

    # LR
    cfg_learning_rate="1e-5"
    cfg_lr_vhead="5e-4"
    cfg_init_kl_coef="0.02"

    # PPO
    cfg_ppo_epochs="4"
    cfg_mini_batch_size="4"
    cfg_kl_estimator="k1"
    cfg_adap_kl_ctrl="true"
    cfg_target="70.0"
    cfg_target_kl="0.3"
    cfg_early_stopping="true"
    cfg_max_new_tokens="30"
    cfg_min_new_tokens="0"

    # LoRA
    cfg_lora_r="16"
    cfg_lora_alpha="32"
    cfg_lora_target_modules=""

    # Curation
    cfg_filter_frac="1.0"
    cfg_use_second_order="false"
    cfg_n_val="0"
    cfg_val_batch_size="256"
    cfg_val_loss_type="reward"
    cfg_update_compressor_freq="200"

    # Evaluation
    cfg_eval_interval="1"
    cfg_n_eval="500"
    cfg_eval_batch_size="256"
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
                section="$top_key"; continue
            fi
            section=""
            key="$top_key"
            val="$top_val"
        fi

        case "$key" in
            method)                              cfg_method="$val" ;;
            finetuning)                          cfg_finetuning="$val" ;;
            # New nested structure
            scoring.method)                      cfg_scoring_method="$val" ;;
            scoring.compression)                 cfg_score_compression="$val" ;;
            optimizer.compression)               cfg_opt_compression="$val" ;;
            optimizer.refresh_freq)              cfg_update_compressor_freq="$val" ;;
            scoring_method)                      cfg_scoring_method="$val" ;;
            subset_mode)                         cfg_subset_mode="$val" ;;
            # Legacy keys (backward compat)
            score_grad_compression.sparsifier)   cfg_score_compression="$val" ;;
            score_grad_compression.projector)    ;; # Ignored
            opt_grad_compression.sparsifier)     cfg_opt_compression="$val" ;;
            opt_grad_compression.projector)      ;; # Ignored
            model)                               cfg_model="$val" ;;
            reward_model)                        cfg_reward_model="$val" ;;
            task)                                cfg_task="$val" ;;
            seed)                                cfg_seed="$val" ;;
            batch_size)                          cfg_batch_size="$val" ;;
            epochs)                              cfg_epochs="$val" ;;
            max_steps)                           cfg_max_steps="$val" ;;
            use_flash_attention)                 cfg_use_flash_attention="$val" ;;
            lr_scheduler_type)                   cfg_lr_scheduler_type="$val" ;;
            warmup_ratio)                        cfg_warmup_ratio="$val" ;;
            weight_decay)                        cfg_weight_decay="$val" ;;
            learning_rate)                       cfg_learning_rate="$val" ;;
            lr_vhead)                            cfg_lr_vhead="$val" ;;
            init_kl_coef)                        cfg_init_kl_coef="$val" ;;
            ppo_epochs)                          cfg_ppo_epochs="$val" ;;
            mini_batch_size)                     cfg_mini_batch_size="$val" ;;
            kl_estimator)                        cfg_kl_estimator="$val" ;;
            adap_kl_ctrl)                        cfg_adap_kl_ctrl="$val" ;;
            target)                              cfg_target="$val" ;;
            target_kl)                           cfg_target_kl="$val" ;;
            early_stopping)                      cfg_early_stopping="$val" ;;
            max_new_tokens)                      cfg_max_new_tokens="$val" ;;
            min_new_tokens)                      cfg_min_new_tokens="$val" ;;
            lora_r)                              cfg_lora_r="$val" ;;
            lora_alpha)                          cfg_lora_alpha="$val" ;;
            lora_target_modules)                 cfg_lora_target_modules="$val" ;;
            filter_frac)                         cfg_filter_frac="$val" ;;
            use_second_order)                    cfg_use_second_order="$val" ;;
            n_val)                               cfg_n_val="$val" ;;
            val_batch_size)                      cfg_val_batch_size="$val" ;;
            val_loss_type)                       cfg_val_loss_type="$val" ;;
            update_compressor_freq)              cfg_update_compressor_freq="$val" ;;
            eval_interval)                       cfg_eval_interval="$val" ;;
            n_eval)                              cfg_n_eval="$val" ;;
            eval_batch_size)                     cfg_eval_batch_size="$val" ;;
        esac
    done < "$file"
}

# =============================================================================
# Method resolution (categories auto-discover from config dir)
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
            baseline)  for m in "${available[@]}"; do [[ "$m" == FullTraining-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            iif)       for m in "${available[@]}"; do [[ "$m" == IIF-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            layer-wise-subset) for m in "${available[@]}"; do [[ "$m" == LayerWiseSubset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            global-subset)    for m in "${available[@]}"; do [[ "$m" == GlobalSubset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            lora)      for m in "${available[@]}"; do [[ "$m" == *-LoRA ]] && resolved="${resolved:+$resolved,}$m"; done ;;
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
    [[ -n "$lr_vhead_override" ]] && cfg_lr_vhead="$lr_vhead_override"
    [[ -n "$init_kl_coef_override" ]] && cfg_init_kl_coef="$init_kl_coef_override"

    # Validate required fields
    if [[ -z "$cfg_task" ]] || [[ -z "$cfg_reward_model" ]]; then
        echo "ERROR: task and reward_model must be set (in defaults.yaml or method config)"
        return 1
    fi

    # Derived values
    local internal_method="NA"
    [[ "$cfg_method" != "FullTraining" ]] && internal_method="$cfg_method"

    local model_name=$(basename "$cfg_model")
    local method_str="${cfg_method}"
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && method_str="${method_str}-2nd"

    # Build job name
    local val_type_short
    case "$cfg_val_loss_type" in
        reward)      val_type_short="rew" ;;
        token-pg)    val_type_short="tpg" ;;
        train-loss)  val_type_short="tloss" ;;
        *)           val_type_short="$cfg_val_loss_type" ;;
    esac
    local val_str="v${cfg_n_val}-${val_type_short}"
    [[ "$cfg_n_val" -gt 0 ]] && val_str="${val_str}-b${cfg_val_batch_size}"

    local JOB_NAME="${cfg_task}-${model_name}-${method_str}-${cfg_finetuning}-lr${cfg_learning_rate}-b${cfg_batch_size}-${val_str}-pe${cfg_ppo_epochs}-mb${cfg_mini_batch_size}-kl${cfg_init_kl_coef}-s${cfg_seed}"

    local output_dir="$SCRATCH_DIR/Dr.Post-Training/RLHF/${JOB_NAME}"
    mkdir -p "$output_dir"

    echo ""
    echo "=============================================="
    echo "  Running: $exp_name"
    echo "=============================================="
    echo "Config: $config_file"
    echo "Job: $JOB_NAME"
    echo "Model: $cfg_model | Task: $cfg_task"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    echo "LR: $cfg_learning_rate | LR vhead: $cfg_lr_vhead | KL: $cfg_init_kl_coef"
    echo "Batch: $cfg_batch_size | PPO epochs: $cfg_ppo_epochs | Mini-batch: $cfg_mini_batch_size"
    echo "Output: $output_dir"
    echo "=============================================="

    # Build command
    local cmd="python RLHF/train/train.py \
$FIXED_ARGS \
--lr_scheduler_type=$cfg_lr_scheduler_type \
--warmup_ratio=$cfg_warmup_ratio \
--weight_decay=$cfg_weight_decay \
--task=$cfg_task \
--method=$internal_method \
--model_name_or_path=$cfg_model \
--reward_model_name=$cfg_reward_model \
--per_device_train_batch_size=$cfg_batch_size \
--learning_rate=$cfg_learning_rate \
--learning_rate_vhead=$cfg_lr_vhead \
--filter_frac=$cfg_filter_frac \
--max_steps=$cfg_max_steps \
--num_train_epochs=$cfg_epochs \
--seed=$cfg_seed \
--ppo_epochs=$cfg_ppo_epochs \
--mini_batch_size=$cfg_mini_batch_size \
--init_kl_coef=$cfg_init_kl_coef \
--kl_estimator=$cfg_kl_estimator \
--adap_kl_ctrl=$cfg_adap_kl_ctrl \
--target=$cfg_target \
--target_kl=$cfg_target_kl \
--early_stopping=$cfg_early_stopping \
--max_new_tokens=$cfg_max_new_tokens \
--min_new_tokens=$cfg_min_new_tokens \
--output_dir=$output_dir"

    # LoRA
    case "$cfg_finetuning" in
        LoRA|MeSO-LoRA)
            cmd="$cmd --lora=True --lora_r=$cfg_lora_r --lora_alpha=$cfg_lora_alpha"
            [[ -n "$cfg_lora_target_modules" ]] && cmd="$cmd --lora_target_modules $cfg_lora_target_modules"
            ;;
        *)
            cmd="$cmd --lora=False"
            ;;
    esac

    # Flash attention
    cmd="$cmd --use_flash_attention=$([[ "$cfg_use_flash_attention" == "true" ]] && echo True || echo False)"

    # Scoring method
    cmd="$cmd --scoring_method=$cfg_scoring_method --subset_mode=$cfg_subset_mode"

    # Scoring compression: parse "SPARSIFIER" or "SPARSIFIER/PROJECTOR" format
    if [[ -n "$cfg_score_compression" && "$cfg_score_compression" != "none" ]]; then
        cmd="$cmd --score_compression=${cfg_score_compression%%/*}"
    fi

    # Optimizer compression: parse "SPARSIFIER" or "SPARSIFIER/PROJECTOR" format
    if [[ -n "$cfg_opt_compression" && "$cfg_opt_compression" != "none" ]]; then
        local opt_sparsifier="${cfg_opt_compression%%/*}"
        cmd="$cmd --sparsification=$opt_sparsifier --update_compressor_freq=$cfg_update_compressor_freq"
        if [[ "$cfg_opt_compression" == */* ]]; then
            local opt_projector="${cfg_opt_compression##*/}"
            [[ "$opt_projector" != "none" ]] && cmd="$cmd --projection=$opt_projector"
        fi
    fi

    # Second-order
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && \
        cmd="$cmd --use_second_order=True"

    # Validation / Evaluation
    cmd="$cmd --n_val=$cfg_n_val --val_batch_size=$cfg_val_batch_size --val_loss_type=$cfg_val_loss_type"
    cmd="$cmd --eval_interval=$cfg_eval_interval --n_eval=$cfg_n_eval --eval_batch_size=$cfg_eval_batch_size"

    cmd="$cmd 2>&1 | tee $output_dir/train.log"

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
echo "  RLHF Training"
echo "========================================================"
echo "Config dir: $config_dir"
echo "Methods: $resolved_methods ($TOTAL total)"
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
