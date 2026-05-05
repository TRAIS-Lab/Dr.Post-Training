#!/bin/bash
#
# SFT Training Runner
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

SCRIPT_DIR="$CODE_DIR/Dr.Post-Training/SFT/train"

# Fixed training args (infra-level, always the same)
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
seed_override=""
lr_override=""
eval_split_override=""
dry_run=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)  config_dir="$2"; shift 2 ;;
        --methods|-m)     methods="$2"; shift 2 ;;
        --seed)           seed_override="$2"; shift 2 ;;
        --lr)             lr_override="$2"; shift 2 ;;
        --eval_split)     eval_split_override="$2"; shift 2 ;;
        --dry-run)        dry_run=true; shift ;;
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
            echo "Categories: all, full-training, layer-wise-subset, global-subset, full, lora, meso"
            exit 0
            ;;
        --help|-h)
            cat <<'HELP'
Usage: bash train.sh -c <config_dir> -m <methods> [options]

All experiment settings (model, batch_size, dataset, LR, etc.) live in config files.
Each config directory has a defaults.yaml for shared settings, plus per-method configs.

Required:
  -c, --config_dir <dir>  Config directory (relative to SFT/train/ or absolute)
  -m, --methods <list>    Methods or categories (comma-separated)

Optional:
  --seed <seed>           Override seed from config
  --lr <lr>               Override learning rate from config
  --eval_split <split>    Override eval split ("test" or "lr")
  --dry-run               Print commands without executing
  --list                  List available methods and exit

Categories: all, full-training, layer-wise-subset, global-subset, full, lora, meso

Examples:
  bash train.sh -c configs/tulu3_tydiqa -m all
  bash train.sh -c configs/tulu3_tydiqa -m "LayerWiseSubset-Full,GlobalSubset-Full" --seed 123
  bash train.sh -c configs/tulu3_tydiqa -m full-training --dry-run
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
    cfg_finetuning="Full"

    # Scoring (nested under scoring: in YAML)
    cfg_scoring_method="reduced_ghost"
    cfg_score_compression=""     # Single string: "normal-64*64" or "normal-64*64/sjlt-512"

    # Optimizer compression (nested under optimizer: in YAML)
    cfg_opt_compression=""       # Single string: "normal-512*512" or "normal-512*512/sjlt-256"

    # Curation
    cfg_selection_frac="0.5"
    cfg_selection_mode="topk"
    cfg_n_val="8"
    cfg_val_batch_size="1"
    cfg_val_strategy="merged_batch"
    cfg_use_second_order="false"
    cfg_subset_mode="one_pass"

    # LoRA
    cfg_lora_r="32"
    cfg_lora_alpha="1"
    cfg_lora_dropout="0.1"

    # Experiment
    cfg_model="meta-llama/Llama-3.2-1B"
    cfg_seed="42"
    cfg_batch_size="8"
    cfg_gradient_accumulation_steps="1"
    cfg_n_eval="500"
    cfg_optim="adamw_torch"
    cfg_use_flash_attention="true"
    cfg_learning_rate=""

    # Training hyperparameters
    cfg_max_seq_length="512"
    cfg_lr_scheduler_type="linear"
    cfg_warmup_ratio="0.03"
    cfg_weight_decay="0.0"
    cfg_num_train_epochs="1"
    cfg_eval_steps="50"

    # Dataset
    cfg_train_dataset=""
    cfg_target_task=""
    cfg_percentage=""

    # Extras
    cfg_record_selections="false"
    cfg_record_selections_freq="1"
    cfg_update_compressor_freq="200"
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
            # New nested structure: scoring.method, scoring.compression
            scoring.method)                      cfg_scoring_method="$val" ;;
            scoring.compression)                 cfg_score_compression="$val" ;;
            # New nested structure: optimizer.compression, optimizer.refresh_freq
            optimizer.compression)               cfg_opt_compression="$val" ;;
            optimizer.refresh_freq)              cfg_update_compressor_freq="$val" ;;
            # Legacy keys (backward compatibility)
            scoring_method)                      cfg_scoring_method="$val" ;;
            score_grad_compression.sparsifier)   cfg_score_compression="$val" ;;
            score_grad_compression.projector)    ;; # Ignored in new format
            opt_grad_compression.sparsifier)     cfg_opt_compression="$val" ;;
            opt_grad_compression.projector)      ;; # Ignored in new format
            # Curation
            selection_frac)                      cfg_selection_frac="$val" ;;
            selection_mode)                      cfg_selection_mode="$val" ;;
            n_val)                               cfg_n_val="$val" ;;
            val_batch_size)                      cfg_val_batch_size="$val" ;;
            val_strategy)                        cfg_val_strategy="$val" ;;
            use_second_order)                    cfg_use_second_order="$val" ;;
            subset_mode)                         cfg_subset_mode="$val" ;;
            lora_r)                              cfg_lora_r="$val" ;;
            lora_alpha)                          cfg_lora_alpha="$val" ;;
            lora_dropout)                        cfg_lora_dropout="$val" ;;
            model)                               cfg_model="$val" ;;
            seed)                                cfg_seed="$val" ;;
            batch_size)                          cfg_batch_size="$val" ;;
            gradient_accumulation_steps)         cfg_gradient_accumulation_steps="$val" ;;
            n_eval)                              cfg_n_eval="$val" ;;
            optim)                               cfg_optim="$val" ;;
            use_flash_attention)                 cfg_use_flash_attention="$val" ;;
            learning_rate)                       cfg_learning_rate="$val" ;;
            max_seq_length)                      cfg_max_seq_length="$val" ;;
            lr_scheduler_type)                   cfg_lr_scheduler_type="$val" ;;
            warmup_ratio)                        cfg_warmup_ratio="$val" ;;
            weight_decay)                        cfg_weight_decay="$val" ;;
            num_train_epochs)                    cfg_num_train_epochs="$val" ;;
            eval_steps)                          cfg_eval_steps="$val" ;;
            train_dataset)                       cfg_train_dataset="$val" ;;
            target_task)                         cfg_target_task="$val" ;;
            percentage)                          cfg_percentage="$val" ;;
            record_selections)                   cfg_record_selections="$val" ;;
            record_selections_freq)              cfg_record_selections_freq="$val" ;;
            update_compressor_freq)              cfg_update_compressor_freq="$val" ;;
        esac
    done < "$file"
}

# =============================================================================
# Method resolution (categories auto-discover from config dir)
# =============================================================================
resolve_methods() {
    local input="$1"

    # Discover available methods
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
            full-training)       for m in "${available[@]}"; do [[ "$m" == FullTraining-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            layer-wise-subset)      for m in "${available[@]}"; do [[ "$m" == LayerWiseSubset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            global-subset)         for m in "${available[@]}"; do [[ "$m" == GlobalSubset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            full)           for m in "${available[@]}"; do [[ "$m" == *-Full ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            lora)           for m in "${available[@]}"; do [[ "$m" == *-LoRA ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            meso)           for m in "${available[@]}"; do [[ "$m" == *-MeSO ]] && resolved="${resolved:+$resolved,}$m"; done ;;
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

    # Validate required fields
    if [[ -z "$cfg_target_task" ]] || [[ -z "$cfg_percentage" ]]; then
        echo "ERROR: target_task and percentage must be set (in defaults.yaml or method config)"
        return 1
    fi

    # Derived values
    local internal_method="NA"
    [[ "$cfg_method" != "FullTraining" ]] && internal_method="$cfg_method"

    local use_lora="false"
    [[ "$cfg_finetuning" == "LoRA" || "$cfg_finetuning" == "MeSO-LoRA" ]] && use_lora="true"

    # LR fallback if not specified anywhere (every YAML should set this explicitly)
    if [[ -z "$cfg_learning_rate" ]]; then
        [[ "$use_lora" == "true" ]] && cfg_learning_rate="5e-04" || cfg_learning_rate="2e-05"
    fi

    local model_name=$(basename "$cfg_model")
    local method_str="$exp_name"
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && method_str="${method_str}-2nd"

    # Build job name
    local train_str="${cfg_train_dataset:-default}"
    local JOB_NAME="${train_str}_${cfg_target_task}-${model_name}-${method_str}-p${cfg_percentage}-lr${cfg_learning_rate}-b${cfg_batch_size}-v${cfg_n_val}-s${cfg_seed}"

    local data_dir="$SCRATCH_DIR/Dr.Post-Training/SFT/data"
    local output_dir="$SCRATCH_DIR/Dr.Post-Training/SFT/${JOB_NAME}"
    mkdir -p "$output_dir"

    echo ""
    echo "=============================================="
    echo "  Running: $exp_name"
    echo "=============================================="
    echo "Config: $config_file"
    echo "Job: $JOB_NAME"
    echo "Model: $cfg_model | Task: $cfg_target_task | LR: $cfg_learning_rate"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    echo "Batch: $cfg_batch_size | Val: $cfg_val_batch_size | Curation: $cfg_selection_frac"
    echo "Output: $output_dir"
    echo "=============================================="

    # FSDP for large models
    local fsdp_args=""
    case "$cfg_model" in
        *Llama-2-13b*|*llama-2-13b*)
            fsdp_args="--fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune" ;;
        *Mistral-7B*|*mistral-7b*)
            fsdp_args="--fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune" ;;
    esac

    local DATA_SEED=$((cfg_seed + 1))
    local PORT=$((29400 + RANDOM % 10000))

    # Build command
    local cmd="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv_id=$RANDOM --rdzv_backend c10d --rdzv_endpoint=localhost:$PORT \
-m SFT.train.train \
$FIXED_ARGS \
$fsdp_args \
--max_seq_length $cfg_max_seq_length \
--lr_scheduler_type $cfg_lr_scheduler_type \
--warmup_ratio $cfg_warmup_ratio \
--weight_decay $cfg_weight_decay \
--num_train_epochs $cfg_num_train_epochs \
--eval_steps $cfg_eval_steps \
--model_name_or_path $cfg_model \
--output_dir $output_dir \
--data_dir $data_dir \
--percentage $cfg_percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $cfg_batch_size \
--method $internal_method \
--n_val $cfg_n_val \
--n_eval $cfg_n_eval \
--analysis_dataset $cfg_target_task \
--learning_rate $cfg_learning_rate \
--gradient_accumulation_steps $cfg_gradient_accumulation_steps \
--seed $cfg_seed \
--optim $cfg_optim \
--selection_frac $cfg_selection_frac \
--selection_mode $cfg_selection_mode \
--val_strategy $cfg_val_strategy \
--scoring_method $cfg_scoring_method \
--subset_mode $cfg_subset_mode \
--use_flash_attention $cfg_use_flash_attention"

    # Optional args
    [[ -n "$cfg_train_dataset" ]] && cmd="$cmd --train_dataset_names $cfg_train_dataset"
    [[ -n "$cfg_val_batch_size" ]] && cmd="$cmd --val_batch_size_for_selection $cfg_val_batch_size"

    # LoRA
    if [[ "$use_lora" == "true" ]]; then
        cmd="$cmd --lora True --lora_r $cfg_lora_r --lora_alpha $cfg_lora_alpha --lora_dropout $cfg_lora_dropout"
    else
        cmd="$cmd --lora False"
    fi

    # Scoring compression: parse "SPARSIFIER" or "SPARSIFIER/PROJECTOR" format
    if [[ -n "$cfg_score_compression" && "$cfg_score_compression" != "none" ]]; then
        cmd="$cmd --score_compression ${cfg_score_compression%%/*}"
    fi

    # Optimizer compression: parse "SPARSIFIER" or "SPARSIFIER/PROJECTOR" format
    if [[ -n "$cfg_opt_compression" && "$cfg_opt_compression" != "none" ]]; then
        local opt_sparsifier="${cfg_opt_compression%%/*}"
        cmd="$cmd --sparsification $opt_sparsifier --update_compressor_freq $cfg_update_compressor_freq"
        if [[ "$cfg_opt_compression" == */* ]]; then
            local opt_projector="${cfg_opt_compression##*/}"
            [[ "$opt_projector" != "none" ]] && cmd="$cmd --projection $opt_projector"
        fi
    fi

    # Second-order
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && \
        cmd="$cmd --use_second_order True"

    # Recording
    [[ "$cfg_record_selections" == "true" ]] && \
        cmd="$cmd --record_selections True --record_selections_freq $cfg_record_selections_freq"

    # Eval split override
    [[ -n "$eval_split_override" ]] && cmd="$cmd --eval_split $eval_split_override"

    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
    else
        eval $cmd 2>&1 | tee $output_dir/train.log
        exit ${PIPESTATUS[0]}
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
echo "  SFT Training"
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
