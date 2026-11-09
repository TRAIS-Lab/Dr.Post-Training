#!/bin/bash
# SFT Training Runner
#
# Yaml keys forwarded to train.py besides the shared ones: init_eot_from_eos (Qwen3 <|im_end|> row init),
# loss_reduction (sample_mean | token_mean; the paper uses sample_mean), score_normalization (none | layer_meanabs |
# layer_std, layer-normalized Global Subset control), selection_mode (topk | random), lora_target_modules (comma-separated
# module-name suffixes, e.g. q_proj,k_proj; default all-linear), add_bos (BOS id on every sequence), chat_template (less_tulu),
# subset_sampler / subset_seed (how and with which fixed seed the pool subset is drawn), target_max_seq_length, torch_dtype /
# bf16 (float32 weights; bf16: false turns bf16 autocast off), stop_at_step (end early, LR
# schedule keeps the full horizon), warmup_steps (overrides warmup_ratio), save_at_steps (full checkpoints). Run dirs land under
# --runs_root (default $SCRATCH_DIR/Dr.Post-Training/SFT/runs, the sample-mean tree the paper reads).
# CLI overrides of yaml keys: --gradient_checkpointing, --subset_mode, --eval_steps (48 GB cards; evaluation cadence).
# Never edit this file while jobs are running it (bash reads scripts incrementally): copy, edit, launch the copy.
#
# All experiment settings live in config files.
# Each config directory has defaults.yaml (shared settings) + per-method configs.
#
# Usage: bash train.sh -c <config_dir> -m <methods> [options]
#

# cluster_env.sh: $DRPT_CLUSTER_ENV if set, else the one at the root of this checkout
# (set DRPT_CLUSTER_ENV when the script runs from a Slurm spool copy).
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
n_val_override=""
runs_root_override=""
eval_split_override=""
gc_override=""
subset_mode_override=""
eval_steps_override=""
save_at_steps_override=""
dry_run=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir|-c)  config_dir="$2"; shift 2 ;;
        --methods|-m)     methods="$2"; shift 2 ;;
        --seed)           seed_override="$2"; shift 2 ;;
        --lr)             lr_override="$2"; shift 2 ;;
        --n_val)          n_val_override="$2"; shift 2 ;;
        --runs_root)      runs_root_override="$2"; shift 2 ;;
        --eval_split)     eval_split_override="$2"; shift 2 ;;
        --gradient_checkpointing) gc_override="$2"; shift 2 ;;
        --subset_mode)    subset_mode_override="$2"; shift 2 ;;
        --eval_steps)     eval_steps_override="$2"; shift 2 ;;
        --save_at_steps)  save_at_steps_override="$2"; shift 2 ;;
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
            echo "Categories: all, full-training, layer-wise-subset, global-subset, block-wise-subset, sublayer-wise-subset, group-wise-subset, full, lora, meso"
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
  --runs_root <dir>       Parent directory for run dirs (default $SCRATCH_DIR/Dr.Post-Training/SFT/runs)
  --n_val <n>             Override the number of D* rows (n_val) from config
  --eval_split <split>    Override eval split ("test" or "lr")
  --gradient_checkpointing <bool>  Override gradient_checkpointing from config (48 GB cards)
  --subset_mode <one_pass|two_pass>  Override subset_mode from config (two_pass avoids the one-pass retention memory)
  --eval_steps <n>        Override the periodic-evaluation interval (evaluation does not affect training)
  --save_at_steps <list>  Write full checkpoints (model + optimizer) at these comma-separated steps (default: none)
  --dry-run               Print commands without executing
  --list                  List available methods and exit

Categories: all, full-training, layer-wise-subset, global-subset,
            block-wise-subset, sublayer-wise-subset, group-wise-subset, full, lora, meso

Group-wise curation (method: GroupWiseSubset, or the aliases BlockWiseSubset /
SublayerWiseSubset) takes two extra YAML keys:
  selection_granularity: layer | sublayer | block | global | custom
  selection_groups:      per-block rules, e.g. "attn.qkv=q_proj,k_proj,v_proj;attn.o=o_proj;mlp=gate_proj,up_proj,down_proj"
                         (no ':' or quotes inside; setting it implies custom)

Examples:
  bash train.sh -c configs/alpaca_samsum -m all
  bash train.sh -c configs/less_tydiqa -m "LayerWiseSubset-LoRA,GlobalSubset-LoRA" --seed 123
  bash train.sh -c configs/dolci_reason_mathpersona -m "GlobalSubset-Full-eot,LayerWiseSubset-Full-eot"
  bash train.sh -c configs/alpaca_samsum -m full-training --dry-run
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
    cfg_scoring_method="pip"
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
    cfg_selection_granularity=""       # GroupWiseSubset only; empty -> train.py default (block)
    cfg_selection_groups=""            # GroupWiseSubset only; custom per-block rules (implies custom)

    # LoRA
    cfg_lora_r="8"
    cfg_lora_alpha="16"
    cfg_lora_dropout="0.1"
    cfg_lora_target_modules=""        # comma-separated suffixes; empty -> train.py default (all-linear)

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
    cfg_lr_scheduler_type="cosine"
    cfg_warmup_ratio="0.1"
    cfg_weight_decay="0.0"
    cfg_num_train_epochs="1"
    cfg_init_eot_from_eos=""
    cfg_loss_reduction=""              # sample_mean | token_mean; empty -> train.py default (sample_mean)
    cfg_score_normalization=""         # none | layer_meanabs | layer_std; empty -> train.py default (none)
    cfg_eval_steps="50"

    # Dataset
    cfg_train_dataset=""
    cfg_target_task=""
    cfg_percentage=""

    # Extras
    cfg_record_selections="false"
    cfg_record_selections_freq="1"
    cfg_update_compressor_freq="200"
    cfg_gradient_checkpointing="false"
    cfg_val_seq_length_multiplier=""   # empty -> train.py default (1.2); 0 disables rejection
    cfg_eval_split=""                  # empty -> "test"; CLI --eval_split overrides
    cfg_add_bos=""                     # true -> BOS id on every sequence (GREATS / LESS / torchtune encoding)
    cfg_chat_template=""               # less_tulu -> exact LESS / open-instruct tulu rendering (SFT/data/chat_format.py)
    cfg_subset_sampler=""              # torch | numpy (LESS / GREATS np.random.permutation)
    cfg_subset_seed=""                 # pool-subset seed fixed across training seeds (default seed + 1)
    cfg_target_max_seq_length=""       # truncation of D* / held-out rows (default max_seq_length)
    cfg_torch_dtype=""                 # float32 -> fp32 weights (default: bfloat16 under bf16 training)
    cfg_bf16=""                        # false -> no bf16 autocast (pure fp32 with torch_dtype float32)
    cfg_stop_at_step=""                # end training at this step; the LR schedule keeps the full-run horizon
    cfg_warmup_steps=""                # absolute warmup steps (overrides warmup_ratio when > 0)
    cfg_save_at_steps=""               # comma-separated full-checkpoint steps (CLI --save_at_steps overrides)
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
            # Nested structure: scoring.method, scoring.compression
            scoring.method)                      cfg_scoring_method="$val" ;;
            scoring.compression)                 cfg_score_compression="$val" ;;
            # Nested structure: optimizer.compression, optimizer.refresh_freq
            optimizer.compression)               cfg_opt_compression="$val" ;;
            optimizer.refresh_freq)              cfg_update_compressor_freq="$val" ;;
            # Flat keys (backward compatibility)
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
            selection_granularity)               cfg_selection_granularity="$val" ;;
            selection_groups)                    cfg_selection_groups="$val" ;;
            lora_r)                              cfg_lora_r="$val" ;;
            lora_alpha)                          cfg_lora_alpha="$val" ;;
            lora_dropout)                        cfg_lora_dropout="$val" ;;
            lora_target_modules)                 cfg_lora_target_modules="$val" ;;
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
            gradient_checkpointing)              cfg_gradient_checkpointing="$val" ;;
            val_seq_length_multiplier)           cfg_val_seq_length_multiplier="$val" ;;
            eval_split)                          cfg_eval_split="$val" ;;
            init_eot_from_eos)                   cfg_init_eot_from_eos="$val" ;;
            loss_reduction)                      cfg_loss_reduction="$val" ;;
            score_normalization)                 cfg_score_normalization="$val" ;;
            add_bos)                             cfg_add_bos="$val" ;;
            chat_template)                       cfg_chat_template="$val" ;;
            subset_sampler)                      cfg_subset_sampler="$val" ;;
            subset_seed)                         cfg_subset_seed="$val" ;;
            target_max_seq_length)               cfg_target_max_seq_length="$val" ;;
            torch_dtype)                         cfg_torch_dtype="$val" ;;
            bf16)                                cfg_bf16="$val" ;;
            stop_at_step)                        cfg_stop_at_step="$val" ;;
            warmup_steps)                        cfg_warmup_steps="$val" ;;
            save_at_steps)                       cfg_save_at_steps="$val" ;;
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
            block-wise-subset)     for m in "${available[@]}"; do [[ "$m" == BlockWiseSubset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            sublayer-wise-subset)  for m in "${available[@]}"; do [[ "$m" == SublayerWiseSubset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
            group-wise-subset)     for m in "${available[@]}"; do [[ "$m" == GroupWiseSubset-* || "$m" == BlockWiseSubset-* || "$m" == SublayerWiseSubset-* ]] && resolved="${resolved:+$resolved,}$m"; done ;;
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
    [[ -n "$gc_override" ]] && cfg_gradient_checkpointing="$gc_override"
    [[ -n "$subset_mode_override" ]] && cfg_subset_mode="$subset_mode_override"
    [[ -n "$eval_steps_override" ]] && cfg_eval_steps="$eval_steps_override"

    # CLI overrides
    [[ -n "$seed_override" ]] && cfg_seed="$seed_override"
    [[ -n "$lr_override" ]] && cfg_learning_rate="$lr_override"
    [[ -n "$n_val_override" ]] && cfg_n_val="$n_val_override"

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
        [[ "$use_lora" == "true" ]] && cfg_learning_rate="1e-04" || cfg_learning_rate="1e-05"
    fi

    local model_name=$(basename "$cfg_model")
    local method_str="$exp_name"
    [[ "$internal_method" != "NA" && "$cfg_use_second_order" == "true" ]] && method_str="${method_str}-2nd"

    # Build job name
    local train_str="${cfg_train_dataset:-default}"
    local JOB_NAME="${train_str}_${cfg_target_task}-${model_name}-${method_str}-p${cfg_percentage}-lr${cfg_learning_rate}-b${cfg_batch_size}-v${cfg_n_val}-s${cfg_seed}"

    local data_dir="$SCRATCH_DIR/Dr.Post-Training/SFT/data"
    local runs_root="${runs_root_override:-$SCRATCH_DIR/Dr.Post-Training/SFT/runs}"
    local output_dir="$runs_root/${JOB_NAME}"
    mkdir -p "$output_dir"

    echo ""
    echo "=============================================="
    echo "  Running: $exp_name"
    echo "=============================================="
    echo "Config: $config_file"
    echo "Job: $JOB_NAME"
    echo "Model: $cfg_model | Task: $cfg_target_task | LR: $cfg_learning_rate"
    echo "Method: $cfg_method | Finetuning: $cfg_finetuning"
    case "$cfg_method" in
        GroupWiseSubset|BlockWiseSubset|SublayerWiseSubset)
            echo "Grouping: granularity=${cfg_selection_granularity:-<default>} groups=${cfg_selection_groups:-<none>}" ;;
    esac
    echo "Batch: $cfg_batch_size | Val: $cfg_val_batch_size | Curation: $cfg_selection_frac | Loss reduction: ${cfg_loss_reduction:-sample_mean (default)}"
    echo "Seq len: $cfg_max_seq_length | Grad ckpt: $cfg_gradient_checkpointing | n_val: $cfg_n_val | n_eval: $cfg_n_eval"
    echo "Output: $output_dir"
    echo "=============================================="

    # No FSDP: LESS wrapped Llama-2-13B / Mistral-7B in FSDP for multi-GPU runs, but this launcher starts one process
    # (--nproc_per_node 1), where full sharding only adds a wrapper around the model and its gradient hooks.
    local fsdp_args=""

    local DATA_SEED=$((cfg_seed + 1))
    # Derive port from SLURM_JOB_ID (or PID fallback) so different jobs landing
    # on the same node get distinct ports. Random ports caused C10D rendezvous
    # collisions when slurm packed multiple jobs per node.
    local PORT=$((20000 + (${SLURM_JOB_ID:-$$} % 40000)))

    # Build command
    local cmd="torchrun --standalone --nproc_per_node 1 --nnodes 1 \
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
--use_flash_attention $cfg_use_flash_attention \
--gradient_checkpointing $cfg_gradient_checkpointing"

    # Optional args
    [[ -n "$cfg_val_seq_length_multiplier" ]] && cmd="$cmd --val_seq_length_multiplier $cfg_val_seq_length_multiplier"
    [[ -n "$cfg_init_eot_from_eos" ]] && cmd="$cmd --init_eot_from_eos $cfg_init_eot_from_eos"
    [[ -n "$cfg_loss_reduction" ]] && cmd="$cmd --loss_reduction $cfg_loss_reduction"
    [[ -n "$cfg_score_normalization" ]] && cmd="$cmd --score_normalization $cfg_score_normalization"
    # GroupWiseSubset grouping (single-quoted: the rules string contains ';' and spaces)
    [[ -n "$cfg_selection_granularity" ]] && cmd="$cmd --selection_granularity $cfg_selection_granularity"
    [[ -n "$cfg_selection_groups" ]] && cmd="$cmd --selection_groups '$cfg_selection_groups'"
    [[ -n "$cfg_train_dataset" ]] && cmd="$cmd --train_dataset_names $cfg_train_dataset"
    [[ -n "$cfg_val_batch_size" ]] && cmd="$cmd --val_batch_size_for_selection $cfg_val_batch_size"
    if [[ -n "$save_at_steps_override" ]]; then cmd="$cmd --save_at_steps $save_at_steps_override"
    elif [[ -n "$cfg_save_at_steps" ]]; then cmd="$cmd --save_at_steps $cfg_save_at_steps"; fi
    [[ "$cfg_add_bos" == "true" ]] && cmd="$cmd --add_bos True"
    [[ -n "$cfg_chat_template" ]] && cmd="$cmd --chat_template $cfg_chat_template"
    [[ -n "$cfg_subset_sampler" ]] && cmd="$cmd --subset_sampler $cfg_subset_sampler"
    [[ -n "$cfg_subset_seed" ]] && cmd="$cmd --subset_seed $cfg_subset_seed"
    [[ -n "$cfg_target_max_seq_length" ]] && cmd="$cmd --target_max_seq_length $cfg_target_max_seq_length"
    [[ -n "$cfg_torch_dtype" ]] && cmd="$cmd --torch_dtype $cfg_torch_dtype"
    [[ "$cfg_bf16" == "false" ]] && cmd="$cmd --bf16 False"   # overrides FIXED_ARGS --bf16=True (last one wins)
    [[ -n "$cfg_stop_at_step" ]] && cmd="$cmd --stop_at_step $cfg_stop_at_step"
    [[ -n "$cfg_warmup_steps" ]] && cmd="$cmd --warmup_steps $cfg_warmup_steps"

    # LoRA
    if [[ "$use_lora" == "true" ]]; then
        cmd="$cmd --lora True --lora_r $cfg_lora_r --lora_alpha $cfg_lora_alpha --lora_dropout $cfg_lora_dropout"
        [[ -n "$cfg_lora_target_modules" ]] && cmd="$cmd --lora_target_modules ${cfg_lora_target_modules//,/ }"
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

    # Eval split: CLI override > yaml eval_split > train.py default ("test")
    if [[ -n "$eval_split_override" ]]; then
        cmd="$cmd --eval_split $eval_split_override"
    elif [[ -n "$cfg_eval_split" ]]; then
        cmd="$cmd --eval_split $cfg_eval_split"
    fi

    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
    else
        eval $cmd 2>&1 | tee $output_dir/train.log
        local status=${PIPESTATUS[0]}
        if [[ $status -ne 0 ]]; then
            echo "ERROR: $exp_name failed with exit code $status"
            return $status
        fi
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
failed=()
for method_name in "${method_list[@]}"; do
    current=$((current + 1))
    echo ""
    echo "[$current/$TOTAL] $method_name"
    run_method "$method_name" || failed+=("$method_name")
done

echo ""
echo "========================================================"
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "  All $TOTAL methods completed!"
    echo "========================================================"
else
    echo "  ${#failed[@]}/$TOTAL methods FAILED: ${failed[*]}"
    echo "========================================================"
    exit 1
fi
