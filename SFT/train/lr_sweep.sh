#!/bin/bash

#SBATCH --job-name=LR-Sweep
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdzy-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Gradient-Streaming/SFT/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

# =============================================================================
# LR Sweep Script
# =============================================================================
# Performs learning rate grid search for SFT experiments.
# Runs short training (default 5% of data) with multiple LRs and selects best.
#
# Usage:
#   bash SFT/train/lr_sweep.sh --experiments all --task samsum --train alpaca
#
# Output:
#   - Updates SFT/train/lr_config.json with best LRs
#   - Creates detailed results in SFT/lr_sweep_results/
# =============================================================================

cd $HOME/Project/Gradient-Streaming

export PYTHONPATH="$HOME/Project/Gradient-Streaming:$PYTHONPATH"

# Base training arguments (same as train.sh but with fewer epochs/steps)
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
--report_to=none \
--seed=0 \
--percentage=1.0 \
--selection_frac=0.5 \
--use_flash_attention=True"

# Default values
data_selection="NA"
optim="adamw_torch"
data_dir="SFT/data"
sweep_percentage=0.05  # Use 5% of data for LR sweep
n_val=8
n_eval=100  # Fewer eval examples for speed
model="llama3-1b"
batch_size=8
val_batch_size=""
seed=42
gradient_accumulation_steps=1
task="mmlu"
train_dataset=""
subject="sociology"
compression=""
use_second_order=false
use_lora=false
use_flash_attention=true

# LR grid defaults
lr_grid="1e-6,5e-6,1e-5,5e-5,1e-4"
lr_grid_lora="5e-5,1e-4,2e-4,5e-4,1e-3"

# Multi-experiment mode
experiments=""
dry_run=false

# Output paths
lr_config_file="SFT/train/lr_config.json"
sweep_results_dir="SFT/lr_sweep_results"

# LoRA defaults
lora_alpha=1
lora_r=8
lora_dropout=0.1
update_compressor_freq=200

# Experiment definitions (same as train.sh)
declare -A EXPERIMENT_DEFS=(
    ["NA-NA-full"]="NA::false:false"
    ["NA-NA-lora"]="NA::true:false"
    ["Streaming-NA-full"]="Streaming::false:true"
    ["Streaming-NA-lora"]="Streaming::true:true"
    ["GREATS-NA-full"]="GREATS::false:true"
    ["GREATS-NA-lora"]="GREATS::true:true"
    ["Streaming-LoGra-full"]="Streaming:LoGra:false:true"
    ["GREATS-LoGra-full"]="GREATS:LoGra:false:true"
)

# Category mappings (same as train.sh)
declare -A CATEGORY_EXPERIMENTS=(
    ["all"]="NA-NA-full,NA-NA-lora,Streaming-NA-full,Streaming-NA-lora,GREATS-NA-full,GREATS-NA-lora,Streaming-LoGra-full,GREATS-LoGra-full"
    ["baseline"]="NA-NA-full,NA-NA-lora"
    ["streaming"]="Streaming-NA-full,Streaming-NA-lora,Streaming-LoGra-full"
    ["greats"]="GREATS-NA-full,GREATS-NA-lora,GREATS-LoGra-full"
    ["full"]="NA-NA-full,Streaming-NA-full,GREATS-NA-full,Streaming-LoGra-full,GREATS-LoGra-full"
    ["lora"]="NA-NA-lora,Streaming-NA-lora,GREATS-NA-lora"
    ["compression"]="Streaming-LoGra-full,GREATS-LoGra-full"
    ["no-compression"]="NA-NA-full,NA-NA-lora,Streaming-NA-full,Streaming-NA-lora,GREATS-NA-full,GREATS-NA-lora"
)

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_selection)
            data_selection="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --val_batch_size)
            val_batch_size="$2"
            shift 2
            ;;
        --sweep_percentage)
            sweep_percentage="$2"
            shift 2
            ;;
        --n_val)
            n_val="$2"
            shift 2
            ;;
        --n_eval)
            n_eval="$2"
            shift 2
            ;;
        --model)
            model="$2"
            shift 2
            ;;
        --lr_grid)
            lr_grid="$2"
            shift 2
            ;;
        --lr_grid_lora)
            lr_grid_lora="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            gradient_accumulation_steps="$2"
            shift 2
            ;;
        --task)
            task="$2"
            shift 2
            ;;
        --train)
            train_dataset="$2"
            shift 2
            ;;
        --subject)
            subject="$2"
            shift 2
            ;;
        --compression)
            compression="$2"
            shift 2
            ;;
        --use_second_order)
            use_second_order=true
            shift 1
            ;;
        --update_compressor_freq)
            update_compressor_freq="$2"
            shift 2
            ;;
        --lora)
            use_lora=true
            shift 1
            ;;
        --lora_alpha)
            lora_alpha="$2"
            shift 2
            ;;
        --lora_r)
            lora_r="$2"
            shift 2
            ;;
        --lora_dropout)
            lora_dropout="$2"
            shift 2
            ;;
        --data_dir)
            data_dir="$2"
            shift 2
            ;;
        --lr_config)
            lr_config_file="$2"
            shift 2
            ;;
        --experiments)
            experiments="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Perform learning rate grid search for SFT experiments."
            echo ""
            echo "LR Sweep Options:"
            echo "  --lr_grid <lrs>                        Comma-separated LRs for full fine-tuning"
            echo "                                         (default: 1e-6,5e-6,1e-5,5e-5,1e-4)"
            echo "  --lr_grid_lora <lrs>                   Comma-separated LRs for LoRA"
            echo "                                         (default: 5e-5,1e-4,2e-4,5e-4,1e-3)"
            echo "  --sweep_percentage <pct>               Data percentage for sweep (default: 0.05)"
            echo "  --lr_config <path>                     Output config file (default: SFT/train/lr_config.json)"
            echo ""
            echo "Multi-Experiment Mode:"
            echo "  --experiments <list>                   Run sweep for multiple experiments"
            echo "  --dry-run                              Print commands without executing"
            echo ""
            echo "  Experiment names: NA-NA-full, NA-NA-lora, Streaming-NA-full, etc."
            echo "  Categories: all, baseline, streaming, greats, full, lora, compression"
            echo ""
            echo "Shared Options (same as train.sh):"
            echo "  --model <model>                        Model: llama3-1b, llama2-7b, etc."
            echo "  --task <task>                          Task: mmlu, bbh, tydiqa, samsum, gsm8k"
            echo "  --train <dataset>                      Training dataset"
            echo "  --subject <subject>                    MMLU/BBH subject"
            echo "  --batch_size <size>                    Training batch size"
            echo "  --n_val <n>                            Validation examples"
            echo "  --n_eval <n>                           Evaluation examples (default: 100)"
            echo "  --seed <seed>                          Random seed"
            echo ""
            echo "Output:"
            echo "  - Updates lr_config.json with best LRs per experiment"
            echo "  - Detailed results in SFT/lr_sweep_results/"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# ========================================
# Resolve experiment names from categories
# ========================================
resolve_experiments() {
    local input="$1"
    local resolved=""
    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)
        if [[ -n "${CATEGORY_EXPERIMENTS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,${CATEGORY_EXPERIMENTS[$item]}"
            else
                resolved="${CATEGORY_EXPERIMENTS[$item]}"
            fi
        elif [[ -n "${EXPERIMENT_DEFS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,$item"
            else
                resolved="$item"
            fi
        else
            echo "ERROR: Unknown experiment or category: $item"
            exit 1
        fi
    done
    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# ========================================
# Extract validation loss from training output
# ========================================
# IMPORTANT: We use val_loss (validation set) for LR selection, NOT eval_loss (test set).
# This prevents test set leakage into hyperparameter tuning.
extract_val_loss() {
    local output_dir="$1"
    local val_loss=""

    # Primary: Read from evaluation_results.json (structured output)
    local eval_json="$output_dir/evaluation_results.json"
    if [[ -f "$eval_json" ]]; then
        # Get the last val_loss from the JSON array
        val_loss=$(python3 -c "
import json
import sys
try:
    with open('$eval_json', 'r') as f:
        results = json.load(f)
    if results and len(results) > 0:
        last_val_loss = results[-1].get('val_loss')
        if last_val_loss is not None:
            print(f'{last_val_loss:.10e}')
        else:
            sys.exit(1)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)
    fi

    # Fallback: Try parsing train.log if JSON not available
    if [[ -z "$val_loss" ]]; then
        local log_file="$output_dir/train.log"
        if [[ -f "$log_file" ]]; then
            val_loss=$(grep -oP "val_loss['\"]?\s*[:=]\s*\K[0-9]+\.[0-9]+" "$log_file" 2>/dev/null | tail -1)
        fi
    fi

    # Return default if no valid value found
    if [[ -z "$val_loss" ]]; then
        echo "999.0"
    else
        echo "$val_loss"
    fi
}

# ========================================
# Check if a string is a valid number
# ========================================
is_valid_number() {
    local val="$1"
    # Check if it's a valid floating point number (including scientific notation)
    [[ "$val" =~ ^[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]]
}

# ========================================
# Clean up model weights to save disk space
# ========================================
# Keeps only essential files: evaluation_results.json, train.log, training_args.bin
cleanup_model_weights() {
    local output_dir="$1"

    if [[ -d "$output_dir" ]]; then
        # Remove large model files (full fine-tuning)
        rm -f "$output_dir/model.safetensors" 2>/dev/null
        rm -f "$output_dir/pytorch_model.bin" 2>/dev/null
        rm -f "$output_dir/model*.safetensors" 2>/dev/null

        # Remove LoRA adapter files
        rm -f "$output_dir/adapter_model.safetensors" 2>/dev/null
        rm -f "$output_dir/adapter_model.bin" 2>/dev/null
        rm -f "$output_dir/adapter_config.json" 2>/dev/null

        # Remove tokenizer and config files
        rm -f "$output_dir/tokenizer.json" 2>/dev/null
        rm -f "$output_dir/tokenizer_config.json" 2>/dev/null
        rm -f "$output_dir/special_tokens_map.json" 2>/dev/null
        rm -f "$output_dir/config.json" 2>/dev/null
        rm -f "$output_dir/generation_config.json" 2>/dev/null
        rm -f "$output_dir/README.md" 2>/dev/null

        # Remove TensorBoard logs
        rm -rf "$output_dir/runs" 2>/dev/null

        echo "  Cleaned up model weights in $output_dir"
    fi
}

# ========================================
# Run single LR sweep trial
# ========================================
run_lr_trial() {
    local exp_data_selection="$1"
    local exp_compression="$2"
    local exp_use_lora="$3"
    local exp_use_second_order="$4"
    local trial_lr="$5"
    local exp_name="$6"
    local trial_output_dir="$7"

    local exp_base_training_args="$base_training_args"

    # Compression configuration
    local sparsification=""
    local projection=""
    if [[ -n "$exp_compression" ]]; then
        case "$exp_compression" in
            LoGra)
                if [ "$exp_use_lora" = true ]; then
                    sparsification="normal-128*128"
                else
                    sparsification="normal-512*512"
                fi
                ;;
            GraSS)
                if [ "$exp_use_lora" = true ]; then
                    sparsification="random_mask-256*256"
                    projection="sjlt-16384"
                else
                    sparsification="random_mask-1024*1024"
                    projection="sjlt-262144"
                fi
                ;;
        esac
    fi

    # Map model to path
    local MODEL_PATH
    case $model in
        llama3-1b) MODEL_PATH="meta-llama/Llama-3.2-1B" ;;
        llama2-7b) MODEL_PATH="meta-llama/Llama-2-7b-hf" ;;
        llama2-13b)
            MODEL_PATH="meta-llama/Llama-2-13b-hf"
            exp_base_training_args="$exp_base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
            ;;
        mistral-7b)
            MODEL_PATH="mistralai/Mistral-7B-v0.1"
            exp_base_training_args="$exp_base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
            ;;
        *) MODEL_PATH="$model" ;;
    esac

    local DATA_SEED=$((seed + 1))
    local ID=$RANDOM
    local PORT=$((29400 + RANDOM % 10000))

    local header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d --rdzv-endpoint=localhost:$PORT \
-m SFT.train.train"

    local training_args="$exp_base_training_args \
--model_name_or_path $MODEL_PATH \
--output_dir $trial_output_dir \
--data_dir $data_dir \
--percentage $sweep_percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $batch_size \
--method $exp_data_selection \
--subject $subject \
--n_val $n_val \
--n_eval $n_eval \
--analysis_dataset $task \
--learning_rate $trial_lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed \
--optim $optim"

    if [[ -n "$train_dataset" ]]; then
        training_args="$training_args --train_dataset_names $train_dataset"
    fi
    if [[ -n "$val_batch_size" ]]; then
        training_args="$training_args --val_batch_size_for_selection $val_batch_size"
    fi
    if [ "$exp_use_lora" = true ]; then
        training_args="$training_args --lora True --lora_alpha $lora_alpha --lora_r $lora_r --lora_dropout $lora_dropout --lora_target_modules q_proj k_proj v_proj o_proj"
    else
        training_args="$training_args --lora False"
    fi
    if [[ -n "$sparsification" ]]; then
        training_args="$training_args --sparsification $sparsification"
    fi
    if [[ -n "$projection" ]]; then
        training_args="$training_args --projection $projection"
    fi
    if [[ -n "$exp_compression" ]]; then
        training_args="$training_args --update_compressor_freq $update_compressor_freq"
    fi
    if [[ "$exp_data_selection" != "NA" ]] && [ "$exp_use_second_order" = true ]; then
        training_args="$training_args --use_second_order True"
    fi

    local log_file="$trial_output_dir/train.log"

    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] $header $training_args 2>&1 | tee $log_file"
        echo "0.0"  # Return dummy loss for dry run
    else
        mkdir -p "$trial_output_dir"
        # Run training and capture output to log file only (not stdout)
        # This prevents training output from being mixed with val_loss return value
        eval "$header" "$training_args" > "$log_file" 2>&1
        # Extract val_loss from output directory (reads evaluation_results.json)
        extract_val_loss "$trial_output_dir"
    fi
}

# ========================================
# Update lr_config.json with best LR
# ========================================
update_lr_config() {
    local config_key="$1"
    local exp_name="$2"
    local best_lr="$3"
    local best_val_loss="$4"

    # Initialize config file if it doesn't exist
    if [[ ! -f "$lr_config_file" ]]; then
        echo "{}" > "$lr_config_file"
    fi

    # Use Python to update JSON (more reliable than jq for complex updates)
    python3 << EOF
import json
from datetime import datetime

config_file = "$lr_config_file"
config_key = "$config_key"
exp_name = "$exp_name"
best_lr = "$best_lr"
best_val_loss = float("$best_val_loss") if "$best_val_loss" != "" else None

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

if config_key not in config:
    config[config_key] = {}

config[config_key][exp_name] = {
    "lr": float(best_lr),
    "val_loss": best_val_loss,
    "sweep_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "sweep_percentage": float("$sweep_percentage"),
    "model": "$model"
}

with open(config_file, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Updated {config_file}: {config_key}/{exp_name} -> lr={best_lr}")
EOF
}

# ========================================
# Main execution
# ========================================

# Build config key (includes subject for mmlu/bbh)
train_str="${train_dataset:-default}"
if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
    config_key="${train_str}_${task}_${subject}"
else
    config_key="${train_str}_${task}"
fi

echo ""
echo "========================================================"
echo "  Learning Rate Sweep"
echo "========================================================"
echo "Config key: $config_key"
echo "Task: $task"
echo "Training data: ${train_dataset:-default}"
echo "Model: $model"
echo "Sweep percentage: $sweep_percentage"
echo ""
echo "LR grid (full): $lr_grid"
echo "LR grid (LoRA): $lr_grid_lora"
echo ""
echo "Output config: $lr_config_file"
echo "Results dir: $sweep_results_dir"
echo "========================================================"

# Create results directory
mkdir -p "$sweep_results_dir"

# Initialize summary file
summary_file="$sweep_results_dir/${config_key}_summary.txt"
echo "LR Sweep Summary: $config_key" > "$summary_file"
echo "Date: $(date)" >> "$summary_file"
echo "Model: $model" >> "$summary_file"
echo "Sweep percentage: $sweep_percentage" >> "$summary_file"
echo "" >> "$summary_file"

if [[ -n "$experiments" ]]; then
    resolved_experiments=$(resolve_experiments "$experiments")
    echo "Experiments: $resolved_experiments"
    echo "Experiments: $resolved_experiments" >> "$summary_file"
    echo "" >> "$summary_file"

    IFS=',' read -ra exp_list <<< "$resolved_experiments"

    for exp_name in "${exp_list[@]}"; do
        echo ""
        echo "========================================================"
        echo "  Sweeping LR for: $exp_name"
        echo "========================================================"

        # Parse experiment definition
        IFS=':' read -ra exp_parts <<< "${EXPERIMENT_DEFS[$exp_name]}"
        exp_data_selection="${exp_parts[0]}"
        exp_compression="${exp_parts[1]}"
        exp_use_lora="${exp_parts[2]}"
        exp_use_second_order="${exp_parts[3]}"

        # Select appropriate LR grid
        if [ "$exp_use_lora" = true ]; then
            current_lr_grid="$lr_grid_lora"
        else
            current_lr_grid="$lr_grid"
        fi

        echo "Testing LRs: $current_lr_grid"
        echo "" >> "$summary_file"
        echo "=== $exp_name ===" >> "$summary_file"
        echo "LR grid: $current_lr_grid" >> "$summary_file"

        best_lr=""
        best_val_loss=999.0

        # Run trials for each LR
        IFS=',' read -ra lr_values <<< "$current_lr_grid"
        for trial_lr in "${lr_values[@]}"; do
            trial_lr=$(echo "$trial_lr" | xargs)  # Trim whitespace
            echo ""
            echo "--- Testing LR: $trial_lr ---"

            trial_output_dir="$sweep_results_dir/${config_key}/${exp_name}/lr_${trial_lr}"

            val_loss=$(run_lr_trial "$exp_data_selection" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$trial_lr" "$exp_name" "$trial_output_dir")

            echo "LR $trial_lr -> val_loss: $val_loss"
            echo "  LR $trial_lr: val_loss=$val_loss" >> "$summary_file"

            # Check if this is the best (lower val_loss is better)
            # Validate that val_loss is a proper number before comparing
            if is_valid_number "$val_loss" && is_valid_number "$best_val_loss"; then
                if (( $(echo "$val_loss < $best_val_loss" | bc -l) )); then
                    best_val_loss="$val_loss"
                    best_lr="$trial_lr"
                fi
            else
                echo "  Warning: Invalid val_loss value '$val_loss', skipping comparison"
            fi

            # Clean up model weights to save disk space (keeps eval results and logs)
            cleanup_model_weights "$trial_output_dir"
        done

        echo ""
        echo "Best LR for $exp_name: $best_lr (val_loss: $best_val_loss)"
        echo "  BEST: $best_lr (val_loss=$best_val_loss)" >> "$summary_file"

        # Update config file
        if [[ -n "$best_lr" ]]; then
            update_lr_config "$config_key" "$exp_name" "$best_lr" "$best_val_loss"
        fi
    done
else
    echo "ERROR: Please specify experiments with --experiments"
    echo "Example: --experiments all"
    echo "         --experiments baseline"
    echo "         --experiments NA-NA-full,Streaming-NA-full"
    exit 1
fi

echo ""
echo "========================================================"
echo "  LR Sweep Complete!"
echo "========================================================"
echo "Summary: $summary_file"
echo "Config updated: $lr_config_file"
echo ""
echo "To run full training with these LRs:"
echo "  bash SFT/train/train.sh --experiments all --task $task --train ${train_dataset:-default}"
echo "========================================================"
