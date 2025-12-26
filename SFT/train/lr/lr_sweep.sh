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
--eval_steps=99999 \
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
model="meta-llama/Llama-3.2-1B"
batch_size=8
val_batch_size="1"
seed=42
gradient_accumulation_steps=1
task="mmlu"
train_dataset=""
subject="sociology"
compression=""
use_second_order=true
use_lora=false
use_flash_attention=true

# LR grid defaults
lr_grid="1e-6,5e-6,1e-5,5e-5,1e-4"
lr_grid_lora="5e-5,1e-4,2e-4,5e-4,1e-3"

# Sweep mode
sweep_mode="grid"  # "grid" or "binary"

# Binary search defaults
binary_lr_min="1e-7"
binary_lr_max="1e-3"
binary_lr_min_lora="1e-6"
binary_lr_max_lora="1e-2"
binary_max_iters=7  # ~6 iterations narrows range by ~10x

# Stability margin: prefer smaller LR unless larger LR is significantly better
lr_margin=0.05  # 5% - larger LR must have loss at least 5% lower to be preferred

# Multi-experiment mode
experiments=""
dry_run=false

# Output paths
lr_config_file="SFT/train/lr/config.json"
sweep_results_dir="SFT/train/lr/results"

# LoRA defaults
lora_alpha=1
lora_r=32
lora_dropout=0.1
update_compressor_freq=200

# Experiment definitions (same as train.sh)
declare -A EXPERIMENT_DEFS=(
    ["NA-NA-Full"]="NA::false:false"
    ["NA-NA-LoRA"]="NA::true:false"
    ["Streaming-NA-Full"]="Streaming::false:true"
    ["Streaming-NA-LoRA"]="Streaming::true:true"
    ["GREATS-NA-Full"]="GREATS::false:true"
    ["GREATS-NA-LoRA"]="GREATS::true:true"
    ["Streaming-LoGra-Full"]="Streaming:LoGra:false:true"
    ["GREATS-LoGra-Full"]="GREATS:LoGra:false:true"
)

# Category mappings (same as train.sh)
declare -A CATEGORY_EXPERIMENTS=(
    ["all"]="NA-NA-Full,NA-NA-LoRA,Streaming-NA-Full,Streaming-NA-LoRA,GREATS-NA-Full,GREATS-NA-LoRA,Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["baseline"]="NA-NA-Full,NA-NA-LoRA"
    ["streaming"]="Streaming-NA-Full,Streaming-NA-LoRA,Streaming-LoGra-Full"
    ["greats"]="GREATS-NA-Full,GREATS-NA-LoRA,GREATS-LoGra-Full"
    ["full"]="NA-NA-Full,Streaming-NA-Full,GREATS-NA-Full,Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["lora"]="NA-NA-LoRA,Streaming-NA-LoRA,GREATS-NA-LoRA"
    ["compression"]="Streaming-LoGra-Full,GREATS-LoGra-Full"
    ["no-compression"]="NA-NA-Full,NA-NA-LoRA,Streaming-NA-Full,Streaming-NA-LoRA,GREATS-NA-Full,GREATS-NA-LoRA"
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
        --mode)
            sweep_mode="$2"
            shift 2
            ;;
        --binary_lr_min)
            binary_lr_min="$2"
            shift 2
            ;;
        --binary_lr_max)
            binary_lr_max="$2"
            shift 2
            ;;
        --binary_lr_min_lora)
            binary_lr_min_lora="$2"
            shift 2
            ;;
        --binary_lr_max_lora)
            binary_lr_max_lora="$2"
            shift 2
            ;;
        --binary_max_iters)
            binary_max_iters="$2"
            shift 2
            ;;
        --lr_margin)
            lr_margin="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Perform learning rate search for SFT experiments."
            echo ""
            echo "Sweep Mode:"
            echo "  --mode <mode>                          Sweep mode: 'grid' (default) or 'binary'"
            echo ""
            echo "Grid Search Options (--mode grid):"
            echo "  --lr_grid <lrs>                        Comma-separated LRs for full fine-tuning"
            echo "                                         (default: 1e-6,5e-6,1e-5,5e-5,1e-4)"
            echo "  --lr_grid_lora <lrs>                   Comma-separated LRs for LoRA"
            echo "                                         (default: 5e-5,1e-4,2e-4,5e-4,1e-3)"
            echo ""
            echo "Binary Search Options (--mode binary):"
            echo "  --binary_lr_min <lr>                   Min LR for full fine-tuning (default: 1e-6)"
            echo "  --binary_lr_max <lr>                   Max LR for full fine-tuning (default: 1e-3)"
            echo "  --binary_lr_min_lora <lr>              Min LR for LoRA (default: 1e-5)"
            echo "  --binary_lr_max_lora <lr>              Max LR for LoRA (default: 1e-2)"
            echo "  --binary_max_iters <n>                 Max iterations (default: 6)"
            echo ""
            echo "Stability Options:"
            echo "  --lr_margin <pct>                      Prefer smaller LR unless larger LR is this much"
            echo "                                         better (default: 0.05 = 5%)"
            echo ""
            echo "Common Options:"
            echo "  --sweep_percentage <pct>               Data percentage for sweep (default: 0.05)"
            echo "  --lr_config <path>                     Output config file (default: SFT/train/lr/config.json)"
            echo ""
            echo "Multi-Experiment Mode:"
            echo "  --experiments <list>                   Run sweep for multiple experiments"
            echo "  --dry-run                              Print commands without executing"
            echo ""
            echo "  Experiment names: NA-NA-Full, NA-NA-LoRA, Streaming-NA-Full, etc."
            echo "  Categories: all, baseline, streaming, greats, Full, LoRA, compression"
            echo ""
            echo "Shared Options (same as train.sh):"
            echo "  --model <model>                        HuggingFace model path (default: meta-llama/Llama-3.2-1B)"
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

# Extract model name from path for use in output directories
# e.g., "meta-llama/Llama-3.2-1B" -> "Llama-3.2-1B"
model_name=$(basename "$model")

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
# Extract evaluation loss from training output
# ========================================
# We use eval_loss (evaluation/test set) for LR selection, NOT val_loss (validation set).
# This prevents overfitting to the validation set which is also used for data selection
# during Streaming/GREATS training.
extract_eval_loss() {
    local output_dir="$1"
    local eval_loss=""

    # Primary: Read from evaluation_results.json (structured output)
    local eval_json="$output_dir/evaluation_results.json"
    if [[ -f "$eval_json" ]]; then
        # Get the last eval_loss from the JSON array
        eval_loss=$(python3 -c "
import json
import sys
try:
    with open('$eval_json', 'r') as f:
        results = json.load(f)
    if results and len(results) > 0:
        last_eval_loss = results[-1].get('eval_loss')
        if last_eval_loss is not None:
            print(f'{last_eval_loss:.10e}')
        else:
            sys.exit(1)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null)
    fi

    # Fallback: Try parsing train.log if JSON not available
    if [[ -z "$eval_loss" ]]; then
        local log_file="$output_dir/train.log"
        if [[ -f "$log_file" ]]; then
            eval_loss=$(grep -oP "eval_loss['\"]?\s*[:=]\s*\K[0-9]+\.[0-9]+" "$log_file" 2>/dev/null | tail -1)
        fi
    fi

    # Return default if no valid value found
    if [[ -z "$eval_loss" ]]; then
        echo "999.0"
    else
        echo "$eval_loss"
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

    # Use model path directly
    local MODEL_PATH="$model"

    # Model-specific configurations (match by path pattern)
    case "$model" in
        *Llama-2-13b*|*llama-2-13b*)
            exp_base_training_args="$exp_base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
            ;;
        *Mistral-7B*|*mistral-7b*)
            exp_base_training_args="$exp_base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
            ;;
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
        # This prevents training output from being mixed with eval_loss return value
        eval "$header" "$training_args" > "$log_file" 2>&1
        # Extract eval_loss from output directory (reads evaluation_results.json)
        extract_eval_loss "$trial_output_dir"
    fi
}

# ========================================
# Update lr_config.json with best LR
# ========================================
update_lr_config() {
    local config_key="$1"
    local exp_name="$2"
    local best_lr="$3"
    local best_eval_loss="$4"

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
best_eval_loss = float("$best_eval_loss") if "$best_eval_loss" != "" else None

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

if config_key not in config:
    config[config_key] = {}

config[config_key][exp_name] = {
    "lr": float(best_lr),
    "eval_loss": best_eval_loss,
    "sweep_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "sweep_percentage": float("$sweep_percentage"),
    "model": "$model_name"
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
echo "Sweep mode: $sweep_mode"
echo ""
if [[ "$sweep_mode" == "grid" ]]; then
    echo "LR grid (Full): $lr_grid"
    echo "LR grid (LoRA): $lr_grid_lora"
elif [[ "$sweep_mode" == "binary" ]]; then
    echo "LR range (Full): $binary_lr_min -> $binary_lr_max"
    echo "LR range (LoRA): $binary_lr_min_lora -> $binary_lr_max_lora"
    echo "Max iterations: $binary_max_iters"
fi
echo "LR margin: $lr_margin (prefer smaller LR for stability)"
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

        echo "" >> "$summary_file"
        echo "=== $exp_name ===" >> "$summary_file"

        best_lr=""
        best_eval_loss=999.0

        if [[ "$sweep_mode" == "binary" ]]; then
            # ========================================
            # Binary Search Mode (Golden Section Search)
            # ========================================
            # Select appropriate LR bounds
            if [ "$exp_use_lora" = true ]; then
                current_lr_min="$binary_lr_min_lora"
                current_lr_max="$binary_lr_max_lora"
            else
                current_lr_min="$binary_lr_min"
                current_lr_max="$binary_lr_max"
            fi

            echo "Running Binary Search: $current_lr_min -> $current_lr_max (max $binary_max_iters iterations)"
            echo "Mode: Binary Search ($current_lr_min -> $current_lr_max, max $binary_max_iters iters)" >> "$summary_file"

            # Golden section search in log space
            # φ = (1 + √5) / 2 ≈ 1.618, 1/φ ≈ 0.618
            inv_phi="0.6180339887"

            # Work in log10 space
            log_a=$(python3 -c "import math; print(math.log10($current_lr_min))")
            log_b=$(python3 -c "import math; print(math.log10($current_lr_max))")

            # Track all evaluated points to find best (reset for each experiment)
            declare -A lr_losses
            lr_losses=()

            for ((iter=1; iter<=binary_max_iters; iter++)); do
                echo ""
                echo "--- Binary Search Iteration $iter/$binary_max_iters ---"
                echo "  Current range: [10^$log_a, 10^$log_b]"

                # Calculate two interior points using golden ratio
                log_c=$(python3 -c "print($log_b - $inv_phi * ($log_b - $log_a))")
                log_d=$(python3 -c "print($log_a + $inv_phi * ($log_b - $log_a))")

                lr_c=$(python3 -c "print(f'{10**$log_c:.2e}')")
                lr_d=$(python3 -c "print(f'{10**$log_d:.2e}')")

                echo "  Testing LR_c=$lr_c and LR_d=$lr_d"

                # Evaluate lr_c if not already evaluated
                if [[ -z "${lr_losses[$lr_c]}" ]]; then
                    trial_output_dir="$sweep_results_dir/${config_key}/${exp_name}/binary_iter${iter}_lr_${lr_c}"
                    loss_c=$(run_lr_trial "$exp_data_selection" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$lr_c" "$exp_name" "$trial_output_dir")
                    lr_losses[$lr_c]="$loss_c"
                    echo "  LR $lr_c -> eval_loss: $loss_c"
                    echo "  Iter $iter: LR $lr_c -> eval_loss=$loss_c" >> "$summary_file"
                    cleanup_model_weights "$trial_output_dir"
                else
                    loss_c="${lr_losses[$lr_c]}"
                    echo "  LR $lr_c -> eval_loss: $loss_c (cached)"
                fi

                # Evaluate lr_d if not already evaluated
                if [[ -z "${lr_losses[$lr_d]}" ]]; then
                    trial_output_dir="$sweep_results_dir/${config_key}/${exp_name}/binary_iter${iter}_lr_${lr_d}"
                    loss_d=$(run_lr_trial "$exp_data_selection" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$lr_d" "$exp_name" "$trial_output_dir")
                    lr_losses[$lr_d]="$loss_d"
                    echo "  LR $lr_d -> eval_loss: $loss_d"
                    echo "  Iter $iter: LR $lr_d -> eval_loss=$loss_d" >> "$summary_file"
                    cleanup_model_weights "$trial_output_dir"
                else
                    loss_d="${lr_losses[$lr_d]}"
                    echo "  LR $lr_d -> eval_loss: $loss_d (cached)"
                fi

                # Narrow the search interval (bias toward lower half if losses within margin)
                # Only go to upper half if loss_d is significantly better than loss_c
                prefer_lower=$(python3 -c "
loss_c = float('$loss_c')
loss_d = float('$loss_d')
margin = float('$lr_margin')
# Prefer lower half unless upper half is significantly better (by margin)
if loss_d < loss_c * (1 - margin):
    print('0')  # Upper half is significantly better
else:
    print('1')  # Prefer lower half for stability
" 2>/dev/null)
                if [[ "$prefer_lower" == "1" ]]; then
                    # Prefer lower half (smaller LRs)
                    log_b="$log_d"
                    echo "  -> Narrowing to lower half: [10^$log_a, 10^$log_b] (prefer smaller LR)"
                else
                    # Upper half is significantly better
                    log_a="$log_c"
                    echo "  -> Narrowing to upper half: [10^$log_a, 10^$log_b] (significantly better)"
                fi
            done

            # Find the best LR from all evaluated points (prefer smaller LR for stability)
            best_lr=""
            best_eval_loss="999.0"
            # Sort LRs from smallest to largest and iterate
            sorted_lrs=$(python3 -c "
lrs = '${!lr_losses[*]}'.split()
for lr in sorted(lrs, key=float):
    print(lr)
" 2>/dev/null)
            for lr in $sorted_lrs; do
                loss="${lr_losses[$lr]}"
                if is_valid_number "$loss"; then
                    should_update=$(python3 -c "
loss = float('$loss')
best_loss = float('$best_eval_loss')
margin = float('$lr_margin')
best_lr = float('${best_lr:-0}') if '${best_lr:-}' else 0
current_lr = float('$lr')

if best_lr == 0:
    print('1')  # First result
elif current_lr < best_lr:
    # Smaller LR: update if loss is not worse (within margin)
    if loss <= best_loss * (1 + margin):
        print('1')
    else:
        print('0')
else:
    # Larger LR: only update if significantly better
    if loss < best_loss * (1 - margin):
        print('1')
    else:
        print('0')
" 2>/dev/null)
                    if [[ "$should_update" == "1" ]]; then
                        best_eval_loss="$loss"
                        best_lr="$lr"
                    fi
                fi
            done

            echo ""
            echo "Binary search complete. Best LR: $best_lr (eval_loss: $best_eval_loss)"

        elif [[ "$sweep_mode" == "grid" ]]; then
            # ========================================
            # Grid Search Mode
            # ========================================
            # Select appropriate LR grid
            if [ "$exp_use_lora" = true ]; then
                current_lr_grid="$lr_grid_lora"
            else
                current_lr_grid="$lr_grid"
            fi

            echo "Testing LRs: $current_lr_grid"
            echo "LR grid: $current_lr_grid" >> "$summary_file"

            # Run trials for each LR
            IFS=',' read -ra lr_values <<< "$current_lr_grid"
            for trial_lr in "${lr_values[@]}"; do
                trial_lr=$(echo "$trial_lr" | xargs)  # Trim whitespace
                echo ""
                echo "--- Testing LR: $trial_lr ---"

                trial_output_dir="$sweep_results_dir/${config_key}/${exp_name}/lr_${trial_lr}"

                eval_loss=$(run_lr_trial "$exp_data_selection" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$trial_lr" "$exp_name" "$trial_output_dir")

                echo "LR $trial_lr -> eval_loss: $eval_loss"
                echo "  LR $trial_lr: eval_loss=$eval_loss" >> "$summary_file"

                # Check if this is the best (prefer smaller LR for stability)
                # Only update to larger LR if improvement exceeds lr_margin
                if is_valid_number "$eval_loss" && is_valid_number "$best_eval_loss"; then
                    # Compute threshold: new loss must be at least (margin)% better
                    # threshold = best_eval_loss * (1 - lr_margin)
                    should_update=$(python3 -c "
eval_loss = float('$eval_loss')
best_loss = float('$best_eval_loss')
margin = float('$lr_margin')
trial_lr = float('$trial_lr')
best_lr = float('${best_lr:-0}') if '${best_lr:-}' else 0

if best_lr == 0:
    # First valid result
    print('1')
elif trial_lr < best_lr:
    # Smaller LR: update if loss is not worse (within margin)
    if eval_loss <= best_loss * (1 + margin):
        print('1')
    else:
        print('0')
else:
    # Larger LR: only update if loss is significantly better (exceeds margin)
    if eval_loss < best_loss * (1 - margin):
        print('1')
    else:
        print('0')
" 2>/dev/null)
                    if [[ "$should_update" == "1" ]]; then
                        echo "  -> New best (margin=$lr_margin)"
                        best_eval_loss="$eval_loss"
                        best_lr="$trial_lr"
                    fi
                else
                    echo "  Warning: Invalid eval_loss value '$eval_loss', skipping comparison"
                fi

                # Clean up model weights to save disk space (keeps eval results and logs)
                cleanup_model_weights "$trial_output_dir"
            done
        else
            echo "ERROR: Unknown sweep mode '$sweep_mode'. Use 'grid' or 'binary'."
            exit 1
        fi

        echo ""
        echo "Best LR for $exp_name: $best_lr (eval_loss: $best_eval_loss)"
        echo "  BEST: $best_lr (eval_loss=$best_eval_loss)" >> "$summary_file"

        # Update config file
        if [[ -n "$best_lr" ]]; then
            update_lr_config "$config_key" "$exp_name" "$best_lr" "$best_eval_loss"
        fi
    done
else
    echo "ERROR: Please specify experiments with --experiments"
    echo "Example: --experiments all"
    echo "         --experiments baseline"
    echo "         --experiments NA-NA-Full,Streaming-NA-Full"
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
