#!/bin/bash

#SBATCH --job-name=RLHF-LR-Sweep
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdzy-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Gradient-Streaming/RLHF/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

# =============================================================================
# LR Sweep Script for RLHF
# =============================================================================
# Performs learning rate grid search for RLHF experiments.
# Runs short training with multiple LRs and selects best based on reward.
#
# Usage:
#   bash RLHF/train/lr/lr_sweep.sh --experiments all --task toxicity
#
# Output:
#   - Updates RLHF/train/lr/config.json with best LRs
#   - Creates detailed results in RLHF/train/lr/results/
# =============================================================================

cd $HOME/Project/Gradient-Streaming

export PYTHONPATH="$HOME/Project/Gradient-Streaming:$PYTHONPATH"

# ========================================
# Default values
# ========================================
task="toxicity"
model="EleutherAI/gpt-neo-125m"
reward_model="facebook/roberta-hate-speech-dynabench-r4-target"

# Sweep settings
sweep_max_steps=50  # Fewer steps for LR sweep
batch_size=64
n_val=16
val_batch_size=""
selection_frac=0.5
seed=42

# LoRA settings
use_lora=true
lora_r=16
lora_alpha=32

# PPO settings
ppo_epochs=4
kl_coef=0.04
max_new_tokens=30
min_new_tokens=10

# Validation settings
val_strategy="random"
val_loss_type="reward_weighted"  # Sequence-level attribution: f^seq(θ) = -E[log π_θ(y|x) * Â(x,y)]

# LR grid defaults
lr_grid="1e-6,5e-6,1e-5,5e-5,1e-4"
lr_grid_lora="5e-5,1e-4,2e-4,5e-4,1e-3"

# Sweep mode
sweep_mode="grid"  # "grid" or "binary"

# Binary search defaults
binary_lr_min="1e-6"
binary_lr_max="1e-3"
binary_lr_min_lora="1e-5"
binary_lr_max_lora="1e-2"
binary_max_iters=6

# Stability margin
lr_margin=0.05

# Multi-experiment mode
experiments=""
dry_run=false

# Output paths
lr_config_file="RLHF/train/lr/config.json"
sweep_results_dir="RLHF/train/lr/results"

# Compression settings
sparsification=""
projection=""
update_compressor_freq=200

# ========================================
# Experiment Definitions (8 experiments, matching SFT)
# ========================================
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

# Category mappings (matching SFT)
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

# ========================================
# Parse named arguments
# ========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            task="$2"
            shift 2
            ;;
        --model)
            model="$2"
            shift 2
            ;;
        --reward_model)
            reward_model="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --sweep_max_steps)
            sweep_max_steps="$2"
            shift 2
            ;;
        --n_val)
            n_val="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
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
        --lr_config)
            lr_config_file="$2"
            shift 2
            ;;
        --lora_r)
            lora_r="$2"
            shift 2
            ;;
        --lora_alpha)
            lora_alpha="$2"
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
            echo "Perform learning rate search for RLHF experiments."
            echo ""
            echo "Sweep Mode:"
            echo "  --mode <mode>              Sweep mode: 'grid' (default) or 'binary'"
            echo ""
            echo "Grid Search Options (--mode grid):"
            echo "  --lr_grid <lrs>            Comma-separated LRs for full fine-tuning"
            echo "  --lr_grid_lora <lrs>       Comma-separated LRs for LoRA"
            echo ""
            echo "Binary Search Options (--mode binary):"
            echo "  --binary_lr_min <lr>       Min LR for full fine-tuning"
            echo "  --binary_lr_max <lr>       Max LR for full fine-tuning"
            echo "  --binary_lr_min_lora <lr>  Min LR for LoRA"
            echo "  --binary_lr_max_lora <lr>  Max LR for LoRA"
            echo "  --binary_max_iters <n>     Max iterations"
            echo ""
            echo "Common Options:"
            echo "  --sweep_max_steps <steps>  Training steps for sweep (default: 50)"
            echo "  --lr_config <path>         Output config file"
            echo "  --lr_margin <pct>          Prefer smaller LR margin (default: 0.05)"
            echo ""
            echo "Multi-Experiment Mode:"
            echo "  --experiments <list>       Run sweep for multiple experiments"
            echo "  --dry-run                  Print commands without executing"
            echo ""
            echo "  Categories: all, baseline, streaming, greats, lora, full, compression"
            echo ""
            echo "Shared Options:"
            echo "  --task <task>              Task: toxicity, imdb"
            echo "  --model <model>            Policy model"
            echo "  --batch_size <size>        Training batch size"
            echo "  --seed <seed>              Random seed"
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
# Extract mean reward from training output
# ========================================
extract_mean_reward() {
    local output_dir="$1"
    local log_file="$output_dir/train.log"
    local mean_reward="-999.0"

    if [[ -f "$log_file" ]]; then
        # Extract the last reward/mean value from logs
        mean_reward=$(grep -oP "reward/mean['\"]?\s*[:=]\s*\K[-0-9]+\.[0-9]+" "$log_file" 2>/dev/null | tail -1)
    fi

    if [[ -z "$mean_reward" ]]; then
        echo "-999.0"
    else
        echo "$mean_reward"
    fi
}

# ========================================
# Check if a string is a valid number
# ========================================
is_valid_number() {
    local val="$1"
    [[ "$val" =~ ^-?[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]]
}

# ========================================
# Run single LR sweep trial
# ========================================
run_lr_trial() {
    local exp_method="$1"
    local exp_compression="$2"
    local exp_use_lora="$3"
    local exp_use_second_order="$4"
    local trial_lr="$5"
    local exp_name="$6"
    local trial_output_dir="$7"

    # Configure compression
    local exp_sparsification=""
    local exp_projection=""
    if [[ -n "$exp_compression" ]]; then
        case "$exp_compression" in
            GraSS)
                exp_sparsification="Rademacher-64*64"
                ;;
            LoGra)
                exp_projection="Gaussian-256"
                ;;
        esac
    fi

    # Build command
    local cmd="python RLHF/train/train.py \
    --task=$task \
    --method=$exp_method \
    --model_name_or_path=$model \
    --reward_model_name=$reward_model \
    --per_device_train_batch_size=$batch_size \
    --learning_rate=$trial_lr \
    --n_val=$n_val \
    --selection_frac=$selection_frac \
    --max_steps=$sweep_max_steps \
    --seed=$seed \
    --ppo_epochs=$ppo_epochs \
    --kl_coef=$kl_coef \
    --max_new_tokens=$max_new_tokens \
    --min_new_tokens=$min_new_tokens \
    --val_strategy=$val_strategy \
    --val_loss_type=$val_loss_type \
    --output_dir=$trial_output_dir \
    --bf16=True \
    --lr_scheduler_type=linear \
    --warmup_ratio=0.03 \
    --weight_decay=0.0 \
    --logging_steps=1 \
    --report_to=none"

    # Add LoRA settings
    if [[ "$exp_use_lora" == "true" ]]; then
        cmd="$cmd --lora=True --lora_r=$lora_r --lora_alpha=$lora_alpha"
    else
        cmd="$cmd --lora=False"
    fi

    # Add compression settings
    if [[ -n "$exp_sparsification" ]]; then
        cmd="$cmd --sparsification=$exp_sparsification"
    fi
    if [[ -n "$exp_projection" ]]; then
        cmd="$cmd --projection=$exp_projection"
    fi
    if [[ -n "$exp_sparsification" || -n "$exp_projection" ]]; then
        cmd="$cmd --update_compressor_freq=$update_compressor_freq"
    fi

    # Add second-order flag
    if [[ "$exp_use_second_order" == "true" ]]; then
        cmd="$cmd --use_second_order=True"
    fi

    local log_file="$trial_output_dir/train.log"

    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] $cmd 2>&1 | tee $log_file"
        echo "0.0"
    else
        mkdir -p "$trial_output_dir"
        eval "$cmd" > "$log_file" 2>&1
        extract_mean_reward "$trial_output_dir"
    fi
}

# ========================================
# Update lr_config.json with best LR
# ========================================
update_lr_config() {
    local config_key="$1"
    local exp_name="$2"
    local best_lr="$3"
    local best_reward="$4"

    if [[ ! -f "$lr_config_file" ]]; then
        echo "{}" > "$lr_config_file"
    fi

    python3 << EOF
import json
from datetime import datetime

config_file = "$lr_config_file"
config_key = "$config_key"
exp_name = "$exp_name"
best_lr = "$best_lr"
best_reward = float("$best_reward") if "$best_reward" != "" else None

try:
    with open(config_file, 'r') as f:
        config = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    config = {}

if config_key not in config:
    config[config_key] = {}

config[config_key][exp_name] = {
    "lr": float(best_lr),
    "mean_reward": best_reward,
    "sweep_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "sweep_max_steps": int("$sweep_max_steps"),
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

config_key="${task}"

echo ""
echo "========================================================"
echo "  RLHF Learning Rate Sweep"
echo "========================================================"
echo "Config key: $config_key"
echo "Task: $task"
echo "Model: $model"
echo "Sweep max steps: $sweep_max_steps"
echo "Sweep mode: $sweep_mode"
echo ""
if [[ "$sweep_mode" == "grid" ]]; then
    echo "LR grid (full): $lr_grid"
    echo "LR grid (LoRA): $lr_grid_lora"
elif [[ "$sweep_mode" == "binary" ]]; then
    echo "LR range (full): $binary_lr_min -> $binary_lr_max"
    echo "LR range (LoRA): $binary_lr_min_lora -> $binary_lr_max_lora"
    echo "Max iterations: $binary_max_iters"
fi
echo "LR margin: $lr_margin"
echo ""
echo "Output config: $lr_config_file"
echo "Results dir: $sweep_results_dir"
echo "========================================================"

mkdir -p "$sweep_results_dir"

summary_file="$sweep_results_dir/${config_key}_summary.txt"
echo "LR Sweep Summary: $config_key" > "$summary_file"
echo "Date: $(date)" >> "$summary_file"
echo "Model: $model" >> "$summary_file"
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

        IFS=':' read -ra exp_parts <<< "${EXPERIMENT_DEFS[$exp_name]}"
        exp_method="${exp_parts[0]}"
        exp_compression="${exp_parts[1]}"
        exp_use_lora="${exp_parts[2]}"
        exp_use_second_order="${exp_parts[3]}"

        echo "" >> "$summary_file"
        echo "=== $exp_name ===" >> "$summary_file"

        best_lr=""
        best_reward="-999.0"

        if [[ "$sweep_mode" == "grid" ]]; then
            # Grid Search Mode
            if [ "$exp_use_lora" = "true" ]; then
                current_lr_grid="$lr_grid_lora"
            else
                current_lr_grid="$lr_grid"
            fi

            echo "Testing LRs: $current_lr_grid"
            echo "LR grid: $current_lr_grid" >> "$summary_file"

            IFS=',' read -ra lr_values <<< "$current_lr_grid"
            for trial_lr in "${lr_values[@]}"; do
                trial_lr=$(echo "$trial_lr" | xargs)
                echo ""
                echo "--- Testing LR: $trial_lr ---"

                trial_output_dir="$sweep_results_dir/${config_key}/${exp_name}/lr_${trial_lr}"

                mean_reward=$(run_lr_trial "$exp_method" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$trial_lr" "$exp_name" "$trial_output_dir")

                echo "LR $trial_lr -> mean_reward: $mean_reward"
                echo "  LR $trial_lr: mean_reward=$mean_reward" >> "$summary_file"

                # For RLHF, higher reward is better
                if is_valid_number "$mean_reward" && is_valid_number "$best_reward"; then
                    should_update=$(python3 -c "
reward = float('$mean_reward')
best = float('$best_reward')
margin = float('$lr_margin')
trial_lr = float('$trial_lr')
best_lr = float('${best_lr:-0}') if '${best_lr:-}' else 0

if best_lr == 0:
    print('1')
elif trial_lr < best_lr:
    # Smaller LR: update if reward is not worse
    if reward >= best * (1 - margin):
        print('1')
    else:
        print('0')
else:
    # Larger LR: only update if significantly better
    if reward > best * (1 + margin):
        print('1')
    else:
        print('0')
" 2>/dev/null)
                    if [[ "$should_update" == "1" ]]; then
                        echo "  -> New best (margin=$lr_margin)"
                        best_reward="$mean_reward"
                        best_lr="$trial_lr"
                    fi
                fi
            done

        elif [[ "$sweep_mode" == "binary" ]]; then
            # Binary Search Mode
            if [ "$exp_use_lora" = "true" ]; then
                current_lr_min="$binary_lr_min_lora"
                current_lr_max="$binary_lr_max_lora"
            else
                current_lr_min="$binary_lr_min"
                current_lr_max="$binary_lr_max"
            fi

            echo "Binary Search: $current_lr_min -> $current_lr_max (max $binary_max_iters iterations)"
            echo "Mode: Binary Search" >> "$summary_file"

            inv_phi="0.6180339887"
            log_a=$(python3 -c "import math; print(math.log10($current_lr_min))")
            log_b=$(python3 -c "import math; print(math.log10($current_lr_max))")

            declare -A lr_rewards
            lr_rewards=()

            for ((iter=1; iter<=binary_max_iters; iter++)); do
                echo ""
                echo "--- Iteration $iter/$binary_max_iters ---"

                log_c=$(python3 -c "print($log_b - $inv_phi * ($log_b - $log_a))")
                log_d=$(python3 -c "print($log_a + $inv_phi * ($log_b - $log_a))")

                lr_c=$(python3 -c "print(f'{10**$log_c:.2e}')")
                lr_d=$(python3 -c "print(f'{10**$log_d:.2e}')")

                if [[ -z "${lr_rewards[$lr_c]}" ]]; then
                    trial_output_dir="$sweep_results_dir/${config_key}/${exp_name}/binary_iter${iter}_lr_${lr_c}"
                    reward_c=$(run_lr_trial "$exp_method" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$lr_c" "$exp_name" "$trial_output_dir")
                    lr_rewards[$lr_c]="$reward_c"
                    echo "  LR $lr_c -> reward: $reward_c"
                else
                    reward_c="${lr_rewards[$lr_c]}"
                fi

                if [[ -z "${lr_rewards[$lr_d]}" ]]; then
                    trial_output_dir="$sweep_results_dir/${config_key}/${exp_name}/binary_iter${iter}_lr_${lr_d}"
                    reward_d=$(run_lr_trial "$exp_method" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$lr_d" "$exp_name" "$trial_output_dir")
                    lr_rewards[$lr_d]="$reward_d"
                    echo "  LR $lr_d -> reward: $reward_d"
                else
                    reward_d="${lr_rewards[$lr_d]}"
                fi

                # For reward, higher is better, so prefer upper half if significantly better
                prefer_lower=$(python3 -c "
reward_c = float('$reward_c')
reward_d = float('$reward_d')
margin = float('$lr_margin')
if reward_d > reward_c * (1 + margin):
    print('0')  # Upper half significantly better
else:
    print('1')  # Prefer lower half for stability
" 2>/dev/null)
                if [[ "$prefer_lower" == "1" ]]; then
                    log_b="$log_d"
                else
                    log_a="$log_c"
                fi
            done

            # Find best from all evaluated
            for lr in "${!lr_rewards[@]}"; do
                reward="${lr_rewards[$lr]}"
                if is_valid_number "$reward"; then
                    if [[ -z "$best_lr" ]] || (( $(echo "$reward > $best_reward" | bc -l) )); then
                        best_reward="$reward"
                        best_lr="$lr"
                    fi
                fi
            done
        fi

        echo ""
        echo "Best LR for $exp_name: $best_lr (mean_reward: $best_reward)"
        echo "  BEST: $best_lr (mean_reward=$best_reward)" >> "$summary_file"

        if [[ -n "$best_lr" ]]; then
            update_lr_config "$config_key" "$exp_name" "$best_lr" "$best_reward"
        fi
    done
else
    echo "ERROR: Please specify experiments with --experiments"
    echo "Example: --experiments all"
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
echo "  bash RLHF/train/train.sh --experiments all --task $task"
echo "========================================================"
