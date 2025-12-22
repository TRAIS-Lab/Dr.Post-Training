#!/bin/bash

#SBATCH --job-name=RLHF-Train
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdzy-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Gradient-Streaming/RLHF/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

cd $HOME/Project/Gradient-Streaming

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$HOME/Project/Gradient-Streaming:$PYTHONPATH"

# Base training arguments (shared across all experiments, matching SFT)
# Note: warmup_ratio=0 for PPO (pretrained policy doesn't need warmup, and LR=0 breaks first batch)
# Note: max_grad_norm removed to match reference (reference doesn't use gradient clipping)
export base_training_args="--bf16=True \
--lr_scheduler_type=constant \
--warmup_ratio=0 \
--weight_decay=0.0 \
--logging_steps=1 \
--report_to=none \
--selection_frac=0.5"

# Default values
task="toxicity"
method="NA"  # NA, Streaming, or GREATS
model="EleutherAI/gpt-neo-2.7B"
reward_model="facebook/roberta-hate-speech-dynabench-r4-target"

# Training hyperparameters
batch_size=256  # Reference uses 256 for stable gradients
lr=""  # Learning rate (looked up from config if not specified)
lr_override=""  # Set when --lr is explicitly passed
n_val=128
val_batch_size="32"
selection_frac=0.5
max_steps=1000
seed=42

# LR configuration
# Reference: archive/LDA-ORL-main/rlhf-toxicity/scripts/run_train_std.sh uses 1e-5
lr_config_file="RLHF/train/lr/config.json"
fallback_lr_full="1e-5"
fallback_lr_lora="1e-5"

# LoRA settings
use_lora=false
lora_r=16
lora_alpha=32

# Compression settings
compression=""  # LoGra or GraSS
update_compressor_freq=200

# PPO settings (matching reference implementation)
# Reference: archive/LDA-ORL-main/rlhf-toxicity/scripts/run_train_std.sh
ppo_epochs=4
mini_batch_size=1  # Reference uses 1: one optimizer.step() per sample within each PPO epoch
forward_batch_size=256  # Reference uses tracin_batch_size=256 for forward passes (GPU efficiency)
init_kl_coef=0.04  # Reference uses 0.04 (NOT 0.2)
adap_kl_ctrl=true  # Reference uses adaptive KL control
target_kl=6.0      # Reference default
max_new_tokens=30
min_new_tokens=20

# Validation settings
val_strategy="random"
val_loss_type="reward_weighted"  # Sequence-level attribution: f^seq(θ) = -E[log π_θ(y|x) * Â(x,y)]
use_second_order=false

# Output
output_dir=""

# Multi-experiment mode
experiments=""
dry_run=false
use_sbatch=false

# ========================================
# Experiment Definitions (8 experiments, matching SFT)
# ========================================
# Format: "method:compression:use_lora:use_second_order"
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

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            task="$2"
            shift 2
            ;;
        --method)
            method="$2"
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
        --lr)
            lr_override="$2"
            shift 2
            ;;
        --lr_config)
            lr_config_file="$2"
            shift 2
            ;;
        --n_val)
            n_val="$2"
            shift 2
            ;;
        --val_batch_size)
            val_batch_size="$2"
            shift 2
            ;;
        --selection_frac)
            selection_frac="$2"
            shift 2
            ;;
        --max_steps)
            max_steps="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --lora)
            use_lora=true
            shift 1
            ;;
        --lora_r)
            lora_r="$2"
            shift 2
            ;;
        --lora_alpha)
            lora_alpha="$2"
            shift 2
            ;;
        --compression)
            compression="$2"
            shift 2
            ;;
        --update_compressor_freq)
            update_compressor_freq="$2"
            shift 2
            ;;
        --ppo_epochs)
            ppo_epochs="$2"
            shift 2
            ;;
        --forward_batch_size)
            forward_batch_size="$2"
            shift 2
            ;;
        --kl_coef)
            kl_coef="$2"
            shift 2
            ;;
        --max_new_tokens)
            max_new_tokens="$2"
            shift 2
            ;;
        --val_strategy)
            val_strategy="$2"
            shift 2
            ;;
        --val_loss_type)
            val_loss_type="$2"
            shift 2
            ;;
        --use_second_order)
            use_second_order=true
            shift 1
            ;;
        --output_dir)
            output_dir="$2"
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
        --sbatch)
            use_sbatch=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run single or multiple RLHF training experiments."
            echo ""
            echo "Multi-Experiment Mode:"
            echo "  --experiments <list>       Run multiple experiments (see below)"
            echo "  --dry-run                  Print commands without executing"
            echo "  --sbatch                   Use sbatch instead of bash"
            echo ""
            echo "  Experiment names:"
            echo "    NA-NA-full, NA-NA-lora, Streaming-NA-full, Streaming-NA-lora,"
            echo "    GREATS-NA-full, GREATS-NA-lora, Streaming-LoGra-full, GREATS-LoGra-full"
            echo ""
            echo "  Categories: all, baseline, streaming, greats, full, lora, compression, no-compression"
            echo ""
            echo "Single Experiment Options:"
            echo "  --task <task>              Task: toxicity, imdb (default: toxicity)"
            echo "  --method <method>          Selection: NA, Streaming, GREATS (default: NA)"
            echo "  --model <model>            Policy model"
            echo "  --reward_model <model>     Reward model"
            echo ""
            echo "  --lora                     Enable LoRA fine-tuning (default: full fine-tuning)"
            echo "  --lora_r <r>               LoRA rank (default: 16)"
            echo "  --lora_alpha <alpha>       LoRA alpha (default: 32)"
            echo ""
            echo "  --compression <method>     Compression: LoGra, GraSS (implies MeSO optimizer)"
            echo "  --use_second_order         Enable second-order selection"
            echo ""
            echo "Shared Options:"
            echo "  --batch_size <size>        Training batch size (default: 64)"
            echo "  --lr <lr>                  Learning rate override (ignores config file)"
            echo "  --lr_config <path>         LR config file (default: RLHF/train/lr/config.json)"
            echo "  --n_val <n>                Validation samples (default: 128)"
            echo "  --selection_frac <frac>    Fraction to select (default: 0.5)"
            echo "  --max_steps <steps>        Maximum training steps (default: 200)"
            echo "  --seed <seed>              Random seed (default: 42)"
            echo ""
            echo "Learning Rate Resolution:"
            echo "  1. If --lr is specified, use that value"
            echo "  2. Otherwise, look up from lr/config.json based on task + experiment"
            echo "  3. If not found, use fallback (1e-5)"
            echo ""
            echo "  Run lr/lr_sweep.sh first to populate config.json with optimal LRs."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
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
            echo "Valid experiments: ${!EXPERIMENT_DEFS[*]}"
            echo "Valid categories: ${!CATEGORY_EXPERIMENTS[*]}"
            exit 1
        fi
    done

    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# ========================================
# Look up LR from config file
# ========================================
lookup_lr() {
    local config_key="$1"
    local exp_name="$2"
    local is_lora="$3"

    # If lr_override is set, use it
    if [[ -n "$lr_override" ]]; then
        echo "$lr_override"
        return
    fi

    # Try to look up from config file
    if [[ -f "$lr_config_file" ]]; then
        local looked_up_lr
        looked_up_lr=$(python3 -c "
import json
import sys
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

    # Fallback to defaults
    if [ "$is_lora" = "true" ]; then
        echo "$fallback_lr_lora"
    else
        echo "$fallback_lr_full"
    fi
}

# ========================================
# Function to run a single experiment
# ========================================
run_single_experiment() {
    local exp_method="$1"
    local exp_compression="$2"
    local exp_use_lora="$3"
    local exp_use_second_order="$4"
    local exp_lr="$5"
    local exp_name="$6"

    # Start with base training args
    local exp_training_args="$base_training_args"

    # ========================================
    # Compression Configuration
    # ========================================
    local exp_sparsification=""
    local exp_projection=""
    if [[ -n "$exp_compression" ]]; then
        case "$exp_compression" in
            LoGra)
                exp_projection="Gaussian-256"
                ;;
            GraSS)
                exp_sparsification="Rademacher-64*64"
                ;;
            *)
                echo "ERROR: Invalid compression method: $exp_compression"
                exit 1
                ;;
        esac
    fi

    # Derive compression name for job
    local compression_name="NA"
    if [[ -n "$exp_sparsification" ]]; then
        compression_name="GraSS"
    elif [[ -n "$exp_projection" ]]; then
        compression_name="LoGra"
    fi

    # ========================================
    # Build Job Name
    # ========================================
    local model_short=$(echo "$model" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]')

    local training_type
    if [[ "$exp_use_lora" == "true" ]]; then
        training_type="lora"
    else
        training_type="full"
    fi

    local JOB_NAME="${task}-${exp_method}-${compression_name}-${model_short}-${training_type}-lr${exp_lr}-b${batch_size}-v${n_val}-s${seed}"

    local exp_output_dir
    if [[ -z "$output_dir" ]]; then
        exp_output_dir="/scratch/pbb/Project/Gradient-Streaming/RLHF/${JOB_NAME}"
    else
        exp_output_dir="$output_dir"
    fi

    if [[ ! -d $exp_output_dir ]]; then
        mkdir -p $exp_output_dir
    fi

    # Indicate LR source
    local lr_source
    if [[ -n "$lr_override" ]]; then
        lr_source="(--lr override)"
    elif [[ -f "$lr_config_file" ]]; then
        lr_source="(from config)"
    else
        lr_source="(fallback)"
    fi

    echo ""
    echo "=============================================="
    if [[ -n "$exp_name" ]]; then
        echo "  Running: $exp_name"
    fi
    echo "=============================================="
    echo "Job name: $JOB_NAME"
    echo "Task: $task"
    echo "Method: $exp_method"
    echo "Model: $model"
    echo "Compression: $compression_name"
    echo "Training type: $training_type"
    if [[ "$exp_method" != "NA" ]]; then
        echo "Second-order: $([ "$exp_use_second_order" = "true" ] && echo "yes" || echo "no")"
    fi
    echo "Learning rate: $exp_lr $lr_source"
    echo "Output: $exp_output_dir"
    echo "=============================================="

    # ========================================
    # Build training arguments
    # ========================================
    local training_args="$exp_training_args \
--task=$task \
--method=$exp_method \
--model_name_or_path=$model \
--reward_model_name=$reward_model \
--per_device_train_batch_size=$batch_size \
--learning_rate=$exp_lr \
--n_val=$n_val \
--selection_frac=$selection_frac \
--max_steps=$max_steps \
--seed=$seed \
--ppo_epochs=$ppo_epochs \
--mini_batch_size=$mini_batch_size \
--forward_batch_size=$forward_batch_size \
--init_kl_coef=$init_kl_coef \
--adap_kl_ctrl=$adap_kl_ctrl \
--target_kl=$target_kl \
--max_new_tokens=$max_new_tokens \
--min_new_tokens=$min_new_tokens \
--val_strategy=$val_strategy \
--val_loss_type=$val_loss_type \
--output_dir=$exp_output_dir"

    # Add LoRA settings
    if [[ "$exp_use_lora" == "true" ]]; then
        training_args="$training_args --lora=True --lora_r=$lora_r --lora_alpha=$lora_alpha"
    else
        training_args="$training_args --lora=False"
    fi

    # Add compression settings
    if [[ -n "$exp_sparsification" ]]; then
        training_args="$training_args --sparsification=$exp_sparsification"
    fi
    if [[ -n "$exp_projection" ]]; then
        training_args="$training_args --projection=$exp_projection"
    fi
    if [[ -n "$exp_sparsification" || -n "$exp_projection" ]]; then
        training_args="$training_args --update_compressor_freq=$update_compressor_freq"
    fi

    # Add val_batch_size if specified
    if [[ -n "$val_batch_size" ]]; then
        training_args="$training_args --val_batch_size=$val_batch_size"
    fi

    # Add second-order flag
    if [[ "$exp_use_second_order" == "true" ]]; then
        training_args="$training_args --use_second_order=True"
    fi

    # Add log file
    training_args="$training_args 2>&1 | tee $exp_output_dir/train.log"

    local cmd="python RLHF/train/train.py $training_args"

    echo "Running command:"
    echo "$cmd"
    echo ""

    # Execute or print command
    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] Would execute above command"
    else
        eval "$cmd"
    fi
}

# ========================================
# Main execution logic
# ========================================
config_key="${task}"

if [[ -n "$experiments" ]]; then
    # ========================================
    # Multi-experiment mode
    # ========================================
    resolved_experiments=$(resolve_experiments "$experiments")

    echo ""
    echo "========================================================"
    echo "  RLHF Multi-Experiment Mode"
    echo "========================================================"
    echo "Task: $task"
    echo "Model: $model"
    echo "Seed: $seed"
    echo ""
    echo "Training Settings:"
    echo "  Batch size: $batch_size"
    echo "  Val batch size: ${val_batch_size:-same as n_val}"
    echo "  N_val: $n_val"
    echo "  Max steps: $max_steps"
    echo "  Selection fraction: $selection_frac"
    echo ""
    echo "PPO Settings:"
    echo "  PPO epochs: $ppo_epochs"
    echo "  Init KL coefficient: $init_kl_coef"
    echo "  Adaptive KL control: $adap_kl_ctrl"
    echo "  Target KL: $target_kl"
    echo ""
    echo "LR Config: $lr_config_file"
    echo "Experiments to run: $resolved_experiments"
    echo "========================================================"

    exp_count=$(echo "$resolved_experiments" | tr ',' '\n' | wc -l)
    current=0

    IFS=',' read -ra exp_list <<< "$resolved_experiments"
    for exp_name in "${exp_list[@]}"; do
        current=$((current + 1))
        echo ""
        echo "========================================================"
        echo "  [$current/$exp_count] $exp_name"
        echo "========================================================"

        IFS=':' read -ra exp_parts <<< "${EXPERIMENT_DEFS[$exp_name]}"
        exp_method="${exp_parts[0]}"
        exp_compression="${exp_parts[1]}"
        exp_use_lora="${exp_parts[2]}"
        exp_use_second_order="${exp_parts[3]}"

        # Look up learning rate
        exp_lr=$(lookup_lr "$config_key" "$exp_name" "$exp_use_lora")

        run_single_experiment "$exp_method" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$exp_lr" "$exp_name"
    done

    echo ""
    echo "========================================================"
    echo "  All experiments completed!"
    echo "========================================================"
    echo "Total: $exp_count experiments"

else
    # ========================================
    # Single experiment mode
    # ========================================

    # Validate method
    if [[ ! "$method" =~ ^(NA|Streaming|GREATS)$ ]]; then
        echo "Error: method must be NA, Streaming, or GREATS"
        exit 1
    fi

    # ========================================
    # Compression Configuration
    # ========================================
    sparsification=""
    projection=""
    if [[ -n "$compression" ]]; then
        case "$compression" in
            LoGra)
                projection="Gaussian-256"
                ;;
            GraSS)
                sparsification="Rademacher-64*64"
                ;;
            *)
                echo "ERROR: Invalid compression method: $compression"
                echo "Valid options: LoGra, GraSS"
                exit 1
                ;;
        esac
    fi

    # Build experiment name for LR lookup
    local_compression=""
    if [[ -n "$compression" ]]; then
        local_compression="$compression"
    fi

    if [[ "$use_lora" == "true" ]]; then
        exp_name="${method}-${local_compression:-NA}-lora"
    else
        exp_name="${method}-${local_compression:-NA}-full"
    fi

    # Look up learning rate
    lr=$(lookup_lr "$config_key" "$exp_name" "$use_lora")

    # Run single experiment
    run_single_experiment "$method" "$local_compression" "$use_lora" "$use_second_order" "$lr" ""
fi
