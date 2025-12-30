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

# Base training arguments (shared across all methods, matching SFT)
# Note: warmup_ratio=0 for PPO (pretrained policy doesn't need warmup, and LR=0 breaks first batch)
# Note: max_grad_norm removed to match reference (reference doesn't use gradient clipping)
export base_training_args="--bf16=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--report_to=none"

# Default values
task="toxicity"
method="NA"  # NA, Streaming, or GREATS
model="EleutherAI/gpt-neo-2.7B"
reward_model="facebook/roberta-hate-speech-dynabench-r4-target"
batch_size=256
max_steps=-1  # -1 means no step limit (use epochs instead)
epochs=1  # Number of training epochs (used when max_steps <= 0)
seed=42

# PPO settings
ppo_epochs=4
mini_batch_size=1
filter_frac=1.0
init_kl_coef=0.04
kl_penalty="kl"  # Options: kl, abs, mse, full (use "kl" for token-level stability)
adap_kl_ctrl=true
target_kl=0.1
max_new_tokens=30
min_new_tokens=10
use_second_order=false
# Toxicity evaluation settings
enable_toxicity_eval=true
eval_interval=1  # 0 = end of epoch only, N > 0 = every N steps
eval_n_samples=500
eval_batch_size=256
eval_on_step_generations=true  # Evaluate toxicity on each step's generations
use_flash_attention=true

# LR configuration
lr_config_file="RLHF/train/lr/config.json"
lr_override=""  # Set when --lr is explicitly passed
# Default LRs (used when lr_config.json has no entry for the method)
default_lr_full="1e-5"
default_lr_lora="5e-6"

# LoRA settings
use_lora=false
lora_r=16
lora_alpha=32
lora_target_modules="q_proj k_proj v_proj o_proj"


# Compression settings
compression=""  # LoGra or GraSS
update_compressor_freq=200

# Output
output_dir=""

# Multi-method mode
methods=""
dry_run=false
use_sbatch=false

# ========================================
# Experiment Definitions (8 methods, matching SFT)
# ========================================
# Format: "method:compression:use_lora:use_second_order"
declare -A METHOD_DEFS=(
    ["NA-NA-Full"]="NA::false:false"
    ["NA-NA-LoRA"]="NA::true:false"
    ["Streaming-NA-Full"]="Streaming::false:true"
    ["Streaming-NA-LoRA"]="Streaming::true:true"
    ["GREATS-NA-Full"]="GREATS::false:true"
    ["GREATS-NA-LoRA"]="GREATS::true:true"
    ["Streaming-LoGra-Full"]="Streaming:LoGra:false:true"
    ["GREATS-LoGra-Full"]="GREATS:LoGra:false:true"
)

# Category mappings (matching SFT)
declare -A CATEGORY_METHODS=(
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
        --filter_frac)
            filter_frac="$2"
            shift 2
            ;;
        --max_steps)
            max_steps="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
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
        --lora_target_modules)
            lora_target_modules="$2"
            shift 2
            ;;
        --flash_attention)
            use_flash_attention=true
            shift 1
            ;;
        --no_flash_attention)
            use_flash_attention=false
            shift 1
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
        --mini_batch_size)
            mini_batch_size="$2"
            shift 2
            ;;
        --kl_coef)
            kl_coef="$2"
            shift 2
            ;;
        --kl_penalty)
            kl_penalty="$2"
            shift 2
            ;;
        --max_new_tokens)
            max_new_tokens="$2"
            shift 2
            ;;
        --use_second_order)
            use_second_order=true
            shift 1
            ;;
        --enable_toxicity_eval)
            enable_toxicity_eval=true
            shift 1
            ;;
        --disable_toxicity_eval|--no_toxicity_eval)
            enable_toxicity_eval=false
            shift 1
            ;;
        --eval_interval)
            eval_interval="$2"
            shift 2
            ;;
        --eval_n_samples)
            eval_n_samples="$2"
            shift 2
            ;;
        --eval_batch_size)
            eval_batch_size="$2"
            shift 2
            ;;
        --eval_on_step_generations)
            eval_on_step_generations=true
            shift 1
            ;;
        --no_eval_on_step_generations)
            eval_on_step_generations=false
            shift 1
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --methods)
            methods="$2"
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
            echo "Run single or multiple RLHF training methods."
            echo ""
            echo "Multi-Experiment Mode:"
            echo "  --methods <list>       Run multiple methods (see below)"
            echo "  --dry-run                  Print commands without executing"
            echo "  --sbatch                   Use sbatch instead of bash"
            echo ""
            echo "  Experiment names:"
            echo "    NA-NA-Full, NA-NA-LoRA, Streaming-NA-Full, Streaming-NA-LoRA,"
            echo "    GREATS-NA-Full, GREATS-NA-LoRA, Streaming-LoGra-Full, GREATS-LoGra-Full"
            echo ""
            echo "  Categories: all, baseline, streaming, greats, Full, LoRA, compression, no-compression"
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
            echo "  --lora_target_modules <m>  Target modules (default: q_proj k_proj v_proj o_proj)"
            echo ""
            echo "  --flash_attention          Enable Flash Attention 2 (default: enabled)"
            echo "  --no_flash_attention       Disable Flash Attention 2"
            echo ""
            echo "  --compression <method>     Compression: LoGra, GraSS (implies MeSO optimizer)"
            echo "  --use_second_order         Enable second-order selection"
            echo ""
            echo "Shared Options:"
            echo "  --batch_size <size>        Training batch size (default: 256)"
            echo "  --lr <lr>                  Learning rate override (ignores config file)"
            echo "  --lr_config <path>         LR config file (default: RLHF/train/lr/config.json)"
            echo "  --filter_frac <frac>       Fraction of negative samples to drop (default: 1.0)"
            echo "  --max_steps <steps>        Maximum training steps (default: -1, meaning no limit)"
            echo "  --epochs <n>               Number of training epochs (default: 1, used when max_steps <= 0)"
            echo "  --seed <seed>              Random seed (default: 42)"
            echo ""
            echo "PPO Options:"
            echo "  --ppo_epochs <n>           PPO epochs per batch (default: 4)"
            echo "  --mini_batch_size <n>      Mini-batch size for PPO updates (default: 8)"
            echo "  --kl_coef <coef>           Initial KL penalty coefficient (default: 0.2)"
            echo "  --kl_penalty <mode>        KL penalty mode: kl, abs, mse, full (default: full)"
            echo "  --max_new_tokens <n>       Maximum new tokens to generate (default: 30)"
            echo ""
            echo "Note: Uses self-referencing validation (training buffer as validation set)"
            echo ""
            echo "Toxicity Evaluation Options:"
            echo "  --enable_toxicity_eval     Enable toxicity evaluation (default: true)"
            echo "  --disable_toxicity_eval    Disable toxicity evaluation"
            echo "  --eval_interval <n>        Evaluate every N steps (0=epoch end only, default: 1)"
            echo "  --eval_n_samples <n>       Samples for full evaluation (default: 500)"
            echo "  --eval_batch_size <n>      Batch size for generation (default: 256)"
            echo "  --eval_on_step_generations Evaluate toxicity on each step's generations (default: true)"
            echo "  --no_eval_on_step_generations Disable per-step toxicity evaluation"
            echo ""
            echo "  Note: Evaluation uses a DIFFERENT classifier (DaNLP/da-electra-hatespeech-detection)"
            echo "        than the reward model to provide unbiased toxicity measurement."
            echo ""
            echo "Learning Rate Resolution:"
            echo "  1. If --lr is specified, use that value"
            echo "  2. Otherwise, look up from lr/config.json based on task + method"
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
# Resolve method names from categories
# ========================================
resolve_methods() {
    local input="$1"
    local resolved=""

    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)

        if [[ -n "${CATEGORY_METHODS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,${CATEGORY_METHODS[$item]}"
            else
                resolved="${CATEGORY_METHODS[$item]}"
            fi
        elif [[ -n "${METHOD_DEFS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,$item"
            else
                resolved="$item"
            fi
        else
            echo "ERROR: Unknown method or category: $item"
            echo "Valid methods: ${!METHOD_DEFS[*]}"
            echo "Valid categories: ${!CATEGORY_METHODS[*]}"
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
        echo "$default_lr_lora"
    else
        echo "$default_lr_full"
    fi
}

# ========================================
# Function to run a single method
# ========================================
run_single_method() {
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
        training_type="LoRA"
    else
        training_type="Full"
    fi

    local JOB_NAME="${task}-${model_short}-${exp_method}-${compression_name}-${training_type}-lr${exp_lr}-b${batch_size}-s${seed}"

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
--filter_frac=$filter_frac \
--max_steps=$max_steps \
--num_train_epochs=$epochs \
--seed=$seed \
--ppo_epochs=$ppo_epochs \
--mini_batch_size=$mini_batch_size \
--init_kl_coef=$init_kl_coef \
--kl_penalty=$kl_penalty \
--adap_kl_ctrl=$adap_kl_ctrl \
--target_kl=$target_kl \
--max_new_tokens=$max_new_tokens \
--min_new_tokens=$min_new_tokens \
--output_dir=$exp_output_dir"

    # Add LoRA settings
    if [[ "$exp_use_lora" == "true" ]]; then
        training_args="$training_args --lora=True --lora_r=$lora_r --lora_alpha=$lora_alpha --lora_target_modules $lora_target_modules"
    else
        training_args="$training_args --lora=False"
    fi

    # Add flash attention setting
    if [[ "$use_flash_attention" == "true" ]]; then
        training_args="$training_args --use_flash_attention=True"
    else
        training_args="$training_args --use_flash_attention=False"
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

    # Add second-order flag
    if [[ "$exp_use_second_order" == "true" ]]; then
        training_args="$training_args --use_second_order=True"
    fi

    # Add toxicity evaluation settings
    if [[ "$enable_toxicity_eval" == "true" ]]; then
        training_args="$training_args --enable_toxicity_eval=True"
        training_args="$training_args --eval_interval=$eval_interval"
        training_args="$training_args --eval_n_samples=$eval_n_samples"
        training_args="$training_args --eval_batch_size=$eval_batch_size"
        if [[ "$eval_on_step_generations" == "true" ]]; then
            training_args="$training_args --eval_on_step_generations=True"
        else
            training_args="$training_args --eval_on_step_generations=False"
        fi
    else
        training_args="$training_args --enable_toxicity_eval=False"
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

if [[ -n "$methods" ]]; then
    # ========================================
    # Multi-method mode
    # ========================================
    resolved_methods=$(resolve_methods "$methods")

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
    echo "  Epochs: $epochs"
    echo "  Max steps: $max_steps (use -1 for epoch-based training)"
    echo "  Filter fraction: $filter_frac"
    echo "  Validation: self-reference (training buffer)"
    echo ""
    echo "PPO Settings:"
    echo "  PPO epochs: $ppo_epochs"
    echo "  Mini-batch size: $mini_batch_size"
    echo "  Init KL coefficient: $init_kl_coef"
    echo "  Adaptive KL control: $adap_kl_ctrl"
    echo "  Target KL: $target_kl"
    echo ""
    echo "Toxicity Evaluation:"
    echo "  Enabled: $enable_toxicity_eval"
    if [[ "$enable_toxicity_eval" == "true" ]]; then
        echo "  Eval interval: $eval_interval (0=epoch end only)"
        echo "  Eval samples: $eval_n_samples"
        echo "  Eval batch size: $eval_batch_size"
        echo "  Eval on step generations: $eval_on_step_generations"
    fi
    echo ""
    echo "LR Config: $lr_config_file"
    echo "Experiments to run: $resolved_methods"
    echo "========================================================"

    method_count=$(echo "$resolved_methods" | tr ',' '\n' | wc -l)
    current=0

    IFS=',' read -ra exp_list <<< "$resolved_methods"
    for exp_name in "${exp_list[@]}"; do
        current=$((current + 1))
        echo ""
        echo "========================================================"
        echo "  [$current/$method_count] $exp_name"
        echo "========================================================"

        IFS=':' read -ra exp_parts <<< "${METHOD_DEFS[$exp_name]}"
        exp_method="${exp_parts[0]}"
        exp_compression="${exp_parts[1]}"
        exp_use_lora="${exp_parts[2]}"
        exp_use_second_order="${exp_parts[3]}"

        # Look up learning rate
        exp_lr=$(lookup_lr "$config_key" "$exp_name" "$exp_use_lora")

        run_single_method "$exp_method" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$exp_lr" "$exp_name"
    done

    echo ""
    echo "========================================================"
    echo "  All methods completed!"
    echo "========================================================"
    echo "Total: $method_count methods"

else
    # ========================================
    # Single method mode
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

    # Build method name for LR lookup
    local_compression=""
    if [[ -n "$compression" ]]; then
        local_compression="$compression"
    fi

    if [[ "$use_lora" == "true" ]]; then
        exp_name="${method}-${local_compression:-NA}-LoRA"
    else
        exp_name="${method}-${local_compression:-NA}-Full"
    fi

    # Look up learning rate
    lr=$(lookup_lr "$config_key" "$exp_name" "$use_lora")

    # Run single method
    run_single_method "$method" "$local_compression" "$use_lora" "$use_second_order" "$lr" ""
fi
