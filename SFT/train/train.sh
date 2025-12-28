#!/bin/bash

#SBATCH --job-name=Train
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bdzy-delta-gpu
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
#SBATCH --output=/u/%u/Project/Gradient-Streaming/SFT/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=none
#SBATCH --mail-user=pbb@illinois.edu
#SBATCH --mail-type="END"

cd $HOME/Project/Gradient-Streaming

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$HOME/Project/Gradient-Streaming:$PYTHONPATH"

# Base training arguments
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
--percentage=0.5 \
--selection_frac=0.5"

# Default values
data_selection="NA"  # NA, Streaming (per-layer), or GREATS (global)
optim="adamw_torch"  # Standard HF optimizer (adamw_torch, adamw_hf, etc.)
data_dir="SFT/data"
percentage=0.05
n_val=8
n_eval=500
model="meta-llama/Llama-3.2-1B"
batch_size=8
val_batch_size="1"  # Defaults to batch_size if not specified
lr=""  # Learning rate (looked up from lr_config.json if not specified)
seed=42

# LR configuration
lr_config_file="SFT/train/lr/config.json"
lr_override=""  # Set when --lr is explicitly passed
gradient_accumulation_steps=1
task="mmlu"
train_dataset=""  # Training dataset (if empty, uses task-based default)
subject="sociology"
compression=""  # NA, GraSS, or LoGra. Compression implies MeSO optimizer.
use_second_order=false  # If true, use greedy selection with second-order interactions
selection_frac="0.5"  # Fraction of samples to select (for Streaming/GREATS)
val_strategy="merged_batch"  # Validation strategy: separate_batch_factorized, separate_batch, merged_batch
use_lora=false
use_flash_attention=true
enable_profile=false
profile_steps=10

# Multi-method mode
methods=""  # Comma-separated list of methods or categories
dry_run=false
use_sbatch=false

# Fallback LRs (used only when lr_config.json has no entry)
fallback_lr_full="5e-05"
fallback_lr_lora="2e-04"

# Method definitions
# Format: "NAME:data_selection:compression:use_lora"
# Note: use_second_order is controlled by the --use_second_order flag, not per-method
declare -A METHOD_DEFS=(
    ["NA-NA-Full"]="NA::false"
    ["NA-NA-LoRA"]="NA::true"
    ["Streaming-NA-Full"]="Streaming::false"
    ["Streaming-NA-LoRA"]="Streaming::true"
    ["GREATS-NA-Full"]="GREATS::false"
    ["GREATS-NA-LoRA"]="GREATS::true"
    ["Streaming-LoGra-Full"]="Streaming:LoGra:false"
    ["GREATS-LoGra-Full"]="GREATS:LoGra:false"
)

# Category mappings
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

# LoRA-specific defaults (only used if --lora is passed)
lora_alpha=1
lora_r=32
lora_dropout=0.1

update_compressor_freq=200

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_selection)
            data_selection="$2"
            shift 2
            ;;
        --optim)
            optim="$2"
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
        --percentage)
            percentage="$2"
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
        --lr)
            lr_override="$2"
            shift 2
            ;;
        --lr_config)
            lr_config_file="$2"
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
        --selection_frac)
            selection_frac="$2"
            shift 2
            ;;
        --val_strategy)
            val_strategy="$2"
            shift 2
            ;;
        --update_compressor_freq)
            update_compressor_freq="$2"
            shift 2
            ;;
        --lora)
            use_lora=true
            shift 1
            ;;
        --flash_attention)
            use_flash_attention=true
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
        --lr_lora)
            # Deprecated: LRs are now managed via lr_config.json
            # This sets the fallback LR for LoRA methods
            fallback_lr_lora="$2"
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
        --profile)
            enable_profile=true
            shift
            ;;
        --profile_steps)
            profile_steps="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Run single or multiple training methods."
            echo ""
            echo "Multi-Method Mode:"
            echo "  --methods <list>                       Run multiple methods (see below)"
            echo "  --dry-run                              Print commands without executing"
            echo "  --sbatch                               Use sbatch instead of bash"
            echo ""
            echo "  Method names: NA-NA-Full, NA-NA-LoRA, Streaming-NA-Full, Streaming-NA-LoRA,"
            echo "                GREATS-NA-Full, GREATS-NA-LoRA, Streaming-LoGra-Full, GREATS-LoGra-Full"
            echo ""
            echo "  Categories: all, baseline, streaming, greats, Full, LoRA, compression, no-compression"
            echo ""
            echo "  Examples:"
            echo "    --methods all                        Run all 8 methods"
            echo "    --methods baseline                   Run NA-NA-Full and NA-NA-LoRA"
            echo "    --methods \"streaming,greats\"         Run all Streaming and GREATS methods"
            echo "    --methods \"NA-NA-Full,Streaming-LoGra-Full\"  Run specific methods"
            echo ""
            echo "Single Method Options:"
            echo "  --data_selection <method>              Data selection: NA, Streaming, GREATS (default: NA)"
            echo "  --compression <method>                 Compression: LoGra, GraSS (implies MeSO optimizer)"
            echo "  --use_second_order                     Use greedy selection with second-order interactions"
            echo "  --val_strategy <strategy>              Validation strategy: separate_batch_factorized (default), separate_batch, merged_batch"
            echo ""
            echo "  --lora                                 Use LoRA fine-tuning (default: full fine-tuning)"
            echo "  --lora_alpha <alpha>                   LoRA alpha (default: 1)"
            echo "  --lora_r <r>                           LoRA rank (default: 32)"
            echo "  --lora_dropout <dropout>               LoRA dropout (default: 0.1)"
            echo "  --lora_target_modules <modules>        Target modules (default: q_proj k_proj v_proj o_proj)"
            echo ""
            echo "Shared Options (apply to all methods):"
            echo "  --model <model>                        HuggingFace model path (default: meta-llama/Llama-3.2-1B)"
            echo "  --task <task>                          Task: mmlu, bbh, tydiqa, samsum, gsm8k (default: mmlu)"
            echo "  --train <dataset>                      Training dataset (default: task-based)"
            echo "  --subject <subject>                    MMLU/BBH subject (default: world_religions)"
            echo ""
            echo "  --batch_size <size>                    Training batch size (default: 8)"
            echo "  --val_batch_size <size>                Validation batch size for selection (default: 1)"
            echo "  --lr <lr>                              Learning rate override (ignores config file)"
            echo "  --lr_config <path>                     LR config file (default: SFT/train/lr/config.json)"
            echo "  --percentage <pct>                     Data sampling percentage (default: 0.05)"
            echo ""
            echo "Learning Rate Resolution:"
            echo "  1. If --lr is specified, use that value"
            echo "  2. Otherwise, look up from lr_config.json based on {train}_{task} + method"
            echo "  3. If not found, use fallback (5e-05 for full, 2e-04 for LoRA)"
            echo ""
            echo "  Run lr_sweep.sh first to populate lr_config.json with optimal LRs."
            echo "  --n_val <n>                            Validation examples (default: 8)"
            echo "  --n_eval <n>                           Evaluation examples (default: 500)"
            echo "  --seed <seed>                          Random seed (default: 42)"
            echo ""
            echo "  --update_compressor_freq <steps>       Projector refresh interval (default: 200)"
            echo "  --flash_attention                      Enable Flash Attention 2 (default: enabled)"
            echo "  --data_dir <dir>                       Data directory (default: SFT/data)"
            echo ""
            echo "Profiling:"
            echo "  --profile                              Enable PyTorch profiler (profiles first N steps)"
            echo "  --profile_steps <n>                    Number of steps to profile (default: 10)"
            echo ""
            echo "Naming convention: {selection}-{compression}-{training_type}"
            echo "  Examples: Streaming-NA-Full, GREATS-LoGra-Full, NA-NA-LoRA"
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
# Resolve method names from categories
# ========================================
resolve_methods() {
    local input="$1"
    local resolved=""

    # Split by comma
    IFS=',' read -ra items <<< "$input"
    for item in "${items[@]}"; do
        item=$(echo "$item" | xargs)  # trim whitespace

        # Check if it's a category
        if [[ -n "${CATEGORY_METHODS[$item]}" ]]; then
            if [[ -n "$resolved" ]]; then
                resolved="$resolved,${CATEGORY_METHODS[$item]}"
            else
                resolved="${CATEGORY_METHODS[$item]}"
            fi
        # Check if it's a valid method name
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

    # Remove duplicates while preserving order
    echo "$resolved" | tr ',' '\n' | awk '!seen[$0]++' | tr '\n' ',' | sed 's/,$//'
}

# ========================================
# Look up LR from config file
# ========================================
lookup_lr() {
    local config_key="$1"  # e.g., "alpaca_samsum"
    local exp_name="$2"    # e.g., "NA-NA-Full"
    local is_lora="$3"     # "true" or "false"

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
    if [ "$is_lora" = true ]; then
        echo "$fallback_lr_lora"
    else
        echo "$fallback_lr_full"
    fi
}

# ========================================
# Function to run a single method
# ========================================
run_single_method() {
    local exp_data_selection="$1"
    local exp_compression="$2"
    local exp_use_lora="$3"
    local exp_use_second_order="$4"
    local exp_lr="$5"
    local exp_name="$6"

    local exp_base_training_args="$base_training_args"

    # ========================================
    # Compression Configuration
    # ========================================
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
            *)
                echo "ERROR: Invalid compression method: $exp_compression"
                exit 1
                ;;
        esac
    fi

    # ========================================
    # Build Job Name
    # ========================================
    local job_type
    if [ "$exp_use_lora" = true ]; then
        job_type="LoRA"
    else
        job_type="Full"
    fi

    local method_str="${exp_data_selection}-${exp_compression:-NA}"
    if [[ "$exp_data_selection" != "NA" ]] && [ "$exp_use_second_order" = true ]; then
        method_str="${method_str}-2nd"
    fi

    local train_str="${train_dataset:-default}"
    local JOB_NAME
    if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
        JOB_NAME="${train_str}_${task}_${subject}-${model_name}-${method_str}-${job_type}-p${percentage}-lr${exp_lr}-b${batch_size}-v${n_val}-s${seed}"
    else
        JOB_NAME="${train_str}_${task}-${model_name}-${method_str}-${job_type}-p${percentage}-lr${exp_lr}-b${batch_size}-v${n_val}-s${seed}"
    fi

    local output_dir=/scratch/pbb/Project/Gradient-Streaming/SFT/${JOB_NAME}
    if [[ ! -d $output_dir ]]; then
        mkdir -p $output_dir
    fi

    echo ""
    echo "=============================================="
    if [[ -n "$exp_name" ]]; then
        echo "  Running: $exp_name"
    fi
    echo "=============================================="
    echo "Job name: $JOB_NAME"
    echo "Method: ${method_str}-${job_type}"
    echo "Model: $model"
    echo "Training type: $([ "$exp_use_lora" = true ] && echo "LoRA (alpha=$lora_alpha, r=$lora_r)" || echo "Full")"
    echo "Task: $task"
    echo "Training data: ${train_dataset:-default for $task}"
    echo "Selection: $exp_data_selection"
    if [[ "$exp_data_selection" != "NA" ]]; then
        echo "  Second-order: $([ "$exp_use_second_order" = true ] && echo "yes" || echo "no")"
    fi
    echo "Compression: ${exp_compression:-None}"
    echo "Batch size: $batch_size (val: ${val_batch_size:-$batch_size})"
    echo "Learning rate: $exp_lr"
    echo "Data percentage: $percentage"
    echo "Validation examples: $n_val"
    echo "Output: $output_dir"
    echo "=============================================="

    local DATA_SEED=$((seed + 1))

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

    local ID=$RANDOM
    local PORT=$((29400 + RANDOM % 10000))
    local header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d --rdzv-endpoint=localhost:$PORT \
-m SFT.train.train"

    # Build training arguments
    local training_args="$exp_base_training_args \
--model_name_or_path $MODEL_PATH \
--output_dir $output_dir \
--data_dir $data_dir \
--percentage $percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $batch_size \
--method $exp_data_selection \
--subject $subject \
--n_val $n_val \
--n_eval $n_eval \
--analysis_dataset $task \
--learning_rate $exp_lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed \
--optim $optim \
--selection_frac $selection_frac \
--val_strategy $val_strategy \
--use_flash_attention $use_flash_attention"

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

    # Add profiling arguments
    if [ "$enable_profile" = true ]; then
        training_args="$training_args --profile True --profile_steps $profile_steps"
    fi

    training_args="$training_args 2>&1 | tee $output_dir/train.log"

    # Execute or print command
    if [ "$dry_run" = true ]; then
        echo "[DRY-RUN] Would execute: $header $training_args"
    elif [ "$use_sbatch" = true ]; then
        echo "Submitting with sbatch..."
        # For sbatch, we need to write a temporary script or use sbatch --wrap
        # For now, we'll use the current script with sbatch
        echo "sbatch $0 (with current arguments)"
        # Note: sbatch mode would need the script to be called without --sbatch
        # This is a simplified version - in practice you might want separate handling
    else
        eval "$header" "$training_args"
    fi
}

# ========================================
# Main execution logic
# ========================================
if [[ -n "$methods" ]]; then
    # Multi-method mode
    resolved_methods=$(resolve_methods "$methods")

    echo ""
    echo "========================================================"
    echo "  Multi-Method Mode"
    echo "========================================================"
    echo "Task: $task"
    echo "Training data: ${train_dataset:-default}"
    echo "Model: $model"
    if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
        echo "Subject: $subject"
    fi
    echo "Percentage: $percentage"
    echo "Seed: $seed"
    echo ""
    echo "Training Settings:"
    echo "  Batch size: $batch_size"
    echo "  Val batch size: ${val_batch_size:-same as batch_size}"
    echo "  Learning rate (Full): $lr"
    echo "  Learning rate (LoRA): $lr_lora"
    echo "  N_val: $n_val"
    echo ""
    echo "Methods to run: $resolved_methods"
    echo "========================================================"

    # Count methods
    method_count=$(echo "$resolved_methods" | tr ',' '\n' | wc -l)
    current=0

    # Run each method
    IFS=',' read -ra method_list <<< "$resolved_methods"
    for method_name in "${method_list[@]}"; do
        current=$((current + 1))
        echo ""
        echo "========================================================"
        echo "  [$current/$method_count] $method_name"
        echo "========================================================"

        # Parse method definition
        IFS=':' read -ra exp_parts <<< "${METHOD_DEFS[$method_name]}"
        exp_data_selection="${exp_parts[0]}"
        exp_compression="${exp_parts[1]}"
        exp_use_lora="${exp_parts[2]}"
        # use_second_order is controlled globally via --use_second_order flag
        exp_use_second_order="$use_second_order"

        # Build config key for LR lookup (includes subject for mmlu/bbh)
        train_str="${train_dataset:-default}"
        if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
            config_key="${train_str}_${task}_${subject}"
        else
            config_key="${train_str}_${task}"
        fi

        # Look up learning rate from config
        exp_lr=$(lookup_lr "$config_key" "$method_name" "$exp_use_lora")

        run_single_method "$exp_data_selection" "$exp_compression" "$exp_use_lora" "$exp_use_second_order" "$exp_lr" "$method_name"
    done

    echo ""
    echo "========================================================"
    echo "  All methods completed!"
    echo "========================================================"
    echo "Total: $method_count methods"

else
    # Single method mode (original behavior)

    # ========================================
    # Compression Configuration
    # ========================================
# Compression implies MeSO optimizer. No compression = standard optimizer.

# Validate and configure compression method
# Default: LoGra (Gaussian projection). GraSS (random mask + SJLT) also available.
sparsification=""
projection=""
if [[ -n "$compression" ]]; then
    case "$compression" in
        LoGra)
            # LoGra: Gaussian random projection, no additional projection
            if [ "$use_lora" = true ]; then
                sparsification="normal-128*128"
                projection=""
            else
                sparsification="normal-512*512"
                projection=""
            fi
            ;;
        GraSS)
            # GraSS: Random mask sparsification + SJLT projection
            # (Available but not used in default methods)
            if [ "$use_lora" = true ]; then
                sparsification="random_mask-256*256"
                projection="sjlt-16384"
            else
                sparsification="random_mask-1024*1024"
                projection="sjlt-262144"
            fi
            ;;
        *)
            echo "ERROR: Invalid compression method: $compression"
            echo "Valid options: LoGra (default), GraSS"
            exit 1
            ;;
    esac
fi


# ========================================
# Build Job Name
# ========================================
# Naming convention: {selection}-{compression}-{training_type}
# Examples: Streaming-NA-Full, GREATS-GraSS-LoRA, NA-LoGra-Full

if [ "$use_lora" = true ]; then
    job_type="LoRA"
else
    job_type="Full"
fi

# Validate data_selection
# - NA: no data selection
# - Streaming: per-layer selection
# - GREATS: global selection
case "$data_selection" in
    NA|Streaming|GREATS)
        ;;
    *)
        echo "ERROR: Invalid data_selection: $data_selection"
        echo "Valid options: NA, Streaming, GREATS"
        exit 1
        ;;
esac

# Build method string: {selection}-{compression}
method_str="${data_selection}-${compression:-NA}"

# Add second-order suffix for data selection methods
if [[ "$data_selection" != "NA" ]] && [ "$use_second_order" = true ]; then
    method_str="${method_str}-2nd"
fi

# Build train string for job name
train_str="${train_dataset:-default}"

# Build method name for LR lookup
exp_name="${data_selection}-${compression:-NA}-${job_type}"

# Look up learning rate from config (includes subject for mmlu/bbh)
if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
    config_key="${train_str}_${task}_${subject}"
else
    config_key="${train_str}_${task}"
fi
lr=$(lookup_lr "$config_key" "$exp_name" "$use_lora")

# For MMLU/BBH, include subject in job name
if [[ "$task" == "mmlu" ]] || [[ "$task" == "bbh" ]]; then
    JOB_NAME="${train_str}_${task}_${subject}-${model_name}-${method_str}-${job_type}-p${percentage}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"
else
    JOB_NAME="${train_str}_${task}-${model_name}-${method_str}-${job_type}-p${percentage}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"
fi

output_dir=/scratch/pbb/Project/Gradient-Streaming/SFT/${JOB_NAME}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

echo "=== Training Configuration ==="
echo "Job name: $JOB_NAME"
echo "Method: ${method_str}-${job_type}"
echo ""
echo "Model: $model"
echo "Training type: $([ "$use_lora" = true ] && echo "LoRA (alpha=$lora_alpha, r=$lora_r)" || echo "Full")"
echo "Task: $task"
echo "Training data: ${train_dataset:-default for $task}"
echo ""
echo "Selection: $data_selection"
if [[ "$data_selection" != "NA" ]]; then
    echo "  Second-order: $([ "$use_second_order" = true ] && echo "yes" || echo "no")"
fi
echo "Compression: ${compression:-None}"
if [[ -n "$compression" ]]; then
    echo "  Optimizer: MeSO (compressed)"
else
    echo "  Optimizer: AdamW (standard)"
fi
echo ""
# Indicate LR source
if [[ -n "$lr_override" ]]; then
    lr_source="(--lr override)"
elif [[ -f "$lr_config_file" ]]; then
    lr_source="(from $lr_config_file)"
else
    lr_source="(fallback default)"
fi

echo "Batch size: $batch_size (val: ${val_batch_size:-$batch_size})"
echo "Learning rate: $lr $lr_source"
echo "Data percentage: $percentage"
echo "Validation examples: $n_val"
echo "Evaluation examples: $n_eval"
echo ""
echo "Output: $output_dir"
echo "==============================="


DATA_SEED=$((seed + 1))

# Use model path directly
MODEL_PATH="$model"

# Model-specific configurations (match by path pattern)
case "$model" in
    *Llama-2-13b*|*llama-2-13b*)
        base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama2_13b_finetune"
        ;;
    *Mistral-7B*|*mistral-7b*)
        base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
        ;;
esac

ID=$RANDOM
# Generate a unique port for this job to avoid conflicts with other jobs
PORT=$((29400 + RANDOM % 10000))
export header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d --rdzv-endpoint=localhost:$PORT \
-m SFT.train.train"

# Build training arguments
training_args="$base_training_args \
--model_name_or_path $MODEL_PATH \
--output_dir $output_dir \
--data_dir $data_dir \
--percentage $percentage \
--data_seed $DATA_SEED \
--per_device_train_batch_size $batch_size \
--method $data_selection \
--subject $subject \
--n_val $n_val \
--n_eval $n_eval \
--analysis_dataset $task \
--learning_rate $lr \
--gradient_accumulation_steps $gradient_accumulation_steps \
--seed $seed \
--optim $optim \
--selection_frac $selection_frac \
--val_strategy $val_strategy \
--use_flash_attention $use_flash_attention"

# Add train_dataset_names if specified
if [[ -n "$train_dataset" ]]; then
    training_args="$training_args --train_dataset_names $train_dataset"
fi

# Add val_batch_size if specified (defaults to train batch size if not specified)
if [[ -n "$val_batch_size" ]]; then
    training_args="$training_args --val_batch_size_for_selection $val_batch_size"
fi

# Add LoRA arguments if using LoRA
if [ "$use_lora" = true ]; then
    training_args="$training_args --lora True --lora_alpha $lora_alpha --lora_r $lora_r --lora_dropout $lora_dropout --lora_target_modules q_proj k_proj v_proj o_proj"
else
    training_args="$training_args --lora False"
fi

# Add gradient compression arguments (compression implies MeSO optimizer)
if [[ -n "$sparsification" ]]; then
    training_args="$training_args --sparsification $sparsification"
fi
if [[ -n "$projection" ]]; then
    training_args="$training_args --projection $projection"
fi
if [[ -n "$compression" ]]; then
    training_args="$training_args --update_compressor_freq $update_compressor_freq"
fi

# Add use_second_order for data selection
if [[ "$data_selection" != "NA" ]] && [ "$use_second_order" = true ]; then
    training_args="$training_args --use_second_order True"
fi

# Add profiling arguments
if [ "$enable_profile" = true ]; then
    training_args="$training_args --profile True --profile_steps $profile_steps"
fi

training_args="$training_args 2>&1 | tee $output_dir/train.log"

eval "$header" "$training_args"

fi  # End of single/multi method mode
