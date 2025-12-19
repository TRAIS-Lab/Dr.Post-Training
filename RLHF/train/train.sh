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

# Set PYTHONPATH
export PYTHONPATH="$HOME/Project/Gradient-Streaming:$PYTHONPATH"

# Default values
task="toxicity"
method="NA"  # NA, Streaming, or GREATS
model="EleutherAI/gpt-neo-125m"
reward_model="facebook/roberta-hate-speech-dynabench-r4-target"

# Training hyperparameters
batch_size=64
lr=1e-5
n_val=16
val_batch_size=""  # Defaults to n_val if not specified
selection_frac=0.5
max_steps=200
seed=42

# LoRA settings
use_lora=true
lora_r=16
lora_alpha=32

# Compression settings (empty = no compression)
sparsification=""  # e.g., "Rademacher-64*64"
projection=""      # e.g., "Gaussian-256"
update_compressor_freq=200

# PPO settings
ppo_epochs=4
kl_coef=0.04
max_new_tokens=30
min_new_tokens=10

# Validation settings
val_strategy="random"  # random or top
val_loss_type="logprob"  # logprob, reward_weighted, advantage_weighted
use_second_order=false

# Output
output_dir=""

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
            lr="$2"
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
        --no_lora)
            use_lora=false
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
        --sparsification)
            sparsification="$2"
            shift 2
            ;;
        --projection)
            projection="$2"
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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate method
if [[ ! "$method" =~ ^(NA|Streaming|GREATS)$ ]]; then
    echo "Error: method must be NA, Streaming, or GREATS"
    exit 1
fi

# Derive compression name
compression="NA"
if [[ -n "$sparsification" ]]; then
    compression="GraSS"
elif [[ -n "$projection" ]]; then
    compression="LoGra"
fi

# Get model short name
model_short=$(echo "$model" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]')

# Training type
if [[ "$use_lora" == "true" ]]; then
    training_type="lora"
else
    training_type="full"
fi

# Build experiment name
# Format: {task}-{method}-{compression}-{model}-{training_type}-lr{lr}-b{batch}-v{nval}-s{seed}
experiment_name="${task}-${method}-${compression}-${model_short}-${training_type}-lr${lr}-b${batch_size}-v${n_val}-s${seed}"

# Output directory
if [[ -z "$output_dir" ]]; then
    output_dir="/scratch/pbb/Project/Gradient-Streaming/RLHF/${experiment_name}"
fi

echo "========================================"
echo "RLHF Training Configuration"
echo "========================================"
echo "Task: $task"
echo "Method: $method"
echo "Model: $model"
echo "Compression: $compression"
echo "Training type: $training_type"
echo "Experiment: $experiment_name"
echo "Output: $output_dir"
echo "========================================"

# Build command
cmd="python RLHF/train/train.py \
    --task=$task \
    --method=$method \
    --model_name_or_path=$model \
    --reward_model_name=$reward_model \
    --per_device_train_batch_size=$batch_size \
    --learning_rate=$lr \
    --n_val=$n_val \
    --selection_frac=$selection_frac \
    --max_steps=$max_steps \
    --seed=$seed \
    --ppo_epochs=$ppo_epochs \
    --kl_coef=$kl_coef \
    --max_new_tokens=$max_new_tokens \
    --min_new_tokens=$min_new_tokens \
    --val_strategy=$val_strategy \
    --val_loss_type=$val_loss_type \
    --output_dir=$output_dir \
    --bf16=True \
    --report_to=none"

# Add LoRA settings
if [[ "$use_lora" == "true" ]]; then
    cmd="$cmd --lora=True --lora_r=$lora_r --lora_alpha=$lora_alpha"
else
    cmd="$cmd --lora=False"
fi

# Add compression settings
if [[ -n "$sparsification" ]]; then
    cmd="$cmd --sparsification=$sparsification"
fi
if [[ -n "$projection" ]]; then
    cmd="$cmd --projection=$projection"
fi
if [[ -n "$sparsification" || -n "$projection" ]]; then
    cmd="$cmd --update_compressor_freq=$update_compressor_freq"
fi

# Add val_batch_size if specified
if [[ -n "$val_batch_size" ]]; then
    cmd="$cmd --val_batch_size=$val_batch_size"
fi

# Add second-order flag
if [[ "$use_second_order" == "true" ]]; then
    cmd="$cmd --use_second_order=True"
fi

echo "Running command:"
echo "$cmd"
echo ""

# Execute
$cmd
