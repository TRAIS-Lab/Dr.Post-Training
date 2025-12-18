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
#SBATCH --output=/u/%u/Project/Efficient-Fine-Tuning/experiment_rlhf/log/%x-%j.log

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none

cd $HOME/Project/Efficient-Fine-Tuning

# Set PYTHONPATH to include project root for imports
export PYTHONPATH="$HOME/Project/Efficient-Fine-Tuning:$PYTHONPATH"

# ========================================
# Default Values
# ========================================

# Model
model="llama3-1b"

# Selection
selection_method="GREATS"  # GREATS, GradNorm, or NA
selection_frac=0.5
selection_mode="per_layer"  # global or per_layer (consistent with SFT)
val_loss_type="logprob"  # logprob, reward_weighted, advantage_weighted
use_second_order=false

# Compression
compression="GraSS"  # GraSS, LoGra
sparsifier_dim=32    # Sparsifier projection dimension
projector_dim=512    # Projector projection dimension
sparsifier_type="random_mask"  # random_mask, identity
projector_type="normal"  # normal, identity
update_compressor_freq=200

# Training (consistent naming with SFT)
lr=1.41e-5
per_device_train_batch_size=32  # Renamed from batch_size for consistency with SFT
mini_batch_size=4
val_size=256  # Total validation pool size (n_val equivalent)
val_batch_size=""  # Validation batch size for selection (defaults to val_size if not specified)
epochs=1  # Renamed from ppo_epochs for consistency with SFT
steps=1000
seed=42

# LoRA
use_lora=true
lora_r=16
lora_alpha=32

# PPO-specific
kl_coef=0.1
gamma=1.0
lam=0.95
cliprange=0.2
cliprange_value=0.2

# Logging
output_dir="experiment_rlhf/outputs"
log_interval=""  # Log every N steps (defaults to steps // 10 if empty)

# Dataset
prompt_dataset="allenai/real-toxicity-prompts"
val_dataset="Anthropic/hh-rlhf"

# Flags
dry_run=false
MeSO=false

# ========================================
# Parse Arguments
# ========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            model="$2"
            shift 2
            ;;
        --selection_method)
            selection_method="$2"
            shift 2
            ;;
        --selection_frac)
            selection_frac="$2"
            shift 2
            ;;
        --selection_mode)
            selection_mode="$2"
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
        --compression)
            compression="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --per_device_train_batch_size|--batch_size)
            per_device_train_batch_size="$2"
            shift 2
            ;;
        --mini_batch_size)
            mini_batch_size="$2"
            shift 2
            ;;
        --val_size)
            val_size="$2"
            shift 2
            ;;
        --val_batch_size)
            val_batch_size="$2"
            shift 2
            ;;
        --epochs|--ppo_epochs)
            epochs="$2"
            shift 2
            ;;
        --steps)
            steps="$2"
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
        --kl_coef)
            kl_coef="$2"
            shift 2
            ;;
        --MeSO)
            MeSO=true
            shift 1
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        --prompt_dataset)
            prompt_dataset="$2"
            shift 2
            ;;
        --val_dataset)
            val_dataset="$2"
            shift 2
            ;;
        --log_interval)
            log_interval="$2"
            shift 2
            ;;
        --dry_run)
            dry_run=true
            shift 1
            ;;
        --help|-h)
            echo "RLHF Training with Layer-wise Selection"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Model Options:"
            echo "  --model <model>                    Model: llama3-1b, llama2-7b, etc. (default: llama3-1b)"
            echo ""
            echo "Selection Options:"
            echo "  --selection_method <m>             Selection: GREATS, GradNorm, NA (default: GREATS)"
            echo "  --selection_frac <f>               Fraction to select (default: 0.5)"
            echo "  --selection_mode <mode>            Selection mode: global, per_layer (default: per_layer)"
            echo "  --val_loss_type <type>             Validation loss: logprob, reward_weighted, advantage_weighted"
            echo "  --use_second_order                 Use second-order selection"
            echo ""
            echo "Compression Options:"
            echo "  --compression <method>             Compression: GraSS, LoGra (default: GraSS)"
            echo "  --MeSO                             Use MeSO compressed optimizer"
            echo ""
            echo "Training Options (consistent with SFT naming):"
            echo "  --lr <lr>                          Learning rate (default: 1.41e-5)"
            echo "  --per_device_train_batch_size <n>  Training batch size (default: 32)"
            echo "  --mini_batch_size <size>           PPO mini-batch size (default: 4)"
            echo "  --val_size <size>                  Total validation pool size (default: 256)"
            echo "  --val_batch_size <size>            Validation batch size for selection (default: val_size)"
            echo "  --epochs <n>                       Epochs per PPO step (default: 1)"
            echo "  --steps <n>                        Total training steps (default: 1000)"
            echo "  --seed <seed>                      Random seed (default: 42)"
            echo ""
            echo "LoRA Options:"
            echo "  --lora                             Enable LoRA (default: on)"
            echo "  --no_lora                          Disable LoRA (full fine-tuning)"
            echo "  --lora_r <r>                       LoRA rank (default: 16)"
            echo "  --lora_alpha <alpha>               LoRA alpha (default: 32)"
            echo ""
            echo "PPO Options:"
            echo "  --kl_coef <coef>                   KL penalty coefficient (default: 0.1)"
            echo ""
            echo "Other Options:"
            echo "  --output_dir <dir>                 Output directory"
            echo "  --log_interval <n>                 Log every N steps (default: steps // 10)"
            echo "  --prompt_dataset <name>            Prompt dataset (default: allenai/real-toxicity-prompts)"
            echo "  --val_dataset <name>               Validation dataset (default: Anthropic/hh-rlhf)"
            echo "  --dry_run                          Print command without executing"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ========================================
# Map Model Names
# ========================================
case $model in
    llama3-1b)
        MODEL_PATH="meta-llama/Llama-3.2-1B"
        ;;
    llama2-7b)
        MODEL_PATH="meta-llama/Llama-2-7b-hf"
        ;;
    llama2-13b)
        MODEL_PATH="meta-llama/Llama-2-13b-hf"
        ;;
    mistral-7b)
        MODEL_PATH="mistralai/Mistral-7B-v0.1"
        ;;
    *)
        MODEL_PATH="$model"
        ;;
esac

# ========================================
# Configure Compression
# ========================================
if [[ "$selection_method" != "NA" ]] || [[ "$MeSO" == "true" ]]; then
    if [[ -z "$compression" ]]; then
        compression="GraSS"
    fi
fi

# Configure compression settings based on method
if [[ -n "$compression" ]]; then
    case "$compression" in
        GraSS)
            # GraSS: random mask sparsification + projection
            sparsifier_type="random_mask"
            if [ "$use_lora" = true ]; then
                sparsifier_dim=32
                projector_dim=512
            else
                sparsifier_dim=64
                projector_dim=4096
            fi
            projector_type="normal"
            ;;
        LoGra)
            # LoGra: normal projection only (identity sparsification)
            sparsifier_type="identity"
            sparsifier_dim=-1  # -1 means no sparsification
            if [ "$use_lora" = true ]; then
                projector_dim=512
            else
                projector_dim=2048
            fi
            projector_type="normal"
            ;;
    esac
fi

# ========================================
# Build Job Name
# ========================================
if [ "$use_lora" = true ]; then
    job_type="lora"
else
    job_type="full"
fi

if [ "$MeSO" = true ]; then
    optimizer_str="MeSO"
else
    optimizer_str="Regular"
fi

method_str="${selection_method}-${optimizer_str}-${compression:-NA}"

if [ "$use_second_order" = true ]; then
    method_str="${method_str}-2nd"
fi

JOB_NAME="rlhf-${method_str}-${model}-${job_type}-lr${lr}-b${per_device_train_batch_size}-s${seed}"

# Create output directory
full_output_dir="${output_dir}/${JOB_NAME}"
mkdir -p "$full_output_dir"

# ========================================
# Print Configuration
# ========================================
echo "=== RLHF Training with Layer-wise Selection ==="
echo ""
echo "Model: $MODEL_PATH"
echo "Training mode: $([ "$use_lora" = true ] && echo "LoRA (r=$lora_r, alpha=$lora_alpha)" || echo "Full Model")"
echo ""
echo "Selection:"
echo "  Method: $selection_method"
echo "  Mode: $selection_mode"
echo "  Fraction: $selection_frac"
echo "  Val loss type: $val_loss_type"
echo "  Second-order: $([ "$use_second_order" = true ] && echo "ENABLED" || echo "DISABLED")"
echo ""
echo "Compression: ${compression:-None}"
echo "MeSO Optimizer: $([ "$MeSO" = true ] && echo "ENABLED" || echo "DISABLED")"
echo ""
echo "Training:"
echo "  Learning rate: $lr"
echo "  per_device_train_batch_size: $per_device_train_batch_size"
echo "  Mini-batch size: $mini_batch_size"
echo "  Epochs: $epochs"
echo "  Total steps: $steps"
echo "  Seed: $seed"
echo ""
echo "PPO:"
echo "  KL coefficient: $kl_coef"
echo ""
echo "Datasets:"
echo "  Prompts: $prompt_dataset"
echo "  Validation: $val_dataset"
echo "  Val size (total pool): $val_size"
echo "  Val batch size: ${val_batch_size:-$val_size (same as val_size)}"
echo ""
echo "Logging:"
echo "  Log interval: ${log_interval:-$(($steps / 10)) (steps // 10)}"
echo ""
echo "Output: $full_output_dir"
echo "==========================================="

# ========================================
# Build Command
# ========================================
# Use PYTHON env var if set, otherwise default to python
PYTHON=${PYTHON:-python}

CMD="$PYTHON experiment_rlhf/train_rlhf.py \
    --model_name $MODEL_PATH \
    --learning_rate $lr \
    --per_device_train_batch_size $per_device_train_batch_size \
    --mini_batch_size $mini_batch_size \
    --val_size $val_size \
    --epochs $epochs \
    --steps $steps \
    --seed $seed \
    --init_kl_coef $kl_coef \
    --output_dir $full_output_dir \
    --run_name $JOB_NAME"

# Add val_batch_size if specified (different from val_size)
if [[ -n "$val_batch_size" ]]; then
    CMD="$CMD --val_batch_size $val_batch_size"
fi

# Add log_interval if specified
if [[ -n "$log_interval" ]]; then
    CMD="$CMD --log_interval $log_interval"
fi

# Add selection or disable it
if [ "$selection_method" = "NA" ]; then
    CMD="$CMD --no_selection"
else
    CMD="$CMD --selection_method $selection_method --selection_frac $selection_frac --selection_mode $selection_mode --val_loss_type $val_loss_type"
fi

# Add LoRA args
if [ "$use_lora" = true ]; then
    CMD="$CMD --use_lora --lora_r $lora_r --lora_alpha $lora_alpha"
fi
# Note: --use_lora is a bool flag in Python, omitting it defaults to False

# Add compression args for selection
if [ "$selection_method" != "NA" ]; then
    CMD="$CMD --use_compression \
        --sparsifier_dim $sparsifier_dim \
        --projector_dim $projector_dim \
        --sparsifier_type $sparsifier_type \
        --projector_type $projector_type \
        --update_freq $update_compressor_freq"
fi

# Add second-order flag
if [ "$use_second_order" = true ]; then
    CMD="$CMD --use_second_order"
fi

# ========================================
# Execute
# ========================================
if [ "$dry_run" = true ]; then
    echo ""
    echo "[DRY-RUN] Would execute:"
    echo "$CMD"
else
    echo ""
    echo "Executing..."
    eval "$CMD" 2>&1 | tee "$full_output_dir/train.log"
fi
