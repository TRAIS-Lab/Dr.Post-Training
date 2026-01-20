#!/bin/bash

#SBATCH --job-name=RLVR-MATH
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -x

# Custom configurations

source /workspace-vast/pbb/GradStream_RLVR/bin/activate

export WANDB_API_KEY="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"

SCRATCH_DIR=/workspace-vast/pbb
CODE_DIR=/workspace-vast/pbb

# ============================================================================
# GPU Configuration
# ============================================================================
if [ -z "$N_GPUS" ]; then
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    else
        N_GPUS=1
    fi
fi
echo "Using N_GPUS=$N_GPUS"

# ============================================================================
# Data paths
# ============================================================================
DATA_DIR=$SCRATCH_DIR/Gradient-Streaming/RLVR/data
math_train_path=$DATA_DIR/math/train.parquet
math_test_path=$DATA_DIR/math/test.parquet

# Validation prompts for online rollout generation
VAL_PROMPTS_PATH=$DATA_DIR/math/val_prompts.parquet

train_files=$math_train_path
test_files=$math_test_path

# Output directory
OUTPUT_BASE=$SCRATCH_DIR/Gradient-Streaming/RLVR/output

# ============================================================================
# Experiment Configuration
# ============================================================================
SEED=${SEED:-42}
SELECTION_ENABLED=${SELECTION_ENABLED:-True}
SELECTION_METHOD=${SELECTION_METHOD:-Streaming}
SELECTION_FRAC=${SELECTION_FRAC:-1.0}
VAL_POOL_SIZE=${VAL_POOL_SIZE:-500}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
REFRESH_FREQ=${REFRESH_FREQ:-1}
RESUME_MODE=${RESUME_MODE:-disable}

# Parse Hydra overrides from command line args
for arg in "$@"; do
    case "$arg" in
        *seed=*)
            SEED=$(echo "$arg" | sed 's/.*seed=//')
            ;;
        *selection.enable=False*|*selection.enable=false*)
            SELECTION_ENABLED=False
            ;;
        *selection.enable=True*|*selection.enable=true*)
            SELECTION_ENABLED=True
            ;;
        *selection.method=*)
            SELECTION_METHOD=$(echo "$arg" | sed 's/.*selection.method=//')
            ;;
        *selection.frac=*)
            SELECTION_FRAC=$(echo "$arg" | sed 's/.*selection.frac=//')
            ;;
        *selection.val_pool_size=*)
            VAL_POOL_SIZE=$(echo "$arg" | sed 's/.*selection.val_pool_size=//')
            ;;
        *selection.val_batch_size=*)
            VAL_BATCH_SIZE=$(echo "$arg" | sed 's/.*selection.val_batch_size=//')
            ;;
        *selection.refresh_freq=*)
            REFRESH_FREQ=$(echo "$arg" | sed 's/.*selection.refresh_freq=//')
            ;;
        *trainer.resume_mode=*)
            RESUME_MODE=$(echo "$arg" | sed 's/.*trainer.resume_mode=//')
            ;;
    esac
done

# Build experiment name
if [ "$SELECTION_ENABLED" = "True" ]; then
    EXP_NAME="qwen3_1.7b_grpo_math_online_${SELECTION_METHOD}_frac${SELECTION_FRAC}_vb${VAL_BATCH_SIZE}"
else
    EXP_NAME="qwen3_1.7b_grpo_math_baseline"
fi

OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}"
HYDRA_DIR="${OUTPUT_BASE}/hydra/${EXP_NAME}"

echo "Experiment: $EXP_NAME"
echo "Output dir: $OUTPUT_DIR"
echo "Random seed: $SEED"
echo "Selection config:"
echo "  enabled=$SELECTION_ENABLED"
echo "  method=$SELECTION_METHOD"
echo "  frac=$SELECTION_FRAC"
echo "  val_pool_size=$VAL_POOL_SIZE"
echo "  val_batch_size=$VAL_BATCH_SIZE"
echo "  refresh_freq=$REFRESH_FREQ"

# Add project root to PYTHONPATH
export PYTHONPATH=$CODE_DIR/Gradient-Streaming/RLVR:$CODE_DIR/Gradient-Streaming:${PYTHONPATH}

# Unset ROCR_VISIBLE_DEVICES to avoid conflict with CUDA_VISIBLE_DEVICES
# (ROCR is for AMD ROCm, we're using NVIDIA GPUs)
unset ROCR_VISIBLE_DEVICES

# ============================================================================
# Check/Prepare validation prompts
# ============================================================================
if [ "$SELECTION_ENABLED" = "True" ] && [ ! -f "$VAL_PROMPTS_PATH" ]; then
    echo "Validation prompts not found at $VAL_PROMPTS_PATH"
    echo "Creating validation prompts from training data..."
    python3 $CODE_DIR/Gradient-Streaming/RLVR/data/prepare_data.py \
        --train_data $math_train_path \
        --output $VAL_PROMPTS_PATH \
        --num_samples $VAL_POOL_SIZE \
        --seed $SEED
fi

# ============================================================================
# Run Training
# ============================================================================
echo "Starting training with $N_GPUS GPUs..."

# Note: MATH problems and solutions are typically longer than GSM8K
# Increased max_response_length to 2048 to accommodate more detailed reasoning
python3 $CODE_DIR/Gradient-Streaming/RLVR/main_ppo_online_selection.py \
    hydra.run.dir=$HYDRA_DIR \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.seed=$SEED \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.seed=$SEED \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.seed=$SEED \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.seed=$SEED \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_math_online_selection' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.resume_mode=$RESUME_MODE \
	trainer.balance_batch=True \
    trainer.log_val_generations=5 \
    +selection.enable=$SELECTION_ENABLED \
    +selection.method=$SELECTION_METHOD \
    +selection.frac=$SELECTION_FRAC \
    +selection.use_second_order=False \
    +selection.val_prompts_path=$VAL_PROMPTS_PATH \
    +selection.val_pool_size=$VAL_POOL_SIZE \
    +selection.val_batch_size=$VAL_BATCH_SIZE \
    +selection.val_max_prompt_length=1024 \
    +selection.val_max_response_length=4096 \
    +selection.val_seed=$SEED \
    +selection.refresh_freq=$REFRESH_FREQ \
    +selection.val_loss_type=seqloss-reward \
    "$@"
