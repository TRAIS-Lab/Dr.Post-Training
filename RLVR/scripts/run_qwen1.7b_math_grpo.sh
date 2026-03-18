#!/bin/bash

set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Source cluster config (skip if already set by submit.sh)
if [[ -z "$CODE_DIR" ]]; then
    source "$REPO_ROOT/cluster_env.sh" || { echo "ERROR: cluster_env.sh not found."; exit 1; }
    activate_env
fi

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
DATA_DIR=$SCRATCH_DIR/Dr.Post-Training/RLVR/data
math_train_path=$DATA_DIR/math/train.parquet
math_test_path=$DATA_DIR/math/test.parquet
math_test_cleaned_path=$DATA_DIR/math/test_cleaned.parquet

# Validation prompts for online rollout generation
# Options: val_from_test.parquet (default) or val_from_train.parquet (ablation)
VAL_PROMPTS_PATH=${VAL_PROMPTS_PATH:-$DATA_DIR/math/val_from_train.parquet}

train_files=$math_train_path
# Use cleaned test set (excludes validation samples) for evaluation
test_files=$math_test_cleaned_path

# Output directory
OUTPUT_BASE=$SCRATCH_DIR/Dr.Post-Training/RLVR/output

# ============================================================================
# Experiment Configuration
# ============================================================================
SEED=${SEED:-42}
SELECTION_ENABLED=${SELECTION_ENABLED:-True}
SELECTION_METHOD=${SELECTION_METHOD:-Layerwise}
SELECTION_FRAC=${SELECTION_FRAC:-1.0}
VAL_POOL_SIZE=${VAL_POOL_SIZE:-512}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
REFRESH_FREQ=${REFRESH_FREQ:-1}
RESUME_MODE=${RESUME_MODE:-disable}
# Validation loss type options:
# - reward: L = -E[normalized_reward * log_prob] (batch normalization)
# - train-loss: L = -E[advantages * log_prob] (GRPO normalization, matches training)
VAL_LOSS_TYPE=${VAL_LOSS_TYPE:-train-loss}

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
        *selection.val_loss_type=*)
            VAL_LOSS_TYPE=$(echo "$arg" | sed 's/.*selection.val_loss_type=//')
            ;;
        *trainer.resume_mode=*)
            RESUME_MODE=$(echo "$arg" | sed 's/.*trainer.resume_mode=//')
            ;;
    esac
done

# Build experiment name
if [ "$SELECTION_ENABLED" = "True" ]; then
    EXP_NAME="Qwen3-1.7B_${SELECTION_METHOD}_s${SEED}"
else
    EXP_NAME="Qwen3-1.7B_s${SEED}"
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
echo "  val_loss_type=$VAL_LOSS_TYPE"

# Add project root to PYTHONPATH
export PYTHONPATH=$REPO_ROOT/RLVR:$REPO_ROOT:${PYTHONPATH}

# Unset ROCR_VISIBLE_DEVICES to avoid conflict with CUDA_VISIBLE_DEVICES
# (ROCR is for AMD ROCm, we're using NVIDIA GPUs)
unset ROCR_VISIBLE_DEVICES

# ============================================================================
# Check/Prepare validation prompts
# ============================================================================
VAL_FROM_TEST_PATH=$DATA_DIR/math/val_from_test.parquet
VAL_FROM_TRAIN_PATH=$DATA_DIR/math/val_from_train.parquet

if [ "$SELECTION_ENABLED" = "True" ]; then
    # Generate val_from_test and test_cleaned if needed
    if [ ! -f "$VAL_FROM_TEST_PATH" ]; then
        echo "Generating val_from_test.parquet and test_cleaned.parquet..."
        python3 $REPO_ROOT/RLVR/data/prepare_data.py \
            --test_data $math_test_path \
            --output $VAL_FROM_TEST_PATH \
            --output_test $math_test_cleaned_path \
            --num_samples $VAL_POOL_SIZE \
            --seed $SEED
    fi

    # Generate val_from_train if needed
    if [ ! -f "$VAL_FROM_TRAIN_PATH" ]; then
        echo "Generating val_from_train.parquet..."
        python3 $REPO_ROOT/RLVR/data/prepare_data.py \
            --train_data $math_train_path \
            --output $VAL_FROM_TRAIN_PATH \
            --num_samples $VAL_POOL_SIZE \
            --seed $SEED
    fi

    echo "Using validation prompts from: $VAL_PROMPTS_PATH"
fi

# ============================================================================
# Run Training
# ============================================================================
echo "Starting training with $N_GPUS GPUs..."

# Note: MATH problems and solutions are typically longer than GSM8K
# Increased max_response_length to 1024 to accommodate more detailed reasoning
python3 $REPO_ROOT/RLVR/train.py \
    hydra.run.dir=$HYDRA_DIR \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.seed=$SEED \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-1.7B-Base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.seed=$SEED \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.seed=$SEED \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_math' \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=3 \
    trainer.total_epochs=5 \
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
    +selection.val_max_response_length=1024 \
    +selection.val_seed=$SEED \
    +selection.refresh_freq=$REFRESH_FREQ \
    +selection.val_loss_type=$VAL_LOSS_TYPE \
    "$@"
