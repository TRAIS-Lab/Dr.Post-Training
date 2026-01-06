#!/bin/bash
set -x

# ============================================================================
# RLVR Training Script (Unified Entry Point)
#
# All selection methods use main_ppo_rlvr as the single entry point.
# Selection is controlled via config flags:
#   - "teacher": Original DOTS with teacher model (epoch-level)
#   - "GREATS": RLVR online per-batch selection
#   - "Streaming": RLVR per-layer selection
#   - "random": Random selection (baseline)
# ============================================================================

# Kill existing Ray cluster if running
if ray status &>/dev/null; then
    echo "Ray cluster detected. Stopping existing Ray..."
    ray stop
fi

export VLLM_ATTENTION_BACKEND=XFORMERS
ray start --head --port 6379
echo "VLLM_ATTENTION_BACKEND is set to: $VLLM_ATTENTION_BACKEND"

# ============================================================================
# Model and Data Configuration
# ============================================================================
MODEL_PATHS=(
    "Qwen/Qwen2.5-Math-1.5B"
)

DATASET_PATHS=(
    "data/deepscaler.parquet"
)

# ============================================================================
# Training Hyperparameters
# ============================================================================
NUM_EPOCHS=30
LEARNING_RATE=1e-6
USE_TEMP_LOG_PROB=True

MU=2                    # Steps per epoch
BETA=0                  # KL loss coefficient
ENTROPY_COEFF=0

NUM_GENERATIONS=8
WORLD_SIZE=8

# ============================================================================
# Selection Method Configuration
# Options: "teacher", "GREATS", "Streaming", "random"
# ============================================================================
SELECTION_TYPE="GREATS"

# Common selection parameters (used by all methods except random)
ALPHA=0.5                       # Target difficulty (Goldilocks zone)
TAU=0.1                         # Selection temperature (lower = greedier)
REF_SIZE=256                    # Reference set size

# RLVR-specific parameters (for GREATS and Streaming)
RLVR_SELECTION_RATIO=0.5        # Fraction of batch to select
RLVR_SELECTION_MODE="soft"      # "soft" (probabilistic) or "hard" (top-k)
RLVR_AGGREGATION="mean"         # Layer aggregation: "mean", "max", "min"

# Teacher model parameters (for teacher mode only)
EMBEDDING_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"
TEACHER_CHECKPOINT="adaptive_prediction_training/adaptive_prediction_model.pt"
TEACHER_HIDDEN_SIZE=896
TEACHER_SCALING="group_logit_temp"
TEACHER_NUM_LAYERS=3

# Replay buffer (for teacher mode only)
SIGMA=0.25
BUFFER_SIZE=512
REPLAY_STRATEGY="random"

# ============================================================================
# Training Loop
# ============================================================================
for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    for MODEL_PATH in "${MODEL_PATHS[@]}"; do

        MODEL_NAME=$(basename "$MODEL_PATH")
        DATASET_NAME=$(basename "$DATASET_PATH" .parquet)

        # Set max completion length based on model
        MAX_COMPLETION_LENGTH=4096
        if [[ "$MODEL_PATH" == *Math* ]]; then
            MAX_COMPLETION_LENGTH=3072
        fi
        echo "MAX_COMPLETION_LENGTH=$MAX_COMPLETION_LENGTH"
        PPO_MAX_TOKEN_LEN_PER_GPU=$((8 * 1024 + MAX_COMPLETION_LENGTH))

        # Set batch size based on model size
        if [[ "$MODEL_PATH" == *7B* ]]; then
            EFFECTIVE_BATCH_SIZE=$((WORLD_SIZE * 32))
        else
            EFFECTIVE_BATCH_SIZE=$((WORLD_SIZE * 64))
        fi

        # Build experiment name based on selection type
        if [[ "$SELECTION_TYPE" == "teacher" ]]; then
            WANDB_NAME="dots_teacher_${MODEL_NAME}_${DATASET_NAME}_ep${NUM_EPOCHS}_bs${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}_alpha${ALPHA}_tau${TAU}_sigma${SIGMA}"
        elif [[ "$SELECTION_TYPE" == "random" ]]; then
            WANDB_NAME="random_${MODEL_NAME}_${DATASET_NAME}_ep${NUM_EPOCHS}_bs${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}"
        else
            WANDB_NAME="rlvr_${SELECTION_TYPE}_${MODEL_NAME}_${DATASET_NAME}_ep${NUM_EPOCHS}_bs${EFFECTIVE_BATCH_SIZE}_lr${LEARNING_RATE}_alpha${ALPHA}_tau${TAU}_ratio${RLVR_SELECTION_RATIO}"
        fi

        echo "=============================================="
        echo "Starting Training"
        echo "  Model: $MODEL_PATH"
        echo "  Dataset: $DATASET_PATH"
        echo "  Selection Type: $SELECTION_TYPE"
        echo "  Alpha: $ALPHA"
        echo "  Tau: $TAU"
        if [[ "$SELECTION_TYPE" == "GREATS" || "$SELECTION_TYPE" == "Streaming" ]]; then
            echo "  Selection Ratio: $RLVR_SELECTION_RATIO"
        fi
        echo "=============================================="

        # Build selection-specific arguments
        if [[ "$SELECTION_TYPE" == "teacher" ]]; then
            # Teacher mode: rlvr_enabled=False, random_selection=False
            SELECTION_ARGS=(
                "+data.random_selection=False"
                "+data.rlvr_enabled=False"
                "+data.tau=$TAU"
                "+data.alpha=$ALPHA"
                "+data.sigma=$SIGMA"
                "+data.buffer_size=$BUFFER_SIZE"
                "+data.replay_strategy=$REPLAY_STRATEGY"
                "+teacher_model.embedding_path=adaptive_prediction_training"
                "+teacher_model.model_name=$EMBEDDING_MODEL"
                "+teacher_model.batch_size=32"
                "+teacher_model.checkpoint_path=$TEACHER_CHECKPOINT"
                "+teacher_model.num_layers=$TEACHER_NUM_LAYERS"
                "+teacher_model.scaling=$TEACHER_SCALING"
                "+teacher_model.hidden_size=$TEACHER_HIDDEN_SIZE"
            )
        elif [[ "$SELECTION_TYPE" == "random" ]]; then
            # Random mode: rlvr_enabled=False, random_selection=True
            SELECTION_ARGS=(
                "+data.random_selection=True"
                "+data.rlvr_enabled=False"
            )
        else
            # RLVR mode (GREATS or Streaming): rlvr_enabled=True
            SELECTION_ARGS=(
                "+data.random_selection=False"
                "+data.rlvr_enabled=True"
                "+data.rlvr_selection_method=$SELECTION_TYPE"
                "+data.alpha=$ALPHA"
                "+data.tau=$TAU"
                "+data.rlvr_selection_ratio=$RLVR_SELECTION_RATIO"
                "+data.rlvr_selection_mode=$RLVR_SELECTION_MODE"
                "+data.rlvr_aggregation=$RLVR_AGGREGATION"
            )
        fi

        # Single unified entry point for all selection methods
        python3 -m verl.trainer.main_ppo_rlvr \
            algorithm.adv_estimator=grpo \
            data.train_files=$DATASET_PATH \
            data.val_files=$DATASET_PATH \
            data.train_batch_size=$EFFECTIVE_BATCH_SIZE \
            data.val_batch_size=512 \
            data.max_prompt_length=1024 \
            data.max_response_length=$MAX_COMPLETION_LENGTH \
            +data.mu=$MU \
            +data.ref_size=$REF_SIZE \
            "${SELECTION_ARGS[@]}" \
            actor_rollout_ref.model.path=$MODEL_PATH \
            actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=64 \
            actor_rollout_ref.actor.use_dynamic_bsz=True \
            actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
            actor_rollout_ref.actor.kl_loss_coef=$BETA \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.grad_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            +actor_rollout_ref.actor.use_temp_log_prob=$USE_TEMP_LOG_PROB \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.temperature=0.6 \
            actor_rollout_ref.rollout.val_temperature=0.6 \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
            actor_rollout_ref.rollout.n=$NUM_GENERATIONS \
            actor_rollout_ref.rollout.n_val=8 \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name="rl" \
            trainer.experiment_name=$WANDB_NAME \
            +trainer.val_before_train=False \
            trainer.n_gpus_per_node=$WORLD_SIZE \
            trainer.nnodes=1 \
            trainer.save_freq=-1 \
            trainer.test_freq=10000000 \
            trainer.default_hdfs_dir=null \
            trainer.default_local_dir="output/models/${WANDB_NAME}" \
            trainer.total_epochs=$NUM_EPOCHS \
            trainer.resume_mode=auto \
            trainer.resume_from_path="final_checkpoints" \
            +trainer.max_retries=1 \
            +trainer.retry_delay=60

    done
done

ray stop

echo "=============================================="
echo "Training Complete!"
echo "  Selection Type: $SELECTION_TYPE"
echo "=============================================="
