set -x

# Streaming Online Data Selection for RLVR
# This script runs RLVR with gradient-based data selection (Streaming method)
#
# Key differences from baseline DOTS and GREATS:
# - DOTS: Uses difficulty to select training samples offline (per-epoch)
# - GREATS: Per-step global selection (same selection for all layers)
# - Streaming: Per-step, per-layer selection (different selection per layer)

# Kill existing Ray cluster if running
if ray status &>/dev/null; then
    echo "Ray cluster detected. Stopping existing Ray..."
    ray stop
fi

# Add gradstream to Python path for Ray workers
export PYTHONPATH="/home/pbb/Project/Gradient-Streaming:${PYTHONPATH}"
echo "PYTHONPATH is set to: $PYTHONPATH"

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
export VLLM_ATTENTION_BACKEND=XFORMERS
ray start --head --port 6379
echo "VLLM_ATTENTION_BACKEND is set to: $VLLM_ATTENTION_BACKEND"

MODEL_PATHs=(
    "Qwen/Qwen2.5-Math-1.5B"
)

DATASET_PATHS=(
    "data/deepscaler.parquet"
)

EMBEDDING_MODEL="Qwen/Qwen2.5-Math-1.5B-Instruct"

NUM_EPOCHS=30
LEARNING_RATE=1e-6
USE_TEMP_LOG_PROB=True

# Streaming Selection Parameters
SELECTION_METHOD="Streaming"  # Options: NA, GREATS, Streaming
SELECTION_ALPHA=0.5           # Target difficulty for validation set (Goldilocks zone)
SELECTION_TAU=0.1             # Temperature for validation selection
VAL_RATIO=0.2                 # Fraction of validation batch to use after difficulty selection (Goldilocks)
TRAIN_SELECTION_RATIO=1.0     # Fraction of negative-influence samples to DROP (filtering mode) - 1.0 = drop ALL negative samples
VAL_SELECTION_MODE="hard"     # "soft" (probabilistic) or "hard" (top-k)
USE_SECOND_ORDER=False        # Whether to use second-order greedy selection
SELECTION_VAL_BATCH_SIZE=64   # Batch size for separate validation set (used once per epoch)

MU=2
TAU=1e-3
ALPHA=0.5
BETA=0
ENTROPY_COEFF=0

# Replay buffer parameters (required by ray_trainer.py)
SIGMA=0.5
BUFFER_SIZE=512
REPLAY_STRATEGY="random"

NUM_GENERATIONS=8
WORLD_SIZE=4

for DATASET_PATH in "${DATASET_PATHS[@]}"; do
    for MODEL_PATH in "${MODEL_PATHs[@]}"; do

        TEACHER_MODEL_NAME="$EMBEDDING_MODEL"
        TEACHER_MODEL_CHECKPOINT_PATH="adaptive_prediction_training/adaptive_prediction_model.pt"
        TEACHER_MODEL_HIDDEN_SIZE=896
        TEACHER_MODEL_SCALING=group_logit_temp

        MODEL_NAME=$(basename "$MODEL_PATH")
        DATASET_NAME=$(basename "$DATASET_PATH" .parquet)

        MAX_COMPLETION_LENGTH=4096
        if [[ "$MODEL_PATH" == *Math* ]]; then
            MAX_COMPLETION_LENGTH=3072
        fi
        echo "MAX_COMPLETION_LENGTH=$MAX_COMPLETION_LENGTH"
        PPO_MAX_TOKEN_LEN_PER_GPU=$((8 * 1024 + MAX_COMPLETION_LENGTH))

        if [[ "$MODEL_PATH" == *7B* ]]; then
            EFFECTIVE_BATCH_SIZE=$((WORLD_SIZE * 32))
        else
            EFFECTIVE_BATCH_SIZE=$((WORLD_SIZE * 64))
        fi

        WANDB_NAME="${SELECTION_METHOD}_selection_${MODEL_NAME}_${DATASET_NAME}_epoch_${NUM_EPOCHS}_bs_${EFFECTIVE_BATCH_SIZE}_lr_${LEARNING_RATE}_alpha_${SELECTION_ALPHA}_tau_${SELECTION_TAU}_val_${VAL_RATIO}_train_${TRAIN_SELECTION_RATIO}"
        echo "MODEL_PATH=$MODEL_PATH"
        echo "DATASET_PATH=$DATASET_PATH"
        echo "SELECTION_METHOD=$SELECTION_METHOD"

        # Train with wandb logging
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=$DATASET_PATH \
            data.val_files=$DATASET_PATH \
            data.train_batch_size=$EFFECTIVE_BATCH_SIZE \
            data.val_batch_size=512 \
            data.max_prompt_length=1024 \
            data.max_response_length=$MAX_COMPLETION_LENGTH \
            +data.mu=$MU \
            +data.ref_size=256 \
            +data.tau=$TAU \
            +data.alpha=$ALPHA \
            +data.sigma=$SIGMA \
            +data.buffer_size=$BUFFER_SIZE \
            +data.replay_strategy=$REPLAY_STRATEGY \
            +data.random_selection=False \
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
            +actor_rollout_ref.actor.selection_method=$SELECTION_METHOD \
            +actor_rollout_ref.actor.selection_alpha=$SELECTION_ALPHA \
            +actor_rollout_ref.actor.selection_tau=$SELECTION_TAU \
            +actor_rollout_ref.actor.val_ratio=$VAL_RATIO \
            +actor_rollout_ref.actor.train_selection_ratio=$TRAIN_SELECTION_RATIO \
            +actor_rollout_ref.actor.val_selection_mode=$VAL_SELECTION_MODE \
            +actor_rollout_ref.actor.use_second_order=$USE_SECOND_ORDER \
            +actor_rollout_ref.actor.selection_val_batch_size=$SELECTION_VAL_BATCH_SIZE \
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
            +trainer.retry_delay=60 \
            +teacher_model.embedding_path="adaptive_prediction_training" \
            +teacher_model.model_name="$TEACHER_MODEL_NAME" \
            +teacher_model.batch_size=32 \
            +teacher_model.checkpoint_path=$TEACHER_MODEL_CHECKPOINT_PATH \
            +teacher_model.num_layers=3 \
            +teacher_model.scaling=$TEACHER_MODEL_SCALING \
            +teacher_model.hidden_size=$TEACHER_MODEL_HIDDEN_SIZE
    done
done

ray stop
