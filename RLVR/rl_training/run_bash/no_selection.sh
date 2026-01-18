set -x

# Kill existing Ray cluster if running
if ray status &>/dev/null; then
    echo "Ray cluster detected. Stopping existing Ray..."
    ray stop
fi

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
export VLLM_ATTENTION_BACKEND=XFORMERS
ray start --head --port 6379
echo "VLLM_ATTENTION_BACKEND is set to: $VLLM_ATTENTION_BACKEND"

MODEL_PATHs=(
    "Qwen/Qwen2.5-Math-1.5B"
)

# Use original small training set (1024 samples) for comparison with baselines
# Use fresh held-out validation set for selection methods
TRAIN_DATASET_PATH="data/deepscaler.parquet"
VAL_DATASET_PATH="data/deepscaler_val.parquet"

# Dataset sample limits (-1 for all samples)
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1

NUM_EPOCHS=30
LEARNING_RATE=1e-6
USE_RANDOM_SELECTION=True  # Random selection baseline
USE_TEMP_LOG_PROB=True

MU=2
TAU=1e-3
ALPHA=0.5
BETA=0
ENTROPY_COEFF=0

NUM_GENERATIONS=8
WORLD_SIZE=4

for MODEL_PATH in "${MODEL_PATHs[@]}"; do

    MODEL_NAME=$(basename "$MODEL_PATH")
    DATASET_NAME=$(basename "$TRAIN_DATASET_PATH" .parquet)

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

        SIGMA=0.5
        BUFFER_SIZE=512
        REPLAY_STRATEGY="random"

        WANDB_NAME="random_selection_baseline_${MODEL_NAME}_dataset_${DATASET_NAME}_epoch_${NUM_EPOCHS}_bs_${EFFECTIVE_BATCH_SIZE}_lr_${LEARNING_RATE}_sigma_${SIGMA}_buffer_${BUFFER_SIZE}"
        echo "MODEL_PATH=$MODEL_PATH"
        echo "DATASET_PATH=$DATASET_PATH"

        # Train with wandb logging - NO teacher model for random selection
        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=$TRAIN_DATASET_PATH \
            data.val_files=$VAL_DATASET_PATH \
            data.train_batch_size=$EFFECTIVE_BATCH_SIZE \
            data.val_batch_size=512 \
            data.train_max_samples=$TRAIN_MAX_SAMPLES \
            data.val_max_samples=$VAL_MAX_SAMPLES \
            data.max_prompt_length=1024 \
            data.max_response_length=$MAX_COMPLETION_LENGTH \
            +data.mu=$MU \
            +data.ref_size=256 \
            +data.tau=$TAU \
            +data.alpha=$ALPHA \
            +data.sigma=$SIGMA \
            +data.buffer_size=$BUFFER_SIZE \
            +data.replay_strategy=$REPLAY_STRATEGY \
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
            +trainer.retry_delay=60 \
            +data.random_selection=$USE_RANDOM_SELECTION
done

ray stop
