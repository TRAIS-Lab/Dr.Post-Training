# RLVR Experiments

This folder contains the training and evaluation code and method configurations for Reinforcement Learning with Verifiable Rewards (RLVR).

## Experiment Summary

The following methods have been run and can be rerun with the commands below.

| Dataset    | Model             | Batch | Epochs | LR   | Num Generations |
| ---------- | ----------------- | ----- | ------ | ---- | --------------- |
| DeepScaler | Qwen2.5-Math-1.5B | 256   | 30     | 1e-6 | 8               |

### Experiment Configurations

We consider the following 4 methods for the RLVR task:

| #   | Selection | Description                                                       |
| --- | --------- | ----------------------------------------------------------------- |
| 1   | NA        | Baseline GRPO with random selection                               |
| 2   | DOTS+RR   | Difficulty-targeted Online daTa Selection with Rollout Replay     |
| 3   | Streaming | Per-layer, per-step gradient-based selection during GRPO training |
| 4   | GREATS    | Global selection across all layers, per-step during GRPO training |

**Selection Methods:**
- **NA**: No data selection (random baseline)
- **DOTS+RR**: Uses adaptive difficulty prediction to select training samples and replays rollouts
- **Streaming**: Per-layer, per-mini-batch selection during GRPO training
- **GREATS**: Global selection across all layers, per-mini-batch during GRPO training

## Adaptive Difficulty Prediction Framework

The core of difficulty-targeted online data selection lies in the attention-based adaptive difficulty prediction framework. To achieve this efficiently, we freeze a backbone LLM model (e.g., Qwen2.5-Math-1.5B-Instruct) and augment it with a lightweight adapter and a calibration head.

### Training the Difficulty Predictor

1. Prepare the training data

   In `adaptive_difficulty_prediction/load_data.py`, replace `data_train.pkl` and `data_ref.pkl` with your customized datasets.
   You can refer to the example formats provided in the `adaptive_difficulty_prediction/datasets/` directory.

2. Launch embedding inference and training

   ```bash
   cd RLVR/adaptive_difficulty_prediction
   bash run_bash/run_embed.sh
   bash run_bash/run_train.sh
   ```

## Training Commands

All methods are launched using individual training scripts in `rl_training/run_bash/`.

```bash
cd RLVR/rl_training

# Baseline - Random selection
bash run_bash/random_selection_baseline.sh

# DOTS+RR - Difficulty-targeted selection with rollout replay
bash run_bash/final_ds_teacher_replay.sh

# Streaming - Per-layer gradient-based selection
bash run_bash/streaming_selection.sh

# GREATS - Global gradient-based selection
bash run_bash/greats_selection.sh
```

<details>
  <summary>Detailed Training Script Configuration</summary>

### Core Training Parameters

| Parameter              | Default           | Description                              |
| ---------------------- | ----------------- | ---------------------------------------- |
| `MODEL_PATH`           | Qwen2.5-Math-1.5B | HuggingFace model path                   |
| `DATASET_PATH`         | deepscaler        | Training dataset (parquet format)        |
| `NUM_EPOCHS`           | 30                | Number of training epochs                |
| `LEARNING_RATE`        | 1e-6              | Learning rate                            |
| `EFFECTIVE_BATCH_SIZE` | 256               | Batch size (WORLD_SIZE * 64)             |
| `NUM_GENERATIONS`      | 8                 | Number of rollout generations per prompt |
| `WORLD_SIZE`           | 4                 | Number of GPUs                           |

### Selection Parameters (Streaming/GREATS)

| Parameter               | Default | Description                                            |
| ----------------------- | ------- | ------------------------------------------------------ |
| `SELECTION_METHOD`      | NA      | Selection method: `NA`, `GREATS`, `Streaming`          |
| `SELECTION_ALPHA`       | 0.5     | Target difficulty for validation set (Goldilocks zone) |
| `SELECTION_TAU`         | 0.1     | Temperature for validation selection                   |
| `VAL_RATIO`             | 0.2     | Fraction of batch to use as validation                 |
| `TRAIN_SELECTION_RATIO` | 0.5     | Fraction of negative-influence samples to drop         |
| `VAL_SELECTION_MODE`    | soft    | Selection mode: `soft` (probabilistic) or `hard`       |
| `USE_SECOND_ORDER`      | False   | Enable greedy selection with second-order interactions |

### Replay Buffer Parameters (DOTS+RR)

| Parameter         | Default | Description                 |
| ----------------- | ------- | --------------------------- |
| `SIGMA`           | 0.5     | Replay sampling temperature |
| `BUFFER_SIZE`     | 512     | Replay buffer size          |
| `REPLAY_STRATEGY` | random  | Replay strategy             |

### Teacher Model Parameters (DOTS+RR)

| Parameter                       | Default                      | Description                    |
| ------------------------------- | ---------------------------- | ------------------------------ |
| `TEACHER_MODEL_NAME`            | Qwen2.5-Math-1.5B-Instruct   | Embedding model for difficulty |
| `TEACHER_MODEL_CHECKPOINT_PATH` | adaptive_prediction_model.pt | Trained predictor checkpoint   |
| `TEACHER_MODEL_HIDDEN_SIZE`     | 896                          | Hidden size of predictor       |
| `TEACHER_MODEL_SCALING`         | group_logit_temp             | Scaling method for calibration |

### GRPO Algorithm Parameters

| Parameter       | Default | Description                            |
| --------------- | ------- | -------------------------------------- |
| `MU`            | 2       | GRPO advantage normalization parameter |
| `TAU`           | 1e-3    | Temperature for advantage computation  |
| `ALPHA`         | 0.5     | Mixing coefficient                     |
| `BETA`          | 0       | KL loss coefficient                    |
| `ENTROPY_COEFF` | 0       | Entropy bonus coefficient              |

</details>
