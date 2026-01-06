# RLVR Experiments

This folder contains the training code for Reinforcement Learning with Variable Replay (RLVR) based on difficulty-targeted online data selection (DOTS).

Reference: [Data-Efficient Reinforcement Learning via Difficulty-Targeted Online Data Selection](https://arxiv.org/pdf/2506.05316)

## Method Overview

RLVR implements difficulty-based sample selection using activation similarity:

1. **Reference Set**: Maintain a small set of samples with ground-truth difficulties (computed from rollout rewards)
2. **Difficulty Prediction**: Use attention-based similarity between training sample activations and reference activations to predict difficulty
3. **Goldilocks Selection**: Select samples with medium difficulty (close to target α=0.5) - the "Goldilocks zone"
4. **Replay Buffer**: Optionally replay past experiences with difficulty-based selection

Key difference from Gradient Streaming:
- **Gradient Streaming**: Similarity computed on gradients (backward pass)
- **RLVR**: Similarity computed on activations (forward pass)

## Selection Methods

RLVR supports three selection methods:

### 1. Original DOTS (Epoch-Level)
Select samples once per epoch based on predicted difficulties using a teacher model.
This is the original paper implementation.

**Use:** `train.sh` with `random_selection=False`

### 2. RLVR-GREATS (Online Per-Batch)
**NEW:** Select samples online for each mini-batch using activation similarity.
Similar to IIF but with Goldilocks zone selection (medium difficulty).

**Flow:**
```
1. Forward pass: Capture activations at each layer
2. After forward: Compute aggregated difficulty using activation similarity
3. Selection: Select samples close to target difficulty (α=0.5)
4. Loss: Compute loss only on selected samples
5. Backward: Standard backward on selected samples
```

**Advantages:**
- Online: reacts to training dynamics each step
- No teacher model needed: uses activation similarity
- Simpler: standard backward pass (no custom autograd)

**Use:** `train_rlvr.sh` with `RLVR_SELECTION_METHOD=GREATS`

### 3. RLVR-Streaming (Online Per-Layer)
**NEW:** Select samples online for each layer during backward pass.
Each layer independently selects samples based on layer-specific difficulty.

**Flow:**
```
1. Forward pass: Capture activations at each layer
2. After forward: Compute per-layer difficulties
3. Selection: Each layer selects samples close to target difficulty
4. Loss: Compute loss on full batch
5. Backward: RLVRStreamingLinearBackward aggregates gradients from per-layer selections
```

**Advantages:**
- Most adaptive: different layers select different samples
- Online: reacts to training dynamics each step
- Per-layer: some layers may find certain samples harder

**Use:** `train_rlvr.sh` with `RLVR_SELECTION_METHOD=Streaming`

### Comparison

| Method | Selection Timing | Selection Scope | Reference |
|--------|-----------------|-----------------|-----------|
| Original DOTS | Per-epoch | Global | Teacher model predictions |
| RLVR-GREATS | Per-batch | Global | Activation similarity |
| RLVR-Streaming | Per-batch | Per-layer | Activation similarity |

## Directory Structure

```
RLVR/
├── __init__.py                       # Module exports (RLVRHook, etc.)
├── hook.py                           # NEW: Unified RLVRHook for activation-based selection
├── DESIGN.md                         # Design document
│
├── adaptive_difficulty_prediction/   # Difficulty prediction model (teacher)
│   ├── model.py                      # FewShotRegressor, TextEncoder, scaling methods
│   ├── train.py                      # Training script for difficulty predictor
│   ├── load_data.py                  # TeacherDataset for training
│   ├── save_embedding.py             # Pre-compute embeddings
│   ├── utils.py                      # Metrics and calibration utilities
│   └── run_bash/                     # Training scripts
│
├── difficulty/                       # RLVR difficulty prediction components
│   ├── __init__.py                   # Module exports
│   ├── activation_hook.py            # ActivationHook for capturing layer activations
│   ├── predictor.py                  # DifficultyPredictor (attention-based)
│   ├── state.py                      # RLVRState, RLVRStreamingState
│   └── backward.py                   # RLVRStreamingLinearBackward autograd function
│
├── data/                             # Training data (parquet)
│
└── train/
    ├── __init__.py                   # NEW: Module exports
    ├── train.sh                      # Original DOTS training script
    ├── train_rlvr.sh                 # NEW: RLVR-GREATS/Streaming training script
    ├── rlvr_selection.py             # NEW: RLVRSelectionManager for trainer integration
    ├── rlvr_streaming.py             # RLVR Streaming manager
    ├── integration_example.py        # NEW: Integration examples
    └── verl/                         # VERL framework (Ray-based distributed RL)
        └── verl/
            ├── trainer/
            │   ├── main_ppo.py           # Original entry point
            │   ├── main_ppo_rlvr.py      # NEW: RLVR entry point
            │   └── ppo/
            │       ├── ray_trainer_teacher.py     # Original DOTS trainer
            │       ├── ray_trainer_rlvr.py        # NEW: RLVR trainer
            │       ├── teacher_utils.py           # TeacherModelWorker
            │       └── core_algos.py              # PPO algorithms
            ├── workers/              # Distributed workers (Actor, Critic, Rollout)
            ├── utils/                # Utilities (FSDP, dataset, distributed)
            └── protocol.py           # DataProto for data transfer
```

## RLVR Difficulty Module

The RLVR-specific difficulty prediction and selection components are in `RLVR/difficulty/`:

```
RLVR/difficulty/
├── __init__.py           # Module exports
├── activation_hook.py    # ActivationHook for capturing layer activations
├── predictor.py          # DifficultyPredictor (attention-based)
├── state.py              # RLVRState, RLVRStreamingState
└── backward.py           # RLVRStreamingLinearBackward autograd function
```

Note: Unlike gradient-based selection (in `gradstream/`), RLVR uses activation similarity computed during forward pass, so it has its own separate module.

### Using RLVR Streaming

```python
from RLVR.train.rlvr_streaming import create_rlvr_streaming_manager

# Initialize
rlvr_manager = create_rlvr_streaming_manager(
    model=model,
    layer_names=layer_names,
    device='cuda',
    predictor_type='simple',
)
rlvr_manager.set_selection_params(alpha=0.5, tau=0.1, selection_ratio=0.5)

# Set reference from rollout rewards
rlvr_manager.capture_reference_from_rollouts(ref_input_ids, ref_attention_mask, rewards)

# Training step
for batch in dataloader:
    # Select samples based on difficulty
    result = rlvr_manager.select_samples(
        batch['input_ids'], batch['attention_mask']
    )
    selected_indices = result['selected_indices']

    # Filter batch to selected samples
    filtered_batch = {k: v[selected_indices] for k, v in batch.items()}

    # Forward/backward on filtered batch
    outputs = model(**filtered_batch)
    loss = compute_loss(outputs)
    loss.backward()

    optimizer.step()
```

## Quick Start

### Option 1: RLVR-GREATS (Recommended)

Online per-batch selection using activation similarity. No teacher model needed.

```bash
cd RLVR/train
bash train_rlvr.sh
```

Key configuration in `train_rlvr.sh`:
```bash
RLVR_ENABLED=True
RLVR_SELECTION_METHOD="GREATS"   # or "Streaming"
RLVR_ALPHA=0.5                   # Target difficulty
RLVR_TAU=0.1                     # Selection temperature
RLVR_SELECTION_RATIO=0.5         # Fraction to select
```

### Option 2: Original DOTS (Teacher Model)

Epoch-level selection using pre-trained difficulty predictor.

#### Step 1: Pre-compute Embeddings

```bash
cd RLVR/adaptive_difficulty_prediction
bash run_bash/run_embed.sh
```

#### Step 2: Train Difficulty Predictor

```bash
cd RLVR/adaptive_difficulty_prediction
bash run_bash/run_train.sh
```

#### Step 3: Run DOTS Training

```bash
cd RLVR/train
bash train.sh
```

## DOTS Parameters

Key hyperparameters for difficulty-targeted selection:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu` | 2 | Steps per epoch (data efficiency factor) |
| `tau` | 1e-3 | Temperature for selection softmax (lower = greedier) |
| `alpha` | 0.5 | Target difficulty (0.5 = medium, Goldilocks zone) |
| `sigma` | 0.25 | Fraction of buffer to replay |
| `buffer_size` | 512 | Replay buffer size |
| `ref_size` | 256 | Reference set size for difficulty prediction |

### Selection Score

RLVR selects samples close to target difficulty:

```python
score = -|predicted_difficulty - alpha|  # Higher = closer to target
probs = softmax(score / tau)              # Temperature-scaled
selected = multinomial(probs, budget)     # Probabilistic selection
```

## VERL Framework

The training uses VERL (Volcano Engine RL), a Ray-based distributed RL framework:

### Key Components

- **RayPPOTrainer** (`ray_trainer_teacher_replay.py`): Main trainer with DOTS + replay
- **TeacherModelWorker** (`teacher_utils.py`): Difficulty prediction worker
- **DataProto** (`protocol.py`): Efficient data transfer protocol
- **FSDP Workers**: Fully sharded data parallel for actor/critic

### Worker Architecture

```
Driver (RayPPOTrainer)
├── ActorRolloutRefWorker  - Policy generation and training
├── CriticWorker           - Value function computation
├── RewardModelWorker      - Reward scoring
├── TeacherModelWorker     - Difficulty prediction
└── RolloutWorker (vLLM)   - Fast generation engine
```

## Difficulty Prediction Model

### FewShotRegressor Architecture

```python
class FewShotRegressor:
    # Components:
    - TextEncoder: Frozen LLM backbone (e.g., Qwen2.5-Math)
    - RegressionHead: MLP projection for query/reference
    - ResidualHead: Calibration scaling (platt, temperature, group_logit_temp)
```

### Inference

```python
# 1. Encode questions using frozen LLM
query_emb = encoder(query_input_ids)      # [batch, hidden]
ref_emb = encoder(ref_input_ids)          # [n_ref, hidden]

# 2. Project embeddings
query_proj = regression_head(query_emb)   # [batch, proj_dim]
ref_proj = regression_head(ref_emb)       # [n_ref, proj_dim]

# 3. Attention-based prediction
similarity = query_proj @ ref_proj.T / sqrt(d)  # [batch, n_ref]
weights = softmax(similarity / tau)
difficulty = weights @ ref_difficulties          # [batch]

# 4. Calibration
difficulty = residual_head(difficulty)    # Scaled prediction
```

## Example Configuration

```yaml
# DOTS + Replay configuration
data:
  mu: 2                    # 2 steps per epoch
  tau: 1e-3               # Selection temperature
  alpha: 0.5              # Target difficulty
  sigma: 0.25             # Replay fraction
  buffer_size: 512        # Buffer size
  ref_size: 256           # Reference set size
  replay_strategy: random # or "teacher"

teacher_model:
  model_name: Qwen/Qwen2.5-Math-1.5B-Instruct
  checkpoint_path: adaptive_prediction_model.pt
  scaling: group_logit_temp
  hidden_size: 896
  num_layers: 3
```
