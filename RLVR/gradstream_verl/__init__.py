"""
Gradient streaming module for RLVR with Verl.

This is a specialized version of gradstream optimized for multi-GPU training
with Verl's packed sequence format (use_remove_padding=True).

Key features:
- Native cu_seqlens support for packed sequences
- Vectorized operations to avoid D2H memory copies
- FSDP-aware layer detection
- seqloss-reward validation gradient capture for GRPO
- Streaming and GREATS selection methods
- Online validation rollout generation (matching RLHF design)

Usage:
    from gradstream_verl import SelectionAwareActor, SelectionConfig

    config = SelectionConfig(method="Streaming", frac=0.5)
    selection_actor = SelectionAwareActor(actor, config)

    # During training:
    selection_actor.capture_validation_gradients(val_batch)
    selection_actor.setup_selection(train_batch_size, lr)
    metrics = selection_actor.update_policy_with_selection(data)

For online validation rollouts (recommended):
    from gradstream_verl import ValidationDataManager, ValidationConfig
    from gradstream_verl import SelectionRayPPOTrainerWithOnlineVal, SelectionTrainerConfig

    # Create validation manager with prompt pool
    val_config = ValidationConfig(
        val_prompts_path="data/val_prompts.parquet",
        val_pool_size=500,
        val_batch_size=32,
    )
    val_manager = ValidationDataManager(val_config, tokenizer)

    # Training with online validation rollouts
    trainer = SelectionRayPPOTrainerWithOnlineVal(
        config=config,
        selection_config=SelectionTrainerConfig(enable=True, method="Streaming"),
        ...
    )
"""

from .hook import GradientHookVerl
from .selection_state import SelectionStateVerl, StreamingStateVerl, GREATSStateVerl
from .validation import capture_seqloss_reward_gradients, prepare_val_batch_from_rollout
from .actor_selection import SelectionAwareActor, SelectionConfig, get_trainable_linear_layers

# Import the archive's complete actor implementation
# This is a drop-in replacement for DataParallelPPOActor with selection support
try:
    from .dp_actor_selection import DataParallelPPOActorWithSelection
except ImportError as e:
    # verl may not be installed, make it optional
    DataParallelPPOActorWithSelection = None

# Import validation data manager (online rollout support)
try:
    from .validation_manager import (
        ValidationConfig,
        ValidationDataManager,
        ValidationPromptDataset,
        prepare_validation_batch_for_verl,
        create_validation_prompts_from_train,
    )
except ImportError as e:
    ValidationConfig = None
    ValidationDataManager = None
    ValidationPromptDataset = None
    prepare_validation_batch_for_verl = None
    create_validation_prompts_from_train = None

# Import selection trainer (online validation rollouts)
try:
    from .selection_trainer import (
        SelectionTrainerConfig,
        SelectionRayPPOTrainerWithOnlineVal,
        create_selection_trainer,
    )
except ImportError as e:
    SelectionTrainerConfig = None
    SelectionRayPPOTrainerWithOnlineVal = None
    create_selection_trainer = None

__all__ = [
    # Hook manager
    "GradientHookVerl",
    # Selection states
    "SelectionStateVerl",
    "StreamingStateVerl",
    "GREATSStateVerl",
    # Validation gradient capture
    "capture_seqloss_reward_gradients",
    "prepare_val_batch_from_rollout",
    # Actor wrapper (simple)
    "SelectionAwareActor",
    "SelectionConfig",
    "get_trainable_linear_layers",
    # Actor with selection (complete, from archive)
    "DataParallelPPOActorWithSelection",
    # Validation data manager (online rollout support)
    "ValidationConfig",
    "ValidationDataManager",
    "ValidationPromptDataset",
    "prepare_validation_batch_for_verl",
    "create_validation_prompts_from_train",
    # Selection trainer (online validation rollouts)
    "SelectionTrainerConfig",
    "SelectionRayPPOTrainerWithOnlineVal",
    "create_selection_trainer",
]
