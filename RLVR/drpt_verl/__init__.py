"""
Gradient-based selection module for RLVR with Verl.

This module provides gradient-based data selection for PPO training with Verl.

Key features:
- LayerWiseSubset and GlobalSubset selection methods
- Online validation rollout generation
- Zero-variance handling for validation gradients
- FSDP-compatible gradient capture

Usage:
    from drpt_verl import (
        DataParallelPPOActorWithSelection,
        SelectionRayPPOTrainerWithOnlineVal,
        SelectionTrainerConfig,
    )

    # Create trainer with selection
    trainer = SelectionRayPPOTrainerWithOnlineVal(
        config=config,
        selection_config=SelectionTrainerConfig(enable=True, method="LayerWiseSubset"),
        ...
    )
"""

from .hook import GradientHookVerl
from .selection_state import SelectionStateVerl, LayerWiseSubsetStateVerl, GlobalSubsetStateVerl
from .validation import (
    normalize_rewards_for_validation,
    VAR_THRESHOLD,
    BINARY_BASELINE,
)

# Import the main actor implementation with selection support
try:
    from .dp_actor_selection import DataParallelPPOActorWithSelection
except ImportError:
    DataParallelPPOActorWithSelection = None

# Import validation data manager
try:
    from .validation_manager import (
        ValidationConfig,
        ValidationDataManager,
        ValidationPromptDataset,
        prepare_validation_batch_for_verl,
        create_validation_prompts_from_train,
    )
except ImportError:
    ValidationConfig = None
    ValidationDataManager = None
    ValidationPromptDataset = None
    prepare_validation_batch_for_verl = None
    create_validation_prompts_from_train = None

# Import selection trainer
try:
    from .selection_trainer import (
        SelectionTrainerConfig,
        SelectionRayPPOTrainerWithOnlineVal,
        create_selection_trainer,
    )
except ImportError:
    SelectionTrainerConfig = None
    SelectionRayPPOTrainerWithOnlineVal = None
    create_selection_trainer = None

__all__ = [
    # Hook manager
    "GradientHookVerl",
    # Selection states
    "SelectionStateVerl",
    "LayerWiseSubsetStateVerl",
    "GlobalSubsetStateVerl",
    # Validation utilities
    "normalize_rewards_for_validation",
    "VAR_THRESHOLD",
    "BINARY_BASELINE",
    # Actor with selection
    "DataParallelPPOActorWithSelection",
    # Validation data manager
    "ValidationConfig",
    "ValidationDataManager",
    "ValidationPromptDataset",
    "prepare_validation_batch_for_verl",
    "create_validation_prompts_from_train",
    # Selection trainer
    "SelectionTrainerConfig",
    "SelectionRayPPOTrainerWithOnlineVal",
    "create_selection_trainer",
]
