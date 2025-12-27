"""
Selection module for gradient-based data selection.

This module provides two families of strategies for computing validation gradients:

1. **JointBatch Strategies**:
   - Train and val samples are merged into a single batch
   - Val gradients computed during the same forward/backward pass
   - Factory: create_joint_batch_strategy()
   - Note: Has padding overhead when val/train have different sequence lengths

2. **CachedVal Strategies**:
   - Val gradients are pre-captured and cached before training
   - Training uses cached val gradients for selection scoring
   - Factory: create_cached_val_strategy()
   - Avoids padding overhead - val and train can have different seq lengths
   - Supports two caching modes via start_val_capture(use_factorized=...):
     * Cached grad mode (use_factorized=False): Stores total gradient [O, I] per layer.
       Better when validation batch is large (e.g., self-reference validation in RLHF).
     * Cached factors mode (use_factorized=True): Stores [V, S, O] and [V, S, I] components.
       More memory-efficient during training as it avoids materializing [B_train, O, I].
       Better when validation batch is small (e.g., external validation set in SFT).
"""

from .state import SelectionState, StreamingState, GREATSState
from .backward import (
    CompressedLinearBackward,
    StreamingLinearBackward,
    GREATSLinearBackward,
)
from .strategies import (
    # JointBatch strategies
    JointBatchStrategy,
    JointBatchNoSelectionStrategy,
    JointBatchStreamingStrategy,
    JointBatchGREATSStrategy,
    create_joint_batch_strategy,
    # CachedVal strategies
    CachedValStrategy,
    CachedValNoSelectionStrategy,
    CachedValStreamingStrategy,
    CachedValGREATSStrategy,
    create_cached_val_strategy,
)

__all__ = [
    # State classes
    "SelectionState",
    "StreamingState",
    "GREATSState",
    # Autograd functions
    "CompressedLinearBackward",
    "StreamingLinearBackward",
    "GREATSLinearBackward",
    # JointBatch strategies
    "JointBatchStrategy",
    "JointBatchNoSelectionStrategy",
    "JointBatchStreamingStrategy",
    "JointBatchGREATSStrategy",
    "create_joint_batch_strategy",
    # CachedVal strategies
    "CachedValStrategy",
    "CachedValNoSelectionStrategy",
    "CachedValStreamingStrategy",
    "CachedValGREATSStrategy",
    "create_cached_val_strategy",
]
