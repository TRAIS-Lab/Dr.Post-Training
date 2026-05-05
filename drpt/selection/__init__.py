"""
Curation module for gradient-based data curation (Dr. Post-Training).

This module provides two families of strategies for computing validation gradients:

1. **MergedBatch Strategies**:
   - Train and val samples are merged into a single batch
   - Val gradients computed during the same forward/backward pass
   - Factory: create_merged_batch_strategy()
   - Note: Has padding overhead when val/train have different sequence lengths

2. **SeparateBatch Strategies**:
   - Val gradients are pre-captured and cached before training
   - Training uses cached val gradients for curation scoring
   - Factory: create_separate_batch_strategy()
   - Avoids padding overhead - val and train can have different seq lengths
   - Val storage mode is derived from scoring_method in start_val_capture():
       * reduced_ghost/direct: Stores total gradient [O, I] per layer.
       Better when validation batch is large (e.g., self-reference validation in RLHF).
       * full_ghost: Stores [V, S, O] and [V, S, I] components (for pairwise scoring).
       More memory-efficient during training as it avoids materializing [B_train, O, I].
       Better when validation batch is small (e.g., external validation set in SFT).
"""

from .state import SelectionState, LayerWiseSubsetState, GlobalSubsetState
from .backward import (
    CompressedLinearBackward,
    LayerWiseSubsetLinearBackward,
    GlobalSubsetLinearBackward,
)
from .strategies import (
    # MergedBatch strategies
    MergedBatchStrategy,
    MergedBatchNoSelectionStrategy,
    MergedBatchLayerWiseSubsetStrategy,
    MergedBatchGlobalSubsetStrategy,
    MergedBatchGlobalSubsetOnePassStrategy,
    create_merged_batch_strategy,
    # SeparateBatch strategies
    SeparateBatchStrategy,
    SeparateBatchNoSelectionStrategy,
    SeparateBatchLayerWiseSubsetStrategy,
    SeparateBatchGlobalSubsetStrategy,
    SeparateBatchGlobalSubsetOnePassStrategy,
    create_separate_batch_strategy,
)

__all__ = [
    # State classes
    "SelectionState",
    "LayerWiseSubsetState",
    "GlobalSubsetState",
    # Autograd functions
    "CompressedLinearBackward",
    "LayerWiseSubsetLinearBackward",
    "GlobalSubsetLinearBackward",
    # MergedBatch strategies
    "MergedBatchStrategy",
    "MergedBatchNoSelectionStrategy",
    "MergedBatchLayerWiseSubsetStrategy",
    "MergedBatchGlobalSubsetStrategy",
    "MergedBatchGlobalSubsetOnePassStrategy",
    "create_merged_batch_strategy",
    # SeparateBatch strategies
    "SeparateBatchStrategy",
    "SeparateBatchNoSelectionStrategy",
    "SeparateBatchLayerWiseSubsetStrategy",
    "SeparateBatchGlobalSubsetStrategy",
    "SeparateBatchGlobalSubsetOnePassStrategy",
    "create_separate_batch_strategy",
]
