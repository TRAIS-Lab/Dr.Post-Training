"""
Selection module for gradient-based data selection.

This module provides two families of strategies:

1. **Merged Batch Strategies** (for SFT):
   - Train and val samples are merged into a single batch
   - Val gradients computed during the same forward/backward pass
   - Factory: create_selection_strategy()

2. **Stored Val Strategies** (for RLHF):
   - Val gradients are pre-captured and stored before training
   - Training uses stored val gradients for selection scoring
   - Factory: create_stored_val_strategy()
"""

from .state import SelectionState, StreamingState, GREATSState
from .backward import (
    CompressedLinearBackward,
    StreamingLinearBackward,
    GREATSLinearBackward,
)
from .strategies import (
    # Merged batch strategies (SFT)
    SelectionStrategy,
    NoSelectionStrategy,
    StreamingStrategy,
    GREATSStrategy,
    create_selection_strategy,
    # Stored val strategies (RLHF)
    StoredValStrategy,
    StoredValNoSelectionStrategy,
    StoredValStreamingStrategy,
    StoredValGREATSStrategy,
    create_stored_val_strategy,
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
    # Merged batch strategies (SFT)
    "SelectionStrategy",
    "NoSelectionStrategy",
    "StreamingStrategy",
    "GREATSStrategy",
    "create_selection_strategy",
    # Stored val strategies (RLHF)
    "StoredValStrategy",
    "StoredValNoSelectionStrategy",
    "StoredValStreamingStrategy",
    "StoredValGREATSStrategy",
    "create_stored_val_strategy",
]
