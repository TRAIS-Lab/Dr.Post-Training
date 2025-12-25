"""
Selection module for gradient-based data selection.

This module provides clean separation between:
- Streaming: Per-layer selection, single-pass
- GREATS: Global selection, two-pass
"""

from .state import SelectionState, StreamingState, GREATSState
from .backward import StreamingLinearBackward, GREATSLinearBackward
from .strategies import (
    SelectionStrategy,
    NoSelectionStrategy,
    StreamingStrategy,
    GREATSStrategy,
    create_selection_strategy,
)

__all__ = [
    # State classes
    "SelectionState",
    "StreamingState",
    "GREATSState",
    # Autograd functions
    "StreamingLinearBackward",
    "GREATSLinearBackward",
    # Strategy classes
    "SelectionStrategy",
    "NoSelectionStrategy",
    "StreamingStrategy",
    "GREATSStrategy",
    "create_selection_strategy",
]
