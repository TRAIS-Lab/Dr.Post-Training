"""
Gradient streaming module for RLVR with Verl.

This is a specialized version of gradstream optimized for multi-GPU training
with Verl's packed sequence format (use_remove_padding=True).

Key differences from base gradstream:
- Native cu_seqlens support for packed sequences
- Vectorized operations to avoid D2H memory copies
- FSDP-aware layer detection
"""

from .hook_verl import GradientHookVerl
from .selection_state import SelectionStateVerl, StreamingStateVerl, GREATSStateVerl

__all__ = [
    "GradientHookVerl",
    "SelectionStateVerl",
    "StreamingStateVerl",
    "GREATSStateVerl",
]
