"""
RLVR Training utilities.

Provides integration with VERL PPO trainer for activation-based selection.
"""

from .rlvr_selection import (
    RLVRSelectionManager,
    filter_batch_by_indices,
    compute_goldilocks_scores,
    softmax_sample,
)

__all__ = [
    "RLVRSelectionManager",
    "filter_batch_by_indices",
    "compute_goldilocks_scores",
    "softmax_sample",
]
