"""
Gradient streaming module for unified data selection and model update.

This module provides core components for on-the-fly gradient streaming during training:
- Gradient hooks for capturing and processing gradients layer-by-layer
- Compressors (sparsifiers and projectors) for reducing gradient dimensionality
- MeSO optimizer for memory-efficient subspace optimization
- Selection module with clean separation of Streaming vs GREATS methods

Note: Task-specific trainers (e.g., StreamingTrainer for SFT) are in their respective
experiment directories (e.g., SFT/train/trainer.py).
"""

from .hook import GradientHook
from .compressor import setup_model_compressors
from .optimizer import MeSOAdamW
from .utils import create_sample_inputs

# Selection module exports
from .selection import (
    SelectionState,
    StreamingState,
    GREATSState,
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
    # Core components
    "GradientHook",
    "MeSOAdamW",
    "setup_model_compressors",
    "create_sample_inputs",
    # Selection state classes
    "SelectionState",
    "StreamingState",
    "GREATSState",
    # JointBatch strategy classes
    "JointBatchStrategy",
    "JointBatchNoSelectionStrategy",
    "JointBatchStreamingStrategy",
    "JointBatchGREATSStrategy",
    "create_joint_batch_strategy",
    # CachedVal strategy classes
    "CachedValStrategy",
    "CachedValNoSelectionStrategy",
    "CachedValStreamingStrategy",
    "CachedValGREATSStrategy",
    "create_cached_val_strategy",
]
