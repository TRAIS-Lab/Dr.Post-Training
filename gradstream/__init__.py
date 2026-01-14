"""
Gradient streaming module for unified data selection and model update.

This module provides core components for on-the-fly gradient streaming during training:
- Gradient hooks for capturing and processing gradients layer-by-layer
- Compressors (sparsifiers and projectors) for reducing gradient dimensionality
- MeSO optimizer for memory-efficient subspace optimization
- Selection module with clean separation of Streaming vs GREATS methods
- CompressionMode enum for clear compression configuration
- ValidationCache for unified validation gradient management
"""

from .hook import GradientHook
from .compressor import setup_model_compressors
from .optimizer import MeSOAdamW
from .utils import create_sample_inputs

# Compression mode configuration
from .compression_mode import CompressionMode

# Validation gradient cache
from .validation_cache import ValidationCache, ValidationStorageMode

# Selection module exports (gradient-based)
from .selection import (
    SelectionState,
    StreamingState,
    GREATSState,
    # MergedBatch strategies
    MergedBatchStrategy,
    MergedBatchNoSelectionStrategy,
    MergedBatchStreamingStrategy,
    MergedBatchGREATSStrategy,
    create_merged_batch_strategy,
    # SeparateBatch strategies
    SeparateBatchStrategy,
    SeparateBatchNoSelectionStrategy,
    SeparateBatchStreamingStrategy,
    SeparateBatchGREATSStrategy,
    create_separate_batch_strategy,
)

__all__ = [
    # Core components
    "GradientHook",
    "MeSOAdamW",
    "setup_model_compressors",
    "create_sample_inputs",
    # Compression configuration
    "CompressionMode",
    # Validation cache
    "ValidationCache",
    "ValidationStorageMode",
    # Selection state classes
    "SelectionState",
    "StreamingState",
    "GREATSState",
    # MergedBatch strategy classes
    "MergedBatchStrategy",
    "MergedBatchNoSelectionStrategy",
    "MergedBatchStreamingStrategy",
    "MergedBatchGREATSStrategy",
    "create_merged_batch_strategy",
    # SeparateBatch strategy classes
    "SeparateBatchStrategy",
    "SeparateBatchNoSelectionStrategy",
    "SeparateBatchStreamingStrategy",
    "SeparateBatchGREATSStrategy",
    "create_separate_batch_strategy",
]
