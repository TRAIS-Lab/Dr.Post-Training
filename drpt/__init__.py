"""
Dr. Post-Training (drpt): Data Regularization for LLM Post-Training.

Architecture overview:

  GradientHook
    Monkey-patches Linear layers with custom autograd Functions.
    Maintains two independent compressor lists:
      score_compressors  — for influence score computation (data curation)
      update_compressors — for MeSO optimizer updates
    When both use the same config, they share objects (zero overhead).

  Compressor (compressor.py)
    Two-stage gradient compression: Sparsifier → Projector.
    Named schemes: LoGra (normal + none), GraSS (random_mask + sjlt).
    See compressor.py docstring for details.

  MeSOAdamW (optimizer.py)
    Memory-efficient optimizer maintaining states in compressed space.
    Reads compressed gradients from update_compressors via the hook.

  Curation (selection/)
    Layerwise: per-layer curation in a single backward pass.
    Subset: global curation via two-pass (score accumulation → filtered update).
    Both use score_compressors for influence scoring.

  CompressionMode (compression_mode.py)
    Derived from which compressor lists are populated:
      NONE, SCORE_ONLY, UPDATE_ONLY, FULL.

  ValidationCache (validation_cache.py)
    Stores validation gradients in factorized, full, or compressed form.
"""

from .hook import GradientHook
from .compressor import setup_model_compressors
from .optimizer import MeSOAdamW
from .utils import create_sample_inputs

# Compression mode configuration
from .compression_mode import CompressionMode

# Validation gradient cache
from .validation_cache import ValidationCache, ValidationStorageMode

# Curation module exports (gradient-based)
from .selection import (
    SelectionState,
    LayerwiseState,
    SubsetState,
    # MergedBatch strategies
    MergedBatchStrategy,
    MergedBatchNoSelectionStrategy,
    MergedBatchLayerwiseStrategy,
    MergedBatchSubsetStrategy,
    create_merged_batch_strategy,
    # SeparateBatch strategies
    SeparateBatchStrategy,
    SeparateBatchNoSelectionStrategy,
    SeparateBatchLayerwiseStrategy,
    SeparateBatchSubsetStrategy,
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
    # Curation state classes
    "SelectionState",
    "LayerwiseState",
    "SubsetState",
    # MergedBatch strategy classes
    "MergedBatchStrategy",
    "MergedBatchNoSelectionStrategy",
    "MergedBatchLayerwiseStrategy",
    "MergedBatchSubsetStrategy",
    "create_merged_batch_strategy",
    # SeparateBatch strategy classes
    "SeparateBatchStrategy",
    "SeparateBatchNoSelectionStrategy",
    "SeparateBatchLayerwiseStrategy",
    "SeparateBatchSubsetStrategy",
    "create_separate_batch_strategy",
]
