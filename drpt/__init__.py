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
    LayerWiseSubset: per-layer curation in a single backward pass.
    GlobalSubset: global curation. Two modes:
      one_pass (Algorithm 4.2): score + retain during backward, post-hoc gradient assembly.
      two_pass (Algorithm 4.3): scoring pass, then gradient pass on selected subset.
    Scoring methods: reduced_ghost (default), full_ghost, direct, compress.
    Both use score_compressors for influence scoring when configured.

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
    LayerWiseSubsetState,
    GlobalSubsetState,
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
    "LayerWiseSubsetState",
    "GlobalSubsetState",
    # MergedBatch strategy classes
    "MergedBatchStrategy",
    "MergedBatchNoSelectionStrategy",
    "MergedBatchLayerWiseSubsetStrategy",
    "MergedBatchGlobalSubsetStrategy",
    "MergedBatchGlobalSubsetOnePassStrategy",
    "create_merged_batch_strategy",
    # SeparateBatch strategy classes
    "SeparateBatchStrategy",
    "SeparateBatchNoSelectionStrategy",
    "SeparateBatchLayerWiseSubsetStrategy",
    "SeparateBatchGlobalSubsetStrategy",
    "SeparateBatchGlobalSubsetOnePassStrategy",
    "create_separate_batch_strategy",
]
