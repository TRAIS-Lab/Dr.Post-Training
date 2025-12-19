"""
Gradient streaming module for unified data selection and model update.

This module provides core components for on-the-fly gradient streaming during training:
- Gradient hooks for capturing and processing gradients layer-by-layer
- Compressors (sparsifiers and projectors) for reducing gradient dimensionality
- MeSO optimizer for memory-efficient subspace optimization

Note: Task-specific trainers (e.g., StreamingTrainer for SFT) are in their respective
experiment directories (e.g., SFT/train/trainer.py).
"""

from .hook import GradientHook
from .compressor import setup_model_compressors
from .optimizer import MeSOAdamW
from .utils import create_sample_inputs

__all__ = [
    "GradientHook",
    "MeSOAdamW",
    "setup_model_compressors",
    "create_sample_inputs",
]
