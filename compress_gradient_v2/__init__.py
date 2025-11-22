"""
Gradient compression module for efficient fine-tuning.

This module provides components for compressing gradients during training:
- Gradient hooks for capturing and processing gradients
- Compressors (sparsifiers and projectors) for reducing gradient dimensionality
- Custom trainer with gradient compression support
"""

from .hook import GradientHook
from .trainer import CompGradTrainer
from .compressor import setup_model_compressors
from .utils import create_sample_inputs

__all__ = [
    "GradientHook",
    "CompGradTrainer",
    "setup_model_compressors",
    "create_sample_inputs",
]
