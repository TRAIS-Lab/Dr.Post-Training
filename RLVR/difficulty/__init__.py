"""
RLVR difficulty prediction and selection module.

This module provides RLVR-specific difficulty prediction and selection:
- ActivationHook: Captures layer activations during forward pass
- DifficultyPredictor: Attention-based few-shot regressor for difficulty prediction
- RLVRState: Selection state for difficulty-based sample selection
- RLVRStreamingLinearBackward: Autograd function for per-layer activation-based selection
"""

from .activation_hook import ActivationHook
from .predictor import (
    DifficultyPredictor,
    LayerWiseDifficultyPredictor,
    SimpleCosineSimilarityPredictor,
    create_difficulty_predictor,
)
from .state import RLVRState, RLVRStreamingState
from .backward import RLVRStreamingLinearBackward

__all__ = [
    # Activation capture
    "ActivationHook",
    # Predictor classes
    "DifficultyPredictor",
    "LayerWiseDifficultyPredictor",
    "SimpleCosineSimilarityPredictor",
    "create_difficulty_predictor",
    # State classes
    "RLVRState",
    "RLVRStreamingState",
    # Backward functions
    "RLVRStreamingLinearBackward",
]
