"""
RLVR (Reinforcement Learning with Verifiable Rewards) experiment.

RLVR implements difficulty-based sample selection for PPO training:
- Reference set with ground-truth difficulties from rollouts
- Attention-based activation similarity for difficulty prediction
- Selection of medium-difficulty samples (Goldilocks zone)

Selection Methods:
- GREATS: Online per-batch selection, filter batch before loss
- Streaming: Per-layer selection, filter gradients during backward

Key Components:
- RLVRHook: Unified hook manager for activation capture + selection
- difficulty/: Predictors, states, and backward functions
- adaptive_difficulty_prediction/: Pre-training pipeline for learned predictors
"""

from .hook import RLVRHook, create_rlvr_hook, get_trainable_layer_names

# Re-export from difficulty module
from .difficulty import (
    # Activation capture
    ActivationHook,
    # Predictor classes
    DifficultyPredictor,
    LayerWiseDifficultyPredictor,
    SimpleCosineSimilarityPredictor,
    create_difficulty_predictor,
    # State classes
    RLVRState,
    RLVRStreamingState,
    # Backward functions
    RLVRStreamingLinearBackward,
)

__all__ = [
    # Main hook
    "RLVRHook",
    "create_rlvr_hook",
    "get_trainable_layer_names",
    # Activation capture (legacy, use RLVRHook instead)
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
