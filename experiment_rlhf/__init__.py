"""
RLHF experiments with layer-wise selection and compressed gradients.

This module integrates the layer-wise selection streaming paradigm into
RLHF (PPO) training, enabling efficient data selection during policy optimization.

Key Components:
- LayerwiseSelectionMixin: Mixin class for adding selection to any trainer
- PPOTrainerWithSelection: Wrapper for PPO trainer with layer-wise selection

Usage:
    from experiment_rlhf import PPOTrainerWithSelection
    from compress_gradient import GradientHook

    # Setup gradient hook
    grad_hook = GradientHook(model, layer_names, device)

    # Wrap PPO trainer
    selection_trainer = PPOTrainerWithSelection(
        ppo_trainer=ppo_trainer,
        grad_hook=grad_hook,
        selection_frac=0.5,
        selection_method="GREATS",
    )

    # Train with layer-wise selection
    stats = selection_trainer.step_with_layerwise_selection(...)
"""

from .selection_mixin import LayerwiseSelectionMixin
from .ppo_trainer import PPOTrainerWithSelection

__all__ = [
    "LayerwiseSelectionMixin",
    "PPOTrainerWithSelection",
]
