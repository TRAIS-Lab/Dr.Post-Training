"""
RLVR Streaming integration for online per-batch selection.

This module provides utilities to integrate activation-based RLVR selection
with training frameworks, enabling:
- Online per-batch selection (instead of epoch-level selection)
- Difficulty-based sample filtering using activation similarity
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Dict, Any
if TYPE_CHECKING:
    from torch import Tensor

import torch
import torch.nn as nn
import logging

from RLVR.difficulty import (
    ActivationHook,
    DifficultyPredictor,
    SimpleCosineSimilarityPredictor,
    RLVRState,
)

logger = logging.getLogger(__name__)


class RLVRStreamingManager:
    """
    Manager for RLVR streaming selection.

    Handles the workflow:
    1. Capture activations during forward pass
    2. Predict difficulties using reference set
    3. Select samples based on difficulty (Goldilocks zone)
    4. Return selected indices for filtered training
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        device: str = 'cuda',
        pooling: str = 'last',
        predictor_type: str = 'simple',  # 'simple' or 'learned'
        predictor_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize RLVR streaming manager.

        Args:
            model: The model to hook
            layer_names: Names of layers to capture activations from
            device: Device for tensor operations
            pooling: Pooling strategy for activations ('last', 'mean', 'first')
            predictor_type: Type of difficulty predictor ('simple' or 'learned')
            predictor_kwargs: Additional kwargs for difficulty predictor
        """
        self.model = model
        self.layer_names = layer_names
        self.device = device
        self.num_layers = len(layer_names)

        # Create activation hook
        self.activation_hook = ActivationHook(
            model=model,
            layer_names=layer_names,
            device=device,
            pooling=pooling,
        )

        # Create difficulty predictor
        predictor_kwargs = predictor_kwargs or {}
        if predictor_type == 'simple':
            self.difficulty_predictor = SimpleCosineSimilarityPredictor(
                tau=predictor_kwargs.get('tau', 1.0)
            )
        elif predictor_type == 'learned':
            # Get hidden dim from model
            hidden_dim = self._get_hidden_dim()
            self.difficulty_predictor = DifficultyPredictor(
                hidden_dim=hidden_dim,
                proj_dim=predictor_kwargs.get('proj_dim', 256),
                tau=predictor_kwargs.get('tau', 1.0),
            ).to(device)
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

        # RLVR selection parameters
        self.alpha = 0.5  # Target difficulty
        self.tau = 0.1    # Selection temperature
        self.selection_ratio = 0.5  # Fraction to select
        self.selection_mode = 'soft'  # 'soft' or 'hard'
        self.aggregation = 'mean'  # How to aggregate layer difficulties

        # Reference data (set via set_reference)
        self._ref_activations: Optional[List[Tensor]] = None
        self._ref_difficulties: Optional[Tensor] = None

        logger.info(
            f"Initialized RLVRStreamingManager with {self.num_layers} layers, "
            f"predictor={predictor_type}, pooling={pooling}"
        )

    def _get_hidden_dim(self) -> int:
        """Get hidden dimension from first layer."""
        for name, module in self.model.named_modules():
            if name in self.layer_names and isinstance(module, nn.Linear):
                return module.in_features
        raise ValueError("Could not determine hidden dimension")

    def set_selection_params(
        self,
        alpha: float = 0.5,
        tau: float = 0.1,
        selection_ratio: float = 0.5,
        selection_mode: str = 'soft',
        aggregation: str = 'mean',
    ) -> None:
        """
        Set RLVR selection parameters.

        Args:
            alpha: Target difficulty (0-1, 0.5 = medium)
            tau: Selection temperature (lower = greedier)
            selection_ratio: Fraction of samples to select
            selection_mode: 'soft' (probabilistic) or 'hard' (top-k)
            aggregation: How to aggregate layer difficulties ('mean', 'max', 'min')
        """
        self.alpha = alpha
        self.tau = tau
        self.selection_ratio = selection_ratio
        self.selection_mode = selection_mode
        self.aggregation = aggregation

    def set_reference(
        self,
        ref_activations: List[Tensor],
        ref_difficulties: Tensor,
    ) -> None:
        """
        Set reference activations and difficulties.

        Args:
            ref_activations: List of [num_ref, hidden_dim] per layer
            ref_difficulties: [num_ref] ground-truth difficulties
        """
        self._ref_activations = ref_activations
        self._ref_difficulties = ref_difficulties

        # Also set on activation hook for convenience
        self.activation_hook.set_reference(ref_activations, ref_difficulties)

        logger.debug(
            f"Set reference: {ref_difficulties.shape[0]} samples, "
            f"{len(ref_activations)} layers"
        )

    def capture_reference_from_rollouts(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        rewards: Tensor,
    ) -> None:
        """
        Capture reference activations from rollout data.

        Args:
            input_ids: [num_ref, seq_len]
            attention_mask: [num_ref, seq_len]
            rewards: [num_ref] reward/difficulty scores
        """
        # Capture activations
        ref_activations = self.activation_hook.capture_activations_for_batch(
            self.model, input_ids, attention_mask
        )

        # Normalize rewards to [0, 1] for difficulty
        ref_difficulties = (rewards - rewards.min()) / (rewards.max() - rewards.min() + 1e-8)

        self.set_reference(ref_activations, ref_difficulties)

    def compute_difficulties(
        self,
        train_activations: List[Tensor],
    ) -> Tensor:
        """
        Compute aggregated difficulties for training samples.

        Args:
            train_activations: List of [batch_size, hidden_dim] per layer

        Returns:
            [batch_size] aggregated difficulty predictions
        """
        if self._ref_activations is None or self._ref_difficulties is None:
            raise RuntimeError("No reference set. Call set_reference() first.")

        layer_difficulties = []

        with torch.no_grad():
            for layer_idx, (train_act, ref_act) in enumerate(
                zip(train_activations, self._ref_activations)
            ):
                if train_act is None or ref_act is None:
                    continue

                # Predict difficulty for this layer
                difficulty = self.difficulty_predictor(
                    train_act, ref_act, self._ref_difficulties, tau=self.tau
                )
                layer_difficulties.append(difficulty)

        if not layer_difficulties:
            raise RuntimeError("No valid layer activations for difficulty prediction")

        # Aggregate across layers
        stacked = torch.stack(layer_difficulties, dim=0)  # [num_layers, batch_size]

        if self.aggregation == 'mean':
            return stacked.mean(dim=0)
        elif self.aggregation == 'max':
            return stacked.max(dim=0)[0]
        elif self.aggregation == 'min':
            return stacked.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def select_samples(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Dict[str, Any]:
        """
        Select samples based on activation-based difficulty prediction.

        This is the main entry point for online RLVR selection.

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            Dict with:
                - 'selected_indices': [num_selected] indices of selected samples
                - 'difficulties': [batch_size] predicted difficulties
                - 'stats': selection statistics
        """
        batch_size = input_ids.shape[0]

        # Capture activations
        train_activations = self.activation_hook.capture_activations_for_batch(
            self.model, input_ids, attention_mask
        )

        # Compute difficulties
        difficulties = self.compute_difficulties(train_activations)

        # Create selection state
        state = RLVRState(
            train_batch_size=batch_size,
            alpha=self.alpha,
            tau=self.tau,
            selection_ratio=self.selection_ratio,
            device=self.device,
            selection_mode=self.selection_mode,
        )

        # Select indices
        selected_indices = state.process_batch(difficulties)

        # Compute stats
        stats = {
            'batch_size': batch_size,
            'num_selected': len(selected_indices),
            'selection_ratio': len(selected_indices) / batch_size,
            'alpha': self.alpha,
            'tau': self.tau,
            'difficulty_mean': difficulties.mean().item(),
            'difficulty_std': difficulties.std().item() if len(difficulties) > 1 else 0,
            'difficulty_min': difficulties.min().item(),
            'difficulty_max': difficulties.max().item(),
        }

        return {
            'selected_indices': selected_indices,
            'difficulties': difficulties,
            'stats': stats,
        }


def create_rlvr_streaming_manager(
    model: nn.Module,
    layer_names: List[str],
    device: str = 'cuda',
    **kwargs,
) -> RLVRStreamingManager:
    """
    Factory function to create RLVR streaming manager.

    Args:
        model: The model to hook
        layer_names: Layer names to capture activations from
        device: Device for tensors
        **kwargs: Additional arguments for RLVRStreamingManager

    Returns:
        RLVRStreamingManager instance
    """
    return RLVRStreamingManager(
        model=model,
        layer_names=layer_names,
        device=device,
        **kwargs,
    )


# Example integration with training:
"""
from RLVR.train.rlvr_streaming import create_rlvr_streaming_manager

# Initialize
rlvr_manager = create_rlvr_streaming_manager(
    model=actor_model,
    layer_names=layer_names,
    device='cuda',
    predictor_type='simple',
)
rlvr_manager.set_selection_params(alpha=0.5, tau=0.1, selection_ratio=0.5)

# Set reference from rollout rewards
rlvr_manager.capture_reference_from_rollouts(ref_input_ids, ref_attention_mask, rewards)

# In training step:
for batch in train_dataloader:
    # Select samples based on difficulty
    result = rlvr_manager.select_samples(
        batch['input_ids'], batch['attention_mask']
    )
    selected_indices = result['selected_indices']

    # Filter batch to selected samples
    filtered_batch = {k: v[selected_indices] for k, v in batch.items()}

    # Forward/backward on filtered batch
    outputs = model(**filtered_batch)
    loss = compute_loss(outputs)
    loss.backward()

    optimizer.step()
"""
