"""
RLVR Selection Integration for VERL PPO Trainer.

This module provides utilities to integrate RLVR activation-based selection
with the existing VERL training loop:
- GREATS: Online per-batch selection (filter before loss)
- Streaming: Per-layer selection (filter during backward)

Usage in ray_trainer_teacher.py:
    from RLVR.train.rlvr_selection import RLVRSelectionManager

    # In __init__:
    self.rlvr_manager = RLVRSelectionManager(config)

    # After rollout collection:
    self.rlvr_manager.update_reference_from_rollouts(rollout_data, model)

    # In training loop (GREATS mode):
    selected_indices = self.rlvr_manager.select_batch(batch, model)
    # ... filter batch and compute loss on selected samples

    # In training loop (Streaming mode):
    self.rlvr_manager.setup_streaming(batch, model)
    # ... compute loss normally, backward will filter gradients
    self.rlvr_manager.cleanup_streaming()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
if TYPE_CHECKING:
    from torch import Tensor
    from omegaconf import DictConfig

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RLVRSelectionManager:
    """
    Manager for RLVR selection in VERL training.

    Handles:
    1. Reference set management (from rollouts)
    2. GREATS: batch-level selection before loss
    3. Streaming: per-layer selection during backward
    """

    def __init__(
        self,
        config: "DictConfig",
        device: str = 'cuda',
    ):
        """
        Initialize RLVR selection manager.

        Args:
            config: Training config with RLVR parameters
            device: Device for tensor operations
        """
        self.config = config
        self.device = device

        # RLVR parameters (from config.data)
        self.alpha = getattr(config.data, 'alpha', 0.5)
        self.tau = getattr(config.data, 'tau', 0.1)
        self.selection_ratio = getattr(config.data, 'selection_ratio', 0.5)
        self.selection_mode = getattr(config.data, 'selection_mode', 'soft')
        self.selection_method = getattr(config.data, 'selection_method', 'GREATS')
        self.difficulty_aggregation = getattr(config.data, 'difficulty_aggregation', 'mean')
        self.predictor_type = getattr(config.data, 'predictor_type', 'simple')

        # RLVRHook instance (created lazily when model is available)
        self.rlvr_hook: Optional["RLVRHook"] = None
        self._hook_initialized = False

        # Reference data cache
        self._ref_input_ids: Optional[Tensor] = None
        self._ref_attention_mask: Optional[Tensor] = None
        self._ref_difficulties: Optional[Tensor] = None

        # Statistics
        self.selection_stats: Dict[str, Any] = {}

        logger.info(
            f"Initialized RLVRSelectionManager: method={self.selection_method}, "
            f"alpha={self.alpha}, tau={self.tau}, ratio={self.selection_ratio}"
        )

    def _initialize_hook(self, model: nn.Module) -> None:
        """Initialize RLVR hook on model (lazy initialization)."""
        if self._hook_initialized:
            return

        from RLVR import RLVRHook, get_trainable_layer_names

        # Get trainable layer names
        layer_names = get_trainable_layer_names(model)

        if len(layer_names) == 0:
            logger.warning("No trainable layers found, RLVR selection disabled")
            return

        # Create hook
        self.rlvr_hook = RLVRHook(
            model=model,
            layer_names=layer_names,
            device=self.device,
            predictor_type=self.predictor_type,
        )

        # Patch layers for Streaming mode
        if self.selection_method == 'Streaming':
            self.rlvr_hook.patch_linear_layers()

        self._hook_initialized = True
        logger.info(f"Initialized RLVRHook with {len(layer_names)} layers")

    # ==========================================
    # Reference Set Management
    # ==========================================

    def update_reference_from_rollouts(
        self,
        rollout_data: Dict[str, Any],
        model: nn.Module,
    ) -> None:
        """
        Update reference set from rollout data.

        Called per rollout generation to update the reference activations
        and difficulties.

        Args:
            rollout_data: Dict with 'input_ids', 'attention_mask', 'rewards'
            model: The actor model for activation capture
        """
        self._initialize_hook(model)

        if self.rlvr_hook is None:
            return

        # Extract reference data from rollouts
        input_ids = rollout_data.get('input_ids')
        attention_mask = rollout_data.get('attention_mask')
        rewards = rollout_data.get('rewards')

        if input_ids is None or rewards is None:
            logger.warning("Missing input_ids or rewards in rollout_data")
            return

        # Convert to tensors if needed
        if isinstance(input_ids, np.ndarray):
            input_ids = torch.from_numpy(input_ids)
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask)
        if isinstance(rewards, np.ndarray):
            rewards = torch.from_numpy(rewards)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        else:
            attention_mask = torch.ones_like(input_ids)
        rewards = rewards.float().to(self.device)

        # Capture reference activations
        self.rlvr_hook.capture_reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            difficulties=rewards,  # Use rewards as difficulties
        )

        # Cache reference data
        self._ref_input_ids = input_ids
        self._ref_attention_mask = attention_mask
        self._ref_difficulties = rewards

        logger.debug(f"Updated reference: {len(rewards)} samples")

    def update_reference_from_batch_data(
        self,
        ref_data: Dict[str, Any],
        model: nn.Module,
    ) -> None:
        """
        Update reference from processed batch data (alternative to rollouts).

        Args:
            ref_data: Dict with keys as sample indices, values with 'rewards' list
            model: The actor model
        """
        self._initialize_hook(model)

        if self.rlvr_hook is None:
            return

        # Extract mean rewards as difficulties
        difficulties = []
        indices = []
        for key, data in ref_data.items():
            if 'rewards' in data:
                difficulties.append(np.mean(data['rewards']))
                indices.append(key)

        difficulties = torch.tensor(difficulties, dtype=torch.float32, device=self.device)

        # Note: This requires input_ids to be available
        # For now, just store the difficulties
        self._ref_difficulties = difficulties
        logger.debug(f"Updated reference difficulties: {len(difficulties)} samples")

    def has_reference(self) -> bool:
        """Check if reference is set."""
        if self.rlvr_hook is None:
            return False
        return self.rlvr_hook.has_reference()

    # ==========================================
    # GREATS Mode: Batch-level Selection
    # ==========================================

    def select_batch(
        self,
        batch: Dict[str, Tensor],
        model: nn.Module,
        return_difficulties: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Select samples from batch using GREATS mode.

        Captures activations, computes difficulties, and returns selected indices.
        Filter the batch using these indices before computing loss.

        Args:
            batch: Dict with 'input_ids', 'attention_mask'
            model: The actor model
            return_difficulties: If True, also return difficulty predictions

        Returns:
            selected_indices: [num_selected] indices into the batch
            difficulties: [batch_size] if return_difficulties=True
        """
        self._initialize_hook(model)

        if self.rlvr_hook is None or not self.has_reference():
            # No RLVR: return all indices
            batch_size = batch['input_ids'].shape[0]
            indices = torch.arange(batch_size, device=self.device)
            if return_difficulties:
                return indices, torch.ones(batch_size, device=self.device) * 0.5
            return indices

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Setup selection
        batch_size = input_ids.shape[0]
        self.rlvr_hook.setup_selection(
            selection_method='GREATS',
            train_batch_size=batch_size,
            alpha=self.alpha,
            tau=self.tau,
            selection_ratio=self.selection_ratio,
            selection_mode=self.selection_mode,
        )

        # Capture activations
        self.rlvr_hook.start_capture(attention_mask)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        self.rlvr_hook.end_capture()

        # Compute selections
        self.rlvr_hook.compute_selections(aggregation=self.difficulty_aggregation)

        # Get results
        selected_indices = self.rlvr_hook.get_global_selection()
        self.selection_stats = self.rlvr_hook.get_selection_stats()

        # Cleanup
        self.rlvr_hook.clear_selection()
        self.rlvr_hook.clear_activations()

        if return_difficulties:
            difficulties = self.rlvr_hook.compute_aggregated_difficulty(self.difficulty_aggregation)
            return selected_indices, difficulties

        return selected_indices

    # ==========================================
    # Streaming Mode: Per-layer Selection
    # ==========================================

    def setup_streaming_selection(
        self,
        batch: Dict[str, Tensor],
        model: nn.Module,
        labels: Optional[Tensor] = None,
    ) -> None:
        """
        Setup Streaming mode selection.

        Call before forward pass. After forward pass, call compute_streaming_selections().
        The backward pass will automatically use pre-computed selections.

        Args:
            batch: Dict with 'input_ids', 'attention_mask'
            model: The actor model
            labels: Optional labels for token count computation
        """
        self._initialize_hook(model)

        if self.rlvr_hook is None or not self.has_reference():
            return

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        batch_size = input_ids.shape[0]

        # Setup selection state
        self.rlvr_hook.setup_selection(
            selection_method='Streaming',
            train_batch_size=batch_size,
            alpha=self.alpha,
            tau=self.tau,
            selection_ratio=self.selection_ratio,
            selection_mode=self.selection_mode,
        )

        # Set token counts for gradient scaling
        if labels is not None:
            self.rlvr_hook.set_token_counts(labels)

        # Start activation capture (will be captured during forward)
        self.rlvr_hook.start_capture(attention_mask)

    def compute_streaming_selections(self) -> None:
        """
        Compute per-layer selections after forward pass.

        Call after forward pass, before backward pass.
        """
        if self.rlvr_hook is None:
            return

        self.rlvr_hook.end_capture()
        self.rlvr_hook.compute_selections(aggregation=self.difficulty_aggregation)
        self.selection_stats = self.rlvr_hook.get_selection_stats()

    def cleanup_streaming_selection(self) -> None:
        """
        Cleanup after backward pass.
        """
        if self.rlvr_hook is None:
            return

        self.rlvr_hook.clear_selection()
        self.rlvr_hook.clear_activations()
        self.rlvr_hook.clear_token_counts()

    # ==========================================
    # Utility Methods
    # ==========================================

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get selection statistics from last selection."""
        return self.selection_stats

    def is_enabled(self) -> bool:
        """Check if RLVR selection is enabled."""
        return self.selection_method in ['GREATS', 'Streaming']

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.rlvr_hook is not None:
            self.rlvr_hook.remove_hooks()
            if self.selection_method == 'Streaming':
                self.rlvr_hook.unpatch_linear_layers()


def filter_batch_by_indices(
    batch: Dict[str, Any],
    indices: Tensor,
) -> Dict[str, Any]:
    """
    Filter batch tensors by selected indices.

    Args:
        batch: Dict with tensor values
        indices: Selected indices

    Returns:
        Filtered batch
    """
    filtered = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            filtered[key] = value[indices]
        elif isinstance(value, np.ndarray):
            filtered[key] = value[indices.cpu().numpy()]
        elif isinstance(value, list):
            indices_list = indices.cpu().tolist()
            filtered[key] = [value[i] for i in indices_list]
        else:
            filtered[key] = value
    return filtered


def compute_goldilocks_scores(
    difficulties: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """
    Compute Goldilocks zone scores.

    Score = -|difficulty - alpha|
    Higher score = closer to target difficulty.

    Args:
        difficulties: [batch_size] predicted difficulties
        alpha: Target difficulty (0-1)

    Returns:
        [batch_size] selection scores
    """
    return -torch.abs(difficulties - alpha)


def softmax_sample(
    scores: Tensor,
    num_samples: int,
    tau: float = 0.1,
    replacement: bool = False,
) -> Tensor:
    """
    Sample indices based on softmax of scores.

    Args:
        scores: [batch_size] selection scores
        num_samples: Number of samples to select
        tau: Temperature (lower = greedier)
        replacement: Whether to sample with replacement

    Returns:
        [num_samples] selected indices
    """
    logits = scores / tau
    logits = logits - logits.max()  # Numerical stability
    probs = torch.softmax(logits, dim=0)
    return torch.multinomial(probs, num_samples, replacement=replacement)
