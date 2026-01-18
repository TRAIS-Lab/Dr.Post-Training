"""
Selection-aware PPO Actor wrapper for VERL.

This module provides a wrapper around VERL's DataParallelPPOActor that integrates
gradient-based data selection (Streaming or GREATS) during policy updates.

Usage:
    1. Create SelectionAwareActor with selection config
    2. Before each training step, capture validation gradients
    3. Call update_policy_with_selection() instead of update_policy()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List, Optional
    from torch import Tensor
    from verl import DataProto

import torch
import torch.nn as nn
import logging

from .hook import GradientHookVerl
from .validation import capture_seqloss_reward_gradients, prepare_val_batch_from_rollout

logger = logging.getLogger(__name__)


@dataclass
class SelectionConfig:
    """Configuration for gradient-based data selection."""

    # Selection method: "Streaming", "GREATS", or "NA" (disabled)
    method: str = "NA"

    # Fraction of samples to select (0.0 to 1.0)
    frac: float = 0.5

    # Use second-order selection (similarity matrix for greedy selection)
    use_second_order: bool = False

    # Selection mode: "topk" or "filtering"
    selection_mode: str = "topk"

    # Validation data path (parquet file)
    val_data_path: Optional[str] = None

    # Validation batch size
    val_batch_size: int = 64

    # How often to refresh validation gradients (1 = every step)
    refresh_freq: int = 1

    # Micro-batch size for validation gradient capture (None = full batch)
    val_micro_batch_size: Optional[int] = None


def get_trainable_linear_layers(model: nn.Module) -> List[str]:
    """
    Get names of trainable Linear layers in the model.

    Args:
        model: PyTorch model

    Returns:
        List of fully-qualified layer names
    """
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.requires_grad:
            layer_names.append(name)
    return layer_names


class SelectionAwareActor:
    """
    Wrapper that adds gradient selection to VERL's PPO actor.

    This wrapper integrates with VERL's DataParallelPPOActor to enable
    gradient-based data selection during policy updates.

    Key features:
    - Validation gradient capture using seqloss-reward objective
    - Streaming selection (per-layer, single pass)
    - GREATS selection (global, two pass)
    - FSDP compatibility

    Example:
        actor = DataParallelPPOActor(...)
        selection_actor = SelectionAwareActor(actor, SelectionConfig(method="Streaming"))

        # During training:
        selection_actor.capture_validation_gradients(val_batch)
        selection_actor.setup_selection(train_batch_size, lr)
        metrics = selection_actor.update_policy_with_selection(data)
    """

    def __init__(
        self,
        actor: Any,  # DataParallelPPOActor - avoid import cycle
        config: SelectionConfig,
    ):
        """
        Initialize selection-aware actor.

        Args:
            actor: VERL's DataParallelPPOActor instance
            config: Selection configuration
        """
        self.actor = actor
        self.config = config
        self.grad_hook: Optional[GradientHookVerl] = None
        self._step_count = 0

        if config.method != "NA":
            self._setup_gradient_hook()

    def _setup_gradient_hook(self) -> None:
        """Initialize gradient hooks on trainable Linear layers."""
        actor_module = self.actor.actor_module

        # Get trainable Linear layer names
        layer_names = get_trainable_linear_layers(actor_module)
        logger.info(f"Setting up gradient hooks on {len(layer_names)} Linear layers")

        # Determine device
        device = next(actor_module.parameters()).device

        # Initialize hook manager
        self.grad_hook = GradientHookVerl(
            model=actor_module,
            layer_names=layer_names,
            device=device,
        )

    @property
    def actor_module(self) -> nn.Module:
        """Get the underlying actor module."""
        return self.actor.actor_module

    def capture_validation_gradients(
        self,
        val_batch: Dict[str, Tensor],
        force: bool = False,
    ) -> float:
        """
        Capture validation gradients before training step.

        Args:
            val_batch: Validation batch containing:
                - input_ids: [batch, seq_len]
                - attention_mask: [batch, seq_len]
                - response_ids: [batch, response_len]
                - response_mask: [batch, response_len]
                - rewards: [batch]
            force: Force capture even if not at refresh interval

        Returns:
            Validation loss value
        """
        if self.grad_hook is None or self.config.method == "NA":
            return 0.0

        # Check if we should capture this step
        should_capture = force or (self._step_count % self.config.refresh_freq == 0)
        if not should_capture:
            return 0.0

        # Clear any existing validation gradients
        self.grad_hook.clear_val_cache()

        # Capture validation gradients
        val_loss = capture_seqloss_reward_gradients(
            model=self.actor_module,
            grad_hook=self.grad_hook,
            val_batch=val_batch,
            micro_batch_size=self.config.val_micro_batch_size,
        )

        logger.debug(f"Captured validation gradients, loss={val_loss:.4f}")
        return val_loss

    def capture_validation_gradients_from_rollout(
        self,
        rollout_batch: Dict[str, Tensor],
        rewards: Tensor,
        force: bool = False,
    ) -> float:
        """
        Capture validation gradients from rollout data.

        Convenience method that prepares the validation batch from rollout format.

        Args:
            rollout_batch: Rollout data from verl
            rewards: Per-sequence rewards
            force: Force capture even if not at refresh interval

        Returns:
            Validation loss value
        """
        val_batch = prepare_val_batch_from_rollout(rollout_batch, rewards)
        return self.capture_validation_gradients(val_batch, force=force)

    def setup_selection(
        self,
        train_batch_size: int,
        lr: float,
    ) -> None:
        """
        Set up selection state after validation gradient capture.

        Must be called after capture_validation_gradients() and before
        update_policy_with_selection().

        Args:
            train_batch_size: Number of training samples in the batch
            lr: Learning rate (used for gradient scaling)
        """
        if self.grad_hook is None or self.config.method == "NA":
            return

        if not self.grad_hook.has_val_grads():
            raise RuntimeError(
                "No validation gradients captured. Call capture_validation_gradients() first."
            )

        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=train_batch_size,
            selection_method=self.config.method,
            frac=self.config.frac,
            lr=lr,
            use_second_order=self.config.use_second_order,
            selection_mode=self.config.selection_mode,
        )
        logger.debug(f"Set up {self.config.method} selection for {train_batch_size} samples")

    def update_policy_with_selection(self, data: "DataProto") -> Dict[str, Any]:
        """
        Update policy with gradient selection enabled.

        For Streaming: Hooks are enabled during backward pass to perform
        per-layer selection.

        For GREATS: Two passes are performed - score accumulation followed
        by gradient computation on selected samples.

        Args:
            data: Training data (verl DataProto)

        Returns:
            Metrics dictionary from policy update
        """
        self._step_count += 1

        if self.grad_hook is None or self.config.method == "NA":
            return self.actor.update_policy(data)

        if self.config.method == "Streaming":
            return self._update_policy_streaming(data)
        elif self.config.method == "GREATS":
            return self._update_policy_greats(data)
        else:
            raise ValueError(f"Unknown selection method: {self.config.method}")

    def _update_policy_streaming(self, data: "DataProto") -> Dict[str, Any]:
        """Update policy with Streaming selection."""
        # Enable hooks for selection during backward
        self.grad_hook.enable_hooks()

        try:
            # Call original update_policy
            # Hooks will intercept backward and perform per-layer selection
            metrics = self.actor.update_policy(data)
        finally:
            # Disable hooks and clear state
            self.grad_hook.disable_hooks()
            self.grad_hook.clear_selection()

        return metrics

    def _update_policy_greats(self, data: "DataProto") -> Dict[str, Any]:
        """
        Update policy with GREATS selection.

        GREATS requires two passes:
        1. Score accumulation pass (hooks compute and accumulate scores)
        2. Gradient computation pass (standard backward on selected samples)

        NOTE: This is a simplified implementation that doesn't yet support
        the two-pass mechanism. For now, it falls back to Streaming.
        Full GREATS implementation requires modifying the training loop.
        """
        # TODO: Implement full GREATS two-pass mechanism
        # For now, use Streaming as fallback
        logger.warning(
            "GREATS two-pass not fully implemented yet, falling back to Streaming"
        )
        return self._update_policy_streaming(data)

    def clear_caches(self) -> None:
        """Clear all cached gradients and selection state."""
        if self.grad_hook is not None:
            self.grad_hook.clear_val_cache()
            self.grad_hook.clear_selection()

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get statistics about the last selection."""
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            return {}

        state = self.grad_hook.selection_state
        return {
            "selection/method": self.config.method,
            "selection/frac": self.config.frac,
            "selection/train_batch_size": state.train_batch_size if hasattr(state, 'train_batch_size') else 0,
        }

    # Delegate attribute access to the wrapped actor
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped actor."""
        return getattr(self.actor, name)
