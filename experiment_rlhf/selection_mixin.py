"""
Layer-wise selection mixin for RLHF training.

This module provides the core selection logic for integrating the layer-wise
selection streaming paradigm into PPO training.

Key Design:
- Two-pass approach: capture validation gradients first, then train with selection
- Validation loss uses likelihood on high-quality samples (good direction)
- Training loss uses PPO (policy + value)
- Selection happens layer-by-layer during backward pass
"""

from typing import Dict, List, Optional, Tuple, Any
import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class LayerwiseSelectionMixin:
    """
    Mixin class providing layer-wise selection for RLHF training.

    This implements the two-pass approach:
    1. Capture validation gradients (from high-quality samples)
    2. Train with PPO loss, selecting samples layer-by-layer

    Attributes:
        grad_hook: GradientHook instance for gradient compression and selection
        selection_frac: Fraction of samples to select per layer
        selection_method: Selection method ('GREATS' or 'GradNorm')
        val_loss_type: Type of validation loss ('logprob', 'reward_weighted', etc.)

    Usage:
        class MyPPOTrainer(PPOTrainer, LayerwiseSelectionMixin):
            def __init__(self, ...):
                PPOTrainer.__init__(self, ...)
                LayerwiseSelectionMixin.__init__(self, grad_hook, ...)
    """

    def init_selection(
        self,
        grad_hook,
        selection_frac: float = 0.5,
        selection_method: str = "GREATS",
        val_loss_type: str = "logprob",
        use_second_order: bool = False,
    ):
        """
        Initialize selection-related attributes.

        Args:
            grad_hook: GradientHook instance
            selection_frac: Fraction of samples to select (0-1)
            selection_method: 'GREATS' (gradient alignment) or 'GradNorm' (gradient magnitude)
            val_loss_type: Type of validation loss:
                - 'logprob': Simple negative log-likelihood
                - 'reward_weighted': Log-likelihood weighted by reward
                - 'advantage_weighted': Log-likelihood weighted by advantage
            use_second_order: Whether to use second-order selection (slower but more accurate)
        """
        self.grad_hook = grad_hook
        self.selection_frac = selection_frac
        self.selection_method = selection_method
        self.val_loss_type = val_loss_type
        self.use_second_order = use_second_order

        logger.info("=" * 60)
        logger.info("Initialized LayerwiseSelectionMixin for RLHF")
        logger.info(f"  Selection method: {selection_method}")
        logger.info(f"  Selection fraction: {selection_frac}")
        logger.info(f"  Validation loss type: {val_loss_type}")
        logger.info(f"  Second-order selection: {use_second_order}")
        logger.info("=" * 60)

    def compute_validation_loss(
        self,
        model: nn.Module,
        val_queries: List[Tensor],
        val_responses: List[Tensor],
        val_model_inputs: Dict[str, Tensor],
        val_logprobs: Optional[Tensor] = None,
        val_rewards: Optional[Tensor] = None,
        val_advantages: Optional[Tensor] = None,
        val_masks: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute validation loss for capturing the "good direction".

        The validation loss should represent what we want the model to learn.
        For RLHF, this is typically the likelihood of high-quality responses.

        Args:
            model: The model
            val_queries: Validation query tokens
            val_responses: Validation response tokens
            val_model_inputs: Model inputs (input_ids, attention_mask)
            val_logprobs: Pre-computed log probabilities (optional)
            val_rewards: Rewards for validation samples (optional)
            val_advantages: Advantages for validation samples (optional)
            val_masks: Mask for valid tokens (optional)

        Returns:
            Scalar validation loss
        """
        if self.val_loss_type == "logprob":
            # Simple negative log-likelihood
            # Forward pass to get logits
            outputs = model(
                input_ids=val_model_inputs["input_ids"],
                attention_mask=val_model_inputs["attention_mask"],
                return_dict=True,
            )

            # Get logits and shift for causal LM
            logits = outputs.logits[:, :-1, :]  # [batch, seq-1, vocab]
            labels = val_model_inputs["input_ids"][:, 1:]  # [batch, seq-1]

            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            ).reshape(logits.size(0), -1)  # [batch, seq-1]

            # Apply mask if provided
            if val_masks is not None:
                # Ensure mask matches loss shape
                mask = val_masks[:, :per_token_loss.size(1)]
                per_token_loss = per_token_loss * mask
                val_loss = per_token_loss.sum() / mask.sum().clamp(min=1)
            else:
                val_loss = per_token_loss.mean()

            return val_loss

        elif self.val_loss_type == "reward_weighted":
            # Log-likelihood weighted by reward
            if val_logprobs is None or val_rewards is None:
                raise ValueError("reward_weighted requires val_logprobs and val_rewards")

            if val_masks is None:
                val_loss = -(val_logprobs * val_rewards.unsqueeze(-1)).mean()
            else:
                masked_term = val_logprobs * val_rewards.unsqueeze(-1) * val_masks
                val_loss = -masked_term.sum() / val_masks.sum().clamp(min=1)

            return val_loss

        elif self.val_loss_type == "advantage_weighted":
            # Log-likelihood weighted by advantage (similar to PPO policy loss direction)
            if val_logprobs is None or val_advantages is None:
                raise ValueError("advantage_weighted requires val_logprobs and val_advantages")

            if val_masks is None:
                val_loss = -(val_logprobs * val_advantages).mean()
            else:
                masked_term = val_logprobs * val_advantages * val_masks
                val_loss = -masked_term.sum() / val_masks.sum().clamp(min=1)

            return val_loss

        else:
            raise ValueError(f"Unknown val_loss_type: {self.val_loss_type}")

    def capture_validation_gradients(
        self,
        model: nn.Module,
        val_queries: List[Tensor],
        val_responses: List[Tensor],
        val_model_inputs: Dict[str, Tensor],
        optimizer: torch.optim.Optimizer,
        val_logprobs: Optional[Tensor] = None,
        val_rewards: Optional[Tensor] = None,
        val_advantages: Optional[Tensor] = None,
        val_masks: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Phase 1: Forward/backward on validation data to capture reference gradients.

        This captures the compressed validation gradients for each layer and stores
        them in the hook's val_grad_buffer for use in the training phase.

        Args:
            model: The model
            val_queries: Validation query tokens
            val_responses: Validation response tokens
            val_model_inputs: Model inputs (input_ids, attention_mask)
            optimizer: Optimizer (for zero_grad)
            val_logprobs: Pre-computed log probabilities
            val_rewards: Rewards for validation samples
            val_advantages: Advantages for validation samples
            val_masks: Mask for valid tokens

        Returns:
            Validation loss (for logging)
        """
        # Set hook to capture mode
        self.grad_hook.set_mode("capture_val")
        self.grad_hook.enable_hooks()

        # Compute validation loss
        val_loss = self.compute_validation_loss(
            model=model,
            val_queries=val_queries,
            val_responses=val_responses,
            val_model_inputs=val_model_inputs,
            val_logprobs=val_logprobs,
            val_rewards=val_rewards,
            val_advantages=val_advantages,
            val_masks=val_masks,
        )

        # Backward - hooks will capture compressed gradients per layer
        val_loss.backward()

        # Clear optimizer gradients (we only wanted to capture compressed grads)
        optimizer.zero_grad()

        logger.debug(f"Captured validation gradients from {len(self.grad_hook.val_grad_buffer)} layers")

        return val_loss.detach()

    def setup_training_selection(
        self,
        train_batch_size: int,
        lr: float,
        compute_scores_only: bool = False,
    ) -> None:
        """
        Set up for training phase with selection.

        Args:
            train_batch_size: Number of training samples in batch
            lr: Current learning rate
            compute_scores_only: If True, only accumulate scores (for global selection)
        """
        # Set hook to train_select mode
        self.grad_hook.set_mode("train_select")

        # Setup selection state
        self.grad_hook.setup_selection(
            train_batch_size=train_batch_size,
            selection_method=self.selection_method,
            selection_frac=self.selection_frac,
            lr=lr,
            compute_scores_only=compute_scores_only,
            use_second_order=self.use_second_order,
        )

    def finish_training_step(self) -> None:
        """
        Clean up after training step.
        """
        self.grad_hook.clear_selection()
        self.grad_hook.set_mode("standard")

    def get_selection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current selection state.

        Returns:
            Dictionary with selection statistics
        """
        stats = {
            "mode": self.grad_hook.mode,
            "val_grad_buffer_size": len(self.grad_hook.val_grad_buffer),
            "selection_frac": self.selection_frac,
            "selection_method": self.selection_method,
        }

        if self.grad_hook.selection_state is not None:
            state = self.grad_hook.selection_state
            stats["train_batch_size"] = state.train_batch_size
            stats["num_selected"] = state.num_selected

            if state.grad_dot_scores is not None:
                stats["mean_score"] = state.grad_dot_scores.mean().item()
                stats["max_score"] = state.grad_dot_scores.max().item()
                stats["min_score"] = state.grad_dot_scores.min().item()

        return stats
