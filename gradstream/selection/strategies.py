"""
Selection strategies for gradient-based data selection in trainers.

This module provides the strategy pattern implementation for clean
separation of selection methods in trainer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Callable, Any

import torch
import torch.nn as nn
from torch import Tensor

from .state import StreamingState, GREATSState

if TYPE_CHECKING:
    from ..hook import GradientHook


class SelectionStrategy(ABC):
    """
    Abstract strategy for data selection in trainers.

    Encapsulates the selection logic so trainers don't need to
    branch on method type directly.
    """

    def __init__(
        self,
        grad_hook: Optional["GradientHook"],
        frac: float,
        use_second_order: bool = False,
        selection_mode: str = "topk"
    ):
        """
        Initialize selection strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Selection fraction / filter fraction
            use_second_order: Use greedy selection with second-order
            selection_mode: "topk" (SFT) or "filtering" (RLHF)
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode

    @property
    def has_compression(self) -> bool:
        """Check if compression (MeSO) is enabled."""
        if self.grad_hook is None:
            return False
        return any(c is not None for c in self.grad_hook.compressors)

    @abstractmethod
    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Execute a complete training step with selection.

        Args:
            model: The model to train
            merged_batch: Merged train+val batch
            train_batch_size: Number of train samples in merged batch
            compute_loss_fn: Function to compute loss
            **kwargs: Additional arguments (lr, batch_train, etc.)

        Returns:
            (loss, stats) tuple where stats includes selection info
        """
        pass


class NoSelectionStrategy(SelectionStrategy):
    """
    Baseline strategy: no data selection, standard training.

    Uses all training samples without selection.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Standard training step without selection."""
        # Disable hooks if present (use standard gradient computation)
        if self.grad_hook is not None and not self.has_compression:
            self.grad_hook.disable_hooks()

        # Extract train-only portion (no need for val in baseline)
        train_batch = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        model.zero_grad()
        loss = compute_loss_fn(model, train_batch)
        loss.backward()

        # Re-enable hooks
        if self.grad_hook is not None and not self.has_compression:
            self.grad_hook.enable_hooks()

        return loss.detach(), {"selection/frac": 1.0, "selection/n_selected": train_batch_size}


class StreamingStrategy(SelectionStrategy):
    """
    Streaming strategy: single-pass, per-layer selection.

    Selection and gradient aggregation happen layer-by-layer
    during the backward pass.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Training step with per-layer selection."""
        lr = kwargs.get('lr', 1e-4)

        # Set up streaming state
        self._setup_state(train_batch_size, lr)

        # Set token counts for proper gradient scaling
        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()  # Per-layer selection happens in backward hooks

        # Get stats
        state = self.grad_hook.selection_state
        stats = {
            "selection/n_selected": state.num_selected if state else train_batch_size,
            "selection/frac": (state.num_selected / train_batch_size) if state else 1.0
        }

        # Cleanup
        self._cleanup()

        return loss.detach(), stats

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up StreamingState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = StreamingState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class GREATSStrategy(SelectionStrategy):
    """
    GREATS strategy: two-pass, global selection.

    Pass 1: Compute selection scores across all layers
    Pass 2: Forward/backward only on globally selected samples
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Training step with global selection."""
        lr = kwargs.get('lr', 1e-4)
        batch_train = kwargs.get('batch_train')  # Original train batch for pass 2

        # === PASS 1: Score Accumulation ===
        self._setup_state(train_batch_size, lr)

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss_for_scoring = compute_loss_fn(model, merged_batch)
        loss_for_scoring.backward()

        # Get globally selected indices
        state: GREATSState = self.grad_hook.selection_state
        selected_indices = state.get_final_selection()
        n_selected = len(selected_indices)

        self._cleanup()

        # === PASS 2: Gradient Computation on Selected ===
        if batch_train is None:
            # Fall back to extracting from merged batch
            batch_train = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        filtered_inputs = {
            'input_ids': batch_train['input_ids'][selected_indices],
            'attention_mask': batch_train['attention_mask'][selected_indices],
            'labels': batch_train['labels'][selected_indices]
        }

        # For pass 2, disable hooks if no compression (we want full gradients)
        if not self.has_compression:
            self.grad_hook.disable_hooks()
        else:
            # For MeSO, set token counts for selected batch
            self.grad_hook.set_token_counts(filtered_inputs['labels'])

        model.zero_grad()
        loss = compute_loss_fn(model, filtered_inputs)
        loss.backward()

        # Re-enable hooks / clear token counts
        if not self.has_compression:
            self.grad_hook.enable_hooks()
        else:
            self.grad_hook.clear_token_counts()

        stats = {
            "selection/n_selected": n_selected,
            "selection/frac": n_selected / train_batch_size
        }

        return loss.detach(), stats

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up GREATSState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = GREATSState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class StreamingWithStoredValStrategy(StreamingStrategy):
    """
    Streaming strategy variant for RLHF with stored validation gradients.

    Uses pre-captured validation gradients instead of merged batch.
    """

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up StreamingState with stored val flag."""
        super()._setup_state(train_batch_size, lr)
        # Mark that we're using stored validation gradients
        self.grad_hook.selection_state._use_stored_val = True


class GREATSWithStoredValStrategy(GREATSStrategy):
    """
    GREATS strategy variant for RLHF with stored validation gradients.

    Uses pre-captured validation gradients instead of merged batch.
    """

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up GREATSState with stored val flag."""
        super()._setup_state(train_batch_size, lr)
        # Mark that we're using stored validation gradients
        self.grad_hook.selection_state._use_stored_val = True


def create_selection_strategy(
    method: str,
    grad_hook: Optional["GradientHook"],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
    use_stored_val: bool = False
) -> SelectionStrategy:
    """
    Factory function to create appropriate selection strategy.

    Args:
        method: Selection method ("NA", "Streaming", "GREATS")
        grad_hook: GradientHook instance
        frac: Selection/filter fraction
        use_second_order: Use greedy selection with second-order
        selection_mode: "topk" (SFT) or "filtering" (RLHF)
        use_stored_val: Use stored validation gradients (RLHF mode)

    Returns:
        Appropriate SelectionStrategy instance
    """
    if method == "NA":
        return NoSelectionStrategy(grad_hook, frac, use_second_order, selection_mode)

    if method == "Streaming":
        if use_stored_val:
            return StreamingWithStoredValStrategy(grad_hook, frac, use_second_order, selection_mode)
        return StreamingStrategy(grad_hook, frac, use_second_order, selection_mode)

    if method == "GREATS":
        if use_stored_val:
            return GREATSWithStoredValStrategy(grad_hook, frac, use_second_order, selection_mode)
        return GREATSStrategy(grad_hook, frac, use_second_order, selection_mode)

    raise ValueError(f"Unknown selection method: {method}")
