"""
Selection strategies for gradient-based data selection in trainers.

This module provides two families of strategies based on how validation
gradients are obtained (merged validation batch vs. separate pass).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional, Callable, Tuple
    from ..hook import GradientHook
    from torch import Tensor

import torch.nn as nn
from .state import StreamingState, GREATSState


# ============================================================
# MERGED BATCH STRATEGIES for when target function = train loss
# Val gradients computed from merged train+val batch
# ============================================================

class SelectionStrategy(ABC):
    """
    Abstract strategy for merged-batch data selection.

    Used when train and val samples are merged into a single batch,
    and val gradients are computed during the same forward/backward pass.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
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
    ) -> Tensor:
        """
        Execute a complete training step with selection.

        Args:
            model: The model to train
            merged_batch: Merged train+val batch
            train_batch_size: Number of train samples in merged batch
            compute_loss_fn: Function to compute loss
            **kwargs: Additional arguments (lr, batch_train, etc.)

        Returns:
            Detached loss tensor
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
    ) -> Tensor:
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

        return loss.detach()


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
    ) -> Tensor:
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

        # Cleanup
        self._cleanup()

        return loss.detach()

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
    ) -> Tensor:
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

        return loss.detach()

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


def create_selection_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
) -> SelectionStrategy:
    """
    Factory function to create merged-batch selection strategy.

    Args:
        method: Selection method ("NA", "Streaming", "GREATS")
        grad_hook: GradientHook instance
        frac: Selection/filter fraction
        use_second_order: Use greedy selection with second-order
        selection_mode: "topk" (SFT) or "filtering" (RLHF)

    Returns:
        Appropriate SelectionStrategy instance
    """
    if method == "NA":
        return NoSelectionStrategy(grad_hook, frac, use_second_order, selection_mode)

    if method == "Streaming":
        return StreamingStrategy(grad_hook, frac, use_second_order, selection_mode)

    if method == "GREATS":
        return GREATSStrategy(grad_hook, frac, use_second_order, selection_mode)

    raise ValueError(f"Unknown selection method: {method}")


# ============================================================
# STORED VAL STRATEGIES (for RLHF)
# Val gradients pre-captured and stored before training
# ============================================================

class StoredValStrategy(ABC):
    """
    Abstract strategy for stored-val data selection.

    Used when val gradients are pre-captured and stored before training,
    rather than computed from a merged batch during the same forward pass.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
        frac: float,
        use_second_order: bool = False,
    ):
        """
        Initialize stored-val selection strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Fraction of negative-influence samples to DROP (0-1)
            use_second_order: Use greedy selection with second-order
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order

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
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """
        Execute a complete training step with selection.

        Args:
            model: The model to train
            batch_size: Number of samples in the batch
            compute_loss_fn: Zero-arg function that computes loss and returns (loss, stats)
            lr: Learning rate for score scaling
            **kwargs: Additional arguments (filter_batch_fn for GREATS)

        Returns:
            Tuple of (loss, stats_dict) where stats includes selection metrics
        """
        pass


class StoredValNoSelectionStrategy(StoredValStrategy):
    """
    Baseline strategy: no data selection, standard training.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Standard training step without selection."""
        # Disable hooks for baseline (use standard gradient computation)
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()

        # Re-enable hooks
        if self.grad_hook is not None:
            self.grad_hook.enable_hooks()

        return loss.detach(), stats


class StoredValStreamingStrategy(StoredValStrategy):
    """
    Streaming strategy with stored val: per-layer selection.

    Selection and gradient aggregation happen layer-by-layer during backward,
    using pre-captured validation gradients.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with per-layer selection using stored val grads."""
        # Set up streaming state with stored validation gradients
        self._setup_state(batch_size, lr)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()  # Per-layer selection happens in backward hooks

        # Get selection stats before cleanup
        if self.grad_hook.selection_state is not None:
            n_selected = self.grad_hook.selection_state.num_selected
            stats["selection/n_selected"] = n_selected
            stats["selection/frac"] = n_selected / batch_size

        # Cleanup
        self._cleanup()

        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up StreamingState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="Streaming",
            frac=self.frac,
            lr=lr,
            compute_scores_only=False,
            use_second_order=self.use_second_order,
            selection_mode="filtering",  # Drop negative samples
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()


class StoredValGREATSStrategy(StoredValStrategy):
    """
    GREATS strategy with stored val: global selection.

    Pass 1: Compute selection scores across all layers
    Pass 2: Forward/backward only on globally selected samples
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with global selection using stored val grads."""
        # filter_batch_fn: Callable[[Tensor], Callable] that takes selected_indices
        # and returns a new compute_loss_fn for the filtered batch
        filter_batch_fn = kwargs.get('filter_batch_fn')
        if filter_batch_fn is None:
            raise ValueError("GREATS strategy requires 'filter_batch_fn' in kwargs")

        # === PASS 1: Score Accumulation ===
        self._setup_state(batch_size, lr)

        model.zero_grad()
        loss_for_scoring, _ = compute_loss_fn()
        loss_for_scoring.backward()

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        n_selected = len(selected_indices)

        self._cleanup()

        # === PASS 2: Gradient Computation on Selected ===
        self.grad_hook.disable_hooks()
        model.zero_grad()

        # Get filtered compute_loss_fn for selected samples
        filtered_compute_loss_fn = filter_batch_fn(selected_indices)
        loss, stats = filtered_compute_loss_fn()
        loss.backward()

        # Re-enable hooks for next step
        self.grad_hook.enable_hooks()

        # Add selection stats
        stats["selection/n_selected"] = n_selected
        stats["selection/n_total"] = batch_size
        stats["selection/frac"] = n_selected / batch_size

        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up GREATSState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="GREATS",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,  # Only accumulate scores in pass 1
            use_second_order=self.use_second_order,
            selection_mode="filtering",  # Drop negative samples
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()


def create_stored_val_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
) -> StoredValStrategy:
    """
    Factory function to create stored-val selection strategy.

    Args:
        method: Selection method ("NA", "Streaming", "GREATS")
        grad_hook: GradientHook instance
        frac: Fraction of negative-influence samples to DROP
        use_second_order: Use greedy selection with second-order

    Returns:
        Appropriate StoredValStrategy instance
    """
    if method == "NA":
        return StoredValNoSelectionStrategy(grad_hook, frac, use_second_order)

    if method == "Streaming":
        return StoredValStreamingStrategy(grad_hook, frac, use_second_order)

    if method == "GREATS":
        return StoredValGREATSStrategy(grad_hook, frac, use_second_order)

    raise ValueError(f"Unknown selection method: {method}")
