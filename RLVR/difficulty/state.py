"""
Selection state classes for RLVR difficulty-based data selection.

Provides two state classes:
- RLVRState: Global selection based on difficulty (batch-level)
- RLVRStreamingState: Per-layer selection based on per-layer difficulty
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, List, Tuple
    from torch import Tensor

import torch
import logging

logger = logging.getLogger(__name__)


class RLVRStateBase(ABC):
    """
    Abstract base class for RLVR selection state.

    RLVR selects samples based on difficulty proximity to a target (alpha).
    Unlike gradient-based selection which maximizes alignment,
    RLVR prefers "medium difficulty" samples (Goldilocks zone).
    """

    def __init__(
        self,
        train_batch_size: int,
        alpha: float = 0.5,
        tau: float = 0.1,
        selection_ratio: float = 0.5,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        selection_mode: str = 'soft',  # 'soft' (probabilistic) or 'hard' (deterministic)
    ):
        """
        Initialize RLVR selection state.

        Args:
            train_batch_size: Number of training samples in batch
            alpha: Target difficulty level (0-1, default 0.5 = medium)
            tau: Temperature for selection softmax (lower = greedier)
            selection_ratio: Fraction of samples to select (for hard mode)
            device: Device for tensors
            dtype: Data type for tensors
            selection_mode: 'soft' (multinomial sampling) or 'hard' (top-k)
        """
        self.train_batch_size = train_batch_size
        self.alpha = alpha
        self.tau = tau
        self.selection_ratio = selection_ratio
        self.device = device
        self.dtype = dtype
        self.selection_mode = selection_mode

        # Number of samples to select (for hard mode)
        self.num_to_select = max(1, int(train_batch_size * selection_ratio))

        # Track selection stats
        self.num_selected: int = 0
        self._selection_stats: dict = {}

    def compute_selection_scores(self, difficulties: Tensor) -> Tensor:
        """
        Compute selection scores from difficulty predictions.

        RLVR prefers samples close to target difficulty alpha.
        Score = -|difficulty - alpha| (higher = closer to target)

        Args:
            difficulties: [batch_size] predicted difficulty scores

        Returns:
            [batch_size] selection scores (higher = more likely to select)
        """
        # Score = negative absolute distance from target
        scores = -torch.abs(difficulties - self.alpha)
        return scores

    def select_indices(
        self,
        difficulties: Tensor,
        num_to_select: Optional[int] = None,
    ) -> Tensor:
        """
        Select sample indices based on difficulties.

        Args:
            difficulties: [batch_size] predicted difficulty scores
            num_to_select: Override number of samples to select

        Returns:
            [num_selected] selected indices
        """
        if num_to_select is None:
            num_to_select = self.num_to_select

        scores = self.compute_selection_scores(difficulties)

        if self.selection_mode == 'hard':
            # Deterministic: select top-k by score
            _, indices = torch.topk(scores, k=num_to_select)
        else:
            # Soft: probabilistic sampling based on scores
            # Apply temperature and softmax
            logits = scores / self.tau
            logits = logits - logits.max()  # Numerical stability
            probs = torch.softmax(logits, dim=0)

            # Sample without replacement
            indices = torch.multinomial(probs, num_samples=num_to_select, replacement=False)

        self.num_selected = len(indices)
        return indices.sort()[0]  # Sort for cache locality

    @abstractmethod
    def process_batch(self, *args, **kwargs) -> Tensor:
        """Process a batch and return selected indices."""
        pass

    def get_stats(self) -> dict:
        """Get selection statistics."""
        return {
            "num_selected": self.num_selected,
            "selection_ratio": self.num_selected / self.train_batch_size,
            **self._selection_stats,
        }


class RLVRState(RLVRStateBase):
    """
    Global RLVR selection state (batch-level).

    Selects samples once per batch based on aggregated difficulty
    across all layers (or single-layer difficulty).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Accumulated difficulties across layers (for multi-layer aggregation)
        self._layer_difficulties: List[Tensor] = []

    def process_batch(
        self,
        difficulties: Tensor,
        num_to_select: Optional[int] = None,
    ) -> Tensor:
        """
        Select indices for the batch based on difficulties.

        Args:
            difficulties: [batch_size] aggregated difficulty predictions
            num_to_select: Override number of samples to select

        Returns:
            [num_selected] selected indices
        """
        # Track difficulty stats
        self._selection_stats.update({
            "difficulty_mean": difficulties.mean().item(),
            "difficulty_std": difficulties.std().item() if len(difficulties) > 1 else 0,
            "difficulty_min": difficulties.min().item(),
            "difficulty_max": difficulties.max().item(),
        })

        return self.select_indices(difficulties, num_to_select)

    def accumulate_layer_difficulty(self, layer_difficulty: Tensor, layer_idx: int) -> None:
        """
        Accumulate difficulty prediction from a layer.

        For global selection with multiple layers, accumulate predictions
        and then aggregate for final selection.

        Args:
            layer_difficulty: [batch_size] difficulty for this layer
            layer_idx: Layer index
        """
        # Extend list if needed
        while len(self._layer_difficulties) <= layer_idx:
            self._layer_difficulties.append(None)

        self._layer_difficulties[layer_idx] = layer_difficulty

    def get_aggregated_difficulty(self, aggregation: str = 'mean') -> Tensor:
        """
        Get aggregated difficulty across accumulated layers.

        Args:
            aggregation: 'mean', 'max', 'min'

        Returns:
            [batch_size] aggregated difficulties
        """
        valid = [d for d in self._layer_difficulties if d is not None]
        if not valid:
            raise RuntimeError("No layer difficulties accumulated")

        stacked = torch.stack(valid, dim=0)  # [num_layers, batch_size]

        if aggregation == 'mean':
            return stacked.mean(dim=0)
        elif aggregation == 'max':
            return stacked.max(dim=0)[0]
        elif aggregation == 'min':
            return stacked.min(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def reset(self) -> None:
        """Reset state for next batch."""
        self._layer_difficulties = []
        self.num_selected = 0
        self._selection_stats = {}


class RLVRStreamingState(RLVRStateBase):
    """
    Per-layer RLVR selection state (streaming).

    Each layer independently selects samples based on its own
    difficulty prediction, similar to gradient Streaming.
    """

    def __init__(self, num_layers: int, **kwargs):
        super().__init__(**kwargs)

        self.num_layers = num_layers

        # Track per-layer selection stats
        self._layer_selections: List[Tuple[int, int]] = []  # (layer_idx, num_selected)
        self._layer_stats: List[dict] = []

    def process_layer(
        self,
        layer_difficulties: Tensor,
        layer_idx: int,
        num_to_select: Optional[int] = None,
    ) -> Tensor:
        """
        Select indices for a specific layer based on its difficulties.

        Args:
            layer_difficulties: [batch_size] difficulty predictions for this layer
            layer_idx: Index of the current layer
            num_to_select: Override number of samples to select

        Returns:
            [num_selected] selected indices for this layer
        """
        # Track stats for this layer
        layer_stat = {
            "difficulty_mean": layer_difficulties.mean().item(),
            "difficulty_std": layer_difficulties.std().item() if len(layer_difficulties) > 1 else 0,
            "difficulty_min": layer_difficulties.min().item(),
            "difficulty_max": layer_difficulties.max().item(),
        }
        self._layer_stats.append(layer_stat)

        # Select indices
        indices = self.select_indices(layer_difficulties, num_to_select)

        # Track selection
        self._layer_selections.append((layer_idx, len(indices)))

        return indices

    def process_batch(
        self,
        difficulties: Tensor,
        num_to_select: Optional[int] = None,
    ) -> Tensor:
        """
        Fallback for batch-level processing.

        For streaming, use process_layer instead.
        """
        return self.select_indices(difficulties, num_to_select)

    def get_selection_summary(self) -> dict:
        """Get summary of per-layer selections."""
        if not self._layer_selections:
            return {"num_layers": 0, "mean_selected": 0}

        selections = [n for _, n in self._layer_selections]
        return {
            "num_layers": len(selections),
            "mean_selected": sum(selections) / len(selections),
            "min_selected": min(selections),
            "max_selected": max(selections),
        }

    def get_difficulty_summary(self) -> dict:
        """Get summary of difficulty predictions across layers."""
        if not self._layer_stats:
            return {}

        means = [s["difficulty_mean"] for s in self._layer_stats]
        return {
            "difficulty_mean_avg": sum(means) / len(means),
            "difficulty_mean_min": min(means),
            "difficulty_mean_max": max(means),
        }

    def reset(self) -> None:
        """Reset state for next batch."""
        self._layer_selections = []
        self._layer_stats = []
        self.num_selected = 0
        self._selection_stats = {}
