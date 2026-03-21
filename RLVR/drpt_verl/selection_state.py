"""
Selection state classes for RLVR with Verl.

Extends base library selection states with:
- cu_seqlens support for packed sequences
- Optimized vectorized operations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional
    from torch import Tensor

import torch

# Import selection utilities from base library
import sys
import os
# Add parent path to import from drpt
drpt_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if drpt_path not in sys.path:
    sys.path.insert(0, drpt_path)

from drpt.utils import greedy_selection, topk_selection, negative_filtering


class SelectionStateVerl(ABC):
    """
    Abstract base class for selection state management in RLVR/Verl.

    Extends base SelectionState with cu_seqlens support for packed sequences.
    """

    def __init__(
        self,
        train_batch_size: int,
        num_layers: int,
        frac: float,
        lr: float,
        device: str = 'cpu',
        dtype: torch.dtype = torch.float32,
        use_second_order: bool = False,
        selection_mode: str = "topk"
    ):
        """
        Initialize selection state.

        Args:
            train_batch_size: Number of training samples
            num_layers: Total number of layers
            frac: Fraction parameter (topk: select frac, filtering: drop frac of negative)
            lr: Learning rate for score scaling
            device: Device for tensors
            dtype: Data type for tensors
            use_second_order: If True, use greedy selection with similarity matrix
            selection_mode: "topk" or "filtering"
        """
        self.train_batch_size = train_batch_size
        self.num_layers = num_layers
        self.frac = frac
        self.lr = lr
        self.device = device
        self.dtype = dtype
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode

        self.num_selected = max(1, int(train_batch_size * frac))

        # Token-based scaling
        self.tokens_per_sample: Optional[Tensor] = None
        self.train_total_tokens_tensor: Optional[Tensor] = None

        # Cumulative sequence lengths for packed sequences
        self.cu_seqlens: Optional[Tensor] = None

    def set_token_counts(
        self,
        tokens_per_sample: Tensor,
        total_train_tokens: Tensor,
        packed_tokens_per_sample: Optional[Tensor] = None
    ) -> None:
        """
        Set token counts for gradient scaling.

        Args:
            tokens_per_sample: Response token count per training sample [train_batch_size]
            total_train_tokens: Sum of response tokens in training samples
            packed_tokens_per_sample: Total token count per sample for packed sequences
        """
        # Move all tensors to the correct device to avoid CPU/GPU mismatches
        self.tokens_per_sample = tokens_per_sample.to(self.device)
        self.train_total_tokens_tensor = total_train_tokens.to(device=self.device, dtype=self.dtype)

        # Compute cu_seqlens for packed sequence handling
        cu_source = packed_tokens_per_sample if packed_tokens_per_sample is not None else tokens_per_sample
        cu_source = cu_source.to(self.device)
        cu_seqlens = torch.zeros(len(cu_source) + 1, device=self.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(cu_source, dim=0)
        self.cu_seqlens = cu_seqlens

    def _select_indices(
        self,
        scores: Tensor,
        similarity: Optional[Tensor] = None
    ) -> Tensor:
        """Select sample indices based on scores."""
        scores_scaled = scores * self.lr
        if similarity is not None:
            similarity = similarity * (self.lr ** 2)

        if self.selection_mode == "filtering":
            return negative_filtering(scores_scaled, self.frac)
        elif self.use_second_order and similarity is not None:
            return greedy_selection(scores_scaled, similarity, self.num_selected)
        else:
            return topk_selection(scores_scaled, self.num_selected)

    def _compute_scale_factor(self, selected_indices: Tensor) -> Tensor:
        """Compute token-based gradient scale factor."""
        if self.tokens_per_sample is None or self.train_total_tokens_tensor is None:
            raise RuntimeError("Token counts not set. Call set_token_counts() first.")
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        # All tensors should be on self.device after set_token_counts
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.train_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens == 0, torch.ones_like(scale), scale)

    def is_packed(self) -> bool:
        """Check if using packed sequences."""
        return self.cu_seqlens is not None

    @abstractmethod
    def get_final_selection(self) -> Tensor:
        """Get final selected indices."""
        pass


class LayerwiseStateVerl(SelectionStateVerl):
    """
    Layerwise (layer-wise descent) method state for RLVR/Verl.

    Per-layer selection, single-pass with packed sequence support.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_selected_indices: Optional[Tensor] = None
        self._layer_selections: list = []

    def get_final_selection(self) -> Tensor:
        raise NotImplementedError("LayerwiseState uses per-layer selection.")


class SubsetStateVerl(SelectionStateVerl):
    """
    Subset method state for RLVR/Verl.

    Global selection, two-pass with packed sequence support.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.grad_dot_scores = torch.zeros(
            self.train_batch_size,
            device=self.device,
            dtype=self.dtype
        )

        self.similarity_matrix: Optional[Tensor] = None
        if self.use_second_order:
            self.similarity_matrix = torch.zeros(
                self.train_batch_size, self.train_batch_size,
                device=self.device,
                dtype=self.dtype
            )

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
    ) -> None:
        """Accumulate pre-computed scores."""
        self.grad_dot_scores += scores.to(self.dtype)
        if self.similarity_matrix is not None and similarity is not None:
            self.similarity_matrix += similarity.to(self.dtype)

    def get_final_selection(self) -> Tensor:
        """Compute global selection after all layers."""
        scores = self.grad_dot_scores * self.lr
        similarity = None
        if self.similarity_matrix is not None:
            similarity = self.similarity_matrix * (self.lr ** 2)

        if self.selection_mode == "filtering":
            selected_indices = negative_filtering(scores, self.frac)
        elif self.use_second_order and similarity is not None:
            selected_indices = greedy_selection(scores, similarity, self.num_selected)
        else:
            selected_indices = topk_selection(scores, self.num_selected)

        self.num_selected = len(selected_indices)
        return selected_indices

    def reset_accumulators(self) -> None:
        """Reset accumulators for next batch."""
        self.grad_dot_scores.zero_()
        if self.similarity_matrix is not None:
            self.similarity_matrix.zero_()
