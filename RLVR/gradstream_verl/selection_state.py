"""
Selection state classes for RLVR with Verl.

Extends base gradstream selection states with:
- cu_seqlens support for packed sequences
- Optimized vectorized operations
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor

import torch

# Import selection utilities from base gradstream
import sys
import os
# Add parent path to import from gradstream
gradstream_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if gradstream_path not in sys.path:
    sys.path.insert(0, gradstream_path)

from gradstream.utils import greedy_selection, topk_selection, negative_filtering


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

        # Score correction for joint batch mode
        self.score_correction: Optional[Tensor] = None

        # Cumulative sequence lengths for packed sequences
        self.cu_seqlens: Optional[Tensor] = None

    def set_token_counts(
        self,
        tokens_per_sample: Tensor,
        total_train_tokens: Tensor,
        batch_total_tokens: Tensor,
        packed_tokens_per_sample: Optional[Tensor] = None
    ) -> None:
        """
        Set token counts for gradient scaling and score correction.

        Args:
            tokens_per_sample: Response token count per training sample [train_batch_size]
            total_train_tokens: Sum of response tokens in training samples
            batch_total_tokens: Sum of tokens in entire batch
            packed_tokens_per_sample: Total token count per sample for packed sequences
        """
        self.tokens_per_sample = tokens_per_sample
        self.train_total_tokens_tensor = total_train_tokens.to(dtype=self.dtype)

        # Compute cu_seqlens for packed sequence handling
        cu_source = packed_tokens_per_sample if packed_tokens_per_sample is not None else tokens_per_sample
        cu_seqlens = torch.zeros(len(cu_source) + 1, device=cu_source.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(cu_source, dim=0)
        self.cu_seqlens = cu_seqlens

        # Precompute score correction for joint batch mode
        val_tokens = batch_total_tokens - total_train_tokens
        correction = (batch_total_tokens.to(self.dtype) ** 2) / (
            total_train_tokens.to(self.dtype) * val_tokens.to(self.dtype)
        )
        valid_mask = (val_tokens > 0) & (total_train_tokens > 0)
        self.score_correction = torch.where(
            valid_mask,
            correction,
            torch.ones((), device=tokens_per_sample.device, dtype=self.dtype)
        )

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
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.train_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens == 0, torch.ones_like(scale), scale)

    def is_packed(self) -> bool:
        """Check if using packed sequences."""
        return self.cu_seqlens is not None

    @abstractmethod
    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> Optional[Tuple[Tensor, int]]:
        """Process gradients for a single layer."""
        pass

    @abstractmethod
    def get_final_selection(self) -> Tensor:
        """Get final selected indices."""
        pass


class StreamingStateVerl(SelectionStateVerl):
    """
    Streaming method state for RLVR/Verl.

    Per-layer selection, single-pass with packed sequence support.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_selected_indices: Optional[Tensor] = None
        self._layer_selections: list = []

    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """Immediately select and aggregate at this layer."""
        import logging
        logger = logging.getLogger(__name__)

        # Compute scores
        scores = train_grads @ val_grad
        if score_correction is not None:
            scores = scores * score_correction

        # Debug: Log score distribution for this layer (every 50th layer to reduce spam)
        if layer_idx % 50 == 0:
            pos_count = (scores > 0).sum().item()
            neg_count = (scores < 0).sum().item()
            logger.info(
                f"[DEBUG] Streaming Layer {layer_idx}: "
                f"scores mean={scores.mean().item():.4f}, std={scores.std().item():.4f}, "
                f"pos={pos_count}, neg={neg_count}"
            )

        # Compute similarity if second-order
        similarity = None
        if self.use_second_order:
            similarity = train_grads @ train_grads.T
            if score_correction is not None:
                similarity = similarity * (score_correction ** 2)

        # Select indices
        selected_indices = self._select_indices(scores, similarity)
        selected_indices = selected_indices.sort()[0]
        self._last_selected_indices = selected_indices
        num_selected = selected_indices.shape[0]

        self._layer_selections.append((layer_idx, num_selected))

        # Debug: Log selection details for this layer (every 50th layer)
        if layer_idx % 50 == 0:
            logger.info(
                f"[DEBUG] Streaming Layer {layer_idx}: "
                f"selected {num_selected}/{self.train_batch_size} samples"
            )

        # Aggregate selected gradients
        selected_grads = train_grads[selected_indices]
        reduced_grad = selected_grads.sum(dim=0, keepdim=True)

        # Apply scale factor
        scale_factor = self._compute_scale_factor(selected_indices)
        reduced_grad = reduced_grad * scale_factor

        self.num_selected = num_selected
        return reduced_grad, num_selected

    def get_final_selection(self) -> Tensor:
        raise NotImplementedError("StreamingState uses per-layer selection.")


class GREATSStateVerl(SelectionStateVerl):
    """
    GREATS method state for RLVR/Verl.

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

    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> None:
        """Accumulate scores - no immediate selection."""
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        alpha = score_correction.item() if score_correction is not None else 1.0
        torch.addmv(self.grad_dot_scores, train_grads, val_grad, alpha=alpha, out=self.grad_dot_scores)

        if self.similarity_matrix is not None:
            alpha_sq = alpha ** 2
            torch.addmm(self.similarity_matrix, train_grads, train_grads.t(), alpha=alpha_sq, out=self.similarity_matrix)

        return None

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
        score_correction: Optional[Tensor] = None,
    ) -> None:
        """Accumulate pre-computed scores."""
        if score_correction is not None:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)

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
