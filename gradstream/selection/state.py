"""
Selection state classes for gradient-based data selection.

This module provides two distinct state classes:
- StreamingState: Per-layer selection, single-pass
- GREATSState: Global selection, two-pass score accumulation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor

import torch

from ..utils import greedy_selection, topk_selection, negative_filtering


class SelectionState(ABC):
    """
    Abstract base class for selection state management during backward pass.

    Subclasses implement different selection strategies:
    - StreamingState: Per-layer selection, immediate gradient aggregation
    - GREATSState: Score accumulation, global selection after all layers
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
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select
                  - "filtering": Fraction of negative-influence samples to DROP
            lr: Learning rate for score scaling
            device: Device for tensors
            dtype: Data type for tensors
            use_second_order: If True, use greedy selection with second-order interactions
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        """
        self.train_batch_size = train_batch_size
        self.num_layers = num_layers
        self.frac = frac
        self.lr = lr
        self.device = device
        self.dtype = dtype
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode

        # Number of samples to select (for top-k mode)
        self.num_selected = max(1, int(train_batch_size * frac))

        # Token-based scaling (set via set_token_counts)
        self.tokens_per_sample: Optional[Tensor] = None
        self.train_total_tokens_tensor: Optional[Tensor] = None

        # Precomputed score correction for joint batch mode (1.0 = no correction)
        # Stored as Tensor to avoid D2H memory copies during backward
        self.score_correction: Optional[Tensor] = None

    def set_token_counts(
        self,
        tokens_per_sample: Tensor,
        total_train_tokens: Tensor,
        batch_total_tokens: Tensor,
    ) -> None:
        """
        Set token counts for gradient scaling and score correction.

        Args:
            tokens_per_sample: Response token count per training sample [train_batch_size].
                              Used for gradient scaling (sum over selected samples).
            total_train_tokens: Sum of response tokens in training samples (scalar Tensor)
            batch_total_tokens: Sum of tokens in entire batch (train + val for joint batch, scalar Tensor)
        """
        # Store for gradient scaling: train_total_tokens / selected_tokens
        self.tokens_per_sample = tokens_per_sample
        self.train_total_tokens_tensor = total_train_tokens.to(dtype=self.dtype)

        # Precompute score correction for joint batch mode
        # All operations kept on device as Tensors to avoid D2H memcpy
        val_tokens = batch_total_tokens - total_train_tokens
        # Use torch.where to handle the conditional without branching on CPU values
        correction = (batch_total_tokens.to(self.dtype) ** 2) / (total_train_tokens.to(self.dtype) * val_tokens.to(self.dtype))
        # Set to 1.0 if val_tokens <= 0 or total_train_tokens <= 0
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
        """
        Select sample indices based on scores.

        Args:
            scores: Per-sample scores [train_batch_size]
            similarity: Optional similarity matrix [train_batch_size, train_batch_size]

        Returns:
            Selected indices tensor
        """
        # Apply lr scaling
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
        """
        Compute token-based gradient scale factor for selected samples.

        Returns train_total_tokens / selected_tokens to maintain proper gradient magnitude.
        Returns 1.0 if no samples are selected (empty selection case).
        """
        if self.tokens_per_sample is None or self.train_total_tokens_tensor is None:
            raise RuntimeError(
                "Token counts not set. Call set_token_counts() before selection. "
                "For SeparateBatch strategies, pass 'labels' in kwargs to execute_training_step()."
            )
        # Handle empty selection to avoid division by zero
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.train_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens == 0, torch.ones_like(scale), scale)

    @abstractmethod
    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> Optional[Tuple[Tensor, int]]:
        """
        Process gradients for a single layer.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim] (sum over val samples)
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode (scalar Tensor).
                For joint batch: T_total²/(T_train × T_val) to convert to standalone scaling.
                For cached mode: None (no correction needed).

        Returns:
            For Streaming: (reduced_grad, num_selected) tuple
            For GREATS: None (scores accumulated internally)
        """
        pass

    @abstractmethod
    def get_final_selection(self) -> Tensor:
        """
        Get final selected indices.

        For Streaming: Raises NotImplementedError (selection is per-layer)
        For GREATS: Returns globally selected indices after all layers
        """
        pass


class StreamingState(SelectionState):
    """
    State for Streaming method: per-layer selection, single-pass.

    At each layer, immediately computes scores, selects samples,
    and aggregates gradients. No global accumulation needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Track last selected indices for stats
        self._last_selected_indices: Optional[Tensor] = None

        # Track selection stats across all layers
        self._layer_selections: list = []  # (layer_idx, n_selected) tuples

    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> Tuple[Tensor, int]:
        """
        Immediately select and aggregate at this layer.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim]
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode (scalar Tensor or None)

        Returns:
            (reduced_grad, num_selected) tuple
        """
        # Step 1: Compute scores (gradient alignment)
        scores = train_grads @ val_grad

        if score_correction is not None:
            scores = scores * score_correction

        # Step 2: Compute similarity if second-order
        similarity = None
        if self.use_second_order:
            similarity = train_grads @ train_grads.T
            if score_correction is not None:
                similarity = similarity * (score_correction ** 2)

        # Step 3: Select indices
        selected_indices = self._select_indices(scores, similarity)
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        self._last_selected_indices = selected_indices
        num_selected = selected_indices.shape[0]

        # Track selection for this layer
        self._layer_selections.append((layer_idx, num_selected))

        # Step 4: Aggregate selected gradients
        # Note: empty selection (num_selected=0) naturally produces zero gradients
        # since train_grads[empty_indices].sum() = zeros
        selected_grads = train_grads[selected_indices]
        reduced_grad = selected_grads.sum(dim=0, keepdim=True)

        # Step 5: Apply token-based gradient scaling
        # _compute_scale_factor handles empty selection internally (returns 1.0)
        scale_factor = self._compute_scale_factor(selected_indices)
        reduced_grad = reduced_grad * scale_factor

        self.num_selected = num_selected
        return reduced_grad, num_selected

    def get_final_selection(self) -> Tensor:
        """Streaming uses per-layer selection, not global."""
        raise NotImplementedError(
            "StreamingState uses per-layer selection. "
            "Use process_layer_gradients() at each layer instead."
        )


class GREATSState(SelectionState):
    """
    State for GREATS method: global selection, two-pass.

    Pass 1: Accumulates scores across all layers
    Pass 2: Uses global selection for gradient computation on selected samples
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Accumulators for global scoring
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
        """
        Accumulate scores - no immediate selection.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim]
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode (scalar Tensor or None)

        Returns:
            None (scores accumulated internally)
        """
        # Cast to accumulator dtype if needed
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        # Accumulate first-order scores: train_grads @ val_grad
        # Apply score_correction via the alpha parameter (convert to scalar for addmv)
        alpha = score_correction.item() if score_correction is not None else 1.0
        torch.addmv(self.grad_dot_scores, train_grads, val_grad, alpha=alpha, out=self.grad_dot_scores)

        # Accumulate similarity matrix if second-order
        if self.similarity_matrix is not None:
            # Scale similarity by score_correction² to be consistent with scores
            alpha_sq = alpha ** 2
            torch.addmm(self.similarity_matrix, train_grads, train_grads.t(), alpha=alpha_sq, out=self.similarity_matrix)

        return None

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
        score_correction: Optional[Tensor] = None,
    ) -> None:
        """
        Accumulate pre-computed scores (for full gradient path).

        This method is used when scores are computed externally (e.g., from
        factorized grad_output and input) rather than from flattened gradients.

        Args:
            scores: Pre-computed scores [train_batch_size]
            similarity: Pre-computed similarity matrix [train_batch_size, train_batch_size] or None
            score_correction: Correction factor for joint batch mode (scalar Tensor or None)
        """
        # Apply score correction
        if score_correction is not None:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)

        # Accumulate to state
        self.grad_dot_scores += scores.to(self.dtype)
        if self.similarity_matrix is not None and similarity is not None:
            self.similarity_matrix += similarity.to(self.dtype)

    def get_final_selection(self) -> Tensor:
        """
        Compute global selection after all layers processed.

        Returns:
            Tensor of selected indices
        """
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
