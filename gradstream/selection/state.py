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
        self.train_total_tokens: int = 0
        self.train_total_tokens_tensor: Optional[Tensor] = None

        # Precomputed score correction for joint batch mode (1.0 = no correction)
        self.score_correction: float = 1.0

    def set_token_counts(
        self,
        tokens_per_sample: Tensor,
        total_train_tokens: int,
        batch_total_tokens: int
    ) -> None:
        """
        Set token counts for gradient scaling and score correction.

        Args:
            tokens_per_sample: Token count per training sample [train_batch_size]
            total_train_tokens: Sum of tokens in training samples
            batch_total_tokens: Sum of tokens in entire batch (train + val for joint batch)
        """
        # Store for gradient scaling: train_total_tokens / selected_tokens
        self.tokens_per_sample = tokens_per_sample
        self.train_total_tokens = total_train_tokens
        self.train_total_tokens_tensor = torch.tensor(
            float(total_train_tokens),
            device=tokens_per_sample.device,
            dtype=torch.float32,
        )

        # Precompute score correction for joint batch mode
        val_tokens = batch_total_tokens - total_train_tokens
        if val_tokens > 0 and total_train_tokens > 0:
            self.score_correction = float(batch_total_tokens ** 2) / float(total_train_tokens * val_tokens)
        else:
            self.score_correction = 1.0

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
        # NOTE(liuxs): return tensor instead of float to avoid D2H memcpy
        if self.tokens_per_sample is None or self.train_total_tokens_tensor is None:
            raise RuntimeError(
                "Token counts not set. Call set_token_counts() before selection. "
                "For SeparateBatch strategies, pass 'labels' in kwargs to execute_training_step()."
            )
        # Handle empty selection to avoid division by zero
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=selected_indices.device, dtype=torch.float32)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.train_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens > 0, scale, torch.ones_like(scale))

    @abstractmethod
    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: float = 1.0,
    ) -> Optional[Tuple[Tensor, int]]:
        """
        Process gradients for a single layer.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim] (sum over val samples)
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode.
                For joint batch: T_total²/(T_train × T_val) to convert to standalone scaling.
                For cached mode: 1.0 (no correction needed).

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

        # Track score statistics for debugging instability
        self._score_stats: list = []  # List of per-layer score stats
        self._scale_factors: list = []  # List of scale factors applied

    def get_selection_stats(self) -> dict:
        """Get aggregated selection statistics across all layers."""
        if not self._layer_selections:
            return {"n_layers": 0, "mean": 0, "min": 0, "max": 0}

        selections = [n for _, n in self._layer_selections]

        # Separate by layer type (even=lora_A, odd=lora_B)
        lora_a_selections = [n for idx, n in self._layer_selections if idx % 2 == 0]
        lora_b_selections = [n for idx, n in self._layer_selections if idx % 2 == 1]

        stats = {
            "n_layers": len(selections),
            "mean": sum(selections) / len(selections),
            "min": min(selections),
            "max": max(selections),
        }

        if lora_a_selections:
            stats["lora_a_mean"] = sum(lora_a_selections) / len(lora_a_selections)
        if lora_b_selections:
            stats["lora_b_mean"] = sum(lora_b_selections) / len(lora_b_selections)

        return stats

    def reset_layer_stats(self):
        """Reset per-layer stats for a new mini-batch."""
        self._layer_selections = []
        self._score_stats = []
        self._scale_factors = []

    def get_score_stats(self) -> dict:
        """
        Get aggregated score statistics for debugging.

        Returns statistics about gradient alignment scores which can help
        diagnose training instability:
        - score_mean: Average alignment score (should be positive for good selection)
        - score_std: Score variance (high variance = unstable selection)
        - score_pos_frac: Fraction of positive scores (low = most samples hurt validation)
        - scale_factor_mean: Average gradient scaling applied
        - scale_factor_max: Maximum scaling (very high = potential instability)
        """
        if not self._score_stats:
            return {}

        # Aggregate across layers
        all_means = [s["mean"] for s in self._score_stats]
        all_stds = [s["std"] for s in self._score_stats]
        all_pos_fracs = [s["pos_frac"] for s in self._score_stats]
        all_mins = [s["min"] for s in self._score_stats]
        all_maxs = [s["max"] for s in self._score_stats]

        stats = {
            "score_mean": sum(all_means) / len(all_means),
            "score_std": sum(all_stds) / len(all_stds),
            "score_pos_frac": sum(all_pos_fracs) / len(all_pos_fracs),
            "score_min": min(all_mins),
            "score_max": max(all_maxs),
        }

        if self._scale_factors:
            stats["scale_factor_mean"] = sum(self._scale_factors) / len(self._scale_factors)
            stats["scale_factor_max"] = max(self._scale_factors)
            stats["scale_factor_min"] = min(self._scale_factors)

        return stats

    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: float = 1.0,
    ) -> Tuple[Tensor, int]:
        """
        Immediately select and aggregate at this layer.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim]
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode

        Returns:
            (reduced_grad, num_selected) tuple
        """
        # Step 1: Compute scores (gradient alignment)
        from .backward import _nvtx_push, _nvtx_pop
        _nvtx_push(f"layer_{layer_idx}/score_dot")
        scores = train_grads @ val_grad
        _nvtx_pop()

        if score_correction != 1.0:
            _nvtx_push(f"layer_{layer_idx}/score_scale")
            scores = scores * score_correction
            _nvtx_pop()

        # NOTE(liuxs): Score stats disabled to avoid per-layer D2H sync during profiling.
        # with torch.no_grad():
        #     self._score_stats.append({
        #         "mean": scores.mean().item(),
        #         "std": scores.std().item() if scores.numel() > 1 else 0.0,
        #         "min": scores.min().item(),
        #         "max": scores.max().item(),
        #         "pos_frac": (scores > 0).float().mean().item(),
        #     })

        # Step 2: Compute similarity if second-order
        similarity = None
        if self.use_second_order:
            _nvtx_push(f"layer_{layer_idx}/score_similarity")
            similarity = train_grads @ train_grads.T
            if score_correction != 1.0:
                similarity = similarity * (score_correction ** 2)
            _nvtx_pop()

        # Step 3: Select indices
        _nvtx_push(f"layer_{layer_idx}/select_indices")
        selected_indices = self._select_indices(scores, similarity)
        # NOTE: Sorting disabled to avoid extra syncs in profiling.
        # selected_indices = selected_indices.sort()[0]
        self._last_selected_indices = selected_indices
        num_selected = selected_indices.shape[0]
        _nvtx_pop()

        # Track selection for this layer
        self._layer_selections.append((layer_idx, num_selected))

        # Step 4: Aggregate selected gradients
        # Note: empty selection (num_selected=0) naturally produces zero gradients
        # since train_grads[empty_indices].sum() = zeros
        _nvtx_push(f"layer_{layer_idx}/reduce_grad")
        selected_grads = train_grads[selected_indices]
        reduced_grad = selected_grads.sum(dim=0, keepdim=True)
        _nvtx_pop()

        # Step 5: Apply token-based gradient scaling
        # _compute_scale_factor handles empty selection internally (returns 1.0)
        _nvtx_push(f"layer_{layer_idx}/scale_grad")
        scale_factor = self._compute_scale_factor(selected_indices)
        reduced_grad = reduced_grad * scale_factor
        _nvtx_pop()

        # Track scale factor for debugging
        self._scale_factors.append(scale_factor)

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
        score_correction: float = 1.0,
    ) -> None:
        """
        Accumulate scores - no immediate selection.

        Args:
            train_grads: Per-sample gradients [train_batch_size, feature_dim]
            val_grad: Total validation gradient [feature_dim]
            layer_idx: Index of the current layer
            score_correction: Correction factor for joint batch mode

        Returns:
            None (scores accumulated internally)
        """
        # Cast to accumulator dtype if needed
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        # Accumulate first-order scores: train_grads @ val_grad
        # Apply score_correction via the alpha parameter
        torch.addmv(self.grad_dot_scores, train_grads, val_grad, alpha=score_correction, out=self.grad_dot_scores)

        # Accumulate similarity matrix if second-order
        if self.similarity_matrix is not None:
            # Scale similarity by score_correction² to be consistent with scores
            torch.addmm(self.similarity_matrix, train_grads, train_grads.t(), alpha=score_correction**2, out=self.similarity_matrix)

        return None

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
        score_correction: float = 1.0,
    ) -> None:
        """
        Accumulate pre-computed scores (for full gradient path).

        This method is used when scores are computed externally (e.g., from
        factorized grad_output and input) rather than from flattened gradients.

        Args:
            scores: Pre-computed scores [train_batch_size]
            similarity: Pre-computed similarity matrix [train_batch_size, train_batch_size] or None
            score_correction: Correction factor for joint batch mode
        """
        # Apply score correction
        if score_correction != 1.0:
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
