"""
Curation state classes for gradient-based data curation.

This module provides three state classes:
- LayerWiseSubsetState: Per-layer curation (layer_wise_subset descent), single-pass
- GlobalSubsetState: Global curation (subset descent), two-pass score accumulation
- GroupWiseSubsetState: Per-group curation (any partition of the hooked layers,
  e.g. per transformer block or per attention/MLP sub-block), single-pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, Hashable, List, Optional, Sequence, Tuple
    from torch import Tensor

import torch

from ..utils import greedy_selection, topk_selection, negative_filtering


class SelectionState(ABC):
    """
    Abstract base class for curation state management during backward pass.

    Subclasses implement different curation strategies:
    - LayerWiseSubsetState: Per-layer curation (layer_wise_subset descent), immediate gradient aggregation
    - GlobalSubsetState: Score accumulation (subset descent), global curation after all layers
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
        selection_mode: str = "topk",
        record_selections: bool = False,
    ):
        """
        Initialize curation state.

        Args:
            train_batch_size: Number of training samples
            num_layers: Total number of layers
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select
                  - "filtering": Fraction of negative-influence samples to DROP
            lr: Learning rate for score scaling
            device: Device for tensors
            dtype: Data type for tensors
            use_second_order: If True, use greedy curation with second-order interactions
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record selected indices and scores per layer for case study
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

        # Curation recording for case study analysis
        self._record_selections = record_selections
        self._selection_records: list = []

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
        self.batch_total_tokens_tensor = batch_total_tokens.to(dtype=self.dtype)

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

        Returns batch_total_tokens / selected_tokens so the curated gradient
        matches the magnitude of a forward/backward on only the selected samples.

        Uses batch_total_tokens (not train_total_tokens) because grad_output from
        autograd is normalized by 1/batch_total_tokens. In separate-batch mode,
        batch_total == train_total, so this is equivalent.
        """
        if self.tokens_per_sample is None or self.batch_total_tokens_tensor is None:
            raise RuntimeError(
                "Token counts not set. Call set_token_counts() before curation. "
                "For SeparateBatch strategies, pass 'labels' in kwargs to execute_training_step()."
            )
        # Handle empty curation to avoid division by zero
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.batch_total_tokens_tensor / selected_tokens
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
            For GlobalSubset: None (scores accumulated internally)
        """
        pass

    @abstractmethod
    def get_final_selection(self) -> Tensor:
        """
        Get final selected indices.

        For Streaming: Raises NotImplementedError (curation is per-layer)
        For GlobalSubset: Returns globally selected indices after all layers
        """
        pass


class LayerWiseSubsetState(SelectionState):
    """
    State for layer_wise_subset descent: per-layer curation, single-pass.

    At each layer, immediately computes scores, selects samples,
    and aggregates gradients. No global accumulation needed.
    """

    def __init__(self, scoring_method: str = "reduced_ghost", direct_batch_size: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.scoring_method = scoring_method
        self.direct_batch_size = direct_batch_size
        # Track last selected indices for stats
        self._last_selected_indices: Optional[Tensor] = None

        # Track curation stats across all layers
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

        # Track curation for this layer
        self._layer_selections.append((layer_idx, num_selected))

        # Record curation data for case study analysis
        if self._record_selections:
            self._selection_records.append({
                'layer_idx': layer_idx,
                'selected_indices': selected_indices.tolist(),
                'scores': scores.detach().float().cpu().tolist(),
            })

        # Step 4: Aggregate selected gradients
        # Note: empty curation (num_selected=0) naturally produces zero gradients
        # since train_grads[empty_indices].sum() = zeros
        selected_grads = train_grads[selected_indices]
        reduced_grad = selected_grads.sum(dim=0, keepdim=True)

        # Step 5: Apply token-based gradient scaling
        # _compute_scale_factor handles empty curation internally (returns 1.0)
        scale_factor = self._compute_scale_factor(selected_indices)
        reduced_grad = reduced_grad * scale_factor

        self.num_selected = num_selected
        return reduced_grad, num_selected

    def get_final_selection(self) -> Tensor:
        """Layer-Wise Subset descent uses per-layer curation, not global."""
        raise NotImplementedError(
            "LayerWiseSubsetState uses per-layer curation. "
            "Use process_layer_gradients() at each layer instead."
        )


class GlobalSubsetState(SelectionState):
    """
    State for GlobalSubset method: global curation, two-pass.

    Pass 1: Accumulates scores across all layers
    Pass 2: Uses global curation for gradient computation on selected samples

    Args (in addition to SelectionState):
        scoring_method: "reduced_ghost" for ghost/factored inner product (default),
                        "direct" for explicit per-sample gradient materialization
                        (Algorithm 4.4 in the paper).
    """

    def __init__(self, scoring_method: str = "reduced_ghost", one_pass: bool = False, direct_batch_size: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.scoring_method = scoring_method
        self.direct_batch_size = direct_batch_size
        self.one_pass = one_pass

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
        Accumulate scores - no immediate curation.

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
        # Use tensor ops to avoid D2H sync from .item()
        layer_scores = torch.mv(train_grads, val_grad)
        if score_correction is not None:
            layer_scores = layer_scores * score_correction
        self.grad_dot_scores.add_(layer_scores)

        # Accumulate similarity matrix if second-order
        if self.similarity_matrix is not None:
            layer_sim = torch.mm(train_grads, train_grads.t())
            if score_correction is not None:
                layer_sim = layer_sim * (score_correction ** 2)
            self.similarity_matrix.add_(layer_sim)

        return None

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
        score_correction: Optional[Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        """
        Accumulate pre-computed scores (for full gradient path).

        This method is used when scores are computed externally (e.g., from
        factorized grad_output and input) rather than from flattened gradients.

        Args:
            scores: Pre-computed scores [train_batch_size]
            similarity: Pre-computed similarity matrix [train_batch_size, train_batch_size] or None
            score_correction: Correction factor for joint batch mode (scalar Tensor or None)
            layer_idx: Index of the layer these scores came from. Ignored here;
                used by GroupWiseSubsetState to route scores to the layer's group.
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

    def on_layer_processed(self, layer_idx: int, hook_manager) -> None:
        """
        Hook called by the GlobalSubset autograd Functions once a layer's scores
        have been accumulated (and, in one-pass mode, its data retained).

        No-op for global curation; GroupWiseSubsetState uses it to finalize a
        group as soon as all of its layers have run backward.
        """
        return None

    def _select_from_accumulators(
        self,
        grad_dot_scores: Tensor,
        similarity_matrix: Optional[Tensor],
    ) -> Tensor:
        """Apply lr scaling and the configured selection rule to accumulated scores."""
        scores = grad_dot_scores * self.lr

        similarity = None
        if similarity_matrix is not None:
            similarity = similarity_matrix * (self.lr ** 2)

        k = max(1, int(self.train_batch_size * self.frac))
        if self.selection_mode == "filtering":
            return negative_filtering(scores, self.frac)
        elif self.use_second_order and similarity is not None:
            return greedy_selection(scores, similarity, k)
        else:
            return topk_selection(scores, k)

    def get_final_selection(self) -> Tensor:
        """
        Compute global curation after all layers processed.

        Returns:
            Tensor of selected indices
        """
        selected_indices = self._select_from_accumulators(
            self.grad_dot_scores, self.similarity_matrix
        )

        self.num_selected = len(selected_indices)

        # Record curation data for case study analysis
        if self._record_selections:
            self._selection_records = [{
                'selected_indices': selected_indices.tolist(),
                'scores': self.grad_dot_scores.detach().float().cpu().tolist(),
            }]

        return selected_indices

    def _compute_scale_factor_for_assembly(self, selected_indices: Tensor) -> Tensor:
        """
        Compute scale factor for one-pass gradient assembly (exact parity with two-pass).

        Uses batch_total_tokens (not train_total_tokens) to correct for merged-batch
        normalization. In two-pass mode, pass 2 computes loss on selected samples only,
        giving grad = (1/selected_tokens) * raw_grad. In one-pass, grad_output is scaled
        by 1/batch_total_tokens, so we need scale = batch_total_tokens / selected_tokens.

        For SeparateBatch, batch_total_tokens == train_total_tokens, so this is equivalent
        to the standard scale factor.
        """
        if self.tokens_per_sample is None or self.batch_total_tokens_tensor is None:
            raise RuntimeError(
                "Token counts not set. Call set_token_counts() before curation."
            )
        if selected_indices.numel() == 0:
            return torch.tensor(1.0, device=self.device, dtype=self.dtype)
        selected_tokens = self.tokens_per_sample[selected_indices].sum()
        scale = self.batch_total_tokens_tensor / selected_tokens
        return torch.where(selected_tokens == 0, torch.ones_like(scale), scale)

    def reset_accumulators(self) -> None:
        """Reset accumulators for next batch."""
        self.grad_dot_scores.zero_()
        if self.similarity_matrix is not None:
            self.similarity_matrix.zero_()


class GroupWiseSubsetState(GlobalSubsetState):
    """
    State for GroupWiseSubset: curation at an arbitrary layer-group granularity.

    The hooked layers are partitioned into groups (see ``drpt.selection.grouping``).
    Scores are accumulated per group; as soon as every layer of a group has run
    backward, the group selects its samples and assembles the curated gradient
    for its layers from the retained (grad_output, input) pairs — all inside
    ``loss.backward()``. LayerWiseSubset is the singleton-group special case and
    one-pass GlobalSubset the single-group special case.

    Correctness never depends on autograd's execution order: each group is
    keyed independently and finalized on its own completion counter. Peak
    memory does depend on it — a group whose layers are far apart in backward
    order retains its activations for longer (see ``non_contiguous_groups``).

    Always one-pass (the autograd Functions return None for weight gradients;
    ``hook_manager.assemble_gradients_from_retained`` writes ``.grad``).
    """

    def __init__(self, layer_groups: "Sequence[Hashable]", **kwargs):
        kwargs["one_pass"] = True
        super().__init__(**kwargs)

        if len(layer_groups) != self.num_layers:
            raise ValueError(
                f"layer_groups has {len(layer_groups)} entries but the hook has "
                f"{self.num_layers} layers"
            )
        self.layer_groups = list(layer_groups)

        # group key -> hooked layer indices (first-appearance order)
        self.group_layers: "Dict[Hashable, List[int]]" = {}
        for idx, key in enumerate(self.layer_groups):
            self.group_layers.setdefault(key, []).append(idx)

        # Per-group accumulators (created lazily on first contribution)
        self._group_scores: "Dict[Hashable, Tensor]" = {}
        self._group_similarity: "Dict[Hashable, Tensor]" = {}

        # Bookkeeping
        self._group_done: "Dict[Hashable, set]" = {}
        self._finalized: set = set()
        self._processing_order: "List[int]" = []     # layer_idx in backward order
        self._group_selected: "Dict[Hashable, Tensor]" = {}

        # (group_key, n_selected) per finalized group — mirrors LayerWiseSubsetState
        self._layer_selections: list = []

    # ------------------------------------------------------------------ accumulate

    def _accumulators(self, key: "Hashable") -> "Tuple[Tensor, Optional[Tensor]]":
        scores = self._group_scores.get(key)
        if scores is None:
            scores = torch.zeros(self.train_batch_size, device=self.device, dtype=self.dtype)
            self._group_scores[key] = scores
        sim = None
        if self.use_second_order:
            sim = self._group_similarity.get(key)
            if sim is None:
                sim = torch.zeros(
                    self.train_batch_size, self.train_batch_size,
                    device=self.device, dtype=self.dtype,
                )
                self._group_similarity[key] = sim
        return scores, sim

    def process_layer_gradients(
        self,
        train_grads: Tensor,
        val_grad: Tensor,
        layer_idx: int,
        score_correction: Optional[Tensor] = None,
    ) -> None:
        """Accumulate compressed-gradient scores into the layer's group."""
        if train_grads.dtype != self.dtype:
            train_grads = train_grads.to(self.dtype)
        if val_grad.dtype != self.dtype:
            val_grad = val_grad.to(self.dtype)

        layer_scores = torch.mv(train_grads, val_grad)
        layer_sim = torch.mm(train_grads, train_grads.t()) if self.use_second_order else None
        self.accumulate_precomputed_scores(
            layer_scores, layer_sim, score_correction, layer_idx=layer_idx
        )
        return None

    def accumulate_precomputed_scores(
        self,
        scores: Tensor,
        similarity: Optional[Tensor],
        score_correction: Optional[Tensor] = None,
        layer_idx: Optional[int] = None,
    ) -> None:
        """Accumulate scores into the group of ``layer_idx`` (required)."""
        if layer_idx is None:
            raise ValueError("GroupWiseSubsetState.accumulate_precomputed_scores requires layer_idx")
        key = self.layer_groups[layer_idx]
        if key in self._finalized:
            raise RuntimeError(
                f"Layer {layer_idx} contributed scores to group {key!r} after the group "
                f"was finalized — a hooked layer ran backward twice in one step?"
            )

        if score_correction is not None:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)

        acc_scores, acc_sim = self._accumulators(key)
        acc_scores.add_(scores.to(self.dtype))
        if acc_sim is not None and similarity is not None:
            acc_sim.add_(similarity.to(self.dtype))

    # ------------------------------------------------------------------ finalize

    def on_layer_processed(self, layer_idx: int, hook_manager) -> None:
        """Mark ``layer_idx`` done; finalize its group once every member is done."""
        key = self.layer_groups[layer_idx]
        self._processing_order.append(layer_idx)
        done = self._group_done.setdefault(key, set())
        done.add(layer_idx)
        if key not in self._finalized and len(done) == len(self.group_layers[key]):
            self._finalize_group(key, hook_manager)

    def _finalize_group(self, key: "Hashable", hook_manager) -> None:
        """Select for one group and assemble its layers' gradients."""
        scores = self._group_scores.pop(key, None)
        similarity = self._group_similarity.pop(key, None)
        if scores is None:
            # No layer of this group produced scores (e.g. no cached val gradient).
            # Match GlobalSubset behaviour: select on all-zero scores.
            scores = torch.zeros(self.train_batch_size, device=self.device, dtype=self.dtype)

        selected_indices = self._select_from_accumulators(scores, similarity)
        selected_indices = selected_indices.sort()[0]
        n_selected = selected_indices.numel()

        self._group_selected[key] = selected_indices
        self._layer_selections.append((key, n_selected))
        self.num_selected = n_selected

        if self._record_selections:
            self._selection_records.append({
                'group': str(key),
                'layer_indices': list(self.group_layers[key]),
                'selected_indices': selected_indices.tolist(),
                'scores': scores.detach().float().cpu().tolist(),
            })

        scale_factor = self._compute_scale_factor_for_assembly(selected_indices)
        hook_manager.assemble_gradients_from_retained(
            selected_indices, scale_factor, layer_indices=self.group_layers[key]
        )
        self._finalized.add(key)

    def finalize_remaining(self, hook_manager) -> "List[Hashable]":
        """
        Finalize groups that never completed during backward (a hooked layer did
        not run, e.g. an unused or frozen layer). Returns the affected group keys
        so the caller can warn. Normally returns an empty list.
        """
        pending = [key for key in self.group_layers if key not in self._finalized]
        for key in pending:
            self._finalize_group(key, hook_manager)
        return pending

    def non_contiguous_groups(self) -> "List[Hashable]":
        """
        Group keys whose layers were *not* processed as one contiguous run in
        backward order. Such groups are still correct, but they retain their
        activations for longer and raise peak memory.
        """
        order = self._processing_order
        first_last: "Dict[Hashable, List[int]]" = {}
        for pos, layer_idx in enumerate(order):
            key = self.layer_groups[layer_idx]
            span = first_last.setdefault(key, [pos, pos])
            span[1] = pos
        bad = []
        for key, (first, last) in first_last.items():
            if last - first + 1 != len(self.group_layers[key]):
                bad.append(key)
        return bad

    def get_final_selection(self) -> Tensor:
        """GroupWiseSubset selects per group during backward, not globally."""
        raise NotImplementedError(
            "GroupWiseSubsetState selects per group inside backward; "
            "there is no single global selection."
        )

    def reset_accumulators(self) -> None:
        super().reset_accumulators()
        self._group_scores.clear()
        self._group_similarity.clear()
        self._group_done.clear()
        self._finalized.clear()
        self._processing_order.clear()
        self._group_selected.clear()
        self._layer_selections.clear()
