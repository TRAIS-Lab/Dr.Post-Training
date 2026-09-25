"""
Curation strategies for gradient-based data curation in trainers.

This module provides two families of strategies based on how validation
gradients are obtained (each with NA / LayerWiseSubset / GlobalSubset /
GroupWiseSubset variants; GroupWiseSubset curates per layer group, e.g. per
transformer block — see drpt.selection.grouping):

1. **MergedBatch Strategies**:
   - Train and val samples are merged into a single batch
   - Val gradients computed during the same forward/backward pass
   - Factory: create_merged_batch_strategy()
   - Note: Has padding overhead when val/train have different sequence lengths

2. **SeparateBatch Strategies**:
   - Val gradients are pre-captured and cached before training
   - Training uses cached val gradients for curation scoring
   - Factory: create_separate_batch_strategy()
   - Val storage mode is derived from scoring_method in start_val_capture():
       * pip/direct: Stores total gradient [O, I] per layer.
       Better when validation batch is large (e.g., self-reference validation in RLHF).
       * gip: Stores [V, S, O] and [V, S, I] components (for pairwise scoring).
       More memory-efficient during training as it avoids materializing [B_train, O, I].
       Better when validation batch is small (e.g., external validation set in SFT).

Loss convention: ``compute_loss_fn`` must follow ``grad_hook.loss_reduction``
(see ``drpt.losses``). The strategies hand the batch labels to
``grad_hook.set_token_counts`` so the per-sample item counts (ones under
``"sample_mean"``, token counts under ``"token_mean"``) match the loss, which
makes the score correction and the curated-update scale exact.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict, Optional, Callable, Tuple
    from ..hook import GradientHook
    from torch import Tensor

import torch
import torch.nn as nn
from .state import LayerWiseSubsetState, GlobalSubsetState, GroupWiseSubsetState

logger = logging.getLogger(__name__)

# Curation methods that need gradient hooks (shared by the SFT/RLHF trainers).
SELECTION_METHODS = ("LayerWiseSubset", "GlobalSubset", "GroupWiseSubset")


def _finish_groupwise_backward(strategy, grad_hook: "GradientHook") -> None:
    """Post-backward bookkeeping shared by the GroupWiseSubset strategies.

    Every group normally finalizes (select + assemble .grad) inside backward.
    Groups whose layers did not all run backward are finalized here with a
    warning; on the first step we also warn about groups that are not
    contiguous in backward order (correct, but higher peak memory).
    """
    state: GroupWiseSubsetState = grad_hook.selection_state
    pending = state.finalize_remaining(grad_hook)
    if pending and not getattr(strategy, "_warned_pending", False):
        strategy._warned_pending = True
        logger.warning(
            f"GroupWiseSubset: {len(pending)} group(s) had layers that did not run backward "
            f"and were finalized after loss.backward(): {[str(k) for k in pending[:5]]}"
            f"{' ...' if len(pending) > 5 else ''}. Check that every hooked layer is used "
            f"in the forward pass and trainable."
        )
    if not getattr(strategy, "_contiguity_checked", False):
        strategy._contiguity_checked = True
        bad = state.non_contiguous_groups()
        if bad:
            logger.warning(
                f"GroupWiseSubset: {len(bad)} group(s) are not contiguous in backward order "
                f"(e.g. {[str(k) for k in bad[:3]]}). Selection is still exact, but their "
                f"activations stay retained until the group completes, raising peak memory."
            )
        else:
            logger.info(
                f"GroupWiseSubset: all {len(state.group_layers)} groups are contiguous in "
                f"backward order (retained activations are released per group)."
            )


# ============================================================
# JOINT BATCH STRATEGIES
# Val gradients computed from merged train+val batch
# ============================================================

class MergedBatchStrategy(ABC):
    """
    Abstract strategy for merged-batch data curation.

    Used when train and val samples are merged into a single batch,
    and val gradients are computed during the same forward/backward pass.

    Note: Has padding overhead when val/train have different sequence lengths.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
        frac: float,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "pip",
        score_normalization: str = "none",
    ):
        """
        Initialize curation strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Curation fraction / filter fraction
            use_second_order: Use greedy curation with second-order
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record curation data for case study analysis
            scoring_method: Scoring method for influence scores ("pip", "gip", "direct")
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode
        self.record_selections = record_selections
        self.scoring_method = scoring_method
        self.last_selection_record = None
        # Random Subset control: one CPU generator per strategy, seeded from the run seed
        # (torch.initial_seed() reflects set_seed), so draws are reproducible per seed and
        # leave the global RNG (data order, dropout) untouched.
        self.score_normalization = score_normalization
        self.random_generator = None
        if selection_mode == "random":
            self.random_generator = torch.Generator().manual_seed(int(torch.initial_seed()) % (2**63 - 1))

    @property
    def has_update_compression(self) -> bool:
        """Check if update compression (MeSO) is enabled.

        When True, hooks stay enabled during GlobalSubset pass 2 so that
        CompressedLinearBackward stores compressed gradients for MeSO.
        """
        if self.grad_hook is None:
            return False
        return self.grad_hook.compression_mode.uses_compressed_updates

    def _extract_selection_records(self):
        """Extract curation records from state before cleanup."""
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            self.last_selection_record = None
            return
        state = self.grad_hook.selection_state
        if state._record_selections and state._selection_records:
            self.last_selection_record = list(state._selection_records)
        else:
            self.last_selection_record = None

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
        Execute a complete training step with curation.

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


class MergedBatchNoSelectionStrategy(MergedBatchStrategy):
    """
    Baseline strategy: no data curation, standard training.

    Uses all training samples without curation.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Standard training step without curation."""
        # Disable hooks if present (use standard gradient computation)
        if self.grad_hook is not None and not self.has_update_compression:
            self.grad_hook.disable_hooks()

        # Extract train-only portion (no need for val in baseline)
        train_batch = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        model.zero_grad()
        loss = compute_loss_fn(model, train_batch)
        loss.backward()

        # Re-enable hooks
        if self.grad_hook is not None and not self.has_update_compression:
            self.grad_hook.enable_hooks()

        return loss.detach()


class MergedBatchLayerWiseSubsetStrategy(MergedBatchStrategy):
    """
    Layer-Wise Subset strategy with merged batch: single-pass, per-layer curation.

    Curation and gradient aggregation happen layer-by-layer
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
        """Training step with per-layer curation."""
        lr = kwargs.get('lr', 1e-4)

        # Set up streaming state
        self._setup_state(train_batch_size, lr)

        # Set item counts for gradient scaling and the merged-batch score correction
        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()  # Per-layer curation happens in backward hooks

        # Extract curation records before cleanup
        self._extract_selection_records()

        # Cleanup
        self._cleanup()

        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up LayerWiseSubsetState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = LayerWiseSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class MergedBatchGlobalSubsetStrategy(MergedBatchStrategy):
    """
    GlobalSubset strategy with merged batch: two-pass, global curation.

    Pass 1: Compute curation scores across all layers
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
        """Training step with global curation."""
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
        state: GlobalSubsetState = self.grad_hook.selection_state
        selected_indices = state.get_final_selection()
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        self._cleanup()

        # Handle empty curation: skip pass 2 and return zero loss
        if n_selected == 0:
            # Re-enable hooks for next step
            if not self.has_update_compression:
                self.grad_hook.enable_hooks()
            else:
                self.grad_hook.clear_token_counts()

            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            return zero_loss

        # === PASS 2: Gradient Computation on Selected ===
        if batch_train is None:
            # Fall back to extracting from merged batch
            batch_train = {k: v[:train_batch_size] for k, v in merged_batch.items()}

        filtered_inputs = {
            'input_ids': batch_train['input_ids'][selected_indices],
            'attention_mask': batch_train['attention_mask'][selected_indices],
            'labels': batch_train['labels'][selected_indices]
        }

        # For pass 2, disable hooks if no compression (we want full gradients for selected samples)
        if not self.has_update_compression:
            self.grad_hook.disable_hooks()
        else:
            # For MeSO, set item counts for selected batch
            self.grad_hook.set_token_counts(filtered_inputs['labels'])

        model.zero_grad()
        loss = compute_loss_fn(model, filtered_inputs)
        loss.backward()

        # Re-enable hooks / clear token counts
        if not self.has_update_compression:
            self.grad_hook.enable_hooks()
        else:
            self.grad_hook.clear_token_counts()

        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = GlobalSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class MergedBatchGlobalSubsetOnePassStrategy(MergedBatchStrategy):
    """
    One-pass subset strategy with merged batch (Algorithm 4.2).

    Single forward+backward pass: scoring and data retention happen during backward,
    then post-hoc gradient assembly from retained (grad_output, input) per layer.
    Saves the second forward+backward at the cost of higher peak memory.

    Non-linear layers (RMSNorm) are wrapped with TrainOnlyRMSNormBackward so
    their grad_weight is computed from the train slice only, preventing
    validation gradient leakage.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with one-pass global curation."""
        lr = kwargs.get('lr', 1e-4)

        # Set up GlobalSubsetState with one_pass=True
        self._setup_state(train_batch_size, lr)

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()
        # After backward:
        # - Scores accumulated in state, layer data retained in hook
        # - Hooked linear layers: .grad is None (GlobalSubsetLinearBackward returns None)
        # - RMSNorm layers: .grad contains train-only gradient (TrainOnlyRMSNormBackward)

        # Get globally selected indices
        state: GlobalSubsetState = self.grad_hook.selection_state
        selected_indices = state.get_final_selection()
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        if n_selected == 0:
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # Compute scale factor for exact parity with two-pass
        scale_factor = state._compute_scale_factor_for_assembly(selected_indices)

        # Post-hoc gradient assembly for linear layers.
        # No model.zero_grad() here — non-linear layers retain their train-only
        # gradients from backward, and hooked linear layers have None grad
        # (GlobalSubsetLinearBackward suppresses weight/bias grads).
        self.grad_hook.assemble_gradients_from_retained(selected_indices, scale_factor)

        self._cleanup()
        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState with one_pass=True."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = GlobalSubsetState(
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()
        self.grad_hook.clear_retained_data()  # Safety net


class MergedBatchGroupWiseSubsetStrategy(MergedBatchStrategy):
    """
    GroupWiseSubset strategy with merged batch: single-pass, per-group curation.

    The hooked layers are partitioned into groups (per transformer block, per
    attention/MLP sub-block, ...; see ``drpt.selection.grouping``). During
    backward each group accumulates scores over its layers and, once all of
    them have run, selects and assembles the curated gradient for exactly
    those layers. Non-linear layers (RMSNorm) are wrapped with
    TrainOnlyRMSNormBackward as in one-pass GlobalSubset.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        merged_batch: Dict[str, Tensor],
        train_batch_size: int,
        compute_loss_fn: Callable[[nn.Module, Dict[str, Tensor]], Tensor],
        **kwargs
    ) -> Tensor:
        """Training step with per-group curation."""
        lr = kwargs.get('lr', 1e-4)

        self._setup_state(train_batch_size, lr)

        if 'labels' in merged_batch:
            self.grad_hook.set_token_counts(merged_batch['labels'], train_batch_size)

        model.zero_grad()
        loss = compute_loss_fn(model, merged_batch)
        loss.backward()  # Groups select + assemble .grad as they complete

        _finish_groupwise_backward(self, self.grad_hook)

        self._extract_selection_records()
        self._cleanup()
        return loss.detach()

    def _setup_state(self, train_batch_size: int, lr: float) -> None:
        """Set up GroupWiseSubsetState for this step."""
        dtype = next(self.grad_hook.model.parameters()).dtype

        state = GroupWiseSubsetState(
            layer_groups=self.grad_hook._require_layer_groups(),
            train_batch_size=train_batch_size,
            num_layers=len(self.grad_hook.layer_names),
            frac=self.frac,
            lr=lr,
            device=self.grad_hook.device,
            dtype=dtype,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
        )

        self.grad_hook.selection_state = state

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()
        self.grad_hook.clear_retained_data()  # Safety net


def create_merged_batch_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
    record_selections: bool = False,
    scoring_method: str = "pip",
    subset_mode: str = "one_pass",
    score_normalization: str = "none",
) -> MergedBatchStrategy:
    """
    Factory function to create merged-batch curation strategy.

    Note: Has padding overhead when val/train have different sequence lengths.

    Args:
        method: Curation method ("NA", "LayerWiseSubset", "GlobalSubset", "GroupWiseSubset")
        grad_hook: GradientHook instance (GroupWiseSubset: with layer_groups set)
        frac: Curation/filter fraction
        use_second_order: Use greedy curation with second-order
        selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        record_selections: If True, record curation data for case study analysis
        scoring_method: Scoring method ("pip", "gip", "direct", "compress")
        subset_mode: For GlobalSubset method: "one_pass" (Algorithm 4.2) or "two_pass" (Algorithm 4.3)

    Returns:
        Appropriate MergedBatchStrategy instance
    """
    kwargs = dict(grad_hook=grad_hook, frac=frac, use_second_order=use_second_order,
                  selection_mode=selection_mode, record_selections=record_selections,
                  scoring_method=scoring_method, score_normalization=score_normalization)

    if method == "NA":
        return MergedBatchNoSelectionStrategy(**kwargs)

    if method == "LayerWiseSubset":
        strategy = MergedBatchLayerWiseSubsetStrategy(**kwargs)
        # Wrap non-linear layers so their grad_weight comes from train slice only.
        grad_hook.wrap_nonlinear_layers()
        return strategy

    if method == "GlobalSubset":
        if subset_mode == "one_pass":
            strategy = MergedBatchGlobalSubsetOnePassStrategy(**kwargs)
            grad_hook.wrap_nonlinear_layers()
            return strategy
        else:
            return MergedBatchGlobalSubsetStrategy(**kwargs)

    if method == "GroupWiseSubset":
        grad_hook._require_layer_groups()
        strategy = MergedBatchGroupWiseSubsetStrategy(**kwargs)
        grad_hook.wrap_nonlinear_layers()
        return strategy

    raise ValueError(f"Unknown curation method: {method}")




# ============================================================
# CACHED VAL STRATEGIES
# Val gradients pre-captured and cached before training
# Avoids padding overhead when val/train have different seq lengths
# ============================================================

class SeparateBatchStrategy(ABC):
    """
    Abstract strategy for separate-batch data curation.

    Used when val gradients are pre-captured and cached before training,
    rather than computed from a merged batch during the same forward pass.

    Val storage mode is derived from scoring_method in start_val_capture():
    - pip/direct: Stores total gradient [O, I] per layer.
      Better when validation batch is large (e.g., self-reference validation in RLHF).
    - gip: Stores [V, S, O] and [V, S, I] components (for pairwise scoring).
      More memory-efficient during training. Better when validation batch is small.
    """

    def __init__(
        self,
        grad_hook: Optional[GradientHook],
        frac: float,
        use_second_order: bool = False,
        selection_mode: str = "topk",
        record_selections: bool = False,
        scoring_method: str = "pip",
        score_normalization: str = "none",
    ):
        """
        Initialize stored-val curation strategy.

        Args:
            grad_hook: GradientHook instance (can be None for NoSelection)
            frac: Fraction parameter. Meaning depends on selection_mode:
                  - "topk": Fraction of samples to select (top frac by score)
                  - "filtering": Fraction of negative-influence samples to DROP
            use_second_order: Use greedy curation with second-order
            selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
            record_selections: If True, record curation data for case study analysis
            scoring_method: Scoring method ("pip", "gip", "direct", "compress")
        """
        self.grad_hook = grad_hook
        self.frac = frac
        self.use_second_order = use_second_order
        self.selection_mode = selection_mode
        self.record_selections = record_selections
        self.scoring_method = scoring_method
        self.last_selection_record = None
        # Random Subset control: one CPU generator per strategy, seeded from the run seed
        # (torch.initial_seed() reflects set_seed), so draws are reproducible per seed and
        # leave the global RNG (data order, dropout) untouched.
        self.score_normalization = score_normalization
        self.random_generator = None
        if selection_mode == "random":
            self.random_generator = torch.Generator().manual_seed(int(torch.initial_seed()) % (2**63 - 1))

    @property
    def has_update_compression(self) -> bool:
        """Check if update compression (MeSO) is enabled.

        When True, hooks stay enabled during GlobalSubset pass 2 so that
        CompressedLinearBackward stores compressed gradients for MeSO.
        """
        if self.grad_hook is None:
            return False
        return self.grad_hook.compression_mode.uses_compressed_updates

    def _extract_selection_records(self):
        """Extract curation records from state before cleanup."""
        if self.grad_hook is None or self.grad_hook.selection_state is None:
            self.last_selection_record = None
            return
        state = self.grad_hook.selection_state
        if state._record_selections and state._selection_records:
            self.last_selection_record = list(state._selection_records)
        else:
            self.last_selection_record = None

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
        Execute a complete training step with curation.

        Args:
            model: The model to train
            batch_size: Number of samples in the batch
            compute_loss_fn: Zero-arg function that computes loss and returns (loss, stats)
            lr: Learning rate for score scaling
            **kwargs: Additional arguments (filter_batch_fn for GlobalSubset)

        Returns:
            Tuple of (loss, stats_dict) where stats includes curation metrics
        """
        pass


class SeparateBatchNoSelectionStrategy(SeparateBatchStrategy):
    """
    Baseline strategy: no data curation, standard training.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Standard training step without curation."""
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


class SeparateBatchLayerWiseSubsetStrategy(SeparateBatchStrategy):
    """
    Layer-Wise Subset strategy with cached val: per-layer curation.

    Curation and gradient aggregation happen layer-by-layer during backward,
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
        """Training step with per-layer curation using stored val grads.

        Args:
            model: The model to train
            batch_size: Number of samples in the batch
            compute_loss_fn: Zero-arg function that computes loss and returns (loss, stats)
            lr: Learning rate for score scaling
            **kwargs:
                labels: Label tensor [batch_size, seq_len] (-100 = ignored) used to
                        derive per-sample item counts for the curated-update scale.
        """
        # Set up streaming state with stored validation gradients
        self._setup_state(batch_size, lr)

        # Set item counts for gradient scaling (if labels provided)
        # In SeparateBatch mode, entire batch is train, so pass batch_size as train_batch_size
        labels = kwargs.get('labels')
        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()  # Per-layer curation happens in backward hooks

        # Add curation stats from streaming state
        sel_state = self.grad_hook.selection_state
        if hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
            n_selected_list = [n for _, n in sel_state._layer_selections]
            stats["selection/mean_selected"] = sum(n_selected_list) / len(n_selected_list)
            stats["selection/min_selected"] = min(n_selected_list)
            # Use min for n_selected check - if any layer had 0, flag it
            stats["selection/n_selected"] = min(n_selected_list)

        # Extract curation records before cleanup
        self._extract_selection_records()

        # Cleanup
        self._cleanup()

        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up LayerWiseSubsetState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="LayerWiseSubset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=False,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()


class SeparateBatchGlobalSubsetStrategy(SeparateBatchStrategy):
    """
    GlobalSubset strategy with cached val: global curation.

    Pass 1: Compute curation scores across all layers
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
        """Training step with global curation using stored val grads."""
        # filter_batch_fn: Callable[[Tensor], Callable] that takes selected_indices
        # and returns a new compute_loss_fn for the filtered batch
        filter_batch_fn = kwargs.get('filter_batch_fn')
        if filter_batch_fn is None:
            raise ValueError("GlobalSubset strategy requires 'filter_batch_fn' in kwargs")

        # === PASS 1: Score Accumulation ===
        self._setup_state(batch_size, lr)

        # Item counts (not needed for pass-1 scoring in cached-val mode, where no
        # correction applies; kept uniform with the one-pass strategies so the state
        # is fully populated for stats / MeSO pass 2)
        labels = kwargs.get('labels')
        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss_for_scoring, _ = compute_loss_fn()
        loss_for_scoring.backward()

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        # Sort indices for sequential memory access (better cache locality)
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        self._cleanup()

        # Handle empty curation: skip pass 2 and return zero loss
        if n_selected == 0:
            # Re-enable hooks for next step
            self.grad_hook.enable_hooks()

            # Pass 1 left .grad on parameters that are not routed through the
            # curation hooks (e.g. the PPO value head); drop them so the caller's
            # optimizer.step() does not apply an unfiltered update.
            model.zero_grad()

            # Return zero loss and stats indicating batch was skipped
            # (keys match compute_ppo_loss / _compute_ppo_stats for aggregation)
            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            stats = {
                "loss/total": 0.0,
                "loss/policy": 0.0,
                "loss/value": 0.0,
                "policy/approxkl": 0.0,
                "policy/policykl": 0.0,
                "policy/clipfrac": 0.0,
                "policy/ratio": 1.0,
                "val/mean": 0.0,
                "selection/n_selected": 0,
            }
            return zero_loss, stats

        # === PASS 2: Gradient Computation on Selected ===
        # The loss on the selected batch (same convention as pass 1) already is the
        # update we want: under "sample_mean" the mean over the k selected samples.
        # Disable hooks only if no update compression (standard optimizer for selected samples).
        # With MeSO, keep hooks enabled so CompressedLinearBackward stores
        # compressed gradients for the optimizer.
        if not self.has_update_compression:
            self.grad_hook.disable_hooks()

        model.zero_grad()

        # Get filtered compute_loss_fn for selected samples
        filtered_compute_loss_fn = filter_batch_fn(selected_indices)
        loss, stats = filtered_compute_loss_fn()
        loss.backward()

        # Re-enable hooks if we disabled them
        if not self.has_update_compression:
            self.grad_hook.enable_hooks()

        # Add curation stats
        stats["selection/n_selected"] = n_selected

        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="GlobalSubset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,  # Only accumulate scores in pass 1
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()


class SeparateBatchGlobalSubsetOnePassStrategy(SeparateBatchStrategy):
    """
    One-pass subset strategy with separate val batch (Algorithm 4.2).

    Recommended one-pass mode: exact scale factor parity with two-pass since
    batch_total == train_total (item counts) in separate batch mode.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with one-pass global curation using stored val grads."""
        labels = kwargs.get('labels')

        # Set up GlobalSubsetState with one_pass=True and stored val
        self._setup_state(batch_size, lr)

        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()
        # After backward: scores accumulated, layer data retained

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        selected_indices = selected_indices.sort()[0]
        n_selected = len(selected_indices)

        # Extract curation records before cleanup
        self._extract_selection_records()

        if n_selected == 0:
            self.grad_hook.clear_retained_data()
            self._cleanup()
            import torch
            zero_loss = torch.tensor(0.0, device=next(model.parameters()).device)
            stats["selection/n_selected"] = 0
            return zero_loss, stats

        # Compute scale factor for exact parity with two-pass
        state = self.grad_hook.selection_state
        scale_factor = state._compute_scale_factor_for_assembly(selected_indices)

        # Post-hoc gradient assembly for linear layers.
        # Non-linear layers (LayerNorm, embeddings, etc.) retain their autograd
        # gradients from backward — GlobalSubsetLinearBackward returns None for
        # weight/bias, so hooked linear layers have no stale grad to clear.
        self.grad_hook.assemble_gradients_from_retained(selected_indices, scale_factor)

        stats["selection/n_selected"] = n_selected
        self._cleanup()
        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up GlobalSubsetState with one_pass=True and stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="GlobalSubset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_retained_data()  # Safety net


class SeparateBatchGroupWiseSubsetStrategy(SeparateBatchStrategy):
    """
    GroupWiseSubset strategy with cached val: single-pass, per-group curation.

    Uses pre-captured validation gradients. During backward each group
    accumulates scores over its layers and, once all of them have run, selects
    and assembles the curated gradient for exactly those layers. Exact scale
    factor parity with LayerWiseSubset/GlobalSubset since
    batch_total == train_total (item counts) in separate batch mode.
    """

    def execute_training_step(
        self,
        model: nn.Module,
        batch_size: int,
        compute_loss_fn: Callable[[], Tuple[Tensor, Dict]],
        lr: float,
        **kwargs
    ) -> Tuple[Tensor, Dict]:
        """Training step with per-group curation using stored val grads."""
        labels = kwargs.get('labels')

        self._setup_state(batch_size, lr)

        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

        model.zero_grad()
        loss, stats = compute_loss_fn()
        loss.backward()  # Groups select + assemble .grad as they complete

        _finish_groupwise_backward(self, self.grad_hook)

        # Curation stats (same keys as LayerWiseSubset)
        sel_state = self.grad_hook.selection_state
        if sel_state._layer_selections:
            n_selected_list = [n for _, n in sel_state._layer_selections]
            stats["selection/mean_selected"] = sum(n_selected_list) / len(n_selected_list)
            stats["selection/min_selected"] = min(n_selected_list)
            stats["selection/n_selected"] = min(n_selected_list)

        self._extract_selection_records()
        self._cleanup()
        return loss.detach(), stats

    def _setup_state(self, batch_size: int, lr: float) -> None:
        """Set up GroupWiseSubsetState with stored val gradients."""
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method="GroupWiseSubset",
            frac=self.frac,
            lr=lr,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode=self.selection_mode,
            record_selections=self.record_selections,
            random_generator=self.random_generator,
            score_normalization=self.score_normalization,
            scoring_method=self.scoring_method,
            one_pass=True,
        )

    def _cleanup(self) -> None:
        """Clean up after training step."""
        self.grad_hook.clear_selection()
        self.grad_hook.clear_retained_data()  # Safety net


def create_separate_batch_strategy(
    method: str,
    grad_hook: Optional[GradientHook],
    frac: float = 0.5,
    use_second_order: bool = False,
    selection_mode: str = "topk",
    record_selections: bool = False,
    scoring_method: str = "pip",
    subset_mode: str = "one_pass",
    score_normalization: str = "none",
) -> SeparateBatchStrategy:
    """
    Factory function to create separate-batch curation strategy.

    Avoids padding overhead when val/train have different sequence lengths.

    Args:
        method: Curation method ("NA", "LayerWiseSubset", "GlobalSubset", "GroupWiseSubset")
        grad_hook: GradientHook instance (GroupWiseSubset: with layer_groups set)
        frac: Fraction parameter. Meaning depends on selection_mode:
              - "topk": Fraction of samples to select (top frac by score)
              - "filtering": Fraction of negative-influence samples to DROP
        use_second_order: Use greedy curation with second-order
        selection_mode: "topk" (select top frac) or "filtering" (drop bottom frac of negative)
        record_selections: If True, record curation data for case study analysis
        scoring_method: Scoring method ("pip", "gip", "direct", "compress")
        subset_mode: For GlobalSubset method: "one_pass" (Algorithm 4.2) or "two_pass" (Algorithm 4.3)

    Returns:
        Appropriate SeparateBatchStrategy instance
    """
    kwargs = dict(grad_hook=grad_hook, frac=frac, use_second_order=use_second_order,
                  selection_mode=selection_mode, record_selections=record_selections,
                  scoring_method=scoring_method, score_normalization=score_normalization)

    if method == "NA":
        return SeparateBatchNoSelectionStrategy(**kwargs)

    if method == "LayerWiseSubset":
        return SeparateBatchLayerWiseSubsetStrategy(**kwargs)

    if method == "GlobalSubset":
        if subset_mode == "one_pass":
            strategy = SeparateBatchGlobalSubsetOnePassStrategy(**kwargs)
            # Safety check: warn about trainable params not covered by hooks.
            # No non-linear wrapping needed (no val contamination in separate batch),
            # but unhooked params get full-batch train grad instead of curated.
            grad_hook.check_unhooked_trainable_params()
            return strategy
        else:
            return SeparateBatchGlobalSubsetStrategy(**kwargs)

    if method == "GroupWiseSubset":
        grad_hook._require_layer_groups()
        strategy = SeparateBatchGroupWiseSubsetStrategy(**kwargs)
        grad_hook.check_unhooked_trainable_params()
        return strategy

    raise ValueError(f"Unknown curation method: {method}")
