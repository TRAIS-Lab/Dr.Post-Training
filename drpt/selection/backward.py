"""
Autograd functions for gradient-based data selection and compression.

This module provides three distinct autograd Functions:
- CompressedLinearBackward: Pure gradient compression (no selection)
- LayerwiseLinearBackward: Per-layer selection, single-pass
- SubsetLinearBackward: Score accumulation, two-pass

Each function routes to specific handlers based on CompressionMode:
- NONE: Full gradients for scoring and updates
- SCORE_ONLY: Compressed scoring, full gradient updates
- FULL: Compressed scoring and gradient updates (MeSO)
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor
    from .state import LayerwiseState, SubsetState
    from ..hook import GradientHook
    from ..compressor import Compressor

import torch
import torch.nn.functional as F
from torch.autograd import Function

from ..compression_mode import CompressionMode
from .utils import (
    augment_input_for_bias,
    split_train_val_batch,
    compute_scores_and_similarity,
    compute_scores_direct_materialization,
    compute_selected_gradients,
)


# =============================================================================
# Helper functions for backward passes
# =============================================================================

def _get_val_components(
    hook_manager: GradientHook,
    layer_idx: int
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Get validation gradient components from hook manager.

    Returns:
        (val_grad_output, val_input, val_grad_total) tuple.
        Either (val_grad_output, val_input, None) for factorized mode,
        or (None, None, val_grad_total) for full gradient mode.
    """
    val_cache = hook_manager._val_cache

    # Try factorized mode first
    val_grad_output, val_input = val_cache.get_factorized(layer_idx)
    if val_grad_output is not None and val_input is not None:
        return val_grad_output, val_input, None

    # Fall back to full gradient mode
    val_grad_total = val_cache.get_full(layer_idx)
    return None, None, val_grad_total


def _compute_scale_factor(state: LayerwiseState, selected_indices: Tensor) -> Tensor:
    """Compute token-based gradient scale factor for selected samples."""
    return state._compute_scale_factor(selected_indices)


# =============================================================================
# Shared helpers for Layerwise backward paths
# =============================================================================

def _do_selection(
    state: "LayerwiseState",
    layer_idx: int,
    scores: Tensor,
    similarity: Optional[Tensor],
) -> Tensor:
    """Run per-layer selection: pick indices, record stats."""
    selected_indices = state._select_indices(scores, similarity)
    selected_indices = selected_indices.sort()[0]
    state._last_selected_indices = selected_indices
    state.num_selected = selected_indices.shape[0]

    if hasattr(state, '_layer_selections'):
        state._layer_selections.append((layer_idx, state.num_selected))
    if state._record_selections:
        state._selection_records.append({
            'layer_idx': layer_idx,
            'selected_indices': selected_indices.tolist(),
            'scores': scores.detach().float().cpu().tolist(),
        })
    return selected_indices


def _produce_gradient_update(
    hook_manager: "GradientHook",
    update_compressor: Optional["Compressor"],
    state: "LayerwiseState",
    layer_idx: int,
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
) -> Tuple[Optional[Tensor], Optional[Tensor]]:
    """After selection, produce the gradient update.

    If update_compressor exists: compress selected gradients → store for MeSO → return (None, None).
    Otherwise: compute full gradients for selected samples → return (grad_weight, grad_bias).
    """
    scale_factor = _compute_scale_factor(state, selected_indices)

    if update_compressor is not None:
        sel_go = train_grad_output[selected_indices]
        sel_inp = augment_input_for_bias(train_input[selected_indices], has_bias)
        update_compressed = update_compressor.forward((sel_go, sel_inp))
        reduced_grad = update_compressed.mean(dim=0, keepdim=True) * scale_factor
        hook_manager._store_compressed_grad(layer_idx, reduced_grad)
        return None, None
    else:
        grad_weight, grad_bias = compute_selected_gradients(
            train_grad_output, train_input, selected_indices, has_bias, scale_factor
        )
        return grad_weight, grad_bias


def _store_update_grad(
    hook_manager: "GradientHook",
    update_compressor: Optional["Compressor"],
    score_compressor: Optional["Compressor"],
    layer_idx: int,
    grad_output: Tensor,
    input_aug: Tensor,
    has_bias: bool,
    score_compressed_reduced: Tensor,
) -> None:
    """Store compressed gradient for MeSO when no selection is active.

    Reuses score-compressed grad if compressors are shared; otherwise re-compresses.
    """
    if update_compressor is None:
        return
    if update_compressor is score_compressor:
        hook_manager._store_compressed_grad(layer_idx, score_compressed_reduced)
    else:
        update_compressed = update_compressor.forward((grad_output, input_aug))
        hook_manager._store_compressed_grad(layer_idx, update_compressed.sum(dim=0, keepdim=True))


# =============================================================================
# Autograd Functions
# =============================================================================

class CompressedLinearBackward(Function):
    """
    Autograd Function for pure gradient compression (no data selection).

    Used when compression is enabled (MeSO optimizer) but no data selection is active.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        """Backward pass: compress gradients and store for MeSO optimizer."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        compressor = hook_manager.update_compressors[layer_idx]

        with torch.no_grad():
            input_aug = augment_input_for_bias(input, bias is not None)
            compressed_grad = compressor.forward((grad_output, input_aug))
            compressed_grad = compressed_grad.sum(dim=0, keepdim=True)
            hook_manager._store_compressed_grad(layer_idx, compressed_grad)

        return grad_input, None, None, None, None


class LayerwiseLinearBackward(Function):
    """
    Autograd Function for layer-wise descent (per-layer selection).

    Single-pass: At each layer, computes scores, selects samples,
    and aggregates gradients immediately.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], None, None]:
        """Backward pass with per-layer selection."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        score_compressor = hook_manager.score_compressors[layer_idx]
        update_compressor = hook_manager.update_compressors[layer_idx]
        state: Optional[LayerwiseState] = hook_manager.selection_state
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            getattr(state, '_use_stored_val', False)
        )

        with torch.no_grad():
            if score_compressor is not None:
                grad_weight, grad_bias = LayerwiseLinearBackward._backward_compressed(
                    hook_manager, score_compressor, update_compressor, state, layer_idx,
                    input, grad_output, bias, capture_val_mode, use_stored_val
                )
            else:
                grad_weight, grad_bias = LayerwiseLinearBackward._backward_full(
                    hook_manager, update_compressor, state, layer_idx,
                    input, bias, grad_output,
                    capture_val_mode, use_stored_val
                )

        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def _backward_compressed(
        hook_manager: "GradientHook",
        score_compressor: "Compressor",
        update_compressor: Optional["Compressor"],
        state: Optional["LayerwiseState"],
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Layerwise backward with score compression.

        Score computation uses score_compressor. Gradient updates are either:
        - Compressed via update_compressor (MeSO): returns (None, None)
        - Full gradients (no MeSO): returns (grad_weight, grad_bias)
        When update_compressor is score_compressor, reuses already-compressed grads.
        """
        has_bias = bias is not None
        input_aug = augment_input_for_bias(input, has_bias)

        # --- Compress ---

        score_compressed = score_compressor.forward((grad_output, input_aug))


        # --- Validation capture: store compressed val gradient ---
        if capture_val_mode:
            total_grad = score_compressed.sum(dim=0)
            val_cache = hook_manager.val_cache
            if val_cache._compressed[layer_idx] is None:
                val_cache._compressed[layer_idx] = total_grad
            else:
                val_cache._compressed[layer_idx] = val_cache._compressed[layer_idx] + total_grad
            return None, None

        # --- No selection state: just store MeSO update if needed ---
        if state is None:
            _store_update_grad(hook_manager, update_compressor, score_compressor,
                               layer_idx, grad_output, input_aug, has_bias,
                               score_compressed.sum(dim=0, keepdim=True))
            return None, None

        # --- Compute scores from score-compressed gradients ---

        if use_stored_val:
            train_grads = score_compressed
            val_grad = hook_manager._val_cache.get_compressed(layer_idx)
            score_correction = None
        else:
            train_grads, val_grads = split_train_val_batch(score_compressed, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)
            score_correction = state.score_correction

        if val_grad is None:
    
            _store_update_grad(hook_manager, update_compressor, score_compressor,
                               layer_idx, grad_output, input_aug, has_bias,
                               score_compressed.mean(dim=0, keepdim=True))
            return None, None

        scores = train_grads @ val_grad
        if score_correction is not None:
            scores = scores * score_correction

        similarity = None
        if state.use_second_order:
            similarity = train_grads @ train_grads.T
            if score_correction is not None:
                similarity = similarity * (score_correction ** 2)


        # --- Shared compressors: delegate to state for efficient select+reduce ---
        if update_compressor is not None and update_compressor is score_compressor:

            reduced_grad, _ = state.process_layer_gradients(
                train_grads, val_grad, layer_idx, score_correction
            )
            hook_manager._store_compressed_grad(layer_idx, reduced_grad)

            return None, None

        # --- Select ---

        selected_indices = _do_selection(state, layer_idx, scores, similarity)


        # --- w.grad ---

        if use_stored_val:
            train_grad_output, train_input = grad_output, input
        else:
            train_grad_output, _ = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, _ = split_train_val_batch(input, state.train_batch_size)

        result = _produce_gradient_update(
            hook_manager, update_compressor, state, layer_idx,
            train_grad_output, train_input, selected_indices, has_bias
        )

        return result

    @staticmethod
    def _backward_full(
        hook_manager: "GradientHook",
        update_compressor: Optional["Compressor"],
        state: Optional["LayerwiseState"],
        layer_idx: int,
        input: Tensor,
        bias: Optional[Tensor],
        grad_output: Tensor,
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Layerwise backward without score compression (full gradient scoring).

        If update_compressor is set, compresses selected gradients for MeSO.
        """
        has_bias = bias is not None

        # --- Validation capture: store full gradients ---
        if capture_val_mode:
            hook_manager.val_cache.store_layer(
                layer_idx=layer_idx,
                grad_output=grad_output.detach(),
                input=input.detach(),
                compressor=None
            )
            return None, None

        # --- No selection state: just store MeSO update if needed ---
        if state is None:
            if update_compressor is not None:
                input_aug = augment_input_for_bias(input, has_bias)
                update_compressed = update_compressor.forward((grad_output, input_aug))
                hook_manager._store_compressed_grad(layer_idx, update_compressed.sum(dim=0, keepdim=True))
            return None, None

        # --- Compute scores from full gradients ---
        if use_stored_val:
            train_grad_output, train_input = grad_output, input
            val_grad_output, val_input, val_grad_total = _get_val_components(hook_manager, layer_idx)
            if val_grad_output is None and val_grad_total is None:
                return None, None
            score_correction = None
        else:
            train_grad_output, val_grad_output = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_input = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            score_correction = state.score_correction

        scores, similarity = compute_scores_and_similarity(
            train_grad_output, train_input, val_grad_output, val_input, val_grad_total,
            state.use_second_order
        )

        if score_correction is not None:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)

        # --- Select, then produce gradient update ---
        selected_indices = _do_selection(state, layer_idx, scores, similarity)

        return _produce_gradient_update(
            hook_manager, update_compressor, state, layer_idx,
            train_grad_output, train_input, selected_indices, has_bias
        )


class SubsetLinearBackward(Function):
    """
    Autograd Function for Subset method (global selection).

    Pass 1: Accumulates scores across all layers (no gradient output).
    Pass 2: Forward/backward on selected samples only.
            Without MeSO: hooks disabled, standard autograd gradients.
            With MeSO: hooks stay enabled, CompressedLinearBackward stores
            compressed gradients for the optimizer.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: "GradientHook",
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        """Backward pass: accumulate scores only."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        grad_input = grad_output @ weight.to(grad_output.dtype)

        score_compressor = hook_manager.score_compressors[layer_idx]
        state: Optional[SubsetState] = hook_manager.selection_state

        if state is None:
            return grad_input, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        with torch.no_grad():
            if score_compressor is not None:
                SubsetLinearBackward._accumulate_compressed(
                    hook_manager, score_compressor, state, layer_idx,
                    input, grad_output, bias, use_stored_val
                )
            else:
                SubsetLinearBackward._accumulate_full(
                    hook_manager, state, layer_idx,
                    input, grad_output, use_stored_val
                )

        return grad_input, None, None, None, None

    @staticmethod
    def _accumulate_compressed(
        hook_manager: "GradientHook",
        compressor: "Compressor",
        state: "SubsetState",
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from compressed gradients."""
        input_aug = augment_input_for_bias(input, bias is not None)
        compressed_grad = compressor.forward((grad_output, input_aug))

        if use_stored_val:
            train_grads = compressed_grad
            val_grad = hook_manager._val_cache.get_compressed(layer_idx)
            # Cached mode: gradients already correctly scaled
            score_correction = None
        else:
            train_grads, val_grads = split_train_val_batch(compressed_grad, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)  # Sum, not mean, for token-weighted semantics
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction  # Tensor

        if val_grad is not None:
            state.process_layer_gradients(train_grads, val_grad, layer_idx, score_correction)

    @staticmethod
    def _accumulate_full(
        hook_manager: "GradientHook",
        state: "SubsetState",
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from full gradients."""
        if use_stored_val:
            train_grad_output, train_input = grad_output, input
            val_go, val_inp, val_grad_total = _get_val_components(hook_manager, layer_idx)
            if val_go is None and val_grad_total is None:
                return
            # Cached mode: gradients already correctly scaled, no correction needed
            score_correction = None
        else:
            # Joint batch mode: split merged batch
            train_grad_output, val_go = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_inp = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction  # Tensor

        # Route to scoring method based on state configuration
        if state.scoring_method == "direct":
            # Algorithm 4.4: Explicitly materialize per-sample gradients g_i ∈ R^{O×I}
            scores, similarity = compute_scores_direct_materialization(
                train_grad_output, train_input, val_go, val_inp, val_grad_total,
                state.use_second_order
            )
        else:
            # Default: factored scoring (avoids materializing [B, O, I])
            scores, similarity = compute_scores_and_similarity(
                train_grad_output, train_input, val_go, val_inp, val_grad_total,
                state.use_second_order
            )

        # Accumulate scores using the state method (handles correction internally)
        state.accumulate_precomputed_scores(scores, similarity, score_correction)
