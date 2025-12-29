"""
Autograd functions for gradient-based data selection and compression.

This module provides three distinct autograd Functions:
- CompressedLinearBackward: Pure gradient compression (no selection)
- StreamingLinearBackward: Per-layer selection, single-pass
- GREATSLinearBackward: Score accumulation, two-pass
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor
    from .state import StreamingState, GREATSState
    from ..hook import GradientHook
    from ..compressor import Compressor

import torch
import torch.nn.functional as F
from torch.autograd import Function

from .utils import (
    augment_input_for_bias,
    split_train_val_batch,
    compute_total_gradient,
    compute_scores_and_similarity,
    compute_selected_gradients,
)


# =============================================================================
# Helper functions for backward passes
# =============================================================================

def _get_val_components(
    hook_manager: "GradientHook",
    layer_idx: int
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    """
    Get validation gradient components from hook manager.

    Returns:
        (val_grad_output, val_input, val_grad_total) tuple.
        Either (val_grad_output, val_input, None) for factorized mode,
        or (None, None, val_grad_total) for full gradient mode.
    """
    val_grad_output = hook_manager.val_grad_output_buffer[layer_idx]
    val_input = hook_manager.val_input_buffer[layer_idx]

    if val_grad_output is not None and val_input is not None:
        return val_grad_output, val_input, None

    val_grad_total = hook_manager.val_grad_buffer[layer_idx]
    return None, None, val_grad_total


def _compute_scale_factor(state: "StreamingState", selected_indices: "Tensor") -> float:
    """Compute token-based gradient scale factor for selected samples."""
    return state._compute_scale_factor(selected_indices)


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

        compressor = hook_manager.compressors[layer_idx]

        with torch.no_grad():
            input_aug = augment_input_for_bias(input, bias is not None)
            compressed_grad = compressor.forward((grad_output, input_aug))
            compressed_grad = compressed_grad.sum(dim=0, keepdim=True)
            hook_manager._store_compressed_grad(layer_idx, compressed_grad)

        return grad_input, None, None, None, None


class StreamingLinearBackward(Function):
    """
    Autograd Function for Streaming method (per-layer selection).

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

        compressor = hook_manager.compressors[layer_idx]
        state: Optional[StreamingState] = hook_manager.selection_state
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            getattr(state, '_use_stored_val', False)
        )

        with torch.no_grad():
            if compressor is not None:
                grad_weight, grad_bias = StreamingLinearBackward._backward_compressed(
                    hook_manager, compressor, state, layer_idx,
                    input, grad_output, bias, capture_val_mode, use_stored_val
                )
            else:
                grad_weight, grad_bias = StreamingLinearBackward._backward_full(
                    hook_manager, state, layer_idx,
                    input, bias, grad_output,
                    capture_val_mode, use_stored_val
                )

        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def _backward_compressed(
        hook_manager: "GradientHook",
        compressor: "Compressor",
        state: Optional["StreamingState"],
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[None, None]:
        """Handle compressed gradient path."""
        input_aug = augment_input_for_bias(input, bias is not None)
        compressed_grad = compressor.forward((grad_output, input_aug))

        if capture_val_mode:
            # Val capture: accumulate total compressed gradient (sum, not mean)
            total_grad = compressed_grad.sum(dim=0)
            if hook_manager.val_grad_buffer[layer_idx] is None:
                hook_manager.val_grad_buffer[layer_idx] = total_grad
            else:
                hook_manager.val_grad_buffer[layer_idx] += total_grad
            return None, None

        if state is None:
            # No selection: sum and store
            hook_manager._store_compressed_grad(layer_idx, compressed_grad.sum(dim=0, keepdim=True))
            return None, None

        # Get train/val gradients
        if use_stored_val:
            train_grads = compressed_grad
            val_grad = hook_manager.val_grad_buffer[layer_idx]
            # Cached mode: gradients already correctly scaled
            score_correction = 1.0
        else:
            train_grads, val_grads = split_train_val_batch(compressed_grad, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)  # Sum, not mean, for token-weighted semantics
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction

        if val_grad is None:
            hook_manager._store_compressed_grad(layer_idx, compressed_grad.mean(dim=0, keepdim=True))
            return None, None

        # Per-layer selection and reduction (with score correction for joint batch)
        reduced_grad, _ = state.process_layer_gradients(
            train_grads, val_grad, layer_idx, score_correction
        )
        hook_manager._store_compressed_grad(layer_idx, reduced_grad)
        return None, None

    @staticmethod
    def _backward_full(
        hook_manager: "GradientHook",
        state: Optional["StreamingState"],
        layer_idx: int,
        input: Tensor,
        bias: Optional[Tensor],
        grad_output: Tensor,
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Handle full gradient path."""
        if capture_val_mode:
            # Val capture: store gradients for later use
            if hook_manager.use_factorized_val:
                hook_manager.val_grad_output_buffer[layer_idx] = grad_output.detach()
                hook_manager.val_input_buffer[layer_idx] = input.detach()
            else:
                val_grad_total = compute_total_gradient(grad_output, input)
                if hook_manager.val_grad_buffer[layer_idx] is None:
                    hook_manager.val_grad_buffer[layer_idx] = val_grad_total
                else:
                    hook_manager.val_grad_buffer[layer_idx] += val_grad_total
            return None, None

        if state is None:
            return None, None

        # Get train/val components and compute scores
        if use_stored_val:
            train_grad_output, train_input = grad_output, input
            val_grad_output, val_input, val_grad_total = _get_val_components(hook_manager, layer_idx)
            if val_grad_output is None and val_grad_total is None:
                return None, None
            # Cached mode: gradients already correctly scaled, no correction needed
            score_correction = 1.0
        else:
            # Joint batch mode: split merged batch
            train_grad_output, val_grad_output = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_input = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction

        scores, similarity = compute_scores_and_similarity(
            train_grad_output, train_input, val_grad_output, val_input, val_grad_total,
            state.use_second_order
        )

        # Apply correction for joint batch mode
        if score_correction != 1.0:
            scores = scores * score_correction
            if similarity is not None:
                similarity = similarity * (score_correction ** 2)

        # Per-layer selection
        selected_indices = state._select_indices(scores, similarity)
        state.num_selected = selected_indices.shape[0]

        # Compute gradients for selected samples
        # Note: empty selection naturally produces zero gradients via einsum on empty tensors
        # _compute_scale_factor handles empty selection internally (returns 1.0)
        scale_factor = _compute_scale_factor(state, selected_indices)
        grad_weight, grad_bias = compute_selected_gradients(
            train_grad_output, train_input, selected_indices, bias is not None, scale_factor
        )

        return grad_weight, grad_bias


class GREATSLinearBackward(Function):
    """
    Autograd Function for GREATS method (global selection).

    Pass 1: Accumulates scores across all layers (no gradient output)
    Pass 2: Standard forward/backward on selected samples (hooks disabled)
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

        compressor = hook_manager.compressors[layer_idx]
        state: Optional[GREATSState] = hook_manager.selection_state

        if state is None:
            return grad_input, None, None, None, None

        use_stored_val = getattr(state, '_use_stored_val', False)

        with torch.no_grad():
            if compressor is not None:
                GREATSLinearBackward._accumulate_compressed(
                    hook_manager, compressor, state, layer_idx,
                    input, grad_output, bias, use_stored_val
                )
            else:
                GREATSLinearBackward._accumulate_full(
                    hook_manager, state, layer_idx,
                    input, grad_output, use_stored_val
                )

        return grad_input, None, None, None, None

    @staticmethod
    def _accumulate_compressed(
        hook_manager: "GradientHook",
        compressor: "Compressor",
        state: "GREATSState",
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
            val_grad = hook_manager.val_grad_buffer[layer_idx]
            # Cached mode: gradients already correctly scaled
            score_correction = 1.0
        else:
            train_grads, val_grads = split_train_val_batch(compressed_grad, state.train_batch_size)
            val_grad = val_grads.sum(dim=0)  # Sum, not mean, for token-weighted semantics
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction

        if val_grad is not None:
            state.process_layer_gradients(train_grads, val_grad, layer_idx, score_correction)

    @staticmethod
    def _accumulate_full(
        hook_manager: "GradientHook",
        state: "GREATSState",
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
            score_correction = 1.0
        else:
            # Joint batch mode: split merged batch
            train_grad_output, val_go = split_train_val_batch(grad_output, state.train_batch_size)
            train_input, val_inp = split_train_val_batch(input, state.train_batch_size)
            val_grad_total = None
            # Joint batch needs correction: T_total²/(T_train × T_val)
            score_correction = state.score_correction

        scores, similarity = compute_scores_and_similarity(
            train_grad_output, train_input, val_go, val_inp, val_grad_total,
            state.use_second_order
        )

        # Accumulate scores using the state method (handles correction internally)
        state.accumulate_precomputed_scores(scores, similarity, score_correction)
