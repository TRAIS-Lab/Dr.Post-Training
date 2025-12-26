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
    from torch.autograd import Function
    from .state import StreamingState, GREATSState
    from ..hook import GradientHook
    from ..compressor import Compressor

import torch
import torch.nn.functional as F
from torch.autograd import Function


def _augment_input_for_bias(input: Tensor, has_bias: bool) -> Tensor:
    """Augment input with ones column for bias gradient computation."""
    if not has_bias:
        return input

    batch_size = input.shape[0]
    if input.dim() == 3:
        _, seq_length, in_features = input.shape
        ones = torch.ones(batch_size, seq_length, 1, device=input.device, dtype=input.dtype)
    else:
        ones = torch.ones(batch_size, 1, device=input.device, dtype=input.dtype)
    return torch.cat([input, ones], dim=-1)


def _split_merged_train_val_batch(
    tensor: Tensor,
    train_batch_size: int
) -> Tuple[Tensor, Tensor]:
    """Split merged batch tensor into train and val portions."""
    train_portion = tensor[:train_batch_size]
    val_portion = tensor[train_batch_size:]
    return train_portion, val_portion


def _compute_val_gradient_full(
    val_grad_output: Tensor,
    val_input: Tensor
) -> Tensor:
    """Compute mean validation gradient for full gradient mode.

    Uses matmul instead of einsum for better performance:
    einsum('bso,bsi->oi') -> reshape to [B*S, O] and [B*S, I], then matmul.T @ X
    """
    if val_grad_output.dim() == 3:
        # Reshape: [V, S, O] -> [V*S, O] and [V, S, I] -> [V*S, I]
        O = val_grad_output.shape[-1]
        I = val_input.shape[-1]
        go_flat = val_grad_output.reshape(-1, O)  # [V*S, O]
        inp_flat = val_input.reshape(-1, I)       # [V*S, I]
        # matmul: [O, V*S] @ [V*S, I] -> [O, I]
        val_grad_full = go_flat.T @ inp_flat
    else:
        # 2D case: [V, O] @ [V, I].T -> matmul is the same as einsum
        val_grad_full = val_grad_output.T @ val_input
    return val_grad_full / val_grad_output.shape[0]


def _compute_scores_ghost(
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_full: Tensor,
    use_second_order: bool
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute per-sample scores using ghost inner product (without materializing full gradients).

    Returns:
        (scores, similarity) tuple where similarity is None if not use_second_order
    """
    similarity = None

    if train_grad_output.dim() == 3:
        # 3D case: [batch, seq, features]
        temp = train_input @ val_grad_full.T
        scores = (train_grad_output * temp).sum(dim=(1, 2))

        if use_second_order:
            contracted = torch.bmm(
                train_grad_output.permute(0, 2, 1),
                train_input
            ).flatten(start_dim=1)
            similarity = torch.matmul(contracted, contracted.T)
    else:
        # 2D case: [batch, features]
        temp = train_input @ val_grad_full.T
        scores = (train_grad_output * temp).sum(dim=1)

        if use_second_order:
            dot_g = torch.matmul(train_grad_output, train_grad_output.T)
            dot_x = torch.matmul(train_input, train_input.T)
            similarity = dot_g * dot_x

    return scores, similarity


def _compute_selected_gradients_full(
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    scale_factor: float
) -> Tuple[Tensor, Optional[Tensor]]:
    """Compute gradients for selected samples in full gradient mode.

    Uses matmul instead of einsum for better performance:
    einsum('kso,ksi->oi') -> reshape to [K*S, O] and [K*S, I], then matmul.T @ X
    """
    selected_grad_output = train_grad_output[selected_indices]
    selected_input = train_input[selected_indices]

    if selected_grad_output.dim() == 3:
        # Reshape: [K, S, O] -> [K*S, O] and [K, S, I] -> [K*S, I]
        O = selected_grad_output.shape[-1]
        I = selected_input.shape[-1]
        go_flat = selected_grad_output.reshape(-1, O)  # [K*S, O]
        inp_flat = selected_input.reshape(-1, I)       # [K*S, I]
        # matmul: [O, K*S] @ [K*S, I] -> [O, I]
        grad_weight = (go_flat.T @ inp_flat) * scale_factor
        if has_bias:
            grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor
        else:
            grad_bias = None
    else:
        # 2D case: [K, O].T @ [K, I] -> [O, I]
        grad_weight = (selected_grad_output.T @ selected_input) * scale_factor
        if has_bias:
            grad_bias = selected_grad_output.sum(dim=0) * scale_factor
        else:
            grad_bias = None

    return grad_weight, grad_bias


class CompressedLinearBackward(Function):
    """
    Autograd Function for pure gradient compression (no data selection).

    This is used when:
    - Compression is enabled (MeSO optimizer with compressors)
    - No data selection is active (selection_state is None)

    Simply compresses per-sample gradients, sums them, and stores for MeSO.
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
        """
        Backward pass: compress gradients and store for MeSO optimizer.

        No selection logic - just compress and sum.
        """
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        # Cast input to match grad_output dtype
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        # Compute grad_input (always needed for backprop)
        grad_input = grad_output @ weight.to(grad_output.dtype)

        compressor = hook_manager.compressors[layer_idx]

        with torch.no_grad():
            # Augment input for bias gradient computation
            input_for_compressor = _augment_input_for_bias(input, bias is not None)

            # Compress per-sample gradients
            compressed_grad = compressor.forward((grad_output, input_for_compressor))

            # Sum across batch dimension and store
            compressed_grad = compressed_grad.sum(dim=0, keepdim=True)
            hook_manager._store_compressed_grad(layer_idx, compressed_grad)

        return grad_input, None, None, None, None


class StreamingLinearBackward(Function):
    """
    Autograd Function for Streaming method (per-layer selection).

    Single-pass: At each layer, computes scores, selects samples,
    and aggregates gradients immediately.

    Supports:
    - Compressed mode (with MeSO): stores compressed gradients
    - Full gradient mode: returns grad_weight/grad_bias
    - Merged batch (SFT) and stored val (RLHF) validation modes
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: GradientHook,
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        # Store weakref to avoid preventing garbage collection
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], None, None]:
        """
        Backward pass with per-layer selection.

        For each layer:
        1. Compute scores (train_grad @ val_grad)
        2. Select samples based on scores
        3. Aggregate selected gradients
        """
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        # Cast input to match grad_output dtype
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        # Compute grad_input (always needed for backprop)
        grad_input = grad_output @ weight.to(grad_output.dtype)

        grad_weight = None
        grad_bias = None

        compressor = hook_manager.compressors[layer_idx]
        state: Optional[StreamingState] = hook_manager.selection_state

        # Validation capture mode (RLHF)
        capture_val_mode = hook_manager.capture_val_mode
        use_stored_val = (
            state is not None and
            hasattr(state, '_use_stored_val') and
            state._use_stored_val
        )

        with torch.no_grad():
            if compressor is not None:
                # === COMPRESSED MODE ===
                grad_weight, grad_bias = StreamingLinearBackward._backward_compressed(
                    hook_manager, compressor, state, layer_idx,
                    input, grad_output, bias,
                    capture_val_mode, use_stored_val
                )
            else:
                # === FULL GRADIENT MODE ===
                grad_weight, grad_bias = StreamingLinearBackward._backward_full(
                    hook_manager, state, layer_idx,
                    input, weight, bias, grad_output,
                    capture_val_mode, use_stored_val
                )

        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def _backward_compressed(
        hook_manager: GradientHook,
        compressor: "Compressor",
        state: Optional[StreamingState],
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[None, None]:
        """Handle compressed gradient path for Streaming."""
        # Augment input for bias
        input_for_compressor = _augment_input_for_bias(input, bias is not None)

        # Compress gradients
        compressed_grad = compressor.forward((grad_output, input_for_compressor))

        if capture_val_mode:
            # RLHF val capture: accumulate mean gradient
            mean_grad = compressed_grad.mean(dim=0)
            if hook_manager.val_grad_buffer[layer_idx] is None:
                hook_manager.val_grad_buffer[layer_idx] = mean_grad
            else:
                hook_manager.val_grad_buffer[layer_idx] = (
                    hook_manager.val_grad_buffer[layer_idx] + mean_grad
                )
            return None, None

        if state is None:
            # Baseline: sum compressed gradients
            compressed_grad = compressed_grad.sum(dim=0, keepdim=True)
            hook_manager._store_compressed_grad(layer_idx, compressed_grad)
            return None, None

        # Get train grads and val grad
        if use_stored_val:
            # RLHF: use stored validation gradients
            train_grads = compressed_grad
            val_grad = hook_manager.val_grad_buffer[layer_idx]
        else:
            # SFT: split merged batch
            train_grads, val_grads = _split_merged_train_val_batch(
                compressed_grad, state.train_batch_size
            )
            val_grad = val_grads.mean(dim=0)

        if val_grad is None:
            # No val grad, just average
            hook_manager._store_compressed_grad(layer_idx, compressed_grad.mean(dim=0, keepdim=True))
            return None, None

        # Per-layer selection and reduction
        reduced_grad, num_selected = state.process_layer_gradients(
            train_grads, val_grad, layer_idx
        )

        hook_manager._store_compressed_grad(layer_idx, reduced_grad)
        return None, None

    @staticmethod
    def _backward_full(
        hook_manager: GradientHook,
        state: Optional[StreamingState],
        layer_idx: int,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        grad_output: Tensor,
        capture_val_mode: bool,
        use_stored_val: bool
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Handle full gradient path for Streaming."""

        if capture_val_mode:
            # RLHF val capture: accumulate mean full gradient
            if grad_output.dim() == 3:
                val_grad_full = torch.einsum('bso,bsi->oi', grad_output, input) / grad_output.shape[0]
            else:
                val_grad_full = torch.einsum('bo,bi->oi', grad_output, input) / grad_output.shape[0]

            if hook_manager.val_grad_buffer[layer_idx] is None:
                hook_manager.val_grad_buffer[layer_idx] = val_grad_full
            else:
                hook_manager.val_grad_buffer[layer_idx] = (
                    hook_manager.val_grad_buffer[layer_idx] + val_grad_full
                )
            return None, None

        if state is None:
            # Baseline: let PyTorch compute gradients
            return None, None

        # Get train/val split
        if use_stored_val:
            train_grad_output = grad_output
            train_input = input
            train_batch_size = state.train_batch_size
            val_grad_full = hook_manager.val_grad_buffer[layer_idx]
        else:
            train_batch_size = state.train_batch_size
            train_grad_output, val_grad_output = _split_merged_train_val_batch(
                grad_output, train_batch_size
            )
            train_input, val_input = _split_merged_train_val_batch(
                input, train_batch_size
            )

        if not use_stored_val:
            val_grad_full = _compute_val_gradient_full(val_grad_output, val_input)

        if val_grad_full is None:
            return None, None

        # Compute scores using ghost inner product
        scores, similarity = _compute_scores_ghost(
            train_grad_output, train_input, val_grad_full, state.use_second_order
        )

        # Per-layer selection
        selected_indices = state._select_indices(scores, similarity)
        num_selected = selected_indices.shape[0]
        state.num_selected = num_selected

        if num_selected == 0:
            return torch.zeros_like(weight), torch.zeros_like(bias) if bias is not None else None

        # Compute scale factor
        if state.tokens_per_sample is not None and state.total_tokens is not None:
            selected_tokens = state.tokens_per_sample[selected_indices].sum().item()
            scale_factor = state.total_tokens / selected_tokens if selected_tokens > 0 else 1.0
        else:
            scale_factor = train_batch_size / num_selected

        # Compute selected gradients
        grad_weight, grad_bias = _compute_selected_gradients_full(
            train_grad_output, train_input, selected_indices,
            bias is not None, scale_factor
        )

        return grad_weight, grad_bias


class GREATSLinearBackward(Function):
    """
    Autograd Function for GREATS method (global selection).

    Pass 1 (score accumulation): Accumulates scores across all layers
    Pass 2 (gradient computation): Standard forward/backward on selected samples
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager: GradientHook,
        layer_idx: int
    ) -> Tensor:
        """Forward pass: standard linear transformation."""
        input_compute = input.to(weight.dtype) if input.dtype != weight.dtype else input
        ctx.save_for_backward(input_compute, weight, bias)
        # Store weakref to avoid preventing garbage collection
        ctx.hook_manager_ref = weakref.ref(hook_manager)
        ctx.layer_idx = layer_idx
        return F.linear(input_compute, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: Tensor
    ) -> Tuple[Tensor, None, None, None, None]:
        """
        Backward pass: accumulate scores only, no gradient output.

        For GREATS pass 1, we only accumulate scores across layers.
        The actual gradient computation happens in pass 2 with hooks disabled.
        """
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        # Cast input to match grad_output dtype
        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        # Compute grad_input (always needed for backprop)
        grad_input = grad_output @ weight.to(grad_output.dtype)

        compressor = hook_manager.compressors[layer_idx]
        state: Optional[GREATSState] = hook_manager.selection_state

        # Skip if no state (shouldn't happen for GREATS)
        if state is None:
            return grad_input, None, None, None, None

        use_stored_val = (
            hasattr(state, '_use_stored_val') and
            state._use_stored_val
        )

        with torch.no_grad():
            if compressor is not None:
                GREATSLinearBackward._accumulate_scores_compressed(
                    hook_manager, compressor, state, layer_idx,
                    input, grad_output, bias, use_stored_val
                )
            else:
                GREATSLinearBackward._accumulate_scores_full(
                    hook_manager, state, layer_idx,
                    input, grad_output, use_stored_val
                )

        # GREATS pass 1: only accumulate scores, no gradient output
        return grad_input, None, None, None, None

    @staticmethod
    def _accumulate_scores_compressed(
        hook_manager: GradientHook,
        compressor: "Compressor",
        state: GREATSState,
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        bias: Optional[Tensor],
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from compressed gradients."""
        input_for_compressor = _augment_input_for_bias(input, bias is not None)
        compressed_grad = compressor.forward((grad_output, input_for_compressor))

        if use_stored_val:
            train_grads = compressed_grad
            val_grad = hook_manager.val_grad_buffer[layer_idx]
        else:
            train_grads, val_grads = _split_merged_train_val_batch(
                compressed_grad, state.train_batch_size
            )
            val_grad = val_grads.mean(dim=0)

        if val_grad is not None:
            state.process_layer_gradients(train_grads, val_grad, layer_idx)

    @staticmethod
    def _accumulate_scores_full(
        hook_manager: GradientHook,
        state: GREATSState,
        layer_idx: int,
        input: Tensor,
        grad_output: Tensor,
        use_stored_val: bool
    ) -> None:
        """Accumulate scores from full gradients."""
        if use_stored_val:
            train_grad_output = grad_output
            train_input = input
            val_grad_full = hook_manager.val_grad_buffer[layer_idx]
        else:
            train_batch_size = state.train_batch_size
            train_grad_output, val_grad_output = _split_merged_train_val_batch(
                grad_output, train_batch_size
            )
            train_input, val_input = _split_merged_train_val_batch(
                input, train_batch_size
            )

        if not use_stored_val:
            val_grad_full = _compute_val_gradient_full(val_grad_output, val_input)

        if val_grad_full is None:
            return

        # Compute scores
        scores, similarity = _compute_scores_ghost(
            train_grad_output, train_input, val_grad_full, state.use_second_order
        )

        # Directly update accumulators (more efficient than process_layer_gradients for full grad)
        state.grad_dot_scores += scores.to(state.dtype)
        if state.similarity_matrix is not None and similarity is not None:
            state.similarity_matrix += similarity.to(state.dtype)
