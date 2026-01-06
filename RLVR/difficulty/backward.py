"""
Autograd functions for RLVR activation-based data selection.

This module provides autograd Functions that use pre-computed activation-based
selections during backward pass:
- RLVRStreamingLinearBackward: Per-layer selection using pre-computed indices
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple, List
    from torch import Tensor

import torch
import torch.nn.functional as F
from torch.autograd import Function


def compute_selected_gradients(
    grad_output: Tensor,
    input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    scale_factor: float = 1.0
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute aggregated gradients for selected samples.

    Args:
        grad_output: Gradient of output [B, S, O] or [B, O]
        input: Input tensor [B, S, I] or [B, I]
        selected_indices: Indices of selected samples [K]
        has_bias: Whether to compute bias gradient
        scale_factor: Scaling factor to normalize for selection

    Returns:
        (grad_weight, grad_bias) tuple where grad_bias is None if not has_bias
    """
    if len(selected_indices) == 0:
        # Empty selection: return zero gradients
        out_features = grad_output.shape[-1]
        in_features = input.shape[-1]
        device = grad_output.device
        dtype = grad_output.dtype
        grad_weight = torch.zeros(out_features, in_features, device=device, dtype=dtype)
        grad_bias = torch.zeros(out_features, device=device, dtype=dtype) if has_bias else None
        return grad_weight, grad_bias

    selected_grad_output = grad_output[selected_indices]
    selected_input = input[selected_indices]

    if selected_grad_output.dim() == 3:
        # 3D case: [K, S, O] x [K, S, I] -> [O, I]
        grad_weight = torch.einsum('kso,ksi->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor if has_bias else None
    else:
        # 2D case: [K, O] x [K, I] -> [O, I]
        grad_weight = torch.einsum('ko,ki->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=0) * scale_factor if has_bias else None

    return grad_weight, grad_bias


class RLVRStreamingLinearBackward(Function):
    """
    Autograd Function for RLVR per-layer selection.

    Unlike gradient-based StreamingLinearBackward which computes selection scores
    during backward, this uses pre-computed activation-based selections.

    The selection is computed after forward pass based on activation similarity
    to reference samples, then applied during backward.

    Flow:
    1. Forward pass captures activations (handled by ActivationHook)
    2. After forward: compute per-layer difficulties and selections
    3. Store selections on hook manager (rlvr_layer_selections)
    4. Backward: retrieve pre-computed selection, aggregate gradients
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager,  # Can be GradientHook or a specialized RLVR hook
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
        """Backward pass with pre-computed RLVR selection."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        # Standard grad_input computation
        grad_input = grad_output @ weight.to(grad_output.dtype)

        with torch.no_grad():
            # Retrieve pre-computed RLVR selection for this layer
            selected_indices = hook_manager.rlvr_layer_selections[layer_idx]

            if selected_indices is None:
                # No selection for this layer: use all samples
                grad_weight, grad_bias = RLVRStreamingLinearBackward._compute_full_gradients(
                    grad_output, input, bias is not None
                )
            else:
                # Compute scale factor for proper gradient normalization
                scale_factor = RLVRStreamingLinearBackward._compute_scale_factor(
                    hook_manager, selected_indices, grad_output.shape[0]
                )

                # Aggregate gradients from selected samples only
                grad_weight, grad_bias = compute_selected_gradients(
                    grad_output, input, selected_indices, bias is not None, scale_factor
                )

                # Track selection stats
                if hasattr(hook_manager, 'rlvr_selection_stats'):
                    hook_manager.rlvr_selection_stats.setdefault('layer_selections', []).append(
                        (layer_idx, len(selected_indices))
                    )

        return grad_input, grad_weight, grad_bias, None, None

    @staticmethod
    def _compute_full_gradients(
        grad_output: Tensor,
        input: Tensor,
        has_bias: bool
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Compute gradients using all samples (no selection)."""
        if grad_output.dim() == 3:
            grad_weight = torch.einsum('bso,bsi->oi', grad_output, input)
            grad_bias = grad_output.sum(dim=(0, 1)) if has_bias else None
        else:
            grad_weight = torch.einsum('bo,bi->oi', grad_output, input)
            grad_bias = grad_output.sum(dim=0) if has_bias else None
        return grad_weight, grad_bias

    @staticmethod
    def _compute_scale_factor(
        hook_manager,
        selected_indices: Tensor,
        batch_size: int
    ) -> float:
        """
        Compute scaling factor for proper gradient normalization.

        When selecting a subset of samples, we need to scale gradients to maintain
        the same expected gradient magnitude as if all samples were used.

        For token-averaged loss (LM training):
            scale = total_tokens / selected_tokens

        For sample-averaged loss:
            scale = batch_size / num_selected
        """
        num_selected = len(selected_indices)
        if num_selected == 0:
            return 1.0

        # Token-based scaling (for language models)
        if hasattr(hook_manager, 'tokens_per_sample') and hook_manager.tokens_per_sample is not None:
            tokens_per_sample = hook_manager.tokens_per_sample
            total_tokens = hook_manager.total_tokens or tokens_per_sample.sum()
            selected_tokens = tokens_per_sample[selected_indices].sum()
            if selected_tokens > 0:
                return float(total_tokens / selected_tokens)
            return 1.0

        # Sample-based scaling (fallback)
        return batch_size / num_selected


class RLVRGlobalLinearBackward(Function):
    """
    Autograd Function for RLVR global selection (score accumulation).

    Used when computing scores across all layers before final selection.
    Does not perform selection during backward - just accumulates for
    final global selection decision.
    """

    @staticmethod
    def forward(
        ctx,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        hook_manager,
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
        """Backward pass: just pass through gradients, selection done elsewhere."""
        input, weight, bias = ctx.saved_tensors
        layer_idx = ctx.layer_idx

        hook_manager = ctx.hook_manager_ref()
        if hook_manager is None:
            raise RuntimeError("Hook manager was garbage collected before backward pass")

        if input.dtype != grad_output.dtype:
            input = input.to(grad_output.dtype)

        # Standard grad_input computation
        grad_input = grad_output @ weight.to(grad_output.dtype)

        # For global RLVR, selection is done once after all layers are processed
        # during the first pass, then a second forward/backward is done on selected samples
        # So we just return grad_input and let standard autograd handle the rest

        return grad_input, None, None, None, None
