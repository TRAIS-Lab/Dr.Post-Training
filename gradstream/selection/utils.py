"""
Utility functions for gradient-based data selection.

This module provides low-level helper functions used by the backward autograd functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor

import torch


def augment_input_for_bias(input: Tensor, has_bias: bool) -> Tensor:
    """
    Augment input tensor with ones column for bias gradient computation.

    When computing gradients for a linear layer with bias, we can treat
    bias as an extra weight column by appending ones to the input.

    Args:
        input: Input tensor [B, S, I] or [B, I]
        has_bias: Whether the layer has bias

    Returns:
        Augmented input [B, S, I+1] or [B, I+1] if has_bias, else unchanged input
    """
    if not has_bias:
        return input

    batch_size = input.shape[0]
    if input.dim() == 3:
        seq_length = input.shape[1]
        ones = torch.ones(batch_size, seq_length, 1, device=input.device, dtype=input.dtype)
    else:
        ones = torch.ones(batch_size, 1, device=input.device, dtype=input.dtype)
    return torch.cat([input, ones], dim=-1)


def split_train_val_batch(
    tensor: Tensor,
    train_batch_size: int
) -> Tuple[Tensor, Tensor]:
    """
    Split a merged batch tensor into train and validation portions.

    Args:
        tensor: Merged tensor with train samples first, then val samples
        train_batch_size: Number of training samples (first N in batch)

    Returns:
        (train_portion, val_portion) tuple
    """
    train_portion = tensor[:train_batch_size]
    val_portion = tensor[train_batch_size:]
    return train_portion, val_portion


def compute_total_gradient(
    grad_output: Tensor,
    input: Tensor
) -> Tensor:
    """
    Compute total (summed) gradient [O, I] from grad_output and input.

    This computes the sum of gradients across samples: Σ_b grad_b
    where grad_b[o,i] = Σ_s grad_output[b,s,o] × input[b,s,i].

    Note on scaling:
    - The grad_output already has loss function scaling (1/total_tokens for
      token-averaged loss).
    - We sum (not average) to be consistent with token-weighted loss semantics.
    - Samples with more tokens naturally contribute more through gradient magnitude.

    Args:
        grad_output: Gradient of output [B, S, O] or [B, O]
        input: Input tensor [B, S, I] or [B, I]

    Returns:
        Total gradient [O, I]
    """
    if grad_output.dim() == 3:
        return torch.einsum('bso,bsi->oi', grad_output, input)
    else:
        return torch.einsum('bo,bi->oi', grad_output, input)


@torch.compile
def compute_scores_and_similarity(
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    use_second_order: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Unified score computation that handles both factorized and full gradient modes.

    Args:
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        train_input: Training input [B, S, I] or [B, I]
        val_grad_output: Validation grad_output [V, S, O] or None
        val_input: Validation input [V, S, I] or None
        val_grad_total: Total validation gradient [O, I] or None
        use_second_order: Whether to compute similarity matrix

    Returns:
        (scores, similarity) tuple where similarity is None if not use_second_order
    """
    # Compute scores
    # Priority to factorized mode if available
    if val_grad_output is not None and val_input is not None:
        # Ghost Inner Product computation
        # Optimization: Precompute val_grad_total, then use efficient matmul path
        # This is O(V*T*O*I + B*S*I*O) vs O(B*S*V*T*O*I) for 4-way einsum
        if train_grad_output.dim() == 3:
            # 3D case: precompute val gradient, then matmul
            val_grad_total = torch.einsum('vto,vti->oi', val_grad_output, val_input)
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=(1, 2))
        else:
            # 2D case: precompute val gradient, then matmul
            val_grad_total = torch.einsum('vo,vi->oi', val_grad_output, val_input)
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=1)
    elif val_grad_total is not None:
        # This is used in cached_full mode where we store the summed validation gradient
        if train_grad_output.dim() == 3:
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=(1, 2))
        else:
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=1)
    else:
        raise ValueError("Must provide either (val_grad_output, val_input) or val_grad_total")

    # Compute similarity if needed
    similarity = None
    if use_second_order:
        # S[i,j] = <grad_i, grad_j> where grad_i is the per-sample gradient for sample i.
        if train_grad_output.dim() == 3:
            # 3D: contract over sequence and features
            contracted = torch.bmm(
                train_grad_output.permute(0, 2, 1),
                train_input
            ).flatten(start_dim=1)
            similarity = torch.matmul(contracted, contracted.T)
        else:
            # 2D: Hadamard product of dot products
            dot_g = torch.matmul(train_grad_output, train_grad_output.T)
            dot_x = torch.matmul(train_input, train_input.T)
            similarity = dot_g * dot_x

    return scores, similarity


def compute_selected_gradients(
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    scale_factor: float
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute aggregated gradients for selected samples.

    Args:
        train_grad_output: Training grad_output [B, S, O] or [B, O]
        train_input: Training input [B, S, I] or [B, I]
        selected_indices: Indices of selected samples [K]
        has_bias: Whether to compute bias gradient
        scale_factor: Scaling factor to normalize for selection

    Returns:
        (grad_weight, grad_bias) tuple where grad_bias is None if not has_bias
    """
    selected_grad_output = train_grad_output[selected_indices]
    selected_input = train_input[selected_indices]

    if selected_grad_output.dim() == 3:
        # 3D case: [K, S, O] x [K, S, I] -> [O, I]
        grad_weight = torch.einsum('kso,ksi->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor if has_bias else None
    else:
        # 2D case: [K, O] x [K, I] -> [O, I]
        grad_weight = torch.einsum('ko,ki->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=0) * scale_factor if has_bias else None

    return grad_weight, grad_bias
