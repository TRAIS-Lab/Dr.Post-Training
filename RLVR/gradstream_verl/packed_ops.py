"""
Optimized operations for packed sequences in Verl.

These operations avoid D2H memory copies by using vectorized PyTorch operations
instead of Python loops with .item() calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Optional, Tuple
    from torch import Tensor

import torch


def compute_sample_ids_from_cu_seqlens(
    total_tokens: int,
    cu_seqlens: Tensor,
    device: torch.device,
) -> Tensor:
    """
    Compute sample ID for each token position using cu_seqlens.

    This is a vectorized alternative to looping with .item() calls.

    Args:
        total_tokens: Total number of tokens in packed sequence
        cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        device: Device to create tensor on

    Returns:
        sample_ids: [total_tokens] tensor where sample_ids[i] = sample index for token i
    """
    token_indices = torch.arange(total_tokens, device=device)
    # Ensure cu_seqlens is on the same device
    cu_seqlens_device = cu_seqlens.to(device) if cu_seqlens.device != device else cu_seqlens
    # bucketize assigns each token to the appropriate sample
    # right=True means cu_seqlens[i] <= token_idx < cu_seqlens[i+1] maps to sample i
    sample_ids = torch.bucketize(token_indices, cu_seqlens_device[1:], right=True)
    return sample_ids


def compute_scores_packed_vectorized(
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_total: Tensor,
    cu_seqlens: Tensor,
    use_second_order: bool,
) -> Tuple[Tensor, Optional[Tensor], Tensor]:
    """
    Compute per-sample scores from packed sequences using vectorized operations.

    This avoids D2H memory copies by using torch.bucketize + scatter_add instead
    of Python loops with .item() calls.

    Args:
        train_grad_output: Packed training grad_output [1, total_nnz, O]
        train_input: Packed training input [1, total_nnz, I]
        val_grad_total: Total validation gradient [O, I]
        cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        use_second_order: Whether to compute similarity matrix

    Returns:
        (scores, similarity, sample_ids) tuple where:
        - scores: [batch_size] per-sample scores
        - similarity: [batch_size, batch_size] or None
        - sample_ids: [total_nnz] sample index for each token (for reuse)
    """
    target_dtype = train_grad_output.dtype
    val_grad_total = val_grad_total.to(target_dtype)

    # Squeeze batch dimension: [1, total_nnz, D] -> [total_nnz, D]
    grad_output = train_grad_output.squeeze(0)
    input_tensor = train_input.squeeze(0)

    total_tokens = grad_output.shape[0]
    batch_size = cu_seqlens.shape[0] - 1
    device = grad_output.device

    # Compute per-token scores: <grad_output[t], val_grad_total @ input[t]>
    # = (grad_output * (input @ val_grad_total.T)).sum(dim=1)
    temp = input_tensor @ val_grad_total.T  # [total_nnz, O]
    per_token_scores = (grad_output * temp).sum(dim=1)  # [total_nnz]

    # Aggregate per-token scores to per-sample scores using scatter_add
    sample_ids = compute_sample_ids_from_cu_seqlens(total_tokens, cu_seqlens, device)
    scores = torch.zeros(batch_size, device=device, dtype=target_dtype)
    scores.scatter_add_(0, sample_ids, per_token_scores)

    # Compute similarity if needed
    similarity = None
    if use_second_order:
        # For second-order, we need per-sample gradients
        # grad_i = sum_t grad_output[t] ⊗ input[t] for t in sample i
        #
        # Memory-efficient approach: compute per-sample gradients one at a time
        # instead of materializing the full [total_nnz, O, I] outer product tensor.
        # This trades some compute (Python loop) for massive memory savings.

        out_dim, in_dim = grad_output.shape[1], input_tensor.shape[1]
        sample_grads = torch.zeros(batch_size, out_dim * in_dim, device=device, dtype=target_dtype)

        # Compute per-sample gradients using efficient einsum (no large intermediate)
        for sample_idx in range(batch_size):
            mask = (sample_ids == sample_idx)
            if mask.any():
                sample_go = grad_output[mask]  # [n_tokens_i, O]
                sample_in = input_tensor[mask]  # [n_tokens_i, I]
                # einsum computes sum of outer products without materializing [n, O, I]
                sample_grad = torch.einsum('to,ti->oi', sample_go, sample_in)  # [O, I]
                sample_grads[sample_idx] = sample_grad.flatten()

        # Compute similarity matrix: [batch_size, batch_size]
        similarity = sample_grads @ sample_grads.T

    return scores, similarity, sample_ids


def compute_selected_gradients_packed_vectorized(
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    cu_seqlens: Tensor,
    has_bias: bool,
    scale_factor: Tensor,
    sample_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute aggregated gradients for selected samples from packed sequences.

    This uses vectorized operations to avoid D2H memory copies.

    Args:
        train_grad_output: Packed training grad_output [1, total_nnz, O]
        train_input: Packed training input [1, total_nnz, I]
        selected_indices: Indices of selected samples [K] (sample indices)
        cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        has_bias: Whether to compute bias gradient
        scale_factor: Scaling factor to normalize for selection (scalar Tensor)
        sample_ids: Optional pre-computed sample IDs [total_nnz] (from compute_scores_packed_vectorized)

    Returns:
        (grad_weight, grad_bias) tuple where grad_bias is None if not has_bias
    """
    # Squeeze batch dimension
    grad_output = train_grad_output.squeeze(0)  # [total_nnz, O]
    input_tensor = train_input.squeeze(0)  # [total_nnz, I]

    total_tokens = grad_output.shape[0]
    device = grad_output.device
    dtype = grad_output.dtype

    # Compute sample IDs for all tokens (reuse if provided)
    if sample_ids is None:
        sample_ids = compute_sample_ids_from_cu_seqlens(total_tokens, cu_seqlens, device)

    # Create mask for tokens belonging to selected samples
    # Use broadcasting: sample_ids [total_nnz] vs selected_indices [K]
    # token_mask[t] = True if sample_ids[t] in selected_indices
    token_mask = torch.isin(sample_ids, selected_indices)

    # Extract tokens from selected samples
    selected_grad_output = grad_output[token_mask]  # [N_selected_tokens, O]
    selected_input = input_tensor[token_mask]  # [N_selected_tokens, I]

    # Compute gradient: sum of outer products
    grad_weight = torch.einsum('to,ti->oi', selected_grad_output, selected_input) * scale_factor

    grad_bias = None
    if has_bias:
        grad_bias = selected_grad_output.sum(dim=0) * scale_factor

    return grad_weight, grad_bias


def compute_scores_and_similarity_standard(
    train_grad_output: Tensor,
    train_input: Tensor,
    val_grad_output: Optional[Tensor],
    val_input: Optional[Tensor],
    val_grad_total: Optional[Tensor],
    use_second_order: bool,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Standard score computation for non-packed batches.

    This is the same as gradstream's compute_scores_and_similarity but without
    @torch.compile for flexibility in RLVR context.
    """
    target_dtype = train_grad_output.dtype
    if val_grad_output is not None:
        val_grad_output = val_grad_output.to(target_dtype)
    if val_input is not None:
        val_input = val_input.to(target_dtype)
    if val_grad_total is not None:
        val_grad_total = val_grad_total.to(target_dtype)

    # Compute scores
    if val_grad_output is not None and val_input is not None:
        if train_grad_output.dim() == 3:
            val_grad_total = torch.einsum('vto,vti->oi', val_grad_output, val_input)
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=(1, 2))
        else:
            val_grad_total = torch.einsum('vo,vi->oi', val_grad_output, val_input)
            temp = train_input @ val_grad_total.T
            scores = (train_grad_output * temp).sum(dim=1)
    elif val_grad_total is not None:
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
        if train_grad_output.dim() == 3:
            contracted = torch.bmm(
                train_grad_output.permute(0, 2, 1),
                train_input
            ).flatten(start_dim=1)
            similarity = torch.matmul(contracted, contracted.T)
        else:
            dot_g = torch.matmul(train_grad_output, train_grad_output.T)
            dot_x = torch.matmul(train_input, train_input.T)
            similarity = dot_g * dot_x

    return scores, similarity


def compute_selected_gradients_standard(
    train_grad_output: Tensor,
    train_input: Tensor,
    selected_indices: Tensor,
    has_bias: bool,
    scale_factor: Tensor,
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Standard gradient computation for non-packed batches.
    """
    selected_grad_output = train_grad_output[selected_indices]
    selected_input = train_input[selected_indices]

    if selected_grad_output.dim() == 3:
        grad_weight = torch.einsum('kso,ksi->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=(0, 1)) * scale_factor if has_bias else None
    else:
        grad_weight = torch.einsum('ko,ki->oi', selected_grad_output, selected_input) * scale_factor
        grad_bias = selected_grad_output.sum(dim=0) * scale_factor if has_bias else None

    return grad_weight, grad_bias
