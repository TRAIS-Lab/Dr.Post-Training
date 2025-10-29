"""
Simplified hook-based gradient computation for GREATS.

This follows the GPT2_wikitext pattern: simple loss.backward() + hook capture.
"""

import sys
sys.path.append('/u/phu1/Project/Efficient-Fine-Tuning')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from typing import Dict
from torch import Tensor

from _GradComp.core.hook import HookManager

logger = logging.getLogger(__name__)


def compute_grad_dotprod(
    model: nn.Module,
    hook_manager: HookManager,
    batch_train: Dict[str, Tensor],
    batch_val: Dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    return_similarity: bool = True
):
    """
    Compute gradient dot products using simple backward pass + hooks.

    Args:
        model: The model
        hook_manager: HookManager with hooks attached
        batch_train: Training batch
        batch_val: Validation batch (single batch)
        optimizer: Optimizer
        return_similarity: Whether to return similarity matrix

    Returns:
        - grad_dot_scores: [train_bs] gradient similarity with validation
        - similarity_matrix: [train_bs, train_bs] if return_similarity else None
    """
    # Move validation batch to same device as training
    device = batch_train['input_ids'].device
    batch_val = {k: v.to(device) for k, v in batch_val.items()}

    # Step 1: Compute validation gradients
    model.zero_grad()
    val_outputs = model(**batch_val)
    val_loss = val_outputs.loss
    val_loss.backward()

    # Get validation gradients from hooks
    val_grads = hook_manager.get_compressed_grads()
    # Concatenate all layers: each is [batch_size, grad_dim]
    val_grads_concat = torch.cat([g for g in val_grads if g is not None], dim=1)  # [val_bs, total_dim]
    # Average over validation batch
    val_grad_avg = val_grads_concat.mean(dim=0, keepdim=True)  # [1, total_dim]

    # Step 2: Compute training gradients (per-sample)
    model.zero_grad()
    train_outputs = model(**batch_train)
    train_loss = train_outputs.loss
    train_loss.backward()

    # Get training gradients from hooks
    train_grads = hook_manager.get_compressed_grads()
    # Concatenate all layers
    train_grads_concat = torch.cat([g for g in train_grads if g is not None], dim=1)  # [train_bs, total_dim]

    # Step 3: Compute GradDot scores
    # [train_bs, total_dim] x [total_dim, 1] -> [train_bs]
    grad_dot_scores = torch.matmul(train_grads_concat, val_grad_avg.t()).squeeze(-1)
    grad_dot_scores = grad_dot_scores.cpu().detach()

    # Step 4: Compute similarity matrix if requested
    similarity_matrix = None
    if return_similarity:
        # [train_bs, total_dim] x [total_dim, train_bs] -> [train_bs, train_bs]
        similarity_matrix = torch.matmul(train_grads_concat, train_grads_concat.t())
        similarity_matrix = similarity_matrix.cpu().detach()

    return grad_dot_scores, similarity_matrix
