"""
Loss-reduction conventions shared by the trainers and the curation hook.

drpt reads per-sample gradients off the backward pass of a single batch loss, so
whatever that loss averages over becomes the definition of "one item" in every
influence score and every curated update:

``"sample_mean"`` (default, the paper's convention)
    L = (1/N) * sum_i lbar_i, where lbar_i is the per-example *token-mean* loss.
    Every example is one item. The per-sample gradient seen by the hook is
    grad(lbar_i) / N, the score of sample b is <grad lbar_b, (1/m) sum_j grad lbar_j>
    (up to a common factor), and the curated update is the plain mean over the
    selected samples, (1/k) sum_{b in S} grad lbar_b.

``"token_mean"`` (legacy, the Hugging Face default)
    L = (1/T_tot) * sum over every supervised token in the batch. An item is a
    token, so a sample's gradient is its token-sum gradient over a batch-wide
    constant: scores carry an extra factor of the sample's token count and the
    curated update is a token mean over the selected samples, not a mean over
    samples. Runs before 2026-09-11 used this convention.

The score correction and assembly scale in ``drpt.selection.state`` are exact for
either convention as long as the item counts handed to the hook match the loss
(``GradientHook(loss_reduction=...)`` and ``item_counts_from_labels``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor

LOSS_REDUCTIONS = ("sample_mean", "token_mean")


def validate_loss_reduction(value: str) -> str:
    """Return ``value`` if it names a supported loss reduction, else raise."""
    if value not in LOSS_REDUCTIONS:
        raise ValueError(f"loss_reduction must be one of {LOSS_REDUCTIONS}, got {value!r}")
    return value


def per_example_mean(per_token_loss: "Tensor", mask: "Tensor") -> "Tensor":
    """
    Per-example token-mean loss.

    Args:
        per_token_loss: [B, T] loss per position.
        mask: [B, T] 1/True at supervised positions.

    Returns:
        [B] mean loss over each row's supervised positions; 0 for rows without any.
    """
    mask = mask.to(per_token_loss.dtype)
    counts = mask.sum(dim=-1)
    return (per_token_loss * mask).sum(dim=-1) / counts.clamp(min=1)


def reduce_masked_loss(
    per_token_loss: "Tensor",
    mask: "Tensor",
    reduction: str = "sample_mean",
) -> "Tensor":
    """
    Reduce a [B, T] masked per-token loss to a scalar under ``reduction``.

    - ``"sample_mean"``: mean over examples (rows with at least one supervised
      position) of the per-example token-mean loss.
    - ``"token_mean"``: mean over all supervised positions in the batch.

    Batches without any supervised position return 0 instead of NaN.
    """
    validate_loss_reduction(reduction)
    mask = mask.to(per_token_loss.dtype)
    if reduction == "token_mean":
        return (per_token_loss * mask).sum() / mask.sum().clamp(min=1)
    counts = mask.sum(dim=-1)
    per_example = (per_token_loss * mask).sum(dim=-1) / counts.clamp(min=1)
    n_items = (counts > 0).sum().to(per_token_loss.dtype).clamp(min=1)
    return per_example.sum() / n_items


def causal_lm_loss(
    logits: "Tensor",
    labels: "Tensor",
    reduction: str = "sample_mean",
    ignore_index: int = -100,
) -> "Tensor":
    """
    Next-token cross-entropy with the standard causal shift, reduced per ``reduction``.

    ``reduction="token_mean"`` reproduces Hugging Face's ``ForCausalLMLoss``
    (logits are upcast to float32, positions with ``ignore_index`` are skipped).

    Args:
        logits: [B, S, V] model logits.
        labels: [B, S] token ids with ``ignore_index`` at unsupervised positions.
    """
    validate_loss_reduction(reduction)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    batch, seq = shift_labels.shape
    per_token = F.cross_entropy(
        shift_logits.reshape(batch * seq, -1),
        shift_labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(batch, seq)
    return reduce_masked_loss(per_token, shift_labels != ignore_index, reduction)


def item_counts_from_labels(
    labels: "Tensor",
    reduction: str = "sample_mean",
    ignore_index: int = -100,
) -> "Tensor":
    """
    Number of loss items each row contributes under ``reduction``.

    - ``"sample_mean"``: 1 for rows with at least one supervised position, else 0.
    - ``"token_mean"``: the row's number of supervised positions.

    Args:
        labels: [B, S] with ``ignore_index`` at unsupervised positions.

    Returns:
        [B] integer tensor on the same device as ``labels``.
    """
    validate_loss_reduction(reduction)
    valid = labels != ignore_index
    if reduction == "token_mean":
        return valid.sum(dim=1)
    return valid.any(dim=1).long()
