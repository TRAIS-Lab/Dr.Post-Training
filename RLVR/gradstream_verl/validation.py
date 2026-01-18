"""
Validation gradient capture for VERL GRPO using seqloss-reward.

This module implements the validation gradient capture mechanism for gradient-based
data selection in GRPO training. It uses the seqloss-reward objective:
    L_val = -E[normalize(reward) * log π(y|x)]

Reference: RLHF/train/trainer.py lines 500-565, 1162-1324
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, Optional, Union
    from torch import Tensor

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .hook import GradientHookVerl

logger = logging.getLogger(__name__)


def _get_rank() -> int:
    """Get current rank for logging."""
    return dist.get_rank() if dist.is_initialized() else 0


def _get_world_size() -> int:
    """Get world size for logging."""
    return dist.get_world_size() if dist.is_initialized() else 1


def compute_token_logprobs(
    logits: Tensor,
    token_ids: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """
    Compute token-level log probabilities.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        token_ids: Target token IDs [batch, seq_len]
        temperature: Temperature for softmax (default 1.0)

    Returns:
        Token log probabilities [batch, seq_len]
    """
    if temperature != 1.0:
        logits = logits / temperature
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return torch.gather(log_probs, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)


def capture_seqloss_reward_gradients(
    model: torch.nn.Module,
    grad_hook: GradientHookVerl,
    val_batch: Dict[str, Tensor],
    micro_batch_size: Optional[int] = None,
    temperature: float = 1.0,
) -> float:
    """
    Capture validation gradients using seqloss-reward objective.

    The validation loss is computed as:
        L_val = -E[seq_scores * seq_log_probs]

    where seq_scores are normalized rewards: (R - mean) / (std + eps)

    Args:
        model: The policy model (actor)
        grad_hook: GradientHookVerl instance
        val_batch: Dictionary containing:
            - input_ids: [batch, seq_len] - full input (prompt + response)
            - attention_mask: [batch, seq_len]
            - response_ids: [batch, response_len] - response token IDs
            - response_mask: [batch, response_len] - mask for response tokens
            - rewards: [batch] - per-sequence rewards
        micro_batch_size: If set, process in mini-batches to save memory
        temperature: Temperature for log prob computation (default 1.0)

    Returns:
        Total validation loss (float)
    """
    from .debug_utils import get_debugger
    debugger = get_debugger()

    rewards = val_batch["rewards"]
    device = rewards.device
    rank = _get_rank()
    world_size = _get_world_size()

    # Normalize rewards: seq_scores = (R - μ) / (σ + ε)
    if rewards.numel() > 1:
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        seq_scores = (rewards - reward_mean) / (reward_std + 1e-8)
    else:
        reward_mean = rewards.mean()
        reward_std = torch.tensor(0.0, device=device)
        seq_scores = rewards - reward_mean

    # Start validation gradient capture
    grad_hook.start_val_capture()
    grad_hook.enable_hooks()
    model.train()

    batch_size = val_batch["input_ids"].shape[0]
    total_val_loss = 0.0
    total_tokens = val_batch["response_mask"].sum().item() if "response_mask" in val_batch else 0

    # Debug: Log validation start
    logger.info(
        f"[Val][Rank {rank}] Starting validation gradient capture: "
        f"batch_size={batch_size}, total_tokens={total_tokens}, "
        f"reward_mean={reward_mean.item():.4f}, reward_std={reward_std.item():.4f}"
    )
    debugger.log_validation_start(batch_size, int(total_tokens))

    # Process in micro-batches if specified
    if micro_batch_size is None:
        micro_batch_size = batch_size

    num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

    for mb_idx, start_idx in enumerate(range(0, batch_size, micro_batch_size)):
        end_idx = min(start_idx + micro_batch_size, batch_size)
        mb_slice = slice(start_idx, end_idx)
        mb_size = end_idx - start_idx

        # Forward pass
        outputs = model(
            input_ids=val_batch["input_ids"][mb_slice],
            attention_mask=val_batch["attention_mask"][mb_slice],
        )
        logits = outputs.logits  # [mb, seq_len, vocab_size]

        # Get response portion
        # Compute log probs for response tokens (shift by 1 for next-token prediction)
        # logits[:, :-1, :] predicts tokens at positions 1 to seq_len
        response_logits = logits[:, :-1, :]

        # Compute token log probs
        log_probs = F.log_softmax(response_logits.float() / temperature, dim=-1)

        # Get the response tokens (shifted by 1)
        response_ids = val_batch["response_ids"][mb_slice, 1:]  # [mb, response_len-1]
        response_mask = val_batch["response_mask"][mb_slice, 1:]  # [mb, response_len-1]

        # Handle case where response_ids might need alignment with logits
        # This depends on how the data is structured in verl
        if response_ids.shape[1] > response_logits.shape[1]:
            # Truncate if response is longer
            response_ids = response_ids[:, :response_logits.shape[1]]
            response_mask = response_mask[:, :response_logits.shape[1]]
        elif response_ids.shape[1] < response_logits.shape[1]:
            # Extract the relevant portion of logits
            response_logits = response_logits[:, -response_ids.shape[1]:, :]
            log_probs = F.log_softmax(response_logits.float() / temperature, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1,
            index=response_ids.unsqueeze(-1)
        ).squeeze(-1)  # [mb, response_len-1]

        # Sum to get sequence log prob
        seq_log_probs = (token_log_probs * response_mask.float()).sum(dim=1)  # [mb]

        # Validation loss: L = -E[seq_scores * seq_log_probs]
        mb_seq_scores = seq_scores[mb_slice]
        per_seq_loss = -(mb_seq_scores * seq_log_probs)

        # Normalize by full batch size (not micro-batch size) for proper gradient scaling
        mb_val_loss = per_seq_loss.sum() / batch_size
        mb_val_loss.backward()

        total_val_loss += mb_val_loss.item()

        # Debug: Log micro-batch loss
        logger.debug(
            f"[Val][Rank {rank}] Micro-batch {mb_idx+1}/{num_micro_batches}: "
            f"loss={mb_val_loss.item():.6f}, "
            f"seq_log_probs_mean={seq_log_probs.mean().item():.4f}"
        )
        debugger.log_validation_loss(mb_val_loss.item(), mb_idx)

    # End validation gradient capture
    grad_hook.end_val_capture()

    # Debug: Log captured gradients
    num_layers = grad_hook._val_cache.get_num_captured()
    logger.info(
        f"[Val][Rank {rank}] Validation gradient capture complete: "
        f"total_loss={total_val_loss:.6f}, num_layers={num_layers}"
    )

    # Log per-layer gradient statistics
    total_grad_norm_sq = 0.0
    for layer_idx in range(num_layers):
        val_grad = grad_hook.get_val_grad(layer_idx)
        if val_grad is not None:
            grad_norm = val_grad.norm().item()
            grad_mean = val_grad.mean().item()
            grad_std = val_grad.std().item()
            total_grad_norm_sq += grad_norm ** 2

            # Log first few and last few layers
            if layer_idx < 3 or layer_idx >= num_layers - 3:
                logger.info(
                    f"[Val][Rank {rank}]   Layer {layer_idx}: "
                    f"shape={tuple(val_grad.shape)}, "
                    f"norm={grad_norm:.6f}, mean={grad_mean:.8f}, std={grad_std:.6f}"
                )
            elif layer_idx == 3:
                logger.info(f"[Val][Rank {rank}]   ... ({num_layers - 6} more layers)")

    total_grad_norm = total_grad_norm_sq ** 0.5
    logger.info(f"[Val][Rank {rank}] Total validation gradient norm: {total_grad_norm:.6f}")

    debugger.log_validation_gradients(grad_hook, detailed=False)
    debugger.log_validation_end(total_val_loss)

    # Clear model gradients (we only wanted to capture them in the hook)
    model.zero_grad()

    return total_val_loss


def prepare_val_batch_from_rollout(
    rollout_batch: Dict[str, Tensor],
    rewards: Tensor,
) -> Dict[str, Tensor]:
    """
    Prepare validation batch from rollout data.

    This converts verl's rollout output format to the format expected
    by capture_seqloss_reward_gradients.

    Args:
        rollout_batch: Rollout data from verl containing:
            - input_ids: [batch, seq_len]
            - attention_mask: [batch, seq_len]
            - responses: [batch, response_len]
            - response_mask: [batch, response_len]
        rewards: [batch] - rewards from reward function

    Returns:
        val_batch dictionary for capture_seqloss_reward_gradients
    """
    return {
        "input_ids": rollout_batch["input_ids"],
        "attention_mask": rollout_batch["attention_mask"],
        "response_ids": rollout_batch["responses"],
        "response_mask": rollout_batch["response_mask"],
        "rewards": rewards,
    }
