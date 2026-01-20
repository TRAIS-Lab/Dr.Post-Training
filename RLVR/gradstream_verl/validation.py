"""
Validation gradient capture for VERL GRPO using seqloss-reward.

This module implements the validation gradient capture mechanism for gradient-based
data selection in GRPO training. It uses the seqloss-reward objective:
    L_val = -E[normalize(reward) * log π(y|x)]

Reference: RLHF/train/trainer.py lines 500-565, 1162-1324
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, Optional
    from torch import Tensor

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from .hook import GradientHookVerl


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
    n_responses: int = 1,
    norm_adv_by_std: bool = True,
    norm_type: str = "batch",
) -> float:
    """
    Capture validation gradients using seqloss-reward objective.

    The validation loss is computed as:
        L_val = -E[seq_scores * seq_log_probs]

    where seq_scores are normalized rewards.

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
        n_responses: Number of responses per prompt (for GRPO grouping, default 1)
        norm_adv_by_std: Whether to divide by std (True for GRPO, False for Dr.GRPO)
        norm_type: Normalization type - "batch" or "grpo"
            - "batch": batch-level normalization (rewards - mean) / std
            - "grpo": per-prompt-group GRPO normalization (matches training)

    Returns:
        Total validation loss (float)
    """
    rewards = val_batch["rewards"]

    # Normalize rewards based on norm_type
    if norm_type == "grpo" and n_responses > 1:
        # GRPO per-prompt-group normalization
        # This EXACTLY matches verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage
        batch_size = rewards.shape[0]
        num_prompts = batch_size // n_responses
        # Create index array for grouping (same as training)
        index = np.repeat(np.arange(num_prompts), n_responses)

        # EXACT copy of GRPO advantage computation logic
        scores = rewards  # Already sequence-level (not token-level like training)
        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}
        epsilon = 1e-6

        with torch.no_grad():
            bsz = scores.shape[0]
            for i in range(bsz):
                id2score[index[i]].append(scores[i])
            for idx in id2score:
                if len(id2score[idx]) == 1:
                    id2mean[idx] = torch.tensor(0.0)
                    id2std[idx] = torch.tensor(1.0)
                elif len(id2score[idx]) > 1:
                    scores_tensor = torch.stack(id2score[idx])
                    id2mean[idx] = torch.mean(scores_tensor)
                    id2std[idx] = torch.std(scores_tensor)
                else:
                    raise ValueError(f"no score in prompt index: {idx}")
            seq_scores = torch.zeros_like(scores)
            for i in range(bsz):
                if norm_adv_by_std:
                    seq_scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
                else:
                    seq_scores[i] = scores[i] - id2mean[index[i]]
    else:
        # Batch-level normalization (default and fallback)
        if rewards.numel() > 1:
            seq_scores = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            seq_scores = rewards - rewards.mean()

    # Start validation gradient capture
    grad_hook.start_val_capture()
    grad_hook.enable_hooks()
    model.train()

    batch_size = val_batch["input_ids"].shape[0]
    total_val_loss = 0.0

    # Process in micro-batches if specified
    if micro_batch_size is None:
        micro_batch_size = batch_size

    for start_idx in range(0, batch_size, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, batch_size)
        mb_slice = slice(start_idx, end_idx)

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

    # End validation gradient capture
    grad_hook.end_val_capture()

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
