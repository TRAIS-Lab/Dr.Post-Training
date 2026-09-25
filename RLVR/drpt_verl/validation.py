"""
Validation utilities for RLVR gradient-based selection.

This module provides shared utilities for reward normalization in validation
gradient capture. The normalization handles zero-variance cases to ensure
non-zero gradients for selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Dict, Tuple

import logging
import numpy as np
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

# =============================================================================
# Constants for Reward Normalization
# =============================================================================

# Threshold for detecting near-zero variance
VAR_THRESHOLD = 1e-6

# Baseline for binary rewards (0/1) when variance is zero
# For validation gradient capture, we need non-zero gradients even when
# rewards have zero/low variance. Using 0.5 as baseline ensures:
# - reward=0 → score=-0.5 (negative gradient signal)
# - reward=1 → score=+0.5 (positive gradient signal)
BINARY_BASELINE = 0.5


# =============================================================================
# Reward Normalization for Validation Gradients
# =============================================================================

def normalize_rewards_for_validation(
    rewards: torch.Tensor,
    norm_type: str = "batch",
    n_responses: int = 1,
    norm_adv_by_std: bool = True,
    device: torch.device = None,
) -> Tuple[torch.Tensor, Dict[str, any]]:
    """
    Normalize rewards for validation gradient capture.

    IMPORTANT: For validation gradient capture, we need non-zero gradients even when
    rewards have zero/low variance. This is different from training where zero advantage
    is intentional when all responses are equally good/bad.

    When variance is near zero, we use a fallback that subtracts BINARY_BASELINE (0.5)
    to ensure non-zero gradient signal.

    Args:
        rewards: Sequence-level rewards [batch_size]
        norm_type: "batch" for batch-level normalization, "grpo" for per-prompt-group
        n_responses: Number of responses per prompt (for GRPO grouping)
        norm_adv_by_std: Whether to divide by std (True for GRPO, False for Dr.GRPO)
        device: Device for tensors (defaults to rewards.device)

    Returns:
        Tuple of:
            - seq_scores: Normalized scores [batch_size]
            - stats: Dictionary with normalization statistics
    """
    if device is None:
        device = rewards.device

    batch_size = rewards.shape[0]
    stats = {
        'used_fallback': False,
        'zero_var_groups': 0,
        'total_groups': 0,
        'norm_type': norm_type,
    }

    if norm_type == "grpo" and n_responses > 1:
        # GRPO per-prompt-group normalization
        # Based on verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage
        num_prompts = batch_size // n_responses
        index = np.repeat(np.arange(num_prompts), n_responses)

        id2score = defaultdict(list)
        id2mean = {}
        id2std = {}
        epsilon = 1e-6

        with torch.no_grad():
            for i in range(batch_size):
                id2score[index[i]].append(rewards[i])

            stats['total_groups'] = len(id2score)

            for idx in id2score:
                if len(id2score[idx]) == 1:
                    # Single response per prompt - use global baseline
                    id2mean[idx] = torch.tensor(BINARY_BASELINE, device=device)
                    id2std[idx] = torch.tensor(1.0, device=device)
                    stats['zero_var_groups'] += 1
                elif len(id2score[idx]) > 1:
                    scores_tensor = torch.stack(id2score[idx])
                    id2mean[idx] = torch.mean(scores_tensor)
                    id2std[idx] = torch.std(scores_tensor)
                    if id2std[idx] < VAR_THRESHOLD:
                        stats['zero_var_groups'] += 1
                else:
                    raise ValueError(f"no score in prompt index: {idx}")

            seq_scores = torch.zeros_like(rewards)
            for i in range(batch_size):
                group_std = id2std[index[i]]
                group_mean = id2mean[index[i]]

                if group_std < VAR_THRESHOLD:
                    # Zero-variance fallback
                    seq_scores[i] = rewards[i] - BINARY_BASELINE
                    stats['used_fallback'] = True
                elif norm_adv_by_std:
                    seq_scores[i] = (rewards[i] - group_mean) / (group_std + epsilon)
                else:
                    seq_scores[i] = rewards[i] - group_mean

            if stats['zero_var_groups'] > 0:
                logger.info(
                    f"[Val Grad] GRPO normalization: {stats['zero_var_groups']}/{stats['total_groups']} "
                    f"groups had zero variance, using fallback (baseline={BINARY_BASELINE})"
                )
    else:
        # Batch-level normalization
        stats['total_groups'] = 1

        with torch.no_grad():
            if batch_size > 1:
                reward_std = rewards.std()
                reward_mean = rewards.mean()

                if reward_std < VAR_THRESHOLD:
                    seq_scores = rewards - BINARY_BASELINE
                    stats['used_fallback'] = True
                    stats['zero_var_groups'] = 1
                    logger.info(
                        f"[Val Grad] Batch normalization: zero variance detected (std={reward_std:.2e}), "
                        f"using fallback (baseline={BINARY_BASELINE})"
                    )
                else:
                    seq_scores = (rewards - reward_mean) / (reward_std + 1e-8)
            else:
                # Single sample
                seq_scores = rewards - BINARY_BASELINE
                stats['used_fallback'] = True
                stats['zero_var_groups'] = 1
                logger.info(
                    f"[Val Grad] Single sample validation batch, using fallback (baseline={BINARY_BASELINE})"
                )

    return seq_scores, stats
