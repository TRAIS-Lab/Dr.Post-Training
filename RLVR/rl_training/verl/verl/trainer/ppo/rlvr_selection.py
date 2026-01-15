"""
RLVR Online Data Selection with GREATS and Streaming.

This module provides gradient-based online data selection for RLVR training:
- GREATS: Per-step global selection using gradient similarity
- Streaming: Per-step, per-layer selection using gradient similarity

Key Difference from Baseline DOTS:
- Baseline DOTS: Uses difficulty to select training samples offline (per-epoch)
- GREATS/Streaming: Uses difficulty to select the *validation set* dynamically,
  then applies gradient similarity selection for training samples (per-step)

The validation set represents "what we want the model to learn" - medium difficulty
samples are ideal curriculum targets. Gradient similarity then selects training
samples that best align with improving on these curriculum-relevant validation samples.

Usage:
    # Initialize manager (once per training run)
    selection_manager = RLVROnlineSelectionManager(
        model=actor_module,
        config=config,
        device='cuda',
    )

    # Per rollout: update validation set based on difficulty
    selection_manager.update_validation_set(
        batch=rollout_batch,
        advantages=advantages,
        response_mask=response_mask,
    )

    # Per training step: perform selection
    # For GREATS:
    selected_indices = selection_manager.select_batch_greats(
        batch=train_batch,
        advantages=advantages,
        response_mask=response_mask,
    )

    # For Streaming:
    selection_manager.setup_streaming(batch=train_batch, ...)
    loss.backward()  # Selection happens during backward
    selection_manager.cleanup_streaming()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, List, Optional, Tuple
if TYPE_CHECKING:
    from torch import Tensor
    from omegaconf import DictConfig

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def get_trainable_linear_layers(model: nn.Module) -> List[str]:
    """
    Get names of trainable Linear layers in the model.

    For FSDP models, we look at the wrapped module.
    Excludes embedding and lm_head layers.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    # Get the actual module (unwrap FSDP if needed)
    if isinstance(model, FSDP):
        # For FSDP, we need to be careful about how we access modules
        # The layer names should still work through named_modules
        pass

    layer_names = []
    exclude_patterns = ['embed', 'lm_head', 'norm', 'wte', 'wpe']

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check exclusion patterns
            if any(pattern in name.lower() for pattern in exclude_patterns):
                continue
            # Check if trainable
            if any(p.requires_grad for p in module.parameters()):
                layer_names.append(name)

    return layer_names


def compute_goldilocks_scores(
    difficulties: Tensor,
    alpha: float = 0.5,
) -> Tensor:
    """
    Compute Goldilocks zone scores for validation set selection.

    Score = -|difficulty - alpha|
    Higher score = closer to target difficulty (medium difficulty preferred).

    Args:
        difficulties: [batch_size] difficulty scores (e.g., mean rewards per sample)
        alpha: Target difficulty (0-1, default 0.5 = medium difficulty)

    Returns:
        [batch_size] selection scores (higher = more suitable for validation)
    """
    return -torch.abs(difficulties - alpha)


def select_validation_indices(
    difficulties: Tensor,
    num_to_select: int,
    alpha: float = 0.5,
    tau: float = 0.1,
    mode: str = 'soft',
) -> Tensor:
    """
    Select validation set indices based on difficulty (Goldilocks criterion).

    Args:
        difficulties: [batch_size] difficulty scores
        num_to_select: Number of samples to select for validation
        alpha: Target difficulty level
        tau: Temperature for softmax sampling (lower = greedier)
        mode: 'soft' (probabilistic) or 'hard' (deterministic top-k)

    Returns:
        [num_to_select] selected indices for validation set
    """
    scores = compute_goldilocks_scores(difficulties, alpha)

    if mode == 'hard':
        # Deterministic: select top-k by score (closest to alpha)
        _, indices = torch.topk(scores, k=num_to_select)
    else:
        # Soft: probabilistic sampling based on scores
        logits = scores / tau
        logits = logits - logits.max()  # Numerical stability
        probs = torch.softmax(logits, dim=0)
        indices = torch.multinomial(probs, num_samples=num_to_select, replacement=False)

    return indices.sort()[0]  # Sort for cache locality


class RLVROnlineSelectionManager:
    """
    Manager for RLVR online data selection using GREATS or Streaming.

    This integrates gradient-based selection with curriculum-aware validation:
    1. Select validation samples based on DOTS difficulty criterion
    2. Capture validation gradients using sequence-level advantage target
    3. Use gradient similarity to select training samples per step
    """

    def __init__(
        self,
        model: nn.Module,
        config: "DictConfig",
        device: str = 'cuda',
    ):
        """
        Initialize the selection manager.

        Args:
            model: The actor model (can be FSDP wrapped)
            config: Training configuration with selection parameters
            device: Device for tensor operations
        """
        self.model = model
        self.config = config
        self.device = device

        # Selection parameters from config
        self.selection_method = getattr(config.data, 'online_selection_method', 'NA')
        self.alpha = getattr(config.data, 'alpha', 0.5)
        self.tau = getattr(config.data, 'tau', 0.1)
        self.val_selection_mode = getattr(config.data, 'val_selection_mode', 'soft')
        self.val_ratio = getattr(config.data, 'val_ratio', 0.2)  # Fraction of batch for validation
        self.train_selection_ratio = getattr(config.data, 'train_selection_ratio', 0.5)
        self.use_second_order = getattr(config.data, 'use_second_order', False)

        # GradientHook for gradient-based selection (lazy initialization)
        self.grad_hook: Optional["GradientHook"] = None
        self._hook_initialized = False

        # Cached validation gradients
        self._val_grad_buffer: List[Optional[Tensor]] = []
        self._val_batch_size: int = 0

        # Selection statistics
        self.selection_stats: Dict[str, Any] = {}

        # Only initialize if we're using gradient-based selection
        if self.selection_method in ['GREATS', 'Streaming']:
            self._initialize_hook()

        logger.info(
            f"Initialized RLVROnlineSelectionManager: method={self.selection_method}, "
            f"alpha={self.alpha}, tau={self.tau}, val_ratio={self.val_ratio}"
        )

    def _initialize_hook(self) -> None:
        """Initialize GradientHook for gradient-based selection."""
        if self._hook_initialized:
            return

        try:
            from gradstream.hook import GradientHook
        except ImportError:
            logger.error("gradstream not found. Please install gradstream for GREATS/Streaming selection.")
            self.selection_method = 'NA'
            return

        # Get trainable layer names
        layer_names = get_trainable_linear_layers(self.model)

        if len(layer_names) == 0:
            logger.warning("No trainable Linear layers found, falling back to NA selection")
            self.selection_method = 'NA'
            return

        # Create GradientHook
        self.grad_hook = GradientHook(
            model=self.model,
            layer_names=layer_names,
            device=self.device,
        )

        self._hook_initialized = True
        logger.info(f"Initialized GradientHook with {len(layer_names)} layers")

    def is_enabled(self) -> bool:
        """Check if gradient-based selection is enabled."""
        return self.selection_method in ['GREATS', 'Streaming'] and self._hook_initialized

    # =========================================================================
    # Validation Set Management
    # =========================================================================

    def compute_sample_difficulties(
        self,
        rewards: Tensor,
        uids: Optional[Any] = None,
    ) -> Tensor:
        """
        Compute per-sample difficulty from rewards.

        For RLVR, difficulty is based on solve rate:
        - All zeros (solve_none) -> difficulty = 0 (too hard)
        - All ones (solve_all) -> difficulty = 1 (too easy)
        - Mixed -> difficulty in (0, 1) (medium = ideal)

        Args:
            rewards: [batch_size] or [batch_size, response_len] reward tensor
            uids: Optional unique IDs for grouping (for multi-response per prompt)

        Returns:
            [num_prompts] difficulty scores
        """
        # Sum rewards over sequence if needed
        if rewards.dim() > 1:
            rewards = rewards.sum(dim=-1)

        if uids is None:
            # Each sample is independent
            # Normalize to [0, 1]
            rewards_min = rewards.min()
            rewards_max = rewards.max()
            if rewards_max - rewards_min > 1e-8:
                difficulties = (rewards - rewards_min) / (rewards_max - rewards_min)
            else:
                difficulties = torch.ones_like(rewards) * 0.5
            return difficulties

        # Group by uid and compute mean reward per prompt
        import numpy as np
        unique_uids = np.unique(uids)
        difficulties = []

        for uid in unique_uids:
            mask = (uids == uid)
            uid_rewards = rewards[mask]
            # Mean reward as difficulty proxy
            difficulties.append(uid_rewards.mean().item())

        difficulties = torch.tensor(difficulties, device=rewards.device, dtype=rewards.dtype)

        # Normalize to [0, 1]
        d_min, d_max = difficulties.min(), difficulties.max()
        if d_max - d_min > 1e-8:
            difficulties = (difficulties - d_min) / (d_max - d_min)
        else:
            difficulties = torch.ones_like(difficulties) * 0.5

        return difficulties

    def select_validation_set(
        self,
        batch_size: int,
        difficulties: Tensor,
    ) -> Tensor:
        """
        Select validation set indices based on DOTS difficulty criterion.

        Args:
            batch_size: Total batch size
            difficulties: [batch_size] difficulty scores

        Returns:
            [num_val] indices of samples to use as validation set
        """
        num_val = max(1, int(batch_size * self.val_ratio))

        val_indices = select_validation_indices(
            difficulties=difficulties,
            num_to_select=num_val,
            alpha=self.alpha,
            tau=self.tau,
            mode=self.val_selection_mode,
        )

        self._val_batch_size = len(val_indices)

        # Track stats
        selected_difficulties = difficulties[val_indices]
        self.selection_stats['val/num_selected'] = len(val_indices)
        self.selection_stats['val/difficulty_mean'] = selected_difficulties.mean().item()
        self.selection_stats['val/difficulty_std'] = selected_difficulties.std().item() if len(selected_difficulties) > 1 else 0

        return val_indices

    # =========================================================================
    # Validation Gradient Capture
    # =========================================================================

    def capture_validation_gradients(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        responses: Tensor,
        advantages: Tensor,
        response_mask: Tensor,
        val_indices: Tensor,
        temperature: float = 1.0,
    ) -> None:
        """
        Capture validation gradients using sequence-level advantage target.

        Target function: L_val = -E[log π_θ(y|x) * A(x,y)]

        This caches the validation gradients for use in training selection.

        Args:
            input_ids: [batch_size, seq_len] full input (prompt + response)
            attention_mask: [batch_size, seq_len] attention mask
            responses: [batch_size, response_len] response tokens
            advantages: [batch_size, response_len] per-token advantages
            response_mask: [batch_size, response_len] response mask
            val_indices: [num_val] indices of validation samples
            temperature: Temperature for log prob computation
        """
        if not self.is_enabled():
            return

        # Extract validation samples
        val_input_ids = input_ids[val_indices]
        val_attention_mask = attention_mask[val_indices]
        val_responses = responses[val_indices]
        val_advantages = advantages[val_indices]
        val_response_mask = response_mask[val_indices]

        # Extract advantage at the LAST token (Â_{-1})
        # This matches RLHF: use last-token advantage as the sequence-level signal
        last_token_indices = val_response_mask.float().sum(dim=1) - 1  # [batch_size]
        seq_advantages = val_advantages[
            torch.arange(val_advantages.size(0), device=val_advantages.device),
            last_token_indices.long()
        ]

        # Start validation gradient capture
        # Use full mode (not factorized) for efficiency with larger batches
        self.grad_hook.start_val_capture(use_factorized=False)
        self.grad_hook.enable_hooks()

        # Forward pass to get log probabilities
        response_length = val_responses.size(1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=val_input_ids,
                attention_mask=val_attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Get log probs for response tokens
            # Logits at position i predict token at i+1
            logits = logits[:, -response_length - 1:-1, :]  # [val_batch, response_len, vocab]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=val_responses.unsqueeze(-1)
            ).squeeze(-1)  # [val_batch, response_len]

            # Sequence log probability (sum over valid response tokens)
            seq_log_probs = (token_log_probs * val_response_mask.float()).sum(dim=1)

            # Validation loss: -E[log π_θ(y|x) * A(x,y)]
            # We want samples with high advantage to have high log prob
            per_seq_loss = -(seq_advantages * seq_log_probs)
            val_loss = per_seq_loss.mean()

            # Backward to capture gradients
            val_loss.backward()

        # End capture and cleanup
        self.model.zero_grad()  # Clear validation gradients from parameters
        self.grad_hook.end_val_capture()

        logger.debug(f"Captured validation gradients for {len(val_indices)} samples")

    # =========================================================================
    # GREATS Selection (Global, Two-Pass)
    # =========================================================================

    def select_batch_greats(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        responses: Tensor,
        advantages: Tensor,
        response_mask: Tensor,
        old_log_probs: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """
        Select training samples using GREATS (global gradient-based selection).

        Pass 1: Compute gradient similarity scores across all layers
        Pass 2: Return selected indices for actual training

        Args:
            input_ids: [batch_size, seq_len] input tokens
            attention_mask: [batch_size, seq_len] attention mask
            responses: [batch_size, response_len] response tokens
            advantages: [batch_size, response_len] advantages
            response_mask: [batch_size, response_len] response mask
            old_log_probs: [batch_size, response_len] old policy log probs
            temperature: Temperature for log prob computation

        Returns:
            [num_selected] indices of selected training samples
        """
        if not self.is_enabled():
            # Return all indices if selection not enabled
            return torch.arange(input_ids.size(0), device=self.device)

        batch_size = input_ids.size(0)
        num_to_select = max(1, int(batch_size * self.train_selection_ratio))

        # Setup GREATS state for Pass 1 (score accumulation)
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method='GREATS',
            frac=self.train_selection_ratio,
            lr=1.0,  # We handle scaling elsewhere
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode='filtering',
        )
        self.grad_hook.enable_hooks()

        # Pass 1: Forward/backward to accumulate scores
        self.model.zero_grad()

        response_length = responses.size(1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = outputs.logits

            if temperature != 1.0:
                logits = logits / temperature

            # Compute log probs
            logits = logits[:, -response_length - 1:-1, :]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=responses.unsqueeze(-1)
            ).squeeze(-1)

            # PPO-style policy loss for scoring
            ratio = torch.exp(token_log_probs - old_log_probs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 0.8, 1.2)  # Clip ratio
            pg_loss = torch.max(pg_loss1, pg_loss2)

            # Masked mean loss
            loss = (pg_loss * response_mask.float()).sum() / response_mask.float().sum()

            # Backward to accumulate scores
            loss.backward()

        # Get globally selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        selected_indices = selected_indices.sort()[0]  # Sort for cache locality

        # Cleanup after Pass 1
        self.grad_hook.clear_selection()
        self.grad_hook.disable_hooks()  # Disable hooks for Pass 2 (actual training)

        # Track stats
        self.selection_stats['train/num_selected'] = len(selected_indices)
        self.selection_stats['train/selection_ratio'] = len(selected_indices) / batch_size

        return selected_indices

    # =========================================================================
    # Streaming Selection (Per-Layer, Single-Pass)
    # =========================================================================

    def setup_streaming(
        self,
        batch_size: int,
        labels: Optional[Tensor] = None,
    ) -> None:
        """
        Setup Streaming selection state before training forward pass.

        Args:
            batch_size: Number of training samples
            labels: Optional labels for token-based gradient scaling
        """
        if not self.is_enabled():
            return

        # Setup streaming state with stored validation gradients
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method='Streaming',
            frac=self.train_selection_ratio,
            lr=1.0,
            compute_scores_only=False,
            use_second_order=self.use_second_order,
            selection_mode='filtering',
        )
        self.grad_hook.enable_hooks()

        # Set token counts for gradient scaling if labels provided
        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

    def cleanup_streaming(self) -> None:
        """Cleanup after streaming backward pass."""
        if not self.is_enabled():
            return

        # Get selection stats from streaming state
        sel_state = self.grad_hook.selection_state
        if hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
            n_selected_list = [n for _, n in sel_state._layer_selections]
            self.selection_stats['train/mean_selected'] = sum(n_selected_list) / len(n_selected_list)
            self.selection_stats['train/min_selected'] = min(n_selected_list)

        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_selection_stats(self) -> Dict[str, Any]:
        """Get selection statistics from last selection."""
        return self.selection_stats.copy()

    def clear_validation_cache(self) -> None:
        """Clear cached validation gradients."""
        if self.grad_hook is not None:
            self.grad_hook.clear_val_buffer()
        self._val_batch_size = 0

    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()
