"""
Streaming PPO Trainer for RLHF experiments.

This module implements PPO training with gradient streaming for data selection,
following the design patterns from the Snapshot paper (LDA-ORL).

Key Design:
- Separate validation and training passes (different loss functions)
- Validation: sequence-level reward-weighted log probs (f^seq)
- Training: PPO loss (policy + value + KL)
- KL penalty applied via reward shaping (before GAE), not as loss term

Methods:
- NA: Standard PPO (no selection)
- Streaming: Per-layer selection during backward (single-pass for training)
- GREATS: Global selection with validation gradients (two-pass for training)
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards: Per-token rewards (with KL penalty already applied) [batch, seq_len]
        values: Value estimates [batch, seq_len]
        mask: Valid token mask [batch, seq_len]
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        Tuple of (advantages, returns)
    """
    batch_size, seq_len = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device)

    # Next values (shifted by 1, with 0 at terminal)
    next_values = torch.cat([
        values[:, 1:],
        torch.zeros(batch_size, 1, device=values.device)
    ], dim=1)

    for t in reversed(range(seq_len)):
        delta = rewards[:, t] + gamma * next_values[:, t] * mask[:, t] - values[:, t]
        last_gae = delta + gamma * gae_lambda * mask[:, t] * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values
    return advantages, returns


class StreamingPPOTrainer:
    """
    PPO Trainer with gradient streaming for data selection.

    This trainer implements PPO with support for three training methods:
    - NA: Standard PPO (no selection)
    - Streaming: Per-layer selection during backward (uses stored val gradients)
    - GREATS: Global selection with validation gradients (two-pass)

    Key design choices for RLHF (vs SFT):
    1. Validation and training use DIFFERENT loss functions, so we can't merge batches
    2. Validation loss: -E[log π_θ(y|x) * Â(x,y)] (sequence-level, reward-weighted)
    3. Training loss: PPO clipped policy gradient + value loss
    4. KL penalty: Applied via reward shaping (subtracted from rewards before GAE)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module],
        reward_model: nn.Module,
        tokenizer,
        args,
        grad_hook=None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler=None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Policy model (with value head if using value function)
            ref_model: Reference model for KL penalty (None to use initial policy)
            reward_model: Reward model wrapper
            tokenizer: Tokenizer
            args: TrainingArguments
            grad_hook: GradientHook instance (None for NA/baseline)
            optimizer: Optimizer (created if None)
            lr_scheduler: LR scheduler (optional)
        """
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.args = args
        self.grad_hook = grad_hook
        self.lr_scheduler = lr_scheduler

        # Device
        self.device = next(model.parameters()).device

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # PPO hyperparameters
        self.kl_coef = args.kl_coef
        self.cliprange = args.cliprange
        self.cliprange_value = args.cliprange_value
        self.vf_coef = args.vf_coef
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.ppo_epochs = getattr(args, 'ppo_epochs', 4)  # PPO update epochs per batch

        # Selection configuration
        self.method = args.method
        self.selection_frac = args.selection_frac
        self.use_second_order = args.use_second_order
        self.val_loss_type = args.val_loss_type

        # Logging
        self._log_config()

    def _log_config(self):
        """Log trainer configuration."""
        logger.info("=" * 60)
        logger.info("StreamingPPOTrainer Configuration")
        logger.info(f"  Method: {self.method}")
        if self.method != "NA":
            logger.info(f"  Selection fraction: {self.selection_frac}")
            logger.info(f"  Validation loss type: {self.val_loss_type}")
            logger.info(f"  Second-order selection: {self.use_second_order}")
        logger.info(f"  KL coefficient: {self.kl_coef} (reward shaping)")
        logger.info(f"  Clip range: {self.cliprange}")
        logger.info(f"  PPO epochs per batch: {self.ppo_epochs}")
        logger.info(f"  Reference model: {'loaded (frozen)' if self.ref_model is not None else 'None (WARNING: KL penalty disabled!)'}")
        logger.info("=" * 60)

    @torch.no_grad()
    def generate_rollouts(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[str]]:
        """
        Generate responses for given queries.

        Args:
            query_ids: Query token IDs [batch, query_len]
            query_mask: Query attention mask [batch, query_len]

        Returns:
            Tuple of (response_ids, response_mask, response_texts)
        """
        self.model.eval()

        # Generation config
        gen_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "min_new_tokens": self.args.min_new_tokens,
            "do_sample": True,
            "temperature": self.args.temperature,
            "top_k": self.args.top_k if self.args.top_k > 0 else None,
            "top_p": self.args.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        # Generate
        outputs = self.model.generate(
            input_ids=query_ids,
            attention_mask=query_mask,
            **gen_kwargs,
        )

        # Extract responses (remove query prefix)
        query_len = query_ids.shape[1]
        response_ids = outputs[:, query_len:]

        # Create response mask
        response_mask = (response_ids != self.tokenizer.pad_token_id).long()

        # Decode responses
        response_texts = self.tokenizer.batch_decode(
            response_ids,
            skip_special_tokens=True,
        )

        self.model.train()
        return response_ids, response_mask, response_texts

    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_start: int,
    ) -> Tensor:
        """
        Compute log probabilities for response tokens.

        Args:
            model: Model to use
            input_ids: Full sequence [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            response_start: Index where response starts

        Returns:
            Log probabilities [batch, response_len]
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Handle model output format
        # AutoModelForCausalLMWithValueHead returns tuple: (logits, loss, values)
        # Regular model returns object with .logits attribute
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs.logits

        # Shift logits for next-token prediction
        logits = logits[:, response_start - 1:-1, :]
        labels = input_ids[:, response_start:]

        log_probs = F.log_softmax(logits.float(), dim=-1)
        selected_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1),
        ).squeeze(-1)

        return selected_log_probs

    def compute_values(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_start: int,
    ) -> Tensor:
        """
        Compute value estimates for response tokens using the value head.

        Args:
            input_ids: Full sequence [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            response_start: Index where response starts

        Returns:
            Value estimates [batch, response_len]
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # AutoModelForCausalLMWithValueHead returns tuple: (logits, loss, values)
        if isinstance(outputs, tuple) and len(outputs) >= 3:
            values_full = outputs[2]  # [batch, seq_len] or [batch, seq_len, 1]
            # Extract values for response tokens only
            if values_full.dim() == 3:
                values_full = values_full.squeeze(-1)
            values = values_full[:, response_start:]
        else:
            # Fallback if no value head (shouldn't happen)
            response_len = input_ids.shape[1] - response_start
            values = torch.zeros(input_ids.shape[0], response_len, device=self.device)

        return values

    def capture_validation_gradients(
        self,
        val_query_ids: Tensor,
        val_query_mask: Tensor,
    ) -> float:
        """
        Capture validation gradients using sequence-level attribution.

        Implements the sequence-level objective from Snapshot paper:
            f^seq(θ) = -E[log π_θ(y|x) * Â(x,y)]

        This gradient represents the "good direction" - moving towards
        higher reward sequences. The gradient points in the direction
        that increases log probability of high-reward responses.

        Args:
            val_query_ids: Validation query token IDs [batch, query_len]
            val_query_mask: Validation query attention mask [batch, query_len]

        Returns:
            Validation loss value
        """
        t_start = time.time()

        # Start validation capture mode
        self.grad_hook.start_val_capture()
        self.grad_hook.enable_hooks()

        # Generate responses from validation queries
        with torch.no_grad():
            response_ids, response_mask, response_texts = self.generate_rollouts(
                val_query_ids, val_query_mask
            )

            # Compute rewards for validation responses
            query_texts = self.tokenizer.batch_decode(val_query_ids, skip_special_tokens=True)
            rewards = self.reward_model.compute_rewards(query_texts, response_texts)

            # Normalize rewards to get advantages (z-normalization for variance reduction)
            if rewards.std() > 1e-8:
                rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                rewards_normalized = rewards - rewards.mean()

        # Now compute sequence-level log probability with gradients
        self.model.train()

        full_ids = torch.cat([val_query_ids, response_ids], dim=1)
        full_mask = torch.cat([val_query_mask, response_mask], dim=1)
        query_len = val_query_ids.shape[1]

        # Forward pass
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_mask,
        )

        # Handle model output format
        # AutoModelForCausalLMWithValueHead returns tuple: (logits, loss, values)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs.logits

        # Log probs for response tokens
        logits = logits[:, query_len - 1:-1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)

        # Sequence log probability (sum over response tokens)
        seq_log_probs = (token_log_probs * response_mask).sum(dim=1)

        # Sequence-level loss: -E[log π_θ(y|x) * Â(x,y)]
        # Negative because we want gradient towards higher reward
        val_loss = -(rewards_normalized * seq_log_probs).mean()

        # Backward - hooks capture gradients into val_grad_buffer
        val_loss.backward()

        # Clear optimizer gradients (we only wanted to capture compressed grads)
        self.optimizer.zero_grad()

        # End capture mode
        self.grad_hook.end_val_capture()

        logger.debug(
            f"Captured validation gradients (seq-level): "
            f"mean_reward={rewards.mean().item():.4f}, "
            f"loss={val_loss.item():.4f}, time={time.time() - t_start:.3f}s"
        )

        return val_loss.item()

    def compute_ppo_loss(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_values: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute PPO loss.

        Note: KL penalty is already applied to rewards via reward shaping
        (see train method), so we don't add it as a loss term here.

        Args:
            query_ids: Query token IDs [batch, query_len]
            response_ids: Response token IDs [batch, response_len]
            query_mask: Query attention mask
            response_mask: Response attention mask
            old_logprobs: Log probs from rollout [batch, response_len]
            advantages: GAE advantages (from KL-shaped rewards) [batch, response_len]
            returns: Returns for value loss [batch, response_len]
            old_values: Old value estimates [batch, response_len]

        Returns:
            Tuple of (total_loss, stats_dict)
        """
        batch_size = query_ids.shape[0]
        query_len = query_ids.shape[1]

        # Concatenate query and response
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = torch.cat([query_mask, response_mask], dim=1)

        # Forward pass - model returns (logits, loss, values) for AutoModelForCausalLMWithValueHead
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Handle model output format
        # AutoModelForCausalLMWithValueHead returns tuple: (logits, loss, values)
        if isinstance(outputs, tuple):
            logits = outputs[0]
            values_full = outputs[2]  # [batch, seq_len]
        else:
            # Fallback for standard model (shouldn't happen with value head)
            logits = outputs.logits
            values_full = torch.zeros(batch_size, input_ids.shape[1], device=self.device)

        # Extract values for response tokens only
        values = values_full[:, query_len:].squeeze(-1) if values_full.dim() == 3 else values_full[:, query_len:]

        # New log probs
        logits_for_probs = logits[:, query_len - 1:-1, :]
        log_probs = F.log_softmax(logits_for_probs.float(), dim=-1)
        new_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)

        # PPO policy loss with clipping
        logprob_diff = new_logprobs - old_logprobs
        ratio = torch.exp(logprob_diff)
        clipped_ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # Value loss with clipping
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.cliprange_value,
            self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        vf_loss = (vf_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # Total loss (no KL term - it's in the rewards/advantages already)
        total_loss = pg_loss + self.vf_coef * vf_loss

        # Stats for logging
        with torch.no_grad():
            clipfrac = ((ratio - 1).abs() > self.cliprange).float().mean().item()
            approx_kl = (0.5 * (new_logprobs - old_logprobs) ** 2 * response_mask).sum()
            approx_kl = approx_kl / response_mask.sum()

        stats = {
            "loss/total": total_loss.item(),
            "loss/policy": pg_loss.item(),
            "loss/value": vf_loss.item(),
            "policy/approx_kl": approx_kl.item(),
            "policy/clipfrac": clipfrac,
            "policy/ratio_mean": ratio.mean().item(),
            "values/mean": values.mean().item(),
        }

        return total_loss, stats

    def training_step(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_values: Tensor,
    ) -> Dict[str, float]:
        """
        Perform one training step with optional selection.

        Handles all three methods:
        - NA: Standard PPO update
        - Streaming: Per-layer selection during backward (uses stored val grads)
        - GREATS: Global selection (two-pass for training)

        Args:
            query_ids, response_ids: Token IDs
            query_mask, response_mask: Attention masks
            old_logprobs: Log probs from rollout
            advantages: GAE advantages (from KL-shaped rewards)
            returns: Returns
            old_values: Old value estimates

        Returns:
            Training statistics dictionary
        """
        batch_size = query_ids.shape[0]
        lr = self.optimizer.param_groups[0]["lr"]

        # ========================================
        # Method: NA (baseline, no selection)
        # ========================================
        if self.method == "NA":
            self.optimizer.zero_grad()

            # Disable hooks for baseline
            if self.grad_hook is not None:
                self.grad_hook.disable_hooks()

            loss, stats = self.compute_ppo_loss(
                query_ids, response_ids, query_mask, response_mask,
                old_logprobs, advantages, returns, old_values,
            )

            loss.backward()

            # Gradient clipping (disabled if max_grad_norm=0 or None)
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm,
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.grad_hook is not None:
                self.grad_hook.enable_hooks()

            return stats

        # ========================================
        # Method: Streaming (per-layer selection with stored val grads)
        # ========================================
        if self.method == "Streaming":
            self.optimizer.zero_grad()

            # Setup selection using pre-captured validation gradients
            self.grad_hook.setup_selection_with_stored_val(
                train_batch_size=batch_size,
                selection_method="Streaming",
                selection_frac=self.selection_frac,
                lr=lr,
                compute_scores_only=False,  # Per-layer selection during backward
                use_second_order=self.use_second_order,
            )

            loss, stats = self.compute_ppo_loss(
                query_ids, response_ids, query_mask, response_mask,
                old_logprobs, advantages, returns, old_values,
            )

            # Backward with per-layer selection using stored val grads
            loss.backward()

            # Get selection stats
            if self.grad_hook.selection_state is not None:
                n_selected = self.grad_hook.selection_state.num_selected
                stats["selection/n_selected"] = n_selected
                stats["selection/frac"] = n_selected / batch_size

            self.grad_hook.clear_selection()

            # Optimizer step (uses MeSO if compression enabled)
            self.optimizer.step()
            self.optimizer.zero_grad()

            return stats

        # ========================================
        # Method: GREATS (global selection, two-pass for training)
        # ========================================
        if self.method == "GREATS":
            # Pass 1: Compute selection scores using stored val grads
            self.optimizer.zero_grad()
            self.grad_hook.setup_selection_with_stored_val(
                train_batch_size=batch_size,
                selection_method="GREATS",
                selection_frac=self.selection_frac,
                lr=lr,
                compute_scores_only=True,  # Only accumulate scores
                use_second_order=self.use_second_order,
            )

            loss_for_scoring, _ = self.compute_ppo_loss(
                query_ids, response_ids, query_mask, response_mask,
                old_logprobs, advantages, returns, old_values,
            )

            loss_for_scoring.backward()

            # Get selected indices
            selected_indices = self.grad_hook.selection_state.get_selected_indices()
            n_selected = len(selected_indices)

            self.grad_hook.clear_selection()

            # Pass 2: Full gradients on selected samples
            self.grad_hook.disable_hooks()
            self.optimizer.zero_grad()

            # Filter to selected samples
            query_ids_sel = query_ids[selected_indices]
            response_ids_sel = response_ids[selected_indices]
            query_mask_sel = query_mask[selected_indices]
            response_mask_sel = response_mask[selected_indices]
            old_logprobs_sel = old_logprobs[selected_indices]
            advantages_sel = advantages[selected_indices]
            returns_sel = returns[selected_indices]
            old_values_sel = old_values[selected_indices]

            loss, stats = self.compute_ppo_loss(
                query_ids_sel, response_ids_sel, query_mask_sel, response_mask_sel,
                old_logprobs_sel, advantages_sel, returns_sel, old_values_sel,
            )

            loss.backward()

            # Gradient clipping (disabled if max_grad_norm=0 or None)
            if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm,
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Re-enable hooks for next step
            self.grad_hook.enable_hooks()

            # Add selection stats
            stats["selection/n_selected"] = n_selected
            stats["selection/n_total"] = batch_size
            stats["selection/frac"] = n_selected / batch_size

            return stats

        raise ValueError(f"Unknown method: {self.method}")

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_dataloader: DataLoader for training prompts
            val_dataloader: DataLoader for validation data (required for selection)
            num_epochs: Number of training epochs
            max_steps: Maximum steps (None = no limit)
            log_interval: Steps between logging

        Returns:
            Dictionary of training history
        """
        logger.info(f"Starting training with method={self.method}")

        if self.method != "NA" and val_dataloader is None:
            raise ValueError("val_dataloader required for selection methods")

        history = {"loss": [], "reward": [], "kl": []}
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_stats = []

            # Prepare validation queries for gradient capture
            val_query_ids = None
            val_query_mask = None
            if self.method != "NA" and val_dataloader is not None:
                val_batch = next(iter(val_dataloader))
                val_query_ids = val_batch["input_ids"]
                if isinstance(val_query_ids, list):
                    # Left-pad to same length (required for decoder-only generation)
                    max_len = max(t.shape[0] if t.dim() == 1 else t.shape[1] for t in val_query_ids)
                    val_query_ids = torch.stack([
                        F.pad(t.flatten(), (max_len - t.numel(), 0), value=self.tokenizer.pad_token_id)
                        for t in val_query_ids
                    ]).to(self.device)
                else:
                    val_query_ids = val_query_ids.to(self.device)
                val_query_mask = (val_query_ids != self.tokenizer.pad_token_id).long()

            # For GREATS: capture validation gradients once per epoch
            if self.method == "GREATS" and val_query_ids is not None:
                self.capture_validation_gradients(
                    val_query_ids, val_query_mask
                )

            # Training loop
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
            for batch in pbar:
                # Get queries (left-pad for decoder-only generation)
                query_ids = batch["input_ids"]
                if isinstance(query_ids, list):
                    max_len = max(t.shape[0] if t.dim() == 1 else t.shape[1] for t in query_ids)
                    query_ids = torch.stack([
                        F.pad(t.flatten(), (max_len - t.numel(), 0), value=self.tokenizer.pad_token_id)
                        for t in query_ids
                    ]).to(self.device)
                else:
                    query_ids = query_ids.to(self.device)

                query_mask = (query_ids != self.tokenizer.pad_token_id).long()

                # Generate rollouts
                with torch.no_grad():
                    response_ids, response_mask, response_texts = self.generate_rollouts(
                        query_ids, query_mask
                    )

                    # Compute rewards from reward model
                    query_texts = self.tokenizer.batch_decode(query_ids, skip_special_tokens=True)
                    raw_rewards = self.reward_model.compute_rewards(query_texts, response_texts)

                    # Create per-token reward tensor (sparse - only at last token)
                    rewards = torch.zeros_like(response_mask, dtype=torch.float, device=self.device)
                    for i in range(len(response_texts)):
                        last_idx = response_mask[i].sum().item() - 1
                        if last_idx >= 0:
                            rewards[i, last_idx] = raw_rewards[i]

                    # Compute old log probs and ref log probs
                    full_ids = torch.cat([query_ids, response_ids], dim=1)
                    full_mask = torch.cat([query_mask, response_mask], dim=1)
                    query_len = query_ids.shape[1]

                    old_logprobs = self.compute_log_probs(
                        self.model, full_ids, full_mask, query_len
                    )

                    # Compute reference log probs for KL penalty
                    if self.ref_model is not None:
                        ref_logprobs = self.compute_log_probs(
                            self.ref_model, full_ids, full_mask, query_len
                        )
                    else:
                        ref_logprobs = old_logprobs.detach()

                    # ========================================
                    # KL PENALTY VIA REWARD SHAPING
                    # ========================================
                    # This is the correct implementation per the Snapshot paper
                    # and standard PPO for RLHF. The KL penalty is subtracted
                    # from rewards BEFORE computing GAE, not added as a loss term.
                    kl_penalty = old_logprobs - ref_logprobs  # KL(π||π_ref) per token
                    rewards = rewards - self.kl_coef * kl_penalty * response_mask.float()

                    # Compute value estimates from value head
                    old_values = self.compute_values(full_ids, full_mask, query_len)

                    # Compute advantages and returns using KL-shaped rewards
                    advantages, returns = compute_gae(
                        rewards, old_values, response_mask.float(),
                        self.gamma, self.gae_lambda
                    )

                    # Normalize advantages
                    adv_mean = (advantages * response_mask).sum() / response_mask.sum().clamp(min=1)
                    adv_var = ((advantages - adv_mean) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1)
                    advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

                # Capture validation gradients per-step for Streaming
                if self.method == "Streaming" and val_query_ids is not None:
                    self.capture_validation_gradients(
                        val_query_ids, val_query_mask
                    )

                # PPO epochs: multiple updates per rollout batch
                # old_logprobs stays fixed, new_logprobs recomputed each epoch
                for ppo_epoch in range(self.ppo_epochs):
                    stats = self.training_step(
                        query_ids, response_ids, query_mask, response_mask,
                        old_logprobs.detach(), advantages.detach(), returns.detach(), old_values.detach(),
                    )

                # Add reward and KL stats
                stats["reward/mean"] = raw_rewards.mean().item()
                stats["reward/std"] = raw_rewards.std().item()
                stats["objective/kl"] = kl_penalty.mean().item()

                epoch_stats.append(stats)
                global_step += 1

                # LR scheduler step (must be per-step, not per-epoch, for warmup to work)
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{stats['loss/total']:.4f}",
                    "reward": f"{stats['reward/mean']:.2f}",
                })

                # Logging
                if global_step % log_interval == 0:
                    avg_stats = {
                        k: float(np.mean([s[k] for s in epoch_stats[-log_interval:]]))
                        for k in epoch_stats[-1].keys()
                    }
                    logger.info(f"Step {global_step}: {avg_stats}")

                    # Update history
                    history["loss"].append(avg_stats["loss/total"])
                    history["reward"].append(avg_stats["reward/mean"])
                    history["kl"].append(avg_stats["objective/kl"])

                # Check max steps
                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Reached max_steps ({max_steps})")
                    break

            if max_steps is not None and global_step >= max_steps:
                break

        logger.info("Training complete")
        return history

    def save_model(self, output_dir: str):
        """Save model to directory."""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
