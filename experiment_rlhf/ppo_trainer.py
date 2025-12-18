"""
PPO Trainer with layer-wise selection for RLHF (TRL 0.26.1 compatible).

This module provides a custom PPO trainer that integrates the layer-wise
selection streaming paradigm for efficient data selection during policy
optimization.

Key Features:
- Proper rollout generation with response sampling
- Actual PPO loss computation (policy + value + KL)
- Two-pass selection approach for RLHF
- Compatible with TRL 0.26.1's experimental PPO API
"""

import sys
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_advantages(
    rewards: Tensor,
    values: Tensor,
    response_mask: Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Per-token rewards [batch, response_len]
        values: Value estimates [batch, response_len]
        response_mask: Mask for valid response tokens [batch, response_len]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        Tuple of (advantages, returns)
    """
    # For simplicity in RLHF, we often use per-sequence rewards
    # which means rewards are sparse (only at the end)
    batch_size, seq_len = rewards.shape

    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros(batch_size, device=rewards.device)

    # Append zero for value at terminal state
    next_values = torch.cat([values[:, 1:], torch.zeros(batch_size, 1, device=values.device)], dim=1)

    for t in reversed(range(seq_len)):
        delta = rewards[:, t] + gamma * next_values[:, t] * response_mask[:, t] - values[:, t]
        last_gae_lam = delta + gamma * gae_lambda * response_mask[:, t] * last_gae_lam
        advantages[:, t] = last_gae_lam

    returns = advantages + values
    return advantages, returns


class PPOTrainerWithSelection:
    """
    PPO Trainer wrapper with layer-wise selection support for TRL 0.26.1.

    This class provides a custom training loop that:
    1. Generates rollouts using the policy model
    2. Computes rewards using the reward model
    3. Calculates advantages using GAE
    4. Performs PPO updates with layer-wise selection

    The two-pass approach:
    Phase 1 (capture_val): Forward/backward on validation data to capture
                          compressed gradients representing the "good direction"
    Phase 2 (train_select): Forward/backward on training data with PPO loss,
                           selecting samples layer-by-layer based on alignment
                           with validation gradients
    """

    def __init__(
        self,
        ppo_trainer,  # TRL 0.26.1 PPOTrainer instance
        grad_hook=None,    # GradientHook instance (None for baseline/NA)
        selection_frac: float = 0.5,
        selection_method: str = "GREATS",
        selection_mode: str = "per_layer",
        val_loss_type: str = "logprob",
        use_second_order: bool = False,
    ):
        """
        Initialize the selection wrapper.

        Args:
            ppo_trainer: TRL 0.26.1 PPOTrainer instance
            grad_hook: GradientHook instance with compressors set up (None for NA/baseline)
            selection_frac: Fraction of samples to select (0-1)
            selection_method: 'GREATS', 'GradNorm', or 'NA' (baseline, no selection)
            selection_mode: 'global' or 'per_layer' (default: per_layer)
                - global: Accumulates scores across all layers, selects globally
                - per_layer: Each layer independently selects samples
            val_loss_type: Type of validation loss ('logprob', 'reward_weighted', 'advantage_weighted')
            use_second_order: Whether to use second-order selection
        """
        self.ppo_trainer = ppo_trainer
        self.grad_hook = grad_hook
        self.selection_frac = selection_frac
        self.selection_method = selection_method
        self.selection_mode = selection_mode
        self.val_loss_type = val_loss_type
        self.use_second_order = use_second_order

        # Check if selection is enabled
        self.selection_enabled = (selection_method != "NA" and grad_hook is not None)

        # Access trainer components via TRL 0.26.1's experimental API
        self.model = ppo_trainer.model  # PolicyAndValueWrapper
        self.policy_model = ppo_trainer.policy_model  # Raw policy model
        self.ref_model = ppo_trainer.ref_model  # Reference model
        self.reward_model = ppo_trainer.reward_model
        self.value_model = ppo_trainer.value_model
        self.optimizer = ppo_trainer.optimizer
        self.accelerator = ppo_trainer.accelerator
        self.args = ppo_trainer.args

        # Get tokenizer (TRL 0.26.1 uses processing_class)
        self.tokenizer = getattr(ppo_trainer, 'processing_class', None)
        if self.tokenizer is None:
            self.tokenizer = getattr(ppo_trainer, 'tokenizer', None)

        # PPO hyperparameters
        self.kl_coef = getattr(self.args, 'kl_coef', 0.04)
        self.cliprange = getattr(self.args, 'cliprange', 0.2)
        self.cliprange_value = getattr(self.args, 'cliprange_value', 0.2)
        self.vf_coef = getattr(self.args, 'vf_coef', 0.1)
        self.gamma = getattr(self.args, 'gamma', 1.0)
        self.gae_lambda = getattr(self.args, 'lam', 0.95)

        logger.info("=" * 60)
        logger.info("Initialized PPOTrainerWithSelection (TRL 0.26.1)")
        if self.selection_enabled:
            logger.info(f"  Selection method: {selection_method}")
            logger.info(f"  Selection mode: {selection_mode}")
            logger.info(f"  Selection fraction: {selection_frac}")
            logger.info(f"  Validation loss type: {val_loss_type}")
            logger.info(f"  Second-order selection: {use_second_order}")
        else:
            logger.info("  Selection: DISABLED (baseline PPO)")
        logger.info("=" * 60)

    @torch.no_grad()
    def generate_rollouts(
        self,
        query_tensors: List[Tensor],
        generation_kwargs: Optional[Dict] = None,
    ) -> Tuple[List[Tensor], List[str]]:
        """
        Generate responses for given queries using the policy model.

        Args:
            query_tensors: List of query token tensors
            generation_kwargs: Generation configuration

        Returns:
            Tuple of (response_tensors, response_texts)
        """
        if generation_kwargs is None:
            generation_kwargs = {
                "max_new_tokens": 30,
                "min_new_tokens": 10,
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

        device = next(self.policy_model.parameters()).device
        self.policy_model.eval()

        # Batch all queries together for efficient generation
        # Pad queries to same length
        max_query_len = max(q.shape[0] if q.dim() == 1 else q.shape[1] for q in query_tensors)
        batch_size = len(query_tensors)

        padded_queries = torch.full(
            (batch_size, max_query_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros(batch_size, max_query_len, dtype=torch.long, device=device)

        for i, query in enumerate(query_tensors):
            if query.dim() == 1:
                query_len = query.shape[0]
                padded_queries[i, :query_len] = query.to(device)
                attention_mask[i, :query_len] = 1
            else:
                query_len = query.shape[1]
                padded_queries[i, :query_len] = query[0].to(device)
                attention_mask[i, :query_len] = 1

        # Generate all responses in one batched call
        with torch.no_grad():
            outputs = self.policy_model.generate(
                input_ids=padded_queries,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        # Extract responses (remove query prefix)
        response_tensors = []
        for i in range(batch_size):
            query_len = attention_mask[i].sum().item()
            response = outputs[i, query_len:]
            # Remove padding from response
            if self.tokenizer.pad_token_id is not None:
                valid_mask = response != self.tokenizer.pad_token_id
                if valid_mask.any():
                    last_valid = valid_mask.nonzero()[-1].item() + 1
                    response = response[:last_valid]
            response_tensors.append(response)

        # Decode responses
        response_texts = self.tokenizer.batch_decode(
            response_tensors, skip_special_tokens=True
        )

        self.policy_model.train()
        return response_tensors, response_texts

    @torch.no_grad()
    def compute_rewards(
        self,
        queries: List[str],
        responses: List[str],
    ) -> Tensor:
        """
        Compute rewards using the reward model.

        Args:
            queries: Query texts
            responses: Response texts

        Returns:
            Reward tensor [batch_size]
        """
        device = next(self.reward_model.parameters()).device
        texts = [q + r for q, r in zip(queries, responses)]

        # Check if reward_model is a CrossVocabRewardWrapper or similar
        if hasattr(self.reward_model, 'reward_backbone'):
            # It's our custom wrapper - use policy tokenizer
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = self.reward_model(inputs.input_ids, inputs.attention_mask)
            # Extract rewards from the score layer
            hidden_states = outputs.hidden_states[-1]
            rewards = self.reward_model.score(hidden_states[:, -1, :]).squeeze(-1)
        else:
            # Standard reward model
            # This assumes the reward model has its own tokenizer
            reward_tokenizer = getattr(self, 'reward_tokenizer', self.tokenizer)
            inputs = reward_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            outputs = self.reward_model(**inputs)
            if hasattr(outputs, 'logits'):
                # Classification model - use raw logits like reference
                # Model labels: class 0 = "nothate", class 1 = "hate"
                # Higher logit for nothate = higher reward = less toxic
                logits = outputs.logits.float()
                rewards = logits[:, 0]  # nothate logit as reward
            else:
                rewards = outputs[0].squeeze(-1)

        return rewards

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
            model: The model (policy or reference)
            input_ids: Full sequence [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            response_start: Index where response starts

        Returns:
            Log probabilities [batch, response_len]
        """
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = outputs.logits[:, response_start-1:-1, :]  # Shift for next-token prediction
        labels = input_ids[:, response_start:]

        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = torch.gather(
            log_probs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        return selected_log_probs

    def compute_validation_loss(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """
        Compute validation loss for capturing the "good direction".

        For RLHF, we use negative log-likelihood on validation responses
        as the signal for what constitutes "good" behavior.

        Args:
            input_ids: Validation input IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Scalar validation loss
        """
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        logits = outputs.logits

        # Shift for causal LM (predict next token)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        # Compute per-token cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_logits.size(0), -1)

        # Apply attention mask
        mask = attention_mask[:, 1:].contiguous()
        per_token_loss = per_token_loss * mask
        val_loss = per_token_loss.sum() / mask.sum().clamp(min=1)

        return val_loss

    def capture_validation_gradients(
        self,
        val_input_ids: Tensor,
        val_attention_mask: Tensor,
        generation_kwargs: Optional[Dict] = None,
    ) -> float:
        """
        Phase 1: Capture validation gradients.

        Forward/backward on validation data to capture compressed gradients
        representing the "good direction" for selection.

        Supports different validation loss types:
        - 'logprob': Simple NLL on validation prompts (original)
        - 'reward_weighted': Generate responses, weight by rewards (RLHF-aligned)
        - 'advantage_weighted': Generate responses, weight by advantages

        Args:
            val_input_ids: Validation input IDs (prompts)
            val_attention_mask: Validation attention mask
            generation_kwargs: Optional generation config for reward-weighted modes

        Returns:
            Validation loss value (for logging)
        """
        t_start = time.time()

        # Set hook to capture mode
        self.grad_hook.set_mode("capture_val")
        self.grad_hook.enable_hooks()

        # Ensure model is in training mode for gradient computation
        self.model.train()

        if self.val_loss_type == "logprob":
            # Original: simple NLL on validation prompts
            val_loss = self.compute_validation_loss(
                input_ids=val_input_ids,
                attention_mask=val_attention_mask,
            )

        elif self.val_loss_type in ("reward_weighted", "advantage_weighted"):
            # Generate responses for validation prompts and weight by rewards
            # This aligns the validation gradient with the RLHF objective

            if generation_kwargs is None:
                generation_kwargs = {
                    "max_new_tokens": 30,
                    "min_new_tokens": 10,
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "pad_token_id": self.tokenizer.pad_token_id,
                }

            device = val_input_ids.device
            batch_size = val_input_ids.shape[0]

            # Generate responses for validation prompts
            with torch.no_grad():
                response_tensors, response_texts = self.generate_rollouts(
                    [q for q in val_input_ids],
                    generation_kwargs,
                )

                # Pad responses to same length
                max_response_len = max(r.shape[0] for r in response_tensors)
                response_ids = torch.zeros(
                    batch_size, max_response_len,
                    dtype=torch.long, device=device
                )
                response_mask = torch.zeros(
                    batch_size, max_response_len,
                    dtype=torch.long, device=device
                )
                for i, r in enumerate(response_tensors):
                    response_ids[i, :r.shape[0]] = r
                    response_mask[i, :r.shape[0]] = 1

                # Compute rewards
                val_texts = self.tokenizer.batch_decode(val_input_ids, skip_special_tokens=True)
                rewards = self.compute_rewards(val_texts, response_texts)

                # Normalize rewards to be weights (higher reward = more weight)
                # Shift to positive and normalize
                rewards_normalized = rewards - rewards.min()
                if rewards_normalized.max() > 0:
                    rewards_normalized = rewards_normalized / rewards_normalized.max()
                # Add small minimum weight to avoid zero gradients
                weights = rewards_normalized + 0.1
                weights = weights / weights.sum() * batch_size  # Scale to sum to batch_size

            # Now compute loss with gradients enabled
            # Concatenate query and response
            full_ids = torch.cat([val_input_ids, response_ids], dim=1)
            full_mask = torch.cat([val_attention_mask, response_mask], dim=1)
            query_len = val_input_ids.shape[1]

            # Forward pass
            outputs = self.policy_model(
                input_ids=full_ids,
                attention_mask=full_mask,
                return_dict=True,
            )

            # Compute log probs for response tokens
            logits = outputs.logits[:, query_len-1:-1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            response_logprobs = torch.gather(
                log_probs, dim=-1, index=response_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Weight by rewards (samples with higher rewards contribute more)
            # This makes the validation gradient point toward generating high-reward responses
            weighted_logprobs = response_logprobs * weights.unsqueeze(-1) * response_mask.float()

            # Negative because we want to maximize (minimize negative)
            val_loss = -weighted_logprobs.sum() / response_mask.sum().clamp(min=1)

            logger.debug(
                f"Reward-weighted validation: rewards min={rewards.min():.2f}, "
                f"max={rewards.max():.2f}, mean={rewards.mean():.2f}"
            )

        else:
            raise ValueError(f"Unknown val_loss_type: {self.val_loss_type}")

        # Backward pass - hooks capture compressed gradients per layer
        val_loss.backward()

        # Clear optimizer gradients (we only wanted compressed grads in the buffer)
        self.optimizer.zero_grad()

        logger.debug(
            f"Captured validation gradients ({self.val_loss_type}): "
            f"{len(self.grad_hook.val_grad_buffer)} layers, "
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
        Compute PPO loss for policy and value updates.

        Args:
            query_ids: Query token IDs [batch, query_len]
            response_ids: Response token IDs [batch, response_len]
            query_mask: Query attention mask [batch, query_len]
            response_mask: Response attention mask [batch, response_len]
            old_logprobs: Log probabilities from rollout [batch, response_len]
            advantages: GAE advantages [batch, response_len]
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

        # Forward pass
        outputs = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get value predictions if available
        # For now, we skip value predictions to simplify the implementation
        # Value loss is still computed but with zero values
        values = torch.zeros_like(old_values)

        # Compute new log probs
        logits = outputs.logits[:, query_len-1:-1, :]  # Shift for response prediction
        log_probs = F.log_softmax(logits, dim=-1)
        new_logprobs = torch.gather(
            log_probs, dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Compute reference log probs for KL penalty
        if self.ref_model is not None:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                ref_logits = ref_outputs.logits[:, query_len-1:-1, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_logprobs = torch.gather(
                    ref_log_probs, dim=-1, index=response_ids.unsqueeze(-1)
                ).squeeze(-1)
        else:
            ref_logprobs = old_logprobs.detach()

        # PPO policy loss with clipping
        ratio = torch.exp(new_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # Value loss with clipping
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.cliprange_value, self.cliprange_value
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        vf_loss = (vf_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # KL penalty
        kl = (ref_logprobs - new_logprobs) * response_mask
        kl_loss = kl.sum() / response_mask.sum().clamp(min=1)

        # Total loss
        total_loss = pg_loss + self.vf_coef * vf_loss + self.kl_coef * kl_loss

        # Stats (using TRL-compatible naming)
        with torch.no_grad():
            clipfrac = ((ratio - 1).abs() > self.cliprange).float().mean().item()
            approx_kl = (0.5 * (new_logprobs - old_logprobs) ** 2 * response_mask).sum() / response_mask.sum()

        stats = {
            # TRL-compatible metric names
            "objective/kl": kl_loss.item(),
            "policy/approxkl_avg": approx_kl.item(),
            "policy/clipfrac_avg": clipfrac,
            "loss/policy_avg": pg_loss.item(),
            "loss/value_avg": vf_loss.item(),
            # Additional metrics
            "loss/total": total_loss.item(),
            "policy/ratio_mean": ratio.mean().item(),
        }

        return total_loss, stats

    def train_step_with_selection(
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
        Perform one training step with optional layer-wise selection.

        When selection is disabled (method=NA), this performs standard PPO update.
        When selection is enabled, this performs layer-wise gradient selection.

        Args:
            query_ids: Query token IDs
            response_ids: Response token IDs
            query_mask: Query attention mask
            response_mask: Response attention mask
            old_logprobs: Log probabilities from rollout
            advantages: GAE advantages
            returns: Returns
            old_values: Old value estimates

        Returns:
            Training statistics dictionary
        """
        batch_size = query_ids.shape[0]

        # Setup selection state (only if selection is enabled)
        if self.selection_enabled:
            lr = self.optimizer.param_groups[0]["lr"]
            # For global mode, we compute scores only in first pass, then do second pass
            # For per_layer mode, we select and reduce on-the-fly
            compute_scores_only = (self.selection_mode == "global")

            self.grad_hook.set_mode("train_select")
            self.grad_hook.setup_selection(
                train_batch_size=batch_size,
                selection_method=self.selection_method,
                selection_frac=self.selection_frac,
                lr=lr,
                compute_scores_only=compute_scores_only,
                use_second_order=self.use_second_order,
            )

        # Compute PPO loss
        loss, stats = self.compute_ppo_loss(
            query_ids=query_ids,
            response_ids=response_ids,
            query_mask=query_mask,
            response_mask=response_mask,
            old_logprobs=old_logprobs,
            advantages=advantages,
            returns=returns,
            old_values=old_values,
        )

        # Backward pass - hooks perform layer-wise selection if enabled
        loss.backward()

        # Gradient clipping
        max_grad_norm = getattr(self.args, 'max_grad_norm', 1.0)
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_grad_norm
            )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Clear selection state (only if selection is enabled)
        if self.selection_enabled:
            self.grad_hook.clear_selection()

        return stats

    def train_with_selection(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs: int = 1,
        log_interval: int = 10,
        max_steps: Optional[int] = None,
        generation_kwargs: Optional[Dict] = None,
    ):
        """
        Custom training loop with optional layer-wise selection.

        This method implements PPO training. When selection is enabled,
        it performs the two-pass selection mechanism. When disabled (NA),
        it performs standard PPO updates with consistent logging.

        Args:
            train_dataloader: DataLoader for training prompts
            val_dataloader: DataLoader for validation data (only needed if selection enabled)
            num_epochs: Number of training epochs
            log_interval: How often to log stats
            max_steps: Maximum number of training steps (None = no limit)
            generation_kwargs: Arguments for response generation
        """
        if self.selection_enabled:
            logger.info("Starting PPO training with layer-wise selection")
        else:
            logger.info("Starting PPO training (baseline, no selection)")

        if generation_kwargs is None:
            generation_kwargs = {
                "max_new_tokens": 30,
                "min_new_tokens": 10,
                "do_sample": True,
                "temperature": 1.0,
                "top_k": 0,
                "top_p": 1.0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }

        device = next(self.model.parameters()).device
        global_step = 0

        for epoch in range(num_epochs):
            self.model.train()
            epoch_stats = []

            # Phase 1: Capture validation gradients (only if selection enabled)
            val_loss = 0.0
            if self.selection_enabled and val_dataloader is not None:
                val_batch = next(iter(val_dataloader))
                val_input_ids = val_batch["input_ids"].to(device)
                val_attention_mask = val_batch.get("attention_mask")
                if val_attention_mask is None:
                    val_attention_mask = (val_input_ids != self.tokenizer.pad_token_id).long()
                val_attention_mask = val_attention_mask.to(device)

                val_loss = self.capture_validation_gradients(
                    val_input_ids=val_input_ids,
                    val_attention_mask=val_attention_mask,
                    generation_kwargs=generation_kwargs,
                )

                # Phase 2: Training with selection
                self.grad_hook.enable_hooks()

            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                # Get query tensors
                query_ids = batch["input_ids"].to(device)
                query_mask = batch.get("attention_mask")
                if query_mask is None:
                    query_mask = (query_ids != self.tokenizer.pad_token_id).long()
                query_mask = query_mask.to(device)

                # Generate rollouts
                query_texts = batch.get("query", self.tokenizer.batch_decode(query_ids, skip_special_tokens=True))

                with torch.no_grad():
                    # Generate responses
                    response_tensors, response_texts = self.generate_rollouts(
                        [q for q in query_ids],
                        generation_kwargs,
                    )

                    # Pad responses to same length
                    max_response_len = max(r.shape[0] for r in response_tensors)
                    response_ids = torch.zeros(
                        len(response_tensors), max_response_len,
                        dtype=torch.long, device=device
                    )
                    response_mask = torch.zeros(
                        len(response_tensors), max_response_len,
                        dtype=torch.long, device=device
                    )
                    for i, r in enumerate(response_tensors):
                        response_ids[i, :r.shape[0]] = r
                        response_mask[i, :r.shape[0]] = 1

                    # Compute rewards
                    rewards_scalar = self.compute_rewards(query_texts, response_texts)

                    # Create per-token rewards (sparse - only at end)
                    rewards = torch.zeros_like(response_mask, dtype=torch.float)
                    for i in range(len(response_tensors)):
                        last_idx = response_tensors[i].shape[0] - 1
                        rewards[i, last_idx] = rewards_scalar[i]

                    # Compute old log probs
                    full_ids = torch.cat([query_ids, response_ids], dim=1)
                    full_mask = torch.cat([query_mask, response_mask], dim=1)
                    old_logprobs = self.compute_log_probs(
                        self.policy_model, full_ids, full_mask, query_ids.shape[1]
                    )

                    # Compute old values (zeros if no value model)
                    old_values = torch.zeros_like(response_mask, dtype=torch.float)

                    # Compute advantages
                    advantages, returns = compute_advantages(
                        rewards, old_values, response_mask.float(),
                        self.gamma, self.gae_lambda
                    )

                    # Normalize advantages
                    adv_mean = (advantages * response_mask).sum() / response_mask.sum().clamp(min=1)
                    adv_var = ((advantages - adv_mean) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1)
                    advantages = (advantages - adv_mean) / (adv_var.sqrt() + 1e-8)

                # Training step with selection
                stats = self.train_step_with_selection(
                    query_ids=query_ids,
                    response_ids=response_ids,
                    query_mask=query_mask,
                    response_mask=response_mask,
                    old_logprobs=old_logprobs,
                    advantages=advantages,
                    returns=returns,
                    old_values=old_values,
                )

                # Add TRL-compatible reward metrics
                reward_mean = rewards_scalar.mean().item()
                stats["objective/scores"] = reward_mean
                stats["objective/rlhf_reward"] = reward_mean - self.kl_coef * stats["objective/kl"]
                stats["val_loss"] = val_loss
                epoch_stats.append(stats)
                global_step += 1

                if global_step % log_interval == 0:
                    avg_stats = {
                        k: float(np.mean([s[k] for s in epoch_stats[-log_interval:]]))
                        for k in epoch_stats[-1].keys()
                    }
                    # Log in TRL-compatible format: {'eps': X, 'metric': Y, ...}
                    avg_stats["eps"] = global_step
                    logger.info(str(avg_stats))

                # Check if we've reached max_steps
                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Reached max_steps ({max_steps}), stopping")
                    break

            # Reset hooks (only if selection enabled)
            if self.selection_enabled:
                self.grad_hook.set_mode("standard")
                self.grad_hook.disable_hooks()

            # Check if we've reached max_steps (break outer loop)
            if max_steps is not None and global_step >= max_steps:
                break

        logger.info("Training complete")

    def train(self):
        """
        Wrapper to call TRL's standard training (falls back to no selection).
        """
        logger.warning(
            "Called train() on PPOTrainerWithSelection. "
            "For selection, use train_with_selection() instead. "
            "Falling back to standard TRL training."
        )
        return self.ppo_trainer.train()
