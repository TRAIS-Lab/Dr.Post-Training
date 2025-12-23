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

Reference Implementation:
- Matches archive/LDA-ORL-main/rlhf-toxicity/scripts/ppo_trainer.py
- Uses disable_adapter() for PEFT reference logprobs (no separate ref model)
- Adaptive KL control for stable training
"""

import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================
# KL Controllers (matching TRL implementation)
# Reference: https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
# ============================================================

class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://huggingface.co/papers/1909.08593

    Dynamically adjusts KL coefficient based on observed KL divergence.
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: float):
        """
        Args:
            init_kl_coef: Initial KL coefficient
            target: Target KL divergence
            horizon: Horizon for adaptation (larger = slower adaptation)
        """
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """
        Update KL coefficient based on current KL divergence.

        Args:
            current: Current KL divergence
            n_steps: Number of steps (batch_size * num_processes)
        """
        # Proportional error clipped to [-0.2, 0.2]
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        # Multiplicative update
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller that keeps coefficient constant."""

    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """No-op for fixed controller."""
        pass


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation.

    Following the reference implementation (ppo_trainer.py lines 3333-3356):
    - Pre-mask values and rewards before GAE loop
    - Use index bounds checking for next values (not mask multiplication)
    - This ensures correct handling of terminal states

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

    # Pre-mask values and rewards (reference: lines 3343-3344)
    values = values * mask
    rewards = rewards * mask

    advantages_reversed = []
    lastgaelam = torch.zeros(batch_size, device=rewards.device)

    # GAE computation following reference (lines 3346-3350)
    for t in reversed(range(seq_len)):
        # Use index check for next values, not mask (reference line 3347)
        nextvalues = values[:, t + 1] if t < seq_len - 1 else torch.zeros(batch_size, device=values.device)
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * gae_lambda * lastgaelam
        advantages_reversed.append(lastgaelam)

    # Stack and reverse (reference line 3351)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)

    returns = advantages + values
    return advantages, returns


class StreamingPPOTrainer:
    """
    PPO Trainer with gradient streaming for data selection.

    This trainer implements PPO with support for three training methods:
    - NA: Standard PPO (no selection)
    - Streaming: Per-layer selection during backward (uses stored val gradients)
    - GREATS: Global selection with validation gradients (two-pass)

    Key Implementation Details (matching reference ppo_trainer.py):
    - For PEFT models: uses disable_adapter() to compute ref_logprobs (no separate ref model)
    - Adaptive KL control for stable training
    - Model in eval mode during forward passes for consistent logprobs
    - old_logprobs computed once before PPO epochs, new_logprobs computed fresh each mini-batch
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
            ref_model: Reference model for KL penalty (None for PEFT - uses disable_adapter)
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

        # Check if model is PEFT (LoRA)
        self.is_peft_model = getattr(model, "is_peft_model", False)
        if not self.is_peft_model and hasattr(model, "pretrained_model"):
            # AutoModelForCausalLMWithValueHead wraps the PEFT model
            self.is_peft_model = getattr(model.pretrained_model, "is_peft_model", False)

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
        self.cliprange = args.cliprange
        self.cliprange_value = args.cliprange_value
        self.vf_coef = args.vf_coef
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.ppo_epochs = getattr(args, 'ppo_epochs', 4)
        self.mini_batch_size = getattr(args, 'mini_batch_size', 1)
        # Forward batch size for efficient GPU utilization during logprobs computation
        # Reference uses tracin_batch_size=256 for forward passes
        self.forward_batch_size = getattr(args, 'forward_batch_size', 0)  # 0 = full batch

        # KL Controller (matching reference implementation)
        # Reference: ppo_trainer.py lines 294-297
        self.adap_kl_ctrl = getattr(args, 'adap_kl_ctrl', True)
        init_kl_coef = getattr(args, 'init_kl_coef', args.kl_coef)
        target_kl = getattr(args, 'target_kl', 6.0)
        horizon = getattr(args, 'horizon', 10000)

        if self.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, target_kl, horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        # KL penalty mode (matching reference: ppo_trainer.py _kl_penalty method)
        # Options: "kl", "abs", "mse", "full"
        self.kl_penalty_mode = getattr(args, 'kl_penalty', 'kl')

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
        logger.info(f"  KL coefficient: {self.kl_ctl.value} ({'adaptive' if self.adap_kl_ctrl else 'fixed'})")
        logger.info(f"  KL penalty mode: {self.kl_penalty_mode}")
        logger.info(f"  Clip range: {self.cliprange}")
        logger.info(f"  PPO epochs per batch: {self.ppo_epochs}")
        logger.info(f"  Mini-batch size (backward): {self.mini_batch_size}")
        logger.info(f"  Forward batch size: {self.forward_batch_size if self.forward_batch_size > 0 else 'full batch'}")
        logger.info(f"  PEFT model: {self.is_peft_model}")
        if self.is_peft_model:
            logger.info(f"  Reference: using disable_adapter() on policy model")
        else:
            logger.info(f"  Reference model: {'loaded (frozen)' if self.ref_model is not None else 'None'}")
        logger.info("=" * 60)

    def _kl_penalty(
        self,
        logprob: torch.FloatTensor,
        ref_logprob: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute KL penalty based on configured mode.

        Matches reference implementation: ppo_trainer.py lines 3317-3329

        Args:
            logprob: Policy log probabilities [batch, seq_len]
            ref_logprob: Reference log probabilities [batch, seq_len]

        Returns:
            KL penalty tensor [batch, seq_len]
        """
        if self.kl_penalty_mode == "kl":
            # Standard: can be negative if policy assigns lower prob than reference
            return logprob - ref_logprob

        if self.kl_penalty_mode == "abs":
            # Absolute value: always positive, prevents reward bonus from negative KL
            return (logprob - ref_logprob).abs()

        if self.kl_penalty_mode == "mse":
            # Mean squared error: always positive, smoother penalty
            return 0.5 * (logprob - ref_logprob).square()

        if self.kl_penalty_mode == "full":
            # True KL divergence using F.kl_div
            # Note: Flip is required due to PyTorch issue #57459
            return F.kl_div(ref_logprob, logprob, log_target=True, reduction="none").sum(-1)

        raise ValueError(f"Unknown kl_penalty mode: {self.kl_penalty_mode}")

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

        # NOTE: Do NOT set model.train() here - mode is managed by train() method
        # Reference implementation keeps model in eval mode during forward passes
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

    def compute_ref_log_probs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        response_start: int,
    ) -> Tensor:
        """
        Compute reference log probabilities for KL penalty.

        For PEFT models: uses disable_adapter() on the policy model
        For non-PEFT models: uses the separate frozen reference model

        This matches the reference implementation (ppo_trainer.py lines 780-798).

        Args:
            input_ids: Full sequence [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            response_start: Index where response starts

        Returns:
            Reference log probabilities [batch, response_len]
        """
        if self.is_peft_model:
            # For PEFT models, disable adapters to get base model logprobs
            # Reference: ppo_trainer.py lines 780-788
            pretrained_model = self.model.pretrained_model
            if hasattr(pretrained_model, "disable_adapter"):
                with pretrained_model.disable_adapter():
                    ref_logprobs = self.compute_log_probs(
                        self.model, input_ids, attention_mask, response_start
                    )
            else:
                raise ValueError(
                    "PEFT model does not support disable_adapter(). "
                    "Please update your peft version."
                )
        else:
            # For non-PEFT models, use the separate frozen reference model
            if self.ref_model is not None:
                ref_logprobs = self.compute_log_probs(
                    self.ref_model, input_ids, attention_mask, response_start
                )
            else:
                raise ValueError(
                    "No reference model available and model is not PEFT. "
                    "Cannot compute KL penalty."
                )

        return ref_logprobs

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
            # Fix: Values should start from position before first response token
            # V(s_t) = value at state BEFORE taking action t
            # For response tokens, V(s_0) is at position response_start - 1
            values = values_full[:, response_start - 1:-1]
        else:
            # Fallback if no value head (shouldn't happen)
            response_len = input_ids.shape[1] - response_start
            values = torch.zeros(input_ids.shape[0], response_len, device=self.device)

        return values

    def batched_forward_pass(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        batch_size: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute logprobs, logits, and values in batches for efficiency.

        This matches the reference implementation (ppo_trainer.py batched_forward_pass)
        which uses tracin_batch_size for forward passes to maximize GPU utilization.

        Args:
            query_ids: Query token IDs [batch, query_len]
            response_ids: Response token IDs [batch, response_len]
            query_mask: Query attention mask
            response_mask: Response attention mask
            batch_size: Batch size for forward passes (0 = full batch)

        Returns:
            Tuple of (logprobs, logits, values) all with shape [batch, response_len]
        """
        full_batch_size = query_ids.shape[0]
        query_len = query_ids.shape[1]

        # Use full batch if batch_size is 0 or larger than full_batch
        if batch_size <= 0 or batch_size >= full_batch_size:
            batch_size = full_batch_size

        # Concatenate query and response
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = torch.cat([query_mask, response_mask], dim=1)

        all_logprobs = []
        all_logits = []
        all_values = []

        # Process in batches
        for i in range(0, full_batch_size, batch_size):
            end_idx = min(i + batch_size, full_batch_size)

            batch_input_ids = input_ids[i:end_idx]
            batch_attention_mask = attention_mask[i:end_idx]
            batch_response_ids = response_ids[i:end_idx]

            # Forward pass
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
            )

            # Handle output format
            if isinstance(outputs, tuple):
                logits = outputs[0]
                values_full = outputs[2] if len(outputs) >= 3 else None
            else:
                logits = outputs.logits
                values_full = None

            # Extract logits for response tokens
            logits_for_probs = logits[:, query_len - 1:-1, :]

            # Compute log probs
            log_probs = F.log_softmax(logits_for_probs.float(), dim=-1)
            batch_logprobs = torch.gather(
                log_probs,
                dim=-1,
                index=batch_response_ids.unsqueeze(-1),
            ).squeeze(-1)

            all_logprobs.append(batch_logprobs)
            all_logits.append(logits_for_probs)

            # Extract values
            if values_full is not None:
                if values_full.dim() == 3:
                    values_full = values_full.squeeze(-1)
                batch_values = values_full[:, query_len - 1:-1]
            else:
                batch_values = torch.zeros_like(batch_logprobs)

            all_values.append(batch_values)

        # Concatenate all batches
        logprobs = torch.cat(all_logprobs, dim=0)
        logits = torch.cat(all_logits, dim=0)
        values = torch.cat(all_values, dim=0)

        return logprobs, logits, values

    def batched_ref_forward_pass(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        batch_size: int = 0,
    ) -> Tensor:
        """
        Compute reference log probabilities in batches for KL penalty.

        For PEFT models: uses disable_adapter() on the policy model
        For non-PEFT models: uses the separate frozen reference model

        Args:
            query_ids: Query token IDs [batch, query_len]
            response_ids: Response token IDs [batch, response_len]
            query_mask: Query attention mask
            response_mask: Response attention mask
            batch_size: Batch size for forward passes (0 = full batch)

        Returns:
            Reference log probabilities [batch, response_len]
        """
        full_batch_size = query_ids.shape[0]
        query_len = query_ids.shape[1]

        # Use full batch if batch_size is 0 or larger than full_batch
        if batch_size <= 0 or batch_size >= full_batch_size:
            batch_size = full_batch_size

        # Concatenate query and response
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = torch.cat([query_mask, response_mask], dim=1)

        all_ref_logprobs = []

        # Choose model for reference logprobs
        if self.is_peft_model:
            # For PEFT models, use disable_adapter context
            pretrained_model = self.model.pretrained_model
            if not hasattr(pretrained_model, "disable_adapter"):
                raise ValueError(
                    "PEFT model does not support disable_adapter(). "
                    "Please update your peft version."
                )
            context_manager = pretrained_model.disable_adapter()
            ref_model = self.model
        else:
            # For non-PEFT models, use separate reference model
            if self.ref_model is None:
                raise ValueError(
                    "No reference model available and model is not PEFT. "
                    "Cannot compute KL penalty."
                )
            from contextlib import nullcontext
            context_manager = nullcontext()
            ref_model = self.ref_model

        # Process in batches within the context
        with context_manager:
            for i in range(0, full_batch_size, batch_size):
                end_idx = min(i + batch_size, full_batch_size)

                batch_input_ids = input_ids[i:end_idx]
                batch_attention_mask = attention_mask[i:end_idx]
                batch_response_ids = response_ids[i:end_idx]

                # Forward pass
                outputs = ref_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                )

                # Handle output format
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits

                # Extract logits for response tokens
                logits_for_probs = logits[:, query_len - 1:-1, :]

                # Compute log probs
                log_probs = F.log_softmax(logits_for_probs.float(), dim=-1)
                batch_ref_logprobs = torch.gather(
                    log_probs,
                    dim=-1,
                    index=batch_response_ids.unsqueeze(-1),
                ).squeeze(-1)

                all_ref_logprobs.append(batch_ref_logprobs)

        # Concatenate all batches
        ref_logprobs = torch.cat(all_ref_logprobs, dim=0)
        return ref_logprobs

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

        # Compute validation loss based on val_loss_type
        if self.val_loss_type == "logprob":
            # Simple negative log likelihood (no reward weighting)
            val_loss = -seq_log_probs.mean()
        elif self.val_loss_type == "reward_weighted":
            # NLL weighted by normalized rewards
            val_loss = -(rewards_normalized * seq_log_probs).mean()
        elif self.val_loss_type == "advantage_weighted":
            # NLL weighted by advantages (same as reward_weighted for now since we use rewards as advantages)
            val_loss = -(rewards_normalized * seq_log_probs).mean()
        else:
            raise ValueError(f"Unknown val_loss_type: {self.val_loss_type}")

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
        # Fix: Values should start from position before first response token
        # V(s_t) = value at state BEFORE taking action t
        if values_full.dim() == 3:
            values_full = values_full.squeeze(-1)
        values = values_full[:, query_len - 1:-1]

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
            # Compute masked mean of ratio for threshold check
            avg_ratio = (ratio * response_mask).sum() / response_mask.sum().clamp(min=1)
            avg_ratio = avg_ratio.item()

        # CRITICAL: Skip batch if ratio exceeds threshold (prevents divergence)
        # Reference implementation (ppo_trainer.py line 3409) does this
        ratio_threshold = getattr(self.args, 'ratio_threshold', 10.0)
        if avg_ratio > ratio_threshold:
            logger.warning(
                f"Avg ratio ({avg_ratio:.2f}) exceeds threshold ({ratio_threshold:.2f}). "
                f"Skipping batch to prevent divergence."
            )
            # Zero out loss to skip this batch
            total_loss = total_loss * 0.0
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0

        stats = {
            "loss/total": total_loss.item(),
            "loss/policy": pg_loss.item(),
            "loss/value": vf_loss.item(),
            "policy/approx_kl": approx_kl.item(),
            "policy/clipfrac": clipfrac,
            "policy/ratio_mean": avg_ratio,
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
        accumulate: bool = False,
        loss_scale: float = 1.0,
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
            accumulate: If True, only accumulate gradients without optimizer.step()
            loss_scale: Scale factor for loss (for gradient accumulation, use 1/num_accumulation_steps)

        Returns:
            Training statistics dictionary
        """
        batch_size = query_ids.shape[0]
        lr = self.optimizer.param_groups[0]["lr"]

        # ========================================
        # Method: NA (baseline, no selection)
        # ========================================
        if self.method == "NA":
            # Only zero grad if not accumulating (first call of accumulation cycle)
            if not accumulate:
                self.optimizer.zero_grad()

            # Disable hooks for baseline
            if self.grad_hook is not None:
                self.grad_hook.disable_hooks()

            loss, stats = self.compute_ppo_loss(
                query_ids, response_ids, query_mask, response_mask,
                old_logprobs, advantages, returns, old_values,
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss * loss_scale
            scaled_loss.backward()

            # Only step optimizer if not accumulating
            if not accumulate:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.grad_hook is not None:
                self.grad_hook.enable_hooks()

            # Add learning rate to stats
            stats["ppo/learning_rate"] = lr

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
                response_ids, response_mask, response_texts = self.generate_rollouts(
                    query_ids, query_mask
                )

                # Compute rewards from reward model
                with torch.no_grad():
                    query_texts = self.tokenizer.batch_decode(query_ids, skip_special_tokens=True)
                    raw_rewards = self.reward_model.compute_rewards(query_texts, response_texts)

                # Prepare full sequences
                full_ids = torch.cat([query_ids, response_ids], dim=1)
                full_mask = torch.cat([query_mask, response_mask], dim=1)
                query_len = query_ids.shape[1]

                # ========================================
                # IMPORTANT: Keep model in TRAIN mode throughout PPO
                #
                # The reference implementation switches to eval() mode here,
                # but this causes train/eval mode mismatch when computing
                # new_logprobs in the PPO loop (which runs in train mode).
                #
                # With all dropouts set to 0 (lora_dropout=0, summary_dropout_prob=0),
                # there should be no functional difference between train/eval modes.
                # Keeping train mode throughout ensures consistent logprobs.
                #
                # NOTE: generate_rollouts() sets model to eval() for generation.
                # We must restore train() mode here before computing old_logprobs.
                # ========================================
                self.model.train()  # Restore train mode after generation

                with torch.no_grad():
                    # ========================================
                    # CRITICAL FIX: Use mini_batch_size for old_logprobs computation
                    #
                    # The model gives different logprobs for different batch sizes
                    # (due to numerical precision in batched attention with left-padding).
                    # Since PPO loop computes new_logprobs with mini_batch_size,
                    # we MUST compute old_logprobs with the same batch size.
                    #
                    # This is slower but ensures ratio ≈ 1.0 at the start of training.
                    # ========================================
                    fwd_batch_size = self.mini_batch_size  # Use mini_batch_size, NOT forward_batch_size

                    # Compute old log probs and values in batches (policy model in train mode)
                    old_logprobs, _, old_values = self.batched_forward_pass(
                        query_ids, response_ids, query_mask, response_mask,
                        batch_size=fwd_batch_size,
                    )

                    # Compute reference log probs for KL penalty in batches
                    # For PEFT: uses disable_adapter() on same model
                    # For non-PEFT: uses separate frozen reference model
                    # Reference: ppo_trainer.py lines 780-798
                    ref_logprobs = self.batched_ref_forward_pass(
                        query_ids, response_ids, query_mask, response_mask,
                        batch_size=fwd_batch_size,
                    )

                    # Create per-token reward tensor (sparse - only at last token)
                    # Reference: ppo_trainer.py compute_rewards method
                    rewards = torch.zeros_like(response_mask, dtype=torch.float, device=self.device)
                    for i in range(len(response_texts)):
                        last_idx = response_mask[i].sum().item() - 1
                        if last_idx >= 0:
                            rewards[i, last_idx] = raw_rewards[i]

                    # ========================================
                    # KL PENALTY VIA REWARD SHAPING
                    # Reference: ppo_trainer.py compute_rewards lines 3304-3314
                    # ========================================
                    # KL penalty is subtracted from rewards BEFORE computing GAE
                    # Uses configured kl_penalty mode (kl/abs/mse/full)
                    kl_penalty = self._kl_penalty(old_logprobs, ref_logprobs)
                    non_score_rewards = -self.kl_ctl.value * kl_penalty
                    rewards = rewards + non_score_rewards * response_mask.float()

                    # Compute advantages and returns using KL-shaped rewards
                    # Reference: ppo_trainer.py compute_advantages lines 3333-3356
                    advantages, returns = compute_gae(
                        rewards, old_values, response_mask.float(),
                        self.gamma, self.gae_lambda
                    )

                    # Normalize advantages using masked_whiten (reference line 3354)
                    adv_mean = (advantages * response_mask).sum() / response_mask.sum().clamp(min=1)
                    adv_var = ((advantages - adv_mean) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1)
                    advantages = ((advantages - adv_mean) / (adv_var.sqrt() + 1e-8)) * response_mask.float()
                    advantages = advantages.detach()  # Reference line 3355

                # Capture validation gradients per-step for Streaming
                if self.method == "Streaming" and val_query_ids is not None:
                    self.capture_validation_gradients(
                        val_query_ids, val_query_mask
                    )

                # PPO epochs with mini-batching (matching reference implementation)
                # old_logprobs stays fixed, new_logprobs recomputed each mini-batch
                batch_size = query_ids.shape[0]
                all_mini_batch_stats = []

                # Ensure model is in train mode for PPO epochs
                # (Should already be in train mode from epoch start, but call again to be safe)
                self.model.train()

                # Removed duplicate diagnostic - the batch size check above is sufficient

                for ppo_epoch in range(self.ppo_epochs):
                    # Shuffle indices at the start of each epoch (like reference)
                    perm = torch.randperm(batch_size, device=query_ids.device)

                    # Iterate over mini-batches
                    for mb_start in range(0, batch_size, self.mini_batch_size):
                        mb_end = min(mb_start + self.mini_batch_size, batch_size)
                        mb_inds = perm[mb_start:mb_end]

                        # Extract mini-batch
                        mb_query_ids = query_ids[mb_inds]
                        mb_response_ids = response_ids[mb_inds]
                        mb_query_mask = query_mask[mb_inds]
                        mb_response_mask = response_mask[mb_inds]
                        mb_old_logprobs = old_logprobs[mb_inds].detach()
                        mb_advantages = advantages[mb_inds].detach()
                        mb_returns = returns[mb_inds].detach()
                        mb_old_values = old_values[mb_inds].detach()

                        # Training step on mini-batch
                        mb_stats = self.training_step(
                            mb_query_ids, mb_response_ids, mb_query_mask, mb_response_mask,
                            mb_old_logprobs, mb_advantages, mb_returns, mb_old_values,
                        )
                        all_mini_batch_stats.append(mb_stats)

                # Aggregate stats from all mini-batches (use last for simplicity, average key metrics)
                stats = all_mini_batch_stats[-1].copy()
                if len(all_mini_batch_stats) > 1:
                    for key in ["loss/total", "loss/policy", "loss/value", "policy/ratio_mean", "policy/clipfrac"]:
                        if key in stats:
                            stats[key] = float(np.mean([s[key] for s in all_mini_batch_stats if key in s]))

                # Add reward and KL stats
                stats["reward/mean"] = raw_rewards.mean().item()
                stats["reward/std"] = raw_rewards.std().item()

                # Compute KL divergence (sum over sequence, mean over batch)
                # Reference: ppo_trainer.py record_step_stats lines 3467-3470
                kl_per_seq = (kl_penalty * response_mask).sum(dim=-1)
                mean_kl = kl_per_seq.mean().item()
                stats["objective/kl"] = mean_kl
                stats["objective/kl_coef"] = self.kl_ctl.value

                # Warning for negative KL (reference: ppo_trainer.py lines 3474-3480)
                # This indicates the policy assigns lower probability to its own
                # generated tokens than the reference - a pathological state
                if mean_kl < -1.0:
                    warnings.warn(
                        f"KL divergence is starting to become negative: {mean_kl:.2f} - "
                        "this might be a precursor for failed training. "
                        "Consider using kl_penalty='abs' to prevent reward bonus from negative KL, "
                        "or review your training hyperparameters (try increasing mini_batch_size)."
                    )

                # Update KL controller (reference line 933-936)
                # multiply batch_size by num_processes for distributed training
                self.kl_ctl.update(mean_kl, batch_size)

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
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
