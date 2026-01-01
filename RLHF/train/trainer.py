"""
Streaming PPO Trainer for RLHF experiments.

Methods:
- NA: Standard PPO (no selection)
- Streaming: Per-layer selection during backward (single-pass for training)
- GREATS: Global selection with validation gradients (two-pass for training)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from gradstream.selection import create_separate_batch_strategy

logger = logging.getLogger(__name__)

# Following TRL 0.26.1: Use a special value for invalid logprobs at masked positions
# This ensures masked positions don't affect mean calculations
INVALID_LOGPROB = 1.0


def first_true_indices(bools: torch.Tensor, dtype=torch.long) -> torch.Tensor:
    """
    Find the position of the first True in each row.
    Returns the length of the row if no True is found.

    Matches TRL's implementation.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(stop_token_id: int, pad_token_id: int, responses: torch.Tensor) -> torch.Tensor:
    """
    Truncates responses at the first occurrence of stop token, filling rest with pad.

    This is critical for proper KL computation - positions after the stop token
    should be masked out consistently.

    Matches TRL 0.26.1 implementation.

    Args:
        stop_token_id: Token ID where truncation occurs
        pad_token_id: Token ID to fill truncated positions
        responses: Response tensor [batch, seq_len]

    Returns:
        Truncated responses with pad tokens after stop token
    """
    trunc_idxs = first_true_indices(responses == stop_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, pad_token_id)
    return postprocessed_responses


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf

    Adjusts the KL penalty coefficient based on observed KL divergence.
    When current KL > target, increases coefficient to penalize divergence more.
    When current KL < target, decreases coefficient to allow more exploration.

    The clipping range [-0.2, 0.2] matches TRL's implementation for stable updates.
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: float):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        # Proportional error: positive if current > target, negative if current < target
        # Clipping to [-0.2, 0.2] ensures gradual updates (matches TRL reference)
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        pass


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    mask: Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    whiten: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation with optional whitening.

    Matches TRL's compute_advantages + masked_whiten implementation.

    Args:
        rewards: Per-token rewards (with KL penalty already applied) [batch, seq_len]
        values: Value estimates [batch, seq_len]
        mask: Valid token mask [batch, seq_len]
        gamma: Discount factor
        gae_lambda: GAE lambda
        whiten: If True, whiten advantages using masked_whiten with Bessel correction

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

    # Whiten advantages (matching TRL's masked_whiten with Bessel correction)
    # Reference: trl/core.py masked_whiten() - does NOT multiply by mask at end
    if whiten:
        mask_sum = mask.sum()
        adv_mean = (advantages * mask).sum() / mask_sum.clamp(min=1)
        centered = advantages - adv_mean
        # Compute variance with Bessel correction (unbiased estimator)
        adv_var = (centered ** 2 * mask).sum() / mask_sum.clamp(min=1)
        if mask_sum > 1:
            adv_var = adv_var * mask_sum / (mask_sum - 1)
        advantages = centered * torch.rsqrt(adv_var + 1e-8)
        # TRL 0.26.1: Zero out masked positions after whitening
        # Reference: ppo_trainer.py line 613
        advantages = torch.masked_fill(advantages, mask == 0, 0)
        advantages = advantages.detach()

    return advantages, returns


class StreamingPPOTrainer:
    """
    PPO Trainer with gradient streaming for data selection.

    This trainer implements PPO with support for three training methods:
    - NA: Standard PPO (no selection)
    - Streaming: Per-layer selection during backward (uses stored val gradients)
    - GREATS: Global selection with validation gradients (two-pass)
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
        evaluator=None,
        val_dataset=None,
    ):
        """
        Initialize the trainer.

        Args:
            model: Policy model (with value head if using value function)
            ref_model: Reference model for KL penalty (None for PEFT - uses disable_adapter, shared-layer model for non-PEFT)
            reward_model: Reward model wrapper
            tokenizer: Tokenizer
            args: TrainingArguments
            grad_hook: GradientHook instance (None for NA/baseline)
            optimizer: Optimizer (created if None)
            lr_scheduler: LR scheduler (optional)
            evaluator: ToxicityEvaluator instance for evaluation during training (optional)
            val_dataset: Fixed validation dataset for data selection (optional).
                         If provided and args.use_fixed_validation is True, uses this
                         instead of self-referencing validation.
        """
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.args = args
        self.grad_hook = grad_hook
        self.lr_scheduler = lr_scheduler
        self.evaluator = evaluator
        self.val_dataset = val_dataset

        # Cached validation rollout data (generated once and reused)
        self._val_rollout_cache: Optional[Dict[str, Any]] = None
        self._val_rollout_step: int = -1  # Step when cache was generated

        # Device
        self.device = next(model.parameters()).device

        # Check if model is PEFT (LoRA)
        self.is_peft_model = getattr(model, "is_peft_model", False)
        if not self.is_peft_model and hasattr(model, "pretrained_model"):
            # AutoModelForCausalLMWithValueHead wraps the PEFT model
            self.is_peft_model = getattr(model.pretrained_model, "is_peft_model", False)

        # Create optimizer if not provided
        if optimizer is None:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            )
            logger.info(f"Optimizer: lr={args.learning_rate}")
        else:
            self.optimizer = optimizer

        # PPO hyperparameters
        self.cliprange = args.cliprange
        self.cliprange_value = args.cliprange_value
        self.vf_coef = args.vf_coef
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.ppo_epochs = args.ppo_epochs
        self.mini_batch_size = args.mini_batch_size

        # KL Controller (matching reference implementation)
        # Note: `target` is for AdaptiveKLController, `target_kl` is for early stopping
        self.adap_kl_ctrl = args.adap_kl_ctrl
        if self.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(args.init_kl_coef, args.target, args.horizon)
        else:
            self.kl_ctl = FixedKLController(args.init_kl_coef)

        # Early stopping configuration
        self.early_stopping = args.early_stopping
        self.target_kl = args.target_kl

        # KL estimator: "k1", "k2", "k3" (http://joschu.net/blog/kl-approx.html)
        self.kl_estimator = args.kl_estimator

        # Selection configuration
        self.method = args.method
        self.filter_frac = args.filter_frac
        self.use_second_order = args.use_second_order

        # Create selection strategy for clean separation of selection methods
        # RLHF uses filtering mode: keep positive + drop bottom frac of negative
        self.selection_strategy = create_separate_batch_strategy(
            method=self.method,
            grad_hook=self.grad_hook,
            frac=self.filter_frac,
            use_second_order=self.use_second_order,
            selection_mode="filtering",
        )

        # Logging
        self._log_config()

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """
        Compute gradient norms for policy (LoRA) and v_head layers.

        This helps verify that gradient normalization is working correctly.
        For Streaming/GREATS, policy gradients should match NA baseline if
        normalization is correct.

        Supports both compressed gradients (stored in _compressed_grad on weight params)
        and standard gradients (param.grad).

        Returns:
            Dictionary with policy_grad_norm, vhead_grad_norm, and per-layer norms
        """
        policy_grad_norm_sq = 0.0
        vhead_grad_norm_sq = 0.0
        lora_a_grad_norm_sq = 0.0
        lora_b_grad_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            # Check for compressed gradient first (stored on weight parameters only)
            # _compressed_grad contains both weight and bias gradients combined
            compressed_grad = getattr(param, '_compressed_grad', None)

            if compressed_grad is not None:
                # Use compressed gradient norm (represents layer's gradient in compressed space)
                grad_norm_sq = compressed_grad.norm().item() ** 2
                # Only weight params have _compressed_grad, so categorize based on weight name
                if "lora" in name.lower():
                    policy_grad_norm_sq += grad_norm_sq
                    if "lora_A" in name or "lora_a" in name:
                        lora_a_grad_norm_sq += grad_norm_sq
                    elif "lora_B" in name or "lora_b" in name:
                        lora_b_grad_norm_sq += grad_norm_sq

            elif param.grad is not None:
                # Use standard gradient (for non-compressed layers like v_head)
                grad_norm_sq = param.grad.norm().item() ** 2
                if "v_head" in name:
                    vhead_grad_norm_sq += grad_norm_sq
                elif "lora" in name.lower():
                    policy_grad_norm_sq += grad_norm_sq
                    if "lora_A" in name or "lora_a" in name:
                        lora_a_grad_norm_sq += grad_norm_sq
                    elif "lora_B" in name or "lora_b" in name:
                        lora_b_grad_norm_sq += grad_norm_sq

        return {
            "grad_norm/policy": policy_grad_norm_sq ** 0.5,
            "grad_norm/vhead": vhead_grad_norm_sq ** 0.5,
            "grad_norm/lora_a": lora_a_grad_norm_sq ** 0.5,
            "grad_norm/lora_b": lora_b_grad_norm_sq ** 0.5,
        }

    def _log_config(self):
        """Log trainer configuration."""
        logger.info("=" * 60)
        logger.info("StreamingPPOTrainer Configuration")
        logger.info(f"  Method: {self.method}")
        if self.method != "NA":
            logger.info(f"  Filter fraction (negative samples to drop): {self.filter_frac}")
            if getattr(self.args, 'use_fixed_validation', False) and self.val_dataset is not None:
                n_val = len(self.val_dataset)
                val_batch_size = getattr(self.args, 'val_batch_size', 1)
                val_gen_interval = getattr(self.args, 'val_generation_interval', 0)
                logger.info(f"  Validation: fixed dataset (n_val={n_val}, batch_size={val_batch_size})")
                logger.info(f"  Val generation interval: {val_gen_interval} (0=once at start)")
            else:
                logger.info(f"  Validation: self-reference (training buffer)")
            logger.info(f"  Second-order selection: {self.use_second_order}")
        logger.info(f"  KL coefficient: {self.kl_ctl.value} ({'adaptive' if self.adap_kl_ctrl else 'fixed'})")
        logger.info(f"  KL estimator: {self.kl_estimator}")
        logger.info(f"  Clip range: {self.cliprange}")
        max_grad_norm = getattr(self.args, 'max_grad_norm', None)
        logger.info(f"  Max grad norm: {max_grad_norm if max_grad_norm else 'disabled'}")
        logger.info(f"  PPO epochs per batch: {self.ppo_epochs}")
        logger.info(f"  Mini-batch size: {self.mini_batch_size}")
        logger.info(f"  PEFT model: {self.is_peft_model}")
        if self.is_peft_model:
            logger.info(f"  Reference: using disable_adapter() on policy model")
        else:
            logger.info(f"  Reference model: {'shared layers' if self.ref_model is not None else 'None'}")
        logger.info("=" * 60)

    @torch.no_grad()
    def _compute_initial_stats(
        self,
        query_ids: Tensor,
        response_ids: Tensor,
        query_mask: Tensor,
        response_mask: Tensor,
        old_logprobs: Tensor,
        advantages: Tensor,
        returns: Tensor,
        old_values: Tensor,
        raw_rewards: Tensor,
        kl_penalty: Tensor,
        response_texts: List[str],
    ) -> Dict[str, float]:
        """
        Compute all training statistics without performing any gradient updates.

        This is used to log initial stats at Step 0 before training begins.

        Args:
            query_ids: Query token IDs [batch, query_len]
            response_ids: Response token IDs [batch, response_len]
            query_mask: Query attention mask
            response_mask: Response attention mask
            old_logprobs: Log probs from rollout [batch, response_len]
            advantages: GAE advantages [batch, response_len]
            returns: Returns for value loss [batch, response_len]
            old_values: Old value estimates [batch, response_len]
            raw_rewards: Raw rewards from reward model [batch]
            kl_penalty: KL penalty values [batch, response_len]
            response_texts: Generated response texts

        Returns:
            Dictionary of statistics
        """
        batch_size = query_ids.shape[0]
        query_len = query_ids.shape[1]

        # Concatenate query and response
        input_ids = torch.cat([query_ids, response_ids], dim=1)
        attention_mask = torch.cat([query_mask, response_mask], dim=1)

        # Forward pass to get current logprobs and values
        self.model.eval()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # Handle model output format
        if isinstance(outputs, tuple):
            logits = outputs[0]
            values_full = outputs[2]
        else:
            logits = outputs.logits
            values_full = torch.zeros(batch_size, input_ids.shape[1], device=self.device)

        # Extract values and logits for response tokens
        response_len = response_ids.shape[1]
        start_idx = query_len - 1
        end_idx = start_idx + response_len

        if values_full.dim() == 3:
            values_full = values_full.squeeze(-1)
        values = values_full[:, start_idx:end_idx]

        # Compute new log probs
        logits_for_probs = logits[:, start_idx:end_idx, :]
        log_probs = F.log_softmax(logits_for_probs.float(), dim=-1)
        new_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)

        # PPO policy loss computation (without backward)
        logprob_diff = new_logprobs - old_logprobs
        ratio = torch.exp(logprob_diff)
        clipped_ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # Value loss computation
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.cliprange_value,
            self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        vf_loss = (vf_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # Total loss
        total_loss = pg_loss + self.vf_coef * vf_loss

        # Stats for logging (TRL 0.26.1 naming conventions)
        # Policy clip fraction
        pg_clipfrac = ((ratio - 1).abs() > self.cliprange).float().mean().item()

        # Value clip fraction
        vf_clipfrac = ((values - old_values).abs() > self.cliprange_value).float().mean().item()

        # KL approximations
        approx_kl = (0.5 * (new_logprobs - old_logprobs) ** 2 * response_mask).sum()
        approx_kl = approx_kl / response_mask.sum()

        # Ratio statistics
        avg_ratio = (ratio * response_mask).sum() / response_mask.sum().clamp(min=1)
        ratio_var = ((ratio - avg_ratio) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1)

        # Entropy
        entropy = (-new_logprobs * response_mask).sum() / response_mask.sum().clamp(min=1)

        # KL stats
        kl_per_seq = (kl_penalty * response_mask).sum(dim=-1)
        mean_kl = kl_per_seq.mean().item()

        # Entropy per sequence
        entropy_per_seq = (-old_logprobs * response_mask.float()).sum(dim=-1)

        # Non-score reward (KL penalty)
        non_score_reward_per_seq = (-self.kl_ctl.value * kl_per_seq)

        # Build stats dictionary (TRL 0.26.1 naming conventions)
        stats = {
            # Loss metrics
            "loss/policy": pg_loss.item(),
            "loss/value": vf_loss.item(),
            "loss/total": total_loss.item(),
            # Policy metrics
            "policy/approxkl": approx_kl.item(),
            "policy/clipfrac": pg_clipfrac,
            "policy/entropy": entropy.item(),
            "policy/ratio": avg_ratio.item(),
            # Value metrics
            "val/clipfrac": vf_clipfrac,
            "val/ratio_var": ratio_var.item(),
            "val/mean": values.mean().item(),
            # Objective metrics (TRL style)
            "objective/kl": mean_kl,
            "objective/kl_coef": self.kl_ctl.value,
            "objective/entropy": entropy_per_seq.mean().item(),
            "objective/non_score_reward": non_score_reward_per_seq.mean().item(),
            "objective/scores": raw_rewards.mean().item(),
            "objective/rlhf_reward": non_score_reward_per_seq.mean().item() + raw_rewards.mean().item(),
            # Legacy reward stats
            "reward/mean": raw_rewards.mean().item(),
            "reward/std": raw_rewards.std().item(),
        }

        # Number of EOS tokens
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            stats["val/num_eos_tokens"] = (response_ids == eos_token_id).sum().item()

        # Toxicity evaluation stats
        if self.evaluator is not None and getattr(self.args, 'eval_on_step_generations', True):
            eval_results = self.evaluator.evaluate_generations(response_texts)
            stats["eval/toxicity_prob"] = eval_results["mean_toxicity_prob"]
            stats["eval/toxicity_rate"] = eval_results["toxicity_rate"]

        return stats

    def _compute_kl(
        self,
        logprob: torch.FloatTensor,
        ref_logprob: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute KL divergence estimate using configured estimator.

        Based on John Schulman's blog: http://joschu.net/blog/kl-approx.html
        Matches TRL experimental PPO implementation.

        Let r = π_ref(x) / π(x), so log(r) = ref_logprob - logprob

        Args:
            logprob: Policy log probabilities [batch, seq_len]
            ref_logprob: Reference log probabilities [batch, seq_len]

        Returns:
            KL estimate tensor [batch, seq_len]
        """
        # log(r) where r = π_ref / π
        logr = ref_logprob - logprob

        if self.kl_estimator == "k1":
            # k1 = -log(r) = log(π/π_ref) = logprob - ref_logprob
            # Unbiased, but can be negative (higher variance)
            return -logr

        if self.kl_estimator == "k2":
            # k2 = 0.5 * log(r)^2
            # Biased but low variance, useful for logging
            return 0.5 * logr.square()

        if self.kl_estimator == "k3":
            # k3 = (r - 1) - log(r) = exp(logr) - 1 - logr
            # Unbiased, low variance, always positive (recommended)
            return (logr.exp() - 1) - logr

        raise ValueError(f"Unknown kl_estimator: {self.kl_estimator}")

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

        # Generation config for PPO rollouts
        # IMPORTANT: Do NOT use min_new_tokens/min_length during training rollouts!
        # Setting min_length causes the model to ignore EOS token until min_length is reached,
        # which makes the model assign very low log prob to EOS and high probs to other tokens.
        # This leads to negative KL divergence, which the model can exploit for reward hacking.
        # See: https://huggingface.co/docs/trl/main/en/how_to_train
        gen_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "min_length": -1,  # Don't ignore EOS token (prevents negative KL exploitation)
            "do_sample": True,
            "temperature": self.args.temperature,
            "top_k": self.args.top_k,
            "top_p": self.args.top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
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

        # TRL 0.26.1 approach: Truncate responses at EOS and fill rest with pad
        # This ensures consistent handling of positions after stop token
        if self.tokenizer.eos_token_id is not None:
            response_ids = truncate_response(
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
                response_ids,
            )

        # Compute sequence lengths using first_true_indices (matches TRL)
        # sequence_length is the position of first pad token minus 1
        sequence_lengths = first_true_indices(response_ids == self.tokenizer.pad_token_id) - 1
        # Clamp to valid range (at least 0, at most response_len - 1)
        sequence_lengths = sequence_lengths.clamp(min=0, max=response_ids.size(1) - 1)

        # Create padding mask: True for positions > sequence_length (to be masked out)
        response_idxs = torch.arange(response_ids.size(1), device=response_ids.device)
        padding_mask = response_idxs.unsqueeze(0) > sequence_lengths.unsqueeze(1)

        # response_mask is the inverse: 1 for valid positions, 0 for padding
        response_mask = (~padding_mask).long()

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
        which uses mini-batch processing for forward passes to maximize GPU utilization.

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

            # TRL 0.26.1 approach: compute position_ids and mask input_ids
            # This ensures consistent handling of padding tokens
            batch_position_ids = batch_attention_mask.cumsum(1) - batch_attention_mask.long()
            batch_input_ids_masked = torch.masked_fill(batch_input_ids, ~batch_attention_mask.bool(), 0)

            # Forward pass (use_cache=False to prevent KV cache issues after generate())
            outputs = self.model(
                input_ids=batch_input_ids_masked,
                attention_mask=batch_attention_mask,
                position_ids=batch_position_ids,
                use_cache=False,
            )

            # Handle output format
            if isinstance(outputs, tuple):
                logits = outputs[0]
                values_full = outputs[2] if len(outputs) >= 3 else None
            else:
                logits = outputs.logits
                values_full = None

            # Extract logits for response tokens
            # Use explicit response_len to ensure correct slicing regardless of model output shape
            # Indexing: logit at position i predicts token at position i+1, so for response
            # tokens at [query_len, query_len+response_len), we need logits at [query_len-1, query_len+response_len-1)
            # TODO: For models like LLaMA that prepend BOS tokens, verify tokenizer.encode()
            # uses add_special_tokens=False consistently, or adjust indices accordingly.
            response_len = batch_response_ids.shape[1]
            start_idx = query_len - 1
            end_idx = start_idx + response_len
            logits_for_probs = logits[:, start_idx:end_idx, :]

            # Debug assertion to catch shape mismatches early
            if logits_for_probs.shape[1] != response_len:
                raise RuntimeError(
                    f"Logits shape mismatch: expected {response_len} positions, "
                    f"got {logits_for_probs.shape[1]}. "
                    f"logits.shape={logits.shape}, query_len={query_len}, "
                    f"response_len={response_len}, start_idx={start_idx}, end_idx={end_idx}"
                )

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
                batch_values = values_full[:, start_idx:end_idx]
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

                # TRL 0.26.1 approach: compute position_ids and mask input_ids
                batch_position_ids = batch_attention_mask.cumsum(1) - batch_attention_mask.long()
                batch_input_ids_masked = torch.masked_fill(batch_input_ids, ~batch_attention_mask.bool(), 0)

                # Forward pass (use_cache=False to prevent KV cache issues)
                outputs = ref_model(
                    input_ids=batch_input_ids_masked,
                    attention_mask=batch_attention_mask,
                    position_ids=batch_position_ids,
                    use_cache=False,
                )

                # Handle output format
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs.logits

                # Extract logits for response tokens
                # Use explicit response_len to ensure correct slicing
                response_len = batch_response_ids.shape[1]
                start_idx = query_len - 1
                end_idx = start_idx + response_len
                logits_for_probs = logits[:, start_idx:end_idx, :]

                # Debug assertion to catch shape mismatches early
                if logits_for_probs.shape[1] != response_len:
                    raise RuntimeError(
                        f"Ref logits shape mismatch: expected {response_len} positions, "
                        f"got {logits_for_probs.shape[1]}. "
                        f"logits.shape={logits.shape}, query_len={query_len}, "
                        f"response_len={response_len}"
                    )

                # TRL 0.26.1: Apply temperature scaling to logits before computing log_softmax
                # Reference: ppo_trainer.py line 528
                temperature = getattr(self.args, 'temperature', 1.0)
                logits_for_probs = logits_for_probs / (temperature + 1e-7)

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

    def capture_validation_gradients_from_buffer(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
    ) -> float:
        """
        Capture validation gradients using training rollout data (self-referencing validation).
        Validation gradients are computed on the same responses used for training

        The validation loss formula:
            L_val = -E[log π_θ(y|x) * Â_{-1}(x,y)]

        Where:
        - y: Responses from policy model (reused from training rollouts)
        - Â_{-1}: Advantage at the LAST token (emphasizes reward model feedback)
        - log π_θ(y|x): Log probability under the current policy

        Args:
            query_ids: Query token IDs [batch, query_len]
            query_mask: Query attention mask [batch, query_len]
            rollout_data: Dictionary containing training rollout data with keys:
                - response_ids: Generated response token IDs
                - response_mask: Response attention mask
                - advantages: GAE advantages (already computed with KL penalty)

        Returns:
            Validation loss value
        """
        # Reuse training rollout data (matching reference implementation)
        response_ids = rollout_data["response_ids"]
        response_mask = rollout_data["response_mask"]
        advantages = rollout_data["advantages"]

        # Prepare full sequence tensors
        full_ids = torch.cat([query_ids, response_ids], dim=1)
        full_mask = torch.cat([query_mask, response_mask], dim=1)
        query_len = query_ids.shape[1]

        # Extract advantage at the LAST token (Â_{-1})
        # Note: advantages are already whitened (mean=0, std=1) at token-level by compute_gae()
        last_token_indices = response_mask.sum(dim=1) - 1  # [batch_size]
        seq_advantages = advantages[
            torch.arange(advantages.size(0), device=advantages.device),
            last_token_indices.long()
        ]

        # Capture validation gradients
        # Start validation capture mode
        self.grad_hook.start_val_capture(use_factorized=False)
        self.grad_hook.enable_hooks()

        # Set model to train mode for gradient computation
        self.model.train()

        full_batch_size = query_ids.shape[0]
        batch_size = self.mini_batch_size
        total_val_loss = 0.0

        # Process in mini-batches to avoid OOM
        for i in range(0, full_batch_size, batch_size):
            end_idx = min(i + batch_size, full_batch_size)

            # Extract mini-batch
            mb_full_ids = full_ids[i:end_idx]
            mb_full_mask = full_mask[i:end_idx]
            mb_response_ids = response_ids[i:end_idx]
            mb_response_mask = response_mask[i:end_idx]
            mb_seq_advantages = seq_advantages[i:end_idx]

            # Forward pass on mini-batch (log π_θ(y|x))
            outputs = self.model(
                input_ids=mb_full_ids,
                attention_mask=mb_full_mask,
            )

            # Handle model output format
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
                index=mb_response_ids.unsqueeze(-1),
            ).squeeze(-1)

            # Sequence log probability (sum over response tokens)
            seq_log_probs = (token_log_probs * mb_response_mask.float()).sum(dim=1)

            # Compute validation loss: -E[log π_θ(y|x) * Â_{-1}(x,y)]
            # Following reference LDA-ORL: per_seq_loss = -seq_logprob * seq_score, then .mean()
            # For mini-batch gradient accumulation: .sum() / full_batch_size is equivalent
            per_seq_loss = -(mb_seq_advantages * seq_log_probs)
            mb_val_loss = per_seq_loss.sum() / full_batch_size

            # Backward - hooks accumulate gradients into val_grad_buffer
            mb_val_loss.backward()

            total_val_loss += mb_val_loss.item()

        # Clear optimizer gradients (we only wanted to capture compressed grads)
        self.optimizer.zero_grad()

        # End capture mode
        self.grad_hook.end_val_capture()

        return total_val_loss

    @torch.no_grad()
    def _generate_validation_rollouts(self, global_step: int = 0) -> Dict[str, Any]:
        """
        Generate rollouts for the fixed validation dataset.

        This method generates responses for validation prompts and computes
        rewards and advantages. The results are cached and reused across
        training steps unless val_generation_interval triggers regeneration.

        Args:
            global_step: Current training step (used to check if regeneration needed)

        Returns:
            Dictionary containing validation rollout data:
                - query_ids: Query token IDs [n_val, query_len]
                - query_mask: Query attention mask
                - response_ids: Generated response IDs [n_val, response_len]
                - response_mask: Response attention mask
                - response_texts: Generated response texts
                - raw_rewards: Rewards from reward model [n_val]
                - advantages: Computed advantages (last-token style) [n_val]
        """
        # Check if we can reuse cached rollouts
        val_gen_interval = getattr(self.args, 'val_generation_interval', 0)
        if self._val_rollout_cache is not None:
            if val_gen_interval == 0:
                # Generate once and reuse forever
                return self._val_rollout_cache
            elif global_step - self._val_rollout_step < val_gen_interval:
                # Within interval, reuse cache
                return self._val_rollout_cache

        logger.info(f"Generating validation rollouts for fixed validation set (step {global_step})...")

        # Prepare all validation prompts
        all_query_ids = []
        all_query_masks = []

        for sample in self.val_dataset:
            query_ids = sample["input_ids"]
            if isinstance(query_ids, torch.Tensor):
                query_ids = query_ids.unsqueeze(0) if query_ids.dim() == 1 else query_ids
            else:
                query_ids = torch.tensor([query_ids])
            query_ids = query_ids.to(self.device)
            query_mask = (query_ids != self.tokenizer.pad_token_id).long()
            all_query_ids.append(query_ids)
            all_query_masks.append(query_mask)

        # Pad to same length and stack
        max_query_len = max(q.shape[1] for q in all_query_ids)
        padded_query_ids = []
        padded_query_masks = []
        for q, m in zip(all_query_ids, all_query_masks):
            pad_len = max_query_len - q.shape[1]
            if pad_len > 0:
                q = F.pad(q, (pad_len, 0), value=self.tokenizer.pad_token_id)
                m = F.pad(m, (pad_len, 0), value=0)
            padded_query_ids.append(q)
            padded_query_masks.append(m)

        query_ids = torch.cat(padded_query_ids, dim=0)
        query_mask = torch.cat(padded_query_masks, dim=0)

        # Generate responses in batches
        val_batch_size = getattr(self.args, 'val_batch_size', 1)
        n_val = query_ids.shape[0]

        all_response_ids = []
        all_response_masks = []
        all_response_texts = []

        self.model.eval()
        for i in range(0, n_val, val_batch_size):
            end_idx = min(i + val_batch_size, n_val)
            batch_query_ids = query_ids[i:end_idx]
            batch_query_mask = query_mask[i:end_idx]

            response_ids, response_mask, response_texts = self.generate_rollouts(
                batch_query_ids, batch_query_mask
            )
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_texts.extend(response_texts)

        # Pad responses to same length and stack
        max_response_len = max(r.shape[1] for r in all_response_ids)
        padded_response_ids = []
        padded_response_masks = []
        for r, m in zip(all_response_ids, all_response_masks):
            pad_len = max_response_len - r.shape[1]
            if pad_len > 0:
                r = F.pad(r, (0, pad_len), value=self.tokenizer.pad_token_id)
                m = F.pad(m, (0, pad_len), value=0)
            padded_response_ids.append(r)
            padded_response_masks.append(m)

        response_ids = torch.cat(padded_response_ids, dim=0)
        response_mask = torch.cat(padded_response_masks, dim=0)

        # Compute rewards
        query_texts = self.tokenizer.batch_decode(query_ids, skip_special_tokens=True)
        raw_rewards = self.reward_model.compute_rewards(query_texts, all_response_texts)

        # Compute advantages (last-token style, similar to capture_validation_gradients_from_buffer)
        # For fixed validation, we use simple reward-based advantages
        # Normalize rewards to have mean 0, std 1
        if len(raw_rewards) > 1:
            advantages = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + 1e-8)
        else:
            advantages = raw_rewards - raw_rewards.mean()

        # Cache the rollouts
        self._val_rollout_cache = {
            "query_ids": query_ids,
            "query_mask": query_mask,
            "response_ids": response_ids,
            "response_mask": response_mask,
            "response_texts": all_response_texts,
            "raw_rewards": raw_rewards,
            "advantages": advantages,
        }
        self._val_rollout_step = global_step

        logger.info(
            f"Generated {n_val} validation rollouts. "
            f"Mean reward: {raw_rewards.mean():.3f}, Std: {raw_rewards.std():.3f}"
        )

        return self._val_rollout_cache

    def capture_validation_gradients_from_fixed_val(
        self,
        global_step: int = 0,
    ) -> float:
        """
        Capture validation gradients using a fixed validation dataset.

        Unlike capture_validation_gradients_from_buffer which uses the same
        training batch for validation, this method uses a separate fixed
        validation set with pre-generated completions.

        The validation loss formula:
            L_val = -E[log π_θ(y|x) * A(x,y)]

        Where:
        - y: Responses generated from fixed validation prompts
        - A(x,y): Normalized reward (advantage) for each validation sample
        - log π_θ(y|x): Log probability under the current policy

        Args:
            global_step: Current training step (for rollout regeneration check)

        Returns:
            Validation loss value
        """
        # Generate or retrieve cached validation rollouts
        val_rollout = self._generate_validation_rollouts(global_step)

        query_ids = val_rollout["query_ids"]
        query_mask = val_rollout["query_mask"]
        response_ids = val_rollout["response_ids"]
        response_mask = val_rollout["response_mask"]
        advantages = val_rollout["advantages"]  # [n_val] - per-sequence advantages

        # Prepare full sequence tensors
        full_ids = torch.cat([query_ids, response_ids], dim=1)
        full_mask = torch.cat([query_mask, response_mask], dim=1)
        query_len = query_ids.shape[1]

        # Capture validation gradients
        self.grad_hook.start_val_capture(use_factorized=False)
        self.grad_hook.enable_hooks()

        # Set model to train mode for gradient computation
        self.model.train()

        full_batch_size = query_ids.shape[0]
        val_batch_size = getattr(self.args, 'val_batch_size', 1)
        total_val_loss = 0.0

        # Process in mini-batches
        for i in range(0, full_batch_size, val_batch_size):
            end_idx = min(i + val_batch_size, full_batch_size)

            # Extract mini-batch
            mb_full_ids = full_ids[i:end_idx]
            mb_full_mask = full_mask[i:end_idx]
            mb_response_ids = response_ids[i:end_idx]
            mb_response_mask = response_mask[i:end_idx]
            mb_advantages = advantages[i:end_idx]

            # Forward pass (log π_θ(y|x))
            outputs = self.model(
                input_ids=mb_full_ids,
                attention_mask=mb_full_mask,
            )

            # Handle model output format
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
                index=mb_response_ids.unsqueeze(-1),
            ).squeeze(-1)

            # Sequence log probability (sum over response tokens)
            seq_log_probs = (token_log_probs * mb_response_mask.float()).sum(dim=1)

            # Compute validation loss: -E[log π_θ(y|x) * A(x,y)]
            per_seq_loss = -(mb_advantages * seq_log_probs)
            mb_val_loss = per_seq_loss.sum() / full_batch_size

            # Backward - hooks accumulate gradients into val_grad_buffer
            mb_val_loss.backward()

            total_val_loss += mb_val_loss.item()

        # Clear optimizer gradients (we only wanted to capture gradients)
        self.optimizer.zero_grad()

        # End capture mode
        self.grad_hook.end_val_capture()

        return total_val_loss

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
        Compute PPO loss (policy + value).

        Note: KL penalty is already applied to rewards via reward shaping
        (see train method), so we don't add it as a loss term here.

        Safety: If the average policy ratio exceeds ratio_threshold (default 10.0),
        the batch is skipped by zeroing out the loss. This prevents training
        divergence when the policy has drifted too far from the reference.

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

        # TRL 0.26.1 approach: compute position_ids and mask input_ids
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids_masked = torch.masked_fill(input_ids, ~attention_mask.bool(), 0)

        # Forward pass - model returns (logits, loss, values) for AutoModelForCausalLMWithValueHead
        # use_cache=False to prevent KV cache issues
        outputs = self.model(
            input_ids=input_ids_masked,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
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

        # Extract values and logits for response tokens only
        # Use explicit response_len for robust slicing
        # TODO: For LLaMA-style models with BOS tokens, verify index consistency (see batched_forward_pass)
        response_len = response_ids.shape[1]
        start_idx = query_len - 1
        end_idx = start_idx + response_len

        # V(s_t) = value at state BEFORE taking action t
        if values_full.dim() == 3:
            values_full = values_full.squeeze(-1)
        values = values_full[:, start_idx:end_idx]

        # New log probs
        logits_for_probs = logits[:, start_idx:end_idx, :]

        # Debug assertion
        if logits_for_probs.shape[1] != response_len:
            raise RuntimeError(
                f"PPO loss logits shape mismatch: expected {response_len}, "
                f"got {logits_for_probs.shape[1]}. "
                f"logits.shape={logits.shape}, query_len={query_len}"
            )

        # TRL 0.26.1: Apply temperature scaling to logits before computing log_softmax
        # Reference: ppo_trainer.py line 637
        temperature = getattr(self.args, 'temperature', 1.0)
        logits_for_probs = logits_for_probs / (temperature + 1e-7)

        log_probs = F.log_softmax(logits_for_probs.float(), dim=-1)
        new_logprobs = torch.gather(
            log_probs,
            dim=-1,
            index=response_ids.unsqueeze(-1),
        ).squeeze(-1)

        # TRL 0.26.1 approach: Apply INVALID_LOGPROB to masked positions
        padding_mask = (response_mask == 0)
        new_logprobs = torch.masked_fill(new_logprobs, padding_mask, INVALID_LOGPROB)

        # PPO policy loss with clipping
        logprob_diff = new_logprobs - old_logprobs
        ratio = torch.exp(logprob_diff)

        clipped_ratio = torch.clamp(ratio, 1 - self.cliprange, 1 + self.cliprange)

        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * clipped_ratio
        pg_loss = torch.max(pg_loss1, pg_loss2)
        pg_loss = (pg_loss * response_mask).sum() / response_mask.sum().clamp(min=1)

        # TRL 0.26.1: Values use padding_mask_p1 which has one extra valid position
        # compared to logprobs' padding_mask. This is because V(s_t) at the terminal
        # state is still needed for bootstrapping even if no action is taken there.
        # Reference: ppo_trainer.py lines 582-584, 643, 652
        # Compute value_mask (equivalent to ~padding_mask_p1)
        seq_lens = response_mask.sum(dim=1) - 1  # last valid position index
        seq_lens_p1 = (seq_lens + 1).clamp(max=response_len - 1)  # one past, clamped
        response_idxs = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        value_mask = (response_idxs <= seq_lens_p1.unsqueeze(1)).long()

        # Value loss with clipping
        # Zero out values at masked positions first (TRL line 643)
        values = values * value_mask
        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.cliprange_value,
            self.cliprange_value,
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        vf_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
        vf_loss = (vf_loss * value_mask).sum() / value_mask.sum().clamp(min=1)

        # Total loss (no KL term - it's in the rewards/advantages already)
        total_loss = pg_loss + self.vf_coef * vf_loss

        # Stats for logging (TRL 0.26.1 naming conventions)
        with torch.no_grad():
            # Policy clip fraction
            pg_clipfrac = ((ratio - 1).abs() > self.cliprange).float().mean().item()

            # Value clip fraction (TRL: val/clipfrac_avg)
            vf_clipfrac = ((values - old_values).abs() > self.cliprange_value).float().mean().item()

            # Second-order KL approximation (always >= 0)
            approx_kl = (0.5 * (new_logprobs - old_logprobs) ** 2 * response_mask).sum()
            approx_kl = approx_kl / response_mask.sum()

            # First-order KL approximation (can be negative) - matches TRL's policykl
            policykl = ((old_logprobs - new_logprobs) * response_mask).sum()
            policykl = policykl / response_mask.sum()

            # Ratio statistics
            avg_ratio = (ratio * response_mask).sum() / response_mask.sum().clamp(min=1)
            avg_ratio = avg_ratio.item()
            ratio_var = ((ratio - avg_ratio) ** 2 * response_mask).sum() / response_mask.sum().clamp(min=1)
            ratio_var = ratio_var.item()

            # Entropy (TRL: policy/entropy_avg)
            # entropy = logsumexp(logits) - sum(prob * logits)
            # Simplified: entropy = -sum(prob * log_prob) ≈ -new_logprobs for selected tokens
            entropy = (-new_logprobs * response_mask).sum() / response_mask.sum().clamp(min=1)
            entropy = entropy.item()

        # Skip batch if ratio exceeds threshold (prevents divergence)
        ratio_threshold = getattr(self.args, 'ratio_threshold', 10.0)
        if avg_ratio > ratio_threshold:
            total_loss = total_loss * 0.0
            pg_loss = pg_loss * 0.0
            vf_loss = vf_loss * 0.0

        # Stats following TRL 0.26.1 naming conventions
        stats = {
            "loss/policy": pg_loss.item(),
            "loss/value": vf_loss.item(),
            "loss/total": total_loss.item(),
            "policy/approxkl": approx_kl.item(),
            "policy/policykl": policykl.item(),
            "policy/clipfrac": pg_clipfrac,
            "policy/entropy": entropy,
            "policy/ratio": avg_ratio,
            "val/clipfrac": vf_clipfrac,
            "val/ratio_var": ratio_var,
            "val/mean": values.mean().item(),
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

        Uses the selection strategy pattern for clean separation of methods:
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

        # Determine effective strategy (fall back to NA for small batches)
        strategy = self._get_effective_strategy(batch_size)

        # Zero gradients before training step
        self.optimizer.zero_grad()

        # Create compute_loss_fn closure for the current batch
        def compute_loss_fn() -> Tuple[Tensor, Dict]:
            return self.compute_ppo_loss(
                query_ids, response_ids, query_mask, response_mask,
                old_logprobs, advantages, returns, old_values,
            )

        # For GREATS, create filter_batch_fn to get filtered compute_loss_fn
        def filter_batch_fn(selected_indices: Tensor) -> Callable[[], Tuple[Tensor, Dict]]:
            def filtered_compute_loss_fn() -> Tuple[Tensor, Dict]:
                return self.compute_ppo_loss(
                    query_ids[selected_indices],
                    response_ids[selected_indices],
                    query_mask[selected_indices],
                    response_mask[selected_indices],
                    old_logprobs[selected_indices],
                    advantages[selected_indices],
                    returns[selected_indices],
                    old_values[selected_indices],
                )
            return filtered_compute_loss_fn

        # Create pseudo-labels for token-based gradient scaling.
        # TRL 0.26.1: Value loss uses padding_mask_p1 (one extra valid position),
        # while policy loss uses response_mask. Since gradients from both flow through
        # LoRA layers, we use value_mask (the superset) for token counting.
        # Labels format: -100 for ignored positions, any other value for valid tokens.
        response_len = response_mask.size(1)
        seq_lens = response_mask.sum(dim=1) - 1  # last valid position index
        seq_lens_p1 = (seq_lens + 1).clamp(max=response_len - 1)
        response_idxs = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        value_mask = (response_idxs <= seq_lens_p1.unsqueeze(1))

        labels = torch.where(
            value_mask,
            torch.zeros_like(response_mask),  # valid tokens: 0
            torch.full_like(response_mask, -100),  # padding: -100
        )

        # Execute selection strategy
        loss, stats = strategy.execute_training_step(
            model=self.model,
            batch_size=batch_size,
            compute_loss_fn=compute_loss_fn,
            lr=lr,
            filter_batch_fn=filter_batch_fn,
            labels=labels,  # Pass labels for token-based gradient scaling
        )

        # Record selection statistics (for logging via stats dict)
        if self.method != "NA" and self.grad_hook is not None:
            sel_state = getattr(self.grad_hook, 'selection_state', None)
            if sel_state is not None:
                if self.method == "Streaming":
                    if hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
                        n_selected_list = [n for _, n in sel_state._layer_selections]
                        stats["selection/avg_selected"] = sum(n_selected_list) / len(n_selected_list)
                elif self.method == "GREATS":
                    n_selected = getattr(sel_state, 'num_selected', None)
                    if n_selected is not None:
                        stats["selection/n_selected"] = n_selected

        # Compute gradient norms (before clipping, for logging)
        grad_norms = self._compute_gradient_norms()
        stats.update(grad_norms)

        # Gradient clipping
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.args.max_grad_norm,
            )
            stats["grad_norm/total_before_clip"] = stats.get("grad_norm/policy", 0) + stats.get("grad_norm/vhead", 0)
            stats["grad_norm/total_after_clip"] = total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm

        # Add learning rate to stats
        stats["lr"] = lr

        # Optimizer step (skip if accumulating gradients)
        if not accumulate:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return stats

    def _get_effective_strategy(self, batch_size: int):
        """Get the selection strategy."""
        return self.selection_strategy

    def _prepare_batch(self, batch: Dict) -> Tuple[Tensor, Tensor]:
        """
        Prepare query tensors from a batch.

        Args:
            batch: Batch dictionary with 'input_ids'

        Returns:
            Tuple of (query_ids, query_mask)
        """
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
        return query_ids, query_mask

    @torch.no_grad()
    def _generate_rollout_data(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
    ) -> Dict[str, Any]:
        """
        Generate rollouts and compute all data needed for PPO training.

        This includes: response generation, reward computation, logprobs,
        KL penalty, advantages, and returns.

        Args:
            query_ids: Query token IDs [batch, query_len]
            query_mask: Query attention mask

        Returns:
            Dictionary containing all rollout data
        """
        # Generate responses
        response_ids, response_mask, response_texts = self.generate_rollouts(query_ids, query_mask)

        # Compute rewards
        query_texts = self.tokenizer.batch_decode(query_ids, skip_special_tokens=True)

        raw_rewards = self.reward_model.compute_rewards(query_texts, response_texts)

        # Compute logprobs and values (model in eval mode)
        self.model.eval()
        old_logprobs, _, old_values = self.batched_forward_pass(
            query_ids, response_ids, query_mask, response_mask,
            batch_size=self.mini_batch_size,
        )

        # Compute reference logprobs for KL penalty
        ref_logprobs = self.batched_ref_forward_pass(
            query_ids, response_ids, query_mask, response_mask,
            batch_size=self.mini_batch_size,
        )

        # TRL 0.26.1 approach: Mask out invalid logprobs at padded positions
        # This ensures masked positions don't affect mean calculations
        padding_mask = (response_mask == 0)
        old_logprobs = torch.masked_fill(old_logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

        # TRL 0.26.1: Values use padding_mask_p1 (one extra valid position)
        # Reference: ppo_trainer.py lines 582-584
        response_len = response_ids.size(1)
        seq_lens = response_mask.sum(dim=1) - 1  # last valid position index
        seq_lens_p1 = (seq_lens + 1).clamp(max=response_len - 1)
        response_idxs = torch.arange(response_len, device=response_mask.device).unsqueeze(0)
        padding_mask_p1 = response_idxs > seq_lens_p1.unsqueeze(1)
        old_values = torch.masked_fill(old_values, padding_mask_p1, 0)

        # Create per-token reward tensor (sparse - only at last token)
        # Reference: trl/trainer/ppo_trainer.py compute_rewards() uses mask.nonzero()[-1]
        rewards = torch.zeros_like(response_mask, dtype=torch.float, device=self.device)
        for i in range(len(response_texts)):
            # Use nonzero()[-1] to get last non-masked index (matches TRL)
            nonzero_indices = response_mask[i].nonzero()
            if len(nonzero_indices) > 0:
                last_idx = nonzero_indices[-1].item()
                rewards[i, last_idx] = raw_rewards[i]

        # Apply KL penalty via reward shaping
        kl_penalty = self._compute_kl(old_logprobs, ref_logprobs)
        non_score_rewards = -self.kl_ctl.value * kl_penalty
        rewards = rewards + non_score_rewards * response_mask.float()

        # Compute advantages and returns (with whitening inside compute_gae)
        advantages, returns = compute_gae(
            rewards, old_values, response_mask.float(),
            self.gamma, self.gae_lambda
        )

        return {
            "response_ids": response_ids,
            "response_mask": response_mask,
            "response_texts": response_texts,
            "raw_rewards": raw_rewards,
            "old_logprobs": old_logprobs,
            "old_values": old_values,
            "ref_logprobs": ref_logprobs,  # Store for debugging
            "kl_penalty": kl_penalty,
            "advantages": advantages,
            "returns": returns,
        }

    def _run_ppo_epochs(
        self,
        query_ids: Tensor,
        query_mask: Tensor,
        rollout_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Run PPO training epochs on rollout data.

        Args:
            query_ids: Query token IDs
            query_mask: Query attention mask
            rollout_data: Dictionary from _generate_rollout_data()

        Returns:
            Aggregated training statistics
        """
        response_ids = rollout_data["response_ids"]
        response_mask = rollout_data["response_mask"]
        old_logprobs = rollout_data["old_logprobs"]
        old_values = rollout_data["old_values"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]

        batch_size = query_ids.shape[0]
        all_mini_batch_stats = []

        self.model.train()

        early_stopped = False
        for ppo_epoch in range(self.ppo_epochs):
            if early_stopped:
                break

            perm = torch.randperm(batch_size, device=query_ids.device)
            epoch_ratios = []  # DEBUG: track ratios in this epoch

            for mb_start in range(0, batch_size, self.mini_batch_size):
                mb_end = min(mb_start + self.mini_batch_size, batch_size)
                mb_inds = perm[mb_start:mb_end]

                mb_stats = self.training_step(
                    query_ids[mb_inds],
                    response_ids[mb_inds],
                    query_mask[mb_inds],
                    response_mask[mb_inds],
                    old_logprobs[mb_inds].detach(),
                    advantages[mb_inds].detach(),
                    returns[mb_inds].detach(),
                    old_values[mb_inds].detach(),
                )
                all_mini_batch_stats.append(mb_stats)
                epoch_ratios.append(mb_stats.get("policy/ratio_mean", 1.0))

                # Check for early stopping based on policy KL
                # Use policykl (first-order approximation) to match TRL's implementation
                # Reference: ppo_trainer.py line 745 uses train_stats["policy/policykl"]
                policy_kl = mb_stats.get("policy/policykl", 0.0)
                if self._early_stop(policy_kl):
                    early_stopped = True
                    break

        # Aggregate stats across all mini-batches
        stats = all_mini_batch_stats[-1].copy()
        if len(all_mini_batch_stats) > 1:
            # Collect all keys from all mini-batch stats
            all_keys = set()
            for s in all_mini_batch_stats:
                all_keys.update(s.keys())

            for key in all_keys:
                values = [s[key] for s in all_mini_batch_stats if key in s]
                if not values or not isinstance(values[0], (int, float)):
                    continue
                stats[key] = float(np.mean(values))

        # Track early stopping
        stats["ppo/early_stopped"] = 1.0 if early_stopped else 0.0

        return stats

    def _build_step_stats(
        self,
        ppo_stats: Dict[str, float],
        rollout_data: Dict[str, Any],
        response_mask: Tensor,
    ) -> Dict[str, float]:
        """
        Build complete step statistics from PPO stats and rollout data.

        Args:
            ppo_stats: Stats from _run_ppo_epochs()
            rollout_data: Data from _generate_rollout_data()
            response_mask: Response attention mask

        Returns:
            Complete statistics dictionary
        """
        stats = ppo_stats.copy()

        raw_rewards = rollout_data["raw_rewards"]
        kl_penalty = rollout_data["kl_penalty"]
        response_texts = rollout_data["response_texts"]
        response_ids = rollout_data["response_ids"]
        old_logprobs = rollout_data["old_logprobs"]

        # TRL 0.26.1 objective metrics
        # KL stats
        kl_per_seq = (kl_penalty * response_mask).sum(dim=-1)
        mean_kl = kl_per_seq.mean().item()
        stats["objective/kl"] = mean_kl
        stats["objective/kl_coef"] = self.kl_ctl.value

        # Entropy: -sum(logprobs) per sequence (TRL line 698)
        entropy_per_seq = (-old_logprobs * response_mask.float()).sum(dim=-1)
        stats["objective/entropy"] = entropy_per_seq.mean().item()

        # Non-score reward (KL penalty portion)
        non_score_reward_per_seq = (-self.kl_ctl.value * kl_per_seq)
        stats["objective/non_score_reward"] = non_score_reward_per_seq.mean().item()

        # Scores (raw reward model output)
        stats["objective/scores"] = raw_rewards.mean().item()

        # RLHF reward = non_score_reward + scores (TRL line 700)
        stats["objective/rlhf_reward"] = stats["objective/non_score_reward"] + stats["objective/scores"]

        # Number of EOS tokens in responses (TRL line 719)
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            stats["val/num_eos_tokens"] = (response_ids == eos_token_id).sum().item()

        # Legacy reward stats (for backwards compatibility)
        stats["reward/mean"] = raw_rewards.mean().item()
        stats["reward/std"] = raw_rewards.std().item()

        # Toxicity evaluation on step generations
        if self.evaluator is not None and getattr(self.args, 'eval_on_step_generations', True):
            eval_results = self.evaluator.evaluate_generations(response_texts)
            stats["eval/toxicity_prob"] = eval_results["mean_toxicity_prob"]
            stats["eval/toxicity_rate"] = eval_results["toxicity_rate"]

        return stats

    def _update_history(
        self,
        history: Dict[str, List[float]],
        stats: Dict[str, float],
        is_full_eval: bool = False,
        global_step: Optional[int] = None,
    ):
        """
        Update training history with statistics.

        Args:
            history: History dictionary to update
            stats: Statistics to add
            is_full_eval: Whether this is from a full evaluation
            global_step: Current step (required for full_eval)
        """
        if is_full_eval:
            if "full_eval_toxicity_prob" not in history:
                history["full_eval_toxicity_prob"] = []
                history["full_eval_toxicity_rate"] = []
                history["full_eval_steps"] = []
            history["full_eval_toxicity_prob"].append(stats["mean_toxicity_prob"])
            history["full_eval_toxicity_rate"].append(stats["toxicity_rate"])
            history["full_eval_steps"].append(global_step)
        else:
            history["loss"].append(stats["loss/total"])
            history["reward"].append(stats["reward/mean"])
            history["kl"].append(stats["objective/kl"])

            if "eval/toxicity_prob" in stats:
                if "toxicity_prob" not in history:
                    history["toxicity_prob"] = []
                    history["toxicity_rate"] = []
                history["toxicity_prob"].append(stats["eval/toxicity_prob"])
                history["toxicity_rate"].append(stats["eval/toxicity_rate"])

    def _early_stop(self, policy_kl: float) -> bool:
        """
        Check if early stopping should be triggered based on policy KL.

        If the policy KL exceeds 1.5 * target_kl, zero the gradients and skip
        the optimization step. This prevents the policy from diverging too far
        from the reference model.

        Args:
            policy_kl: The current policy KL divergence

        Returns:
            True if early stopping was triggered, False otherwise
        """
        if not self.early_stopping:
            return False

        if policy_kl > 1.5 * self.target_kl:
            self.optimizer.zero_grad()
            logger.warning(
                f"Early stopping triggered: policy KL ({policy_kl:.4f}) > "
                f"1.5 * target_kl ({1.5 * self.target_kl:.4f})"
            )
            return True

        return False

    def _check_divergence_warnings(self, stats: Dict[str, float]) -> None:
        """Check for critical training issues (NaN/Inf losses, high ratio)."""
        # Check for NaN/Inf losses (critical)
        for key in ["loss/total", "loss/policy", "loss/value"]:
            if key in stats:
                val = stats[key]
                if not np.isfinite(val):
                    logger.warning(f"CRITICAL: {key} is {val}")

        # Check policy ratio (precedes loss explosion)
        ratio = stats.get("policy/ratio", 1.0)
        if ratio > 5.0:
            logger.warning(f"High policy ratio ({ratio:.2f}): policy diverging from rollout")

    def train(
        self,
        train_dataloader,
        num_epochs: int = 1,
        max_steps: Optional[int] = None,
        log_interval: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Main training loop with self-referencing validation (Snapshot/IIF style).

        Args:
            train_dataloader: DataLoader for training prompts
            num_epochs: Number of training epochs
            max_steps: Maximum steps (None = no limit)
            log_interval: Steps between logging

        Returns:
            Dictionary of training history
        """
        logger.info(f"Starting training with method={self.method}")
        if self.method != "NA":
            if getattr(self.args, 'use_fixed_validation', False) and self.val_dataset is not None:
                logger.info(f"Using fixed validation dataset (n_val={len(self.val_dataset)})")
            else:
                logger.info("Using self-reference validation (training buffer as validation set)")

        history = {"loss": [], "reward": [], "kl": []}
        global_step = 0
        logged_initial_stats = False

        for epoch in range(num_epochs):
            self.model.train()
            epoch_stats = []
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")

            for batch in pbar:
                # Prepare batch and generate rollout data
                query_ids, query_mask = self._prepare_batch(batch)
                rollout_data = self._generate_rollout_data(query_ids, query_mask)
                response_mask = rollout_data["response_mask"]

                # Log initial stats (Step 0) before any training
                if not logged_initial_stats:
                    initial_stats = self._compute_initial_stats(
                        query_ids, rollout_data["response_ids"], query_mask, response_mask,
                        rollout_data["old_logprobs"], rollout_data["advantages"],
                        rollout_data["returns"], rollout_data["old_values"],
                        rollout_data["raw_rewards"], rollout_data["kl_penalty"],
                        rollout_data["response_texts"],
                    )
                    log_msg = (
                        f"Step 0: "
                        f"loss={initial_stats['loss/total']:.3f} "
                        f"reward={initial_stats['reward/mean']:.2f} "
                        f"kl={initial_stats['objective/kl']:.3f}"
                    )
                    if "eval/toxicity_prob" in initial_stats:
                        log_msg += f" tox={initial_stats['eval/toxicity_prob']:.3f}"
                    # Append stats dict for result.ipynb parsing
                    log_msg += f" {initial_stats}"
                    logger.info(log_msg)
                    self._update_history(history, initial_stats)
                    logged_initial_stats = True

                # Capture validation gradients for selection methods
                if self.method != "NA":
                    if getattr(self.args, 'use_fixed_validation', False) and self.val_dataset is not None:
                        # Use fixed validation dataset (separate from training data)
                        self.capture_validation_gradients_from_fixed_val(global_step)
                    else:
                        # Use self-referencing validation (same rollout data) per LDA-ORL implementation
                        self.capture_validation_gradients_from_buffer(query_ids, query_mask, rollout_data)

                # Run PPO training epochs
                ppo_stats = self._run_ppo_epochs(query_ids, query_mask, rollout_data)

                # Clear validation gradient buffers to free GPU memory before next rollout
                # This prevents OOM during generation by releasing stored gradient tensors
                if self.method != "NA" and self.grad_hook is not None:
                    self.grad_hook.clear_val_buffer()

                # Build complete step statistics
                stats = self._build_step_stats(ppo_stats, rollout_data, response_mask)

                # Update KL controller
                mean_kl = stats["objective/kl"]
                self.kl_ctl.update(mean_kl, query_ids.shape[0])

                # Check for critical divergence issues (NaN/Inf only)
                self._check_divergence_warnings(stats)

                epoch_stats.append(stats)
                global_step += 1

                # LR scheduler step
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Update progress bar with key metrics
                postfix = {
                    "loss": f"{stats['loss/total']:.3f}",
                    "reward": f"{stats['reward/mean']:.2f}",
                    "kl": f"{stats['objective/kl']:.3f}",
                }
                if "eval/toxicity_prob" in stats:
                    postfix["tox"] = f"{stats['eval/toxicity_prob']:.3f}"
                pbar.set_postfix(postfix)

                # Periodic logging (concise format + dict for parsing)
                if global_step % log_interval == 0:
                    avg_stats = {k: float(np.mean([s[k] for s in epoch_stats[-log_interval:]])) for k in stats.keys()}
                    log_msg = (
                        f"Step {global_step}: "
                        f"loss={avg_stats['loss/total']:.3f} "
                        f"reward={avg_stats['reward/mean']:.2f} "
                        f"kl={avg_stats['objective/kl']:.3f} "
                        f"clipfrac={avg_stats.get('policy/clipfrac', 0):.2f}"
                    )
                    if "eval/toxicity_prob" in avg_stats:
                        log_msg += f" tox={avg_stats['eval/toxicity_prob']:.3f}"
                    # Append stats dict for result.ipynb parsing
                    log_msg += f" {avg_stats}"
                    logger.info(log_msg)
                    self._update_history(history, avg_stats)

                # Periodic full evaluation
                eval_interval = getattr(self.args, 'eval_interval', 1)
                if self.evaluator is not None and eval_interval > 0 and global_step % eval_interval == 0:
                    self._run_full_evaluation(global_step, history)

                # Check max steps
                if max_steps is not None and global_step >= max_steps:
                    logger.info(f"Reached max_steps ({max_steps})")
                    break

            # End of epoch evaluation
            if self.evaluator is not None and getattr(self.args, 'eval_interval', 0) == 0:
                self._run_full_evaluation(global_step, history)

            if max_steps is not None and global_step >= max_steps:
                break

        logger.info("Training complete")
        return history

    def _run_full_evaluation(
        self,
        global_step: int,
        history: Dict[str, List[float]],
    ):
        """
        Run full toxicity evaluation on test prompts.

        This uses a separate toxicity classifier (DaNLP/da-electra-hatespeech-detection)
        to evaluate the model's toxicity on a held-out set of prompts.

        Args:
            global_step: Current training step
            history: Training history dictionary to update
        """
        # Disable gradient hooks during evaluation to avoid overhead
        if self.grad_hook is not None:
            self.grad_hook.disable_hooks()

        eval_n_samples = getattr(self.args, 'eval_n_samples', 100)
        eval_batch_size = getattr(self.args, 'eval_batch_size', 16)

        # Run evaluation
        eval_results = self.evaluator.evaluate(
            model=self.model,
            tokenizer=self.tokenizer,
            n_samples=eval_n_samples,
            max_new_tokens=self.args.max_new_tokens,
            min_new_tokens=self.args.min_new_tokens,  # Safe for eval (no KL computation)
            generation_batch_size=eval_batch_size,
            temperature=getattr(self.args, 'temperature', 0.7),
            top_p=getattr(self.args, 'top_p', 0.9),
        )

        # Log results (single line)
        logger.info(
            f"[Eval step {global_step}] toxicity: {eval_results['mean_toxicity_prob']:.4f} (rate={eval_results['toxicity_rate']:.1%})"
        )

        # Update history with full evaluation results
        if "full_eval_toxicity_prob" not in history:
            history["full_eval_toxicity_prob"] = []
            history["full_eval_toxicity_rate"] = []
            history["full_eval_steps"] = []

        history["full_eval_toxicity_prob"].append(eval_results["mean_toxicity_prob"])
        history["full_eval_toxicity_rate"].append(eval_results["toxicity_rate"])
        history["full_eval_steps"].append(global_step)

        # Re-enable hooks after evaluation if selection method is active
        if self.grad_hook is not None and self.method != "NA":
            self.grad_hook.enable_hooks()

    def save_model(self, output_dir: str):
        """Save model to directory."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
