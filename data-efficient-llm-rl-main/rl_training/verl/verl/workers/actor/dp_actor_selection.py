# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor with Online Data Selection (GREATS/Streaming).

This extends DataParallelPPOActor to support gradient-based online data selection:
- GREATS: Per-step global selection using gradient similarity
- Streaming: Per-step, per-layer selection using gradient similarity

Key difference from baseline DOTS:
- DOTS: Uses difficulty to select training samples offline (per-epoch)
- GREATS/Streaming: Uses difficulty to select validation set dynamically,
  then uses gradient similarity for training sample selection (per-step)
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Iterable, Tuple, Optional, Dict, Any, List

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

import logging

logger = logging.getLogger(__name__)

__all__ = ['DataParallelPPOActorWithSelection']


def get_trainable_linear_layers(model: nn.Module) -> List[str]:
    """Get names of trainable Linear layers, excluding embeddings and lm_head."""
    layer_names = []
    exclude_patterns = ['embed', 'lm_head', 'norm', 'wte', 'wpe']

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(pattern in name.lower() for pattern in exclude_patterns):
                continue
            if any(p.requires_grad for p in module.parameters()):
                layer_names.append(name)

    return layer_names


def compute_goldilocks_scores(difficulties: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Compute Goldilocks zone scores: -|difficulty - alpha|."""
    return -torch.abs(difficulties - alpha)


def select_validation_indices(
    difficulties: torch.Tensor,
    num_to_select: int,
    alpha: float = 0.5,
    tau: float = 0.1,
    mode: str = 'soft',
) -> torch.Tensor:
    """Select validation indices using DOTS criterion."""
    scores = compute_goldilocks_scores(difficulties, alpha)

    if mode == 'hard':
        _, indices = torch.topk(scores, k=num_to_select)
    else:
        logits = scores / tau
        logits = logits - logits.max()
        probs = torch.softmax(logits, dim=0)
        indices = torch.multinomial(probs, num_samples=num_to_select, replacement=False)

    return indices.sort()[0]


class DataParallelPPOActorWithSelection(DataParallelPPOActor):
    """
    PPO Actor with GREATS/Streaming online data selection.

    Inherits from DataParallelPPOActor and adds:
    1. GradientHook for gradient-based selection
    2. Dynamic validation set selection based on DOTS difficulty
    3. GREATS: Global selection per training step
    4. Streaming: Per-layer selection during backward

    Selection Flow:
    1. At start of update_policy: compute difficulties, select validation set
    2. Capture validation gradients using sequence-level advantage target
    3. During PPO update loop:
       - GREATS: Two-pass selection per mini-batch
       - Streaming: Per-layer selection during backward
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)

        # Selection configuration
        self.selection_method = config.get('selection_method', 'NA')
        self.selection_alpha = config.get('selection_alpha', 0.5)
        self.selection_tau = config.get('selection_tau', 0.1)
        self.val_ratio = config.get('val_ratio', 0.2)
        self.train_selection_ratio = config.get('train_selection_ratio', 0.5)
        self.val_selection_mode = config.get('val_selection_mode', 'soft')
        self.use_second_order = config.get('use_second_order', False)

        # GradientHook (lazy initialization)
        self.grad_hook = None
        self._hook_initialized = False

        # Selection statistics
        self.selection_stats: Dict[str, Any] = {}

        # Initialize hook if needed
        if self.selection_method in ['GREATS', 'Streaming']:
            self._initialize_grad_hook()

        logger.info(
            f"Initialized DataParallelPPOActorWithSelection: "
            f"method={self.selection_method}, alpha={self.selection_alpha}, "
            f"tau={self.selection_tau}, val_ratio={self.val_ratio}"
        )

    def _initialize_grad_hook(self) -> None:
        """Initialize GradientHook for gradient-based selection."""
        if self._hook_initialized:
            return

        try:
            from gradstream.hook import GradientHook
        except ImportError:
            logger.error(
                "gradstream not found. Install gradstream for GREATS/Streaming selection. "
                "Falling back to NA (no selection)."
            )
            self.selection_method = 'NA'
            return

        # Get trainable layer names
        layer_names = get_trainable_linear_layers(self.actor_module)

        if len(layer_names) == 0:
            logger.warning("No trainable Linear layers found, falling back to NA selection")
            self.selection_method = 'NA'
            return

        # Create GradientHook
        self.grad_hook = GradientHook(
            model=self.actor_module,
            layer_names=layer_names,
            device='cuda',
        )

        self._hook_initialized = True
        logger.info(f"Initialized GradientHook with {len(layer_names)} layers")

    def _is_selection_enabled(self) -> bool:
        """Check if gradient-based selection is enabled."""
        return self.selection_method in ['GREATS', 'Streaming'] and self._hook_initialized

    def _compute_sample_difficulties(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-sample difficulty from rewards.

        Normalizes rewards to [0, 1] range.

        Args:
            rewards: [batch_size] or [batch_size, seq_len] reward tensor

        Returns:
            [batch_size] normalized difficulty scores
        """
        if rewards.dim() > 1:
            rewards = rewards.sum(dim=-1)

        r_min, r_max = rewards.min(), rewards.max()
        if r_max - r_min > 1e-8:
            difficulties = (rewards - r_min) / (r_max - r_min)
        else:
            difficulties = torch.ones_like(rewards) * 0.5

        return difficulties

    def _select_validation_set(
        self,
        batch_size: int,
        difficulties: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select validation set indices using DOTS difficulty criterion.

        Args:
            batch_size: Total batch size
            difficulties: [batch_size] difficulty scores

        Returns:
            [num_val] indices for validation set
        """
        num_val = max(1, int(batch_size * self.val_ratio))

        val_indices = select_validation_indices(
            difficulties=difficulties,
            num_to_select=num_val,
            alpha=self.selection_alpha,
            tau=self.selection_tau,
            mode=self.val_selection_mode,
        )

        # Track stats
        selected_diff = difficulties[val_indices]
        self.selection_stats['val/num_selected'] = len(val_indices)
        self.selection_stats['val/difficulty_mean'] = selected_diff.mean().item()

        return val_indices

    def _capture_validation_gradients(
        self,
        val_batch: Dict[str, torch.Tensor],
        temperature: float,
    ) -> None:
        """
        Capture validation gradients using sequence-level advantage target.

        Target: L_val = -E[log π_θ(y|x) * A(x,y)]

        Args:
            val_batch: Dictionary with validation samples
            temperature: Temperature for log prob computation
        """
        if not self._is_selection_enabled():
            return

        # Extract tensors
        val_input_ids = val_batch['input_ids']
        val_attention_mask = val_batch['attention_mask']
        val_responses = val_batch['responses']
        val_advantages = val_batch['advantages']

        response_length = val_responses.size(1)
        response_mask = val_attention_mask[:, -response_length:]

        # Compute sequence-level advantages
        seq_advantages = (val_advantages * response_mask.float()).sum(dim=1)

        # Start validation gradient capture
        self.grad_hook.start_val_capture(use_factorized=False)

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=val_input_ids,
                attention_mask=val_attention_mask,
                use_cache=False,
            )
            logits = output.logits

            if temperature != 1.0:
                logits = logits / temperature

            # Get log probs for response tokens
            logits = logits[:, -response_length - 1:-1, :]
            log_probs = F.log_softmax(logits.float(), dim=-1)
            token_log_probs = torch.gather(
                log_probs, dim=-1, index=val_responses.unsqueeze(-1)
            ).squeeze(-1)

            # Sequence log probability
            seq_log_probs = (token_log_probs * response_mask.float()).sum(dim=1)

            # Validation loss: -E[log π_θ(y|x) * A(x,y)]
            per_seq_loss = -(seq_advantages * seq_log_probs)
            val_loss = per_seq_loss.mean()

            # Backward to capture gradients
            val_loss.backward()

        # End capture
        self.grad_hook.end_val_capture()

        logger.debug(f"Captured validation gradients for {val_input_ids.size(0)} samples")

    def _select_training_batch_greats(
        self,
        train_batch: Dict[str, torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        """
        Select training samples using GREATS (global gradient-based selection).

        Args:
            train_batch: Dictionary with training samples
            temperature: Temperature for log prob computation

        Returns:
            [num_selected] indices of selected training samples
        """
        batch_size = train_batch['input_ids'].size(0)

        if not self._is_selection_enabled():
            return torch.arange(batch_size, device=train_batch['input_ids'].device)

        # Setup GREATS state
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method='GREATS',
            frac=self.train_selection_ratio,
            lr=1.0,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode='topk',
        )

        # Zero gradients
        self.actor_module.zero_grad()

        # Extract tensors
        input_ids = train_batch['input_ids']
        attention_mask = train_batch['attention_mask']
        responses = train_batch['responses']
        old_log_prob = train_batch['old_log_probs']
        advantages = train_batch['advantages']

        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]

        clip_ratio = self.config.clip_ratio

        # Forward pass
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = output.logits

            if temperature != 1.0:
                logits = logits / temperature

            logits = logits[:, -response_length - 1:-1, :]
            log_prob = logprobs_from_logits(logits, responses)

            # Compute PPO loss for scoring
            pg_loss, _, _ = core_algos.compute_policy_loss(
                old_log_prob=old_log_prob,
                log_prob=log_prob,
                advantages=advantages,
                eos_mask=response_mask,
                cliprange=clip_ratio,
            )

            # Backward to accumulate scores
            pg_loss.backward()

        # Get selected indices
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        selected_indices = selected_indices.sort()[0]

        # Cleanup
        self.grad_hook.clear_selection()

        # Track stats
        self.selection_stats['train/num_selected'] = len(selected_indices)
        self.selection_stats['train/selection_ratio'] = len(selected_indices) / batch_size

        return selected_indices

    def _setup_streaming(self, batch_size: int, labels: Optional[torch.Tensor] = None) -> None:
        """Setup Streaming selection state before forward pass."""
        if not self._is_selection_enabled():
            return

        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method='Streaming',
            frac=self.train_selection_ratio,
            lr=1.0,
            compute_scores_only=False,
            use_second_order=self.use_second_order,
            selection_mode='topk',
        )

        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size)

    def _cleanup_streaming(self) -> None:
        """Cleanup after streaming backward."""
        if not self._is_selection_enabled():
            return

        # Get stats from streaming state
        sel_state = self.grad_hook.selection_state
        if hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
            n_selected = [n for _, n in sel_state._layer_selections]
            self.selection_stats['train/mean_selected_per_layer'] = sum(n_selected) / len(n_selected)

        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()

    def update_policy(self, data: DataProto):
        """
        Update policy with GREATS/Streaming data selection.

        This extends the base update_policy with:
        1. Validation set selection based on DOTS difficulty
        2. Validation gradient capture
        3. Per-step GREATS or Streaming selection during PPO updates
        """
        # If selection not enabled, use base implementation
        if not self._is_selection_enabled():
            return super().update_policy(data)

        # Make sure we're in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']

        # Select keys
        select_keys = [
            'responses', 'input_ids', 'attention_mask', 'position_ids',
            'old_log_probs', 'advantages'
        ]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        # =====================================================================
        # Step 1: Select validation set based on DOTS difficulty criterion
        # =====================================================================
        batch_size = batch.batch_size[0]

        # Compute difficulties from advantages (proxy for reward signal)
        advantages = batch['advantages']
        response_length = batch['responses'].size(1)
        attention_mask = batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        # Use mean advantage per sample as difficulty proxy
        seq_advantages = (advantages * response_mask.float()).sum(dim=1) / response_mask.float().sum(dim=1).clamp(min=1)
        difficulties = self._compute_sample_difficulties(seq_advantages)

        # Select validation indices
        val_indices = self._select_validation_set(batch_size, difficulties)

        # =====================================================================
        # Step 2: Capture validation gradients
        # =====================================================================
        val_batch = {
            'input_ids': batch['input_ids'][val_indices],
            'attention_mask': batch['attention_mask'][val_indices],
            'responses': batch['responses'][val_indices],
            'advantages': batch['advantages'][val_indices],
        }

        # Capture validation gradients
        self._capture_validation_gradients(val_batch, temperature)

        # =====================================================================
        # Step 3: PPO update with selection
        # =====================================================================
        # Create training indices (all samples, selection happens per step)
        train_indices = torch.arange(batch_size, device=batch['input_ids'].device)

        # Create dataloader
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                # Handle dynamic batch sizing
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.cuda()

                    # Extract data
                    responses = micro_batch['responses']
                    resp_length = responses.size(1)
                    attn_mask = micro_batch['attention_mask']
                    resp_mask = attn_mask[:, -resp_length:]
                    old_log_prob = micro_batch['old_log_probs']
                    advs = micro_batch['advantages']

                    micro_batch_size = responses.size(0)

                    # Create train batch dict
                    train_batch_dict = {
                        'input_ids': micro_batch['input_ids'],
                        'attention_mask': attn_mask,
                        'responses': responses,
                        'old_log_probs': old_log_prob,
                        'advantages': advs,
                    }

                    if self.selection_method == 'GREATS':
                        # GREATS: Select samples, then train on selected
                        selected_indices = self._select_training_batch_greats(
                            train_batch_dict, temperature
                        )

                        # Filter batch to selected samples
                        sel_input_ids = micro_batch['input_ids'][selected_indices]
                        sel_attention_mask = attn_mask[selected_indices]
                        sel_responses = responses[selected_indices]
                        sel_old_log_prob = old_log_prob[selected_indices]
                        sel_advantages = advs[selected_indices]
                        sel_response_mask = resp_mask[selected_indices]

                        # Forward on selected
                        entropy, log_prob = self._forward_micro_batch(
                            {
                                'input_ids': sel_input_ids,
                                'attention_mask': sel_attention_mask,
                                'position_ids': micro_batch['position_ids'][selected_indices],
                                'responses': sel_responses,
                            },
                            temperature=temperature,
                        )

                        # Compute loss
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=sel_old_log_prob,
                            log_prob=log_prob,
                            advantages=sel_advantages,
                            eos_mask=sel_response_mask,
                            cliprange=self.config.clip_ratio,
                        )

                        entropy_loss = verl_F.masked_mean(entropy, sel_response_mask)
                        policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = micro_batch['ref_log_prob'][selected_indices]
                            kld = core_algos.kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = masked_mean(kld, sel_response_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics['actor/kl_loss'] = kl_loss.detach().item()
                            metrics['actor/kl_coef'] = self.config.kl_loss_coef

                        loss = policy_loss / self.gradient_accumulation
                        loss.backward()

                    elif self.selection_method == 'Streaming':
                        # Streaming: Setup selection, forward/backward does selection per-layer
                        self._setup_streaming(micro_batch_size, responses)

                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch, temperature=temperature
                        )

                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advs,
                            eos_mask=resp_mask,
                            cliprange=self.config.clip_ratio,
                        )

                        entropy_loss = verl_F.masked_mean(entropy, resp_mask)
                        policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = micro_batch['ref_log_prob']
                            kld = core_algos.kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = masked_mean(kld, resp_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics['actor/kl_loss'] = kl_loss.detach().item()
                            metrics['actor/kl_coef'] = self.config.kl_loss_coef

                        loss = policy_loss / self.gradient_accumulation
                        # Streaming selection happens during backward
                        loss.backward()

                        self._cleanup_streaming()

                    else:
                        # Fallback: standard update without selection
                        entropy, log_prob = self._forward_micro_batch(
                            micro_batch, temperature=temperature
                        )

                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advs,
                            eos_mask=resp_mask,
                            cliprange=self.config.clip_ratio,
                        )

                        entropy_loss = verl_F.masked_mean(entropy, resp_mask)
                        policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = micro_batch['ref_log_prob']
                            kld = core_algos.kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = masked_mean(kld, resp_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics['actor/kl_loss'] = kl_loss.detach().item()
                            metrics['actor/kl_coef'] = self.config.kl_loss_coef

                        loss = policy_loss / self.gradient_accumulation
                        loss.backward()

                    # Track metrics
                    step_data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, step_data)

                # Optimizer step
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

        # Clear validation gradients
        if self._is_selection_enabled():
            self.grad_hook.clear_val_gradients()

        # Add selection stats to metrics
        for key, val in self.selection_stats.items():
            metrics[f'selection/{key}'] = val

        self.actor_optimizer.zero_grad()
        return metrics
