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
Single Process Actor with Streaming/GREATS Online Data Selection.

This extends DataParallelPPOActor to support gradient-based selection:
- Streaming: Per-layer selection during backward using gradient similarity
- GREATS: Two-pass global selection using accumulated gradient similarity

The validation gradients are captured from a fixed validation set (seqloss-reward)
and used to compute gradient similarity scores for training sample selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any, List

import torch
import torch.distributed as dist
from torch import nn

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.torch_functional import logprobs_from_logits

try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
except ImportError:
    pad_input = unpad_input = rearrange = index_first_axis = None

import logging

from .hook import GradientHookVerl

logger = logging.getLogger(__name__)


def _get_rank() -> int:
    """Get current distributed rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def _get_world_size() -> int:
    """Get distributed world size."""
    return dist.get_world_size() if dist.is_initialized() else 1


__all__ = ['DataParallelPPOActorWithSelection']


def get_trainable_linear_layers(model: nn.Module, skip_grad_check: bool = False) -> List[str]:
    """Get names of trainable Linear layers, excluding embeddings and lm_head."""
    layer_names = []
    exclude_patterns = ['embed', 'lm_head', 'norm', 'wte', 'wpe']

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(pattern in name.lower() for pattern in exclude_patterns):
                continue
            if skip_grad_check or any(p.requires_grad for p in module.parameters()):
                layer_names.append(name)

    return layer_names


class DataParallelPPOActorWithSelection(DataParallelPPOActor):
    """
    PPO Actor with Streaming/GREATS online data selection.

    Inherits from DataParallelPPOActor and adds:
    1. GradientHook for gradient-based selection
    2. External validation gradient capture (seqloss-reward)
    3. Streaming: Per-layer selection during backward
    4. GREATS: Two-pass global selection per mini-batch

    Flow:
    1. Before training: capture_validation_gradients_external() with fixed val set
    2. During update_policy():
       - Streaming: Per-layer selection during backward
       - GREATS: Pass 1 accumulates scores, Pass 2 trains on selected samples
    """

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        selection_method: str = 'NA',
        train_selection_ratio: float = 0.5,
        use_second_order: bool = False,
    ):
        super().__init__(config, actor_module, actor_optimizer)

        # Selection configuration (passed as kwargs, not from config)
        self.selection_method = selection_method
        self.train_selection_ratio = train_selection_ratio
        self.use_second_order = use_second_order

        # GradientHook (lazy initialization)
        self.grad_hook = None
        self._hook_initialized = False

        # Selection statistics
        self.selection_stats: Dict[str, Any] = {}

        # Initialize hook if Streaming or GREATS is enabled
        if self.selection_method in ['Streaming', 'GREATS']:
            self._initialize_grad_hook()

    def _initialize_grad_hook(self) -> None:
        """Initialize GradientHook for gradient-based selection."""
        if self._hook_initialized:
            return

        # Try to get unwrapped module for FSDP
        unwrapped_module = self.actor_module
        is_fsdp = hasattr(self.actor_module, '_fsdp_wrapped_module')
        if is_fsdp:
            unwrapped_module = self.actor_module._fsdp_wrapped_module

        # Get trainable layer names
        layer_names = get_trainable_linear_layers(unwrapped_module, skip_grad_check=is_fsdp)

        if len(layer_names) == 0:
            logger.warning("No trainable Linear layers found, falling back to NA selection")
            self.selection_method = 'NA'
            return

        # Create GradientHook
        self.grad_hook = GradientHookVerl(
            model=unwrapped_module,
            layer_names=layer_names,
            device='cuda',
        )

        self._hook_initialized = True
        logger.info(f"Initialized GradientHook with {len(layer_names)} layers")

    def _is_selection_enabled(self) -> bool:
        """Check if gradient-based selection is enabled."""
        return self.selection_method in ['Streaming', 'GREATS'] and self._hook_initialized

    def has_external_validation_gradients(self) -> bool:
        """Check if validation gradients were captured externally."""
        if not self._is_selection_enabled():
            return False
        return self.grad_hook._val_cache.get_num_captured() > 0

    def capture_validation_gradients_external(
        self,
        val_input_ids: torch.Tensor,
        val_attention_mask: torch.Tensor,
        val_position_ids: torch.Tensor,
        val_responses: torch.Tensor,
        val_rewards: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Capture validation gradients from external validation data.

        Uses seqloss-reward objective: L = -E[normalize(reward) * log_prob]

        Args:
            val_input_ids: [batch, seq_len] full input (prompt + response)
            val_attention_mask: [batch, seq_len] attention mask
            val_position_ids: [batch, seq_len] position IDs
            val_responses: [batch, response_len] response tokens
            val_rewards: [batch] rewards for each sample
            temperature: Temperature for log prob computation

        Returns:
            Dictionary with validation capture statistics
        """
        if not self._is_selection_enabled():
            return {}

        rank = _get_rank()
        world_size = _get_world_size()

        self.actor_module.train()
        batch_size = val_input_ids.size(0)
        response_length = val_responses.size(1)

        # Normalize rewards to get sequence scores (seqloss-reward objective)
        if batch_size > 1:
            seq_scores = (val_rewards - val_rewards.mean()) / (val_rewards.std() + 1e-8)
        else:
            seq_scores = val_rewards - val_rewards.mean()

        # Start validation gradient capture
        self.grad_hook.start_val_capture(use_factorized=False)
        self.grad_hook.enable_hooks()

        # Calculate micro_batch_size
        if self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * getattr(self, 'ulysses_sequence_parallel_size', 1)
            seq_lengths = val_attention_mask.sum(dim=1)
            max_seq_len = seq_lengths.max().item()
            micro_batch_size = max(1, int(max_token_len // max_seq_len))
        else:
            # Use ppo_micro_batch_size_per_gpu (the correct attribute name in verl config)
            micro_batch_size = getattr(self.config, 'ppo_micro_batch_size_per_gpu', None)
            if micro_batch_size is None:
                # Fallback to batch_size or a default
                micro_batch_size = min(batch_size, 8)  # Default to 8 or batch_size

        # Process in micro-batches
        # Use tensor for accumulation to avoid D2H transfers in the loop
        total_val_loss = torch.tensor(0.0, device=val_input_ids.device)
        num_micro_batches = (batch_size + micro_batch_size - 1) // micro_batch_size

        for mb_idx in range(num_micro_batches):
            start_idx = mb_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, batch_size)

            mb_input_ids = val_input_ids[start_idx:end_idx]
            mb_attention_mask = val_attention_mask[start_idx:end_idx]
            mb_position_ids = val_position_ids[start_idx:end_idx]
            mb_responses = val_responses[start_idx:end_idx]
            mb_seq_scores = seq_scores[start_idx:end_idx]

            mb_batch_size, mb_seqlen = mb_input_ids.shape
            mb_response_mask = mb_attention_mask[:, -response_length:]

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.use_remove_padding and unpad_input is not None:
                    # Pack sequences for memory-efficient forward pass
                    input_ids_rmpad, indices, *_ = unpad_input(
                        mb_input_ids.unsqueeze(-1), mb_attention_mask
                    )
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                    position_ids_rmpad = index_first_axis(
                        rearrange(mb_position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                        indices
                    ).transpose(0, 1)

                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)

                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                    )
                    logits_rmpad = output.logits.squeeze(0)

                    if temperature != 1.0:
                        logits_rmpad = logits_rmpad / temperature

                    log_probs_rmpad = logprobs_from_logits(logits_rmpad, input_ids_rmpad_rolled)

                    full_log_probs = pad_input(
                        hidden_states=log_probs_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=mb_batch_size,
                        seqlen=mb_seqlen
                    )
                    token_log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
                else:
                    output = self.actor_module(
                        input_ids=mb_input_ids,
                        attention_mask=mb_attention_mask,
                        position_ids=mb_position_ids,
                        use_cache=False,
                    )
                    logits = output.logits

                    if temperature != 1.0:
                        logits = logits / temperature

                    logits = logits[:, -response_length - 1:-1, :]
                    token_log_probs = logprobs_from_logits(logits.float(), mb_responses)

                # Sequence log probability
                seq_log_probs = (token_log_probs * mb_response_mask.float()).sum(dim=1)

                # Validation loss: L = -E[seq_scores * seq_log_probs]
                per_seq_loss = -(mb_seq_scores * seq_log_probs)
                val_loss = per_seq_loss.mean()

                # Backward to capture gradients
                val_loss.backward()

            total_val_loss += val_loss.detach() * (end_idx - start_idx)

            # Zero optimizer gradients but keep val cache accumulating
            self.actor_optimizer.zero_grad()

            # Clear intermediate tensors (no empty_cache - let PyTorch manage memory)
            del output, token_log_probs, seq_log_probs, per_seq_loss, val_loss

        # End capture
        self.grad_hook.end_val_capture()
        self.grad_hook.disable_hooks()

        # Synchronize validation gradients across all ranks
        # This ensures all ranks have the same validation gradient for consistent selection
        self.grad_hook.sync_val_grads()

        # Synchronize loss across ranks for consistent stats
        if world_size > 1:
            loss_tensor = torch.stack([total_val_loss, torch.tensor(float(batch_size), device=val_input_ids.device)])
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            # Single .item() calls at the end of capture (not in hot path)
            total_val_loss_float = loss_tensor[0].item()
            total_samples = int(loss_tensor[1].item())
        else:
            total_val_loss_float = total_val_loss.item()
            total_samples = batch_size

        # Get stats
        num_layers_captured = self.grad_hook._val_cache.get_num_captured()
        avg_val_loss = total_val_loss_float / total_samples if total_samples > 0 else 0.0

        stats = {
            'val/num_samples': total_samples,  # Total across all ranks after sync
            'val/loss': avg_val_loss,  # Average loss across all ranks
            'val/num_layers_captured': num_layers_captured,
            'val/num_micro_batches': num_micro_batches,
            'val/rank': rank,
            'val/world_size': world_size,
        }

        # Compute total validation gradient norm efficiently (no .item() calls in loop)
        grad_norms_sq = []
        for i in range(num_layers_captured):
            val_grad = self.grad_hook.get_val_grad(i)
            if val_grad is not None:
                grad_norms_sq.append(val_grad.norm().square())

        if grad_norms_sq:
            total_grad_norm_sq = torch.stack(grad_norms_sq).sum()
            total_grad_norm = total_grad_norm_sq.sqrt().item()  # Single .item() at the end
        else:
            total_grad_norm = 0.0

        stats['val/total_grad_norm'] = total_grad_norm

        return stats

    def clear_validation_gradients(self) -> None:
        """Clear cached validation gradients."""
        if self._is_selection_enabled():
            self.grad_hook.clear_val_buffer()

    def _setup_streaming(
        self,
        batch_size: int,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Setup Streaming selection state before forward pass.

        IMPORTANT: This should be called per micro-batch, not per full batch.
        The batch_size should be the micro-batch size to ensure:
        1. Selection indices are relative to the micro-batch
        2. tokens_per_sample matches the micro-batch for correct scale factor
        3. cu_seqlens matches the micro-batch for packed sequence handling
        """
        if not self._is_selection_enabled():
            return

        rank = _get_rank()

        num_val_layers = self.grad_hook._val_cache.get_num_captured()
        if num_val_layers == 0:
            logger.warning("No validation gradients captured! Selection may not work.")

        # Note: Removed verbose per-micro-batch logging to reduce noise
        # Log only on first setup or at DEBUG level
        logger.debug(
            f"[Stream] Setting up Streaming selection: "
            f"micro_batch_size={batch_size}, frac={self.train_selection_ratio}, "
            f"num_val_layers={num_val_layers}"
        )

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

        if labels is not None:
            self.grad_hook.set_token_counts(labels, batch_size, attention_mask)

    def _cleanup_streaming(self) -> None:
        """
        Cleanup after streaming backward for a single micro-batch.

        This is called per micro-batch. Stats are tracked but not logged
        per micro-batch to avoid overhead.
        """
        if not self._is_selection_enabled():
            return

        sel_state = self.grad_hook.selection_state
        if sel_state is not None and hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
            layer_selections = sel_state._layer_selections
            n_selected = [n for _, n in layer_selections]
            mean_selected = sum(n_selected) / len(n_selected) if n_selected else 0

            # Update stats for metrics reporting (no printing)
            self.selection_stats['train/mean_selected_per_layer'] = mean_selected
            self.selection_stats['train/num_layers_processed'] = len(n_selected)

        self.grad_hook.clear_selection()
        self.grad_hook.clear_token_counts()

    def _select_training_batch_greats(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        responses: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """
        Select training samples using GREATS (global gradient-based selection).

        Pass 1: Forward/backward to accumulate per-sample scores across all layers.
        The actual training (Pass 2) happens in the caller with selected indices.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            position_ids: [batch, seq_len]
            responses: [batch, response_len]
            old_log_probs: [batch, response_len]
            advantages: [batch, response_len]
            temperature: Temperature for log prob computation

        Returns:
            [num_selected] indices of selected training samples
        """
        from verl.trainer.ppo import core_algos

        batch_size = input_ids.size(0)
        response_length = responses.size(1)
        response_mask = attention_mask[:, -response_length:]

        if not self._is_selection_enabled():
            return torch.arange(batch_size, device=input_ids.device)

        # Setup GREATS state for Pass 1 (score accumulation)
        self.grad_hook.setup_selection_with_stored_val(
            train_batch_size=batch_size,
            selection_method='GREATS',
            frac=self.train_selection_ratio,
            lr=1.0,
            compute_scores_only=True,
            use_second_order=self.use_second_order,
            selection_mode='filtering',
        )
        self.grad_hook.enable_hooks()

        # Zero gradients before scoring pass
        self.actor_module.zero_grad()

        # Forward pass for scoring
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if self.use_remove_padding and unpad_input is not None:
                # Pack sequences
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                    indices
                ).transpose(0, 1)

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )
                logits_rmpad = output.logits.squeeze(0)

                if temperature != 1.0:
                    logits_rmpad = logits_rmpad / temperature

                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)
                log_probs_rmpad = logprobs_from_logits(logits_rmpad, input_ids_rmpad_rolled)

                full_log_probs = pad_input(
                    hidden_states=log_probs_rmpad.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=input_ids.size(1)
                )
                log_prob = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]
            else:
                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                )
                logits = output.logits

                if temperature != 1.0:
                    logits = logits / temperature

                logits = logits[:, -response_length - 1:-1, :]
                log_prob = logprobs_from_logits(logits.float(), responses)

            # Compute PPO loss for scoring
            pg_loss, _ = core_algos.compute_policy_loss_vanilla(
                old_log_prob=old_log_probs,
                log_prob=log_prob,
                advantages=advantages,
                response_mask=response_mask,
                loss_agg_mode=self.config.loss_agg_mode,
                config=self.config,
            )

            # Backward to accumulate scores
            pg_loss.backward()

        # Get selected indices from accumulated scores
        scores = self.grad_hook.selection_state.grad_dot_scores
        selected_indices = self.grad_hook.selection_state.get_final_selection()
        selected_indices = selected_indices.sort()[0]

        # Cleanup after Pass 1
        self.grad_hook.clear_selection()
        self.grad_hook.disable_hooks()
        self.actor_optimizer.zero_grad()

        # Track stats
        self.selection_stats['train/num_selected'] = len(selected_indices)
        self.selection_stats['train/selection_ratio'] = len(selected_indices) / batch_size
        if scores is not None:
            self.selection_stats['train/score_mean'] = scores.mean().item()
            self.selection_stats['train/score_std'] = scores.std().item()
            self.selection_stats['train/num_positive_scores'] = (scores > 0).sum().item()
            self.selection_stats['train/num_negative_scores'] = (scores < 0).sum().item()

        return selected_indices

    def update_policy(self, data: DataProto):
        """
        Update policy with Streaming/GREATS data selection.

        - Streaming: Per-layer selection during backward (hooks intercept gradients)
        - GREATS: Two-pass per micro-batch (Pass 1: score, Pass 2: train on selected)

        If selection is not enabled or no validation gradients captured,
        falls back to base implementation.
        """
        # If selection not enabled, use base implementation
        if not self._is_selection_enabled():
            return super().update_policy(data)

        # If no validation gradients, fall back to base
        if not self.has_external_validation_gradients():
            logger.warning("No validation gradients captured, using base update_policy")
            return super().update_policy(data)

        # For Streaming, we can use the base implementation with hooks enabled
        if self.selection_method == 'Streaming':
            return self._update_policy_streaming(data)
        elif self.selection_method == 'GREATS':
            return self._update_policy_greats(data)
        else:
            return super().update_policy(data)

    def _update_policy_streaming(self, data: DataProto):
        """
        Update policy with Streaming selection (per-layer during backward).

        Unlike the previous implementation that called super().update_policy(),
        this version handles micro-batches explicitly to ensure selection state
        (tokens_per_sample, cu_seqlens, etc.) is correctly set per micro-batch.

        This is critical because:
        1. Selection indices are relative to the micro-batch (0 to micro_batch_size-1)
        2. tokens_per_sample must match the micro-batch for correct scale factor computation
        3. cu_seqlens must match the micro-batch for packed sequence handling
        """
        from verl.trainer.ppo import core_algos
        from verl.utils.py_functional import append_to_dict
        import verl.utils.torch_functional as verl_F
        from verl.utils.torch_functional import masked_mean

        self.actor_module.train()

        # Get micro-batch configuration from verl config
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        assert self.config.ppo_mini_batch_size % micro_batch_size == 0
        gradient_accumulation = self.config.ppo_mini_batch_size // micro_batch_size
        temperature = data.meta_info.get('temperature', 1.0)
        pad_token_id = data.meta_info.get('pad_token_id', 0)

        # Select keys needed for training
        select_keys = [
            'responses', 'input_ids', 'attention_mask', 'position_ids',
            'old_log_probs', 'advantages'
        ]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')

        batch = data.select(batch_keys=select_keys).batch

        # Create dataloader: first split into mini-batches, then micro-batches
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        total_selected = 0
        total_samples = 0

        rank = _get_rank()

        for _ in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                micro_batches = mini_batch.split(micro_batch_size)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.cuda()

                    # Extract data for this micro-batch
                    input_ids = micro_batch['input_ids']
                    attention_mask = micro_batch['attention_mask']
                    position_ids = micro_batch['position_ids']
                    responses = micro_batch['responses']
                    old_log_probs = micro_batch['old_log_probs']
                    advantages = micro_batch['advantages']

                    current_micro_batch_size = input_ids.size(0)
                    response_length = responses.size(1)
                    response_mask = attention_mask[:, -response_length:]

                    # Create labels for token counting (response positions only)
                    labels = torch.full_like(attention_mask, -100)
                    labels[:, -response_length:] = responses * response_mask.long() + (-100) * (~response_mask.bool()).long()

                    # Setup streaming selection for THIS micro-batch
                    # This ensures tokens_per_sample and cu_seqlens match the micro-batch
                    self._setup_streaming(current_micro_batch_size, labels, attention_mask)

                    try:
                        # Forward pass with hooks enabled
                        with torch.autocast(device_type='cuda', dtype=self.param_dtype):
                            if self.use_remove_padding and unpad_input is not None:
                                # Packed sequences path
                                input_ids_rmpad, indices, *_ = unpad_input(
                                    input_ids.unsqueeze(-1), attention_mask
                                )
                                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                                position_ids_rmpad = index_first_axis(
                                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                    indices
                                ).transpose(0, 1)

                                output = self.actor_module(
                                    input_ids=input_ids_rmpad,
                                    attention_mask=None,
                                    position_ids=position_ids_rmpad,
                                    use_cache=False,
                                )
                                logits_rmpad = output.logits.squeeze(0)

                                if temperature != 1.0:
                                    logits_rmpad = logits_rmpad / temperature

                                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)
                                log_probs_rmpad = logprobs_from_logits(logits_rmpad, input_ids_rmpad_rolled)

                                full_log_probs = pad_input(
                                    hidden_states=log_probs_rmpad.unsqueeze(-1),
                                    indices=indices,
                                    batch=current_micro_batch_size,
                                    seqlen=input_ids.size(1)
                                )
                                log_prob = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]

                                # Compute entropy from packed logits first, then pad
                                # (DON'T pad logits - they have vocab_size dimension which would be huge)
                                entropy_rmpad = verl_F.entropy_from_logits(logits_rmpad)  # [total_nnz]
                                full_entropy = pad_input(
                                    hidden_states=entropy_rmpad.unsqueeze(-1),  # [total_nnz, 1]
                                    indices=indices,
                                    batch=current_micro_batch_size,
                                    seqlen=input_ids.size(1)
                                )
                                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # [batch, response_len]
                            else:
                                # Standard (non-packed) path
                                output = self.actor_module(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    use_cache=False,
                                )
                                logits = output.logits

                                if temperature != 1.0:
                                    logits = logits / temperature

                                logits = logits[:, -response_length - 1:-1, :]
                                log_prob = logprobs_from_logits(logits.float(), responses)
                                entropy = verl_F.entropy_from_logits(logits)

                        # Compute PPO loss (on all samples - selection happens in backward)
                        pg_loss, pg_metrics = core_algos.compute_policy_loss_vanilla(
                            old_log_prob=old_log_probs,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                            config=self.config,
                        )
                        pg_clipfrac = pg_metrics.get('pg_clipfrac', 0.0)
                        ppo_kl = pg_metrics.get('ppo_kl', 0.0)

                        entropy_loss = verl_F.masked_mean(entropy, response_mask)
                        policy_loss = pg_loss - entropy_loss * self.config.entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = micro_batch['ref_log_prob']
                            kld = core_algos.kl_penalty(
                                logprob=log_prob,
                                ref_logprob=ref_log_prob,
                                kl_penalty=self.config.kl_loss_type,
                            )
                            kl_loss = masked_mean(kld, response_mask)
                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics['actor/kl_loss'] = metrics.get('actor/kl_loss', 0) + kl_loss.detach().item() / gradient_accumulation

                        loss = policy_loss / gradient_accumulation

                        # Backward pass - hooks intercept gradients and apply selection
                        loss.backward()

                        # Track selection statistics for this micro-batch
                        sel_state = self.grad_hook.selection_state
                        if sel_state is not None and hasattr(sel_state, '_layer_selections') and sel_state._layer_selections:
                            n_selected = [n for _, n in sel_state._layer_selections]
                            mean_selected = sum(n_selected) / len(n_selected) if n_selected else 0
                            total_selected += mean_selected
                            total_samples += current_micro_batch_size

                    finally:
                        # Cleanup streaming state after each micro-batch
                        self._cleanup_streaming()
                        self.grad_hook.disable_hooks()

                    # Track metrics for this micro-batch
                    # Note: pg_clipfrac and ppo_kl are already floats from pg_metrics.get()
                    step_data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac,
                        'actor/ppo_kl': ppo_kl,
                    }
                    append_to_dict(metrics, step_data)

                # Optimizer step after accumulating gradients
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

        # Add selection stats
        if total_samples > 0:
            self.selection_stats['train/total_selected'] = total_selected
            self.selection_stats['train/total_samples'] = total_samples
            self.selection_stats['train/overall_selection_ratio'] = total_selected / total_samples

        for key, val in self.selection_stats.items():
            metrics[f'selection/{key}'] = val

        self.actor_optimizer.zero_grad()
        return metrics

    def _update_policy_greats(self, data: DataProto):
        """Update policy with GREATS selection (two-pass per micro-batch)."""
        from verl.trainer.ppo import core_algos
        from verl.utils.py_functional import append_to_dict
        import verl.utils.torch_functional as verl_F
        from verl.utils.torch_functional import masked_mean

        self.actor_module.train()

        # Use ppo_micro_batch_size_per_gpu (correct attribute name in verl config)
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        assert self.config.ppo_mini_batch_size % micro_batch_size == 0
        gradient_accumulation = self.config.ppo_mini_batch_size // micro_batch_size
        temperature = data.meta_info.get('temperature', 1.0)

        # Select keys
        select_keys = [
            'responses', 'input_ids', 'attention_mask', 'position_ids',
            'old_log_probs', 'advantages'
        ]
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')

        batch = data.select(batch_keys=select_keys).batch

        # Create dataloader
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        total_selected = 0
        total_samples = 0

        for _ in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                micro_batches = mini_batch.split(micro_batch_size)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.cuda()

                    # Extract data
                    input_ids = micro_batch['input_ids']
                    attention_mask = micro_batch['attention_mask']
                    position_ids = micro_batch['position_ids']
                    responses = micro_batch['responses']
                    old_log_probs = micro_batch['old_log_probs']
                    advantages = micro_batch['advantages']

                    response_length = responses.size(1)
                    response_mask = attention_mask[:, -response_length:]
                    micro_batch_size = responses.size(0)

                    # Pass 1: GREATS scoring to get selected indices
                    selected_indices = self._select_training_batch_greats(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        responses=responses,
                        old_log_probs=old_log_probs,
                        advantages=advantages,
                        temperature=temperature,
                    )

                    total_selected += len(selected_indices)
                    total_samples += micro_batch_size

                    if len(selected_indices) == 0:
                        logger.warning("GREATS: No samples selected, skipping micro-batch")
                        continue

                    # Filter to selected samples
                    sel_input_ids = input_ids[selected_indices]
                    sel_attention_mask = attention_mask[selected_indices]
                    sel_position_ids = position_ids[selected_indices]
                    sel_responses = responses[selected_indices]
                    sel_old_log_probs = old_log_probs[selected_indices]
                    sel_advantages = advantages[selected_indices]
                    sel_response_mask = response_mask[selected_indices]

                    # Pass 2: Actual training on selected samples
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        if self.use_remove_padding and unpad_input is not None:
                            # Pack sequences
                            sel_batch_size = sel_input_ids.size(0)
                            sel_seqlen = sel_input_ids.size(1)

                            input_ids_rmpad, indices, *_ = unpad_input(
                                sel_input_ids.unsqueeze(-1), sel_attention_mask
                            )
                            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                            position_ids_rmpad = index_first_axis(
                                rearrange(sel_position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                indices
                            ).transpose(0, 1)

                            output = self.actor_module(
                                input_ids=input_ids_rmpad,
                                attention_mask=None,
                                position_ids=position_ids_rmpad,
                                use_cache=False,
                            )
                            logits_rmpad = output.logits.squeeze(0)

                            if temperature != 1.0:
                                logits_rmpad = logits_rmpad / temperature

                            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1).squeeze(0)
                            log_probs_rmpad = logprobs_from_logits(logits_rmpad, input_ids_rmpad_rolled)

                            full_log_probs = pad_input(
                                hidden_states=log_probs_rmpad.unsqueeze(-1),
                                indices=indices,
                                batch=sel_batch_size,
                                seqlen=sel_seqlen
                            )
                            log_prob = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]

                            # Compute entropy from packed logits first, then pad
                            # (DON'T pad logits - they have vocab_size dimension which would be huge)
                            entropy_rmpad = verl_F.entropy_from_logits(logits_rmpad)  # [total_nnz]
                            full_entropy = pad_input(
                                hidden_states=entropy_rmpad.unsqueeze(-1),  # [total_nnz, 1]
                                indices=indices,
                                batch=sel_batch_size,
                                seqlen=sel_seqlen
                            )
                            entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # [batch, response_len]
                        else:
                            output = self.actor_module(
                                input_ids=sel_input_ids,
                                attention_mask=sel_attention_mask,
                                position_ids=sel_position_ids,
                                use_cache=False,
                            )
                            logits = output.logits

                            if temperature != 1.0:
                                logits = logits / temperature

                            logits = logits[:, -response_length - 1:-1, :]
                            log_prob = logprobs_from_logits(logits.float(), sel_responses)
                            entropy = verl_F.entropy_from_logits(logits)

                        # Compute PPO loss
                        pg_loss, pg_metrics = core_algos.compute_policy_loss_vanilla(
                            old_log_prob=sel_old_log_probs,
                            log_prob=log_prob,
                            advantages=sel_advantages,
                            response_mask=sel_response_mask,
                            loss_agg_mode=self.config.loss_agg_mode,
                            config=self.config,
                        )
                        pg_clipfrac = pg_metrics.get('pg_clipfrac', 0.0)
                        ppo_kl = pg_metrics.get('ppo_kl', 0.0)

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

                        loss = policy_loss / gradient_accumulation
                        loss.backward()

                    # Track metrics
                    # Note: pg_clipfrac and ppo_kl are already floats from pg_metrics.get()
                    step_data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac,
                        'actor/ppo_kl': ppo_kl,
                    }
                    append_to_dict(metrics, step_data)

                # Optimizer step
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})

        # Add selection stats
        if total_samples > 0:
            self.selection_stats['train/total_selected'] = total_selected
            self.selection_stats['train/total_samples'] = total_samples
            self.selection_stats['train/overall_selection_ratio'] = total_selected / total_samples

        for key, val in self.selection_stats.items():
            metrics[f'selection/{key}'] = val

        self.actor_optimizer.zero_grad()
        return metrics
