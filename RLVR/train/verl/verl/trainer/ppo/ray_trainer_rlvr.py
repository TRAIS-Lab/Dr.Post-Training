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
RLVR PPO Trainer with Ray-based single controller.

Extends the base RayPPOTrainer with RLVR activation-based selection:
- GREATS: Online per-batch selection (filter batch before loss)
- Streaming: Per-layer selection (filter gradients during backward)

Key differences from ray_trainer_teacher.py:
1. Uses activation similarity instead of teacher model for difficulty prediction
2. Selection happens per mini-batch (online) instead of per epoch
3. Reference is updated from rollouts each epoch
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, Optional, List, Any

import numpy as np
import torch
from tqdm import tqdm
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer_teacher import (
    RayPPOTrainer,
    Role,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    compute_step_metrics,
    compute_timing_metrics,
    compute_epoch_metrics,
    _timer,
)
from collections import defaultdict
import random


WorkerType = Type[Worker]


class RayRLVRTrainer(RayPPOTrainer):
    """
    PPO Trainer with RLVR activation-based data selection.

    Extends RayPPOTrainer to replace teacher-model selection with
    activation-based difficulty prediction and Goldilocks zone selection.

    Selection Methods:
    - GREATS: Online per-batch selection, filter before loss computation
    - Streaming: Per-layer selection during backward pass
    - Original: Per-epoch selection using teacher model (fallback)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RLVR configuration
        self.rlvr_enabled = getattr(self.config.data, 'rlvr_enabled', False)
        self.rlvr_selection_method = getattr(self.config.data, 'rlvr_selection_method', 'GREATS')

        # RLVR parameters
        self.rlvr_alpha = getattr(self.config.data, 'alpha', 0.5)
        self.rlvr_tau = getattr(self.config.data, 'tau', 0.1)
        self.rlvr_selection_ratio = getattr(self.config.data, 'rlvr_selection_ratio', 0.5)
        self.rlvr_selection_mode = getattr(self.config.data, 'rlvr_selection_mode', 'soft')
        self.rlvr_aggregation = getattr(self.config.data, 'rlvr_aggregation', 'mean')

        # RLVR hook (initialized lazily when model is available)
        self._rlvr_hook = None
        self._rlvr_hook_initialized = False

        # Reference data cache
        self._ref_input_ids: Optional[torch.Tensor] = None
        self._ref_attention_mask: Optional[torch.Tensor] = None
        self._ref_difficulties: Optional[torch.Tensor] = None

        if self.rlvr_enabled:
            print(f"RLVR enabled: method={self.rlvr_selection_method}, "
                  f"alpha={self.rlvr_alpha}, tau={self.rlvr_tau}, "
                  f"ratio={self.rlvr_selection_ratio}")

    def _init_rlvr_hook(self, model: torch.nn.Module) -> None:
        """Initialize RLVR hook on the model (lazy initialization)."""
        if self._rlvr_hook_initialized:
            return

        from RLVR import RLVRHook, get_trainable_layer_names

        # Get trainable layer names
        layer_names = get_trainable_layer_names(model)

        if len(layer_names) == 0:
            print("Warning: No trainable layers found for RLVR, disabling")
            self.rlvr_enabled = False
            return

        # Create hook
        self._rlvr_hook = RLVRHook(
            model=model,
            layer_names=layer_names,
            device='cuda',
            predictor_type='simple',  # Cosine similarity
        )

        # Patch layers for Streaming mode
        if self.rlvr_selection_method == 'Streaming':
            self._rlvr_hook.patch_linear_layers()

        self._rlvr_hook_initialized = True
        print(f"Initialized RLVRHook with {len(layer_names)} layers")

    def _update_rlvr_reference(
        self,
        ref_batches: DataProto,
        model: torch.nn.Module,
    ) -> Dict[str, float]:
        """
        Update RLVR reference from rollout data.

        Args:
            ref_batches: Concatenated reference batches with rollout data
            model: The actor model for activation capture

        Returns:
            Dict of reference statistics for logging
        """
        self._init_rlvr_hook(model)

        if self._rlvr_hook is None:
            return {}

        # Extract data from ref_batches
        # We need to aggregate rewards per unique question (uid)
        input_ids_list = []
        attention_mask_list = []
        rewards_list = []

        # Group by uid to get mean reward per question
        uid_to_data = defaultdict(lambda: {'input_ids': None, 'attention_mask': None, 'rewards': []})

        for batch in ref_batches:
            uid = batch.non_tensor_batch.get('uid', str(uuid.uuid4()))
            if uid_to_data[uid]['input_ids'] is None:
                uid_to_data[uid]['input_ids'] = batch.batch['input_ids']
                uid_to_data[uid]['attention_mask'] = batch.batch['attention_mask']
            # Accumulate rewards for this uid
            reward = batch.batch['token_level_scores'].sum(-1).item()
            uid_to_data[uid]['rewards'].append(reward)

        # Convert to tensors with mean rewards
        for uid, data in uid_to_data.items():
            input_ids_list.append(data['input_ids'])
            attention_mask_list.append(data['attention_mask'])
            mean_reward = np.mean(data['rewards'])
            rewards_list.append(mean_reward)

        if len(input_ids_list) == 0:
            return {}

        # Stack tensors
        ref_input_ids = torch.stack(input_ids_list, dim=0)
        ref_attention_mask = torch.stack(attention_mask_list, dim=0)
        ref_difficulties = torch.tensor(rewards_list, dtype=torch.float32)

        # Move to device
        device = next(model.parameters()).device
        ref_input_ids = ref_input_ids.to(device)
        ref_attention_mask = ref_attention_mask.to(device)
        ref_difficulties = ref_difficulties.to(device)

        # Capture reference activations
        self._rlvr_hook.capture_reference(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            difficulties=ref_difficulties,
        )

        # Cache for stats
        self._ref_difficulties = ref_difficulties

        # Return stats
        return {
            'rlvr/ref_size': len(ref_difficulties),
            'rlvr/ref_difficulty_mean': ref_difficulties.mean().item(),
            'rlvr/ref_difficulty_std': ref_difficulties.std().item() if len(ref_difficulties) > 1 else 0,
            'rlvr/ref_difficulty_min': ref_difficulties.min().item(),
            'rlvr/ref_difficulty_max': ref_difficulties.max().item(),
        }

    def _rlvr_select_batch(
        self,
        batch: DataProto,
        model: torch.nn.Module,
    ) -> tuple[DataProto, Dict[str, float]]:
        """
        Apply RLVR GREATS selection to a batch.

        Args:
            batch: Input batch (DataProto)
            model: The actor model

        Returns:
            Tuple of (filtered_batch, selection_metrics)
        """
        if self._rlvr_hook is None or not self._rlvr_hook.has_reference():
            return batch, {}

        input_ids = batch.batch['input_ids']
        attention_mask = batch.batch['attention_mask']
        batch_size = input_ids.shape[0]

        # Number of samples to select
        num_to_select = max(1, int(batch_size * self.rlvr_selection_ratio))

        # Setup selection
        self._rlvr_hook.setup_selection(
            selection_method='GREATS',
            train_batch_size=batch_size,
            alpha=self.rlvr_alpha,
            tau=self.rlvr_tau,
            selection_ratio=self.rlvr_selection_ratio,
            selection_mode=self.rlvr_selection_mode,
        )

        # Capture activations
        self._rlvr_hook.start_capture(attention_mask)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        self._rlvr_hook.end_capture()

        # Compute selections
        self._rlvr_hook.compute_selections(aggregation=self.rlvr_aggregation)

        # Get selected indices
        selected_indices = self._rlvr_hook.get_global_selection()
        selection_stats = self._rlvr_hook.get_selection_stats()

        # Cleanup
        self._rlvr_hook.clear_selection()
        self._rlvr_hook.clear_activations()

        if selected_indices is None or len(selected_indices) == 0:
            return batch, {}

        # Filter batch
        filtered_batch = self._filter_dataproto(batch, selected_indices)

        # Build metrics
        metrics = {
            'rlvr/batch_size': batch_size,
            'rlvr/num_selected': len(selected_indices),
            'rlvr/selection_ratio': len(selected_indices) / batch_size,
        }
        for key, value in selection_stats.items():
            metrics[f'rlvr/{key}'] = value

        return filtered_batch, metrics

    def _filter_dataproto(
        self,
        batch: DataProto,
        indices: torch.Tensor,
    ) -> DataProto:
        """
        Filter a DataProto by selected indices.

        Args:
            batch: Input DataProto
            indices: Selected indices tensor

        Returns:
            Filtered DataProto
        """
        indices_list = indices.cpu().tolist()

        # Filter tensor batch
        filtered_tensors = {}
        for key, value in batch.batch.items():
            if isinstance(value, torch.Tensor):
                filtered_tensors[key] = value[indices]
            else:
                filtered_tensors[key] = value

        # Filter non-tensor batch
        filtered_non_tensors = {}
        for key, value in batch.non_tensor_batch.items():
            if isinstance(value, (list, tuple)):
                filtered_non_tensors[key] = [value[i] for i in indices_list]
            elif isinstance(value, np.ndarray):
                filtered_non_tensors[key] = value[indices_list]
            else:
                filtered_non_tensors[key] = value

        return DataProto.from_dict(
            tensors=filtered_tensors,
            non_tensors=filtered_non_tensors,
            meta_info=batch.meta_info,
        )

    def fit(self):
        """
        The training loop of PPO with RLVR selection.

        Extends the base fit() method to add RLVR activation-based selection.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        import traceback
        import sys
        import time

        print("Start to train with RLVR!")

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        self._load_checkpoint()

        # Perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        effective_total_steps = self.total_training_steps - self.global_steps
        trained_epochs = self.global_steps // self.config.data.mu

        # We start from step 1
        self.global_steps += 1

        # Maximum number of retry attempts
        max_retries = self.config.trainer.get('max_retries', 3)
        retry_count = 0
        retry_delay = self.config.trainer.get('retry_delay', 60)

        # Initialize epoch_raw outside the loop to handle edge cases
        # (e.g., when resuming past the training loop)
        epoch_raw = {}
        epoch_metrics = {}

        while retry_count <= max_retries:
            try:
                with tqdm(total=effective_total_steps, desc="Training Progress") as pbar:

                    for epoch in range(trained_epochs, self.config.trainer.total_epochs):
                        epoch_raw = {}
                        epoch_metrics = {}

                        with _timer('epoch', epoch_raw):
                            with _timer('data_selection', epoch_raw):
                                from torch.utils.data import DataLoader
                                from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

                                self.train_dataset = RLHFDataset(
                                    parquet_files=self.config.data.train_files,
                                    tokenizer=self.tokenizer,
                                    prompt_key=self.config.data.prompt_key,
                                    max_prompt_length=self.config.data.max_prompt_length,
                                    filter_prompts=True,
                                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                                    truncation='error',
                                    format_reward=self.config.data.get('format_reward', False)
                                )

                                # ================================================
                                # RLVR: Collect reference rollouts
                                # ================================================
                                if self.rlvr_enabled and self.rlvr_selection_method in ['GREATS', 'Streaming']:
                                    print(f"[RLVR] Collecting reference rollouts for epoch {epoch}")

                                    ref_indices = random.sample(
                                        range(len(self.train_dataset)),
                                        min(self.config.data.ref_size, len(self.train_dataset))
                                    )
                                    ref_dataset = torch.utils.data.Subset(self.train_dataset, indices=ref_indices)

                                    ref_dataloader = DataLoader(
                                        dataset=ref_dataset,
                                        batch_size=min(self.config.data.train_batch_size, len(ref_dataset)),
                                        shuffle=False,
                                        drop_last=False,
                                        collate_fn=collate_fn
                                    )

                                    # Collect reference rollouts
                                    ref_batches = []
                                    ref_solve_none = 0
                                    ref_solve_all = 0

                                    for _, batch_dict in enumerate(tqdm(ref_dataloader, desc="Reference Rollout")):
                                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                                        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                                        batch.non_tensor_batch['uid'] = np.array(
                                            [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                            dtype=object
                                        )
                                        batch = batch.repeat(
                                            repeat_times=self.config.actor_rollout_ref.rollout.n,
                                            interleave=True
                                        )
                                        batch = batch.union(gen_batch_output)

                                        if self.use_critic:
                                            values = self.critic_wg.compute_values(batch)
                                            batch = batch.union(values)

                                        if self.use_rm:
                                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                                            batch = batch.union(reward_tensor)

                                        reward_tensor = self.reward_fn(batch)
                                        batch.batch['token_level_scores'] = reward_tensor

                                        # Track solve statistics
                                        uids = batch.non_tensor_batch['uid']
                                        unique_uids = np.unique(uids)
                                        for uid in unique_uids:
                                            uid_mask = uids == uid
                                            uid_rewards = reward_tensor[uid_mask].sum(-1)
                                            if (uid_rewards == 0).all():
                                                ref_solve_none += 1
                                            elif (uid_rewards == 1).all():
                                                ref_solve_all += 1

                                        ref_batches.append(batch)

                                    epoch_metrics['epoch/ref_solve_none'] = ref_solve_none
                                    epoch_metrics['epoch/ref_solve_all'] = ref_solve_all

                                    # Concatenate reference batches
                                    ref_batches_concat = DataProto.concat(ref_batches)

                                    # Update RLVR reference with actor model
                                    # Note: We need to get the actual model from the worker group
                                    # This may require implementing a method to access the model
                                    actor_model = self._get_actor_model()
                                    if actor_model is not None:
                                        ref_metrics = self._update_rlvr_reference(ref_batches_concat, actor_model)
                                        epoch_metrics.update(ref_metrics)
                                        print(f"[RLVR] Updated reference: {ref_metrics}")

                                    # For RLVR, we use all data (selection happens per-batch)
                                    use_train_dataloader = DataLoader(
                                        dataset=self.train_dataset,
                                        batch_size=self.config.data.train_batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        collate_fn=collate_fn
                                    )

                                    # Limit to mu steps per epoch
                                    total_batches = min(len(use_train_dataloader), self.config.data.mu)

                                else:
                                    # Fallback: Use original teacher-based selection
                                    # (Copy the original selection logic from ray_trainer_teacher.py)
                                    print("[RLVR] Using original teacher-based selection")

                                    if not self.config.data.random_selection:
                                        # Original teacher selection logic...
                                        ref_indices = random.sample(range(len(self.train_dataset)), self.config.data.ref_size)
                                        ref_dataset = torch.utils.data.Subset(self.train_dataset, indices=ref_indices)

                                        ref_dataloader = DataLoader(
                                            dataset=ref_dataset,
                                            batch_size=min(self.config.data.train_batch_size, self.config.data.ref_size),
                                            shuffle=False,
                                            drop_last=False,
                                            collate_fn=collate_fn
                                        )

                                        ref_batches = []
                                        for _, batch_dict in enumerate(tqdm(ref_dataloader, desc="Reference Rollout")):
                                            batch: DataProto = DataProto.from_single_dict(batch_dict)
                                            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                            batch.non_tensor_batch['uid'] = np.array(
                                                [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                dtype=object
                                            )
                                            batch = batch.repeat(
                                                repeat_times=self.config.actor_rollout_ref.rollout.n,
                                                interleave=True
                                            )
                                            batch = batch.union(gen_batch_output)
                                            if self.use_critic:
                                                values = self.critic_wg.compute_values(batch)
                                                batch = batch.union(values)
                                            if self.use_rm:
                                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                                batch = batch.union(reward_tensor)
                                            reward_tensor = self.reward_fn(batch)
                                            batch.batch['token_level_scores'] = reward_tensor
                                            ref_batches.append(batch)

                                        ref_batches = DataProto.concat(ref_batches)
                                        ref_data = defaultdict(dict)
                                        for _, batch in enumerate(ref_batches):
                                            key = batch.non_tensor_batch['index']
                                            ref_data[key]['question'] = batch.non_tensor_batch['extra_info']['question']
                                            ref_data[key]['rewards'] = ref_data[key].get('rewards', []) + [
                                                batch.batch['token_level_scores'].sum(-1).item()
                                            ]

                                        ref_questions = []
                                        ref_labels = []
                                        ref_indices_keys = []
                                        for key in ref_data:
                                            ref_questions.append(ref_data[key]['question'])
                                            ref_labels.append(np.mean(ref_data[key]['rewards']))
                                            ref_indices_keys.append(key)

                                        all_questions = [item['extra_info']['question'] for item in self.train_dataset]
                                        predicted_labels = self.teacher_wg.predict(
                                            all_questions, ref_questions, ref_labels, ref_indices_keys,
                                            batch_size=self.config.teacher_model.batch_size
                                        )
                                    else:
                                        predicted_labels = torch.rand(len(self.train_dataset))

                                    # Original selection logic
                                    selection_budget = int(self.config.data.mu * self.config.data.train_batch_size)
                                    dataset_sampling_scores = -torch.abs(predicted_labels - self.config.data.alpha)
                                    dataset_sampling_logits = dataset_sampling_scores / self.config.data.tau
                                    dataset_sampling_logits -= dataset_sampling_logits.max()
                                    dataset_sampling_probabilities = torch.softmax(dataset_sampling_logits, dim=0)
                                    selected_indices = torch.multinomial(
                                        dataset_sampling_probabilities, selection_budget, replacement=False
                                    )
                                    selected_indices = selected_indices[torch.randperm(len(selected_indices))]

                                    use_dataset = torch.utils.data.Subset(
                                        self.train_dataset, indices=selected_indices.tolist()
                                    )
                                    use_train_dataloader = DataLoader(
                                        dataset=use_dataset,
                                        batch_size=self.config.data.train_batch_size,
                                        shuffle=False,
                                        drop_last=False,
                                        collate_fn=collate_fn
                                    )
                                    total_batches = len(use_train_dataloader)

                            # ================================================
                            # Training Loop
                            # ================================================
                            with _timer('Rollout_update', epoch_raw):
                                batch_count = 0
                                for _, batch_dict in enumerate(tqdm(use_train_dataloader, desc="Rollout_update")):
                                    if batch_count >= total_batches:
                                        break
                                    batch_count += 1

                                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                                    metrics = {}

                                    # ================================================
                                    # RLVR GREATS: Per-batch selection
                                    # ================================================
                                    if self.rlvr_enabled and self.rlvr_selection_method == 'GREATS':
                                        actor_model = self._get_actor_model()
                                        if actor_model is not None and self._rlvr_hook is not None:
                                            batch, rlvr_metrics = self._rlvr_select_batch(batch, actor_model)
                                            metrics.update(rlvr_metrics)

                                    # Pop keys for generation
                                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                                    # Generate sequences
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                                    # Match prompt ID with responses
                                    batch.non_tensor_batch['uid'] = np.array(
                                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                        dtype=object
                                    )
                                    batch = batch.repeat(
                                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                                        interleave=True
                                    )
                                    batch = batch.union(gen_batch_output)

                                    # Compute values
                                    if self.use_critic:
                                        values = self.critic_wg.compute_values(batch)
                                        batch = batch.union(values)

                                    # Compute scores
                                    if self.use_rm:
                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                        batch = batch.union(reward_tensor)

                                    reward_tensor = self.reward_fn(batch)
                                    batch.batch['token_level_scores'] = reward_tensor

                                    # Track solve statistics
                                    uids = batch.non_tensor_batch['uid']
                                    unique_uids = np.unique(uids)
                                    solve_none = 0
                                    solve_all = 0
                                    for uid in unique_uids:
                                        uid_mask = uids == uid
                                        uid_rewards = reward_tensor[uid_mask].sum(-1)
                                        if (uid_rewards == 0).all():
                                            solve_none += 1
                                        elif (uid_rewards == 1).all():
                                            solve_all += 1

                                    metrics['batch/solve_none'] = solve_none
                                    metrics['batch/solve_all'] = solve_all

                                    # Compute log probs
                                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                    batch = batch.union(old_log_prob)

                                    if self.use_reference_policy:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                        batch = batch.union(ref_log_prob)

                                    # Set token level rewards
                                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                                    # Compute advantages
                                    batch = compute_advantage(
                                        batch,
                                        adv_estimator=self.config.algorithm.adv_estimator,
                                        gamma=self.config.algorithm.gamma,
                                        lam=self.config.algorithm.lam,
                                        num_repeat=self.config.actor_rollout_ref.rollout.n
                                    )

                                    # Balance batch
                                    self._balance_batch(batch, metrics=metrics)

                                    # Global token num
                                    batch.meta_info['global_token_num'] = torch.sum(
                                        batch.batch['attention_mask'], dim=-1
                                    ).tolist()

                                    # Update critic
                                    if self.use_critic:
                                        critic_output = self.critic_wg.update_critic(batch)
                                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                                        metrics.update(critic_output_metrics)

                                    # Compute temp_log_prob if enabled
                                    if self.config.actor_rollout_ref.actor.get('use_temp_log_prob', False):
                                        temp_log_prob = self.actor_rollout_wg.compute_temp_log_prob(batch)
                                        batch = batch.union(temp_log_prob)

                                    # Update actor (after warmup)
                                    if self.config.trainer.critic_warmup <= self.global_steps:
                                        actor_output = self.actor_rollout_wg.update_actor(batch)
                                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                                        metrics.update(actor_output_metrics)

                                    # Validation
                                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                                            self.global_steps % self.config.trainer.test_freq == 0:
                                        val_metrics: dict = self._validate()
                                        metrics.update(val_metrics)

                                    # Save checkpoint
                                    if self.config.trainer.save_freq > 0 and \
                                            self.global_steps % self.config.trainer.save_freq == 0:
                                        print("Saving model...")
                                        with _timer('save_checkpoint', epoch_raw):
                                            self._save_checkpoint(save_checkpoints=False)

                                    if self.config.trainer.save_freq > 0 and \
                                            self.global_steps % (self.config.trainer.save_freq * 3) == 0:
                                        print("Saving model...")
                                        with _timer('save_checkpoint', epoch_raw):
                                            self._save_checkpoint(save_checkpoints=True)

                                    # Collect metrics
                                    metrics.update(compute_step_metrics(batch=batch, use_critic=self.use_critic))

                                    # Log
                                    logger.log(data=metrics, step=self.global_steps)

                                    self.global_steps += 1
                                    pbar.update(1)

                            epoch_metrics.update(compute_epoch_metrics(epoch_raw=epoch_raw))
                            logger.log(data=epoch_metrics, step=self.global_steps - 1)

                            if self.global_steps > self.total_training_steps:
                                self.global_steps -= 1
                                with _timer('save_final_checkpoint', epoch_raw):
                                    self._save_checkpoint(save_checkpoints=True)
                                print("Training completed successfully!")
                                return

                # Training completed
                self.global_steps -= 1
                with _timer('save_final_checkpoint', epoch_raw):
                    self._save_checkpoint(save_checkpoints=True)
                print("Training completed successfully!")

                epoch_metrics.update(compute_epoch_metrics(epoch_raw=epoch_raw))
                logger.log(data=epoch_metrics, step=self.global_steps)
                return

            except Exception as e:
                retry_count += 1
                error_msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                logger.log(data={"error": error_msg}, step=self.global_steps)

                print(f"Saving emergency checkpoint at step {self.global_steps}")
                self._save_checkpoint(save_checkpoints=True)

                if retry_count <= max_retries:
                    print(f"Attempting to resume training in {retry_delay} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    self._load_checkpoint()
                    print(f"Resumed training from step {self.global_steps}")
                else:
                    print(f"Maximum retry attempts ({max_retries}) reached. Exiting.")
                    raise

    def _get_actor_model(self) -> Optional[torch.nn.Module]:
        """
        Get the actor model from the worker group.

        This is a placeholder - the actual implementation depends on
        how the VERL worker group exposes the model.
        """
        # Try to get model from worker group
        # This may need to be implemented based on VERL's API
        try:
            # Option 1: Direct attribute access
            if hasattr(self.actor_rollout_wg, 'model'):
                return self.actor_rollout_wg.model

            # Option 2: Method call
            if hasattr(self.actor_rollout_wg, 'get_model'):
                return self.actor_rollout_wg.get_model()

            # Option 3: Access through workers
            if hasattr(self.actor_rollout_wg, '_workers') and len(self.actor_rollout_wg._workers) > 0:
                worker = self.actor_rollout_wg._workers[0]
                if hasattr(worker, 'model'):
                    return worker.model

            print("Warning: Could not access actor model for RLVR selection")
            return None

        except Exception as e:
            print(f"Warning: Error accessing actor model: {e}")
            return None

    def cleanup(self):
        """Cleanup RLVR resources."""
        if self._rlvr_hook is not None:
            self._rlvr_hook.remove_hooks()
            if self.rlvr_selection_method == 'Streaming':
                self._rlvr_hook.unpatch_linear_layers()
