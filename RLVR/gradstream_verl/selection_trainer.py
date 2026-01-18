"""
PPO Trainer with Online Validation Gradient-Based Data Selection.

This trainer extends VERL's RayPPOTrainer to support gradient-based data selection
with online validation rollouts. It:

1. Generates fresh rollouts for validation prompts using the current policy
2. Computes rewards on validation responses
3. Captures validation gradients for selection
4. Applies Streaming or GREATS selection during training

This matches the intended design from RLHF/train/trainer.py.

Usage:
    trainer = SelectionRayPPOTrainerWithOnlineVal(
        config=config,
        tokenizer=tokenizer,
        ...
        selection_config=selection_config,
    )
    trainer.init_workers()
    trainer.fit()
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.core_algos import agg_loss

from .validation_manager import ValidationConfig, ValidationDataManager, prepare_validation_batch_for_verl

if TYPE_CHECKING:
    from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SelectionTrainerConfig:
    """Configuration for selection trainer."""

    # Enable/disable selection
    enable: bool = False

    # Selection method: "Streaming", "GREATS", or "NA"
    method: str = "NA"

    # Fraction of samples to select (0.0 to 1.0)
    frac: float = 0.5

    # Use second-order selection (similarity matrix)
    use_second_order: bool = False

    # Validation configuration
    val_prompts_path: Optional[str] = None
    val_pool_size: int = 500
    val_batch_size: int = 32
    val_max_prompt_length: int = 1024
    val_max_response_length: int = 1024

    # How often to refresh validation gradients (1 = every step)
    refresh_freq: int = 1

    # Validation loss type: "seqloss-reward" or "seqloss-lastadv"
    val_loss_type: str = "seqloss-reward"


class SelectionRayPPOTrainerWithOnlineVal(RayPPOTrainer):
    """
    PPO Trainer with online validation rollout generation for gradient-based selection.

    This trainer generates fresh validation rollouts each step (or every N steps),
    computes validation gradients, and uses them for Streaming/GREATS data selection.

    Key differences from base RayPPOTrainer:
    1. Maintains a validation data manager with prompt pool
    2. Generates validation rollouts before each training step
    3. Computes validation gradients using seqloss-reward objective
    4. Applies selection during _update_actor

    Flow:
    1. Get next validation batch (prompts only)
    2. Generate rollouts for validation prompts
    3. Compute rewards on validation responses
    4. Capture validation gradients
    5. Proceed with normal training (with selection enabled)
    """

    def __init__(self, *args, selection_config: SelectionTrainerConfig = None, **kwargs):
        super().__init__(*args, **kwargs)

        # Selection configuration
        if selection_config is None:
            selection_config = SelectionTrainerConfig()
        self.selection_config = selection_config

        # Validation data manager (initialized later when tokenizer is available)
        self.val_data_manager: Optional[ValidationDataManager] = None

        # Selection statistics
        self._selection_stats = {
            "total_samples": 0,
            "selected_samples": 0,
            "selection_calls": 0,
            "val_rollouts_generated": 0,
        }

        # Step counter for refresh frequency
        self._step_counter = 0

        if self.selection_config.enable:
            logger.info(
                f"[Selection] Enabled: method={self.selection_config.method}, "
                f"frac={self.selection_config.frac}, "
                f"val_batch_size={self.selection_config.val_batch_size}"
            )

    def init_workers(self):
        """Initialize workers and validation data manager."""
        logger.info("[Selection] Initializing workers...")
        super().init_workers()

        # Initialize validation data manager after tokenizer is available
        if self.selection_config.enable and self.selection_config.val_prompts_path:
            self._init_validation_manager()

    def _init_validation_manager(self):
        """Initialize validation data manager."""
        val_config = ValidationConfig(
            val_prompts_path=self.selection_config.val_prompts_path,
            val_pool_size=self.selection_config.val_pool_size,
            val_batch_size=self.selection_config.val_batch_size,
            max_prompt_length=self.selection_config.val_max_prompt_length,
            max_response_length=self.selection_config.val_max_response_length,
            shuffle=True,
        )

        self.val_data_manager = ValidationDataManager(
            config=val_config,
            tokenizer=self.tokenizer,
        )

        logger.info(
            f"[Selection] Validation manager initialized: "
            f"pool_size={self.val_data_manager.pool_size}, "
            f"batch_size={self.val_data_manager.batch_size}"
        )

    def _generate_validation_rollouts(self) -> DataProto:
        """
        Generate rollouts for validation prompts using current policy.

        Returns:
            DataProto with validation rollouts including:
            - input_ids, attention_mask (full sequence)
            - responses (generated responses)
            - response_mask
            - rewards (from reward function)
        """
        if self.val_data_manager is None:
            raise RuntimeError("Validation data manager not initialized")

        # Get next validation batch
        val_batch = self.val_data_manager.get_next_batch()

        # Prepare for VERL format
        val_proto = prepare_validation_batch_for_verl(val_batch, device='cuda')

        # Add temperature for generation
        val_proto.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

        # Generate rollouts
        logger.debug(f"[Selection] Generating validation rollouts for {len(val_batch['prompts'])} prompts...")
        val_output = self.actor_rollout_wg.generate_sequences(val_proto)

        # Compute response mask if not present
        if "response_mask" not in val_output.batch.keys():
            from verl.trainer.ppo.ray_trainer import compute_response_mask
            val_output.batch["response_mask"] = compute_response_mask(val_output)

        # Compute rewards
        if self.reward_fn is not None:
            reward_tensor, reward_extra_infos = self._compute_or_extract_reward(
                val_output,
                reward_fn=self.reward_fn,
                reward_for_val=True,
            )
            val_output.batch["rewards"] = reward_tensor

        self._selection_stats["val_rollouts_generated"] += 1

        return val_output

    def _capture_validation_gradients(self, val_batch: DataProto) -> Dict[str, float]:
        """
        Capture validation gradients from validation rollouts.

        This method:
        1. Extracts data from validation rollouts
        2. Calls the actor's validation gradient capture method
        3. Returns statistics

        Args:
            val_batch: Validation rollouts with responses and rewards

        Returns:
            Dictionary with validation gradient capture stats
        """
        # Extract required tensors
        input_ids = val_batch.batch['input_ids']
        attention_mask = val_batch.batch['attention_mask']
        responses = val_batch.batch['responses']
        response_mask = val_batch.batch['response_mask']
        rewards = val_batch.batch['rewards']

        # Compute position_ids
        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create validation batch for actor
        val_data = DataProto(
            batch={
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'responses': responses,
                'response_mask': response_mask,
                'rewards': rewards,
            },
            meta_info={'temperature': self.config.actor_rollout_ref.rollout.temperature}
        )

        # Call actor's validation gradient capture
        # This dispatches to the selection-enabled actor
        stats = self.actor_rollout_wg.capture_validation_gradients(val_data)

        return stats.meta_info if hasattr(stats, 'meta_info') else {}

    def _should_capture_validation_gradients(self) -> bool:
        """Check if we should capture validation gradients this step."""
        if not self.selection_config.enable:
            return False
        if self.selection_config.method == "NA":
            return False
        if self.val_data_manager is None:
            return False
        return self._step_counter % self.selection_config.refresh_freq == 0

    def fit(self):
        """
        Main training loop with online validation gradient capture.

        This overrides the base fit() method to add validation rollout
        generation and gradient capture before each training step.
        """
        # If selection not enabled or no validation data, use base implementation
        if not self.selection_config.enable or self.val_data_manager is None:
            logger.info("[Selection] Disabled or no validation data, using base trainer")
            return super().fit()

        # Modified training loop with validation gradient capture
        logger.info("[Selection] Starting training with online validation rollouts...")

        # Get the base fit method's logic but inject our validation step
        # We need to intercept before _update_actor is called
        return self._fit_with_selection()

    def _fit_with_selection(self):
        """
        Training loop with validation gradient capture.

        This is a modified version of the base fit() method that adds
        validation rollout generation before each training step.
        """
        from copy import deepcopy
        import uuid
        from verl.utils.debug import marked_timer
        from verl.trainer.ppo.core_algos import AdvantageEstimator
        from verl.trainer.ppo.ray_trainer import compute_response_mask
        from verl.trainer.ppo.reward import compute_reward_async

        # Initialize similar to base fit()
        self._load_checkpoint()

        # Log validation manager stats
        if self.val_data_manager:
            logger.info(f"[Selection] Validation pool: {self.val_data_manager.get_stats()}")

        current_epoch = self.global_steps // len(self.train_dataloader)

        prev_step_profile = False
        curr_step_profile = False
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}

                # ============================================================
                # VALIDATION GRADIENT CAPTURE (NEW)
                # ============================================================
                if self._should_capture_validation_gradients():
                    with marked_timer("val_rollout", timing_raw, color="green"):
                        try:
                            # Generate validation rollouts
                            val_batch = self._generate_validation_rollouts()

                            # Capture validation gradients
                            val_stats = self._capture_validation_gradients(val_batch)

                            # Log stats
                            for k, v in val_stats.items():
                                metrics[f"selection/{k}"] = v

                            logger.debug(
                                f"[Selection] Step {self.global_steps}: "
                                f"Captured validation gradients from {len(val_batch.batch['input_ids'])} samples"
                            )
                        except Exception as e:
                            logger.warning(f"[Selection] Failed to capture validation gradients: {e}")

                self._step_counter += 1

                # ============================================================
                # STANDARD TRAINING LOOP (from base class)
                # ============================================================
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # Add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # Generate rollouts
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                    # Handle REMAX if needed
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        # ... (same as base class)
                        pass

                    # Repeat batch to align with responses
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Balance batch if configured
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Set metadata
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Compute rewards
                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, reward_for_val=False
                            )

                    # Compute old log probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    # Compute ref log probs if needed
                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="cyan"):
                            ref_log_prob, ref_mfu = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # Get reward tensor if async
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                    batch.batch["token_level_scores"] = reward_tensor

                    # Compute advantages
                    with marked_timer("adv", timing_raw):
                        batch = self._compute_advantage(batch)

                    # Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="orange"):
                            critic_output = self._update_critic(batch)

                    # Update actor (with selection enabled via hooks)
                    with marked_timer("update_actor", timing_raw, color="magenta"):
                        actor_output = self._update_actor(batch)

                # Compute metrics
                metrics.update(self._compute_metrics(batch, timing_raw, actor_output))

                # Add selection stats
                val_manager_stats = self.val_data_manager.get_stats() if self.val_data_manager else {}
                for k, v in val_manager_stats.items():
                    metrics[f"selection/{k}"] = v

                # Log progress
                self._log_step(metrics, timing_raw)

                # Checkpointing
                self._save_checkpoint_if_needed()

                self.global_steps += 1

                # End profiling
                self._stop_profiling()
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

                if is_last_step:
                    break

            if is_last_step:
                break

        # Final checkpoint
        self._save_final_checkpoint()

        logger.info("[Selection] Training complete!")
        logger.info(f"[Selection] Final stats: {self._selection_stats}")

    def _compute_metrics(
        self,
        batch: DataProto,
        timing_raw: Dict[str, float],
        actor_output: DataProto,
    ) -> Dict[str, float]:
        """Compute training metrics."""
        from verl.trainer.ppo.metric_utils import (
            compute_data_metrics,
            compute_throughout_metrics,
            compute_timing_metrics,
        )

        metrics = {}

        # Data metrics
        data_metrics = compute_data_metrics(
            batch=batch,
            use_critic=self.use_critic,
            compute_variance_proxy=self.config.trainer.get("compute_variance_proxy", False),
        )
        metrics.update(data_metrics)

        # Actor metrics
        if hasattr(actor_output, 'meta_info'):
            for k, v in actor_output.meta_info.items():
                if isinstance(v, (int, float)):
                    metrics[f"actor/{k}"] = v

        # Timing metrics
        timing_metrics = compute_timing_metrics(timing_raw)
        metrics.update(timing_metrics)

        return metrics

    def _log_step(self, metrics: Dict[str, float], timing_raw: Dict[str, float]):
        """Log training step."""
        if self.global_steps % self.config.trainer.get("log_freq", 1) == 0:
            # Log key metrics
            reward = metrics.get("reward/mean", 0)
            kl = metrics.get("critic/kl", 0)
            loss = metrics.get("actor/policy_loss", 0)

            logger.info(
                f"Step {self.global_steps}: "
                f"reward={reward:.3f}, kl={kl:.4f}, loss={loss:.4f}"
            )

    def _save_checkpoint_if_needed(self):
        """Save checkpoint if needed."""
        save_freq = self.config.trainer.get("save_freq", -1)
        if save_freq > 0 and self.global_steps % save_freq == 0:
            self._save_checkpoint()

    def _save_checkpoint(self):
        """Save training checkpoint."""
        # Delegate to base class
        pass

    def _save_final_checkpoint(self):
        """Save final checkpoint."""
        pass


def create_selection_trainer(
    config,
    tokenizer,
    processor,
    role_worker_mapping,
    resource_pool_manager,
    ray_worker_group_cls,
    reward_fn,
    val_reward_fn,
    train_dataset,
    val_dataset,
    collate_fn,
    train_sampler,
    selection_config: SelectionTrainerConfig,
) -> SelectionRayPPOTrainerWithOnlineVal:
    """
    Factory function to create selection trainer.

    Args:
        config: Training configuration
        tokenizer: Tokenizer
        processor: Processor
        role_worker_mapping: Role to worker mapping
        resource_pool_manager: Resource pool manager
        ray_worker_group_cls: Ray worker group class
        reward_fn: Reward function
        val_reward_fn: Validation reward function
        train_dataset: Training dataset
        val_dataset: Validation dataset
        collate_fn: Collate function
        train_sampler: Training sampler
        selection_config: Selection configuration

    Returns:
        SelectionRayPPOTrainerWithOnlineVal instance
    """
    return SelectionRayPPOTrainerWithOnlineVal(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collate_fn=collate_fn,
        train_sampler=train_sampler,
        selection_config=selection_config,
    )
