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

import os
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.reward import compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.experimental.dataset.sampler import AbstractCurriculumSampler

from .validation_manager import ValidationConfig, ValidationDataManager, prepare_validation_batch_for_verl

if TYPE_CHECKING:
    from torch import Tensor


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
    val_seed: Optional[int] = None

    # How often to refresh validation gradients (1 = every step)
    refresh_freq: int = 1

    # Validation loss type: "seqloss-reward" or "seqloss-lastadv"
    val_loss_type: str = "seqloss-reward"

    # Reward normalization for validation gradients: "batch" or "grpo"
    # - "batch": batch-level normalization (rewards - mean) / std
    # - "grpo": per-prompt-group GRPO normalization (matches training)
    val_norm_type: str = "batch"


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

    def init_workers(self):
        """Initialize workers and validation data manager."""
        super().init_workers()

        # Setup selection on the actor worker group
        if self.selection_config.enable and self.selection_config.method != "NA":
            self._setup_worker_selection()

        # Initialize validation data manager after tokenizer is available
        if self.selection_config.enable and self.selection_config.val_prompts_path:
            self._init_validation_manager()

    def _setup_worker_selection(self):
        """Setup gradient-based selection on actor workers."""
        selection_config_dict = {
            "method": self.selection_config.method,
            "frac": self.selection_config.frac,
            "use_second_order": self.selection_config.use_second_order,
        }

        # Check if the worker has setup_selection method
        if hasattr(self.actor_rollout_wg, 'setup_selection'):
            self.actor_rollout_wg.setup_selection(selection_config_dict)
        else:
            print(
                "Warning: Worker group doesn't have setup_selection method. "
                "Make sure you're using SelectionActorRolloutRefWorker."
            )

    def _init_validation_manager(self):
        """Initialize validation data manager."""
        val_config = ValidationConfig(
            val_prompts_path=self.selection_config.val_prompts_path,
            val_pool_size=self.selection_config.val_pool_size,
            val_batch_size=self.selection_config.val_batch_size,
            max_prompt_length=self.selection_config.val_max_prompt_length,
            max_response_length=self.selection_config.val_max_response_length,
            shuffle=True,
            seed=self.selection_config.val_seed if self.selection_config.val_seed is not None else 42,
        )

        self.val_data_manager = ValidationDataManager(
            config=val_config,
            tokenizer=self.tokenizer,
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

        # Prepare for VERL format (keep on CPU - workers will move to their GPUs)
        val_proto = prepare_validation_batch_for_verl(val_batch, device='cpu')

        # Add temperature for generation
        val_proto.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

        # Add required metadata for generation
        val_proto.meta_info["global_steps"] = self.global_steps

        # Repeat for n responses per prompt (same as training)
        n_responses = self.config.actor_rollout_ref.rollout.n
        val_proto_repeated = val_proto.repeat(repeat_times=n_responses, interleave=True)

        # Always use async_rollout_manager - same as training rollouts
        val_output = self.async_rollout_manager.generate_sequences(val_proto_repeated)

        # Union with original batch to preserve metadata (reward_model, data_source, extra_info).
        # This follows the same pattern as the training flow: batch = batch.union(gen_batch_output)
        # The validation batch now uses dummy_tensor instead of input_ids/attention_mask,
        # so there's no tensor conflict during union.
        val_output = val_proto_repeated.union(val_output)

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

        # Ensure rewards are sequence-level [batch] not token-level [batch, seq_len]
        # Token-level rewards need to be summed to get sequence-level rewards
        if rewards.dim() > 1:
            # Sum token-level rewards to get sequence-level rewards
            rewards = rewards.sum(dim=-1)

        # Compute position_ids
        from verl.utils.model import compute_position_id_with_mask
        position_ids = compute_position_id_with_mask(attention_mask)

        # Create validation batch for actor
        val_data = DataProto.from_single_dict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'responses': responses,
            'response_mask': response_mask,
            'rewards': rewards,
        })
        val_data.meta_info['temperature'] = self.config.actor_rollout_ref.rollout.temperature
        # Pass GRPO parameters for per-prompt-group normalization (matching training)
        val_data.meta_info['n_responses'] = self.config.actor_rollout_ref.rollout.n
        val_data.meta_info['norm_adv_by_std'] = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        val_data.meta_info['norm_type'] = self.selection_config.val_norm_type

        # Dispatch to workers for gradient capture
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
        # Capture validation gradients every refresh_freq steps
        return self._step_counter % self.selection_config.refresh_freq == 0

    def fit(self):
        """
        Main training loop with online validation gradient capture.

        This overrides the base fit() method to add validation rollout
        generation and gradient capture before each training step.
        """
        # If selection not enabled or no validation data, use base implementation
        if not self.selection_config.enable or self.val_data_manager is None:
            return super().fit()

        # Get the base fit method's logic but inject our validation step
        # We need to intercept before _update_actor is called
        return self._fit_with_selection()

    def _fit_with_selection(self):
        """
        Training loop with validation gradient capture.

        This is a modified version of the base fit() method that adds
        validation rollout generation before each training step.

        Based on verl's RayPPOTrainer.fit() with selection-specific additions.
        """
        from verl.utils.tracking import Tracking
        from verl.trainer.ppo.utils import Role

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # Log validation manager stats (selection-specific)
        if self.val_data_manager:
            print(f"Validation pool: {self.val_data_manager.get_stats()}")

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                # Selection-specific: increment step counter for validation refresh
                self._step_counter += 1

                # ============================================================
                # SELECTION-SPECIFIC: VALIDATION GRADIENT CAPTURE
                # Generate validation rollouts and capture gradients for selection.
                # This happens BEFORE the standard training loop.
                # ============================================================
                if self._should_capture_validation_gradients():
                    with marked_timer("val_grad", timing_raw, color="green"):
                        val_batch = self._generate_validation_rollouts()
                        val_stats = self._capture_validation_gradients(val_batch)

                        # Log stats (now synchronized across all ranks)
                        for k, v in val_stats.items():
                            metrics[f"selection/{k}"] = v

                # ============================================================
                # STANDARD TRAINING LOOP (from verl's RayPPOTrainer.fit())
                # ============================================================
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, reward_for_val=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

                # Selection-specific: add validation manager stats
                if self.val_data_manager:
                    val_manager_stats = self.val_data_manager.get_stats()
                    for k, v in val_manager_stats.items():
                        metrics[f"selection/{k}"] = v

                # this is experimental and may be changed/removed in the future
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)


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
