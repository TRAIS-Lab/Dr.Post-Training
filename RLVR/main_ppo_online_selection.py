"""
PPO training with online validation gradient-based data selection.

This script implements the full gradient-based selection pipeline:
1. Generates fresh rollouts for validation prompts using current policy
2. Computes rewards on validation responses
3. Captures validation gradients for selection
4. Applies Streaming or GREATS selection during training

This matches the intended design from RLHF/train/trainer.py.

Usage:
    python main_ppo_online_selection.py \
        data.train_files=/path/to/train.parquet \
        selection.enable=True \
        selection.method=Streaming \
        selection.frac=0.5 \
        selection.val_prompts_path=/path/to/val_prompts.parquet
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

import hydra
import ray
import torch
from omegaconf import OmegaConf, open_dict
from pprint import pprint
import socket

from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.dataset.rl_dataset import collate_fn
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.fs import copy_to_local

from gradstream_verl.selection_trainer import (
    SelectionRayPPOTrainerWithOnlineVal,
    SelectionTrainerConfig,
)
from gradstream_verl.validation_manager import ValidationConfig
from gradstream_verl.worker_selection import SelectionActorRolloutRefWorker


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples=-1):
    """Create RL dataset (same as verl's main_ppo.py)."""
    from verl.utils.dataset.rl_dataset import get_dataset_class
    dataset_cls = get_dataset_class(data_config)
    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )


def create_rl_sampler(data_config, dataset):
    """Create RL sampler (same as verl's main_ppo.py)."""
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        return RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        return SequentialSampler(data_source=dataset)


class SelectionTaskRunner(BaseTaskRunner):
    """Task runner for selection-enabled PPO training."""

    def add_actor_rollout_worker(self, config):
        """Override to use selection-enabled worker for FSDP strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # New engine implementation doesn't support selection yet
        if use_legacy_worker_impl == "disable":
            print("[Selection] WARNING: New engine implementation doesn't support selection yet.")
            print("[Selection] Falling back to parent implementation.")
            return super().add_actor_rollout_worker(config)

        # Use selection-enabled worker for FSDP strategy
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            # Check if selection is enabled
            selection_enabled = config.get("selection", {}).get("enable", False)

            if selection_enabled:
                print("[Selection] Using SelectionActorRolloutRefWorker for gradient-based selection")
                actor_rollout_cls = SelectionActorRolloutRefWorker
            else:
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
                actor_rollout_cls = AsyncActorRolloutRefWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            print("[Selection] WARNING: Megatron strategy doesn't support selection yet.")
            return super().add_actor_rollout_worker(config)

        else:
            raise NotImplementedError(f"Unknown strategy: {config.actor_rollout_ref.actor.strategy}")

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    def run(self, config):
        """Execute the main PPO training workflow with online validation selection."""
        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Add worker classes
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # Validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Load model
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load reward functions
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        # Initialize resource pools
        resource_pool_manager = self.init_resource_pool_mgr(config)

        # Create datasets
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Get selection config
        selection_cfg = config.get("selection", {})
        selection_enabled = selection_cfg.get("enable", False)

        if selection_enabled:
            print("[Selection] Creating trainer with online validation rollouts...")

            # Build selection trainer config
            selection_trainer_config = SelectionTrainerConfig(
                enable=True,
                method=selection_cfg.get("method", "Streaming"),
                frac=selection_cfg.get("frac", 0.5),
                use_second_order=selection_cfg.get("use_second_order", False),
                val_prompts_path=selection_cfg.get("val_prompts_path"),
                val_pool_size=selection_cfg.get("val_pool_size", 500),
                val_batch_size=selection_cfg.get("val_batch_size", 32),
                val_max_prompt_length=selection_cfg.get("val_max_prompt_length", 1024),
                val_max_response_length=selection_cfg.get("val_max_response_length", 1024),
                val_seed=selection_cfg.get("val_seed"),
                refresh_freq=selection_cfg.get("refresh_freq", 1),
                val_loss_type=selection_cfg.get("val_loss_type", "seqloss-reward"),
            )

            trainer = SelectionRayPPOTrainerWithOnlineVal(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
                selection_config=selection_trainer_config,
            )
        else:
            print("[Selection] Disabled, using standard RayPPOTrainer")
            trainer = RayPPOTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=ray_worker_group_cls,
                reward_fn=reward_fn,
                val_reward_fn=val_reward_fn,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
            )

        print("[DEBUG] Calling trainer.init_workers()...")
        trainer.init_workers()
        print("[DEBUG] Calling trainer.fit()...")
        trainer.fit()


def run_ppo_online_selection(config):
    """Run PPO with online validation selection."""
    # Initialize Ray
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Run task
    task_runner_class = ray.remote(num_cpus=1)(SelectionTaskRunner)
    runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # Save timeline if configured
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@hydra.main(config_path="verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point."""
    auto_set_device(config)

    # Add default selection config if not present
    with open_dict(config):
        if "selection" not in config:
            config.selection = {
                "enable": False,
                "method": "NA",
                "frac": 0.5,
                "use_second_order": False,
                "val_prompts_path": None,
                "val_pool_size": 500,
                "val_batch_size": 32,
                "val_max_prompt_length": 1024,
                "val_max_response_length": 1024,
                "val_seed": None,
                "refresh_freq": 1,
                "val_loss_type": "seqloss-reward",
            }

        if config.selection.get("enable", False):
            print(f"[Selection] Enabled with online validation rollouts:")
            print(f"  method={config.selection.method}")
            print(f"  frac={config.selection.frac}")
            print(f"  val_prompts_path={config.selection.get('val_prompts_path')}")
            print(f"  val_pool_size={config.selection.get('val_pool_size', 500)}")
            print(f"  val_batch_size={config.selection.get('val_batch_size', 32)}")
            print(f"  refresh_freq={config.selection.get('refresh_freq', 1)}")

    run_ppo_online_selection(config)


if __name__ == "__main__":
    main()
