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
Main entry point for PPO training with RLVR activation-based selection.

This extends main_ppo.py to use the RayRLVRTrainer which implements:
- GREATS: Online per-batch selection based on activation similarity
- Streaming: Per-layer selection during backward pass

Usage:
    python -m verl.trainer.main_ppo_rlvr +data.rlvr_enabled=true +data.rlvr_selection_method=GREATS

Config options:
    data.rlvr_enabled: true/false - Enable RLVR selection
    data.rlvr_selection_method: GREATS/Streaming - Selection method
    data.alpha: 0.5 - Target difficulty (Goldilocks zone)
    data.tau: 0.1 - Selection temperature
    data.rlvr_selection_ratio: 0.5 - Fraction of batch to select
    data.rlvr_selection_mode: soft/hard - Probabilistic or deterministic
    data.rlvr_aggregation: mean/max/min - How to aggregate layer difficulties
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.teacher_utils import TeacherModelWorker

from deepscaler.rewards.math_reward import deepscaler_reward_fn


def _select_rm_score_fn(data_source):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        return deepscaler_reward_fn


class RewardManager():
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, format_reward=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_reward = format_reward

    def __call__(self, data: DataProto):
        """Compute rewards for the batch."""
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        from concurrent.futures import ThreadPoolExecutor

        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            score = compute_score_fn(
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_reward=self.format_reward
            )

            return i, score, valid_response_length

        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Download checkpoint
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # Instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # Define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # Add teacher model worker if not using random selection and RLVR is disabled
    rlvr_enabled = getattr(config.data, 'rlvr_enabled', False)
    if not config.data.random_selection and not rlvr_enabled:
        role_worker_mapping[Role.Teacher] = ray.remote(TeacherModelWorker)
        mapping[Role.Teacher] = global_pool_id

    # Reward model
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        format_reward=config.reward_model.format_reward
    )
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        format_reward=config.reward_model.format_reward
    )

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec,
        mapping=mapping
    )

    # Choose trainer based on RLVR config
    if rlvr_enabled:
        from verl.trainer.ppo.ray_trainer_rlvr import RayRLVRTrainer
        print("=" * 60)
        print("Using RayRLVRTrainer with activation-based selection")
        print(f"  Selection method: {getattr(config.data, 'rlvr_selection_method', 'GREATS')}")
        print(f"  Alpha (target difficulty): {getattr(config.data, 'alpha', 0.5)}")
        print(f"  Tau (temperature): {getattr(config.data, 'tau', 0.1)}")
        print(f"  Selection ratio: {getattr(config.data, 'rlvr_selection_ratio', 0.5)}")
        print("=" * 60)

        trainer = RayRLVRTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
    else:
        # Fallback to original trainer
        from verl.trainer.ppo.ray_trainer_teacher import RayPPOTrainer
        print("Using original RayPPOTrainer with teacher-based selection")

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )

    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
