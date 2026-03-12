# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from functools import partial
from collections import Counter
from typing import Optional, Type
from datasets import Dataset as HFDataset
import time
import random
import threading
import datasets
from verl.trainer.ppo.translation_openai import run_async_translation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"
os.environ["GLOG_minloglevel"] = "2"

# 读取StS调试开关
DEBUGGING_STS = os.environ.get("DEBUGGING_STS", "False").lower() in ["true", "1", "yes"]


import numpy as np
import ray
import torch
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.metric_utils import bootstrap_metric
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.dataset.rl_dataset import BufferedDataLoader
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger


WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, config=None):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.get("pf_ppo_reweight_method", "pow"),
                config.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            # Get length from the initial response mask
            response_length = grpo_calculation_mask.size(1)
            # This mask is the one intended for GRPO
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to "cuda".
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None or config.actor_rollout_ref.rollout.multi_turn.interaction_config_path is not None, "tool_config_path or interaction_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"step_{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.actor_rollout_wg.world_size if not self.async_rollout_mode else self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            scores, result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        # 过滤掉非数值字段，只保留可以计算metric的数值字段
        # 非数值字段如'pred'(可能是None)和'language_code'(字符串)不应该参与metric计算
        non_numeric_fields = {'pred', 'language_code'}
        filtered_reward_extra_infos_dict = {
            key: lst for key, lst in reward_extra_infos_dict.items() 
            if key not in non_numeric_fields
        }

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, filtered_reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val
        
        # 添加额外的奖励信息记录（验证阶段）
        # 这些字段从原始reward_extra_infos_dict中获取，确保是数值型字段
        for key_info in ['accuracy_reward', 'format_reward', 'language_reward', 'language_probability', 'language_weight']:
            if key_info in reward_extra_infos_dict and len(reward_extra_infos_dict[key_info]) > 0:
                # 确保所有值都是数值型
                values = reward_extra_infos_dict[key_info]
                if all(isinstance(v, (int, float, np.number)) for v in values):
                    metric_dict[f"val-aux/reward/{key_info}_mean"] = np.mean(values)
                    metric_dict[f"val-aux/reward/{key_info}_std"] = np.std(values)
                    metric_dict[f"val-aux/reward/{key_info}_max"] = np.max(values)
                    metric_dict[f"val-aux/reward/{key_info}_min"] = np.min(values)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, "worker_nsight_options must be set when profile_steps is set"
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(OmegaConf.select(self.config.trainer, "worker_nsight_options"))

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            scores, reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # 打印样例
                    print(f"\n{'='*80}")
                    print(f"打印样例")
                    print(f"{'='*80}")
                    stage1_sample_idx = 0  # 打印第一个样例
                    stage1_prompt = self.tokenizer.decode(batch.batch['prompts'][stage1_sample_idx], skip_special_tokens=True)
                    stage1_response = self.tokenizer.decode(batch.batch['responses'][stage1_sample_idx], skip_special_tokens=True)
                    print(f"[Prompt]:\n{stage1_prompt}")
                    print(f"\n[Response]:\n{stage1_response}")
                    print(f"{'='*80}\n")
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            
                            # 记录训练阶段的额外奖励信息，与RayStSTrainer保持一致
                            for key_info in ['accuracy_reward', 'format_reward', 'language_reward', 'language_probability', 'language_weight']:
                                if key_info in reward_extra_infos_dict and len(reward_extra_infos_dict[key_info]) > 0:
                                    # 确保所有值都是数值型
                                    values = reward_extra_infos_dict[key_info]
                                    if all(isinstance(v, (int, float, np.number)) for v in values):
                                        metrics[f"reward/{key_info}_mean"] = np.mean(values)
                                        metrics[f"reward/{key_info}_std"] = np.std(values)
                                        metrics[f"reward/{key_info}_max"] = np.max(values)
                                        metrics[f"reward/{key_info}_min"] = np.min(values)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor


                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(max_steps_duration=self.max_steps_duration, redundant_time=self.config.trainer.esi_redundant_time)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

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
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
LANGUAGE_MAP = {
    "ZH": "Chinese",
    "EN": "English",
    "JA": "Japanese",
    "KO": "Korean",
    "FR": "French",
    "DE": "German",
    "ES": "Spanish",
    "IT": "Italian",
    "PT": "Portuguese",
    "RU": "Russian",
    "AR": "Arabic",
    "BN": "Bengali",
    "KO": "Korean",
    "PT": "Portuguese",
    "TH": "Thai"
}

LANGUAGE_REASONING_MAP = {
    "EN": "\nPlease reason step by step, and put your final answer within \\boxed{}.",
    "DE": "\nBitte begründen Sie Schritt für Schritt, und fassen Sie Ihre endgültige Antwort in \\boxed{}.",
    "FR": "\nVeuillez raisonner étape par étape, et placez votre réponse finale dans \\boxed{}.",
    "BN": "\nদয়া করে ধাপে ধাপে যুক্তি দিন এবং আপনার চূড়ান্ত উত্তরটি \\boxed{} এর মধ্যে লিখুন।",
    "ZH": "\n请逐步推理，并将最终答案放在 \\boxed{} 中。",
    "JA": "\n段階的に推理し、最終的な答えを\\boxed{}の中に入れてください。",
    "BN": "\nঅনুগ্রহ করে ধাপে ধাপে বিশ্লেষণ করুন এবং চূড়ান্ত উত্তরের চারপাশে \\boxed{} দিন।",
    "KO": "\n단계별로 논리적으로 설명해 주시고, 최종 답변을 \\boxed{} 안에 넣어 주세요.",
    "TH": "\nโปรดให้เหตุผลทีละขั้นตอน และใส่คำตอบสุดท้ายไว้ใน \\boxed{}.",
    "PT": "\nPor favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{}."
}
LANGUAGE_START_PREFIX_MAP = {
    "ZH": "<think>\n好的",
    "FR": "<think>\nD’accord",
    "BN": "<think>\nঠিক আছে",
    "EN": "<think>\nOkay",
    "JA": "<think>\nさて",
    "BN": "<think>\nঠিক আছে",
    "KO": "<think>\n네",
    "PT": "<think>\nCerto",
    "TH": "<think>\nโอเค"
}
LANGUAGE_START_PREFIX_DISTILL_MAP = {
    "ZH": "好的",
    "FR": "D’accord",
    "BN": "ঠিক আছে",
    "EN": "Okay",
    "JA": "<think>\nさて",
    "BN": "ঠিক আছে",

}
import re
from langdetect import detect_langs, DetectorFactory, LangDetectException
def clean_latex_code(text):
    """
    清理文本中的LaTeX代码
    
    这个函数的目的是移除可能干扰语言检测的LaTeX元素。
    我们希望专注于自然语言部分，而不是技术标记。
    
    Args:
        text (str): 原始文本
    
    Returns:
        str: 清理后的文本
    """
    
    # 移除各种LaTeX数学环境
    # 按照从最具体到最一般的顺序进行清理
    
    # 1. 移除显示模式数学公式 $$...$$
    text = re.sub(r'\$\$.*?\$\$', ' ', text, flags=re.DOTALL)
    
    # 2. 移除LaTeX括号式数学公式 \[...\] 和 \(...\)
    text = re.sub(r'\\\[.*?\\\]', ' ', text, flags=re.DOTALL)
    text = re.sub(r'\\\(.*?\\\)', ' ', text, flags=re.DOTALL)
    
    # 3. 移除行内数学公式 $...$
    # 这里使用更保守的匹配，避免误删普通的美元符号
    text = re.sub(r'\$[^$\n]{1,100}\$', ' ', text)
    
    # 4. 移除LaTeX命令 \command{...}
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', ' ', text)
    
    # 5. 移除单独的LaTeX命令（如 \alpha, \beta）
    text = re.sub(r'\\[a-zA-Z]+\b', ' ', text)
    
    # 6. 清理多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
DetectorFactory.seed = 0
# 定义语言映射
LANGUAGE_CODE_MAP = {
    "ZH": ["zh-cn", "zh"],
    "EN": ["en"],
    "JA": ["ja"],
    "FR": ["fr"],
    "DE": ["de"],
    "ES": ["es"],
    "RU": ["ru"],
    "BN": ["bn"],
    "SW": ["sw"],
    "TE": ["te"],
    "TH": ["th"],
    "PT": ["pt"],
    "KO": ["ko"]
}

def extract_last_translation(text: str, language_code: str) -> str:
    """
    从模型返回的文本中提取最后一个 <TRANSLATION> 标签内的内容。
    如果不存在则返回 None；
    如果存在则检测语言是否与目标 language_code 匹配，不匹配返回 None。
    """
    if "<TRANSLATION>" not in text:
        return None

    # 提取最后一个 <TRANSLATION> 段落
    text_segment = "<TRANSLATION>" + text.split("<TRANSLATION>")[-1]
    matches = re.findall(r'<TRANSLATION>(.*?)</TRANSLATION>', text_segment, re.DOTALL)

    if not matches:
        return None

    translation = matches[-1].strip()
    translation_remove_latex_code = clean_latex_code(translation)
    # 检查语言
    try:
        detected_langs = detect_langs(translation_remove_latex_code)
        # 获取得分最高的语言
        top_lang = detected_langs[0].lang.lower()
    except LangDetectException:
        return None
    # print(detected_langs)
    # 获取可接受的语言列表
    target_langs = [l.lower() for l in LANGUAGE_CODE_MAP.get(language_code.upper(), [])]

    # 如果检测语言在目标语言映射中则返回翻译，否则返回 None
    if top_lang in target_langs:
        if len(translation.split("\n\n")) > 2:
            return None
        return translation
    else:
        return None


class RayTRITTrainer(RayPPOTrainer):
    """Self-improving via Self-translation Trainer"""
    # 第一阶段把大于0的部分全部纳入训练
    def fit(self):
        """
        Self-Translation训练流程，包含三个阶段：
        1. 英文推理与过滤
        2. 翻译成目标语言
        3. 多语言推理与翻译验证
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        
        # 获取StS相关配置
        self.target_language = self.config.data.get("target_language", "ZH")
        self.translation_acc_lower = self.config.data.get("translation_acc_lower", 0.5)
        self.translation_acc_upper = self.config.data.get("translation_acc_upper", 1.0)
        self.qt_training_ratio = self.config.data.get("qt_training_ratio", 1.0)
        
        # 读取翻译prompt模板

        with open("prompts/translation_template.txt", "r", encoding="utf-8") as f:
            self.translation_prompt = f.read().strip()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True) and self.global_steps == 0:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="StS Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0
        n_samples = self.config.actor_rollout_ref.rollout.n

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                do_profile = self.global_steps in self.config.trainer.profile_steps if self.config.trainer.profile_steps is not None else False
                if do_profile:
                    self.actor_rollout_wg.start_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()

                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    ##### 阶段一：英文推理与过滤 #####
                    print(f">>> 阶段一：英文推理与过滤")
                    with marked_timer("stage1_english_reasoning", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    sample_languages = []
                    if "extra_info" in batch.non_tensor_batch:
                        for extra_info in batch.non_tensor_batch["extra_info"]:
                            # 从extra_info中提取lang字段
                            if isinstance(extra_info, dict) and "lang" in extra_info:
                                sample_lang = extra_info["lang"]
                            else:
                                assert False, "No Lang in Extra Info Stage1"
                                # 如果没有lang字段，使用默认的target_language
                                # sample_lang = self.target_language
                            sample_languages.append(sample_lang)
                        # 转换为numpy array
                        batch.non_tensor_batch["sample_lang"] = np.array(sample_languages, dtype=object)
                    else:
                        assert False, "No Extra Info Stage1"
                        # 如果没有extra_info，全部使用默认的target_language
                        # batch.non_tensor_batch["sample_lang"] = np.array([self.target_language] * len(batch.batch), dtype=object)

                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # 打印阶段一样例
                    print(f"\n{'='*80}")
                    print(f"阶段一样例（英文推理）")
                    print(f"{'='*80}")
                    stage1_sample_idx = 0  # 打印第一个样例
                    stage1_prompt = self.tokenizer.decode(batch.batch['prompts'][stage1_sample_idx], skip_special_tokens=True)
                    stage1_response = self.tokenizer.decode(batch.batch['responses'][stage1_sample_idx], skip_special_tokens=True)
                    print(f"[Prompt]:\n{stage1_prompt}")
                    print(f"\n[Response]:\n{stage1_response}")
                    print(f"{'='*80}\n")

                    # Balance the number of valid tokens across DP ranks
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    stage1_use_en_think = os.getenv("STAGE1_USE_EN_THINK")
                    if stage1_use_en_think and stage1_use_en_think.lower() in ["true", "1"]:
                        # print("--------------阶段一使用英文think---------------------------")
                        os.environ["USE_ALL_EN_THINK"] = "1"
                    # 计算英文回答的奖励
                    with marked_timer("stage1_reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            reward_tensor, stage1_reward_extra_infos_dict = ray.get(future_reward)
                        else:
                            reward_tensor, stage1_reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        batch.batch['reward_tensor'] = reward_tensor  # 直接使用reward_tensor，不需要[1]索引
                    
                    metrics['sts/stage1_english_reward'] = torch.mean(torch.sum(batch.batch['reward_tensor'], dim=-1)).item()
                    if stage1_use_en_think and stage1_use_en_think.lower() in ["true", "1"]:
                        os.environ["USE_ALL_EN_THINK"] = "0"
                    # 记录阶段一的额外奖励信息
                    if stage1_reward_extra_infos_dict:
                        for key_info in ['accuracy_reward', 'format_reward', 'language_reward', 'language_probability', 'language_weight']:
                            if key_info in stage1_reward_extra_infos_dict and len(stage1_reward_extra_infos_dict[key_info]) > 0:
                                metrics[f"reward/stage1_{key_info}_mean"] = np.mean(stage1_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage1_{key_info}_std"] = np.std(stage1_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage1_{key_info}_max"] = np.max(stage1_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage1_{key_info}_min"] = np.min(stage1_reward_extra_infos_dict[key_info])

                    sts_metrics = {}
                    
                    # 计算每个样本的准确率
                    sample_accs = {}
                    for idx in range(0, len(batch.non_tensor_batch["uid"])):
                        if batch.non_tensor_batch["uid"][idx] not in sample_accs:
                            sample_accs[batch.non_tensor_batch["uid"][idx]] = []
                        sample_accs[batch.non_tensor_batch["uid"][idx]].append(batch.batch['acc'][idx].item())
                    sample_accs = {k: np.mean(v).item() for k, v in sample_accs.items()}

                    # 阶段一过滤：准确率过滤
                    stage1_filtered_indices = []     # 所有准确率>=下限的数据
                    stage1_training_indices = []     # 用于参数更新的数据（准确率在[lower, upper)区间）
                    stage2_input_indices = []        # 进入阶段二的数据（准确率>=下限的问题，每个问题只选一次）
                    use_all_data = os.getenv("USE_ALL_DATA", None)
                    use_all_stage1_data = True if use_all_data and use_all_data.lower() in ["true", "1"] else False

                    for k in sample_accs.keys():
                        cur_uid_indices = np.where(batch.non_tensor_batch["uid"] == k)[0].tolist()
                        
                        # 准确率>=下限的数据都可以进入后续阶段
                        if sample_accs[k] >= self.translation_acc_lower or use_all_stage1_data:
                            stage1_filtered_indices.extend(cur_uid_indices)
                            
                            # 进入翻译阶段：准确率>=下限的问题进入翻译
                            stage2_input_indices.append(cur_uid_indices[0])
                            
                            # 用于参数更新：准确率在[lower, upper)区间内的数据
                            if 0 < sample_accs[k] < self.translation_acc_upper:
                                stage1_training_indices.extend(cur_uid_indices)
                        elif sample_accs[k] > 0:
                            stage1_training_indices.extend(cur_uid_indices)

                    if len(stage2_input_indices) == 0:
                        print("阶段一过滤后没有合适的数据进入阶段二，跳过此iteration")
                        continue
                    
                    sts_metrics["StS-Valid-Ratio/Stage1-all-filtered"] = np.round(len(stage1_filtered_indices) * 100 / len(batch), 2).item()
                    sts_metrics["StS-Valid-Ratio/Stage1-training"] = np.round(len(stage1_training_indices) * 100 / len(batch), 2).item()
                    sts_metrics["StS-Valid-Ratio/Stage2-input"] = np.round(len(stage2_input_indices) * 100 / len(batch), 2).item()

                    ##### 阶段二：翻译成目标语言 #####
                    print(f">>> 阶段二：翻译成目标语言 ({self.target_language})")
                    
                    # 准备翻译输入
                    stage2_batch = batch.select_idxs(stage2_input_indices)
                    stage2_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(stage2_batch.batch))], dtype=object)
                    stage2_languages = stage2_batch.non_tensor_batch["sample_lang"].tolist()
                    
                    # 获取英文问题文本（优先使用数据集中的en_question字段）
                    if "en_question" in stage2_batch.non_tensor_batch:
                        english_questions = stage2_batch.non_tensor_batch["en_question"].tolist()
                    else:
                        # 如果没有en_question字段，则从聚天模板中提取
                        english_questions = self.tokenizer.batch_decode(stage2_batch.batch['prompts'], skip_special_tokens=True)
                    
                    # 构建翻译输入
                    translation_inputs = []
                    for eq, lang in zip(english_questions, stage2_languages):
                        # 如果是从数据集获取的en_question，直接使用
                        if "en_question" in stage2_batch.non_tensor_batch:
                            clean_question = eq
                        else:
                            # 如果是从聚天模板解码的，需要提取原始问题
                            if "<|im_start|>user\n" in eq and "<|im_end|>" in eq:
                                clean_question = eq.split("<|im_start|>user\n")[1].split("<|im_end|>")[0].strip()
                            else:
                                clean_question = eq.strip()
                        
                        translation_input = self.translation_prompt.format(
                            language=LANGUAGE_MAP[lang.upper()],  # 使用样本特定的语言
                            question=clean_question
                        )
                        translation_input = self.tokenizer.apply_chat_template(
                            [{"content": translation_input, "role": "user"}], 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        translation_inputs.append(translation_input)
                    
                    # 构建翻译生成batch
                    translation_model_inputs = self.tokenizer(
                        translation_inputs, 
                        return_tensors="pt", 
                        add_special_tokens=False, 
                        padding=True, 
                        truncation=True, 
                        max_length=self.config.data.max_prompt_length
                    )
                    
                    trans_input_ids = translation_model_inputs.pop("input_ids")
                    trans_attention_mask = translation_model_inputs.pop("attention_mask")
                    trans_input_ids, trans_attention_mask = verl_F.postprocess_data(
                        input_ids=trans_input_ids,
                        attention_mask=trans_attention_mask,
                        max_length=self.config.data.max_prompt_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,
                        truncation=self.config.data.get("truncation", "error"),
                    )
                    trans_position_ids = compute_position_id_with_mask(trans_attention_mask)
                    trans_raw_prompt_ids = np.array(self.tokenizer.batch_encode_plus(translation_inputs, add_special_tokens=False)['input_ids'], dtype=object)
                    trans_tools_kwargs = np.array([{} for _ in range(len(trans_position_ids))], dtype=object)
                    
                    trans_gen_dict = {
                        "input_ids": trans_input_ids,
                        "attention_mask": trans_attention_mask,
                        "position_ids": trans_position_ids,
                        "raw_prompt_ids": trans_raw_prompt_ids,
                        "tools_kwargs": trans_tools_kwargs,
                    }
                    
                    # 生成翻译结果
                    with marked_timer("stage2_translation", timing_raw, color="blue"):
                        trans_gen_batch = DataProto.from_single_dict(trans_gen_dict)
                        # 按照SvS的模式，显式进行repeat
                        trans_gen_batch = trans_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.translation_sample_n, interleave=True)
                        
                        trans_gen_batch.meta_info = {
                            "translation": True, 
                            "temperature": self.config.actor_rollout_ref.rollout.get("translation_temperature", 0.7), 
                            "top_p": self.config.actor_rollout_ref.rollout.get("translation_top_p", 0.9), 
                            "n": 1
                        }
                        
                        # 打印阶段二样例（翻译输入）
                        print(f"\n{'='*80}")
                        print(f"阶段二样例（翻译）- 输入")
                        print(f"{'='*80}")
                        stage2_sample_idx = 0  # 打印第一个样例
                        stage2_prompt = self.tokenizer.decode(trans_gen_batch.batch['input_ids'][stage2_sample_idx], skip_special_tokens=True)
                        print(f"[Prompt]:\n{stage2_prompt}")
                        print(f"{'='*80}\n")
                        
                        trans_gen_batch_padded, pad_size = pad_dataproto_to_divisor(trans_gen_batch, self.actor_rollout_wg.world_size)
                        if not self.async_rollout_mode:
                            trans_output_padded = self.actor_rollout_wg.generate_sequences(trans_gen_batch_padded)
                        else:
                            self.async_rollout_manager.wake_up()
                            trans_output_padded = self.async_rollout_manager.generate_sequences(trans_gen_batch_padded)
                            self.async_rollout_manager.sleep()
                        trans_output = unpad_dataproto(trans_output_padded, pad_size=pad_size)
                        
                        # 打印阶段二样例（翻译输出）
                        print(f"\n{'='*80}")
                        print(f"阶段二样例（翻译）- 输出")
                        print(f"{'='*80}")
                        stage2_response = self.tokenizer.decode(trans_output.batch['responses'][stage2_sample_idx], skip_special_tokens=True)
                        print(f"[Response]:\n{stage2_response}")
                        print(f"{'='*80}\n")
                    
                    # 更新stage2_batch
                    # 按照SvS的模式，对stage2_batch也进行相同的repeat操作
                    print(f">>> Debug: 在repeat前 stage2_batch batch size: {len(stage2_batch.batch)}")
                    print(f">>> Debug: trans_output batch size: {len(trans_output.batch)}")
                    stage2_batch = stage2_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.translation_sample_n, interleave=True)
                    
                    print(f">>> Debug: 在repeat后 stage2_batch batch size: {len(stage2_batch.batch)}")
                    stage2_batch_keys_to_pop = ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids']
                    stage2_non_tensor_batch_keys_to_pop = ["tools_kwargs"]
                    stage2_batch.pop(batch_keys=stage2_batch_keys_to_pop, non_tensor_batch_keys=stage2_non_tensor_batch_keys_to_pop)
                    stage2_batch = stage2_batch.union(trans_output)
                    stage2_batch.batch["response_mask"] = compute_response_mask(stage2_batch)
                    stage2_languages = stage2_batch.non_tensor_batch["sample_lang"].tolist()
                    # 提取翻译结果
                    valid_translation_indices = []
                    translated_questions = []
                    responses_strs = self.tokenizer.batch_decode(stage2_batch.batch['responses'], skip_special_tokens=True)
                    
                    for idx, response in enumerate(responses_strs):
                        try:
                            # 提取翻译结果
                            sample_lang = stage2_languages[idx]
                            extracted = extract_last_translation(response, sample_lang)
                            if extracted and len(extracted) > 10:  # 基本长度检查
                                translated_questions.append(extracted)
                                valid_translation_indices.append(idx)
                            else:
                                translated_questions.append(None)
                        except:
                            translated_questions.append(None)
                    
                    # 过滤有效翻译
                    valid_indices = [i for i in valid_translation_indices if translated_questions[i] is not None]
                    if len(valid_indices) == 0:
                        print("阶段二翻译后没有有效结果，跳过此iteration")
                        continue
                    
                    sts_metrics["StS-Valid-Ratio/Stage2-extracted"] = np.round(len(valid_indices) * 100 / len(stage2_batch), 2).item()
                    
                    # 计算翻译阶段的奖励（稀疏奖励，只在最后一个token）
                    translation_reward_tensor = torch.zeros_like(stage2_batch.batch["responses"], dtype=torch.float32)
                    valid_response_length = stage2_batch.batch["attention_mask"][:, stage2_batch.batch["prompts"].shape[-1]:].sum(dim=-1)
                    
                    # 暂时给所有有效翻译分配奖励，具体值将在阶段三确定
                    for i in valid_indices:
                        translation_reward_tensor[i, valid_response_length[i].item() - 1] = 0.0  # 占位符，将在阶段三更新
                    
                    stage2_batch.batch['reward_tensor'] = translation_reward_tensor
                    stage2_batch.batch['acc'] = torch.sum(translation_reward_tensor, dim=1)
                    
                    ##### 阶段三：多语言推理与翻译验证 #####
                    print(f">>> 阶段三：多语言推理与翻译验证")
                    
                    # 为有效翻译构建多语言推理输入
                    stage3_batch = stage2_batch.select_idxs(valid_indices)
                    if DEBUGGING_STS:
                        print(f">>> Debug: 阶段三初始 stage3_batch size: {len(stage3_batch.batch)}")
                        
                        # ===== Debug信息：验证select_idxs后的uid继承 =====
                        print(f"\n========== 阶段三batch构建 ==========")
                        print(f"从stage2选取的valid_indices: {valid_indices[:5]}... (前5个)")
                        print(f"stage3_batch初始大小: {len(stage3_batch)}")
                        print(f"即将为stage3_batch生成{len(stage3_batch)}个新uid")
                        print(f"=========================================\n")
                    
                    stage3_batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(stage3_batch.batch))], dtype=object)
                    stage3_languages = stage3_batch.non_tensor_batch["sample_lang"].tolist()
                    multilingual_questions = [translated_questions[i] for i in valid_indices]
                    multilingual_inputs = []
                    for mq, lang in zip(multilingual_questions, stage3_languages):
                        # 使用当前样本的目标语言
                        reasoning_input = mq + LANGUAGE_REASONING_MAP[lang.upper()]  # 使用样本特定的语言
                        
                        reasoning_input = self.tokenizer.apply_chat_template(
                            [{"content": reasoning_input, "role": "user"}], 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        reasoning_input += LANGUAGE_START_PREFIX_MAP[lang.upper()]  # 使用样本特定的语言
                        multilingual_inputs.append(reasoning_input)
                    
                    # 构建多语言推理batch
                    multilingual_model_inputs = self.tokenizer(
                        multilingual_inputs, 
                        return_tensors="pt", 
                        add_special_tokens=False, 
                        padding=True, 
                        truncation=True,
                        max_length=self.config.data.max_prompt_length
                    )
                    
                    multi_input_ids = multilingual_model_inputs.pop("input_ids")
                    multi_attention_mask = multilingual_model_inputs.pop("attention_mask")
                    multi_input_ids, multi_attention_mask = verl_F.postprocess_data(
                        input_ids=multi_input_ids,
                        attention_mask=multi_attention_mask,
                        max_length=self.config.data.max_prompt_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        left_pad=True,
                        truncation=self.config.data.get("truncation", "error"),
                    )
                    multi_position_ids = compute_position_id_with_mask(multi_attention_mask)
                    multi_raw_prompt_ids = np.array(self.tokenizer.batch_encode_plus(multilingual_inputs, add_special_tokens=False)['input_ids'], dtype=object)
                    multi_tools_kwargs = np.array([{} for _ in range(len(multi_position_ids))], dtype=object)
                    
                    multi_gen_dict = {
                        "input_ids": multi_input_ids,
                        "attention_mask": multi_attention_mask,
                        "position_ids": multi_position_ids,
                        "raw_prompt_ids": multi_raw_prompt_ids,
                        "tools_kwargs": multi_tools_kwargs,
                    }
                    
                    # 生成多语言推理结果
                    with marked_timer("stage3_multilingual_reasoning", timing_raw, color="green"):
                        multi_gen_batch = DataProto.from_single_dict(multi_gen_dict)
                        print(f">>> Debug: 阶段三多语言推理前 batch size: {len(multi_gen_batch.batch)}")
                        
                        # 打印阶段三样例（多语言推理输入）
                        print(f"\n{'='*80}")
                        print(f"阶段三样例（多语言推理）- 输入")
                        print(f"{'='*80}")
                        stage3_sample_idx = 0  # 打印第一个样例
                        stage3_prompt = self.tokenizer.decode(multi_gen_batch.batch['input_ids'][stage3_sample_idx], skip_special_tokens=True)
                        print(f"[Prompt]:\n{stage3_prompt}")
                        print(f"{'='*80}\n")
                        
                        # ===== Debug信息：验证repeat操作 =====
                        print(f"\n========== Stage3 Repeat操作验证 ==========")
                        before_repeat_size = len(multi_gen_batch.batch)
                        print(f"repeat前大小: {before_repeat_size}")
                        print(f"repeat参数: repeat_times={self.config.actor_rollout_ref.rollout.n}, interleave=True")
                        
                        multi_gen_batch = multi_gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        after_repeat_size = len(multi_gen_batch.batch)
                        print(f"repeat后大小: {after_repeat_size}")
                        print(f"预期大小: {before_repeat_size * self.config.actor_rollout_ref.rollout.n}")
                        print(f"大小匹配: {after_repeat_size == before_repeat_size * self.config.actor_rollout_ref.rollout.n}")
                        print(f"=========================================\n")
                        # print(f">>> Debug: 阶段三多语言推理repeat后 batch size: {len(multi_gen_batch.batch)}")
                        multi_gen_batch.meta_info = {"multilingual_reasoning": True, "n": 1}
                        
                        multi_gen_batch_padded, pad_size = pad_dataproto_to_divisor(multi_gen_batch, self.actor_rollout_wg.world_size)
                        if not self.async_rollout_mode:
                            multi_output_padded = self.actor_rollout_wg.generate_sequences(multi_gen_batch_padded)
                        else:
                            self.async_rollout_manager.wake_up()
                            multi_output_padded = self.async_rollout_manager.generate_sequences(multi_gen_batch_padded)
                            self.async_rollout_manager.sleep()
                        multi_output = unpad_dataproto(multi_output_padded, pad_size=pad_size)
                        
                        # 打印阶段三样例（多语言推理输出）
                        print(f"\n{'='*80}")
                        print(f"阶段三样例（多语言推理）- 输出")
                        print(f"{'='*80}")
                        stage3_response = self.tokenizer.decode(multi_output.batch['responses'][stage3_sample_idx], skip_special_tokens=True)
                        print(f"[Response]:\n{stage3_response}")
                        print(f"{'='*80}\n")
                        
                        if DEBUGGING_STS:
                            print(f">>> Debug: 阶段三multi_output batch size: {len(multi_output.batch)}")
                    
                    # 更新stage3_batch（需要repeat以匹配multi_output的大小）
                    if DEBUGGING_STS:
                        print(f">>> Debug: stage3_batch union前 batch size: {len(stage3_batch.batch)}")
                        
                        # ===== Debug信息：验证stage3_batch的repeat和uid分布 =====
                        print(f"\n========== Stage3_batch Repeat和UID分布 ==========")
                        print(f"stage3_batch在repeat前的uid数量: {len(set(stage3_batch.non_tensor_batch['uid']))}")
                        original_uids = stage3_batch.non_tensor_batch['uid'].copy()
                        print(f"前3个原始uid: {[uid[:8] + '...' for uid in original_uids[:3]]}")
                    
                    stage3_batch = stage3_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    if DEBUGGING_STS:
                        print(f">>> Debug: stage3_batch repeat后 batch size: {len(stage3_batch.batch)}")
                        print(f"stage3_batch在repeat后的总大小: {len(stage3_batch)}")
                        print(f"stage3_batch在repeat后的唯一uid数量: {len(set(stage3_batch.non_tensor_batch['uid']))}")
                        
                        # 验证interleave=True的效果
                        print(f"\n验证interleave=True的效果（前{min(15, len(stage3_batch))}个样本的uid）:")
                        for i in range(min(15, len(stage3_batch))):
                            uid_short = stage3_batch.non_tensor_batch['uid'][i][:8]
                            original_idx = i % len(original_uids) if self.config.actor_rollout_ref.rollout.n > 1 else i
                            expected_uid = original_uids[original_idx][:8]
                            match = "✓" if uid_short == expected_uid else "✗"
                            print(f"  索引{i:2d}: uid={uid_short}... (预期={expected_uid}...) {match}")
                        print(f"=========================================\n")
                    stage3_batch_keys_to_pop = ['prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids']
                    stage3_non_tensor_batch_keys_to_pop = ["tools_kwargs"]
                    meta_info_keys_to_pop = ["timing"]
                    stage3_batch.pop(batch_keys=stage3_batch_keys_to_pop, 
                        non_tensor_batch_keys=stage3_non_tensor_batch_keys_to_pop, 
                        meta_info_keys=meta_info_keys_to_pop)
                    # stage3_batch.pop(batch_keys=stage3_batch_keys_to_pop, non_tensor_batch_keys=stage3_non_tensor_batch_keys_to_pop)
                    stage3_batch = stage3_batch.union(multi_output)
                    stage3_batch.batch["response_mask"] = compute_response_mask(stage3_batch)
                    if DEBUGGING_STS:
                        print(f">>> Debug: stage3_batch union后 batch size: {len(stage3_batch.batch)}")
                    
                    # 计算多语言推理的奖励
                    with marked_timer("stage3_reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(stage3_batch)
                            stage3_batch = stage3_batch.union(reward_tensor)
                        
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(stage3_batch, self.config, self.tokenizer)
                            reward_tensor, stage3_reward_extra_infos_dict = ray.get(future_reward)
                        else:
                            reward_tensor, stage3_reward_extra_infos_dict = compute_reward(stage3_batch, self.reward_fn)
                        stage3_batch.batch['multilingual_reward_tensor'] = reward_tensor  # 直接使用reward_tensor，不需要[1]索引
                    
                    # 记录阶段三的额外奖励信息
                    if stage3_reward_extra_infos_dict:
                        for key_info in ['accuracy_reward', 'format_reward', 'language_reward', 'language_probability', 'language_weight']:
                            if key_info in stage3_reward_extra_infos_dict and len(stage3_reward_extra_infos_dict[key_info]) > 0:
                                metrics[f"reward/stage3_{key_info}_mean"] = np.mean(stage3_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage3_{key_info}_std"] = np.std(stage3_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage3_{key_info}_max"] = np.max(stage3_reward_extra_infos_dict[key_info])
                                metrics[f"reward/stage3_{key_info}_min"] = np.min(stage3_reward_extra_infos_dict[key_info])
                    
                    # 计算多语言推理准确率
                    stage3_sample_accs = {}
                    for idx in range(0, len(stage3_batch.non_tensor_batch["uid"])):
                        if stage3_batch.non_tensor_batch["uid"][idx] not in stage3_sample_accs:
                            stage3_sample_accs[stage3_batch.non_tensor_batch["uid"][idx]] = []
                        stage3_sample_accs[stage3_batch.non_tensor_batch["uid"][idx]].append(stage3_batch.batch['acc'][idx].item())
                    
                    if DEBUGGING_STS:
                        # ===== Debug信息：验证数据结构 =====
                        print(f"\n========== 阶段三数据结构分析 ==========")
                        print(f"valid_indices长度: {len(valid_indices)}")
                        print(f"stage3_batch总长度: {len(stage3_batch)}")
                        print(f"stage3中唯一uid数量: {len(stage3_sample_accs)}")
                        print(f"每个uid的样本数: {[len(v) for v in list(stage3_sample_accs.values())[:3]]}... (前3个)")
                        print(f"rollout.n配置: {self.config.actor_rollout_ref.rollout.n}")
                        
                        # 展示uid分布（前10个样本）
                        print(f"\n前10个stage3样本的uid:")
                        for idx in range(min(10, len(stage3_batch))):
                            print(f"  索引{idx}: uid={stage3_batch.non_tensor_batch['uid'][idx][:8]}..., acc={stage3_batch.batch['acc'][idx].item()}")
                        print(f"=========================================\n")
                    
                    # 更新翻译阶段的奖励（基于多语言推理结果）
                    # 第二阶段：收集所有翻译尝试（包括失败的翻译）用于参数更新
                    # stage2_final_indices = list(range(len(stage2_batch)))  # 收集所有翻译尝试
                    stage3_final_indices = []  # 只保留多语言推理准确率>0的数据
                    
                    # 初始化所有翻译的奖励为0（失败翻译保持0奖励）
                    for idx in range(len(stage2_batch)):
                        stage2_batch.batch['reward_tensor'][idx, valid_response_length[idx].item() - 1] = 0.0
                    
                    if DEBUGGING_STS:
                        print(f"\n========== 开始处理翻译奖励更新 ==========")
                        print(f"收集所有翻译尝试: {len(stage2_final_indices)} 个（包括失败翻译）")
                        print(f"其中成功翻译: {len(valid_indices)} 个")
                        print(f"失败翻译: {len(stage2_final_indices) - len(valid_indices)} 个")
                        print(f"zip循环将执行次数: {min(len(valid_indices), len(stage3_batch))}")
                    
                    # 注意：翻译失败的数据本身就不会进入第三阶段，其奖励保持0
                    # 只有成功翻译的数据才会根据第三阶段的结果更新奖励
                    if DEBUGGING_STS:
                        failed_translation_count = len(stage2_final_indices) - len(valid_indices)
                        print(f"\n特别说明：")
                        print(f"  - 翻译失败的{failed_translation_count}个数据不会进入第三阶段，奖励保持0")
                        print(f"  - 只有成功翻译的{len(valid_indices)}个数据才会根据第三阶段结果更新奖励")
                        print(f"  - 这样设计的目的：鼓励模型尝试翻译，但只奖励正确的翻译")
                        print(f"  - stage3_batch中每个翻译对应{self.config.actor_rollout_ref.rollout.n}个推理结果")
                    
                    # 构建stage2索引到stage3 uid的映射
                    # stage3_batch在repeat前为每个成功翻译生成了唯一的uid
                    # repeat后，每个uid对应n个推理结果
                    stage3_uids_list = list(stage3_sample_accs.keys())  # 所有唯一的uid
                    
                    if DEBUGGING_STS:
                        print(f"\n构建索引映射：")
                        print(f"  - valid_indices数量: {len(valid_indices)}")
                        print(f"  - stage3唯一uid数量: {len(stage3_uids_list)}")
                        print(f"  - 应该相等: {len(valid_indices) == len(stage3_uids_list)}")
                    
                    # 为每个成功翻译的stage2索引分配对应的stage3 uid
                    # valid_indices[i] 对应 stage3_uids_list[i]
                    for i, stage2_idx in enumerate(valid_indices):
                        if i >= len(stage3_uids_list):
                            if DEBUGGING_STS:
                                print(f"警告: valid_indices索引{i}超出stage3_uids_list范围")
                            break
                            
                        stage3_uid = stage3_uids_list[i]
                        multilingual_acc = np.mean(stage3_sample_accs[stage3_uid])
                        
                        if DEBUGGING_STS:
                            # Debug: 每10次循环打印一次
                            if i % 10 == 0 or i < 3:
                                same_uid_indices = np.where(stage3_batch.non_tensor_batch["uid"] == stage3_uid)[0].tolist()
                                print(f"  循环{i}: stage2_idx={stage2_idx}")
                                print(f"    -> stage3_uid={stage3_uid[:8]}..., 该uid在stage3中的所有索引: {same_uid_indices}")
                                print(f"    -> 该uid的平均准确率: {multilingual_acc:.4f}, 样本数: {len(stage3_sample_accs[stage3_uid])}")
                        
                        # 更新翻译奖励（符合memory中的规范：奖励为1或0）
                        if multilingual_acc > 0:
                            # 翻译验证成功，给予1奖励
                            stage2_batch.batch['reward_tensor'][stage2_idx, valid_response_length[stage2_idx].item() - 1] = 1.0
                            # 阶段三只保留多语言推理准确率>0的数据（所有n个推理结果都保留）
                            stage3_indices_to_add = np.where(stage3_batch.non_tensor_batch["uid"] == stage3_uid)[0].tolist()
                            if multilingual_acc < 1:
                                stage3_final_indices.extend(stage3_indices_to_add)
                            if DEBUGGING_STS and i < 3:  # 前3次详细打印
                                print(f"    -> ✓ 翻译验证成功，stage2索引{stage2_idx}奖励=1.0，添加stage3索引{stage3_indices_to_add}")
                        else:
                            # 翻译成功但验证失败（多语言推理acc=0），保持0奖励
                            # 注意：这些数据不会进入第三阶段训练
                            if DEBUGGING_STS and i < 3:  # 前3次详细打印
                                print(f"    -> ✗ 翻译成功但多语言推理acc=0，stage2索引{stage2_idx}奖励=0.0，不进入第三阶段训练")
                    
                    if DEBUGGING_STS:
                        print(f"=========================================\n")
                    
                    # 更新acc
                    stage2_batch.batch['acc'] = torch.sum(stage2_batch.batch['reward_tensor'], dim=1)
                    stage2_filtered_indices = []

                    # 获取stage2_batch中每个样本对应的原始stage2问题的uid
                    # 由于stage2_batch是通过repeat得到的，我们需要重建原始uid的映射
                    translation_sample_n = self.config.actor_rollout_ref.rollout.translation_sample_n

                    # 为stage2_batch中的每个样本分配原始问题id
                    # stage2_batch = stage2_batch.repeat(repeat_times=translation_sample_n, interleave=True)
                    # 所以索引 i 对应的原始问题索引是 i % (len(stage2_batch) // translation_sample_n)
                    original_question_count = len(stage2_batch) // translation_sample_n

                    # 按原始问题分组检查奖励
                    for orig_q_idx in range(original_question_count):
                        # 获取该原始问题的所有翻译尝试的索引
                        translation_indices = []
                        if translation_sample_n > 1:
                            # interleave=True的情况：索引分布为 0, 1, 2, ..., orig_q_idx, orig_q_idx+original_question_count, ...
                            for sample_idx in range(translation_sample_n):
                                idx = orig_q_idx + sample_idx * original_question_count
                                translation_indices.append(idx)
                        else:
                            translation_indices = [orig_q_idx]
                        
                        # 获取这些翻译的奖励值（每个翻译的总奖励）
                        rewards = []
                        for idx in translation_indices:
                            reward_sum = stage2_batch.batch['reward_tensor'][idx].sum().item()
                            rewards.append(reward_sum)
                        
                        # 检查是否所有翻译的奖励都相同
                        # 由于使用稀疏奖励，每个样本的总奖励只能是0或1
                        unique_rewards = set(rewards)
                        
                        # 如果奖励有区分度（既有0又有1），则保留这些翻译用于训练
                        if len(unique_rewards) > 1:
                            stage2_filtered_indices.extend(translation_indices)
                            if DEBUGGING_STS and orig_q_idx < 3:
                                print(f"原始问题{orig_q_idx}: 奖励{rewards}，有区分度，保留用于训练")
                        else:
                            if DEBUGGING_STS and orig_q_idx < 3:
                                print(f"原始问题{orig_q_idx}: 奖励{rewards}，无区分度（全{'0' if 0.0 in unique_rewards else '1'}），过滤掉")

                    # 更新stage2_final_indices为过滤后的索引
                    stage2_final_indices = stage2_filtered_indices
                    
                    metrics['sts/stage2_translation_reward'] = torch.mean(torch.sum(stage2_batch.batch['reward_tensor'], dim=-1)).item()
                    metrics['sts/stage3_multilingual_reward'] = torch.mean(torch.sum(stage3_batch.batch['multilingual_reward_tensor'], dim=-1)).item()
                    
                    sts_metrics["StS-Valid-Ratio/Stage2-final"] = np.round(len(stage2_final_indices) * 100 / len(stage2_batch), 2).item()
                    sts_metrics["StS-Valid-Ratio/Stage3-final"] = np.round(len(stage3_final_indices) * 100 / len(stage3_batch), 2).item()
                    
                    if DEBUGGING_STS:
                        # ===== Debug信息：验证最终结果 =====
                        print(f"\n========== 翻译奖励更新结果 ==========")
                        print(f"stage2_final_indices数量: {len(stage2_final_indices)} （收集所有翻译尝试，100%）")
                        print(f"其中成功翻译: {len(valid_indices)} 个 ({len(valid_indices)*100/len(stage2_batch):.2f}%)")
                        print(f"失败翻译: {len(stage2_final_indices) - len(valid_indices)} 个 ({(len(stage2_final_indices) - len(valid_indices))*100/len(stage2_batch):.2f}%) [奖励保持0]")
                        print(f"stage3_final_indices数量: {len(stage3_final_indices)} (占stage3_batch的{len(stage3_final_indices)*100/len(stage3_batch):.2f}%)")
                        print(f"stage2_batch中非零奖励数量: {(stage2_batch.batch['reward_tensor'].sum(dim=1) > 0).sum().item()}")
                        print(f"stage2_batch中零奖励数量: {(stage2_batch.batch['reward_tensor'].sum(dim=1) == 0).sum().item()} [包括翻译失败+验证失败]")
                        successful_verified = (stage2_batch.batch['reward_tensor'].sum(dim=1) > 0).sum().item()
                        print(f"理论上stage3_final应该是验证成功翻译的{self.config.actor_rollout_ref.rollout.n}倍")
                        print(f"验证成功的翻译数: {successful_verified}")
                        print(f"实际倍数: {len(stage3_final_indices) / max(successful_verified, 1):.2f}")
                        print(f"=========================================\n")
                    
                    ##### 组合最终训练数据 #####
                    # 阶段一：排除全对的英文回答
                    stage1_final_batch = batch.select_idxs(stage1_training_indices)
                    
                    # 阶段二：所有翻译尝试（包括失败的翻译），按照qt_training_ratio采样
                    if self.qt_training_ratio < 1.0:
                        stage2_final_indices = random.sample(stage2_final_indices, int(len(stage2_final_indices) * self.qt_training_ratio))
                    stage2_final_batch = stage2_batch.select_idxs(stage2_final_indices)
                    
                    # 阶段三：只保留准确率非0的多语言推理数据
                    stage3_final_batch = stage3_batch.select_idxs(stage3_final_indices)
                    # 获取环境变量，是否拒绝将一阶段数据加入训练
                    reject_stage1_data = os.getenv("REJECT_STAGE1_DATA")
                    if reject_stage1_data and reject_stage1_data.lower() in ["true", "1"]:
                        stage1_final_batch = []
                    # 获取环境变量，是否拒绝将二阶段数据加入训练
                    reject_stage2_data = os.getenv("REJECT_STAGE2_DATA")
                    if reject_stage2_data and reject_stage2_data.lower() in ["true", "1"]:
                        stage2_final_batch = []
                    # 获取环境变量，是否拒绝将三阶段数据加入训练
                    reject_stage3_data = os.getenv("REJECT_STAGE3_DATA")
                    if reject_stage3_data and reject_stage3_data.lower() in ["true", "1"]:
                        stage3_final_batch = []

                    # 合并所有阶段的数据
                    final_batches = []
                    if len(stage1_final_batch) > 0:
                        final_batches.append(stage1_final_batch)
                    if len(stage2_final_batch) > 0:
                        final_batches.append(stage2_final_batch)
                    if len(stage3_final_batch) > 0:
                        stage3_final_batch.batch['reward_tensor'] = stage3_final_batch.batch.pop('multilingual_reward_tensor')
                        final_batches.append(stage3_final_batch)
                    
                    if len(final_batches) == 0:
                        print("最终没有可用于训练的数据，跳过此iteration")
                        continue
                    
                    batch = DataProto.concat(final_batches)
                    
                    print(f">>> 最终训练数据组成: 阶段一: {len(stage1_final_batch)} | 阶段二: {len(stage2_final_batch)} | 阶段三: {len(stage3_final_batch)} | 总计: {len(batch)}")
                    print_sts_metrics = {k.replace('StS-Valid-Ratio/', ''): v for k, v in sts_metrics.items()}
                    print(f">>> StS数据有效比例: {print_sts_metrics}")
                    
                    # 添加数据统计
                    sts_metrics['StS-Valid-Ratio/Num-Stage1'] = len(stage1_final_batch)
                    sts_metrics['StS-Valid-Ratio/Num-Stage2'] = len(stage2_final_batch) 
                    sts_metrics['StS-Valid-Ratio/Num-Stage3'] = len(stage3_final_batch)
                    sts_metrics['StS-Valid-Ratio/Num-Total'] = len(batch)
                    metrics.update(sts_metrics)
                    
                    # 裁剪数据以保证能被world_size整除
                    ori_batch_length = len(batch)
                    remain_batch_index = [i for i in range(ori_batch_length - ori_batch_length % self.actor_rollout_wg.world_size)]
                    if len(remain_batch_index) == 0:
                        print(f"数据不足以被{self.actor_rollout_wg.world_size}整除，跳过此iteration")
                        continue
                    batch = batch.select_idxs(remain_batch_index)
                    print(f">>> 为保证能被world_size整除，数据从{ori_batch_length}裁剪到{len(batch)}")
                    
                    # 重新计算old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # 设置token_level_scores
                        if "reward_tensor" in batch.batch.keys():
                            reward_tensor = batch.batch['reward_tensor']
                            reward_extra_infos_dict = {}
                            batch.batch["token_level_scores"] = reward_tensor
                        else:
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(max_steps_duration=self.max_steps_duration, redundant_time=self.config.trainer.esi_redundant_time)
                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

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
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if do_profile:
                    self.actor_rollout_wg.stop_profile()
                    if self.use_reference_policy:
                        self.ref_policy_wg.stop_profile()
                    if self.use_critic:
                        self.critic_wg.stop_profile()
                    if self.use_rm:
                        self.rm_wg.stop_profile()

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

