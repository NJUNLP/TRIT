# Copyright 2024 PRIME team and/or its affiliates
# Licensed under the Apache License, Version 2.0

import ray
import torch
import time
import psutil
import os
import logging
from typing import Callable, Optional, List, Dict, Any
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register

# 配置日志
logger = logging.getLogger(__name__)

# =============================================================================
# Helper Functions
# =============================================================================

def _init_ray_if_needed(
    ray_address: Optional[str] = None,
    ray_runtime_env: Optional[Dict] = None,
    ignore_reinit_error: bool = True
) -> Dict[str, Any]:
    """
    智能初始化Ray，支持单机和多机模式自动适配

    参数:
        ray_address: Ray集群地址
            - None: 自动检测环境变量 RAY_ADDRESS，如果没有则单机模式
            - 'auto': 连接到本地已启动的Ray集群
            - 'local': 强制单机模式
            - 'ip:port': 连接到指定的Ray head节点
        ray_runtime_env: Ray runtime环境配置（用于多机依赖同步）
        ignore_reinit_error: 是否忽略重复初始化错误

    返回:
        包含Ray初始化信息的字典
    """
    if ray.is_initialized():
        context = ray.get_runtime_context()
        return {
            "mode": "already_initialized",
            "node_id": context.get_node_id(),
            "cluster_resources": ray.cluster_resources()
        }

    # 1. 确定Ray地址
    if ray_address is None:
        ray_address = os.getenv('RAY_ADDRESS', 'local')

    # 2. 准备初始化参数
    init_kwargs = {
        "ignore_reinit_error": ignore_reinit_error,
        "logging_level": logging.WARNING,  # 减少Ray日志输出
    }

    # 3. 根据地址类型初始化
    try:
        if ray_address == 'local':
            # 单机模式：显式指定本地启动
            logger.info("Initializing Ray in local mode (single machine)")
            init_kwargs["num_cpus"] = psutil.cpu_count(logical=True)
            ray.init(**init_kwargs)
            mode = "local"

        elif ray_address == 'auto':
            # 自动模式：连接到已有集群，如果没有则单机启动
            logger.info("Attempting to connect to existing Ray cluster...")
            try:
                ray.init(address='auto', **init_kwargs)
                mode = "cluster"
                logger.info("Successfully connected to Ray cluster")
            except Exception as e:
                logger.warning(f"Failed to connect to cluster: {e}. Falling back to local mode.")
                ray.init(**init_kwargs)
                mode = "local_fallback"

        else:
            # 多机模式：连接到指定地址
            logger.info(f"Connecting to Ray cluster at {ray_address}")
            init_kwargs["address"] = ray_address

            # 如果提供了runtime_env，添加到配置中（用于依赖同步）
            if ray_runtime_env:
                init_kwargs["runtime_env"] = ray_runtime_env

            ray.init(**init_kwargs)
            mode = "cluster"
            logger.info(f"Successfully connected to Ray cluster at {ray_address}")

        # 4. 收集集群信息
        context = ray.get_runtime_context()
        cluster_resources = ray.cluster_resources()

        info = {
            "mode": mode,
            "address": ray_address,
            "node_id": context.get_node_id(),
            "cluster_resources": cluster_resources,
            "num_nodes": len(ray.nodes()),
        }

        logger.info(f"Ray initialized: {info}")
        return info

    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        # 最后的降级：尝试最基本的本地初始化
        try:
            ray.init(ignore_reinit_error=True)
            return {"mode": "emergency_local", "error": str(e)}
        except Exception as e2:
            logger.error(f"Emergency Ray init also failed: {e2}")
            raise RuntimeError(f"Cannot initialize Ray: {e2}") from e


# =============================================================================
# Ray Remote Worker Function
# =============================================================================

@ray.remote
def _ray_compute_single_score(
    index: int,
    solution_str: str,
    ground_truth: str,
    data_source: str,
    extra_info: Any,
    score_fn: Callable,
    prompt_str: str,
    do_print: bool
) -> Dict[str, Any]:
    """
    单个样本的评分逻辑，将在 Ray Worker 中运行。
    """
    start_time = time.time()
    try:
        # 执行具体的评分函数
        # 注意：这里假设 score_fn 是纯函数，或者已经包含在闭包中
        result = score_fn(
            data_source,
            solution_str,
            ground_truth,
            extra_info
        )
        
        # 适配不同的返回格式 (Dict 或 Scalar)
        if isinstance(result, dict):
            score = result.get('score', 0.0)
            acc = result.get('acc', 0.0)
            # 提取除 score 以外的所有 info
            info = {k: v for k, v in result.items() if k != 'score'}
        else:
            score = float(result)
            acc = score  # 传统格式假设 score 即 acc
            info = {}

    except Exception as e:
        # 捕获所有异常，防止 Worker 崩溃导致整个 Batch 失败
        # print(f"[Ray Worker Error] Sample {index} failed: {e}")
        score = 0.0
        acc = 0.0
        info = {}
    
    # 构建打印日志（如果需要）
    log_message = ""
    if do_print:
        # 仅在需要打印时才构建这个字符串，节省内存
        log_message = (
            f"<<<<<<<<<<  Prompt-{index}  >>>>>>>>>>:\n{prompt_str}\n"
            f"<<<<<<<<<< Response-{index} >>>>>>>>>>:\n{solution_str}\n"
            f"<<<<<<<<<< Evaluate-{index} >>>>>>>>>>:\n"
            f"GT: {ground_truth} | Verifier: {score} | Data Source: {data_source}"
        )

    return {
        "index": index,
        "score": score,
        "acc": acc,
        "extra_info": info,
        "log_message": log_message,
        "data_source": data_source
    }


# =============================================================================
# Ray-based Reward Manager
# =============================================================================

@register("prime")
class PrimeRewardManager:
    """
    The Reward Manager used in https://github.com/PRIME-RL/PRIME
    Re-implemented using Ray for high-concurrency stability.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Callable] = None,
        reward_fn_key: str = "data_source",
        num_cpus_per_task: float = 4.0, # 默认每个任务占1个核 (144核机器建议设为 1.0)
        timeout: int = 300,             # 全局超时时间
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.num_cpus_per_task = num_cpus_per_task
        self.timeout = timeout

        # 自动初始化 Ray，避免用户忘记调用
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True) 
            except Exception:
                # 在某些集群环境可能无法自动 init，这通常没问题
                pass

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        Execute reward scoring using Ray for parallelism.
        """
        # 1. 如果已有 rm_scores，直接返回
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        # 2. 数据准备 (主线程解码，效率最高)
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        
        # 批量解码字符串
        prompts_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in data]
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_infos = data.non_tensor_batch.get("extra_info", [None] * len(data))

        # 3. 确定哪些样本需要打印日志 (控制台输出控制)
        print_counts = {}
        sample_do_print = []
        for ds in data_sources:
            curr = print_counts.get(ds, 0)
            if curr < self.num_examine:
                sample_do_print.append(True)
                print_counts[ds] = curr + 1
            else:
                sample_do_print.append(False)

        # 4. 分发任务到 Ray
        # 将评分函数放入 Object Store，避免重复序列化传输
        score_fn_ref = ray.put(self.compute_score)
        
        futures = []
        for i in range(len(data)):
            future = _ray_compute_single_score.options(
                num_cpus=self.num_cpus_per_task 
            ).remote(
                index=i,
                solution_str=responses_str[i],
                ground_truth=ground_truths[i],
                data_source=data_sources[i],
                extra_info=extra_infos[i],
                score_fn=score_fn_ref,
                prompt_str=prompts_str[i],
                do_print=sample_do_print[i]
            )
            futures.append(future)

        # 5. 等待结果 (带超时保护)
        try:
            results = ray.get(futures, timeout=self.timeout)
        except ray.exceptions.GetTimeoutError:
            print(f"[PrimeRewardManager] Timeout! Batch execution exceeded {self.timeout}s.")
            # 紧急回收：获取已完成的，未完成的设为默认值
            finished, pending = ray.wait(futures, num_returns=len(futures), timeout=0.1)
            
            # 将 ObjectRef 映射回索引很难，所以我们用 map 收集已完成的
            finished_results = ray.get(finished)
            results_map = {res['index']: res for res in finished_results}
            
            results = []
            for i in range(len(data)):
                if i in results_map:
                    results.append(results_map[i])
                else:
                    # 处理超时样本
                    print(f"[Timeout] Sample {i} dropped due to timeout.")
                    results.append({
                        "index": i, "score": 0.0, "acc": 0.0, 
                        "extra_info": {}, "log_message": "", "data_source": data_sources[i]
                    })

        # 6. 组装结果
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        acc_list = []
        score_list = []
        
        # 动态收集所有可能的 extra keys
        all_extra_keys = set()
        for res in results:
            if isinstance(res['extra_info'], dict):
                all_extra_keys.update(res['extra_info'].keys())
        
        reward_extra_info = {key: [] for key in all_extra_keys}

        for i, res in enumerate(results):
            score = res['score']
            
            # 填充 Tensor (注意 valid_response_length 是 tensor)
            idx = valid_response_length[i].item() - 1
            if idx >= 0:
                reward_tensor[i, idx] = score
            
            score_list.append(score)
            acc_list.append(res['acc'])
            
            # 填充 extra info
            for key in all_extra_keys:
                reward_extra_info[key].append(res['extra_info'].get(key, 0.0))
            
            # 打印日志
            if res['log_message']:
                print(res['log_message'])

        # 设置 batch acc
        data.batch["acc"] = torch.tensor(acc_list, dtype=torch.float32, device=prompt_ids.device)

        if return_dict:
            result_dict = {"reward_tensor": reward_tensor}
            if reward_extra_info:
                result_dict["reward_extra_info"] = reward_extra_info
            return score_list, result_dict
        else:
            return score_list, reward_tensor
