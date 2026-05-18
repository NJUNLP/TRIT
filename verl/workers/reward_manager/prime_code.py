# Copyright 2024 PRIME team and/or its affiliates
# Licensed under the Apache License, Version 2.0

"""
PrimeCodeRewardManager - 专门用于代码验证的奖励管理器

特点：
1. 不使用 Ray，在主进程中直接处理，避免超时问题
2. 直接调用 code_verification 模块的函数，不经过 default_compute_score
3. 使用批量处理接口控制并发，提高效率
4. 支持代码验证的特殊格式（format_score, acc, score）
"""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score.code_verification import (
    compute_score,
    compute_code_score_batch,
    CodeVerificationRewardFunction,
)
from verl.workers.reward_manager import register

logger = logging.getLogger(__name__)

# 注意：code_verification 模块的日志级别会在 PrimeCodeRewardManager.__init__ 中设置


@register("prime_code")
class PrimeCodeRewardManager:
    """
    专门用于代码验证任务的奖励管理器。
    
    与 PrimeRewardManager 的区别：
    - 不使用 Ray，在主进程中直接处理，避免超时问题
    - 直接调用 code_verification 模块，不经过 default_compute_score
    - 使用批量处理接口控制并发，提高效率
    - 支持代码验证的特殊返回格式（format_score, acc, score）
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        num_examine: int,
        compute_score: Optional[Any] = None,  # 不使用，保留接口兼容性
        reward_fn_key: str = "data_source",
        max_workers: Optional[int] = None,  # ThreadPoolExecutor 的最大worker数
        timeout: int = 10,  # 单个代码执行的超时时间（秒）
        use_batch_processing: bool = True,  # 是否使用批量处理
        batch_size: Optional[int] = None,  # 批量处理的大小，None表示处理整个batch
        verbose_logging: bool = False,  # 是否显示详细的警告日志（默认False以减少日志噪音）
        **kwargs
    ) -> None:
        """
        初始化 PrimeCodeRewardManager。

        Args:
            tokenizer: Tokenizer用于解码
            num_examine: 每个数据源打印的样本数量
            compute_score: 不使用，保留接口兼容性
            reward_fn_key: 用于标识数据源的key
            max_workers: ThreadPoolExecutor的最大worker数，None表示自动选择
            timeout: 单个代码执行的超时时间（秒）
            use_batch_processing: 是否使用批量处理接口
            batch_size: 批量处理的大小，None表示处理整个batch
            **kwargs: 其他参数（传递给 code_verification 函数）
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.max_workers = max_workers
        self.timeout = timeout
        self.use_batch_processing = use_batch_processing
        self.batch_size = batch_size
        self.verbose_logging = verbose_logging
        
        # code_verification 的额外参数
        self.code_verification_kwargs = kwargs
        
        # 设置 code_verification 模块的日志级别
        # 默认只显示 ERROR，减少批量处理时的警告噪音
        code_verification_logger = logging.getLogger("verl.utils.reward_score.code_verification")
        if verbose_logging:
            code_verification_logger.setLevel(logging.WARNING)  # 显示 WARNING 和 ERROR
        else:
            code_verification_logger.setLevel(logging.ERROR)  # 只显示 ERROR
        
        # 创建奖励函数实例（可选，用于批量处理）
        if use_batch_processing:
            self.reward_function = CodeVerificationRewardFunction(
                timeout=timeout,
                max_workers=max_workers,
                **kwargs
            )
        else:
            self.reward_function = None

    def __call__(self, data: DataProto, return_dict: bool = False):
        """
        计算奖励分数。

        Args:
            data: DataProto包含批次数据
            return_dict: 如果为True，返回包含reward_tensor和reward_extra_info的字典

        Returns:
            reward_tensor 或包含 reward_tensor 和 reward_extra_info 的字典
        """
        # 如果已有 rm_scores，直接返回（并行生成时可能已计算）
        if "rm_scores" in data.batch.keys():
            reward_tensor = data.batch["rm_scores"]
            
            # 尝试收集额外的信息（format_score, acc等）
            reward_extra_info = {}
            if "acc" in data.batch:
                reward_extra_info["acc"] = data.batch["acc"].tolist()
            if "format_score" in data.non_tensor_batch:
                reward_extra_info["format_score"] = data.non_tensor_batch["format_score"]
            
            # 计算 score_list（从 reward_tensor 中提取）
            score_list = reward_tensor.sum(dim=-1).cpu().tolist()
            
            # 返回格式需要与 PrimeRewardManager 保持一致
            if return_dict:
                result_dict = {"reward_tensor": reward_tensor}
                if reward_extra_info:
                    result_dict["reward_extra_info"] = reward_extra_info
                return score_list, result_dict
            else:
                return score_list, reward_tensor

        # 准备数据
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        
        # 批量解码字符串
        prompts_str = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        # 获取 ground_truth 和 data_source
        ground_truths = [item.non_tensor_batch['reward_model']['ground_truth'] for item in data]
        data_sources = data.non_tensor_batch.get(self.reward_fn_key, ["code"] * len(data))
        extra_infos = data.non_tensor_batch.get("extra_info", [None] * len(data))

        # 确定哪些样本需要打印日志
        print_counts = {}
        sample_do_print = []
        for ds in data_sources:
            curr = print_counts.get(ds, 0)
            if curr < self.num_examine:
                sample_do_print.append(True)
                print_counts[ds] = curr + 1
            else:
                sample_do_print.append(False)

        # 使用批量处理或逐个处理
        if self.use_batch_processing:
            results = self._process_batch(
                responses_str, ground_truths, prompts_str, sample_do_print
            )
        else:
            results = self._process_individual(
                responses_str, ground_truths, prompts_str, sample_do_print
            )

        # 组装结果
        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32)
        acc_list = []
        score_list = []
        format_score_list = []
        
        reward_extra_info = defaultdict(list)
        
        # 统计信息
        stats = {
            "total": len(results),
            "success": 0,
            "no_code": 0,
            "syntax_error": 0,
            "timeout": 0,
            "execution_error": 0,
        }

        for i, result in enumerate(results):
            # 处理返回结果（可能是字典或标量）
            if isinstance(result, dict):
                score = result.get('score', 0.0)
                acc = result.get('acc', 0.0)
                format_score = result.get('format_score', 0.0)
                
                # 收集所有额外的信息
                for key, value in result.items():
                    if key not in ['score', 'acc', 'format_score']:
                        reward_extra_info[key].append(value)
            else:
                # 标量结果（向后兼容）
                score = float(result)
                acc = score
                format_score = 0.0
            
            # 填充 reward_tensor
            idx = valid_response_length[i].item() - 1
            if idx >= 0:
                reward_tensor[i, idx] = score
            
            score_list.append(score)
            acc_list.append(acc)
            format_score_list.append(format_score)
            
            # 填充 extra_info
            reward_extra_info['acc'].append(acc)
            reward_extra_info['format_score'].append(format_score)
            
            # 更新统计信息
            if acc >= 0.99:
                stats["success"] += 1
            elif score == 0.0:
                # 尝试推断失败原因（基于分数和格式）
                if format_score == 0:
                    stats["no_code"] += 1
                else:
                    stats["execution_error"] += 1
            
            # 打印日志（只对需要打印的样本）
            if sample_do_print[i] and isinstance(result, dict):
                print(
                    f"<<<<<<<<<<  Prompt-{i}  >>>>>>>>>>:\n{prompts_str[i]}\n"
                    f"<<<<<<<<<< Response-{i} >>>>>>>>>>:\n{responses_str[i]}\n"
                    f"<<<<<<<<<< Evaluate-{i} >>>>>>>>>>:\n"
                    f"GT: {ground_truths[i][:100]}... | Score: {score} | Acc: {acc} | Format: {format_score}"
                )
        
        # 打印批量统计信息（每批次只打印一次）
        if stats["total"] > 0:
            success_rate = stats["success"] / stats["total"] * 100
            logger.info(
                f"[PrimeCodeRewardManager] Batch stats: "
                f"total={stats['total']}, success={stats['success']} ({success_rate:.1f}%), "
                f"no_code={stats['no_code']}, syntax_error={stats['syntax_error']}, "
                f"timeout={stats['timeout']}, execution_error={stats['execution_error']}"
            )

        # 设置 batch acc
        data.batch["acc"] = torch.tensor(acc_list, dtype=torch.float32, device=prompt_ids.device)
        
        # 设置 format_score（如果存在）
        # 注意：non_tensor_batch 中的值必须是 np.ndarray 类型
        if format_score_list:
            data.non_tensor_batch["format_score"] = np.array(format_score_list, dtype=np.float32)

        # 返回格式需要与 PrimeRewardManager 保持一致
        # return_dict=True: 返回 (score_list, result_dict)
        # return_dict=False: 返回 (score_list, reward_tensor)
        if return_dict:
            result_dict = {"reward_tensor": reward_tensor}
            if reward_extra_info:
                result_dict["reward_extra_info"] = dict(reward_extra_info)
            return score_list, result_dict
        else:
            return score_list, reward_tensor

    def _process_batch(
        self,
        responses_str: List[str],
        ground_truths: List[str],
        prompts_str: List[str],
        sample_do_print: List[bool],
    ) -> List[Dict[str, Any]]:
        """
        使用批量处理接口处理所有样本。

        Args:
            responses_str: 响应字符串列表
            ground_truths: 标准答案列表
            prompts_str: 提示字符串列表（用于日志）
            sample_do_print: 是否打印每个样本的日志

        Returns:
            结果列表，每个元素是字典
        """
        try:
            # 使用批量处理接口获取基础分数
            if self.reward_function:
                # 使用 CodeVerificationRewardFunction 类
                scores = self.reward_function.compute_batch(responses_str, ground_truths)
            else:
                # 使用 compute_code_score_batch 函数
                scores = compute_code_score_batch(
                    responses_str,
                    ground_truths,
                    timeout=self.timeout,
                    max_workers=self.max_workers,
                    **self.code_verification_kwargs
                )
            
            # 批量处理返回的是分数列表，我们需要转换为字典格式
            # 为了获取 format_score 和 acc，我们需要调用 compute_score
            # 但为了效率，我们只对需要详细信息的样本调用（需要打印日志的）
            
            results = []
            need_details_indices = [i for i, do_print in enumerate(sample_do_print) if do_print]
            
            # 批量获取需要详细信息的样本的结果
            if need_details_indices:
                detailed_responses = [responses_str[i] for i in need_details_indices]
                detailed_ground_truths = [ground_truths[i] for i in need_details_indices]
                
                # 对需要详细信息的样本调用 compute_score
                detailed_results = []
                for response, ground_truth in zip(detailed_responses, detailed_ground_truths):
                    try:
                        result = compute_score(response, ground_truth, **self.code_verification_kwargs)
                        detailed_results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to get detailed result: {e}")
                        idx = detailed_responses.index(response)
                        score = scores[need_details_indices[idx]]
                        detailed_results.append({"score": score, "acc": score, "format_score": 0.0})
                
                # 创建详细结果的映射
                detailed_map = {need_details_indices[i]: detailed_results[i] for i in range(len(need_details_indices))}
            else:
                detailed_map = {}
            
            # 组装最终结果
            for i, score in enumerate(scores):
                if i in detailed_map:
                    # 使用详细结果
                    results.append(detailed_map[i])
                else:
                    # 使用基础分数，推断 format_score 和 acc
                    # format_score 检查：是否存在 </think> 标记（轻量级检查）
                    has_think = "</think>" in responses_str[i]
                    format_score = 1 if has_think else 0
                    
                    # acc 基于分数推断（score >= 0.99 表示代码执行通过）
                    acc = 1.0 if score >= 0.99 else 0.0
                    
                    # 最终 score：只有当 format_score 和 acc 都为 1 时才为 1
                    final_score = 1.0 if (format_score == 1 and acc == 1) else score
                    
                    results.append({
                        "score": final_score,
                        "acc": acc,
                        "format_score": format_score
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}, falling back to individual processing")
            # 降级为逐个处理
            return self._process_individual(responses_str, ground_truths, prompts_str, sample_do_print)

    def _process_individual(
        self,
        responses_str: List[str],
        ground_truths: List[str],
        prompts_str: List[str],
        sample_do_print: List[bool],
    ) -> List[Dict[str, Any]]:
        """
        逐个处理样本（降级方案）。

        Args:
            responses_str: 响应字符串列表
            ground_truths: 标准答案列表
            prompts_str: 提示字符串列表（用于日志）
            sample_do_print: 是否打印每个样本的日志

        Returns:
            结果列表，每个元素是字典
        """
        results = []
        for i, (response, ground_truth) in enumerate(zip(responses_str, ground_truths)):
            try:
                result = compute_score(response, ground_truth, **self.code_verification_kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process sample {i}: {e}")
                # 返回默认失败结果
                results.append({"score": 0.0, "acc": 0.0, "format_score": 0.0})
        
        return results

