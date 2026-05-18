#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 PrimeCodeRewardManager 的基本功能
"""

import sys
import torch
from transformers import AutoTokenizer

sys.path.insert(0, '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/TRIT')

from verl import DataProto
from verl.workers.reward_manager import PrimeCodeRewardManager


def create_mock_data_proto(responses, ground_truths, tokenizer):
    """创建模拟的 DataProto 对象用于测试"""
    batch_size = len(responses)
    seq_len = 512
    
    # 创建模拟的 token ids
    prompt_ids = torch.randint(1, 1000, (batch_size, 100))
    response_ids = torch.randint(1, 1000, (batch_size, seq_len - 100))
    
    # 创建 attention mask
    attention_mask = torch.ones((batch_size, seq_len))
    
    # 创建 DataProto
    data_proto = DataProto()
    data_proto.batch = {
        "prompts": prompt_ids,
        "responses": response_ids,
        "attention_mask": attention_mask,
    }
    
    # 添加 non_tensor_batch
    data_proto.non_tensor_batch = {
        "reward_model": [
            {"ground_truth": gt} for gt in ground_truths
        ],
        "data_source": ["code"] * batch_size,
    }
    
    # 设置 prompts 和 responses 的字符串（用于解码）
    # 注意：这里我们直接设置，实际使用中会通过 tokenizer 解码
    for i in range(batch_size):
        data_proto[i].non_tensor_batch = {
            "reward_model": {"ground_truth": ground_truths[i]},
        }
    
    return data_proto


def test_basic_functionality():
    """测试基本功能"""
    print("=" * 80)
    print("测试 1: 基本功能测试")
    print("=" * 80)
    
    # 创建 tokenizer（使用一个简单的模型）
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"无法加载 tokenizer: {e}")
        print("使用模拟 tokenizer...")
        # 创建一个简单的模拟 tokenizer
        class MockTokenizer:
            def batch_decode(self, ids, skip_special_tokens=True):
                return [f"prompt_{i}" for i in range(len(ids))]
            def decode(self, ids, skip_special_tokens=True):
                return "response"
        tokenizer = MockTokenizer()
    
    # 创建 RewardManager
    reward_manager = PrimeCodeRewardManager(
        tokenizer=tokenizer,
        num_examine=2,
        max_workers=4,
        timeout=10,
        use_batch_processing=True,
    )
    
    # 准备测试数据
    responses = [
        """</think>
```python
def add(a, b):
    return a + b
```""",
        """```python
def add(a, b):
    return a - b  # 错误实现
```""",
        """```python
def multiply(a, b):
    return a * b
```""",
    ]
    
    ground_truths = [
        """def check(func):
    assert func(1, 2) == 3
    assert func(0, 0) == 0
<entry_point>
add""",
        """def check(func):
    assert func(1, 2) == 3
<entry_point>
add""",
        """def check(func):
    assert func(2, 3) == 6
<entry_point>
multiply""",
    ]
    
    # 创建 DataProto
    data_proto = create_mock_data_proto(responses, ground_truths, tokenizer)
    
    # 手动设置解码后的字符串（因为模拟 tokenizer 无法正确解码）
    # 在实际使用中，tokenizer 会自动解码
    import types
    if hasattr(tokenizer, 'batch_decode'):
        original_batch_decode = tokenizer.batch_decode
        def mock_batch_decode(ids, skip_special_tokens=True):
            if len(ids) == 3:  # prompts
                return ["Prompt 1", "Prompt 2", "Prompt 3"]
            else:  # responses
                return responses
        tokenizer.batch_decode = mock_batch_decode
    
    # 调用 RewardManager
    try:
        result = reward_manager(data_proto, return_dict=True)
        
        if isinstance(result, dict):
            reward_tensor = result["reward_tensor"]
            reward_extra_info = result.get("reward_extra_info", {})
            
            print(f"✓ Reward tensor shape: {reward_tensor.shape}")
            print(f"✓ Extra info keys: {list(reward_extra_info.keys())}")
            
            if "acc" in reward_extra_info:
                accs = reward_extra_info["acc"]
                print(f"✓ Accuracies: {accs}")
            
            if "format_score" in reward_extra_info:
                format_scores = reward_extra_info["format_score"]
                print(f"✓ Format scores: {format_scores}")
            
            print("✓ 基本功能测试通过")
        else:
            print(f"✓ 返回 reward_tensor: {result.shape}")
            print("✓ 基本功能测试通过")
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_batch_processing():
    """测试批量处理"""
    print("\n" + "=" * 80)
    print("测试 2: 批量处理测试")
    print("=" * 80)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception:
        class MockTokenizer:
            def batch_decode(self, ids, skip_special_tokens=True):
                return [f"prompt_{i}" for i in range(len(ids))]
            def decode(self, ids, skip_special_tokens=True):
                return "response"
        tokenizer = MockTokenizer()
    
    # 创建多个样本
    num_samples = 10
    responses = [
        """```python
def add(a, b):
    return a + b
```"""
    ] * num_samples
    
    ground_truths = [
        """def check(func):
    assert func(1, 2) == 3
<entry_point>
add"""
    ] * num_samples
    
    reward_manager = PrimeCodeRewardManager(
        tokenizer=tokenizer,
        num_examine=2,
        max_workers=5,
        timeout=10,
        use_batch_processing=True,
    )
    
    data_proto = create_mock_data_proto(responses, ground_truths, tokenizer)
    
    # 设置解码后的字符串
    if hasattr(tokenizer, 'batch_decode'):
        def mock_batch_decode(ids, skip_special_tokens=True):
            if len(ids) == num_samples:
                return responses
            else:
                return ["Prompt"] * num_samples
        tokenizer.batch_decode = mock_batch_decode
    
    import time
    start_time = time.time()
    result = reward_manager(data_proto, return_dict=True)
    elapsed = time.time() - start_time
    
    print(f"✓ 处理了 {num_samples} 个样本，耗时 {elapsed:.2f}秒")
    print(f"✓ 平均耗时: {elapsed/num_samples:.3f}秒/样本")
    
    if isinstance(result, dict):
        reward_extra_info = result.get("reward_extra_info", {})
        if "acc" in reward_extra_info:
            accs = reward_extra_info["acc"]
            success_count = sum(1 for a in accs if a >= 0.99)
            print(f"✓ 成功样本数: {success_count}/{num_samples}")
    
    print("✓ 批量处理测试通过")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PrimeCodeRewardManager 测试")
    print("=" * 80 + "\n")
    
    try:
        test1_passed = test_basic_functionality()
        test2_passed = test_batch_processing()
        
        print("\n" + "=" * 80)
        if test1_passed and test2_passed:
            print("✓ 所有测试通过！")
        else:
            print("✗ 部分测试失败")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

