#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
代码验证奖励函数的全面测试脚本

测试内容：
1. 基本功能测试
2. 边界情况测试
3. 高并发场景测试
4. 线程安全性测试
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# 添加路径以便导入模块
sys.path.insert(0, '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-hldy-nlp/FMG/liujunxiao03/MeiTuan/TRIT')

from verl.utils.reward_score.code_verification import (
    compute_code_score,
    compute_code_score_with_details,
    compute_code_score_batch,
    compute_score,
    CodeVerificationRewardFunction,
    CodeVerificationResult,
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResults:
    """测试结果统计"""
    def __init__(self):
        self.total = 0
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def add_result(self, test_name: str, passed: bool, error: str = ""):
        self.total += 1
        if passed:
            self.passed += 1
            logger.info(f"✓ {test_name}: PASSED")
        else:
            self.failed += 1
            self.errors.append((test_name, error))
            logger.error(f"✗ {test_name}: FAILED - {error}")
    
    def print_summary(self):
        print("\n" + "=" * 80)
        print("测试结果汇总")
        print("=" * 80)
        print(f"总测试数: {self.total}")
        print(f"通过: {self.passed}")
        print(f"失败: {self.failed}")
        print(f"通过率: {self.passed / self.total * 100:.2f}%")
        
        if self.errors:
            print("\n失败的测试:")
            for test_name, error in self.errors:
                print(f"  - {test_name}: {error}")
        print("=" * 80)


def test_basic_functionality(results: TestResults):
    """测试基本功能"""
    print("\n" + "=" * 80)
    print("测试 1: 基本功能测试")
    print("=" * 80)
    
    # 测试 1.1: 正确的代码解决方案
    solution_correct = """
```python
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
```
"""
    tests_correct = """
assert add(1, 2) == 3
assert add(0, 0) == 0
assert multiply(2, 3) == 6
assert multiply(0, 5) == 0
"""
    try:
        result = compute_code_score_with_details(solution_correct, tests_correct)
        passed = result.success and result.score == 1.0
        results.add_result("1.1 正确的代码解决方案", passed, 
                          f"expected success=True, score=1.0, got success={result.success}, score={result.score}")
    except Exception as e:
        results.add_result("1.1 正确的代码解决方案", False, str(e))
    
    # 测试 1.2: 错误的代码解决方案（逻辑错误）
    solution_wrong = """
```python
def add(a, b):
    return a - b  # 错误：应该是 a + b
```
"""
    tests_wrong = "assert add(1, 2) == 3"
    try:
        result = compute_code_score_with_details(solution_wrong, tests_wrong)
        passed = not result.success and result.score == 0.0
        results.add_result("1.2 错误的代码解决方案", passed,
                          f"expected success=False, score=0.0, got success={result.success}, score={result.score}")
    except Exception as e:
        results.add_result("1.2 错误的代码解决方案", False, str(e))
    
    # 测试 1.3: 语法错误
    solution_syntax_error = """
```python
def add(a, b)
    return a + b  # 缺少冒号
```
"""
    tests_syntax = "assert add(1, 2) == 3"
    try:
        result = compute_code_score_with_details(solution_syntax_error, tests_syntax, enable_syntax_check=True)
        passed = not result.success and result.stage == "syntax"
        results.add_result("1.3 语法错误检测", passed,
                          f"expected stage='syntax', got stage={result.stage}")
    except Exception as e:
        results.add_result("1.3 语法错误检测", False, str(e))
    
    # 测试 1.4: 没有代码的情况
    solution_no_code = "这是一个纯文本回答，没有包含任何代码。"
    tests_no_code = "assert True"
    try:
        result = compute_code_score_with_details(solution_no_code, tests_no_code)
        passed = not result.success and result.stage == "extraction"
        results.add_result("1.4 无代码提取", passed,
                          f"expected stage='extraction', got stage={result.stage}")
    except Exception as e:
        results.add_result("1.4 无代码提取", False, str(e))
    
    # 测试 1.5: 运行时错误
    solution_runtime_error = """
```python
def divide(a, b):
    return a / b
```
"""
    tests_runtime = """
assert divide(10, 2) == 5
assert divide(10, 0) == 0  # 会导致 ZeroDivisionError
"""
    try:
        result = compute_code_score_with_details(solution_runtime_error, tests_runtime)
        passed = not result.success
        results.add_result("1.5 运行时错误处理", passed,
                          f"expected success=False, got success={result.success}")
    except Exception as e:
        results.add_result("1.5 运行时错误处理", False, str(e))


def test_edge_cases(results: TestResults):
    """测试边界情况"""
    print("\n" + "=" * 80)
    print("测试 2: 边界情况测试")
    print("=" * 80)
    
    # 测试 2.1: 空字符串
    try:
        result = compute_code_score_with_details("", "assert True")
        passed = not result.success and result.stage == "extraction"
        results.add_result("2.1 空字符串输入", passed,
                          f"expected stage='extraction', got stage={result.stage}")
    except Exception as e:
        results.add_result("2.1 空字符串输入", False, str(e))
    
    # 测试 2.2: 特殊字符和Unicode
    solution_unicode = """
```python
def greet(name):
    return f"你好, {name}!"
```
"""
    tests_unicode = """
assert greet("世界") == "你好, 世界!"
"""
    try:
        result = compute_code_score_with_details(solution_unicode, tests_unicode)
        passed = result.success
        results.add_result("2.2 Unicode字符支持", passed,
                          f"expected success=True, got success={result.success}")
    except Exception as e:
        results.add_result("2.2 Unicode字符支持", False, str(e))
    
    # 测试 2.3: 多个代码块（使用 all_blocks 方法）
    solution_multiple_blocks = """
```python
import math
```

```python
def calculate_area(radius):
    return math.pi * radius ** 2
```
"""
    tests_multiple = "assert abs(calculate_area(1) - 3.14159) < 0.01"
    try:
        result = compute_code_score_with_details(
            solution_multiple_blocks, 
            tests_multiple,
            extraction_method="all_blocks"
        )
        passed = result.success
        results.add_result("2.3 多个代码块提取", passed,
                          f"expected success=True, got success={result.success}")
    except Exception as e:
        results.add_result("2.3 多个代码块提取", False, str(e))
    
    # 测试 2.4: 自定义评分
    solution_custom = """
```python
def add(a, b):
    return a - b  # 错误实现
```
"""
    tests_custom = "assert add(1, 2) == 3"
    try:
        result = compute_code_score_with_details(
            solution_custom,
            tests_custom,
            correct_score=1.0,
            test_failure_score=0.5  # 部分分数
        )
        passed = result.score == 0.5
        results.add_result("2.4 自定义评分", passed,
                          f"expected score=0.5, got score={result.score}")
    except Exception as e:
        results.add_result("2.4 自定义评分", False, str(e))


def test_compute_score_function(results: TestResults):
    """测试 compute_score 函数（GSM8K风格接口）"""
    print("\n" + "=" * 80)
    print("测试 3: compute_score 函数测试")
    print("=" * 80)
    
    # 测试 3.1: 包含 </think> 标记的正确解决方案
    solution_with_think = """
</think>
我需要实现一个加法函数。
```python
def add(a, b):
    return a + b
```
"""
    ground_truth_with_entry = """
def check(func):
    assert func(1, 2) == 3
    assert func(0, 0) == 0
<entry_point>
add
"""
    try:
        result = compute_score(solution_with_think, ground_truth_with_entry)
        passed = result["score"] == 1 and result["acc"] == 1 and result["format_score"] == 1
        results.add_result("3.1 包含思考标记的正确解决方案", passed,
                          f"expected score=1, acc=1, format_score=1, got {result}")
    except Exception as e:
        results.add_result("3.1 包含思考标记的正确解决方案", False, str(e))
    
    # 测试 3.2: 不包含 </think> 标记
    solution_no_think = """
```python
def add(a, b):
    return a + b
```
"""
    try:
        result = compute_score(solution_no_think, ground_truth_with_entry)
        passed = result["format_score"] == 0
        results.add_result("3.2 不包含思考标记", passed,
                          f"expected format_score=0, got format_score={result['format_score']}")
    except Exception as e:
        results.add_result("3.2 不包含思考标记", False, str(e))


def test_batch_processing(results: TestResults):
    """测试批量处理功能"""
    print("\n" + "=" * 80)
    print("测试 4: 批量处理测试")
    print("=" * 80)
    
    # 准备测试数据
    solutions = [
        "```python\ndef add(a, b):\n    return a + b\n```",
        "```python\ndef add(a, b):\n    return a - b\n```",  # 错误
        "```python\ndef multiply(a, b):\n    return a * b\n```",
        "def divide(a, b):\n    return a / b",  # 无markdown
        "没有代码的纯文本",  # 无代码
    ]
    
    tests = [
        "assert add(1, 2) == 3",
        "assert add(1, 2) == 3",
        "assert multiply(2, 3) == 6",
        "assert divide(10, 2) == 5",
        "assert True",
    ]
    
    try:
        scores = compute_code_score_batch(solutions, tests, max_workers=5)
        expected_scores = [1.0, 0.0, 1.0, 1.0, 0.0]
        passed = len(scores) == len(expected_scores) and all(
            abs(s - e) < 0.01 for s, e in zip(scores, expected_scores)
        )
        results.add_result("4.1 批量处理基本功能", passed,
                          f"expected {expected_scores}, got {scores}")
    except Exception as e:
        results.add_result("4.1 批量处理基本功能", False, str(e))
    
    # 测试 4.2: 单个测试应用于所有解决方案
    try:
        single_test = "assert add(1, 2) == 3"
        solutions_single = [
            "```python\ndef add(a, b):\n    return a + b\n```",
            "```python\ndef add(a, b):\n    return a + b\n```",
        ]
        scores = compute_code_score_batch(solutions_single, single_test, max_workers=2)
        passed = len(scores) == 2 and all(s == 1.0 for s in scores)
        results.add_result("4.2 单个测试批量应用", passed,
                          f"expected [1.0, 1.0], got {scores}")
    except Exception as e:
        results.add_result("4.2 单个测试批量应用", False, str(e))


def test_concurrent_execution(results: TestResults):
    """测试高并发场景"""
    print("\n" + "=" * 80)
    print("测试 5: 高并发场景测试")
    print("=" * 80)
    
    # 准备大量测试数据
    num_tasks = 50
    solutions = [
        "```python\ndef add(a, b):\n    return a + b\n```"
    ] * num_tasks
    
    tests = [
        "assert add(1, 2) == 3\nassert add(0, 0) == 0"
    ] * num_tasks
    
    # 测试 5.1: 高并发批量处理
    try:
        start_time = time.time()
        scores = compute_code_score_batch(
            solutions, 
            tests, 
            max_workers=20,
            timeout=10
        )
        elapsed_time = time.time() - start_time
        
        passed = (
            len(scores) == num_tasks and
            all(s == 1.0 for s in scores) and
            elapsed_time < 120  # 应该在2分钟内完成
        )
        results.add_result("5.1 高并发批量处理 (50任务)", passed,
                          f"processed {len(scores)} tasks in {elapsed_time:.2f}s, "
                          f"all scores correct: {all(s == 1.0 for s in scores)}")
    except Exception as e:
        results.add_result("5.1 高并发批量处理 (50任务)", False, str(e))
    
    # 测试 5.2: 使用 CodeVerificationRewardFunction 类
    try:
        reward_func = CodeVerificationRewardFunction(
            timeout=10,
            max_workers=20
        )
        
        start_time = time.time()
        scores = reward_func.compute_batch(solutions[:20], tests[:20])
        elapsed_time = time.time() - start_time
        
        passed = (
            len(scores) == 20 and
            all(s == 1.0 for s in scores)
        )
        results.add_result("5.2 RewardFunction类批量处理", passed,
                          f"processed {len(scores)} tasks in {elapsed_time:.2f}s")
    except Exception as e:
        results.add_result("5.2 RewardFunction类批量处理", False, str(e))
    
    # 测试 5.3: 线程安全性测试（手动并发）
    def worker_task(idx):
        """单个工作线程任务"""
        solution = "```python\ndef add(a, b):\n    return a + b\n```"
        test = f"assert add({idx}, {idx + 1}) == {idx * 2 + 1}"
        try:
            score = compute_code_score(solution, test, timeout=5)
            return idx, score
        except Exception as e:
            logger.error(f"Worker {idx} failed: {e}")
            return idx, 0.0
    
    try:
        num_workers = 30
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_workers)]
            results_list = [future.result() for future in as_completed(futures)]
        
        elapsed_time = time.time() - start_time
        results_list.sort(key=lambda x: x[0])  # 按索引排序
        
        passed = (
            len(results_list) == num_workers and
            all(score == 1.0 for _, score in results_list) and
            elapsed_time < 60
        )
        results.add_result("5.3 手动线程池并发测试 (30线程)", passed,
                          f"processed {len(results_list)} tasks in {elapsed_time:.2f}s, "
                          f"all correct: {all(s == 1.0 for _, s in results_list)}")
    except Exception as e:
        results.add_result("5.3 手动线程池并发测试 (30线程)", False, str(e))
    
    # 测试 5.4: 混合正确和错误的解决方案（压力测试）
    try:
        mixed_solutions = []
        mixed_tests = []
        
        for i in range(20):
            if i % 2 == 0:
                # 正确的解决方案
                mixed_solutions.append("```python\ndef add(a, b):\n    return a + b\n```")
                mixed_tests.append("assert add(1, 2) == 3")
            else:
                # 错误的解决方案
                mixed_solutions.append("```python\ndef add(a, b):\n    return a - b\n```")
                mixed_tests.append("assert add(1, 2) == 3")
        
        start_time = time.time()
        scores = compute_code_score_batch(mixed_solutions, mixed_tests, max_workers=10)
        elapsed_time = time.time() - start_time
        
        expected_scores = [1.0 if i % 2 == 0 else 0.0 for i in range(20)]
        passed = (
            len(scores) == 20 and
            all(abs(s - e) < 0.01 for s, e in zip(scores, expected_scores))
        )
        results.add_result("5.4 混合场景压力测试 (20任务)", passed,
                          f"processed in {elapsed_time:.2f}s, "
                          f"correct pattern: {passed}")
    except Exception as e:
        results.add_result("5.4 混合场景压力测试 (20任务)", False, str(e))


def test_resource_cleanup(results: TestResults):
    """测试资源清理"""
    print("\n" + "=" * 80)
    print("测试 6: 资源清理测试")
    print("=" * 80)
    
    # 测试 6.1: 多次调用后资源是否正常清理
    try:
        solution = "```python\ndef add(a, b):\n    return a + b\n```"
        test = "assert add(1, 2) == 3"
        
        # 连续多次调用
        for i in range(10):
            score = compute_code_score(solution, test, timeout=5)
            if score != 1.0:
                raise ValueError(f"Iteration {i} failed: score={score}")
        
        results.add_result("6.1 连续多次调用资源清理", True, "")
    except Exception as e:
        results.add_result("6.1 连续多次调用资源清理", False, str(e))
    
    # 测试 6.2: 异常情况下的资源清理
    try:
        # 使用会导致超时的代码（如果支持）
        solution_timeout = "```python\nimport time\ntime.sleep(100)\n```"
        test_timeout = "assert True"
        
        result = compute_code_score_with_details(
            solution_timeout,
            test_timeout,
            timeout=2  # 短超时
        )
        # 应该超时，但不应该崩溃
        timed_out = result.execution_result.timed_out if result.execution_result else False
        passed = timed_out or not result.success
        results.add_result("6.2 超时情况资源清理", passed,
                          f"timeout handled: timed_out={timed_out}, success={result.success}")
    except Exception as e:
        results.add_result("6.2 超时情况资源清理", False, str(e))


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("代码验证奖励函数 - 全面测试")
    print("=" * 80)
    
    results = TestResults()
    
    try:
        # 运行所有测试
        test_basic_functionality(results)
        test_edge_cases(results)
        test_compute_score_function(results)
        test_batch_processing(results)
        test_concurrent_execution(results)
        test_resource_cleanup(results)
        
    except KeyboardInterrupt:
        logger.warning("测试被用户中断")
    except Exception as e:
        logger.error(f"测试过程中发生未预期的错误: {e}", exc_info=True)
        results.add_result("未预期的错误", False, str(e))
    
    # 打印测试结果汇总
    results.print_summary()
    
    # 返回退出码
    return 0 if results.failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

