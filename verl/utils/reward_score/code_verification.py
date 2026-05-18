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
Code verification and reward scoring for reinforcement learning.

This module provides the main interface for:
1. Extracting code from model responses
2. Executing code in a sandbox environment
3. Running test cases to verify correctness
4. Computing reward scores based on execution results

Usage:
    from verl.utils.reward_score.code_verification import compute_code_score

    solution_str = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
    ground_truth = "assert add(1, 2) == 3\nassert add(0, 0) == 0"

    score = compute_code_score(solution_str, ground_truth)
    print(f"Score: {score}")  # 1.0 if all tests pass, 0.0 otherwise
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Union

from verl.utils.reward_score.code_extractor import extract_and_validate_code
from verl.utils.reward_score.code_sandbox_executor import CodeSandboxExecutor, ExecutionResult

logger = logging.getLogger(__name__)


class CodeVerificationResult:
    """Container for code verification results with detailed information."""

    def __init__(
        self,
        score: float,
        success: bool,
        extracted_code: Optional[str] = None,
        execution_result: Optional[ExecutionResult] = None,
        error_message: str = "",
        stage: str = "",
    ):
        """
        Args:
            score: The reward score (0.0 to 1.0)
            success: Whether verification succeeded
            extracted_code: The extracted code from solution
            execution_result: The ExecutionResult from sandbox
            error_message: Error message if failed
            stage: Stage where failure occurred (extraction, syntax, execution, tests)
        """
        self.score = score
        self.success = success
        self.extracted_code = extracted_code
        self.execution_result = execution_result
        self.error_message = error_message
        self.stage = stage

    def __repr__(self):
        return (
            f"CodeVerificationResult(score={self.score}, success={self.success}, "
            f"stage={self.stage}, error={self.error_message[:50] if self.error_message else 'None'})"
        )

    def to_dict(self):
        """Convert to dictionary for logging/serialization."""
        return {
            "score": self.score,
            "success": self.success,
            "extracted_code": self.extracted_code,
            "error_message": self.error_message,
            "stage": self.stage,
            "execution_success": self.execution_result.success if self.execution_result else None,
            "execution_output": self.execution_result.output if self.execution_result else None,
            "execution_error": self.execution_result.error if self.execution_result else None,
            "timed_out": self.execution_result.timed_out if self.execution_result else None,
        }


def compute_code_score(
    solution_str: str,
    ground_truth: str,
    config: Optional[dict] = None,
    timeout: int = 10,
    extraction_method: str = "markdown",
    correct_score: float = 1.0,
    syntax_error_score: float = 0.0,
    execution_error_score: float = 0.0,
    test_failure_score: float = 0.0,
    no_code_score: float = 0.0,
    enable_syntax_check: bool = True,
) -> float:
    """
    Compute reward score for code generation based on test execution.

    This function:
    1. Extracts Python code from the model's response
    2. Verifies syntax (optional)
    3. Executes the code in a sandbox
    4. Runs test cases (assert statements) to verify correctness
    5. Returns a reward score

    Args:
        solution_str: The model's response text (may contain markdown, explanations, etc.)
        ground_truth: Test code containing assert statements to verify correctness
        config: Optional configuration for sandbox executor
        timeout: Execution timeout in seconds (default: 10)
        extraction_method: Method to extract code ("markdown", "all_blocks", "last_block")
        correct_score: Score when all tests pass (default: 1.0)
        syntax_error_score: Score when code has syntax errors (default: 0.0)
        execution_error_score: Score when code fails to execute (default: 0.0)
        test_failure_score: Score when tests fail (default: 0.0)
        no_code_score: Score when no code is extracted (default: 0.0)
        enable_syntax_check: Whether to check syntax before execution (default: True)

    Returns:
        Reward score (float between 0.0 and 1.0)

    Examples:
        >>> solution = "```python\\ndef add(a, b):\\n    return a + b\\n```"
        >>> tests = "assert add(1, 2) == 3\\nassert add(0, 0) == 0"
        >>> score = compute_code_score(solution, tests)
        >>> print(score)  # 1.0
    """
    result = compute_code_score_with_details(
        solution_str=solution_str,
        ground_truth=ground_truth,
        config=config,
        timeout=timeout,
        extraction_method=extraction_method,
        correct_score=correct_score,
        syntax_error_score=syntax_error_score,
        execution_error_score=execution_error_score,
        test_failure_score=test_failure_score,
        no_code_score=no_code_score,
        enable_syntax_check=enable_syntax_check,
    )
    return result.score


def compute_code_score_with_details(
    solution_str: str,
    ground_truth: str,
    config: Optional[dict] = None,
    timeout: int = 10,
    extraction_method: str = "markdown",
    correct_score: float = 1.0,
    syntax_error_score: float = 0.0,
    execution_error_score: float = 0.0,
    test_failure_score: float = 0.0,
    no_code_score: float = 0.0,
    enable_syntax_check: bool = True,
) -> CodeVerificationResult:
    """
    Compute reward score with detailed results.

    Same as compute_code_score but returns a CodeVerificationResult object
    with detailed information about the verification process.

    Returns:
        CodeVerificationResult object containing score and details
    """
    # Step 1: Extract code from solution
    code, is_valid, error = extract_and_validate_code(solution_str, method=extraction_method)
    # print("--------------------------------")
    # print(f"code: {code}")
    # print("--------------------------------")
    if not is_valid:
        logger.warning(f"Code extraction failed: {error}")
        return CodeVerificationResult(
            score=no_code_score,
            success=False,
            extracted_code=None,
            error_message=error,
            stage="extraction",
        )

    # Step 2: Optional syntax check
    if enable_syntax_check:
        executor = CodeSandboxExecutor(config=config, execution_timeout=timeout)
        syntax_valid, syntax_error = executor.verify_code_syntax(code)

        if not syntax_valid:
            logger.warning(f"Syntax error: {syntax_error}")
            return CodeVerificationResult(
                score=syntax_error_score,
                success=False,
                extracted_code=code,
                error_message=syntax_error,
                stage="syntax",
            )

    # Step 3: Execute code with tests in sandbox
    executor = CodeSandboxExecutor(config=config, execution_timeout=timeout)

    try:
        exec_result = executor.execute_code_with_tests(
            solution_code=code,
            test_code=ground_truth,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"Unexpected error during execution: {e}")
        return CodeVerificationResult(
            score=execution_error_score,
            success=False,
            extracted_code=code,
            error_message=f"Execution exception: {str(e)}",
            stage="execution",
        )
    
    # Step 4: Determine score based on execution result
    if exec_result.timed_out:
        logger.warning("Execution timed out")
        return CodeVerificationResult(
            score=execution_error_score,
            success=False,
            extracted_code=code,
            execution_result=exec_result,
            error_message="Execution timeout",
            stage="execution",
        )

    if not exec_result.success:
        # Execution failed (runtime error or test failure)
        logger.info(f"Execution failed: {exec_result.error}")
        return CodeVerificationResult(
            score=test_failure_score,
            success=False,
            extracted_code=code,
            execution_result=exec_result,
            error_message=exec_result.error,
            stage="tests",
        )

    # Success: all tests passed
    logger.info("All tests passed successfully")
    return CodeVerificationResult(
        score=correct_score,
        success=True,
        extracted_code=code,
        execution_result=exec_result,
        error_message="",
        stage="complete",
    )


# Convenience function with simple interface (matches gsm8k.py style)
def compute_score(solution_str: str, ground_truth: str, **kwargs) -> Dict[str, int]:
    """
    GSM8K 风格的简化接口，返回格式分与正确性分。
 
    返回:
        {
            "score": 1 或 0，仅当 format_score 与 acc 均为 1 时为 1
            "acc":   1 或 0，代码执行并通过测试为 1
            "format_score": 1 或 0，存在 </think> 为 1
        }
    """
    has_think = "</think>" in solution_str
    format_score = 1 if has_think else 0
 
    if has_think:
        solution_body = solution_str.split("</think>")[-1].strip()
    else:
        solution_body = "Empty"
 
    check_function = ground_truth.split("<entry_point>")[0].strip()
    entry_point = ground_truth.split("<entry_point>")[1].strip()
    prepared_ground_truth = check_function + "\n\n" + f"check({entry_point})"
 
    result = compute_code_score_with_details(
        solution_str=solution_body,
        ground_truth=prepared_ground_truth,
        **kwargs,
    )
 
    acc = 1 if result.success else 0
    score = 1 if (format_score == 1 and acc == 1) else 0
    print(f"score: {score}, acc: {acc}, format_score: {format_score}")
    print(f"Solution: {solution_body}   ")
    print(f"Ground Truth: {prepared_ground_truth}")
 
    return {"score": score, "acc": acc, "format_score": format_score}


def compute_code_score_batch(
    solution_strs: List[str],
    ground_truths: Union[str, List[str]],
    config: Optional[dict] = None,
    timeout: int = 10,
    extraction_method: str = "markdown",
    correct_score: float = 1.0,
    syntax_error_score: float = 0.0,
    execution_error_score: float = 0.0,
    test_failure_score: float = 0.0,
    no_code_score: float = 0.0,
    enable_syntax_check: bool = True,
    max_workers: Optional[int] = None,
) -> List[float]:
    """
    Compute reward scores for a batch of code generation tasks concurrently.

    This function is designed for high-throughput scenarios in VeRL framework,
    using ThreadPoolExecutor to process multiple code verification tasks in parallel.
    Each task gets its own CodeSandboxExecutor instance to ensure thread safety.

    Args:
        solution_strs: List of model responses containing code
        ground_truths: Test code(s). Can be:
            - Single string: same tests applied to all solutions
            - List of strings: one test per solution (must match length)
        config: Optional configuration for sandbox executor
        timeout: Execution timeout in seconds per task (default: 10)
        extraction_method: Method to extract code ("markdown", "all_blocks", "last_block")
        correct_score: Score when all tests pass (default: 1.0)
        syntax_error_score: Score when code has syntax errors (default: 0.0)
        execution_error_score: Score when code fails to execute (default: 0.0)
        test_failure_score: Score when tests fail (default: 0.0)
        no_code_score: Score when no code is extracted (default: 0.0)
        enable_syntax_check: Whether to check syntax before execution (default: True)
        max_workers: Maximum number of concurrent threads (default: None = min(32, len(solution_strs)))

    Returns:
        List of reward scores (same length as solution_strs)

    Examples:
        >>> solutions = ["```python\\ndef add(a,b): return a+b\\n```"] * 20
        >>> tests = "assert add(1,2)==3"
        >>> scores = compute_code_score_batch(solutions, tests, max_workers=20)
        >>> print(f"Processed {len(scores)} tasks")
    """
    # Handle ground_truths input
    if isinstance(ground_truths, str):
        # Single test for all solutions
        ground_truths_list = [ground_truths] * len(solution_strs)
    else:
        # Individual tests
        if len(ground_truths) != len(solution_strs):
            raise ValueError(
                f"Length mismatch: {len(solution_strs)} solutions but {len(ground_truths)} ground_truths"
            )
        ground_truths_list = ground_truths

    # Initialize results array
    scores = [0.0] * len(solution_strs)

    # Define worker function
    def process_single(idx: int, solution: str, ground_truth: str) -> tuple[int, float]:
        """Process a single solution and return (index, score)."""
        try:
            score = compute_code_score(
                solution_str=solution,
                ground_truth=ground_truth,
                config=config,
                timeout=timeout,
                extraction_method=extraction_method,
                correct_score=correct_score,
                syntax_error_score=syntax_error_score,
                execution_error_score=execution_error_score,
                test_failure_score=test_failure_score,
                no_code_score=no_code_score,
                enable_syntax_check=enable_syntax_check,
            )
            return idx, score
        except Exception as e:
            logger.error(f"Error processing task {idx}: {e}", exc_info=True)
            return idx, 0.0

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(process_single, idx, sol, gt): idx
            for idx, (sol, gt) in enumerate(zip(solution_strs, ground_truths_list))
        }

        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                idx, score = future.result()
                scores[idx] = score
            except Exception as e:
                idx = future_to_idx[future]
                logger.error(f"Unexpected error in task {idx}: {e}", exc_info=True)
                scores[idx] = 0.0

    return scores


class CodeVerificationRewardFunction:
    """
    Stateful reward function class for VeRL integration.

    This class provides a callable interface that can be used directly
    as a reward function in VeRL training loops. It maintains configuration
    and provides both single and batch processing methods.

    Thread-safe design: Each call creates its own executor instances.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        timeout: int = 10,
        extraction_method: str = "markdown",
        correct_score: float = 1.0,
        error_score: float = 0.0,
        enable_syntax_check: bool = True,
        max_workers: Optional[int] = None,
    ):
        """
        Initialize reward function with default parameters.

        Args:
            config: Configuration for sandbox executor
            timeout: Execution timeout in seconds
            extraction_method: Code extraction method
            correct_score: Score for correct solutions (default: 1.0)
            error_score: Score for any error condition (default: 0.0)
            enable_syntax_check: Whether to check syntax before execution
            max_workers: Maximum concurrent workers for batch processing
        """
        self.config = config or {}
        self.timeout = 600
        self.extraction_method = extraction_method
        self.correct_score = correct_score
        self.error_score = error_score
        self.enable_syntax_check = enable_syntax_check
        self.max_workers = max_workers

    def __call__(
        self,
        solution_strs: Union[str, List[str]],
        ground_truths: Union[str, List[str]],
    ) -> Union[float, List[float]]:
        """
        Compute reward score(s) for code generation task(s).

        Args:
            solution_strs: Single solution or list of solutions
            ground_truths: Single test or list of tests

        Returns:
            Single score or list of scores
        """
        # Single solution case
        if isinstance(solution_strs, str):
            if not isinstance(ground_truths, str):
                raise ValueError("ground_truths must be str when solution_strs is str")
            return self.compute_single(solution_strs, ground_truths)

        # Batch case
        return self.compute_batch(solution_strs, ground_truths)

    def compute_single(self, solution_str: str, ground_truth: str) -> float:
        """Compute reward for a single solution."""
        return compute_code_score(
            solution_str=solution_str,
            ground_truth=ground_truth,
            config=self.config,
            timeout=self.timeout,
            extraction_method=self.extraction_method,
            correct_score=self.correct_score,
            syntax_error_score=self.error_score,
            execution_error_score=self.error_score,
            test_failure_score=self.error_score,
            no_code_score=self.error_score,
            enable_syntax_check=self.enable_syntax_check,
        )

    def compute_batch(
        self,
        solution_strs: List[str],
        ground_truths: Union[str, List[str]],
    ) -> List[float]:
        """Compute rewards for a batch of solutions concurrently."""
        return compute_code_score_batch(
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            config=self.config,
            timeout=self.timeout,
            extraction_method=self.extraction_method,
            correct_score=self.correct_score,
            syntax_error_score=self.error_score,
            execution_error_score=self.error_score,
            test_failure_score=self.error_score,
            no_code_score=self.error_score,
            enable_syntax_check=self.enable_syntax_check,
            max_workers=self.max_workers,
        )


if __name__ == "__main__":
    # Test cases demonstrating usage
    print("\n" + "=" * 80)
    print("Code Verification Reward Scoring - Test Cases")
    print("=" * 80)

    # Test 1: Correct solution
    print("\n" + "-" * 80)
    print("Test 1: Correct Solution")
    print("-" * 80)
    solution1 = """
Here is my solution to the problem:

```python
def add(a, b):
    '''Add two numbers'''
    return a + b

def multiply(a, b):
    '''Multiply two numbers'''
    return a * b
```

This implementation handles the requirements correctly.
"""
    tests1 = """
assert add(1, 2) == 3, "Failed: add(1, 2)"
assert add(0, 0) == 0, "Failed: add(0, 0)"
assert add(-1, 1) == 0, "Failed: add(-1, 1)"
assert multiply(2, 3) == 6, "Failed: multiply(2, 3)"
assert multiply(0, 5) == 0, "Failed: multiply(0, 5)"
print("All tests passed!")
"""

    result1 = compute_code_score_with_details(solution1, tests1)
    print(f"Result: {result1}")
    print(f"Score: {result1.score}")
    if result1.execution_result:
        print(f"Output: {result1.execution_result.output}")

    # Test 2: Incorrect solution (tests fail)
    print("\n" + "-" * 80)
    print("Test 2: Incorrect Solution (Logic Error)")
    print("-" * 80)
    solution2 = """
```python
def add(a, b):
    return a - b  # Wrong: should be a + b
```
"""
    tests2 = """
assert add(1, 2) == 3
"""

    result2 = compute_code_score_with_details(solution2, tests2)
    print(f"Result: {result2}")
    print(f"Score: {result2.score}")

    # Test 3: Syntax error
    print("\n" + "-" * 80)
    print("Test 3: Syntax Error")
    print("-" * 80)
    solution3 = """
```python
def add(a, b)
    return a + b  # Missing colon
```
"""
    tests3 = """
assert add(1, 2) == 3
"""

    result3 = compute_code_score_with_details(solution3, tests3)
    print(f"Result: {result3}")
    print(f"Score: {result3.score}")
    print(f"Error: {result3.error_message}")

    # Test 4: No code extracted
    print("\n" + "-" * 80)
    print("Test 4: No Code in Response")
    print("-" * 80)
    solution4 = "I think you should use a for loop to solve this problem."
    tests4 = "assert True"

    result4 = compute_code_score_with_details(solution4, tests4)
    print(f"Result: {result4}")
    print(f"Score: {result4.score}")
    print(f"Error: {result4.error_message}")

    # Test 5: Runtime error
    print("\n" + "-" * 80)
    print("Test 5: Runtime Error")
    print("-" * 80)
    solution5 = """
```python
def divide(a, b):
    return a / b
```
"""
    tests5 = """
assert divide(10, 2) == 5
assert divide(10, 0) == 0  # This will cause ZeroDivisionError
"""

    result5 = compute_code_score_with_details(solution5, tests5)
    print(f"Result: {result5}")
    print(f"Score: {result5.score}")

    # Test 6: Custom scoring
    print("\n" + "-" * 80)
    print("Test 6: Custom Scoring (Partial Credit)")
    print("-" * 80)
    solution6 = """
```python
def add(a, b):
    return a - b  # Wrong implementation
```
"""
    tests6 = "assert add(1, 2) == 3"

    # Give partial credit for extracting code, even if tests fail
    score6 = compute_code_score(
        solution6,
        tests6,
        correct_score=1.0,
        test_failure_score=0.3,  # Partial credit for having code
        no_code_score=0.0,
    )
    print(f"Score with partial credit: {score6}")

    print("\n" + "=" * 80)
    print("All test cases completed")
    print("=" * 80)
