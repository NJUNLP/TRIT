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
Sandbox executor for running Python code and tests in an isolated Jupyter environment.
"""

import logging
import re
import threading
from typing import Optional

from verl.utils.reward_score.simple_code_executor import SimpleCodeExecutor

logger = logging.getLogger(__name__)

# ThreadLocal storage for shared kernel executors (每个线程一个独立的 kernel)
_thread_local = threading.local()


class ExecutionResult:
    """Container for code execution results."""

    def __init__(
        self,
        success: bool,
        output: str = "",
        error: str = "",
        timed_out: bool = False,
        executed_code: str = "",
    ):
        self.success = success
        self.output = output
        self.error = error
        self.timed_out = timed_out
        self.executed_code = executed_code

    def __repr__(self):
        return (
            f"ExecutionResult(success={self.success}, "
            f"timed_out={self.timed_out}, "
            f"output_len={len(self.output)}, "
            f"error_len={len(self.error)})"
        )


def parse_execution_result(raw_result: str) -> tuple[str, str, bool]:
    """
    Parse the raw execution result from Jupyter kernel.

    Args:
        raw_result: Raw output string from CodeInterpreter

    Returns:
        Tuple of (stdout, stderr, has_error)
    """
    # Pattern to match different output types
    pattern = re.compile(r"(status|stdout|error|execute_result|display_data|stderr):\s*```(.*?)```", re.DOTALL)
    matches = pattern.finditer(raw_result)

    stdout_parts = []
    stderr_parts = []
    has_error = False

    for match in matches:
        output_type = match.group(1).strip()
        content = match.group(2).strip()

        if output_type in ["error", "stderr"]:
            has_error = True
            stderr_parts.append(content)
        elif output_type in ["stdout", "execute_result", "display_data"]:
            stdout_parts.append(content)

    stdout = "\n".join(stdout_parts)
    stderr = "\n".join(stderr_parts)

    return stdout, stderr, has_error


class CodeSandboxExecutor:
    """
    Executor for running Python code in a sandboxed Jupyter environment.
    
    使用共享的 kernel 实例，避免每次执行都创建新的 kernel，提高效率并减少超时问题。
    每个线程使用独立的 kernel 实例（ThreadLocal），确保线程安全。
    """

    def __init__(self, config: Optional[dict] = None, execution_timeout: int = 10):
        """
        Initialize the sandbox executor.

        Args:
            config: Configuration for SimpleCodeExecutor
            execution_timeout: Maximum execution time in seconds
        """
        self.config = config or {}
        self.execution_timeout = execution_timeout

    def _get_shared_executor(self) -> SimpleCodeExecutor:
        """
        获取当前线程的共享 kernel 执行器（ThreadLocal）。
        
        每个线程第一次调用时会创建 kernel，后续调用复用同一个 kernel。
        
        Returns:
            SimpleCodeExecutor instance for current thread
        """
        # 使用固定的属性名，ThreadLocal 会自动为每个线程创建独立的实例
        if not hasattr(_thread_local, 'shared_executor') or _thread_local.shared_executor is None:
            # 创建新的 kernel 实例
            executor = SimpleCodeExecutor(config=self.config)
            executor.__enter__()
            _thread_local.shared_executor = executor
            logger.debug(f"Created shared kernel executor for thread {threading.current_thread().name}")
        
        return _thread_local.shared_executor
    
    @staticmethod
    def cleanup_thread_executor():
        """
        清理当前线程的共享 kernel 执行器。
        
        通常在线程结束时调用，或者在不再需要时手动清理。
        """
        if hasattr(_thread_local, 'shared_executor') and _thread_local.shared_executor is not None:
            try:
                _thread_local.shared_executor.__exit__(None, None, None)
                logger.debug(f"Cleaned up shared kernel executor for thread {threading.current_thread().name}")
            except Exception as e:
                logger.error(f"Error cleaning up executor: {e}")
            finally:
                _thread_local.shared_executor = None

    def execute_code(self, code: str, timeout: Optional[int] = None) -> ExecutionResult:
        """
        Execute Python code in sandbox using shared kernel instance.

        Args:
            code: Python code to execute
            timeout: Optional timeout override (seconds)

        Returns:
            ExecutionResult object containing execution results
        """
        if timeout is None:
            timeout = self.execution_timeout

        try:
            # 使用共享的 kernel 执行器（每个线程一个独立的实例）
            shared_executor = self._get_shared_executor()
            raw_result = shared_executor.execute_code(code, timeout=timeout)

            # Parse the result
            stdout, stderr, has_error = parse_execution_result(raw_result)

            return ExecutionResult(
                success=not has_error,
                output=stdout,
                error=stderr,
                timed_out=False,
                executed_code=code,
            )

        except TimeoutError as e:
            return ExecutionResult(
                success=False,
                error=f"Execution timeout: {str(e)}",
                timed_out=True,
                executed_code=code,
            )
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return ExecutionResult(
                success=False,
                error=f"Execution failed: {str(e)}",
                timed_out=False,
                executed_code=code,
            )

    def execute_code_with_tests(
        self,
        solution_code: str,
        test_code: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute solution code followed by test code.

        Args:
            solution_code: The solution implementation code
            test_code: The test code (containing assert statements)
            timeout: Optional timeout override (seconds)

        Returns:
            ExecutionResult object. success=True means all tests passed.
        """
        # Combine solution and test code
        combined_code = f"{solution_code}\n\n# Running tests\n{test_code}"
        # print("--------------------------------")
        # print(f"combined_code: {combined_code}")
        # print("--------------------------------")

        result = self.execute_code(combined_code, timeout=timeout)
        print("--------------------------------")
        print(f"result: {result}")
        if result.error:
            print(f"错误原因: {result.error}")
        print("--------------------------------")

        # For test execution, we consider it successful only if:
        # 1. No errors occurred
        # 2. No assertion failures
        if result.success:
            # Check if there were any assertion errors in output
            if "AssertionError" in result.output or "AssertionError" in result.error:
                result.success = False
                if not result.error:
                    result.error = "Assertion failed in test code"

        return result

    def verify_code_syntax(self, code: str) -> tuple[bool, str]:
        """
        Verify that code has valid Python syntax without executing it.

        Args:
            code: Python code to check

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            compile(code, "<string>", "exec")
            return True, ""
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return False, f"Invalid Python code: {str(e)}"


def test_executor():
    """Test function for CodeSandboxExecutor."""
    executor = CodeSandboxExecutor(config={}, execution_timeout=10)

    # Test 1: Simple execution
    print("\n" + "=" * 60)
    print("Test 1: Simple execution")
    print("=" * 60)
    result = executor.execute_code("print('Hello, World!')\nx = 1 + 1\nx")
    print(f"Result: {result}")
    print(f"Output: {result.output}")

    # Test 2: Execution with error
    print("\n" + "=" * 60)
    print("Test 2: Execution with error")
    print("=" * 60)
    result = executor.execute_code("print(undefined_variable)")
    print(f"Result: {result}")
    print(f"Error: {result.error}")

    # Test 3: Solution with tests
    print("\n" + "=" * 60)
    print("Test 3: Solution with tests (passing)")
    print("=" * 60)
    solution = """
def add(a, b):
    return a + b
"""
    tests = """
assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
print("All tests passed!")
"""
    result = executor.execute_code_with_tests(solution, tests)
    print(f"Result: {result}")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")

    # Test 4: Solution with failing tests
    print("\n" + "=" * 60)
    print("Test 4: Solution with tests (failing)")
    print("=" * 60)
    solution = """
def add(a, b):
    return a - b  # Wrong implementation
"""
    tests = """
assert add(1, 2) == 3, "add(1, 2) should equal 3"
"""
    result = executor.execute_code_with_tests(solution, tests)
    print(f"Result: {result}")
    print(f"Success: {result.success}")
    print(f"Error: {result.error}")

    # Test 5: Syntax check
    print("\n" + "=" * 60)
    print("Test 5: Syntax verification")
    print("=" * 60)
    valid_code = "def foo():\n    return 42"
    invalid_code = "def foo(\n    return 42"

    is_valid, error = executor.verify_code_syntax(valid_code)
    print(f"Valid code: {is_valid}, Error: {error}")

    is_valid, error = executor.verify_code_syntax(invalid_code)
    print(f"Invalid code: {is_valid}, Error: {error}")


if __name__ == "__main__":
    test_executor()
