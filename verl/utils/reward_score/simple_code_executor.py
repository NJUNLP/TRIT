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
Simplified Jupyter code executor - standalone version without external dependencies.

This module provides a lightweight Jupyter kernel interface for code execution,
extracted from the original CodeInterpreterStateless and simplified for portability.
"""

import asyncio
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

LAUNCH_KERNEL_PY = """
from ipykernel import kernelapp as app
app.launch_new_instance()
"""


class SimpleCodeExecutor:
    """
    Simplified Jupyter kernel-based code executor.

    This class provides a minimal interface to execute Python code in an isolated
    Jupyter kernel environment. It manages the kernel lifecycle and handles
    code execution with proper cleanup.

    Usage:
        with SimpleCodeExecutor(config={"work_dir": "/tmp"}) as executor:
            result = executor.execute_code("print('Hello')")
            print(result)
    """

    def __init__(self, config: Optional[dict] = None):
        """
        Initialize the code executor.

        Args:
            config: Configuration dictionary. Supported keys:
                - work_dir (str): Working directory for temporary files
                - verbose (bool): Enable verbose logging
                - timeout (int): Default timeout in seconds
        """
        self.config = config or {}
        self.work_dir = self.config.get("work_dir", tempfile.gettempdir())
        self.verbose = self.config.get("verbose", False)
        self.default_timeout = self.config.get("timeout", 30)

        # Generate unique identifiers
        self.instance_id = str(uuid.uuid4())
        self.kernel_id = f"{self.instance_id}_{os.getpid()}_{uuid.uuid4()}"

        # Kernel client and process
        self.kc = None
        self.subproc = None
        self._is_initialized = False

    def __enter__(self):
        """Context manager entry - starts the kernel."""
        self.kc, self.subproc = self._start_kernel()
        self._is_initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self._cleanup()
        return False

    def __del__(self):
        """Destructor - cleanup resources."""
        self._cleanup()

    def _start_kernel(self):
        """
        Start a new Jupyter kernel process.

        Returns:
            Tuple of (kernel_client, kernel_process)
        """
        connection_file = os.path.join(
            self.work_dir,
            f"kernel_connection_{self.kernel_id}.json"
        )
        launch_script = os.path.join(
            self.work_dir,
            f"launch_kernel_{self.kernel_id}.py"
        )

        # Clean up any existing files
        for f in [connection_file, launch_script]:
            if os.path.exists(f):
                if self.verbose:
                    logger.warning(f"Removing existing file: {f}")
                os.remove(f)

        # Create working directory
        os.makedirs(self.work_dir, exist_ok=True)

        # Write kernel launch script
        with open(launch_script, "w") as fout:
            fout.write(LAUNCH_KERNEL_PY)

        # Start kernel process
        kernel_process = subprocess.Popen(
            [
                sys.executable,
                os.path.abspath(launch_script),
                "--IPKernelApp.connection_file",
                os.path.abspath(connection_file),
                "--matplotlib=inline",
                "--quiet",
            ],
            cwd=os.path.abspath(self.work_dir),
            stdout=subprocess.DEVNULL if not self.verbose else None,
            stderr=subprocess.DEVNULL if not self.verbose else None,
        )

        if self.verbose:
            logger.info(f"Started kernel process with PID: {kernel_process.pid}")

        # Wait for connection file to be ready
        max_wait = 30  # seconds
        start_time = time.time()
        while time.time() - start_time < max_wait:
            if os.path.isfile(connection_file):
                try:
                    with open(connection_file) as fp:
                        json.load(fp)
                    break
                except json.JSONDecodeError:
                    # File may be partially written
                    time.sleep(0.1)
            else:
                time.sleep(0.1)
        else:
            raise TimeoutError("Kernel connection file not created within timeout")

        # Create kernel client
        from jupyter_client import BlockingKernelClient

        kc = BlockingKernelClient(connection_file=connection_file)

        # Handle asyncio event loop policy for multi-threading
        original_policy = asyncio.get_event_loop_policy()
        try:
            if not isinstance(original_policy, _AnyThreadEventLoopPolicy):
                asyncio.set_event_loop_policy(_AnyThreadEventLoopPolicy())
            kc.load_connection_file()
            kc.start_channels()
            kc.wait_for_ready()
        except RuntimeError as e:
            if self.verbose:
                logger.warning(f"Event loop policy warning: {e}")
        finally:
            asyncio.set_event_loop_policy(original_policy)

        return kc, kernel_process

    def _cleanup(self):
        """Clean up kernel resources and temporary files."""
        # Shutdown kernel client
        if hasattr(self, "kc") and self.kc is not None:
            try:
                self.kc.stop_channels()
                self.kc.shutdown()
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to shutdown kernel client: {e}")
            finally:
                self.kc = None

        # Terminate subprocess
        if hasattr(self, "subproc") and self.subproc is not None:
            try:
                self.subproc.terminate()
                try:
                    self.subproc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if self.verbose:
                        logger.warning(f"Force killing kernel process {self.subproc.pid}")
                    self.subproc.kill()
                    self.subproc.wait()
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Failed to terminate subprocess: {e}")
            finally:
                self.subproc = None

        # Remove temporary files
        if hasattr(self, "kernel_id") and hasattr(self, "work_dir"):
            connection_file = os.path.join(
                self.work_dir,
                f"kernel_connection_{self.kernel_id}.json"
            )
            launch_script = os.path.join(
                self.work_dir,
                f"launch_kernel_{self.kernel_id}.py"
            )
            for f in [connection_file, launch_script]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception as e:
                        if self.verbose:
                            logger.warning(f"Failed to remove {f}: {e}")

    def execute_code(self, code: str, timeout: Optional[int] = None) -> str:
        """
        Execute Python code in the kernel.

        Args:
            code: Python code to execute
            timeout: Optional timeout in seconds (uses default if None)

        Returns:
            String containing formatted execution results with output types

        Raises:
            RuntimeError: If kernel is not initialized
        """
        if not self._is_initialized:
            raise RuntimeError("Kernel not initialized. Use 'with' statement.")

        if timeout is None:
            timeout = self.default_timeout

        # Wait for kernel to be ready
        self.kc.wait_for_ready()

        # Execute code
        self.kc.execute(code)

        # Collect results
        result = ""
        start_time = time.time()

        while True:
            text = ""
            finished = False
            msg_type = "error"

            try:
                remaining_time = timeout - (time.time() - start_time)
                if remaining_time <= 0:
                    text = "Timeout: Code execution exceeded the time limit."
                    finished = True
                else:
                    msg = self.kc.get_iopub_msg(timeout=min(1.0, remaining_time))
                    msg_type = msg["msg_type"]

                    if msg_type == "status":
                        if msg["content"].get("execution_state") == "idle":
                            finished = True
                    elif msg_type == "execute_result":
                        text = msg["content"]["data"].get("text/plain", "")
                    elif msg_type == "display_data":
                        text = msg["content"]["data"].get("text/plain", "")
                    elif msg_type == "stream":
                        msg_type = msg["content"]["name"]  # stdout or stderr
                        text = msg["content"]["text"]
                    elif msg_type == "error":
                        text = _escape_ansi("\n".join(msg["content"]["traceback"]))

            except queue.Empty:
                # Check if we've exceeded timeout
                if time.time() - start_time >= timeout:
                    text = "Timeout: Code execution exceeded the time limit."
                    finished = True
                else:
                    continue
            except Exception as e:
                text = f"Unexpected error during execution: {str(e)}"
                msg_type = "error"
                finished = True

            if text:
                result += f"\n\n{msg_type}:\n\n```\n{text}\n```"

            if finished:
                break

        return result.lstrip("\n")


def _escape_ansi(line: str) -> str:
    """Remove ANSI escape codes from string."""
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


# Asyncio event loop policy for multi-threading support
# Borrowed from Tornado:
# https://www.tornadoweb.org/en/stable/_modules/tornado/platform/asyncio.html

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy
else:
    _BasePolicy = asyncio.DefaultEventLoopPolicy


class _AnyThreadEventLoopPolicy(_BasePolicy):
    """
    Event loop policy that allows loop creation on any thread.

    The default asyncio event loop policy only automatically creates
    event loops in the main threads. This policy allows event loops
    to be created automatically on any thread.
    """

    def get_event_loop(self) -> asyncio.AbstractEventLoop:
        try:
            return super().get_event_loop()
        except RuntimeError:
            loop = self.new_event_loop()
            self.set_event_loop(loop)
            return loop


def test_simple_executor():
    """Test function for SimpleCodeExecutor."""
    print("\n" + "=" * 60)
    print("Testing SimpleCodeExecutor")
    print("=" * 60)

    # Test 1: Basic execution
    print("\nTest 1: Basic execution")
    print("-" * 60)
    with SimpleCodeExecutor(config={"verbose": True}) as executor:
        result = executor.execute_code("print('Hello, World!')\nx = 1 + 1\nx")
        print("Result:")
        print(result)

    # Test 2: Error handling
    print("\nTest 2: Error handling")
    print("-" * 60)
    with SimpleCodeExecutor() as executor:
        result = executor.execute_code("print(undefined_variable)")
        print("Result:")
        print(result)

    # Test 3: Timeout
    print("\nTest 3: Timeout test")
    print("-" * 60)
    with SimpleCodeExecutor() as executor:
        result = executor.execute_code("import time\ntime.sleep(5)", timeout=2)
        print("Result:")
        print(result)

    print("\n" + "=" * 60)
    print("All tests completed")
    print("=" * 60)


if __name__ == "__main__":
    test_simple_executor()
