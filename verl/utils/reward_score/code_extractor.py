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
Code extraction utilities for extracting Python code from model responses.
"""

import re
from typing import Optional


def extract_python_code(solution_str: str, method: str = "markdown") -> Optional[str]:
    """
    Extract Python code from model response text.

    Args:
        solution_str: The model's response text that may contain code
        method: Extraction method, choices are:
            - "markdown": Extract from markdown code blocks (```python or ```)
            - "all_blocks": Extract all code blocks and concatenate
            - "last_block": Extract only the last code block

    Returns:
        The extracted Python code string, or None if no code found

    Examples:
        >>> text = "Here is the solution:\\n```python\\ndef add(a, b):\\n    return a + b\\n```"
        >>> extract_python_code(text)
        'def add(a, b):\\n    return a + b'
    """
    if not solution_str or not solution_str.strip():
        return None

    # Pattern to match code blocks with optional language specifier
    # Matches: ```python\ncode\n``` or ```\ncode\n```
    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    matches = re.findall(pattern, solution_str, re.DOTALL | re.IGNORECASE)

    if not matches:
        # If no code blocks found, try to extract code without markdown
        # This handles cases where code is provided without triple backticks
        return _extract_code_without_markdown(solution_str)

    if method == "markdown" or method == "last_block":
        # Return the last code block (most common case)
        return matches[-1].strip()
    elif method == "all_blocks":
        # Concatenate all code blocks with newlines
        return "\n\n".join(match.strip() for match in matches)
    else:
        raise ValueError(f"Unknown extraction method: {method}")


def _extract_code_without_markdown(text: str) -> Optional[str]:
    """
    Try to extract code from text without markdown formatting.
    This is a fallback when no code blocks are found.

    Looks for common Python code patterns like function definitions,
    class definitions, or import statements.
    """
    # Check if text looks like Python code
    python_indicators = [
        r"^\s*def\s+\w+\s*\(",  # function definition
        r"^\s*class\s+\w+",     # class definition
        r"^\s*import\s+\w+",    # import statement
        r"^\s*from\s+\w+",      # from import
        r"^\s*@\w+",            # decorator
    ]

    for pattern in python_indicators:
        if re.search(pattern, text, re.MULTILINE):
            # Text appears to contain Python code, return as-is
            return text.strip()

    return None


def clean_code(code: str) -> str:
    """
    Clean extracted code by removing common artifacts.

    Args:
        code: The extracted code string

    Returns:
        Cleaned code string
    """
    if not code:
        return code

    # Remove leading/trailing whitespace
    code = code.strip()

    # Remove common markdown artifacts that might slip through
    code = re.sub(r'^```(?:python|py)?\s*\n?', '', code, flags=re.IGNORECASE)
    code = re.sub(r'\n?```\s*$', '', code)

    return code


def extract_and_validate_code(solution_str: str, method: str = "markdown") -> tuple[Optional[str], bool, str]:
    """
    Extract Python code and validate it's not empty.

    Args:
        solution_str: The model's response text
        method: Extraction method (see extract_python_code)

    Returns:
        Tuple of (code, is_valid, error_message)
        - code: Extracted code or None
        - is_valid: Whether valid code was extracted
        - error_message: Error message if invalid, empty string otherwise
    """
    try:
        code = extract_python_code(solution_str, method=method)

        if code is None:
            return None, False, "No Python code found in response"

        code = clean_code(code)

        if not code:
            return None, False, "Extracted code is empty after cleaning"

        return code, True, ""

    except Exception as e:
        return None, False, f"Error extracting code: {str(e)}"


if __name__ == "__main__":
    # Test cases
    test_cases = [
        # Case 1: Standard markdown code block
        """Here is the solution:
```python
def add(a, b):
    return a + b
```
This function adds two numbers.""",

        # Case 2: Multiple code blocks
        """First, let's import:
```python
import math
```

Then the implementation:
```python
def calculate_area(radius):
    return math.pi * radius ** 2
```""",

        # Case 3: Code without markdown
        """def multiply(a, b):
    return a * b""",

        # Case 4: No code
        "This is just a text response without any code.",
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}:")
        print(f"{'='*60}")
        print(f"Input:\n{test}\n")

        code, is_valid, error = extract_and_validate_code(test)
        print(f"Valid: {is_valid}")
        if is_valid:
            print(f"Extracted Code:\n{code}")
        else:
            print(f"Error: {error}")
