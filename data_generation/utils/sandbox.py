"""Sandbox client for code execution and verification.

The sandbox is an HTTP service that executes Python code against test cases.
This module provides an async client for parallel verification.

Expected sandbox API:
    POST /execute
    Body: {"code": str, "input": str, "timeout": int, "memory_limit_mb": int}
    Response: {"stdout": str, "stderr": str, "exit_code": int, "timed_out": bool}
"""

import asyncio
from typing import List, Optional, Tuple

import aiohttp


class SandboxClient:
    """HTTP client for code execution sandbox with round-robin load balancing."""

    def __init__(self, endpoints: List[str], timeout: int = 10, memory_limit_mb: int = 512):
        self.endpoints = endpoints
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self._counter = 0

    def _next_endpoint(self) -> str:
        endpoint = self.endpoints[self._counter % len(self.endpoints)]
        self._counter += 1
        return endpoint

    async def verify_async(
        self,
        code: str,
        test_cases: List[Tuple[str, str]],
        function_name: Optional[str] = None,
    ) -> Optional[float]:
        """Verify code against test cases. Returns pass_rate (0.0-1.0) or None on error."""
        if not test_cases:
            return None

        endpoint = self._next_endpoint()
        passed = 0
        total = len(test_cases)

        async with aiohttp.ClientSession() as session:
            for input_str, expected_output in test_cases:
                # Wrap function-type code with a main block that calls the function
                exec_code = code
                if function_name:
                    exec_code = self._wrap_function_code(code, input_str, function_name)
                    input_str = ""  # Input is embedded in the wrapper

                try:
                    async with session.post(
                        f"{endpoint}/execute",
                        json={
                            "code": exec_code,
                            "input": input_str,
                            "timeout": self.timeout,
                            "memory_limit_mb": self.memory_limit_mb,
                        },
                        timeout=aiohttp.ClientTimeout(total=self.timeout + 5),
                    ) as resp:
                        if resp.status != 200:
                            continue
                        result = await resp.json()
                        if result.get("timed_out"):
                            continue
                        if result.get("exit_code", 1) != 0:
                            continue
                        stdout = result.get("stdout", "").strip()
                        if stdout == expected_output.strip():
                            passed += 1
                except (asyncio.TimeoutError, aiohttp.ClientError):
                    continue

        return passed / total if total > 0 else None

    @staticmethod
    def _wrap_function_code(code: str, input_str: str, function_name: str) -> str:
        """Wrap function-type code with stdin parsing and function call."""
        return f"""{code}

import sys
_input_data = {repr(input_str)}
sys.stdin = __import__('io').StringIO(_input_data)
_lines = _input_data.strip().split('\\n')
# Call the function - adjust argument parsing as needed for your dataset
result = {function_name}(*_lines)
if result is not None:
    print(result)
"""
