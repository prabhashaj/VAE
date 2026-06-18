"""Integration hooks for external agent frameworks.

Provides:
- ``GuardrailClient``: Async HTTP client wrapper for the validation API
- ``guardrail_decorator``: Function decorator to auto-validate prompts
- ``LangChainGuardrail``: LangChain-compatible wrapper
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False


class GuardrailClient:
    """Async HTTP client wrapper for the VAE Guardrail API.

    Parameters
    ----------
    base_url : str
        Base URL of the guardrail API server.
    timeout : float
        Request timeout in seconds.
    fail_open : bool
        If True, allow prompts through when the API is unreachable.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 5.0,
        fail_open: bool = True,
    ) -> None:
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for GuardrailClient. Install with: pip install httpx"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.fail_open = fail_open
        self._client = httpx.AsyncClient(timeout=timeout)

    async def validate(self, text: str) -> dict[str, Any]:
        """Validate a prompt through the guardrail API.

        Returns the full API response as a dictionary.
        Raises ``PromptBlockedError`` if the prompt is blocked
        (unless shadow mode is active on the server side).
        """
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/validate",
                json={"text": text},
            )
            response.raise_for_status()
            result = response.json()

            if result["verdict"] == "block":
                raise PromptBlockedError(
                    f"Prompt blocked by {result.get('blocked_by', 'unknown')} stage",
                    result=result,
                )
            return result

        except httpx.HTTPError as e:
            if self.fail_open:
                logger.warning("Guardrail API unreachable (fail-open): %s", e)
                return {"verdict": "pass", "error": str(e), "fail_open": True}
            raise GuardrailUnavailableError(f"Guardrail API error: {e}") from e

    async def health(self) -> dict[str, Any]:
        """Check the health of the guardrail API."""
        response = await self._client.get(f"{self.base_url}/v1/health")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> GuardrailClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()


class PromptBlockedError(Exception):
    """Raised when a prompt is blocked by the guardrail."""

    def __init__(self, message: str, result: dict | None = None) -> None:
        super().__init__(message)
        self.result = result or {}


class GuardrailUnavailableError(Exception):
    """Raised when the guardrail API is unreachable and fail_open=False."""


def guardrail_decorator(
    base_url: str = "http://localhost:8080",
    fail_open: bool = True,
    timeout: float = 5.0,
) -> Callable:
    """Decorator that validates the first string argument before calling the function.

    Usage::

        @guardrail_decorator()
        async def process_prompt(prompt: str) -> str:
            return await call_llm(prompt)

    Raises ``PromptBlockedError`` if the prompt is blocked.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the first string argument (the prompt)
            prompt = None
            for arg in args:
                if isinstance(arg, str):
                    prompt = arg
                    break
            if prompt is None:
                for v in kwargs.values():
                    if isinstance(v, str):
                        prompt = v
                        break

            if prompt is not None:
                async with GuardrailClient(
                    base_url=base_url, fail_open=fail_open, timeout=timeout
                ) as client:
                    await client.validate(prompt)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


class LangChainGuardrail:
    """LangChain-compatible guardrail wrapper.

    Use as a step in a LangChain pipeline::

        from langchain_core.runnables import RunnablePassthrough

        guardrail = LangChainGuardrail()
        chain = guardrail | llm | parser

    Requires ``langchain-core`` to be installed.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        fail_open: bool = True,
        input_key: str = "input",
    ) -> None:
        self.base_url = base_url
        self.fail_open = fail_open
        self.input_key = input_key

    async def ainvoke(self, input_data: dict | str, **kwargs) -> dict | str:
        """Async invocation compatible with LangChain."""
        text = input_data if isinstance(input_data, str) else input_data.get(self.input_key, "")

        async with GuardrailClient(
            base_url=self.base_url, fail_open=self.fail_open
        ) as client:
            await client.validate(text)

        return input_data

    def invoke(self, input_data: dict | str, **kwargs) -> dict | str:
        """Synchronous invocation (runs async under the hood)."""
        try:
            loop = asyncio.get_running_loop()
            # Already in async context — schedule as task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return loop.run_in_executor(pool, lambda: asyncio.run(self.ainvoke(input_data)))
        except RuntimeError:
            return asyncio.run(self.ainvoke(input_data))
