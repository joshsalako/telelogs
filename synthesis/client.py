"""Minimal asynchronous client for an OpenAI-compatible vLLM server."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import aiohttp
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .config import Settings


class ModelAPIError(RuntimeError):
    """Permanent model API failure that should not be retried."""


class TransientModelAPIError(RuntimeError):
    """Temporary HTTP/server failure eligible for retry."""


class MalformedModelOutput(RuntimeError):
    """Syntactically valid API response whose model output fails validation."""


OutputValidator = Callable[[str], None]


class VLLMClient:
    """Shared-session client with bounded concurrency and exponential retry."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_in_flight_requests)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> VLLMClient:
        timeout = aiohttp.ClientTimeout(
            total=self._settings.request_timeout_seconds,
            connect=60.0,
            sock_read=self._settings.request_timeout_seconds,
        )
        connector = aiohttp.TCPConnector(
            limit=self._settings.max_in_flight_requests,
            enable_cleanup_closed=True,
        )
        self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        include_reasoning_content: bool = True,
        validator: OutputValidator | None = None,
    ) -> str:
        """Generate and validate a response, retrying transient/malformed output."""

        retryable = (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
            TransientModelAPIError,
            MalformedModelOutput,
        )
        retrying = AsyncRetrying(
            stop=stop_after_attempt(self._settings.max_retry_attempts),
            wait=wait_random_exponential(
                multiplier=self._settings.retry_backoff_min_seconds,
                min=self._settings.retry_backoff_min_seconds,
                max=self._settings.retry_backoff_max_seconds,
            ),
            retry=retry_if_exception_type(retryable),
            reraise=True,
        )

        async for attempt in retrying:
            with attempt:
                text = await self._request_once(
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    include_reasoning_content=include_reasoning_content,
                )
                if validator is not None:
                    try:
                        validator(text)
                    except MalformedModelOutput:
                        raise
                    except Exception as exc:
                        raise MalformedModelOutput(str(exc)) from exc
                return text
        raise AssertionError("Tenacity retry loop ended without returning or raising")

    async def _request_once(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int,
        include_reasoning_content: bool,
    ) -> str:
        if self._session is None:
            raise RuntimeError("VLLMClient must be used as an async context manager")

        payload = {
            "model": self._settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        async with self._semaphore:
            async with self._session.post(
                self._settings.chat_completions_url,
                json=payload,
                headers=headers,
            ) as response:
                if response.status in {408, 429} or response.status >= 500:
                    body = await response.text()
                    raise TransientModelAPIError(
                        f"vLLM returned HTTP {response.status}: {body[:500]}"
                    )
                if response.status >= 400:
                    body = await response.text()
                    raise ModelAPIError(
                        f"vLLM returned HTTP {response.status}: {body[:500]}"
                    )

                async for line_bytes in response.content:
                    line = line_bytes.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices")
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        r = delta.get("reasoning_content")
                        if r and isinstance(r, str):
                            reasoning_parts.append(r)
                        c = delta.get("content")
                        if c and isinstance(c, str):
                            content_parts.append(c)
                    except Exception:
                        continue

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)

        response_parts = (
            (reasoning, content) if include_reasoning_content else (content,)
        )
        parts = [part.strip() for part in response_parts if part and part.strip()]
        if not parts:
            raise MalformedModelOutput("Model returned no textual content")
        return "\n\n".join(parts)
