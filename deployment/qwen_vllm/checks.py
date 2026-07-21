"""Small HTTP checks for the OpenAI-compatible vLLM endpoint."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from urllib.request import Request, urlopen

from .config import DeploymentSettings


class CheckError(RuntimeError):
    """Raised when a server readiness check fails."""


Opener = Callable[..., Any]


def _request_json(
    settings: DeploymentSettings,
    path: str,
    *,
    opener: Opener,
    payload: dict[str, Any] | None = None,
) -> Any:
    headers = {"Accept": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
        method = "POST"
    request = Request(
        settings.base_url + path, data=data, headers=headers, method=method
    )
    try:
        with opener(request, timeout=settings.check_timeout_seconds) as response:
            status = response.getcode()
            body = response.read()
    except Exception as exc:
        raise CheckError(f"{path} request failed: {exc}") from exc
    if status < 200 or status >= 300:
        raise CheckError(f"{path} returned HTTP {status}")
    if not body.strip():
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CheckError(f"{path} did not return valid JSON") from exc


def check_server(
    settings: DeploymentSettings,
    *,
    opener: Opener = urlopen,
) -> list[str]:
    """Validate health, model discovery, and one tiny chat completion."""

    settings.validate()
    _request_json(settings, "/health", opener=opener)

    models = _request_json(settings, "/v1/models", opener=opener)
    try:
        model_ids = [item["id"] for item in models["data"]]
    except (KeyError, TypeError) as exc:
        raise CheckError("/v1/models response is missing data[].id") from exc
    if settings.served_model_name not in model_ids:
        raise CheckError(
            f"/v1/models does not advertise {settings.served_model_name!r}"
        )

    completion = _request_json(
        settings,
        "/v1/chat/completions",
        opener=opener,
        payload={
            "model": settings.served_model_name,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "temperature": 0,
            "max_tokens": 8,
        },
    )
    try:
        message = completion["choices"][0]["message"]
        content = message.get("content") or message.get("reasoning_content")
    except (KeyError, IndexError, TypeError) as exc:
        raise CheckError(
            "/v1/chat/completions response is missing choices[0].message"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise CheckError("/v1/chat/completions returned no text")
    return ["health: ok", f"model: {settings.served_model_name}", "chat: ok"]
