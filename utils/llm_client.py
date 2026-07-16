"""Environment-driven OpenAI-compatible client for Ollama, vLLM, or equivalent services."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import socket
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class LLMError(RuntimeError):
    """Base error for local or remote LLM calls."""


class LLMConfigurationError(LLMError):
    """Raised when LLM environment configuration is invalid."""


class LLMResponseError(LLMError):
    """Raised when an LLM response does not satisfy the protocol contract."""


class LLMConnectionError(LLMError):
    """Raised when the configured LLM service cannot be reached."""


class LLMTimeoutError(LLMError):
    """Raised when the configured LLM service exceeds its timeout."""


class LLMHTTPError(LLMError):
    """Raised when the configured LLM service returns an HTTP error."""


def _integer(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(f"{name} must be an integer") from exc


def _float(environment: Mapping[str, str], name: str, default: float) -> float:
    raw = environment.get(name, str(default))
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(f"{name} must be a number") from exc


def _normalize_base_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMConfigurationError("LLM_BASE_URL must be an absolute http(s) URL")
    if not url.endswith("/v1"):
        url += "/v1"
    return url


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model_name: str
    max_tokens: int
    context_length: int
    temperature: float
    timeout_seconds: float
    reasoning_effort: str | None

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "LLMSettings":
        environment = os.environ if environment is None else environment
        max_tokens = _integer(environment, "LLM_MAX_TOKENS", 256)
        context_length = _integer(environment, "LLM_CONTEXT_LENGTH", 4096)
        temperature = _float(environment, "LLM_TEMPERATURE", 0.0)
        timeout_seconds = _float(environment, "LLM_TIMEOUT_SECONDS", 60.0)
        reasoning_effort = environment.get("LLM_REASONING_EFFORT", "none").strip().lower()
        if max_tokens <= 0 or max_tokens > context_length:
            raise LLMConfigurationError(
                "LLM_MAX_TOKENS must be positive and no greater than LLM_CONTEXT_LENGTH"
            )
        if context_length <= 0:
            raise LLMConfigurationError("LLM_CONTEXT_LENGTH must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise LLMConfigurationError("LLM_TEMPERATURE must be between 0 and 2")
        if timeout_seconds <= 0:
            raise LLMConfigurationError("LLM_TIMEOUT_SECONDS must be positive")
        if reasoning_effort not in {"", "none", "low", "medium", "high"}:
            raise LLMConfigurationError(
                "LLM_REASONING_EFFORT must be empty, none, low, medium, or high"
            )
        model_name = environment.get("LLM_MODEL_NAME", "qwen3:4b").strip()
        if not model_name:
            raise LLMConfigurationError("LLM_MODEL_NAME must not be empty")
        api_key = environment.get("LLM_API_KEY", "ollama").strip()
        if not api_key:
            raise LLMConfigurationError("LLM_API_KEY must not be empty")
        return cls(
            base_url=_normalize_base_url(
                environment.get("LLM_BASE_URL", "http://localhost:11434/v1")
            ),
            api_key=api_key,
            model_name=model_name,
            max_tokens=max_tokens,
            context_length=context_length,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            reasoning_effort=reasoning_effort or None,
        )


@dataclass(frozen=True)
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


def parse_chat_response(envelope: Mapping[str, Any]) -> LLMResponse:
    try:
        choices = envelope["choices"]
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMResponseError("LLM response is missing choices/message/content") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError("LLM response content is empty")
    usage = envelope.get("usage") or {}
    return LLMResponse(
        content=content.strip(),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
    )


def parse_json_object(content: str) -> dict[str, Any]:
    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("LLM content is not valid JSON") from exc
    if not isinstance(value, dict):
        raise LLMResponseError("LLM JSON response must be an object")
    return value


def _default_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raise LLMHTTPError(f"LLM service returned HTTP {exc.code}") from exc
    except socket.timeout as exc:
        raise LLMTimeoutError(f"LLM request timed out after {timeout:g} seconds") from exc
    except URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise LLMTimeoutError(f"LLM request timed out after {timeout:g} seconds") from exc
        raise LLMConnectionError(
            "Cannot reach the LLM service; verify LLM_BASE_URL and start the server"
        ) from exc
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMResponseError("LLM service returned a malformed JSON envelope") from exc
    if not isinstance(envelope, dict):
        raise LLMResponseError("LLM service response envelope must be an object")
    return envelope


class LLMClient:
    """Small OpenAI-compatible chat client with injectable transport for tests."""

    def __init__(self, settings: LLMSettings | None = None, transport=None) -> None:
        self.settings = settings or LLMSettings.from_env()
        self._transport = transport or _default_transport

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(LLMSettings.from_env())

    def chat(
        self,
        messages: list[dict[str, str]],
        response_schema: Mapping[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "stream": False,
        }
        if self.settings.reasoning_effort is not None:
            payload["reasoning_effort"] = self.settings.reasoning_effort
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "colacare_response",
                    "strict": True,
                    "schema": dict(response_schema),
                },
            }
        envelope = self._transport(
            f"{self.settings.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self.settings.timeout_seconds,
        )
        if not isinstance(envelope, Mapping):
            raise LLMResponseError("LLM service response envelope must be an object")
        return parse_chat_response(envelope)
