"""llama.cpp OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_platform.domain.errors import (
    BackendHTTPError,
    BackendProtocolError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_platform.domain.models import CompletionCommand, CompletionResult

logger = logging.getLogger(__name__)


class _UpstreamErrorDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str
    type: str = "backend_error"
    param: str | None = None
    code: str | None = None


class _UpstreamErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore")

    error: _UpstreamErrorDetail


class _UpstreamCompletion(BaseModel):
    """Validate the stable response shape while retaining extension fields."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None


class LlamaCppBackend:
    """Translate domain commands to llama-server HTTP requests."""

    def __init__(self, client: httpx.AsyncClient, base_url: str) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")

    async def is_ready(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/health")
        except httpx.RequestError:
            return False
        return response.is_success

    async def complete(self, command: CompletionCommand) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": command.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in command.messages
            ],
            "stream": False,
        }
        if command.temperature is not None:
            payload["temperature"] = command.temperature
        if command.max_tokens is not None:
            payload["max_tokens"] = command.max_tokens

        try:
            response = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
            )
        except httpx.TimeoutException as exc:
            logger.warning("backend request timed out")
            raise BackendTimeoutError from exc
        except httpx.NetworkError as exc:
            logger.warning("backend connection failed")
            raise BackendUnavailableError from exc
        except httpx.ProtocolError as exc:
            logger.warning("backend returned an invalid HTTP response")
            raise BackendProtocolError from exc
        except httpx.RequestError as exc:
            logger.warning("backend transport failed")
            raise BackendUnavailableError from exc

        if not response.is_success:
            raise self._http_error(response)

        try:
            decoded = response.json()
            completion = _UpstreamCompletion.model_validate(decoded)
        except (ValueError, ValidationError) as exc:
            logger.warning("backend returned an invalid completion response")
            raise BackendProtocolError from exc

        return CompletionResult(
            payload=completion.model_dump(mode="json", exclude_unset=True)
        )

    @staticmethod
    def _http_error(response: httpx.Response) -> BackendHTTPError:
        default_message = (
            "The inference backend rejected the request."
            if 400 <= response.status_code < 500
            else "The inference backend returned an error."
        )
        try:
            envelope = _UpstreamErrorEnvelope.model_validate(response.json())
        except (ValueError, ValidationError):
            detail = _UpstreamErrorDetail(message=default_message)
        else:
            detail = envelope.error

        logger.warning("backend returned HTTP error status=%d", response.status_code)
        return BackendHTTPError(
            status_code=response.status_code,
            message=detail.message,
            error_type=detail.type,
            param=detail.param,
            code=detail.code,
        )
