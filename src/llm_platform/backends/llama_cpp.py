"""llama.cpp OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from llm_platform.domain.errors import (
    BackendDisconnectedError,
    BackendError,
    BackendHTTPError,
    BackendMalformedResponseError,
    BackendMalformedStreamError,
    BackendProtocolError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_platform.domain.models import (
    CompletionChunk,
    CompletionCommand,
    CompletionResult,
)

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


class _UpstreamCompletionChunk(BaseModel):
    """Validate an OpenAI-compatible chunk while retaining extension fields."""

    model_config = ConfigDict(extra="allow")

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]


class LlamaCppBackend:
    """Translate domain commands to llama-server HTTP requests."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        *,
        stream_idle_timeout_seconds: float,
        stream_event_max_bytes: int,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._stream_idle_timeout_seconds = stream_idle_timeout_seconds
        self._stream_event_max_bytes = stream_event_max_bytes

    async def is_ready(self) -> bool:
        try:
            response = await self._client.get(f"{self._base_url}/health")
        except httpx.RequestError:
            return False
        return response.is_success

    async def complete(self, command: CompletionCommand) -> CompletionResult:
        try:
            response = await self._client.post(
                f"{self._base_url}/v1/chat/completions",
                json=self._request_payload(command, stream=False),
            )
        except httpx.RequestError as exc:
            raise self._transport_error(exc) from exc

        if not response.is_success:
            raise self._http_error(response)

        try:
            decoded = response.json()
            completion = _UpstreamCompletion.model_validate(decoded)
        except (ValueError, ValidationError) as exc:
            logger.warning("backend returned an invalid completion response")
            raise BackendMalformedResponseError from exc

        return CompletionResult(
            payload=completion.model_dump(mode="json", exclude_unset=True)
        )

    @asynccontextmanager
    async def stream(
        self, command: CompletionCommand
    ) -> AsyncIterator[AsyncIterator[CompletionChunk]]:
        """Open a validated upstream SSE response and close it on every exit."""

        timeout = httpx.Timeout(
            connect=self._client.timeout.connect,
            read=self._stream_idle_timeout_seconds,
            write=self._client.timeout.write,
            pool=self._client.timeout.pool,
        )
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                json=self._request_payload(command, stream=True),
                timeout=timeout,
            ) as response:
                if not response.is_success:
                    await response.aread()
                    raise self._http_error(response)
                content_type = response.headers.get("content-type", "")
                if content_type.split(";", 1)[0].strip().lower() != (
                    "text/event-stream"
                ):
                    logger.warning("backend stream returned unexpected content type")
                    raise BackendMalformedStreamError
                yield self._iter_sse_chunks(response)
        except (
            BackendHTTPError,
            BackendProtocolError,
            BackendTimeoutError,
            BackendUnavailableError,
        ):
            raise
        except httpx.RequestError as exc:
            raise self._transport_error(exc) from exc

    async def _iter_sse_chunks(
        self, response: httpx.Response
    ) -> AsyncIterator[CompletionChunk]:
        buffer = bytearray()
        data_lines: list[bytes] = []
        event_size = 0

        try:
            async for transport_chunk in response.aiter_bytes():
                buffer.extend(transport_chunk)

                while (newline_index := buffer.find(b"\n")) >= 0:
                    line = bytes(buffer[:newline_index])
                    del buffer[: newline_index + 1]
                    if line.endswith(b"\r"):
                        line = line[:-1]
                    event_size += len(line) + 1
                    if event_size > self._stream_event_max_bytes:
                        raise self._malformed_stream("event exceeded size limit")

                    if line:
                        if not line.startswith(b":"):
                            field, separator, value = line.partition(b":")
                            if not separator:
                                raise self._malformed_stream(
                                    "event field was malformed"
                                )
                            if value.startswith(b" "):
                                value = value[1:]
                            if field == b"data":
                                data_lines.append(value)
                        continue

                    if not data_lines:
                        event_size = 0
                        continue
                    chunk = self._decode_sse_data(b"\n".join(data_lines))
                    data_lines = []
                    event_size = 0
                    if chunk is None:
                        return
                    yield chunk
                if len(buffer) + event_size > self._stream_event_max_bytes:
                    raise self._malformed_stream("event exceeded size limit")
        except httpx.RequestError as exc:
            raise self._transport_error(exc) from exc

        if buffer or data_lines:
            raise self._malformed_stream("stream ended during an event")
        raise self._malformed_stream("stream ended without a DONE event")

    def _decode_sse_data(self, data: bytes) -> CompletionChunk | None:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self._malformed_stream("event was not valid UTF-8") from exc
        if text == "[DONE]":
            return None
        try:
            decoded = json.loads(text)
            chunk = _UpstreamCompletionChunk.model_validate(decoded)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise self._malformed_stream("event data was invalid") from exc
        return CompletionChunk(
            payload=chunk.model_dump(mode="json", exclude_unset=True)
        )

    @staticmethod
    def _request_payload(command: CompletionCommand, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": command.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in command.messages
            ],
            "stream": stream,
        }
        if command.temperature is not None:
            payload["temperature"] = command.temperature
        if command.max_tokens is not None:
            payload["max_tokens"] = command.max_tokens
        return payload

    @staticmethod
    def _transport_error(exc: httpx.RequestError) -> BackendError:
        if isinstance(exc, httpx.TimeoutException):
            logger.warning("backend request timed out")
            return BackendTimeoutError()
        if isinstance(exc, httpx.ProtocolError):
            logger.warning("backend returned an invalid HTTP response")
            return BackendDisconnectedError()
        logger.warning("backend connection failed")
        return BackendUnavailableError()

    @staticmethod
    def _malformed_stream(reason: str) -> BackendMalformedStreamError:
        logger.warning("backend returned a malformed SSE stream reason=%s", reason)
        return BackendMalformedStreamError()

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
