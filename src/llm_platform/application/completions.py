"""Completion orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

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
    CompletionMode,
    CompletionOutcome,
    CompletionResult,
    UpstreamErrorType,
)
from llm_platform.domain.ports import CompletionMetrics, InferenceBackend

logger = logging.getLogger(__name__)


class CompletionService:
    """Own buffered and streaming completion lifecycles."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        metrics: CompletionMetrics | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._backend = backend
        self._metrics = metrics
        self._clock = clock

    async def complete(self, command: CompletionCommand) -> CompletionResult:
        """Execute a non-streaming completion without retrying generation."""

        mode: CompletionMode = "buffered"
        started_at = self._clock()
        outcome: CompletionOutcome = "success"
        error_type: UpstreamErrorType | None = None
        generated_tokens: int | None = None
        self._request_started(mode)
        try:
            result = await self._backend.complete(command)
            generated_tokens = result.generated_tokens
            return result
        except asyncio.CancelledError:
            outcome = "client_cancelled"
            error_type = "cancelled"
            raise
        except BackendError as exc:
            outcome, error_type = _classify_backend_error(exc)
            raise
        except Exception:
            outcome = "internal_error"
            error_type = "unknown"
            raise
        finally:
            finished_at = self._clock()
            self._request_finished(
                mode=mode,
                outcome=outcome,
                request_duration_seconds=finished_at - started_at,
                upstream_duration_seconds=finished_at - started_at,
                generated_tokens=generated_tokens,
                error_type=error_type,
            )

    def stream(
        self, command: CompletionCommand, *, request_id: str
    ) -> AbstractAsyncContextManager[AsyncIterator[CompletionChunk]]:
        """Open and observe a streaming completion without buffering it."""

        return self._stream(command, request_id=request_id)

    @asynccontextmanager
    async def _stream(
        self, command: CompletionCommand, *, request_id: str
    ) -> AsyncIterator[AsyncIterator[CompletionChunk]]:
        mode: CompletionMode = "streaming"
        started_at = self._clock()
        first_token_at: list[float | None] = [None]
        upstream_finished_at: list[float | None] = [None]
        generated_tokens: list[int | None] = [None]
        outcome: list[CompletionOutcome] = ["success"]
        error_type: list[UpstreamErrorType | None] = [None]
        self._request_started(mode)
        try:
            async with self._backend.stream(command) as chunks:
                yield self._observe_stream(
                    chunks,
                    started_at=started_at,
                    first_token_at=first_token_at,
                    upstream_finished_at=upstream_finished_at,
                    generated_tokens=generated_tokens,
                    outcome=outcome,
                    error_type=error_type,
                )
        except asyncio.CancelledError:
            outcome[0] = "client_cancelled"
            error_type[0] = "cancelled"
            raise
        except BackendError as exc:
            outcome[0], error_type[0] = _classify_backend_error(exc)
            raise
        except Exception:
            outcome[0] = "internal_error"
            error_type[0] = "unknown"
            raise
        finally:
            finished_at = self._clock()
            upstream_finished = upstream_finished_at[0]
            ttft = (
                first_token_at[0] - started_at
                if first_token_at[0] is not None
                else None
            )
            logger.info(
                "stream completed request_id=%s ttft_seconds=%s "
                "stream_duration_seconds=%.6f outcome=%s",
                request_id,
                f"{ttft:.6f}" if ttft is not None else "null",
                finished_at - started_at,
                outcome[0],
            )
            self._request_finished(
                mode=mode,
                outcome=outcome[0],
                request_duration_seconds=finished_at - started_at,
                upstream_duration_seconds=(
                    (
                        upstream_finished
                        if upstream_finished is not None
                        else finished_at
                    )
                    - started_at
                ),
                generated_tokens=generated_tokens[0],
                error_type=error_type[0],
            )

    async def _observe_stream(
        self,
        chunks: AsyncIterator[CompletionChunk],
        *,
        started_at: float,
        first_token_at: list[float | None],
        upstream_finished_at: list[float | None],
        generated_tokens: list[int | None],
        outcome: list[CompletionOutcome],
        error_type: list[UpstreamErrorType | None],
    ) -> AsyncIterator[CompletionChunk]:
        try:
            async for chunk in chunks:
                if first_token_at[0] is None and chunk.has_content:
                    first_token = self._clock()
                    first_token_at[0] = first_token
                    self._time_to_first_token(first_token - started_at)
                if chunk.generated_tokens is not None:
                    generated_tokens[0] = chunk.generated_tokens
                yield chunk
        except asyncio.CancelledError:
            outcome[0] = "client_cancelled"
            error_type[0] = "cancelled"
            raise
        except BackendError as exc:
            outcome[0], error_type[0] = _classify_backend_error(exc)
            raise
        except Exception:
            outcome[0] = "internal_error"
            error_type[0] = "unknown"
            raise
        finally:
            upstream_finished_at[0] = self._clock()

    def _request_started(self, mode: CompletionMode) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.request_started(mode)
        except Exception:
            logger.exception("completion metrics start observation failed")

    def _time_to_first_token(self, duration_seconds: float) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.time_to_first_token_observed(duration_seconds)
        except Exception:
            logger.exception("completion metrics TTFT observation failed")

    def _request_finished(
        self,
        *,
        mode: CompletionMode,
        outcome: CompletionOutcome,
        request_duration_seconds: float,
        upstream_duration_seconds: float,
        generated_tokens: int | None,
        error_type: UpstreamErrorType | None,
    ) -> None:
        if self._metrics is None:
            return
        try:
            self._metrics.request_finished(
                mode=mode,
                outcome=outcome,
                request_duration_seconds=request_duration_seconds,
                upstream_duration_seconds=upstream_duration_seconds,
                generated_tokens=generated_tokens,
                error_type=error_type,
            )
        except Exception:
            logger.exception("completion metrics terminal observation failed")


def _classify_backend_error(
    exc: BackendError,
) -> tuple[CompletionOutcome, UpstreamErrorType]:
    if isinstance(exc, BackendTimeoutError):
        return "backend_timeout", "timeout"
    if isinstance(exc, BackendUnavailableError):
        return "backend_unavailable", "unavailable"
    if isinstance(exc, BackendDisconnectedError):
        return "backend_error", "disconnect"
    if isinstance(exc, BackendMalformedStreamError):
        return "backend_error", "malformed_stream"
    if isinstance(exc, BackendMalformedResponseError):
        return "backend_error", "malformed_response"
    if isinstance(exc, BackendHTTPError):
        error_type: UpstreamErrorType = (
            "http_4xx" if 400 <= exc.status_code < 500 else "http_5xx"
        )
        return "backend_error", error_type
    if isinstance(exc, BackendProtocolError):
        return "backend_error", "malformed_response"
    return "backend_error", "unknown"
