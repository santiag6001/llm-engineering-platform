"""Completion orchestration."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from llm_platform.domain.models import (
    CompletionChunk,
    CompletionCommand,
    CompletionResult,
)
from llm_platform.domain.ports import InferenceBackend

logger = logging.getLogger(__name__)


class CompletionService:
    """Own buffered and streaming completion lifecycles."""

    def __init__(
        self,
        backend: InferenceBackend,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._backend = backend
        self._clock = clock

    async def complete(self, command: CompletionCommand) -> CompletionResult:
        """Execute a non-streaming completion without retrying generation."""

        return await self._backend.complete(command)

    def stream(
        self, command: CompletionCommand, *, request_id: str
    ) -> AbstractAsyncContextManager[AsyncIterator[CompletionChunk]]:
        """Open and observe a streaming completion without buffering it."""

        return self._stream(command, request_id=request_id)

    @asynccontextmanager
    async def _stream(
        self, command: CompletionCommand, *, request_id: str
    ) -> AsyncIterator[AsyncIterator[CompletionChunk]]:
        started_at = self._clock()
        first_token_at: list[float | None] = [None]
        outcome = ["success"]
        try:
            async with self._backend.stream(command) as chunks:
                yield self._observe_stream(
                    chunks,
                    first_token_at=first_token_at,
                    outcome=outcome,
                )
        except asyncio.CancelledError:
            outcome[0] = "client_cancelled"
            raise
        except Exception:
            outcome[0] = "error"
            raise
        finally:
            finished_at = self._clock()
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

    async def _observe_stream(
        self,
        chunks: AsyncIterator[CompletionChunk],
        *,
        first_token_at: list[float | None],
        outcome: list[str],
    ) -> AsyncIterator[CompletionChunk]:
        try:
            async for chunk in chunks:
                if first_token_at[0] is None and chunk.has_content:
                    first_token_at[0] = self._clock()
                yield chunk
        except asyncio.CancelledError:
            outcome[0] = "client_cancelled"
            raise
        except Exception:
            outcome[0] = "error"
            raise
