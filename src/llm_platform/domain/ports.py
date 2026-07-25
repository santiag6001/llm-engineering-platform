"""Replaceable infrastructure boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from llm_platform.domain.models import (
    CompletionChunk,
    CompletionCommand,
    CompletionMode,
    CompletionOutcome,
    CompletionResult,
    UpstreamErrorType,
)


class InferenceBackend(Protocol):
    async def is_ready(self) -> bool:
        """Return whether the backend can currently serve requests."""
        ...

    async def complete(self, command: CompletionCommand) -> CompletionResult:
        """Perform one non-streaming completion."""
        ...

    def stream(
        self, command: CompletionCommand
    ) -> AbstractAsyncContextManager[AsyncIterator[CompletionChunk]]:
        """Open one incremental completion stream owned by the caller."""
        ...


class CompletionMetrics(Protocol):
    """Backend-neutral completion lifecycle observations."""

    def request_started(self, mode: CompletionMode) -> None:
        """Record that one completion lifecycle became active."""
        ...

    def time_to_first_token_observed(self, duration_seconds: float) -> None:
        """Observe backend start to first valid content-bearing chunk."""
        ...

    def request_finished(
        self,
        *,
        mode: CompletionMode,
        outcome: CompletionOutcome,
        request_duration_seconds: float,
        upstream_duration_seconds: float,
        generated_tokens: int | None,
        error_type: UpstreamErrorType | None,
    ) -> None:
        """Record exactly one terminal completion observation."""
        ...
