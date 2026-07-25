"""Replaceable infrastructure boundaries."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from llm_platform.domain.models import (
    CompletionChunk,
    CompletionCommand,
    CompletionResult,
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
