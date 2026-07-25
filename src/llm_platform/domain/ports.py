"""Replaceable infrastructure boundaries."""

from __future__ import annotations

from typing import Protocol

from llm_platform.domain.models import CompletionCommand, CompletionResult


class InferenceBackend(Protocol):
    async def is_ready(self) -> bool:
        """Return whether the backend can currently serve requests."""
        ...

    async def complete(self, command: CompletionCommand) -> CompletionResult:
        """Perform one non-streaming completion."""
        ...
