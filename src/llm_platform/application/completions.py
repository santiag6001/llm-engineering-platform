"""Completion orchestration."""

from __future__ import annotations

from llm_platform.domain.models import CompletionCommand, CompletionResult
from llm_platform.domain.ports import InferenceBackend


class CompletionService:
    """Own the Phase 1 completion use case."""

    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend

    async def complete(self, command: CompletionCommand) -> CompletionResult:
        """Execute a non-streaming completion without retrying generation."""

        return await self._backend.complete(command)
