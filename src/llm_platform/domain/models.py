"""Backend-neutral completion models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class CompletionCommand:
    model: str
    messages: tuple[Message, ...]
    temperature: float | None
    max_tokens: int | None


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """A validated OpenAI-compatible backend response."""

    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CompletionChunk:
    """A validated, backend-neutral streaming completion chunk."""

    payload: dict[str, Any]

    @property
    def has_content(self) -> bool:
        """Return whether this chunk contains generated assistant content."""

        choices = self.payload.get("choices")
        if not isinstance(choices, list):
            return False
        return any(
            isinstance(choice, dict)
            and isinstance(choice.get("delta"), dict)
            and isinstance(choice["delta"].get("content"), str)
            and bool(choice["delta"]["content"])
            for choice in choices
        )


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
