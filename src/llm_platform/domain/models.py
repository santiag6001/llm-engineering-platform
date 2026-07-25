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
class ModelInfo:
    id: str
