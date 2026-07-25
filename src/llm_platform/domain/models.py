"""Backend-neutral completion models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

CompletionMode = Literal["buffered", "streaming"]
CompletionOutcome = Literal[
    "success",
    "backend_timeout",
    "backend_unavailable",
    "backend_error",
    "internal_error",
    "client_cancelled",
]
UpstreamErrorType = Literal[
    "timeout",
    "unavailable",
    "disconnect",
    "malformed_response",
    "malformed_stream",
    "http_4xx",
    "http_5xx",
    "cancelled",
    "unknown",
]


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

    @property
    def generated_tokens(self) -> int | None:
        """Return trusted backend-reported completion tokens when valid."""

        return _generated_tokens(self.payload)


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

    @property
    def generated_tokens(self) -> int | None:
        """Return trusted backend-reported cumulative completion tokens."""

        return _generated_tokens(self.payload)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str


def _generated_tokens(payload: dict[str, Any]) -> int | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    completion_tokens = usage.get("completion_tokens")
    if (
        not isinstance(completion_tokens, int)
        or isinstance(completion_tokens, bool)
        or completion_tokens < 0
    ):
        return None
    return completion_tokens
