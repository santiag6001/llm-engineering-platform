"""Public OpenAI-compatible API schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, Any] | None = None


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "llm-platform"


class ModelsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]
