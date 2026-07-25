"""Public HTTP routes and error presentation."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from llm_platform.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelObject,
    ModelsResponse,
)
from llm_platform.application.completions import CompletionService
from llm_platform.backends.llama_cpp import LlamaCppBackend
from llm_platform.config import Settings
from llm_platform.domain.errors import (
    BackendHTTPError,
    BackendProtocolError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_platform.domain.models import CompletionCommand, Message

router = APIRouter()


def _error_response(
    status_code: int,
    message: str,
    error_type: str,
    *,
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    envelope = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            param=param,
            code=code,
        )
    )
    return JSONResponse(status_code=status_code, content=envelope.model_dump())


@router.get("/health", response_model=HealthResponse, tags=["health"])
@router.get("/health/live", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Process-local liveness; it never probes the inference backend."""

    return HealthResponse(status="healthy")


@router.get(
    "/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
    tags=["health"],
)
@router.get(
    "/health/ready",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse}},
    tags=["health"],
)
async def ready(request: Request) -> HealthResponse | JSONResponse:
    """Report whether llama-server is reachable and healthy."""

    backend = cast(LlamaCppBackend, request.app.state.backend)
    if await backend.is_ready():
        return HealthResponse(status="ready")
    return JSONResponse(status_code=503, content={"status": "unavailable"})


@router.get("/v1/models", response_model=ModelsResponse, tags=["models"])
async def models(request: Request) -> ModelsResponse:
    settings = cast(Settings, request.app.state.settings)
    return ModelsResponse(data=[ModelObject(id=settings.public_model)])


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_unset=True,
    responses={
        400: {"model": ErrorResponse},
        502: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    tags=["chat"],
)
async def chat_completions(
    body: ChatCompletionRequest, request: Request
) -> ChatCompletionResponse | JSONResponse:
    if body.stream:
        return _error_response(
            400,
            "Streaming is not supported in Phase 1; set stream to false.",
            "unsupported_parameter",
            param="stream",
            code="streaming_not_supported",
        )

    command = CompletionCommand(
        model=body.model,
        messages=tuple(
            Message(role=message.role, content=message.content)
            for message in body.messages
        ),
        temperature=body.temperature,
        max_tokens=body.max_tokens,
    )
    service = cast(CompletionService, request.app.state.completion_service)
    try:
        result = await service.complete(command)
    except BackendTimeoutError:
        return _error_response(
            504,
            "The inference backend timed out.",
            "backend_timeout",
            code="backend_timeout",
        )
    except BackendUnavailableError:
        return _error_response(
            503,
            "The inference backend is unavailable.",
            "backend_unavailable",
            code="backend_unavailable",
        )
    except BackendProtocolError:
        return _error_response(
            502,
            "The inference backend returned an invalid response.",
            "backend_error",
            code="invalid_backend_response",
        )
    except BackendHTTPError as exc:
        status_code = exc.status_code if 400 <= exc.status_code < 500 else 502
        return _error_response(
            status_code,
            exc.message,
            exc.error_type,
            param=exc.param,
            code=exc.code,
        )

    return ChatCompletionResponse.model_validate(result.payload)
