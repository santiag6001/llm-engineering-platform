"""Public HTTP routes and error presentation."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
    BackendError,
    BackendHTTPError,
    BackendProtocolError,
    BackendTimeoutError,
    BackendUnavailableError,
)
from llm_platform.domain.models import CompletionCommand, Message

router = APIRouter()


def _backend_error_detail(exc: BackendError) -> tuple[int, ErrorDetail]:
    if isinstance(exc, BackendTimeoutError):
        return 504, ErrorDetail(
            message="The inference backend timed out.",
            type="backend_timeout",
            code="backend_timeout",
        )
    if isinstance(exc, BackendUnavailableError):
        return 503, ErrorDetail(
            message="The inference backend is unavailable.",
            type="backend_unavailable",
            code="backend_unavailable",
        )
    if isinstance(exc, BackendProtocolError):
        return 502, ErrorDetail(
            message="The inference backend returned an invalid response.",
            type="backend_error",
            code="invalid_backend_response",
        )
    if isinstance(exc, BackendHTTPError):
        status_code = exc.status_code if 400 <= exc.status_code < 500 else 502
        return status_code, ErrorDetail(
            message=exc.message,
            type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )
    return 502, ErrorDetail(
        message="The inference backend returned an error.",
        type="backend_error",
        code="backend_error",
    )


def _backend_error_response(exc: BackendError) -> JSONResponse:
    status_code, detail = _backend_error_detail(exc)
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=detail).model_dump(),
    )


def _sse_frame(payload: dict[str, object] | str) -> bytes:
    data = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    return f"data: {data}\n\n".encode()


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
) -> ChatCompletionResponse | JSONResponse | StreamingResponse:
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
    if body.stream:
        stream_context = service.stream(
            command,
            request_id=cast(str, request.state.request_id),
        )
        try:
            chunks = await stream_context.__aenter__()
        except BackendError as exc:
            return _backend_error_response(exc)

        async def present_stream() -> AsyncIterator[bytes]:
            try:
                async for chunk in chunks:
                    yield _sse_frame(chunk.payload)
                yield _sse_frame("[DONE]")
            except asyncio.CancelledError:
                raise
            except BackendError as exc:
                _, detail = _backend_error_detail(exc)
                yield _sse_frame(ErrorResponse(error=detail).model_dump())
            finally:
                await stream_context.__aexit__(None, None, None)

        return StreamingResponse(
            present_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await service.complete(command)
    except BackendError as exc:
        return _backend_error_response(exc)

    return ChatCompletionResponse.model_validate(result.payload)
