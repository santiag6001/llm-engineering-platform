"""FastAPI composition root."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from llm_platform.api.routes import router
from llm_platform.api.schemas import ErrorDetail, ErrorResponse
from llm_platform.application.completions import CompletionService
from llm_platform.backends.llama_cpp import LlamaCppBackend
from llm_platform.config import Settings

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build an application with lifecycle-managed dependencies."""

    resolved_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        timeout = httpx.Timeout(resolved_settings.llama_server_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
            backend = LlamaCppBackend(
                client=client,
                base_url=resolved_settings.backend_base_url,
                stream_idle_timeout_seconds=(
                    resolved_settings.llama_server_stream_idle_timeout_seconds
                ),
                stream_event_max_bytes=(
                    resolved_settings.llama_server_stream_event_max_bytes
                ),
            )
            app.state.settings = resolved_settings
            app.state.backend = backend
            app.state.completion_service = CompletionService(backend)
            logger.info("application started")
            yield
            logger.info("application stopped")

    app = FastAPI(
        title="LLM Production Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Any:
        supplied_request_id = request.headers.get("x-request-id")
        try:
            request_id = (
                str(UUID(supplied_request_id))
                if supplied_request_id is not None
                else str(uuid4())
            )
        except ValueError:
            request_id = str(uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["x-request-id"] = request_id
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        logger.info(
            "request completed request_id=%s method=%s route=%s status=%d",
            request_id,
            request.method,
            route_path,
            response.status_code,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        envelope = ErrorResponse(
            error=ErrorDetail(
                message="The request body is invalid.",
                type="invalid_request_error",
                code="validation_error",
            )
        )
        return JSONResponse(status_code=422, content=envelope.model_dump())

    app.include_router(router)
    return app


app = create_app()
