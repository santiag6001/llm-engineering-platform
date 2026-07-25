from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI

from llm_platform.config import Settings
from llm_platform.main import create_app

Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def app_factory() -> Callable[[Handler], FastAPI]:
    def build(handler: Handler) -> FastAPI:
        return create_app(
            Settings(
                llama_server_base_url="http://llama.test",
                llama_server_timeout_seconds=2,
                public_model="test-model",
            ),
            transport=httpx.MockTransport(handler),
        )

    return build


@asynccontextmanager
async def application_client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://api.test"
        ) as client:
            yield client
