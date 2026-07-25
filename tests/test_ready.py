from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI

from tests.conftest import Handler, application_client


@pytest.mark.asyncio
async def test_ready_returns_200_when_backend_is_reachable(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(200))

    async with application_client(app) as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@pytest.mark.asyncio
async def test_ready_returns_503_when_backend_is_unavailable(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    app = app_factory(unavailable)

    async with application_client(app) as client:
        response = await client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}


@pytest.mark.asyncio
async def test_ready_returns_503_when_backend_is_unhealthy(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(500))

    async with application_client(app) as client:
        response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
