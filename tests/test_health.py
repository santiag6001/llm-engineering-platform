from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI

from tests.conftest import Handler, application_client


@pytest.mark.asyncio
async def test_health_returns_200(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(500))

    async with application_client(app) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    assert response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_valid_request_id_is_returned(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(500))
    request_id = "42dfc6cc-1350-4dae-ae4a-e06427c18468"

    async with application_client(app) as client:
        response = await client.get(
            "/health/live", headers={"x-request-id": request_id}
        )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
