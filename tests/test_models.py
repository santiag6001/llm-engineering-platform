from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI

from tests.conftest import Handler, application_client


@pytest.mark.asyncio
async def test_models_lists_configured_public_model(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(500))

    async with application_client(app) as client:
        response = await client.get("/v1/models")

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "data": [{"id": "test-model", "object": "model", "owned_by": "llm-platform"}],
    }
