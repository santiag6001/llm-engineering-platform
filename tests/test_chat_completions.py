from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI

from tests.conftest import application_client

UPSTREAM_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1_700_000_000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "backend output"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
}


@pytest.mark.asyncio
async def test_valid_request_is_forwarded(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={**UPSTREAM_COMPLETION, "timings": {"x": 1}})

    app = app_factory(handler)
    body = {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        "temperature": 0.4,
        "max_tokens": 12,
    }

    async with application_client(app) as client:
        response = await client.post("/v1/chat/completions", json=body)

    assert response.status_code == 200
    assert response.json() == {**UPSTREAM_COMPLETION, "timings": {"x": 1}}
    assert len(seen) == 1
    assert seen[0].url == "http://llama.test/v1/chat/completions"
    assert seen[0].method == "POST"
    assert httpx.Response(200, content=seen[0].content).json() == {
        **body,
        "stream": False,
    }


@pytest.mark.asyncio
async def test_stream_false_keeps_buffered_behavior(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=UPSTREAM_COMPLETION)

    app = app_factory(handler)
    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 200
    assert response.json() == UPSTREAM_COMPLETION
    assert response.headers["content-type"] == "application/json"
    assert calls == 1


@pytest.mark.asyncio
async def test_unknown_request_field_is_rejected_with_openai_error(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(200, json=UPSTREAM_COMPLETION))

    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "top_p": 0.9,
            },
        )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "message": "The request body is invalid.",
        "type": "invalid_request_error",
        "param": None,
        "code": "validation_error",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_factory", "expected_status", "expected_code"),
    [
        (
            lambda request: httpx.ReadTimeout("slow", request=request),
            504,
            "backend_timeout",
        ),
        (
            lambda request: httpx.ConnectError("refused", request=request),
            503,
            "backend_unavailable",
        ),
    ],
)
async def test_transport_errors_are_mapped(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
    exception_factory: Callable[[httpx.Request], Exception],
    expected_status: int,
    expected_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_factory(request)

    app = app_factory(handler)
    async with application_client(app) as client:
        response = await _post_completion(client)

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert "refused" not in response.text
    assert "slow" not in response.text


@pytest.mark.asyncio
async def test_backend_4xx_preserves_safe_error(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    app = app_factory(
        lambda _request: httpx.Response(
            400,
            json={
                "error": {
                    "message": "Requested model is not loaded.",
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )
    )

    async with application_client(app) as client:
        response = await _post_completion(client)

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Requested model is not loaded.",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


@pytest.mark.asyncio
async def test_backend_5xx_is_mapped_to_bad_gateway(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    app = app_factory(
        lambda _request: httpx.Response(
            503,
            json={
                "error": {
                    "message": "Backend is overloaded.",
                    "type": "server_error",
                    "code": "overloaded",
                }
            },
        )
    )

    async with application_client(app) as client:
        response = await _post_completion(client)

    assert response.status_code == 502
    assert response.json()["error"]["message"] == "Backend is overloaded."
    assert response.json()["error"]["code"] == "overloaded"


@pytest.mark.asyncio
async def test_malformed_backend_json_is_mapped_to_bad_gateway(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    app = app_factory(
        lambda _request: httpx.Response(
            200, content=b"not-json", headers={"content-type": "application/json"}
        )
    )

    async with application_client(app) as client:
        response = await _post_completion(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_backend_response"
    assert "not-json" not in response.text


@pytest.mark.asyncio
async def test_invalid_backend_shape_is_mapped_to_bad_gateway(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(200, json={"choices": []}))

    async with application_client(app) as client:
        response = await _post_completion(client)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_backend_response"


async def _post_completion(client: httpx.AsyncClient) -> httpx.Response:
    return await client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
