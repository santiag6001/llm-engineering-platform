from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
from fastapi import FastAPI

from llm_platform.main import _normalized_endpoint
from llm_platform.observability import PrometheusMetrics
from tests.conftest import Handler, application_client
from tests.test_chat_completions import UPSTREAM_COMPLETION
from tests.test_streaming import CHUNK_CONTENT, CHUNK_ROLE, ChunkedStream


def _sample(
    metrics: PrometheusMetrics,
    name: str,
    labels: dict[str, str] | None = None,
) -> float:
    value = metrics.registry.get_sample_value(name, labels)
    return 0.0 if value is None else value


def _body(*, stream: bool = False) -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": stream,
    }


def _event(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _stream_response(
    events: list[bytes],
    *,
    terminal_error: Exception | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=ChunkedStream(events, terminal_error=terminal_error),
    )


@pytest.mark.parametrize(
    ("route_path", "endpoint"),
    [
        ("/health", "health"),
        ("/health/live", "health"),
        ("/ready", "ready"),
        ("/health/ready", "ready"),
        ("/metrics", "metrics"),
        ("/v1/models", "models"),
        ("/v1/chat/completions", "chat_completions"),
        ("/private/raw-path", "other"),
    ],
)
def test_endpoint_normalization_uses_only_bounded_labels(
    route_path: str,
    endpoint: str,
) -> None:
    assert _normalized_endpoint(route_path) == endpoint


@pytest.mark.asyncio
async def test_metrics_returns_prometheus_text_from_an_isolated_registry(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    first_app = app_factory(lambda _request: httpx.Response(200))
    second_app = app_factory(lambda _request: httpx.Response(200))

    async with application_client(first_app) as client:
        await client.get("/health")
        response = await client.get("/metrics")
    async with application_client(second_app) as client:
        second_response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=")
    assert "charset=utf-8" in response.headers["content-type"]
    assert "# TYPE llm_platform_http_requests_total counter" in response.text
    assert 'endpoint="health"' in response.text
    assert 'endpoint="metrics"' not in response.text
    assert 'endpoint="health"' not in second_response.text
    assert first_app.state.metrics.registry is not second_app.state.metrics.registry


@pytest.mark.asyncio
async def test_buffered_success_records_lifecycle_and_generated_tokens(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(200, json=UPSTREAM_COMPLETION))

    async with application_client(app) as client:
        response = await client.post("/v1/chat/completions", json=_body())

    metrics = app.state.metrics
    assert response.status_code == 200
    assert (
        _sample(
            metrics,
            "llm_platform_chat_requests_total",
            {"mode": "buffered", "outcome": "success"},
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_http_requests_total",
            {
                "endpoint": "chat_completions",
                "method": "POST",
                "status_class": "2xx",
            },
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_generated_tokens_total",
            {"mode": "buffered"},
        )
        == 2
    )
    assert (
        _sample(
            metrics,
            "llm_platform_request_duration_seconds_count",
            {"mode": "buffered", "outcome": "success"},
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_upstream_duration_seconds_count",
            {"mode": "buffered", "outcome": "success"},
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_active_requests",
            {"mode": "buffered"},
        )
        == 0
    )


@pytest.mark.asyncio
async def test_streaming_success_records_ttft_usage_and_balances_gauges(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    usage = {
        **CHUNK_ROLE,
        "choices": [],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 3,
            "total_tokens": 7,
        },
    }
    app = app_factory(
        lambda _request: _stream_response(
            [
                _event(CHUNK_ROLE),
                _event(CHUNK_CONTENT),
                _event(usage),
                b"data: [DONE]\n\n",
            ]
        )
    )

    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(stream=True),
        )

    metrics = app.state.metrics
    assert response.status_code == 200
    assert response.text.endswith("data: [DONE]\n\n")
    assert (
        _sample(
            metrics,
            "llm_platform_chat_requests_total",
            {"mode": "streaming", "outcome": "success"},
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_time_to_first_token_seconds_count",
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_generated_tokens_total",
            {"mode": "streaming"},
        )
        == 3
    )
    assert (
        _sample(
            metrics,
            "llm_platform_active_requests",
            {"mode": "streaming"},
        )
        == 0
    )
    assert _sample(metrics, "llm_platform_active_streams") == 0


@pytest.mark.asyncio
async def test_no_ttft_is_observed_when_stream_fails_before_content(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: _stream_response([b"data: invalid\n\n"]))

    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(stream=True),
        )

    metrics = app.state.metrics
    assert "[DONE]" not in response.text
    assert _sample(metrics, "llm_platform_time_to_first_token_seconds_count") == 0
    assert (
        _sample(
            metrics,
            "llm_platform_upstream_errors_total",
            {"mode": "streaming", "error_type": "malformed_stream"},
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_chat_requests_total",
            {"mode": "streaming", "outcome": "backend_error"},
        )
        == 1
    )
    assert (
        _sample(
            metrics,
            "llm_platform_active_requests",
            {"mode": "streaming"},
        )
        == 0
    )
    assert _sample(metrics, "llm_platform_active_streams") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_error", "error_type", "secret"),
    [
        (httpx.ReadTimeout, "timeout", "private timeout detail"),
        (httpx.RemoteProtocolError, "disconnect", "private disconnect detail"),
    ],
)
async def test_stream_failures_are_bounded_and_balance_gauges(
    app_factory: Callable[[Handler], FastAPI],
    terminal_error: type[httpx.RequestError],
    error_type: str,
    secret: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _stream_response(
            [],
            terminal_error=terminal_error(secret, request=request),
        )

    app = app_factory(handler)
    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_body(stream=True),
        )
        exposition = (await client.get("/metrics")).text

    metrics = app.state.metrics
    assert "[DONE]" not in response.text
    assert (
        _sample(
            metrics,
            "llm_platform_upstream_errors_total",
            {"mode": "streaming", "error_type": error_type},
        )
        == 1
    )
    assert secret not in exposition
    assert (
        _sample(
            metrics,
            "llm_platform_active_requests",
            {"mode": "streaming"},
        )
        == 0
    )
    assert _sample(metrics, "llm_platform_active_streams") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "endpoint"),
    [
        ("/health", "health"),
        ("/health/live", "health"),
        ("/ready", "ready"),
        ("/health/ready", "ready"),
        ("/v1/models", "models"),
    ],
)
async def test_http_routes_increment_their_normalized_endpoint(
    app_factory: Callable[[Handler], FastAPI],
    path: str,
    endpoint: str,
) -> None:
    app = app_factory(lambda _request: httpx.Response(200))

    async with application_client(app) as client:
        assert (await client.get(path)).status_code == 200

    metrics = app.state.metrics
    assert (
        _sample(
            metrics,
            "llm_platform_http_requests_total",
            {"endpoint": endpoint, "method": "GET", "status_class": "2xx"},
        )
        == 1
    )
    assert _sample(metrics, "llm_platform_chat_requests_total") == 0


@pytest.mark.asyncio
async def test_unavailable_readiness_is_not_counted_as_a_liveness_failure(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(503))

    async with application_client(app) as client:
        assert (await client.get("/ready")).status_code == 503
        assert (await client.get("/health/ready")).status_code == 503

    metrics = app.state.metrics
    assert (
        _sample(
            metrics,
            "llm_platform_http_requests_total",
            {"endpoint": "ready", "method": "GET", "status_class": "5xx"},
        )
        == 2
    )
    assert (
        _sample(
            metrics,
            "llm_platform_http_requests_total",
            {"endpoint": "health", "method": "GET", "status_class": "5xx"},
        )
        == 0
    )


@pytest.mark.asyncio
async def test_unknown_paths_use_other_without_raw_path_or_request_id_labels(
    app_factory: Callable[[Handler], FastAPI],
) -> None:
    app = app_factory(lambda _request: httpx.Response(200))
    raw_path = "/private/tenant-1234"
    request_id = "42dfc6cc-1350-4dae-ae4a-e06427c18468"

    async with application_client(app) as client:
        response = await client.get(raw_path, headers={"x-request-id": request_id})
        exposition = (await client.get("/metrics")).text

    assert response.status_code == 404
    assert (
        _sample(
            app.state.metrics,
            "llm_platform_http_requests_total",
            {"endpoint": "other", "method": "GET", "status_class": "4xx"},
        )
        == 1
    )
    assert raw_path not in exposition
    assert request_id not in exposition
