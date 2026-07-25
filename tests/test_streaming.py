from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, MutableMapping
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from llm_platform.application.completions import CompletionService
from llm_platform.backends.llama_cpp import LlamaCppBackend
from llm_platform.domain.errors import BackendProtocolError
from llm_platform.domain.models import CompletionChunk, CompletionCommand, Message
from tests.conftest import application_client

CHUNK_ROLE = {
    "id": "chatcmpl-stream",
    "object": "chat.completion.chunk",
    "created": 1_700_000_000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant"},
            "finish_reason": None,
        }
    ],
}
CHUNK_CONTENT = {
    "id": "chatcmpl-stream",
    "object": "chat.completion.chunk",
    "created": 1_700_000_000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "delta": {"content": "こんにちは"},
            "finish_reason": None,
        }
    ],
}
CHUNK_STOP = {
    "id": "chatcmpl-stream",
    "object": "chat.completion.chunk",
    "created": 1_700_000_000,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }
    ],
}


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: list[bytes],
        *,
        terminal_error: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._terminal_error = terminal_error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._terminal_error is not None:
            raise self._terminal_error

    async def aclose(self) -> None:
        self.closed = True


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = asyncio.Event()
        self._never = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.started.set()
        await self._never.wait()
        if False:  # pragma: no cover - makes this an async generator
            yield b""

    async def aclose(self) -> None:
        self.closed.set()


def _sse_event(payload: dict[str, object]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _streaming_body() -> dict[str, object]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }


def _command() -> CompletionCommand:
    return CompletionCommand(
        model="test-model",
        messages=(Message(role="user", content="Hello"),),
        temperature=None,
        max_tokens=None,
    )


@pytest.mark.asyncio
async def test_successful_stream_uses_openai_sse_and_emits_done(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[httpx.Request] = []
    upstream = ChunkedStream(
        [
            _sse_event(CHUNK_ROLE),
            _sse_event(CHUNK_CONTENT),
            _sse_event(CHUNK_STOP),
            b"data: [DONE]\n\n",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=upstream,
        )

    app = app_factory(handler)
    with caplog.at_level(logging.INFO):
        async with application_client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_streaming_body(),
                headers={"x-request-id": "42dfc6cc-1350-4dae-ae4a-e06427c18468"},
            )

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    frames = response.text.split("\n\n")
    assert frames[-2] == "data: [DONE]"
    assert frames[-1] == ""
    assert response.text.count("data: [DONE]") == 1
    assert "こんにちは" in response.text
    assert upstream.closed
    assert len(seen) == 1
    assert httpx.Response(200, content=seen[0].content).json()["stream"] is True
    stream_log = next(
        record
        for record in caplog.records
        if record.message.startswith("stream completed")
    )
    assert "request_id=42dfc6cc-1350-4dae-ae4a-e06427c18468" in stream_log.message
    assert "ttft_seconds=" in stream_log.message
    assert "ttft_seconds=null" not in stream_log.message
    assert "stream_duration_seconds=" in stream_log.message


@pytest.mark.asyncio
async def test_split_transport_reads_and_unicode_are_parsed_incrementally() -> None:
    encoded = _sse_event(CHUNK_CONTENT)
    unicode_start = encoded.index("こ".encode())
    first_chunk_delivered = asyncio.Event()
    release_remainder = asyncio.Event()

    class GatedStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield encoded[: unicode_start + 1]
            first_chunk_delivered.set()
            await release_remainder.wait()
            yield encoded[unicode_start + 1 :] + b"data: [DONE]\r\n\r\n"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                stream=GatedStream(),
            )
        ),
        timeout=2,
    ) as client:
        backend = LlamaCppBackend(
            client,
            "http://llama.test",
            stream_idle_timeout_seconds=1,
            stream_event_max_bytes=4096,
        )
        async with backend.stream(_command()) as chunks:
            pending: asyncio.Future[CompletionChunk] = asyncio.ensure_future(
                anext(chunks)
            )
            await first_chunk_delivered.wait()
            assert not pending.done()
            release_remainder.set()
            chunk = await pending
            assert chunk.payload == CHUNK_CONTENT
            with pytest.raises(StopAsyncIteration):
                await anext(chunks)


@pytest.mark.asyncio
async def test_slow_consumer_does_not_pull_or_buffer_the_next_event() -> None:
    second_event_requested = asyncio.Event()

    class BackpressuredStream(httpx.AsyncByteStream):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield _sse_event(CHUNK_ROLE)
            second_event_requested.set()
            yield _sse_event(CHUNK_CONTENT) + b"data: [DONE]\n\n"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=BackpressuredStream(),
            )
        ),
        timeout=2,
    ) as client:
        backend = LlamaCppBackend(
            client,
            "http://llama.test",
            stream_idle_timeout_seconds=1,
            stream_event_max_bytes=4096,
        )
        async with backend.stream(_command()) as chunks:
            first = await anext(chunks)
            assert first.payload == CHUNK_ROLE
            assert not second_event_requested.is_set()
            second = await anext(chunks)
            assert second_event_requested.is_set()
            assert second.payload == CHUNK_CONTENT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "upstream_body",
    [
        b"data: not-json\n\n",
        _sse_event(CHUNK_CONTENT),
        b"invalid-field\n\n",
    ],
)
async def test_malformed_or_truncated_sse_emits_error_without_done(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
    upstream_body: bytes,
) -> None:
    app = app_factory(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedStream([upstream_body]),
        )
    )

    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_streaming_body(),
        )

    assert response.status_code == 200
    assert '"code":"invalid_backend_response"' in response.text
    assert "[DONE]" not in response.text


@pytest.mark.asyncio
async def test_stream_idle_timeout_emits_error_without_done(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedStream(
                [],
                terminal_error=httpx.ReadTimeout("idle", request=request),
            ),
        )

    app = app_factory(handler)
    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_streaming_body(),
        )

    assert response.status_code == 200
    assert '"code":"backend_timeout"' in response.text
    assert "idle" not in response.text
    assert "[DONE]" not in response.text


@pytest.mark.asyncio
async def test_upstream_disconnect_emits_protocol_error_without_done(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=ChunkedStream(
                [_sse_event(CHUNK_CONTENT)],
                terminal_error=httpx.RemoteProtocolError(
                    "upstream disconnected",
                    request=request,
                ),
            ),
        )

    app = app_factory(handler)
    async with application_client(app) as client:
        response = await client.post(
            "/v1/chat/completions",
            json=_streaming_body(),
        )

    assert response.status_code == 200
    assert '"code":"invalid_backend_response"' in response.text
    assert "upstream disconnected" not in response.text
    assert "[DONE]" not in response.text


@pytest.mark.asyncio
async def test_streaming_backend_http_error_is_json_before_sse_headers(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = app_factory(
        lambda _request: httpx.Response(
            503,
            json={
                "error": {
                    "message": "Backend overloaded.",
                    "type": "server_error",
                    "code": "overloaded",
                }
            },
        )
    )

    with caplog.at_level(logging.INFO):
        async with application_client(app) as client:
            response = await client.post(
                "/v1/chat/completions",
                json=_streaming_body(),
            )

    assert response.status_code == 502
    assert response.headers["content-type"] == "application/json"
    assert response.json()["error"]["code"] == "overloaded"
    assert any(
        "ttft_seconds=null" in record.message
        and "stream_duration_seconds=" in record.message
        and "outcome=error" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_cancellation_closes_upstream_and_logs_terminal_path(
    caplog: pytest.LogCaptureFixture,
) -> None:
    upstream = BlockingStream()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=upstream,
            )
        ),
        timeout=2,
    ) as client:
        backend = LlamaCppBackend(
            client,
            "http://llama.test",
            stream_idle_timeout_seconds=1,
            stream_event_max_bytes=4096,
        )
        service = CompletionService(backend)
        with caplog.at_level(logging.INFO):
            async with service.stream(_command(), request_id="cancelled") as chunks:
                pending: asyncio.Future[CompletionChunk] = asyncio.ensure_future(
                    anext(chunks)
                )
                await upstream.started.wait()
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending

    assert upstream.closed.is_set()
    assert any(
        "request_id=cancelled" in record.message
        and "outcome=client_cancelled" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_client_disconnect_cancels_and_closes_upstream(
    app_factory: Callable[[Callable[[httpx.Request], httpx.Response]], FastAPI],
) -> None:
    upstream = BlockingStream()
    app = app_factory(
        lambda _request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=upstream,
        )
    )
    body = json.dumps(_streaming_body()).encode()
    request_delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal request_delivered
        if not request_delivered:
            request_delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await upstream.started.wait()
        return {"type": "http.disconnect"}

    sent: list[MutableMapping[str, Any]] = []

    async def send(message: MutableMapping[str, Any]) -> None:
        sent.append(message)

    scope: dict[str, Any] = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 123),
        "server": ("api.test", 80),
    }

    async with app.router.lifespan_context(app):
        await asyncio.wait_for(app(scope, receive, send), timeout=1)

    assert upstream.closed.is_set()
    assert any(message["type"] == "http.response.start" for message in sent)
    assert not any(
        message["type"] == "http.response.body"
        and b"[DONE]" in message.get("body", b"")
        for message in sent
    )


@pytest.mark.asyncio
async def test_stream_event_size_is_bounded() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=ChunkedStream([b"data: " + b"x" * 100]),
            )
        ),
        timeout=2,
    ) as client:
        backend = LlamaCppBackend(
            client,
            "http://llama.test",
            stream_idle_timeout_seconds=1,
            stream_event_max_bytes=32,
        )
        async with backend.stream(_command()) as chunks:
            with pytest.raises(BackendProtocolError):
                await anext(chunks)
