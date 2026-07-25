from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_platform.evaluation.models import (
    EvaluationCase,
    EvaluatorResult,
    RunnerConfiguration,
)
from llm_platform.evaluation.runner import EvaluationRunner
from tests.evaluation_helpers import evaluation_case, evaluation_dataset


def _response(content: str = "answer") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "backend-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
            },
        },
    )


def _configuration(concurrency: int = 2) -> RunnerConfiguration:
    return RunnerConfiguration(
        base_url="http://platform.test",
        model="requested-model",
        request_timeout_seconds=2,
        maximum_concurrency=concurrency,
    )


@pytest.mark.asyncio
async def test_successful_response_and_request_payload() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await EvaluationRunner(_configuration(), client=client).run(
            evaluation_dataset(evaluation_case())
        )

    assert results[0].status == "completed"
    assert results[0].passed
    assert results[0].measurements.prompt_tokens == 2
    assert results[0].measurements.backend_model == "backend-model"
    payload = json.loads(seen[0].content)
    assert payload["model"] == "requested-model"
    assert payload["stream"] is False
    assert seen[0].url.path == "/v1/chat/completions"


@pytest.mark.asyncio
async def test_successful_response_preview_is_bounded() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response("x" * 1_000))
    ) as client:
        result = (
            await EvaluationRunner(_configuration(), client=client).run(
                evaluation_dataset()
            )
        )[0]
    assert result.response_preview is not None
    assert len(result.response_preview) == 240
    assert result.response_preview.endswith("…")


@pytest.mark.asyncio
async def test_result_order_is_dataset_order_under_concurrency() -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        if prompt == "first":
            first_started.set()
            await release_first.wait()
        else:
            await first_started.wait()
            release_first.set()
        return _response()

    dataset = evaluation_dataset(
        evaluation_case("first", prompt="first"),
        evaluation_case("second", prompt="second"),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await EvaluationRunner(_configuration(), client=client).run(dataset)

    assert [result.id for result in results] == ["first", "second"]


@pytest.mark.asyncio
async def test_concurrency_is_bounded_by_fixed_worker_count() -> None:
    active = 0
    maximum_active = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            two_started.set()
        await release.wait()
        active -= 1
        return _response()

    dataset = evaluation_dataset(
        *(evaluation_case(f"case-{index}") for index in range(5))
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        pending = asyncio.create_task(
            EvaluationRunner(_configuration(2), client=client).run(dataset)
        )
        await asyncio.wait_for(two_started.wait(), timeout=1)
        assert maximum_active == 2
        release.set()
        await pending
    assert maximum_active == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 503])
async def test_platform_http_errors_are_classified(status: int) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(status, text="private body")
        )
    ) as client:
        result = (
            await EvaluationRunner(_configuration(), client=client).run(
                evaluation_dataset()
            )
        )[0]
    assert result.error is not None
    assert result.error.type == "platform_http_error"
    assert result.error.http_status == status
    assert "private body" not in result.error.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exception_type", "expected"),
    [
        (httpx.ReadTimeout, "timeout"),
        (httpx.ConnectError, "connection_failure"),
    ],
)
async def test_transport_failures_are_classified(
    exception_type: type[httpx.RequestError], expected: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exception_type("private transport detail", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = (
            await EvaluationRunner(_configuration(), client=client).run(
                evaluation_dataset()
            )
        )[0]
    assert result.error is not None
    assert result.error.type == expected
    assert "private" not in result.error.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"model": "x", "choices": []}),
        httpx.Response(
            200,
            json={
                "model": "x",
                "choices": [{"message": {"content": 3}}],
            },
        ),
    ],
)
async def test_malformed_responses_are_classified(response: httpx.Response) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: response)
    ) as client:
        result = (
            await EvaluationRunner(_configuration(), client=client).run(
                evaluation_dataset()
            )
        )[0]
    assert result.error is not None
    assert result.error.type == "malformed_platform_response"


@pytest.mark.asyncio
async def test_one_failed_case_does_not_stop_remaining_cases() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500) if calls == 1 else _response()

    dataset = evaluation_dataset(evaluation_case("first"), evaluation_case("second"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await EvaluationRunner(_configuration(1), client=client).run(dataset)
    assert [result.status for result in results] == ["error", "completed"]
    assert calls == 2


@pytest.mark.asyncio
async def test_evaluator_failure_is_safe_and_does_not_stop_workers() -> None:
    def evaluator(_case: EvaluationCase, _actual: str) -> list[EvaluatorResult]:
        raise RuntimeError("private traceback content")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: _response())
    ) as client:
        results = await EvaluationRunner(
            _configuration(1), client=client, evaluator=evaluator
        ).run(evaluation_dataset(evaluation_case("first"), evaluation_case("second")))
    assert len(results) == 2
    assert all(
        result.error and result.error.type == "evaluator_failure" for result in results
    )
    assert all(
        "private" not in result.error.message for result in results if result.error
    )
