"""Bounded asynchronous evaluation runner."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_platform.evaluation.evaluators import bounded_preview, evaluate_response
from llm_platform.evaluation.models import (
    CaseResult,
    EvaluationCase,
    EvaluationDataset,
    EvaluationError,
    EvaluatorResult,
    RequestMeasurements,
    RunnerConfiguration,
)


class _ResponseMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    content: str


class _ResponseChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    message: _ResponseMessage
    finish_reason: str | None = Field(default=None, max_length=100)


class _ResponseUsage(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class _PlatformResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str = Field(max_length=200)
    choices: list[_ResponseChoice]
    usage: _ResponseUsage | None = None


class EvaluationRunner:
    """Evaluate a dataset with a fixed-size worker set and no retries."""

    def __init__(
        self,
        configuration: RunnerConfiguration,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Callable[[], float] = time.perf_counter,
        evaluator: Callable[
            [EvaluationCase, str], list[EvaluatorResult]
        ] = evaluate_response,
    ) -> None:
        self.configuration = configuration
        self._client = client
        self._clock = clock
        self._evaluator = evaluator

    async def run(self, dataset: EvaluationDataset) -> list[CaseResult]:
        """Return results in dataset order despite concurrent execution."""

        timeout = httpx.Timeout(self.configuration.request_timeout_seconds)
        if self._client is not None:
            return await self._run_workers(dataset, self._client)
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await self._run_workers(dataset, client)

    async def _run_workers(
        self, dataset: EvaluationDataset, client: httpx.AsyncClient
    ) -> list[CaseResult]:
        results: list[CaseResult | None] = [None] * len(dataset.cases)
        next_index = 0

        async def worker() -> None:
            nonlocal next_index
            while next_index < len(dataset.cases):
                index = next_index
                next_index += 1
                results[index] = await self._evaluate_case(dataset.cases[index], client)

        worker_count = min(self.configuration.maximum_concurrency, len(dataset.cases))
        async with asyncio.TaskGroup() as group:
            for _ in range(worker_count):
                group.create_task(worker())
        return [result for result in results if result is not None]

    async def _evaluate_case(
        self, case: EvaluationCase, client: httpx.AsyncClient
    ) -> CaseResult:
        started_at = self._clock()
        try:
            response = await client.post(
                f"{str(self.configuration.base_url).rstrip('/')}/v1/chat/completions",
                json=self._payload(case),
                timeout=self.configuration.request_timeout_seconds,
            )
        except httpx.TimeoutException:
            return self._error_result(case, started_at, "timeout", "Request timed out.")
        except httpx.RequestError:
            return self._error_result(
                case,
                started_at,
                "connection_failure",
                "Could not connect to the platform.",
            )

        duration = max(0.0, self._clock() - started_at)
        if not response.is_success:
            return CaseResult(
                id=case.id,
                category=case.category,
                status="error",
                passed=False,
                measurements=RequestMeasurements(duration_seconds=duration),
                error=EvaluationError(
                    type="platform_http_error",
                    message=f"Platform returned HTTP {response.status_code}.",
                    http_status=response.status_code,
                ),
            )
        try:
            parsed = _PlatformResponse.model_validate(response.json())
            if not parsed.choices:
                raise ValueError("response choices were empty")
            content = parsed.choices[0].message.content
            usage = parsed.usage
            if usage is not None and (
                isinstance(usage.prompt_tokens, bool)
                or isinstance(usage.completion_tokens, bool)
                or isinstance(usage.total_tokens, bool)
                or min(
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    usage.total_tokens,
                )
                < 0
            ):
                raise ValueError("response usage was invalid")
        except (ValueError, ValidationError):
            return CaseResult(
                id=case.id,
                category=case.category,
                status="error",
                passed=False,
                measurements=RequestMeasurements(duration_seconds=duration),
                error=EvaluationError(
                    type="malformed_platform_response",
                    message="Platform returned a malformed completion response.",
                ),
            )

        measurements = RequestMeasurements(
            duration_seconds=duration,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=(usage.completion_tokens if usage is not None else None),
            total_tokens=usage.total_tokens if usage is not None else None,
            backend_model=parsed.model,
            finish_reason=parsed.choices[0].finish_reason,
        )
        try:
            evaluator_results = self._evaluator(case, content)
        except Exception:
            return CaseResult(
                id=case.id,
                category=case.category,
                status="error",
                passed=False,
                response_preview=bounded_preview(content),
                measurements=measurements,
                error=EvaluationError(
                    type="evaluator_failure",
                    message="A deterministic evaluator failed.",
                ),
            )
        passed = all(result.passed for result in evaluator_results)
        return CaseResult(
            id=case.id,
            category=case.category,
            status="completed",
            passed=passed,
            response_preview=bounded_preview(content),
            measurements=measurements,
            evaluator_results=evaluator_results,
        )

    def _error_result(
        self,
        case: EvaluationCase,
        started_at: float,
        error_type: str,
        message: str,
    ) -> CaseResult:
        return CaseResult(
            id=case.id,
            category=case.category,
            status="error",
            passed=False,
            measurements=RequestMeasurements(
                duration_seconds=max(0.0, self._clock() - started_at)
            ),
            error=EvaluationError.model_validate(
                {"type": error_type, "message": message}
            ),
        )

    def _payload(self, case: EvaluationCase) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.configuration.model,
            "messages": [message.model_dump(mode="json") for message in case.messages],
            "stream": False,
        }
        if case.generation.temperature is not None:
            payload["temperature"] = case.generation.temperature
        if case.generation.max_tokens is not None:
            payload["max_tokens"] = case.generation.max_tokens
        return payload
