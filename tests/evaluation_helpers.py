from __future__ import annotations

from datetime import UTC, datetime

from llm_platform.evaluation.models import (
    CaseResult,
    EvaluationCase,
    EvaluationDataset,
    EvaluationError,
    EvaluationReport,
    RequestMeasurements,
    RunnerConfiguration,
)
from llm_platform.evaluation.reporting import build_report


def evaluation_case(
    case_id: str = "case-1",
    *,
    expected: dict[str, object] | None = None,
    prompt: str = "prompt",
) -> EvaluationCase:
    return EvaluationCase.model_validate(
        {
            "schema_version": "1.0",
            "id": case_id,
            "category": "test",
            "messages": [{"role": "user", "content": prompt}],
            "expected": expected or {"contains_any": ["answer"]},
        }
    )


def evaluation_dataset(*cases: EvaluationCase) -> EvaluationDataset:
    resolved = list(cases) or [evaluation_case()]
    return EvaluationDataset(
        path="dataset.jsonl",
        content_hash="a" * 64,
        cases=resolved,
    )


def completed_result(
    case_id: str = "case-1",
    *,
    passed: bool = True,
    duration: float = 1.0,
    prompt_tokens: int | None = 2,
    completion_tokens: int | None = 3,
) -> CaseResult:
    return CaseResult(
        id=case_id,
        category="test",
        status="completed",
        passed=passed,
        response_preview="answer",
        measurements=RequestMeasurements(
            duration_seconds=duration,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=(
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            ),
        ),
    )


def error_result(case_id: str = "case-error", *, duration: float = 1.0) -> CaseResult:
    return CaseResult(
        id=case_id,
        category="test",
        status="error",
        passed=False,
        measurements=RequestMeasurements(duration_seconds=duration),
        error=EvaluationError(
            type="connection_failure",
            message="Could not connect to the platform.",
        ),
    )


def evaluation_report(
    *results: CaseResult,
    run_id: str = "run",
) -> EvaluationReport:
    dataset = evaluation_dataset(*(evaluation_case(result.id) for result in results))
    configuration = RunnerConfiguration(
        base_url="http://platform.test",
        model="test-model",
        request_timeout_seconds=2,
        maximum_concurrency=2,
    )
    return build_report(
        dataset,
        configuration,
        list(results),
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        run_id_factory=lambda: run_id,
    )
