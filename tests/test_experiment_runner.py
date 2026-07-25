from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import httpx
import pytest

from llm_platform.experiments.models import GenerationConfiguration, RegressionGates
from llm_platform.experiments.registry import ExperimentRegistry, RunNotFoundError
from llm_platform.experiments.runner import ExperimentConfiguration, ExperimentRunner
from tests.experiment_helpers import environment, source


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "case-1",
                "category": "test",
                "messages": [{"role": "user", "content": "prompt"}],
                "expected": {"contains_any": ["answer"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _response(content: str = "answer") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "backend-model",
            "choices": [
                {
                    "message": {"content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
            },
        },
    )


def _configuration(
    dataset: Path,
    *,
    baseline: str | None = None,
    gates: RegressionGates | None = None,
    prompt_file: Path | None = None,
) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        dataset_path=dataset,
        base_url="http://platform.test",
        requested_model="requested-model",
        maximum_concurrency=2,
        timeout_seconds=4,
        generation=GenerationConfiguration(temperature=0, max_tokens=16),
        baseline=baseline,
        regression_gates=gates or RegressionGates(),
        prompt_file=prompt_file,
        prompt_name="shared" if prompt_file else None,
        prompt_version="1" if prompt_file else None,
    )


def _runner(
    registry: ExperimentRegistry,
    client: httpx.AsyncClient,
    run_id: str,
) -> ExperimentRunner:
    return ExperimentRunner(
        registry,
        client=client,
        now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        run_id_factory=lambda _fingerprint: run_id,
        source_collector=source,
        environment_collector=environment,
    )


@pytest.mark.asyncio
async def test_success_reuses_evaluation_outputs_and_registers_manifest(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _response()

    registry = ExperimentRegistry(tmp_path / "experiments")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _runner(registry, client, "run-success").run(
            _configuration(dataset)
        )

    assert result.exit_code == 0
    assert result.manifest.status == "completed"
    sent_messages = cast(list[dict[str, object]], seen[0]["messages"])
    assert sent_messages[0]["content"] == "prompt"
    assert registry.verify("run-success")
    evaluation = json.loads(
        registry.artifact_path("run-success", "evaluation.json").read_text()
    )
    assert evaluation["report_schema_version"] == "1.0"
    assert evaluation["dataset_path"] == "dataset.jsonl"
    assert (
        str(tmp_path)
        not in registry.artifact_path("run-success", "summary.md").read_text()
    )
    assert evaluation["cases"][0]["response_preview"] == "answer"
    assert result.manifest.aggregate_results.prompt_tokens == 2
    assert result.manifest.model.backend_observed == "backend-model"


@pytest.mark.asyncio
async def test_prompt_identity_and_generation_defaults_are_applied(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    prompt = tmp_path / "prompt.txt"
    _write_dataset(dataset)
    prompt.write_text("Use concise answers.", encoding="utf-8")
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _response()

    registry = ExperimentRegistry(tmp_path / "experiments")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await _runner(registry, client, "run-prompt").run(
            _configuration(dataset, prompt_file=prompt)
        )
    assert result.manifest.prompt is not None
    assert result.manifest.prompt.logical_name == "shared"
    assert seen[0]["temperature"] == 0
    assert seen[0]["max_tokens"] == 16
    messages = cast(list[dict[str, object]], seen[0]["messages"])
    assert messages[0]["content"] == "Use concise answers."


@pytest.mark.asyncio
async def test_regression_pass_and_fail_are_registered_without_alias_update(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    responses = iter([_response("answer"), _response("answer"), _response("wrong")])
    registry = ExperimentRegistry(tmp_path / "experiments")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: next(responses))
    ) as client:
        baseline = await _runner(registry, client, "baseline-run").run(
            _configuration(dataset)
        )
        registry.set_alias("baseline", baseline.manifest.run_id)
        passing = await _runner(registry, client, "passing-run").run(
            _configuration(
                dataset,
                baseline="baseline",
                gates=RegressionGates(minimum_pass_rate=1),
            )
        )
        candidate = await _runner(registry, client, "candidate-run").run(
            _configuration(
                dataset,
                baseline="baseline",
                gates=RegressionGates(minimum_pass_rate=1),
            )
        )
    assert passing.exit_code == 0
    assert passing.manifest.regression.decision == "passed"
    assert candidate.exit_code == 1
    assert candidate.manifest.status == "regression_failed"
    assert candidate.manifest.regression.decision == "failed"
    assert registry.show_alias("baseline") == "baseline-run"
    with pytest.raises(RunNotFoundError):
        registry.show_alias("candidate")


@pytest.mark.asyncio
async def test_case_operational_error_and_unexpected_failure_are_preserved(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    registry = ExperimentRegistry(tmp_path / "experiments")

    def connection_failure(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("private detail", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(connection_failure)
    ) as client:
        result = await _runner(registry, client, "case-error").run(
            _configuration(dataset)
        )
    assert result.exit_code == 3
    assert result.manifest.failure is not None
    assert "private" not in result.manifest.failure.message
    assert registry.verify("case-error")

    def unexpected(_request: httpx.Request) -> httpx.Response:
        raise RuntimeError("private exception")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        failed = await _runner(registry, client, "runner-error").run(
            _configuration(dataset)
        )
    assert failed.exit_code == 3
    assert failed.manifest.status == "failed"
    assert {item.path for item in failed.manifest.artifacts} == {
        "regression.json",
        "summary.md",
    }
    assert failed.manifest.failure is not None
    assert "private" not in failed.manifest.failure.message
