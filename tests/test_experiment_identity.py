from __future__ import annotations

from datetime import UTC, datetime

from llm_platform.experiments.identity import (
    canonical_json,
    create_run_id,
    experiment_fingerprint,
)
from llm_platform.experiments.models import (
    DatasetMetadata,
    EvaluationConfiguration,
    EvaluatorConfiguration,
    GenerationConfiguration,
    PromptIdentity,
    RegressionGates,
    SourceMetadata,
)


def _fingerprint(
    *,
    model: str = "model",
    temperature: float = 0,
    dataset_path: str = "a/dataset.jsonl",
) -> str:
    return experiment_fingerprint(
        dataset=DatasetMetadata(
            identifier="dataset",
            path=dataset_path,
            sha256="a" * 64,
            case_count=2,
        ),
        requested_model=model,
        generation=GenerationConfiguration(temperature=temperature, max_tokens=10),
        evaluation=EvaluationConfiguration(
            evaluator=EvaluatorConfiguration(name="lexical", version="1"),
            concurrency=2,
            timeout_seconds=3,
        ),
        regression_gates=RegressionGates(minimum_pass_rate=0.8),
        baseline="baseline",
        prompt=None,
        source=SourceMetadata(git_commit="b" * 40, git_dirty=True),
    )


def test_canonical_json_is_dictionary_order_independent() -> None:
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_fingerprint_is_stable_and_portable_paths_do_not_affect_it() -> None:
    assert _fingerprint(dataset_path="one/dataset.jsonl") == _fingerprint(
        dataset_path="other/dataset.jsonl"
    )


def test_relevant_configuration_changes_fingerprint() -> None:
    assert _fingerprint(model="one") != _fingerprint(model="two")
    assert _fingerprint(temperature=0) != _fingerprint(temperature=0.5)


def test_execution_timestamp_and_output_metrics_are_not_fingerprint_inputs() -> None:
    first = _fingerprint()
    second = _fingerprint()
    assert first == second
    assert len(first) == 64


def test_prompt_source_path_does_not_change_equivalent_input_identity() -> None:
    def with_prompt(source_file: str) -> str:
        return experiment_fingerprint(
            dataset=DatasetMetadata(
                identifier="dataset",
                path="dataset.jsonl",
                sha256="a" * 64,
                case_count=1,
            ),
            requested_model="model",
            generation=GenerationConfiguration(),
            evaluation=EvaluationConfiguration(
                evaluator=EvaluatorConfiguration(name="lexical", version="1"),
                concurrency=1,
                timeout_seconds=1,
            ),
            regression_gates=RegressionGates(),
            baseline=None,
            source=SourceMetadata(git_commit="b" * 40),
            prompt=PromptIdentity(
                logical_name="shared",
                version="1",
                content_sha256="c" * 64,
                source_file=source_file,
            ),
        )

    first = with_prompt("one/prompt.txt")
    second = with_prompt("two/prompt.txt")
    assert first == second


def test_run_id_is_execution_unique_and_contains_short_fingerprint() -> None:
    def fixed() -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)

    first = create_run_id("a" * 64, now=fixed, nonce=lambda: "11111111")
    second = create_run_id("a" * 64, now=fixed, nonce=lambda: "22222222")
    assert first != second
    assert "a" * 12 in first
