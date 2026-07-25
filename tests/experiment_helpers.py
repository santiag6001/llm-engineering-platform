from __future__ import annotations

from datetime import UTC, datetime

from llm_platform.experiments.models import (
    AggregateResults,
    DatasetMetadata,
    EnvironmentMetadata,
    EvaluationConfiguration,
    EvaluatorConfiguration,
    ExperimentManifest,
    FailureMetadata,
    GenerationConfiguration,
    ModelMetadata,
    RegressionGates,
    RegressionMetadata,
    ReproductionSpecification,
    SourceMetadata,
)
from llm_platform.experiments.registry import ArtifactPayload, ExperimentRegistry

FIXED_TIME = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def source(commit: str = "a" * 40, *, dirty: bool = False) -> SourceMetadata:
    return SourceMetadata(git_commit=commit, git_dirty=dirty, branch="feature/test")


def environment(fingerprint: str = "e" * 64) -> EnvironmentMetadata:
    return EnvironmentMetadata(
        python_version="3.12.11",
        python_implementation="CPython",
        operating_system="Linux",
        platform_release="fixed",
        architecture="x86_64",
        project_version="0.1.0",
        container="none",
        dependencies={"httpx": "0.28.1", "pydantic": "2.13.4"},
        environment_fingerprint=fingerprint,
    )


def manifest(
    run_id: str = "run-1",
    *,
    artifact: ArtifactPayload | None = None,
    created_at: datetime = FIXED_TIME,
    fingerprint: str = "f" * 64,
    status: str = "completed",
) -> tuple[ExperimentManifest, list[ArtifactPayload]]:
    summary = artifact or ArtifactPayload(
        "summary_markdown", "summary.md", b"# summary\n"
    )
    artifacts = [
        ArtifactPayload("regression_json", "regression.json", b"{}\n"),
        summary,
    ]
    if status != "failed":
        artifacts = [
            ArtifactPayload("evaluation_json", "evaluation.json", b"{}\n"),
            ArtifactPayload("evaluation_markdown", "evaluation.md", b"# eval\n"),
            *artifacts,
        ]
    generation = GenerationConfiguration(temperature=0, max_tokens=20)
    gates = (
        RegressionGates(minimum_pass_rate=1)
        if status == "regression_failed"
        else RegressionGates()
    )
    baseline = "baseline" if status == "regression_failed" else None
    baseline_run_id = "baseline-run" if status == "regression_failed" else None
    value = ExperimentManifest.model_validate(
        {
            "run_id": run_id,
            "experiment_fingerprint": fingerprint,
            "created_at": created_at,
            "status": status,
            "source": source(),
            "dataset": DatasetMetadata(
                identifier="dataset",
                path="evaluations/dataset.jsonl",
                sha256="d" * 64,
                case_count=1,
            ),
            "model": ModelMetadata(
                requested="requested-model",
                backend_observed="backend-model",
            ),
            "generation": generation,
            "evaluation": EvaluationConfiguration(
                evaluator=EvaluatorConfiguration(name="deterministic", version="1.0"),
                concurrency=1,
                timeout_seconds=2,
            ),
            "regression": RegressionMetadata(
                baseline=baseline,
                baseline_run_id=baseline_run_id,
                gates=gates,
                decision=(
                    "failed" if status == "regression_failed" else "not_evaluated"
                ),
            ),
            "environment": environment(),
            "artifacts": [payload.metadata() for payload in artifacts],
            "aggregate_results": AggregateResults(
                pass_rate=1,
                error_rate=0,
                p50_duration_seconds=1,
                p95_duration_seconds=1,
                prompt_tokens=2,
                completion_tokens=3,
            ),
            "reproduction": ReproductionSpecification(
                dataset_path="evaluations/dataset.jsonl",
                requested_model="requested-model",
                generation=generation,
                evaluation_concurrency=1,
                evaluation_timeout_seconds=2,
                baseline=baseline_run_id,
                regression_gates=gates,
                source_git_commit="a" * 40,
                project_version="0.1.0",
            ),
            "failure": (
                FailureMetadata(message="Safe operational failure.")
                if status == "failed"
                else None
            ),
        }
    )
    return value, artifacts


def register_manifest(
    registry: ExperimentRegistry,
    run_id: str = "run-1",
    *,
    artifact: ArtifactPayload | None = None,
    created_at: datetime = FIXED_TIME,
    fingerprint: str = "f" * 64,
    status: str = "completed",
) -> ExperimentManifest:
    value, artifacts = manifest(
        run_id,
        artifact=artifact,
        created_at=created_at,
        fingerprint=fingerprint,
        status=status,
    )
    registry.register(value, artifacts)
    return value
