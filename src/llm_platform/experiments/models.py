"""Strict persisted models for reproducible experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MANIFEST_SCHEMA_VERSION: Literal["1.0"] = "1.0"
COMPARISON_SCHEMA_VERSION: Literal["1.0"] = "1.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class StrictModel(BaseModel):
    """Forbid unreviewed fields at persisted experiment boundaries."""

    model_config = ConfigDict(extra="forbid")


class SourceMetadata(StrictModel):
    git_commit: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{7,64}$", max_length=64
    )
    git_dirty: bool | None = None
    branch: str | None = Field(default=None, min_length=1, max_length=255)


class DatasetMetadata(StrictModel):
    identifier: str = Field(min_length=1, max_length=128)
    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=SHA256_PATTERN)
    case_count: int = Field(ge=1, le=10_000)

    @field_validator("path")
    @classmethod
    def path_is_portable(cls, value: str) -> str:
        return _portable_relative_path(value)


class PromptIdentity(StrictModel):
    logical_name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    content_sha256: str = Field(pattern=SHA256_PATTERN)
    source_file: str | None = Field(default=None, max_length=512)

    @field_validator("source_file")
    @classmethod
    def source_is_portable(cls, value: str | None) -> str | None:
        return None if value is None else _portable_relative_path(value)


class ModelMetadata(StrictModel):
    requested: str = Field(min_length=1, max_length=200)
    backend_observed: str | None = Field(default=None, max_length=200)


class GenerationConfiguration(StrictModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class EvaluatorConfiguration(StrictModel):
    name: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=64)
    dataset_defined_expectations: bool = True


class EvaluationConfiguration(StrictModel):
    evaluator: EvaluatorConfiguration
    concurrency: int = Field(gt=0, le=256)
    timeout_seconds: float = Field(gt=0)


class RegressionGates(StrictModel):
    minimum_pass_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_pass_rate_drop: float | None = Field(default=None, ge=0, le=1)
    maximum_error_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_p95_latency_seconds: float | None = Field(default=None, gt=0)
    maximum_p95_latency_increase_percent: float | None = Field(default=None, ge=0)

    @property
    def configured(self) -> bool:
        return any(value is not None for value in self.model_dump().values())


class RegressionMetadata(StrictModel):
    baseline: str | None = Field(default=None, max_length=256)
    baseline_run_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
    )
    gates: RegressionGates = Field(default_factory=RegressionGates)
    decision: Literal["passed", "failed", "not_evaluated"]

    @model_validator(mode="after")
    def configuration_matches_decision(self) -> RegressionMetadata:
        if self.decision != "not_evaluated" and not self.gates.configured:
            raise ValueError("an evaluated decision requires regression gates")
        if self.decision != "not_evaluated" and self.baseline_run_id is None:
            raise ValueError("an evaluated decision requires a baseline run")
        if (self.baseline is None) != (self.baseline_run_id is None):
            raise ValueError(
                "baseline input and resolved baseline run ID must appear together"
            )
        return self


class EnvironmentMetadata(StrictModel):
    python_version: str = Field(min_length=1, max_length=64)
    python_implementation: str = Field(min_length=1, max_length=64)
    operating_system: str = Field(min_length=1, max_length=128)
    platform_release: str = Field(min_length=1, max_length=256)
    architecture: str = Field(min_length=1, max_length=64)
    project_version: str = Field(min_length=1, max_length=64)
    container: Literal["docker", "container", "none", "unknown"]
    dependencies: dict[str, str] = Field(max_length=16)
    environment_fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("dependencies")
    @classmethod
    def dependency_values_are_bounded(cls, value: dict[str, str]) -> dict[str, str]:
        if any(
            not name or len(name) > 64 or not version or len(version) > 64
            for name, version in value.items()
        ):
            raise ValueError("dependency names and versions must be bounded")
        return value


class DeploymentMetadata(StrictModel):
    runtime: Literal["host", "docker", "compose", "other"]
    image_reference: str | None = Field(default=None, min_length=1, max_length=256)
    configuration_name: str | None = Field(default=None, min_length=1, max_length=128)


class RAGChunkConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    chunk_size: int = Field(ge=1, le=65_536)
    overlap: int = Field(ge=0, le=65_535)
    separator_strategy: Literal["character", "line", "paragraph"]

    @model_validator(mode="after")
    def overlap_is_smaller(self) -> RAGChunkConfiguration:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class RAGEmbeddingConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    model: Literal["local-hashing-embedding"]
    model_version: Literal["1.0"]
    dimension: int = Field(ge=8, le=4096)


class RAGRetrieverConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    top_k: int = Field(ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    mmr: bool
    mmr_lambda: float = Field(ge=0, le=1)


class RAGRetrievalMetrics(StrictModel):
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    hit_rate: float = Field(ge=0, le=1)


class RAGCitationMetrics(StrictModel):
    citation_correctness: float = Field(ge=0, le=1)
    context_utilization: float = Field(ge=0, le=1)


class RAGExperimentMetadata(StrictModel):
    """RAG inputs, identities, and quality metrics bound to an experiment."""

    schema_version: Literal["1.0"] = "1.0"
    document_fingerprint: str = Field(pattern=SHA256_PATTERN)
    chunk_configuration: RAGChunkConfiguration
    chunk_fingerprints: list[str] = Field(max_length=100_000)
    embedding_configuration: RAGEmbeddingConfiguration
    index_fingerprint: str = Field(pattern=SHA256_PATTERN)
    retriever_configuration: RAGRetrieverConfiguration
    retrieval_metrics: RAGRetrievalMetrics
    citation_metrics: RAGCitationMetrics

    @field_validator("chunk_fingerprints")
    @classmethod
    def chunk_fingerprints_are_hashes(cls, value: list[str]) -> list[str]:
        if any(
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value
        ):
            raise ValueError("chunk fingerprints must be SHA-256 values")
        if len(value) != len(set(value)):
            raise ValueError("chunk fingerprints must be unique")
        return value


ArtifactKind = Literal[
    "evaluation_json",
    "evaluation_markdown",
    "regression_json",
    "summary_markdown",
    "console_summary",
]


class ArtifactMetadata(StrictModel):
    kind: ArtifactKind
    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_size: int = Field(ge=0, le=MAX_ARTIFACT_BYTES)

    @field_validator("path")
    @classmethod
    def path_is_portable(cls, value: str) -> str:
        return _portable_relative_path(value)


class AggregateResults(StrictModel):
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    p50_duration_seconds: float | None = Field(default=None, ge=0)
    p95_duration_seconds: float | None = Field(default=None, ge=0)
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)


class FailureMetadata(StrictModel):
    message: str = Field(min_length=1, max_length=240)

    @field_validator("message")
    @classmethod
    def message_is_single_line(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("failure message must not contain control characters")
        return value


class ReproductionSpecification(StrictModel):
    dataset_path: str = Field(min_length=1, max_length=512)
    requested_model: str = Field(min_length=1, max_length=200)
    base_url_input: Literal["${LLM_PLATFORM_BASE_URL}"] = "${LLM_PLATFORM_BASE_URL}"
    generation: GenerationConfiguration
    evaluation_concurrency: int = Field(gt=0, le=256)
    evaluation_timeout_seconds: float = Field(gt=0)
    baseline: str | None = Field(default=None, max_length=256)
    regression_gates: RegressionGates
    source_git_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{7,64}$")
    project_version: str = Field(min_length=1, max_length=64)
    deployment: DeploymentMetadata | None = None
    rag_metadata_path: str | None = Field(default=None, max_length=512)

    @field_validator("dataset_path", "rag_metadata_path")
    @classmethod
    def path_is_portable(cls, value: str | None) -> str | None:
        return None if value is None else _portable_relative_path(value)


class ExperimentManifest(StrictModel):
    schema_version: Literal["1.0"] = MANIFEST_SCHEMA_VERSION
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    experiment_fingerprint: str = Field(pattern=SHA256_PATTERN)
    created_at: datetime
    status: Literal["completed", "regression_failed", "failed"]
    source: SourceMetadata
    dataset: DatasetMetadata
    prompt: PromptIdentity | None = None
    model: ModelMetadata
    generation: GenerationConfiguration
    evaluation: EvaluationConfiguration
    regression: RegressionMetadata
    environment: EnvironmentMetadata
    deployment: DeploymentMetadata | None = None
    rag: RAGExperimentMetadata | None = None
    artifacts: list[ArtifactMetadata] = Field(min_length=1, max_length=16)
    aggregate_results: AggregateResults
    reproduction: ReproductionSpecification
    checksums_path: Literal["checksums.json"] = "checksums.json"
    failure: FailureMetadata | None = None

    @field_validator("created_at")
    @classmethod
    def created_at_is_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a timezone")
        return value

    @model_validator(mode="after")
    def status_matches_failure(self) -> ExperimentManifest:
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed runs require bounded failure metadata")
        if self.status != "failed" and self.failure is not None:
            raise ValueError("only failed runs may contain failure metadata")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique")
        kinds = [artifact.kind for artifact in self.artifacts]
        if len(kinds) != len(set(kinds)):
            raise ValueError("artifact kinds must be unique")
        required_paths = {"regression.json", "summary.md"}
        if self.status != "failed":
            required_paths.update({"evaluation.json", "evaluation.md"})
        if not required_paths.issubset(paths):
            raise ValueError("manifest is missing required status artifacts")
        if self.status == "regression_failed" and self.regression.decision != "failed":
            raise ValueError("regression_failed status requires a failed decision")
        if self.status == "completed":
            expected = "passed" if self.regression.gates.configured else "not_evaluated"
            if self.regression.decision != expected:
                raise ValueError(
                    "completed status does not match its regression configuration"
                )
        return self


class Difference(StrictModel):
    field: str = Field(min_length=1, max_length=128)
    left: object | None = None
    right: object | None = None


DifferenceCategory = Literal[
    "input_configuration",
    "source_code",
    "environment",
    "quality",
    "performance",
    "artifacts",
]


class ExperimentComparison(StrictModel):
    schema_version: Literal["1.0"] = COMPARISON_SCHEMA_VERSION
    left_run_id: str
    right_run_id: str
    identical: bool
    differences: dict[DifferenceCategory, list[Difference]]


def _portable_relative_path(value: str) -> str:
    if not value or value.startswith(("/", "\\")) or "\\" in value:
        raise ValueError("path must be a portable relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("path must not contain empty, dot, or parent components")
    return value
