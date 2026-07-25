"""Strict models for evaluation inputs and report output."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

REPORT_SCHEMA_VERSION: Literal["1.0"] = "1.0"
DATASET_SCHEMA_VERSION: Literal["1.0"] = "1.0"
PREVIEW_MAX_CHARACTERS = 240
ExpectedString = Annotated[str, Field(min_length=1, max_length=256)]


class StrictModel(BaseModel):
    """Forbid schema drift at persisted-data boundaries."""

    model_config = ConfigDict(extra="forbid")


class EvaluationMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32_768)


class ExpectedResponse(StrictModel):
    exact_match: str | None = Field(default=None, min_length=1, max_length=4_096)
    contains_all: list[ExpectedString] | None = Field(
        default=None, min_length=1, max_length=32
    )
    contains_any: list[ExpectedString] | None = Field(
        default=None, min_length=1, max_length=32
    )
    not_contains: list[ExpectedString] | None = Field(
        default=None, min_length=1, max_length=32
    )
    minimum_characters: int | None = Field(default=None, ge=0)
    maximum_characters: int | None = Field(default=None, ge=0)
    case_sensitive: bool = False
    normalize_whitespace: bool = True

    @model_validator(mode="after")
    def validate_evaluators(self) -> ExpectedResponse:
        configured = (
            self.exact_match is not None
            or bool(self.contains_all)
            or bool(self.contains_any)
            or bool(self.not_contains)
            or self.minimum_characters is not None
            or self.maximum_characters is not None
        )
        if not configured:
            raise ValueError("at least one expected-response evaluator is required")
        if (
            self.minimum_characters is not None
            and self.maximum_characters is not None
            and self.minimum_characters > self.maximum_characters
        ):
            raise ValueError("minimum_characters must not exceed maximum_characters")
        return self


class GenerationOptions(StrictModel):
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)


class CaseMetadata(StrictModel):
    description: str | None = Field(default=None, max_length=500)
    tags: list[str] | None = Field(default=None, max_length=16)

    @model_validator(mode="after")
    def validate_tags(self) -> CaseMetadata:
        if self.tags is not None:
            if any(not tag or len(tag) > 64 for tag in self.tags):
                raise ValueError("tags must be non-empty and at most 64 characters")
            if len(set(self.tags)) != len(self.tags):
                raise ValueError("tags must be unique")
        return self


class EvaluationCase(StrictModel):
    schema_version: Literal["1.0"]
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    category: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    messages: list[EvaluationMessage] = Field(min_length=1, max_length=128)
    expected: ExpectedResponse
    generation: GenerationOptions = Field(default_factory=GenerationOptions)
    metadata: CaseMetadata = Field(default_factory=CaseMetadata)


class EvaluationDataset(StrictModel):
    path: str
    content_hash: str
    cases: list[EvaluationCase]


class EvaluatorResult(StrictModel):
    name: str = Field(max_length=64)
    passed: bool
    score: float | None = Field(default=None, ge=0, le=1)
    failure_reason: str | None = Field(default=None, max_length=240)
    expected_summary: str = Field(max_length=240)
    actual_summary: str = Field(max_length=240)


class RequestMeasurements(StrictModel):
    duration_seconds: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    backend_model: str | None = Field(default=None, max_length=200)
    finish_reason: str | None = Field(default=None, max_length=100)
    ttft_seconds: None = None


ErrorType = Literal[
    "platform_http_error",
    "timeout",
    "connection_failure",
    "malformed_platform_response",
    "evaluator_failure",
]


class EvaluationError(StrictModel):
    type: ErrorType
    message: str = Field(max_length=240)
    http_status: int | None = None


class CaseResult(StrictModel):
    id: str
    category: str
    status: Literal["completed", "error"]
    passed: bool
    response_preview: str | None = Field(default=None, max_length=240)
    measurements: RequestMeasurements
    evaluator_results: list[EvaluatorResult] = Field(default_factory=list)
    error: EvaluationError | None = None


class AggregateMetrics(StrictModel):
    total_cases: int = Field(ge=0)
    completed_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    error_cases: int = Field(ge=0)
    pass_rate: float | None = Field(default=None, ge=0, le=1)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    average_request_duration_seconds: float | None = Field(default=None, ge=0)
    p50_request_duration_seconds: float | None = Field(default=None, ge=0)
    p95_request_duration_seconds: float | None = Field(default=None, ge=0)
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)


class PlatformConfiguration(StrictModel):
    base_url: HttpUrl
    request_timeout_seconds: float = Field(gt=0)
    maximum_concurrency: int = Field(gt=0)

    @model_validator(mode="after")
    def reject_url_credentials(self) -> PlatformConfiguration:
        if (
            self.base_url.username is not None
            or self.base_url.password is not None
            or self.base_url.query is not None
            or self.base_url.fragment is not None
        ):
            raise ValueError(
                "base_url must not contain credentials, a query, or a fragment"
            )
        return self


class GitMetadata(StrictModel):
    commit_hash: str = Field(pattern=r"^[0-9a-f]{7,64}$")
    dirty: bool


class EvaluationReport(StrictModel):
    report_schema_version: Literal["1.0"] = REPORT_SCHEMA_VERSION
    run_id: str
    timestamp: str
    dataset_path: str
    dataset_content_hash: str
    platform: PlatformConfiguration
    model_requested: str
    aggregate: AggregateMetrics
    cases: list[CaseResult]
    git: GitMetadata | None = None


class RunnerConfiguration(StrictModel):
    base_url: HttpUrl
    model: str = Field(min_length=1)
    request_timeout_seconds: float = Field(gt=0)
    maximum_concurrency: int = Field(gt=0, le=256)

    @model_validator(mode="after")
    def reject_url_credentials(self) -> RunnerConfiguration:
        if (
            self.base_url.username is not None
            or self.base_url.password is not None
            or self.base_url.query is not None
            or self.base_url.fragment is not None
        ):
            raise ValueError(
                "base_url must not contain credentials, a query, or a fragment"
            )
        return self
