"""Experiment orchestration built on the existing evaluation package."""

from __future__ import annotations

import json
import shlex
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from llm_platform.evaluation.dataset import load_dataset
from llm_platform.evaluation.models import (
    EvaluationDataset,
    EvaluationMessage,
    EvaluationReport,
    GenerationOptions,
    GitMetadata,
    RunnerConfiguration,
)
from llm_platform.evaluation.regression import (
    RegressionThresholds,
    compare_reports,
    render_comparison_markdown,
)
from llm_platform.evaluation.reporting import build_report, render_json, render_markdown
from llm_platform.evaluation.runner import EvaluationRunner
from llm_platform.experiments.environment import (
    collect_environment,
    collect_source_metadata,
)
from llm_platform.experiments.identity import (
    create_run_id,
    experiment_fingerprint,
    sha256_bytes,
)
from llm_platform.experiments.models import (
    AggregateResults,
    DatasetMetadata,
    DeploymentMetadata,
    EnvironmentMetadata,
    EvaluationConfiguration,
    EvaluatorConfiguration,
    ExperimentManifest,
    FailureMetadata,
    GenerationConfiguration,
    ModelMetadata,
    PromptIdentity,
    RAGExperimentMetadata,
    RegressionGates,
    RegressionMetadata,
    ReproductionSpecification,
    SourceMetadata,
)
from llm_platform.experiments.registry import ArtifactPayload, ExperimentRegistry

SAFE_OPERATIONAL_FAILURE = "Evaluation did not complete successfully."
SAFE_CASE_FAILURE = "Evaluation completed with operational case errors."
MAX_PROMPT_BYTES = 32_768


class ExperimentConfiguration(BaseModel):
    """Validated configuration for one experiment execution."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    dataset_path: Path
    dataset_identifier: str | None = Field(default=None, min_length=1, max_length=128)
    base_url: HttpUrl
    requested_model: str = Field(min_length=1, max_length=200)
    maximum_concurrency: int = Field(default=1, gt=0, le=256)
    timeout_seconds: float = Field(default=120, gt=0)
    generation: GenerationConfiguration = Field(default_factory=GenerationConfiguration)
    evaluator: EvaluatorConfiguration = Field(
        default_factory=lambda: EvaluatorConfiguration(
            name="llm-platform-deterministic-lexical",
            version="1.0",
        )
    )
    baseline: str | None = Field(default=None, min_length=1, max_length=256)
    regression_gates: RegressionGates = Field(default_factory=RegressionGates)
    prompt_file: Path | None = None
    prompt_name: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_version: str | None = Field(default=None, min_length=1, max_length=64)
    deployment: DeploymentMetadata | None = None
    rag_metadata_file: Path | None = None

    @model_validator(mode="after")
    def validate_related_configuration(self) -> ExperimentConfiguration:
        url = self.base_url
        if (
            url.username is not None
            or url.password is not None
            or url.query is not None
            or url.fragment is not None
        ):
            raise ValueError(
                "base_url must not contain credentials, a query, or a fragment"
            )
        if self.baseline is None and self.regression_gates.configured:
            raise ValueError("regression gates require an explicit baseline")
        if self.baseline is not None and not self.regression_gates.configured:
            raise ValueError("a baseline requires at least one regression gate")
        prompt_fields = (
            self.prompt_file is not None,
            self.prompt_name is not None,
            self.prompt_version is not None,
        )
        if any(prompt_fields) and not all(prompt_fields):
            raise ValueError(
                "prompt_file, prompt_name, and prompt_version must be supplied together"
            )
        return self


class ExperimentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manifest: ExperimentManifest
    exit_code: int = Field(ge=0, le=5)


class ExperimentRunner:
    """Collect metadata, reuse evaluation, and atomically register its artifacts."""

    def __init__(
        self,
        registry: ExperimentRegistry,
        *,
        client: httpx.AsyncClient | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        run_id_factory: Callable[[str], str] = create_run_id,
        source_collector: Callable[[], SourceMetadata] = collect_source_metadata,
        environment_collector: Callable[[], EnvironmentMetadata] = collect_environment,
    ) -> None:
        self.registry = registry
        self._client = client
        self._now = now
        self._run_id_factory = run_id_factory
        self._source_collector = source_collector
        self._environment_collector = environment_collector

    async def run(self, configuration: ExperimentConfiguration) -> ExperimentRunResult:
        dataset = load_dataset(configuration.dataset_path)
        baseline_run_id = (
            self.registry.resolve(configuration.baseline)
            if configuration.baseline is not None
            else None
        )
        if baseline_run_id is not None:
            baseline_manifest = self.registry.verify(baseline_run_id)
            if not any(
                artifact.path == "evaluation.json"
                for artifact in baseline_manifest.artifacts
            ):
                raise ValueError("baseline run has no evaluation report")
        source = self._source_collector()
        environment = self._environment_collector()
        rag = _load_rag_metadata(configuration.rag_metadata_file)
        created_at = self._now().astimezone(UTC)
        dataset_metadata = DatasetMetadata(
            identifier=configuration.dataset_identifier
            or configuration.dataset_path.stem,
            path=_portable_path(configuration.dataset_path),
            sha256=dataset.content_hash,
            case_count=len(dataset.cases),
        )
        prompt, evaluated_dataset = _prepare_dataset(dataset, configuration)
        evaluation_configuration = EvaluationConfiguration(
            evaluator=configuration.evaluator,
            concurrency=configuration.maximum_concurrency,
            timeout_seconds=configuration.timeout_seconds,
        )
        fingerprint = experiment_fingerprint(
            dataset=dataset_metadata,
            requested_model=configuration.requested_model,
            generation=configuration.generation,
            evaluation=evaluation_configuration,
            regression_gates=configuration.regression_gates,
            baseline=baseline_run_id,
            prompt=prompt,
            source=source,
            deployment=configuration.deployment,
            rag=rag,
        )
        run_id = self._run_id_factory(fingerprint)

        runner_configuration = RunnerConfiguration(
            base_url=configuration.base_url,
            model=configuration.requested_model,
            request_timeout_seconds=configuration.timeout_seconds,
            maximum_concurrency=configuration.maximum_concurrency,
        )
        try:
            results = await EvaluationRunner(
                runner_configuration, client=self._client
            ).run(evaluated_dataset)
            report_dataset = dataset.model_copy(update={"path": dataset_metadata.path})
            report = build_report(
                report_dataset,
                runner_configuration,
                results,
                now=lambda: created_at,
                run_id_factory=lambda: run_id,
                git_metadata=_evaluation_git(source),
            )
        except Exception:
            return self._register_operational_failure(
                configuration=configuration,
                source=source,
                environment=environment,
                dataset=dataset_metadata,
                prompt=prompt,
                evaluation=evaluation_configuration,
                fingerprint=fingerprint,
                run_id=run_id,
                created_at=created_at,
                baseline_run_id=baseline_run_id,
                rag=rag,
                message=SAFE_OPERATIONAL_FAILURE,
            )

        return self._register_report(
            configuration=configuration,
            source=source,
            environment=environment,
            dataset=dataset_metadata,
            prompt=prompt,
            evaluation=evaluation_configuration,
            fingerprint=fingerprint,
            run_id=run_id,
            created_at=created_at,
            report=report,
            baseline_run_id=baseline_run_id,
            rag=rag,
        )

    def _register_report(
        self,
        *,
        configuration: ExperimentConfiguration,
        source: SourceMetadata,
        environment: EnvironmentMetadata,
        dataset: DatasetMetadata,
        prompt: PromptIdentity | None,
        evaluation: EvaluationConfiguration,
        fingerprint: str,
        run_id: str,
        created_at: datetime,
        report: EvaluationReport,
        baseline_run_id: str | None,
        rag: RAGExperimentMetadata | None,
    ) -> ExperimentRunResult:
        artifacts: list[ArtifactPayload] = [
            ArtifactPayload(
                "evaluation_json", "evaluation.json", render_json(report).encode()
            )
        ]
        regression_decision = "not_evaluated"
        regression_payload: dict[str, object] = {
            "schema_version": "1.0",
            "baseline": configuration.baseline,
            "baseline_run_id": baseline_run_id,
            "decision": regression_decision,
            "comparison": None,
        }
        regression_markdown: str | None = None
        status = "completed"
        exit_code = 0
        failure: FailureMetadata | None = None

        if report.aggregate.error_cases:
            status = "failed"
            exit_code = 3
            failure = FailureMetadata(message=SAFE_CASE_FAILURE)
        elif configuration.baseline is not None:
            baseline_path = self.registry.artifact_path(
                baseline_run_id or configuration.baseline, "evaluation.json"
            )
            baseline_report = EvaluationReport.model_validate_json(
                baseline_path.read_bytes()
            )
            thresholds = RegressionThresholds.model_validate(
                configuration.regression_gates.model_dump()
            )
            comparison = compare_reports(report, baseline_report, thresholds)
            regression_decision = "passed" if comparison.passed else "failed"
            regression_payload["decision"] = regression_decision
            regression_payload["comparison"] = comparison.model_dump(mode="json")
            regression_markdown = render_comparison_markdown(comparison)
            if not comparison.passed:
                status = "regression_failed"
                exit_code = 1

        artifacts.extend(
            [
                ArtifactPayload(
                    "evaluation_markdown",
                    "evaluation.md",
                    render_markdown(
                        report, regression_markdown=regression_markdown
                    ).encode(),
                ),
                ArtifactPayload(
                    "regression_json",
                    "regression.json",
                    _json_bytes(regression_payload),
                ),
            ]
        )
        regression = RegressionMetadata(
            baseline=configuration.baseline,
            baseline_run_id=baseline_run_id,
            gates=configuration.regression_gates,
            decision=regression_decision,
        )
        aggregate = _aggregate(report)
        summary = _summary_markdown(
            run_id=run_id,
            fingerprint=fingerprint,
            status=status,
            aggregate=aggregate,
            configuration=configuration,
            source=source,
            environment=environment,
            regression=regression,
        )
        artifacts.append(
            ArtifactPayload("summary_markdown", "summary.md", summary.encode())
        )
        manifest = _manifest(
            configuration=configuration,
            source=source,
            environment=environment,
            dataset=dataset,
            prompt=prompt,
            evaluation=evaluation,
            fingerprint=fingerprint,
            run_id=run_id,
            created_at=created_at,
            status=status,
            regression=regression,
            aggregate=aggregate,
            artifacts=artifacts,
            backend_model=_observed_backend_model(report),
            failure=failure,
            rag=rag,
        )
        self.registry.register(manifest, artifacts)
        return ExperimentRunResult(manifest=manifest, exit_code=exit_code)

    def _register_operational_failure(
        self,
        *,
        configuration: ExperimentConfiguration,
        source: SourceMetadata,
        environment: EnvironmentMetadata,
        dataset: DatasetMetadata,
        prompt: PromptIdentity | None,
        evaluation: EvaluationConfiguration,
        fingerprint: str,
        run_id: str,
        created_at: datetime,
        baseline_run_id: str | None,
        rag: RAGExperimentMetadata | None,
        message: str,
    ) -> ExperimentRunResult:
        regression = RegressionMetadata(
            baseline=configuration.baseline,
            baseline_run_id=baseline_run_id,
            gates=configuration.regression_gates,
            decision="not_evaluated",
        )
        aggregate = AggregateResults()
        summary = _summary_markdown(
            run_id=run_id,
            fingerprint=fingerprint,
            status="failed",
            aggregate=aggregate,
            configuration=configuration,
            source=source,
            environment=environment,
            regression=regression,
            failure=message,
        )
        artifacts = [
            ArtifactPayload(
                "regression_json",
                "regression.json",
                _json_bytes(
                    {
                        "schema_version": "1.0",
                        "baseline": configuration.baseline,
                        "baseline_run_id": baseline_run_id,
                        "decision": "not_evaluated",
                        "comparison": None,
                    }
                ),
            ),
            ArtifactPayload("summary_markdown", "summary.md", summary.encode()),
        ]
        manifest = _manifest(
            configuration=configuration,
            source=source,
            environment=environment,
            dataset=dataset,
            prompt=prompt,
            evaluation=evaluation,
            fingerprint=fingerprint,
            run_id=run_id,
            created_at=created_at,
            status="failed",
            regression=regression,
            aggregate=aggregate,
            artifacts=artifacts,
            backend_model=None,
            failure=FailureMetadata(message=message),
            rag=rag,
        )
        self.registry.register(manifest, artifacts)
        return ExperimentRunResult(manifest=manifest, exit_code=3)


def _prepare_dataset(
    dataset: EvaluationDataset, configuration: ExperimentConfiguration
) -> tuple[PromptIdentity | None, EvaluationDataset]:
    prompt: PromptIdentity | None = None
    prompt_content: str | None = None
    if configuration.prompt_file is not None:
        try:
            content = configuration.prompt_file.read_bytes()
        except OSError as exc:
            raise ValueError("could not read shared prompt file") from exc
        if len(content) > MAX_PROMPT_BYTES:
            raise ValueError(f"shared prompt exceeds {MAX_PROMPT_BYTES} byte limit")
        try:
            prompt_content = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("shared prompt must be valid UTF-8") from exc
        if not prompt_content:
            raise ValueError("shared prompt must not be empty")
        assert configuration.prompt_name is not None
        assert configuration.prompt_version is not None
        prompt = PromptIdentity(
            logical_name=configuration.prompt_name,
            version=configuration.prompt_version,
            content_sha256=sha256_bytes(content),
            source_file=_portable_path(configuration.prompt_file),
        )

    cases = []
    for case in dataset.cases:
        messages = list(case.messages)
        if prompt_content is not None:
            messages.insert(0, EvaluationMessage(role="system", content=prompt_content))
        generation = GenerationOptions(
            temperature=(
                case.generation.temperature
                if case.generation.temperature is not None
                else configuration.generation.temperature
            ),
            max_tokens=(
                case.generation.max_tokens
                if case.generation.max_tokens is not None
                else configuration.generation.max_tokens
            ),
        )
        cases.append(
            case.model_copy(update={"messages": messages, "generation": generation})
        )
    return prompt, dataset.model_copy(update={"cases": cases})


def _manifest(
    *,
    configuration: ExperimentConfiguration,
    source: SourceMetadata,
    environment: EnvironmentMetadata,
    dataset: DatasetMetadata,
    prompt: PromptIdentity | None,
    evaluation: EvaluationConfiguration,
    fingerprint: str,
    run_id: str,
    created_at: datetime,
    status: str,
    regression: RegressionMetadata,
    aggregate: AggregateResults,
    artifacts: list[ArtifactPayload],
    backend_model: str | None,
    failure: FailureMetadata | None,
    rag: RAGExperimentMetadata | None,
) -> ExperimentManifest:
    return ExperimentManifest.model_validate(
        {
            "run_id": run_id,
            "experiment_fingerprint": fingerprint,
            "created_at": created_at,
            "status": status,
            "source": source,
            "dataset": dataset,
            "prompt": prompt,
            "model": ModelMetadata(
                requested=configuration.requested_model,
                backend_observed=backend_model,
            ),
            "generation": configuration.generation,
            "evaluation": evaluation,
            "regression": regression,
            "environment": environment,
            "deployment": configuration.deployment,
            "rag": rag,
            "artifacts": [artifact.metadata() for artifact in artifacts],
            "aggregate_results": aggregate,
            "reproduction": ReproductionSpecification(
                dataset_path=dataset.path,
                requested_model=configuration.requested_model,
                generation=configuration.generation,
                evaluation_concurrency=configuration.maximum_concurrency,
                evaluation_timeout_seconds=configuration.timeout_seconds,
                baseline=regression.baseline_run_id,
                regression_gates=configuration.regression_gates,
                source_git_commit=source.git_commit,
                project_version=environment.project_version,
                deployment=configuration.deployment,
                rag_metadata_path=(
                    _portable_path(configuration.rag_metadata_file)
                    if configuration.rag_metadata_file is not None
                    else None
                ),
            ),
            "failure": failure,
        }
    )


def _evaluation_git(source: SourceMetadata) -> GitMetadata | None:
    if source.git_commit is None or source.git_dirty is None:
        return None
    return GitMetadata(commit_hash=source.git_commit, dirty=source.git_dirty)


def _aggregate(report: EvaluationReport) -> AggregateResults:
    aggregate = report.aggregate
    return AggregateResults(
        pass_rate=aggregate.pass_rate,
        error_rate=aggregate.error_rate,
        p50_duration_seconds=aggregate.p50_request_duration_seconds,
        p95_duration_seconds=aggregate.p95_request_duration_seconds,
        prompt_tokens=aggregate.total_prompt_tokens,
        completion_tokens=aggregate.total_completion_tokens,
    )


def _observed_backend_model(report: EvaluationReport) -> str | None:
    observed = {
        case.measurements.backend_model
        for case in report.cases
        if case.measurements.backend_model is not None
    }
    if len(observed) == 1:
        return next(iter(observed))
    return "multiple" if observed else None


def _portable_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
    except (OSError, ValueError):
        return path.name
    value = relative.as_posix()
    return value if value and value != "." else path.name


def _summary_markdown(
    *,
    run_id: str,
    fingerprint: str,
    status: str,
    aggregate: AggregateResults,
    configuration: ExperimentConfiguration,
    source: SourceMetadata,
    environment: EnvironmentMetadata,
    regression: RegressionMetadata,
    failure: str | None = None,
) -> str:
    command = [
        "llm-experiment run",
        f"--dataset {shlex.quote(_portable_path(configuration.dataset_path))}",
        '--base-url "$LLM_PLATFORM_BASE_URL"',
        f"--model {shlex.quote(configuration.requested_model)}",
        f"--max-concurrency {configuration.maximum_concurrency}",
        f"--timeout-seconds {configuration.timeout_seconds:g}",
    ]
    if configuration.generation.temperature is not None:
        command.append(f"--temperature {configuration.generation.temperature:g}")
    if configuration.generation.max_tokens is not None:
        command.append(f"--max-tokens {configuration.generation.max_tokens}")
    if regression.baseline_run_id is not None:
        command.append(f"--baseline {shlex.quote(regression.baseline_run_id)}")
    gate_arguments = (
        ("--min-pass-rate", regression.gates.minimum_pass_rate),
        ("--max-pass-rate-drop", regression.gates.maximum_pass_rate_drop),
        ("--max-error-rate", regression.gates.maximum_error_rate),
        (
            "--max-p95-latency-seconds",
            regression.gates.maximum_p95_latency_seconds,
        ),
        (
            "--max-p95-latency-increase-percent",
            regression.gates.maximum_p95_latency_increase_percent,
        ),
    )
    for option, value in gate_arguments:
        if value is not None:
            command.append(f"{option} {value:g}")
    if configuration.prompt_file is not None:
        prompt_path = shlex.quote(_portable_path(configuration.prompt_file))
        command.extend(
            [
                f"--prompt-file {prompt_path}",
                f"--prompt-name {shlex.quote(configuration.prompt_name or '')}",
                f"--prompt-version {shlex.quote(configuration.prompt_version or '')}",
            ]
        )
    if configuration.deployment is not None:
        command.append(f"--deployment-runtime {configuration.deployment.runtime}")
        if configuration.deployment.image_reference is not None:
            command.append(
                "--deployment-image "
                f"{shlex.quote(configuration.deployment.image_reference)}"
            )
        if configuration.deployment.configuration_name is not None:
            command.append(
                "--deployment-name "
                f"{shlex.quote(configuration.deployment.configuration_name)}"
            )
    if configuration.rag_metadata_file is not None:
        rag_path = _portable_path(configuration.rag_metadata_file)
        command.append(f"--rag-metadata {shlex.quote(rag_path)}")
    command_separator = " \\" + "\n  "
    lines = [
        "# Experiment Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Experiment fingerprint: `{fingerprint}`",
        f"- Status: `{status}`",
        f"- Git commit: `{source.git_commit or 'unavailable'}`",
        f"- Environment fingerprint: `{environment.environment_fingerprint}`",
        f"- Pass rate: `{aggregate.pass_rate}`",
        f"- Error rate: `{aggregate.error_rate}`",
        f"- Regression decision: `{regression.decision}`",
    ]
    if failure is not None:
        lines.append(f"- Failure: {failure}")
    lines.extend(
        [
            "",
            "## Configuration reproduction",
            "",
            "```bash",
            command_separator.join(command),
            "```",
            "",
            "This reconstructs configuration, not bit-for-bit model output. The "
            "manifest records execution-environment traceability and report "
            "structure; model generation may remain nondeterministic.",
            "",
        ]
    )
    return "\n".join(lines)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()


def _load_rag_metadata(path: Path | None) -> RAGExperimentMetadata | None:
    if path is None:
        return None
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise ValueError("could not read RAG metadata file") from exc
    if len(content) > 16 * 1024 * 1024:
        raise ValueError("RAG metadata exceeds 16777216 byte limit")
    try:
        return RAGExperimentMetadata.model_validate_json(content)
    except ValueError as exc:
        raise ValueError("RAG metadata is malformed") from exc
