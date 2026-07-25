"""Machine- and human-readable evaluation reporting."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from llm_platform.evaluation.models import (
    AggregateMetrics,
    CaseResult,
    EvaluationDataset,
    EvaluationReport,
    GitMetadata,
    PlatformConfiguration,
    RunnerConfiguration,
)


def nearest_rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    """Return the nearest-rank percentile (rank = ceil(p * N))."""

    if not values:
        return None
    if not 0 < percentile <= 1:
        raise ValueError("percentile must be in (0, 1]")
    ordered = sorted(values)
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def aggregate_results(results: Sequence[CaseResult]) -> AggregateMetrics:
    total = len(results)
    completed = sum(result.status == "completed" for result in results)
    passed = sum(result.passed for result in results)
    errors = sum(result.status == "error" for result in results)
    failed = completed - passed
    durations = [result.measurements.duration_seconds for result in results]
    return AggregateMetrics(
        total_cases=total,
        completed_cases=completed,
        passed_cases=passed,
        failed_cases=failed,
        error_cases=errors,
        pass_rate=passed / total if total else None,
        error_rate=errors / total if total else None,
        average_request_duration_seconds=(
            sum(durations) / len(durations) if durations else None
        ),
        p50_request_duration_seconds=nearest_rank_percentile(durations, 0.50),
        p95_request_duration_seconds=nearest_rank_percentile(durations, 0.95),
        total_prompt_tokens=sum(
            result.measurements.prompt_tokens or 0 for result in results
        ),
        total_completion_tokens=sum(
            result.measurements.completion_tokens or 0 for result in results
        ),
    )


def build_report(
    dataset: EvaluationDataset,
    configuration: RunnerConfiguration,
    results: list[CaseResult],
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    run_id_factory: Callable[[], str] = lambda: str(uuid4()),
    git_metadata: GitMetadata | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        run_id=run_id_factory(),
        timestamp=now().isoformat(),
        dataset_path=dataset.path,
        dataset_content_hash=dataset.content_hash,
        platform=PlatformConfiguration(
            base_url=configuration.base_url,
            request_timeout_seconds=configuration.request_timeout_seconds,
            maximum_concurrency=configuration.maximum_concurrency,
        ),
        model_requested=configuration.model,
        aggregate=aggregate_results(results),
        cases=results,
        git=git_metadata,
    )


def detect_git_metadata(repository: Path | None = None) -> GitMetadata | None:
    """Return bounded Git provenance, or None outside a usable repository."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if not commit:
        return None
    return GitMetadata(commit_hash=commit, dirty=bool(dirty_result.stdout))


def write_report_files(
    report: EvaluationReport,
    output_directory: Path,
    *,
    regression_markdown: str | None = None,
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"evaluation-{report.run_id}"
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    json_path.write_text(render_json(report), encoding="utf-8")
    markdown_path.write_text(
        render_markdown(report, regression_markdown=regression_markdown),
        encoding="utf-8",
    )
    return json_path, markdown_path


def render_json(report: EvaluationReport) -> str:
    """Render the canonical human-readable JSON report representation."""

    return (
        json.dumps(
            report.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )


def render_markdown(
    report: EvaluationReport, *, regression_markdown: str | None = None
) -> str:
    aggregate = report.aggregate
    lines = [
        "# LLM Evaluation Report",
        "",
        "## Run configuration",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Timestamp: `{report.timestamp}`",
        f"- Dataset: `{report.dataset_path}`",
        f"- Dataset SHA-256: `{report.dataset_content_hash}`",
        f"- Platform: `{report.platform.base_url}`",
        f"- Model requested: `{report.model_requested}`",
        f"- Timeout: `{report.platform.request_timeout_seconds:.3f}` seconds",
        f"- Maximum concurrency: `{report.platform.maximum_concurrency}`",
        "- TTFT: unavailable for buffered evaluation requests",
        "",
        "## Aggregate quality",
        "",
        f"- Total cases: {aggregate.total_cases}",
        f"- Completed: {aggregate.completed_cases}",
        f"- Passed: {aggregate.passed_cases}",
        f"- Failed: {aggregate.failed_cases}",
        f"- Errors: {aggregate.error_cases}",
        f"- Pass rate: {_percent(aggregate.pass_rate)}",
        "",
        "## Latency",
        "",
        f"- Average: {_seconds(aggregate.average_request_duration_seconds)}",
        f"- P50 (nearest rank): {_seconds(aggregate.p50_request_duration_seconds)}",
        f"- P95 (nearest rank): {_seconds(aggregate.p95_request_duration_seconds)}",
        "",
        "## Token usage",
        "",
        f"- Prompt tokens: {aggregate.total_prompt_tokens}",
        f"- Completion tokens: {aggregate.total_completion_tokens}",
        "",
        "## Failed cases",
        "",
    ]
    failed = [
        case for case in report.cases if case.status == "completed" and not case.passed
    ]
    if not failed:
        lines.append("None.")
    else:
        for case in failed:
            reasons = [
                result.failure_reason
                for result in case.evaluator_results
                if not result.passed and result.failure_reason
            ]
            lines.append(f"- `{case.id}`: {'; '.join(reasons)}")
    lines.extend(["", "## Error cases", ""])
    errors = [case for case in report.cases if case.status == "error"]
    if not errors:
        lines.append("None.")
    else:
        for case in errors:
            assert case.error is not None
            lines.append(f"- `{case.id}` ({case.error.type}): {case.error.message}")
    if regression_markdown is not None:
        lines.extend(["", "## Regression comparison", "", regression_markdown])
    return "\n".join(lines) + "\n"


def _percent(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.2%}"


def _seconds(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6f} seconds"
