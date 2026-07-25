from __future__ import annotations

import json
from pathlib import Path

from llm_platform.evaluation.reporting import (
    aggregate_results,
    detect_git_metadata,
    nearest_rank_percentile,
    render_markdown,
    write_report_files,
)
from tests.evaluation_helpers import (
    completed_result,
    error_result,
    evaluation_report,
)


def test_nearest_rank_percentile_is_deterministic() -> None:
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert nearest_rank_percentile(values, 0.50) == 3.0
    assert nearest_rank_percentile(values, 0.95) == 5.0
    assert nearest_rank_percentile([], 0.95) is None


def test_aggregate_metrics_count_quality_latency_and_tokens() -> None:
    aggregate = aggregate_results(
        [
            completed_result("pass", duration=1, passed=True),
            completed_result("fail", duration=2, passed=False),
            error_result(duration=3),
        ]
    )
    assert aggregate.total_cases == 3
    assert aggregate.completed_cases == 2
    assert aggregate.passed_cases == 1
    assert aggregate.failed_cases == 1
    assert aggregate.error_cases == 1
    assert aggregate.pass_rate == 1 / 3
    assert aggregate.error_rate == 1 / 3
    assert aggregate.average_request_duration_seconds == 2
    assert aggregate.p95_request_duration_seconds == 3
    assert aggregate.total_prompt_tokens == 4
    assert aggregate.total_completion_tokens == 6


def test_json_and_markdown_reports_have_required_fields(tmp_path: Path) -> None:
    report = evaluation_report(completed_result("failed", passed=False), error_result())
    json_path, markdown_path = write_report_files(report, tmp_path)
    decoded = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")

    assert decoded["report_schema_version"] == "1.0"
    assert decoded["dataset_content_hash"] == "a" * 64
    assert decoded["platform"]["maximum_concurrency"] == 2
    assert decoded["cases"][0]["response_preview"] == "answer"
    assert "# LLM Evaluation Report" in markdown
    assert "## Failed cases" in markdown
    assert "## Error cases" in markdown
    assert "TTFT: unavailable" in markdown


def test_markdown_can_include_regression_result() -> None:
    markdown = render_markdown(
        evaluation_report(completed_result()), regression_markdown="Overall: **PASS**"
    )
    assert "## Regression comparison" in markdown
    assert "Overall: **PASS**" in markdown


def test_git_metadata_is_optional_outside_repository(tmp_path: Path) -> None:
    assert detect_git_metadata(tmp_path) is None
