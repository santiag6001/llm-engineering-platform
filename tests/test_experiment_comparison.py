from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_platform.experiments.comparison import (
    compare_manifests,
    render_comparison_json,
    render_comparison_markdown,
    write_comparison,
)
from llm_platform.experiments.models import (
    AggregateResults,
    DatasetMetadata,
    EnvironmentMetadata,
    GenerationConfiguration,
    SourceMetadata,
)
from tests.experiment_helpers import manifest


def test_identical_runs_have_no_audited_differences() -> None:
    left, _ = manifest("left")
    right = left.model_copy(update={"run_id": "right"})
    comparison = compare_manifests(left, right)
    assert comparison.identical
    assert not any(comparison.differences.values())


@pytest.mark.parametrize(
    ("update", "category"),
    [
        (
            {
                "source": SourceMetadata(
                    git_commit="b" * 40, git_dirty=False, branch="main"
                )
            },
            "source_code",
        ),
        (
            {"generation": GenerationConfiguration(temperature=1, max_tokens=20)},
            "input_configuration",
        ),
        (
            {
                "dataset": DatasetMetadata(
                    identifier="dataset",
                    path="evaluations/dataset.jsonl",
                    sha256="b" * 64,
                    case_count=1,
                )
            },
            "input_configuration",
        ),
        (
            {
                "environment": EnvironmentMetadata(
                    python_version="3.12",
                    python_implementation="CPython",
                    operating_system="Linux",
                    platform_release="other",
                    architecture="x86_64",
                    project_version="0.1.0",
                    container="none",
                    dependencies={},
                    environment_fingerprint="b" * 64,
                )
            },
            "environment",
        ),
        (
            {
                "aggregate_results": AggregateResults(
                    pass_rate=0,
                    error_rate=1,
                    p50_duration_seconds=2,
                    p95_duration_seconds=3,
                )
            },
            "quality",
        ),
    ],
)
def test_differences_are_classified(update: dict[str, object], category: str) -> None:
    left, _ = manifest("left")
    right = left.model_copy(update={"run_id": "right", **update})
    comparison = compare_manifests(left, right)
    assert not comparison.identical
    assert comparison.differences[category]  # type: ignore[index]


def test_json_markdown_and_files_are_produced(tmp_path: Path) -> None:
    left, _ = manifest("left")
    right = left.model_copy(
        update={
            "run_id": "right",
            "generation": GenerationConfiguration(temperature=1, max_tokens=20),
        }
    )
    comparison = compare_manifests(left, right)
    decoded = json.loads(render_comparison_json(comparison))
    markdown = render_comparison_markdown(comparison)
    json_path, markdown_path = write_comparison(comparison, tmp_path)
    assert decoded["left_run_id"] == "left"
    assert "Input and configuration changes" in markdown
    assert json_path.exists()
    assert markdown_path.exists()


def test_performance_only_difference_is_classified() -> None:
    left, _ = manifest("left")
    aggregate = left.aggregate_results.model_copy(
        update={"p50_duration_seconds": 2, "p95_duration_seconds": 3}
    )
    right = left.model_copy(update={"run_id": "right", "aggregate_results": aggregate})
    comparison = compare_manifests(left, right)
    assert comparison.differences["performance"]
    assert not comparison.differences["quality"]
