from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_platform.evaluation.models import EvaluationReport
from llm_platform.evaluation.regression import (
    RegressionThresholds,
    compare_reports,
)
from tests.evaluation_helpers import completed_result, error_result, evaluation_report


def _reports() -> tuple[EvaluationReport, EvaluationReport]:
    baseline = evaluation_report(
        *(completed_result(f"b-{index}", duration=1) for index in range(10)),
        run_id="baseline",
    )
    current = evaluation_report(
        *(
            [
                *(completed_result(f"c-{index}", duration=1.5) for index in range(8)),
                completed_result("failed", passed=False, duration=1.5),
                error_result(duration=1.5),
            ]
        ),
        run_id="current",
    )
    return current, baseline


@pytest.mark.parametrize(
    ("thresholds", "gate_name"),
    [
        ({"minimum_pass_rate": 0.9}, "minimum_pass_rate"),
        ({"maximum_pass_rate_drop": 0.1}, "maximum_pass_rate_drop"),
        ({"maximum_error_rate": 0.05}, "maximum_error_rate"),
        (
            {"maximum_p95_latency_seconds": 1.0},
            "maximum_p95_latency_seconds",
        ),
        (
            {"maximum_p95_latency_increase_percent": 20},
            "maximum_p95_latency_increase_percent",
        ),
    ],
)
def test_each_regression_gate_can_fail(
    thresholds: dict[str, float], gate_name: str
) -> None:
    current, baseline = _reports()
    comparison = compare_reports(
        current,
        baseline,
        RegressionThresholds.model_validate(thresholds),
    )
    assert not comparison.passed
    assert comparison.gates[0].name == gate_name
    assert not comparison.gates[0].passed


def test_multiple_failures_are_all_reported() -> None:
    current, baseline = _reports()
    comparison = compare_reports(
        current,
        baseline,
        RegressionThresholds(
            minimum_pass_rate=0.95,
            maximum_error_rate=0.01,
            maximum_p95_latency_seconds=1.0,
        ),
    )
    assert len(comparison.gates) == 3
    assert all(not gate.passed for gate in comparison.gates)


def test_successful_comparison() -> None:
    current, baseline = _reports()
    comparison = compare_reports(
        current,
        baseline,
        RegressionThresholds(
            minimum_pass_rate=0.7,
            maximum_pass_rate_drop=0.3,
            maximum_error_rate=0.2,
            maximum_p95_latency_seconds=2,
            maximum_p95_latency_increase_percent=60,
        ),
    )
    assert comparison.passed


def test_missing_baseline_metric_fails_required_relative_gate() -> None:
    current = evaluation_report(completed_result("current"))
    baseline = evaluation_report()
    comparison = compare_reports(
        current,
        baseline,
        RegressionThresholds(maximum_p95_latency_increase_percent=20),
    )
    assert not comparison.passed
    assert "unavailable" in comparison.gates[0].explanation


def test_invalid_or_empty_thresholds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        RegressionThresholds()
    with pytest.raises(ValidationError):
        RegressionThresholds(minimum_pass_rate=1.1)
