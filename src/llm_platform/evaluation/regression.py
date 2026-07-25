"""Deterministic report regression gates."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llm_platform.evaluation.models import EvaluationReport


class RegressionThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    minimum_pass_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_pass_rate_drop: float | None = Field(default=None, ge=0, le=1)
    maximum_error_rate: float | None = Field(default=None, ge=0, le=1)
    maximum_p95_latency_seconds: float | None = Field(default=None, gt=0)
    maximum_p95_latency_increase_percent: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def require_gate(self) -> RegressionThresholds:
        if all(value is None for value in self.model_dump().values()):
            raise ValueError("at least one regression gate is required")
        return self


class GateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    current_value: float | None
    baseline_value: float | None
    threshold: float
    explanation: str


class RegressionComparison(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    gates: list[GateResult]


def compare_reports(
    current: EvaluationReport,
    baseline: EvaluationReport,
    thresholds: RegressionThresholds,
) -> RegressionComparison:
    gates: list[GateResult] = []
    current_aggregate = current.aggregate
    baseline_aggregate = baseline.aggregate

    if thresholds.minimum_pass_rate is not None:
        gates.append(
            _maximum_or_minimum_gate(
                name="minimum_pass_rate",
                current=current_aggregate.pass_rate,
                baseline=None,
                threshold=thresholds.minimum_pass_rate,
                minimum=True,
            )
        )
    if thresholds.maximum_pass_rate_drop is not None:
        current_rate = current_aggregate.pass_rate
        baseline_rate = baseline_aggregate.pass_rate
        if current_rate is None or baseline_rate is None:
            gates.append(
                _missing_gate(
                    "maximum_pass_rate_drop",
                    current_rate,
                    baseline_rate,
                    thresholds.maximum_pass_rate_drop,
                )
            )
        else:
            drop = baseline_rate - current_rate
            passed = drop <= thresholds.maximum_pass_rate_drop
            gates.append(
                GateResult(
                    name="maximum_pass_rate_drop",
                    passed=passed,
                    current_value=current_rate,
                    baseline_value=baseline_rate,
                    threshold=thresholds.maximum_pass_rate_drop,
                    explanation=(
                        f"pass-rate drop {drop:.6f} "
                        f"{'did not exceed' if passed else 'exceeded'} the limit"
                    ),
                )
            )
    if thresholds.maximum_error_rate is not None:
        gates.append(
            _maximum_or_minimum_gate(
                name="maximum_error_rate",
                current=current_aggregate.error_rate,
                baseline=None,
                threshold=thresholds.maximum_error_rate,
                minimum=False,
            )
        )
    if thresholds.maximum_p95_latency_seconds is not None:
        gates.append(
            _maximum_or_minimum_gate(
                name="maximum_p95_latency_seconds",
                current=current_aggregate.p95_request_duration_seconds,
                baseline=None,
                threshold=thresholds.maximum_p95_latency_seconds,
                minimum=False,
            )
        )
    if thresholds.maximum_p95_latency_increase_percent is not None:
        current_p95 = current_aggregate.p95_request_duration_seconds
        baseline_p95 = baseline_aggregate.p95_request_duration_seconds
        threshold = thresholds.maximum_p95_latency_increase_percent
        if current_p95 is None or baseline_p95 is None or baseline_p95 <= 0:
            gates.append(
                _missing_gate(
                    "maximum_p95_latency_increase_percent",
                    current_p95,
                    baseline_p95,
                    threshold,
                )
            )
        else:
            increase = ((current_p95 - baseline_p95) / baseline_p95) * 100
            passed = increase <= threshold
            gates.append(
                GateResult(
                    name="maximum_p95_latency_increase_percent",
                    passed=passed,
                    current_value=current_p95,
                    baseline_value=baseline_p95,
                    threshold=threshold,
                    explanation=(
                        f"P95 increase {increase:.6f}% "
                        f"{'did not exceed' if passed else 'exceeded'} the limit"
                    ),
                )
            )
    return RegressionComparison(
        passed=all(gate.passed for gate in gates),
        gates=gates,
    )


def render_comparison_markdown(comparison: RegressionComparison) -> str:
    lines = [
        f"Overall: **{'PASS' if comparison.passed else 'FAIL'}**",
        "",
        "| Gate | Current | Baseline | Threshold | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for gate in comparison.gates:
        lines.append(
            f"| {gate.name} | {_value(gate.current_value)} | "
            f"{_value(gate.baseline_value)} | {gate.threshold:.6g} | "
            f"{'PASS' if gate.passed else 'FAIL'} |"
        )
    lines.extend(["", *[f"- {gate.explanation}" for gate in comparison.gates]])
    return "\n".join(lines)


def _maximum_or_minimum_gate(
    *,
    name: str,
    current: float | None,
    baseline: float | None,
    threshold: float,
    minimum: bool,
) -> GateResult:
    if current is None:
        return _missing_gate(name, current, baseline, threshold)
    passed = current >= threshold if minimum else current <= threshold
    relation = "met" if passed else "failed"
    return GateResult(
        name=name,
        passed=passed,
        current_value=current,
        baseline_value=baseline,
        threshold=threshold,
        explanation=f"current value {relation} the configured threshold",
    )


def _missing_gate(
    name: str,
    current: float | None,
    baseline: float | None,
    threshold: float,
) -> GateResult:
    return GateResult(
        name=name,
        passed=False,
        current_value=current,
        baseline_value=baseline,
        threshold=threshold,
        explanation="required current or baseline metric was unavailable",
    )


def _value(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"
