"""Deterministic response evaluators."""

from __future__ import annotations

from llm_platform.evaluation.models import (
    PREVIEW_MAX_CHARACTERS,
    EvaluationCase,
    EvaluatorResult,
)


def bounded_preview(text: str, limit: int = PREVIEW_MAX_CHARACTERS) -> str:
    """Return a single-line, character-bounded preview."""

    preview = " ".join(text.split())
    if len(preview) <= limit:
        return preview
    return f"{preview[: max(0, limit - 1)]}…"


def normalize_text(
    text: str, *, case_sensitive: bool, normalize_whitespace: bool
) -> str:
    """Apply documented comparison normalization in a fixed order."""

    normalized = " ".join(text.split()) if normalize_whitespace else text
    return normalized if case_sensitive else normalized.casefold()


def evaluate_response(case: EvaluationCase, actual: str) -> list[EvaluatorResult]:
    """Run non-empty plus every evaluator configured for one case."""

    expected = case.expected
    normalized_actual = normalize_text(
        actual,
        case_sensitive=expected.case_sensitive,
        normalize_whitespace=expected.normalize_whitespace,
    )
    results = [_non_empty(actual)]

    if expected.exact_match is not None:
        normalized_expected = normalize_text(
            expected.exact_match,
            case_sensitive=expected.case_sensitive,
            normalize_whitespace=expected.normalize_whitespace,
        )
        passed = normalized_actual == normalized_expected
        results.append(
            _result(
                "exact_match",
                passed,
                expected=expected.exact_match,
                actual=actual,
                reason=None if passed else "normalized response did not match",
            )
        )
    if expected.contains_all:
        results.append(
            _contains_result(
                name="contains_all",
                expected_values=expected.contains_all,
                actual=actual,
                normalized_actual=normalized_actual,
                case_sensitive=expected.case_sensitive,
                normalize_whitespace=expected.normalize_whitespace,
                require_all=True,
            )
        )
    if expected.contains_any:
        results.append(
            _contains_result(
                name="contains_any",
                expected_values=expected.contains_any,
                actual=actual,
                normalized_actual=normalized_actual,
                case_sensitive=expected.case_sensitive,
                normalize_whitespace=expected.normalize_whitespace,
                require_all=False,
            )
        )
    if expected.not_contains:
        normalized_forbidden = [
            normalize_text(
                value,
                case_sensitive=expected.case_sensitive,
                normalize_whitespace=expected.normalize_whitespace,
            )
            for value in expected.not_contains
        ]
        found = [
            original
            for original, normalized in zip(
                expected.not_contains, normalized_forbidden, strict=True
            )
            if normalized in normalized_actual
        ]
        passed = not found
        results.append(
            _result(
                "not_contains",
                passed,
                expected=f"forbid: {', '.join(expected.not_contains)}",
                actual=actual,
                reason=(
                    None
                    if passed
                    else bounded_preview(
                        f"found forbidden value(s): {', '.join(found)}"
                    )
                ),
            )
        )
    if (
        expected.minimum_characters is not None
        or expected.maximum_characters is not None
    ):
        minimum = expected.minimum_characters
        maximum = expected.maximum_characters
        passed = (minimum is None or len(actual) >= minimum) and (
            maximum is None or len(actual) <= maximum
        )
        bounds = f"{minimum if minimum is not None else 0}.."
        bounds += str(maximum) if maximum is not None else "unbounded"
        results.append(
            EvaluatorResult(
                name="response_length",
                passed=passed,
                score=1.0 if passed else 0.0,
                failure_reason=None if passed else "response length was outside bounds",
                expected_summary=f"{bounds} characters",
                actual_summary=f"{len(actual)} characters",
            )
        )
    return results


def _non_empty(actual: str) -> EvaluatorResult:
    passed = bool(actual.strip())
    return _result(
        "non_empty",
        passed,
        expected="non-empty response",
        actual=actual,
        reason=None if passed else "response was empty",
    )


def _contains_result(
    *,
    name: str,
    expected_values: list[str],
    actual: str,
    normalized_actual: str,
    case_sensitive: bool,
    normalize_whitespace: bool,
    require_all: bool,
) -> EvaluatorResult:
    matches = [
        normalize_text(
            value,
            case_sensitive=case_sensitive,
            normalize_whitespace=normalize_whitespace,
        )
        in normalized_actual
        for value in expected_values
    ]
    passed = all(matches) if require_all else any(matches)
    matched_count = sum(matches)
    reason = None
    if not passed:
        if require_all:
            missing = [
                value
                for value, matched in zip(expected_values, matches, strict=True)
                if not matched
            ]
            reason = bounded_preview(f"missing required value(s): {', '.join(missing)}")
        else:
            reason = "none of the expected values were found"
    return EvaluatorResult(
        name=name,
        passed=passed,
        score=matched_count / len(expected_values),
        failure_reason=reason,
        expected_summary=bounded_preview(", ".join(expected_values)),
        actual_summary=bounded_preview(actual),
    )


def _result(
    name: str,
    passed: bool,
    *,
    expected: str,
    actual: str,
    reason: str | None,
) -> EvaluatorResult:
    return EvaluatorResult(
        name=name,
        passed=passed,
        score=1.0 if passed else 0.0,
        failure_reason=reason,
        expected_summary=bounded_preview(expected),
        actual_summary=bounded_preview(actual),
    )
