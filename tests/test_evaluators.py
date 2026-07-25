from __future__ import annotations

from llm_platform.evaluation.evaluators import (
    bounded_preview,
    evaluate_response,
    normalize_text,
)
from tests.evaluation_helpers import evaluation_case


def _by_name(actual: str, expected: dict[str, object]) -> dict[str, bool]:
    return {
        result.name: result.passed
        for result in evaluate_response(evaluation_case(expected=expected), actual)
    }


def test_exact_match_and_non_empty() -> None:
    results = _by_name("the answer", {"exact_match": "the answer"})
    assert results == {"non_empty": True, "exact_match": True}
    assert not _by_name("", {"exact_match": "answer"})["non_empty"]


def test_case_and_whitespace_normalization_are_explicit() -> None:
    assert (
        normalize_text("  KV\n Cache ", case_sensitive=False, normalize_whitespace=True)
        == "kv cache"
    )
    assert _by_name(
        "  KV\n Cache ",
        {
            "exact_match": "kv cache",
            "case_sensitive": False,
            "normalize_whitespace": True,
        },
    )["exact_match"]
    assert not _by_name(
        "KV Cache",
        {"exact_match": "kv cache", "case_sensitive": True},
    )["exact_match"]


def test_contains_all_success_and_failure() -> None:
    assert _by_name("keys and values are cached", {"contains_all": ["keys", "values"]})[
        "contains_all"
    ]
    assert not _by_name("keys are cached", {"contains_all": ["keys", "values"]})[
        "contains_all"
    ]


def test_contains_any_success_and_failure() -> None:
    assert _by_name("lower latency", {"contains_any": ["latency", "speed"]})[
        "contains_any"
    ]
    assert not _by_name("unrelated", {"contains_any": ["latency", "speed"]})[
        "contains_any"
    ]


def test_forbidden_string_success_and_failure() -> None:
    assert _by_name("safe answer", {"not_contains": ["secret"]})["not_contains"]
    assert not _by_name("SECRET value", {"not_contains": ["secret"]})["not_contains"]


def test_response_length_bounds() -> None:
    assert _by_name("four", {"minimum_characters": 4, "maximum_characters": 4})[
        "response_length"
    ]
    assert not _by_name("longer", {"maximum_characters": 4})["response_length"]


def test_previews_are_bounded_and_single_line() -> None:
    preview = bounded_preview("a\n" + "b" * 300, limit=20)
    assert len(preview) == 20
    assert "\n" not in preview
    assert preview.endswith("…")
