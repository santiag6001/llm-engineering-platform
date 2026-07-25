from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_platform.evaluation.dataset import DatasetValidationError, load_dataset


def _valid_case(case_id: str = "case-1") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "id": case_id,
        "category": "serving",
        "messages": [{"role": "user", "content": "Explain prefill."}],
        "expected": {"contains_any": ["prompt", "input"]},
        "generation": {"temperature": 0.0, "max_tokens": 20},
        "metadata": {"description": "A test.", "tags": ["prefill"]},
    }


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_valid_jsonl_loading_and_hash_stability(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(path, [_valid_case("first"), _valid_case("second")])

    first = load_dataset(path)
    second = load_dataset(path)

    assert [case.id for case in first.cases] == ["first", "second"]
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64


def test_malformed_json_reports_line_without_raw_content(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"schema_version": "1.0"\n', encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="malformed JSON on line 1"):
        load_dataset(path)


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    _write_jsonl(path, [_valid_case(), _valid_case()])

    with pytest.raises(DatasetValidationError, match="duplicate case id"):
        load_dataset(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ({"messages": [{"role": "tool", "content": "x"}]}, "messages.0.role"),
        ({"messages": []}, "messages"),
        ({"expected": {}}, "expected"),
        ({"generation": {"temperature": 3}}, "generation.temperature"),
        ({"generation": {"max_tokens": 0}}, "generation.max_tokens"),
        ({"unknown": True}, "unknown"),
    ],
)
def test_invalid_cases_are_rejected(
    tmp_path: Path, mutation: dict[str, object], match: str
) -> None:
    path = tmp_path / "dataset.jsonl"
    row = {**_valid_case(), **mutation}
    _write_jsonl(path, [row])

    with pytest.raises(DatasetValidationError, match=match):
        load_dataset(path)
