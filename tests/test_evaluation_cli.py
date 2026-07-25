from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_platform.evaluation.cli import (
    EXIT_EVALUATION_ERRORS,
    EXIT_INVALID_INPUT,
    EXIT_REGRESSION_FAILED,
    EXIT_SUCCESS,
    main,
)
from llm_platform.evaluation.runner import EvaluationRunner
from tests.evaluation_helpers import completed_result, error_result, evaluation_report


def _write_dataset(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "case-1",
                "category": "test",
                "messages": [{"role": "user", "content": "prompt"}],
                "expected": {"contains_any": ["answer"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_help_is_useful(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "run" in output
    assert "compare" in output
    assert "regression" in output


def test_dataset_validation_failure_has_stable_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "bad.jsonl"
    dataset.write_text("not-json\n", encoding="utf-8")
    assert main(["run", "--dataset", str(dataset)]) == EXIT_INVALID_INPUT
    assert "Traceback" not in capsys.readouterr().err


def test_successful_run_writes_both_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    output = tmp_path / "reports"
    _write_dataset(dataset)

    async def successful_run(_self: EvaluationRunner, _dataset: object) -> list[object]:
        return [completed_result()]

    monkeypatch.setattr(EvaluationRunner, "run", successful_run)
    exit_code = main(
        [
            "run",
            "--dataset",
            str(dataset),
            "--base-url",
            "http://platform.test",
            "--output-dir",
            str(output),
        ]
    )
    assert exit_code == EXIT_SUCCESS
    assert len(list(output.glob("*.json"))) == 1
    assert len(list(output.glob("*.md"))) == 1


def test_network_failure_run_is_reported_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)

    async def failed_run(_self: EvaluationRunner, _dataset: object) -> list[object]:
        return [error_result()]

    monkeypatch.setattr(EvaluationRunner, "run", failed_run)
    exit_code = main(
        [
            "run",
            "--dataset",
            str(dataset),
            "--output-dir",
            str(tmp_path / "reports"),
        ]
    )
    assert exit_code == EXIT_EVALUATION_ERRORS
    captured = capsys.readouterr()
    assert "1 errors" in captured.out
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("minimum", "expected"),
    [("0.5", EXIT_SUCCESS), ("1.0", EXIT_REGRESSION_FAILED)],
)
def test_compare_exit_codes(tmp_path: Path, minimum: str, expected: int) -> None:
    baseline = evaluation_report(completed_result("baseline"), run_id="baseline")
    current = evaluation_report(
        completed_result("pass"),
        completed_result("fail", passed=False),
        run_id="current",
    )
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    baseline_path.write_text(baseline.model_dump_json(), encoding="utf-8")
    current_path.write_text(current.model_dump_json(), encoding="utf-8")
    exit_code = main(
        [
            "compare",
            "--current",
            str(current_path),
            "--baseline",
            str(baseline_path),
            "--min-pass-rate",
            minimum,
        ]
    )
    assert exit_code == expected
