from __future__ import annotations

from pathlib import Path

import pytest

from llm_platform.experiments.cli import (
    EXIT_INVALID_INPUT,
    EXIT_NOT_FOUND,
    EXIT_OPERATIONAL_FAILURE,
    EXIT_REGISTRY_INTEGRITY_FAILURE,
    EXIT_REGRESSION_FAILED,
    EXIT_SUCCESS,
    main,
)
from llm_platform.experiments.registry import ExperimentRegistry
from llm_platform.experiments.runner import ExperimentRunner, ExperimentRunResult
from tests.experiment_helpers import manifest, register_manifest


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["run", "--help"],
        ["list", "--help"],
        ["show", "--help"],
        ["compare", "--help"],
        ["alias", "--help"],
        ["verify", "--help"],
    ],
)
def test_every_command_has_help(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(arguments)
    assert exc_info.value.code == 0


def test_list_show_alias_compare_and_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "experiments"
    registry = ExperimentRegistry(root)
    register_manifest(registry, "left")
    register_manifest(registry, "right")
    common = ["--registry-dir", str(root)]
    assert main(["list", *common]) == EXIT_SUCCESS
    assert main(["show", "left", *common]) == EXIT_SUCCESS
    assert main(["alias", "set", "baseline", "left", *common]) == EXIT_SUCCESS
    assert main(["alias", "show", "baseline", *common]) == EXIT_SUCCESS
    assert main(["verify", "baseline", *common]) == EXIT_SUCCESS
    assert (
        main(
            [
                "compare",
                "left",
                "right",
                *common,
                "--output-dir",
                str(tmp_path / "comparisons"),
            ]
        )
        == EXIT_SUCCESS
    )
    output = capsys.readouterr().out
    assert "left" in output
    assert "Experiment Comparison" in output


def test_cli_exit_codes_are_stable_and_traceback_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "experiments"
    registry = ExperimentRegistry(root)
    register_manifest(registry)
    common = ["--registry-dir", str(root)]
    assert main(["show", "missing", *common]) == EXIT_NOT_FOUND
    artifact = registry.artifact_path("run-1", "summary.md")
    artifact.write_text("modified", encoding="utf-8")
    assert main(["verify", "run-1", *common]) == EXIT_REGISTRY_INTEGRITY_FAILURE
    assert (
        main(
            [
                "run",
                "--dataset",
                str(tmp_path / "missing.jsonl"),
                *common,
            ]
        )
        == EXIT_INVALID_INPUT
    )
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("exit_code", "status"),
    [
        (EXIT_SUCCESS, "completed"),
        (EXIT_REGRESSION_FAILED, "regression_failed"),
        (EXIT_OPERATIONAL_FAILURE, "failed"),
    ],
)
def test_run_preserves_runner_exit_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    exit_code: int,
    status: str,
) -> None:
    value, _ = manifest(f"run-{exit_code}", status=status)

    async def fake_run(
        _self: ExperimentRunner, _configuration: object
    ) -> ExperimentRunResult:
        return ExperimentRunResult(manifest=value, exit_code=exit_code)

    monkeypatch.setattr(ExperimentRunner, "run", fake_run)
    result = main(
        [
            "run",
            "--dataset",
            str(tmp_path / "dataset.jsonl"),
            "--registry-dir",
            str(tmp_path / "experiments"),
        ]
    )
    assert result == exit_code
    assert "Traceback" not in capsys.readouterr().err
