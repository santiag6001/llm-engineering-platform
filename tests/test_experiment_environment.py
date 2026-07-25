from __future__ import annotations

import subprocess

import pytest

from llm_platform.experiments.environment import (
    DEPENDENCY_ALLOWLIST,
    collect_environment,
    collect_source_metadata,
)


def test_environment_is_allowlisted_stable_and_ignores_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"httpx": "1", "pydantic": "2", "secret-package": "3"}
    first = collect_environment(
        dependency_version=lambda name: versions[name], container="none"
    )
    monkeypatch.setenv("API_TOKEN", "private-value")
    monkeypatch.setenv("HOME", "/private/home")
    second = collect_environment(
        dependency_version=lambda name: versions[name], container="none"
    )
    assert set(first.dependencies) == set(DEPENDENCY_ALLOWLIST)
    assert first.environment_fingerprint == second.environment_fingerprint
    dumped = first.model_dump()
    assert not {
        "hostname",
        "username",
        "home",
        "environment_variables",
        "secret-package",
    } & set(dumped)


def test_source_metadata_captures_commit_dirty_state_and_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def run(
        command: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        if command[1:3] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, stdout="a" * 40 + "\n")
        if command[1] == "diff-index":
            return subprocess.CompletedProcess(command, 1, stdout="")
        if command[1:3] == ["branch", "--show-current"]:
            return subprocess.CompletedProcess(command, 0, stdout="feature/test\n")
        raise AssertionError(command)

    monkeypatch.setattr("llm_platform.experiments.environment.subprocess.run", run)
    metadata = collect_source_metadata()
    assert metadata.git_commit == "a" * 40
    assert metadata.git_dirty is True
    assert metadata.branch == "feature/test"
