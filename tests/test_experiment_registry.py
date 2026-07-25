from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest

from llm_platform.experiments.registry import (
    DuplicateRunError,
    ExperimentRegistry,
    InvalidRegistryNameError,
    RegistryIntegrityError,
    RunNotFoundError,
)
from tests.experiment_helpers import FIXED_TIME, manifest, register_manifest


def test_empty_runs_directory_lists_no_runs(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")

    assert registry.list_runs() == []


def test_gitkeep_placeholder_lists_no_runs(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    registry.initialize()
    (registry.runs_directory / ".gitkeep").touch()

    assert registry.list_runs() == []


def test_gitkeep_placeholder_does_not_affect_newest_run(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    registered = register_manifest(registry)
    (registry.runs_directory / ".gitkeep").touch()

    assert registry.newest() == registered


@pytest.mark.parametrize("filename", ["unexpected.txt", ".unexpected"])
def test_unexpected_files_in_runs_directory_are_rejected(
    tmp_path: Path, filename: str
) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    registry.initialize()
    (registry.runs_directory / filename).touch()

    with pytest.raises(RegistryIntegrityError):
        registry.list_runs()


def test_malformed_run_directory_is_rejected(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    registry.initialize()
    (registry.runs_directory / "malformed-run").mkdir()

    with pytest.raises(RegistryIntegrityError):
        registry.list_runs()


def test_atomic_registration_listing_filters_and_newest(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    older = register_manifest(registry, "older")
    newer_value, newer_artifacts = manifest(
        "newer",
        created_at=FIXED_TIME + timedelta(seconds=1),
        fingerprint="b" * 64,
    )
    registry.register(newer_value, newer_artifacts)

    assert registry.get("older") == older
    assert [run.run_id for run in registry.list_runs()] == ["newer", "older"]
    assert registry.list_runs(experiment_fingerprint="b" * 64) == [newer_value]
    assert registry.list_runs(requested_model="requested-model")
    assert registry.list_runs(git_commit="a" * 40)
    assert registry.newest().run_id == "newer"  # type: ignore[union-attr]
    assert not any(
        path.name.startswith(".staging-") for path in registry.runs_directory.iterdir()
    )


def test_duplicate_and_concurrent_registration_are_safe(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    value, artifacts = manifest()
    registry.register(value, artifacts)
    with pytest.raises(DuplicateRunError):
        registry.register(value, artifacts)

    concurrent, concurrent_artifacts = manifest("concurrent")

    def register() -> str:
        registry.register(concurrent, concurrent_artifacts)
        return "ok"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(register) for _ in range(2)]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except DuplicateRunError:
            outcomes.append("duplicate")
    assert sorted(outcomes) == ["duplicate", "ok"]


def test_aliases_are_atomic_validated_and_target_existing_runs(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    register_manifest(registry)
    registry.set_alias("baseline", "run-1")
    assert registry.show_alias("baseline") == "run-1"
    assert registry.get("baseline").run_id == "run-1"
    with pytest.raises(RunNotFoundError):
        registry.set_alias("missing", "does-not-exist")
    with pytest.raises(InvalidRegistryNameError):
        registry.set_alias("../escape", "run-1")


def test_staging_directories_are_not_listed(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    registry.initialize()
    (registry.runs_directory / ".staging-incomplete").mkdir()
    assert registry.list_runs() == []


def test_verify_detects_missing_modified_size_and_checksum(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    register_manifest(registry)
    assert registry.verify("run-1").run_id == "run-1"
    artifact = registry.artifact_path("run-1", "summary.md")
    artifact.write_bytes(b"short")
    with pytest.raises(RegistryIntegrityError):
        registry.verify("run-1")
    artifact.write_bytes(b"# Summary\n")
    with pytest.raises(RegistryIntegrityError):
        registry.verify("run-1")
    artifact.unlink()
    with pytest.raises(RegistryIntegrityError):
        registry.verify("run-1")


def test_path_traversal_and_symlink_escape_are_rejected(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "experiments")
    register_manifest(registry)
    with pytest.raises(InvalidRegistryNameError):
        registry.artifact_path("run-1", "../manifest.json")
    artifact = registry.artifact_path("run-1", "summary.md")
    artifact.unlink()
    artifact.symlink_to(tmp_path / "outside")
    with pytest.raises(RegistryIntegrityError):
        registry.verify("run-1")
