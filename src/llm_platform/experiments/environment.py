"""Bounded source and runtime environment capture."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from llm_platform import __version__
from llm_platform.experiments.identity import canonical_json, sha256_bytes
from llm_platform.experiments.models import EnvironmentMetadata, SourceMetadata

DEPENDENCY_ALLOWLIST = ("httpx", "pydantic")


def collect_source_metadata(repository: Path | None = None) -> SourceMetadata:
    """Collect bounded Git metadata without making Git availability mandatory."""

    commit = _git(["rev-parse", "HEAD"], repository)
    if commit is None or not _is_hex_commit(commit):
        return SourceMetadata()
    dirty = _git_dirty(repository)
    branch = _git(["branch", "--show-current"], repository)
    return SourceMetadata(
        git_commit=commit,
        git_dirty=dirty,
        branch=branch[:255] if branch else None,
    )


def collect_environment(
    *,
    dependency_version: Callable[[str], str] = importlib.metadata.version,
    container: str | None = None,
) -> EnvironmentMetadata:
    """Capture only the documented reproducibility allowlist."""

    dependencies: dict[str, str] = {}
    for name in DEPENDENCY_ALLOWLIST:
        try:
            dependencies[name] = dependency_version(name)[:64]
        except importlib.metadata.PackageNotFoundError:
            continue
    values = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.system() or "unknown",
        "platform_release": (platform.release() or "unknown")[:256],
        "architecture": platform.machine() or "unknown",
        "project_version": __version__,
        "container": container or _detect_container(),
        "dependencies": dict(sorted(dependencies.items())),
    }
    fingerprint = sha256_bytes(canonical_json(values))
    return EnvironmentMetadata(
        python_version=values["python_version"],
        python_implementation=values["python_implementation"],
        operating_system=values["operating_system"],
        platform_release=values["platform_release"],
        architecture=values["architecture"],
        project_version=values["project_version"],
        container=values["container"],
        dependencies=values["dependencies"],
        environment_fingerprint=fingerprint,
    )


def _git(arguments: list[str], repository: Path | None) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_dirty(repository: Path | None) -> bool | None:
    """Check tracked and untracked state without capturing an unbounded listing."""

    commands = (
        ["git", "diff-index", "--quiet", "HEAD", "--"],
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "--error-unmatch",
            "--",
            "*",
        ],
    )
    for command in commands:
        try:
            result = subprocess.run(
                command,
                cwd=repository,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0:
            if command[1] == "ls-files":
                return True
            continue
        if result.returncode == 1:
            if command[1] == "diff-index":
                return True
            continue
        return None
    return False


def _is_hex_commit(value: str) -> bool:
    return 7 <= len(value) <= 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _detect_container() -> str:
    if Path("/.dockerenv").is_file():
        return "docker"
    try:
        cgroup = Path("/proc/1/cgroup").read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "unknown"
    if any(marker in cgroup for marker in ("containerd", "kubepods", "lxc")):
        return "container"
    return "none" if sys.platform.startswith(("linux", "darwin", "win")) else "unknown"
