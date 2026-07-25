"""Atomic filesystem-backed experiment registry."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from llm_platform.experiments.identity import sha256_bytes
from llm_platform.experiments.models import (
    MAX_ARTIFACT_BYTES,
    ArtifactKind,
    ArtifactMetadata,
    ExperimentManifest,
)

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALIAS = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


class RegistryError(RuntimeError):
    """Base class for safe registry failures."""


class RegistryIntegrityError(RegistryError):
    """Registered content failed structural or artifact verification."""


class RunNotFoundError(RegistryError):
    """A requested run or alias does not exist."""


class DuplicateRunError(RegistryError):
    """A run ID already exists."""


class InvalidRegistryNameError(RegistryError):
    """A run ID, alias, or artifact path was unsafe."""


class _AliasRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str


@dataclass(frozen=True)
class ArtifactPayload:
    """One bounded artifact to be committed with a manifest."""

    kind: ArtifactKind
    path: str
    content: bytes

    def metadata(self) -> ArtifactMetadata:
        _validate_relative_path(self.path)
        if len(self.content) > MAX_ARTIFACT_BYTES:
            raise RegistryIntegrityError("artifact exceeds the registry size limit")
        return ArtifactMetadata(
            kind=self.kind,
            path=self.path,
            sha256=sha256_bytes(self.content),
            byte_size=len(self.content),
        )


class ExperimentRegistry:
    """Store immutable runs and mutable, atomic alias pointers."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.runs_directory = root / "runs"
        self.aliases_directory = root / "aliases"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_directory.mkdir(exist_ok=True)
        self.aliases_directory.mkdir(exist_ok=True)
        for directory in (self.root, self.runs_directory, self.aliases_directory):
            if directory.is_symlink() or not directory.is_dir():
                raise RegistryIntegrityError(
                    "registry directories must not be symlinks"
                )

    def register(
        self,
        manifest: ExperimentManifest,
        artifacts: list[ArtifactPayload],
    ) -> Path:
        """Commit one immutable run through a same-filesystem atomic rename."""

        self.initialize()
        _validate_run_id(manifest.run_id)
        target = self.runs_directory / manifest.run_id
        if target.exists() or target.is_symlink():
            raise DuplicateRunError(f"run already exists: {manifest.run_id}")

        supplied = [artifact.metadata() for artifact in artifacts]
        if supplied != manifest.artifacts:
            raise RegistryIntegrityError(
                "manifest artifact metadata does not match supplied artifacts"
            )

        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.runs_directory))
        try:
            for artifact in artifacts:
                artifact_path = staging / artifact.path
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                _write_new_file(artifact_path, artifact.content)
            checksums = {
                artifact.path: {
                    "byte_size": artifact.byte_size,
                    "sha256": artifact.sha256,
                }
                for artifact in manifest.artifacts
            }
            _write_new_file(
                staging / manifest.checksums_path,
                _json_bytes(checksums),
            )
            _write_new_file(
                staging / "manifest.json",
                _json_bytes(manifest.model_dump(mode="json")),
            )
            _fsync_directory(staging)
            try:
                staging.rename(target)
            except FileExistsError as exc:
                raise DuplicateRunError(
                    f"run already exists: {manifest.run_id}"
                ) from exc
            except OSError as exc:
                if target.exists():
                    raise DuplicateRunError(
                        f"run already exists: {manifest.run_id}"
                    ) from exc
                raise RegistryError(
                    "could not atomically register experiment run"
                ) from exc
            _fsync_directory(self.runs_directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return target

    def get(self, run_id_or_alias: str) -> ExperimentManifest:
        run_id = self.resolve(run_id_or_alias)
        manifest_path = self.runs_directory / run_id / "manifest.json"
        try:
            raw = json.loads(_read_regular_file(manifest_path).decode("utf-8"))
            manifest = ExperimentManifest.model_validate(raw)
        except (
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValidationError,
        ) as exc:
            raise RegistryIntegrityError(f"run manifest is invalid: {run_id}") from exc
        if manifest.run_id != run_id:
            raise RegistryIntegrityError("manifest run ID does not match its directory")
        return manifest

    def resolve(self, run_id_or_alias: str) -> str:
        self.initialize()
        if _RUN_ID.fullmatch(run_id_or_alias):
            run_path = self.runs_directory / run_id_or_alias
            if run_path.is_dir() and not run_path.is_symlink():
                return run_id_or_alias
        if not _ALIAS.fullmatch(run_id_or_alias):
            raise InvalidRegistryNameError("invalid run ID or alias")
        alias_path = self.aliases_directory / f"{run_id_or_alias}.json"
        if not alias_path.exists():
            raise RunNotFoundError(f"run or alias not found: {run_id_or_alias}")
        try:
            record = _AliasRecord.model_validate_json(_read_regular_file(alias_path))
        except (OSError, ValidationError) as exc:
            raise RegistryIntegrityError(
                f"alias is invalid: {run_id_or_alias}"
            ) from exc
        _validate_run_id(record.run_id)
        target = self.runs_directory / record.run_id
        if not target.is_dir() or target.is_symlink():
            raise RegistryIntegrityError(
                f"alias target does not exist: {run_id_or_alias}"
            )
        return record.run_id

    def list_runs(
        self,
        *,
        status: str | None = None,
        experiment_fingerprint: str | None = None,
        requested_model: str | None = None,
        git_commit: str | None = None,
    ) -> list[ExperimentManifest]:
        self.initialize()
        manifests: list[ExperimentManifest] = []
        for candidate in self.runs_directory.iterdir():
            if candidate.name.startswith(".staging-"):
                continue
            if candidate.is_symlink() or not candidate.is_dir():
                raise RegistryIntegrityError("unexpected entry in runs directory")
            manifest = self.get(candidate.name)
            if status is not None and manifest.status != status:
                continue
            if (
                experiment_fingerprint is not None
                and manifest.experiment_fingerprint != experiment_fingerprint
            ):
                continue
            if (
                requested_model is not None
                and manifest.model.requested != requested_model
            ):
                continue
            if git_commit is not None and manifest.source.git_commit != git_commit:
                continue
            manifests.append(manifest)
        return sorted(
            manifests,
            key=lambda item: (item.created_at, item.run_id),
            reverse=True,
        )

    def newest(self, **filters: str | None) -> ExperimentManifest | None:
        matches = self.list_runs(
            status=filters.get("status"),
            experiment_fingerprint=filters.get("experiment_fingerprint"),
            requested_model=filters.get("requested_model"),
            git_commit=filters.get("git_commit"),
        )
        return matches[0] if matches else None

    def set_alias(self, alias: str, run_id: str) -> None:
        self.initialize()
        _validate_alias(alias)
        resolved = self.resolve(run_id)
        if resolved != run_id:
            raise InvalidRegistryNameError("alias targets must be immutable run IDs")
        content = _json_bytes({"run_id": run_id})
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".alias-", dir=self.aliases_directory
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.aliases_directory / f"{alias}.json")
            _fsync_directory(self.aliases_directory)
        finally:
            if temporary.exists():
                temporary.unlink()

    def show_alias(self, alias: str) -> str:
        _validate_alias(alias)
        return self.resolve(alias)

    def artifact_path(self, run_id_or_alias: str, relative_path: str) -> Path:
        run_id = self.resolve(run_id_or_alias)
        _validate_relative_path(relative_path)
        run_directory = self.runs_directory / run_id
        artifact = run_directory / relative_path
        _ensure_no_symlink_components(run_directory, artifact)
        return artifact

    def verify(self, run_id_or_alias: str) -> ExperimentManifest:
        manifest = self.get(run_id_or_alias)
        run_directory = self.runs_directory / manifest.run_id
        expected = {
            artifact.path: {
                "byte_size": artifact.byte_size,
                "sha256": artifact.sha256,
            }
            for artifact in manifest.artifacts
        }
        try:
            checksums = json.loads(
                _read_regular_file(run_directory / manifest.checksums_path).decode(
                    "utf-8"
                )
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryIntegrityError(
                "checksums index is missing or invalid"
            ) from exc
        if checksums != expected:
            raise RegistryIntegrityError("checksums index does not match manifest")
        for artifact in manifest.artifacts:
            path = self.artifact_path(manifest.run_id, artifact.path)
            try:
                content = _read_regular_file(path)
            except OSError as exc:
                raise RegistryIntegrityError(
                    f"artifact is missing or unsafe: {artifact.path}"
                ) from exc
            if len(content) != artifact.byte_size:
                raise RegistryIntegrityError(
                    f"artifact byte size mismatch: {artifact.path}"
                )
            if sha256_bytes(content) != artifact.sha256:
                raise RegistryIntegrityError(
                    f"artifact checksum mismatch: {artifact.path}"
                )
        return manifest


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_new_file(path: Path, content: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _read_regular_file(path: Path, *, max_bytes: int = MAX_ARTIFACT_BYTES) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise OSError("path is not a regular file")
        if file_stat.st_size > max_bytes:
            raise OSError("file exceeds the registry size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            return stream.read()
    finally:
        os.close(descriptor)


def _ensure_no_symlink_components(root: Path, target: Path) -> None:
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise InvalidRegistryNameError("artifact path escaped its run") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise RegistryIntegrityError("artifact path contains a symlink")


def _validate_relative_path(value: str) -> None:
    if (
        not value
        or value.startswith(("/", "\\"))
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise InvalidRegistryNameError("invalid relative artifact path")


def _validate_run_id(value: str) -> None:
    if not _RUN_ID.fullmatch(value):
        raise InvalidRegistryNameError("invalid run ID")


def _validate_alias(value: str) -> None:
    if not _ALIAS.fullmatch(value):
        raise InvalidRegistryNameError("invalid alias")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
