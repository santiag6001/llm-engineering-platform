"""Bounded UTF-8 loading and atomic local document registration."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from llm_platform.rag.document import (
    DocumentRecord,
    DocumentRegistryManifest,
    document_fingerprint,
    fingerprint,
    sha256_bytes,
)

MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = frozenset(
    {"text/plain", "text/markdown", "text/x-rst", "application/json"}
)


class DocumentRegistryError(RuntimeError):
    """A local registry operation failed safely."""


class DuplicateDocumentError(DocumentRegistryError):
    """The exact document content is already registered."""


class DocumentNotFoundError(DocumentRegistryError):
    """A requested document does not exist."""


class DocumentValidationError(ValueError):
    """A source document was unreadable or unsupported."""


class DocumentRegistry:
    """Store immutable source bytes and a strictly validated registry manifest."""

    def __init__(
        self,
        root: Path,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.root = root
        self.documents_directory = root / "documents"
        self.manifest_path = root / "documents.json"
        self._now = now

    def register(
        self,
        path: Path,
        *,
        logical_name: str | None = None,
        content_type: str = "text/plain",
    ) -> DocumentRecord:
        """Validate and atomically register one local UTF-8 document."""

        if content_type not in SUPPORTED_CONTENT_TYPES:
            raise DocumentValidationError(f"unsupported content type: {content_type}")
        try:
            size = path.stat().st_size
            if size > MAX_DOCUMENT_BYTES:
                raise DocumentValidationError(
                    f"document exceeds {MAX_DOCUMENT_BYTES} byte limit"
                )
            content = path.read_bytes()
        except OSError as exc:
            raise DocumentValidationError("could not read document") from exc
        try:
            content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentValidationError("document must be valid UTF-8") from exc

        name = logical_name or path.name
        if not name or len(name) > 256 or any(ord(char) < 32 for char in name):
            raise DocumentValidationError(
                "logical name must be 1..256 printable characters"
            )
        content_hash = sha256_bytes(content)
        document_id = f"doc-{content_hash}"
        manifest = self._load_manifest()
        if any(document.sha256 == content_hash for document in manifest.documents):
            raise DuplicateDocumentError(f"document already registered: {document_id}")

        record = DocumentRecord(
            document_id=document_id,
            sha256=content_hash,
            logical_name=name,
            byte_size=len(content),
            content_type=content_type,
            ingestion_timestamp=self._now().astimezone(UTC),
            fingerprint=document_fingerprint(
                content_sha256=content_hash,
                logical_name=name,
                byte_size=len(content),
                content_type=content_type,
            ),
        )
        self._ensure_layout()
        destination = self._content_path(document_id)
        try:
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise DuplicateDocumentError(
                f"document already registered: {document_id}"
            ) from exc
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            updated = DocumentRegistryManifest(
                documents=sorted(
                    [*manifest.documents, record],
                    key=lambda document: document.document_id,
                )
            )
            self._atomic_json(self.manifest_path, updated.model_dump(mode="json"))
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return record

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._load_manifest().documents)

    def get(self, document_id: str) -> DocumentRecord:
        for document in self._load_manifest().documents:
            if document.document_id == document_id:
                return document
        raise DocumentNotFoundError(f"document not found: {document_id}")

    def content_bytes(self, document_id: str) -> bytes:
        self.get(document_id)
        path = self._content_path(document_id)
        if path.is_symlink():
            raise DocumentRegistryError("document content must not be a symlink")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise DocumentRegistryError(
                "registered document content is missing"
            ) from exc
        if sha256_bytes(content) != document_id.removeprefix("doc-"):
            raise DocumentRegistryError("registered document checksum mismatch")
        return content

    def content_text(self, document_id: str) -> str:
        return self.content_bytes(document_id).decode("utf-8")

    def corpus_fingerprint(self) -> str:
        return fingerprint(
            {
                "schema_version": "1.0",
                "document_fingerprints": [
                    document.fingerprint for document in self.list_documents()
                ],
            }
        )

    def _load_manifest(self) -> DocumentRegistryManifest:
        if not self.manifest_path.exists():
            return DocumentRegistryManifest()
        if self.root.is_symlink() or self.manifest_path.is_symlink():
            raise DocumentRegistryError("registry paths must not be symlinks")
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest = DocumentRegistryManifest.model_validate(raw)
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise DocumentRegistryError("document registry is malformed") from exc
        ids = [document.document_id for document in manifest.documents]
        hashes = [document.sha256 for document in manifest.documents]
        duplicate_id = len(ids) != len(set(ids))
        duplicate_hash = len(hashes) != len(set(hashes))
        if ids != sorted(ids) or duplicate_id or duplicate_hash:
            raise DocumentRegistryError("document registry ordering is invalid")
        for document in manifest.documents:
            expected_fingerprint = document_fingerprint(
                content_sha256=document.sha256,
                logical_name=document.logical_name,
                byte_size=document.byte_size,
                content_type=document.content_type,
            )
            if document.document_id != f"doc-{document.sha256}":
                raise DocumentRegistryError("document ID does not match its checksum")
            if document.fingerprint != expected_fingerprint:
                raise DocumentRegistryError("document fingerprint verification failed")
        return manifest

    def _ensure_layout(self) -> None:
        self.documents_directory.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or self.documents_directory.is_symlink():
            raise DocumentRegistryError("registry paths must not be symlinks")

    def _content_path(self, document_id: str) -> Path:
        if not document_id.startswith("doc-") or len(document_id) != 68:
            raise DocumentNotFoundError("invalid document ID")
        return self.documents_directory / f"{document_id}.utf8"

    def _atomic_json(self, destination: Path, value: object) -> None:
        content = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".documents-", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
