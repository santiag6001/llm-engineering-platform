"""Deterministic persistent local vector index."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from llm_platform.rag.chunking import (
    ChunkConfiguration,
    ChunkRecord,
    chunk_document,
)
from llm_platform.rag.document import (
    SHA256_PATTERN,
    StrictModel,
    fingerprint,
    sha256_bytes,
)
from llm_platform.rag.embedding import (
    EmbeddingConfiguration,
    EmbeddingMetadata,
    LocalHashingEmbedder,
)
from llm_platform.rag.loader import DocumentRegistry


class IndexError(RuntimeError):
    """A persistent index failed validation or access."""


class IndexEntry(StrictModel):
    chunk: ChunkRecord
    vector: tuple[float, ...]

    @field_validator("vector")
    @classmethod
    def vector_values_are_finite(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        invalid = any(item != item or abs(item) == float("inf") for item in value)
        if not value or invalid:
            raise ValueError("index vectors must be non-empty and finite")
        return value


class IndexMetadata(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    built_at: datetime
    document_fingerprint: str = Field(pattern=SHA256_PATTERN)
    document_fingerprints: list[str]
    chunk_configuration: ChunkConfiguration
    chunk_fingerprints: list[str]
    embedding_configuration: EmbeddingConfiguration
    embedding_metadata: EmbeddingMetadata
    index_fingerprint: str = Field(pattern=SHA256_PATTERN)
    entry_count: int = Field(ge=0)


class IndexManifest(StrictModel):
    metadata: IndexMetadata
    entries: list[IndexEntry]


class LocalVectorIndex:
    """Build, validate, and query one JSON-backed local vector index."""

    def __init__(
        self,
        path: Path,
        *,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.path = path
        self._now = now

    def build(
        self,
        registry: DocumentRegistry,
        *,
        chunk_configuration: ChunkConfiguration,
        embedding_configuration: EmbeddingConfiguration,
    ) -> IndexManifest:
        documents = registry.list_documents()
        embedder = LocalHashingEmbedder(embedding_configuration)
        entries: list[IndexEntry] = []
        for document in documents:
            chunks = chunk_document(
                document.document_id,
                registry.content_text(document.document_id),
                chunk_configuration,
            )
            entries.extend(
                IndexEntry(chunk=chunk, vector=embedder.embed(chunk.text))
                for chunk in chunks
            )
        entries.sort(
            key=lambda entry: (
                entry.chunk.document_id,
                entry.chunk.chunk_index,
                entry.chunk.chunk_id,
            )
        )
        chunk_fingerprints = [entry.chunk.fingerprint for entry in entries]
        document_fingerprints = [document.fingerprint for document in documents]
        index_fingerprint = fingerprint(
            {
                "schema_version": "1.0",
                "document_fingerprint": registry.corpus_fingerprint(),
                "chunk_configuration": chunk_configuration.model_dump(mode="json"),
                "chunk_fingerprints": chunk_fingerprints,
                "embedding_configuration": embedding_configuration.model_dump(
                    mode="json"
                ),
                "entries": [
                    {
                        "chunk_id": entry.chunk.chunk_id,
                        "vector": list(entry.vector),
                    }
                    for entry in entries
                ],
            }
        )
        manifest = IndexManifest(
            metadata=IndexMetadata(
                built_at=self._now().astimezone(UTC),
                document_fingerprint=registry.corpus_fingerprint(),
                document_fingerprints=document_fingerprints,
                chunk_configuration=chunk_configuration,
                chunk_fingerprints=chunk_fingerprints,
                embedding_configuration=embedding_configuration,
                embedding_metadata=embedder.metadata,
                index_fingerprint=index_fingerprint,
                entry_count=len(entries),
            ),
            entries=entries,
        )
        self._atomic_write(manifest)
        return manifest

    def load(self) -> IndexManifest:
        if self.path.is_symlink():
            raise IndexError("index path must not be a symlink")
        try:
            manifest = IndexManifest.model_validate_json(self.path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise IndexError("vector index is missing or malformed") from exc
        self._validate(manifest)
        return manifest

    def get_chunk(self, chunk_id: str) -> ChunkRecord:
        for entry in self.load().entries:
            if entry.chunk.chunk_id == chunk_id:
                return entry.chunk
        raise IndexError(f"chunk not found: {chunk_id}")

    def _validate(self, manifest: IndexManifest) -> None:
        metadata = manifest.metadata
        entries = manifest.entries
        ordered = sorted(
            entries,
            key=lambda entry: (
                entry.chunk.document_id,
                entry.chunk.chunk_index,
                entry.chunk.chunk_id,
            ),
        )
        if entries != ordered:
            raise IndexError("index entries are not in stable order")
        if metadata.entry_count != len(entries):
            raise IndexError("index entry count does not match metadata")
        if metadata.chunk_fingerprints != [
            entry.chunk.fingerprint for entry in entries
        ]:
            raise IndexError("index chunk fingerprints do not match entries")
        for entry in entries:
            chunk = entry.chunk
            content_hash = sha256_bytes(chunk.text.encode("utf-8"))
            expected_chunk_fingerprint = fingerprint(
                {
                    "schema_version": "1.0",
                    "document_id": chunk.document_id,
                    "chunk_sha256": content_hash,
                    "character_start": chunk.character_start,
                    "character_end": chunk.character_end,
                    "chunk_index": chunk.chunk_index,
                    "chunk_configuration_fingerprint": (
                        metadata.chunk_configuration.fingerprint
                    ),
                }
            )
            if (
                chunk.chunk_sha256 != content_hash
                or chunk.fingerprint != expected_chunk_fingerprint
                or chunk.chunk_id != f"chunk-{expected_chunk_fingerprint}"
            ):
                raise IndexError("index chunk fingerprint verification failed")
        dimension = metadata.embedding_configuration.dimension
        if any(len(entry.vector) != dimension for entry in entries):
            raise IndexError("index vector dimension does not match metadata")
        expected = fingerprint(
            {
                "schema_version": "1.0",
                "document_fingerprint": metadata.document_fingerprint,
                "chunk_configuration": metadata.chunk_configuration.model_dump(
                    mode="json"
                ),
                "chunk_fingerprints": metadata.chunk_fingerprints,
                "embedding_configuration": (
                    metadata.embedding_configuration.model_dump(mode="json")
                ),
                "entries": [
                    {
                        "chunk_id": entry.chunk.chunk_id,
                        "vector": list(entry.vector),
                    }
                    for entry in entries
                ],
            }
        )
        if expected != metadata.index_fingerprint:
            raise IndexError("index fingerprint verification failed")

    def _atomic_write(self, manifest: IndexManifest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.parent.is_symlink():
            raise IndexError("index directory must not be a symlink")
        content = (
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".index-", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
