"""Configuration-sensitive deterministic document chunking."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from llm_platform.rag.document import (
    CHUNK_ID_PATTERN,
    DOCUMENT_ID_PATTERN,
    SHA256_PATTERN,
    StrictModel,
    fingerprint,
    sha256_bytes,
)

SeparatorStrategy = Literal["character", "line", "paragraph"]


class ChunkConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    chunk_size: int = Field(default=800, ge=1, le=65_536)
    overlap: int = Field(default=100, ge=0, le=65_535)
    separator_strategy: SeparatorStrategy = "paragraph"

    @model_validator(mode="after")
    def overlap_is_smaller(self) -> ChunkConfiguration:
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class ChunkRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)
    chunk_sha256: str = Field(pattern=SHA256_PATTERN)
    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    fingerprint: str = Field(pattern=SHA256_PATTERN)
    text: str


def chunk_document(
    document_id: str,
    text: str,
    configuration: ChunkConfiguration,
) -> list[ChunkRecord]:
    """Split text using stable offsets and forward progress on every chunk."""

    chunks: list[ChunkRecord] = []
    start = 0
    index = 0
    while start < len(text):
        target_end = min(len(text), start + configuration.chunk_size)
        end = _separator_end(
            text,
            start=start,
            target_end=target_end,
            strategy=configuration.separator_strategy,
        )
        if end <= start:
            end = target_end
        chunk_text = text[start:end]
        content_hash = sha256_bytes(chunk_text.encode("utf-8"))
        chunk_fingerprint = fingerprint(
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "chunk_sha256": content_hash,
                "character_start": start,
                "character_end": end,
                "chunk_index": index,
                "chunk_configuration_fingerprint": configuration.fingerprint,
            }
        )
        chunks.append(
            ChunkRecord(
                chunk_id=f"chunk-{chunk_fingerprint}",
                chunk_sha256=content_hash,
                document_id=document_id,
                character_start=start,
                character_end=end,
                chunk_index=index,
                fingerprint=chunk_fingerprint,
                text=chunk_text,
            )
        )
        if end == len(text):
            break
        next_start = end - configuration.overlap
        start = next_start if next_start > start else end
        index += 1
    return chunks


def _separator_end(
    text: str,
    *,
    start: int,
    target_end: int,
    strategy: SeparatorStrategy,
) -> int:
    if strategy == "character" or target_end == len(text):
        return target_end
    separator = "\n\n" if strategy == "paragraph" else "\n"
    minimum = start + max(1, (target_end - start) // 2)
    position = text.rfind(separator, minimum, target_end + 1)
    return position + len(separator) if position >= 0 else target_end
