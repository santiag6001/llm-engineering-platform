"""Strict identities and persisted models for local RAG documents."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
DOCUMENT_ID_PATTERN = r"^doc-[0-9a-f]{64}$"
CHUNK_ID_PATTERN = r"^chunk-[0-9a-f]{64}$"


class StrictModel(BaseModel):
    """Forbid silent persisted-schema drift."""

    model_config = ConfigDict(extra="forbid")


class DocumentRecord(StrictModel):
    """Immutable metadata for one registered source document."""

    schema_version: Literal["1.0"] = "1.0"
    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    sha256: str = Field(pattern=SHA256_PATTERN)
    logical_name: str = Field(min_length=1, max_length=256)
    byte_size: int = Field(ge=0, le=16 * 1024 * 1024)
    content_type: str = Field(min_length=1, max_length=128)
    ingestion_timestamp: datetime
    fingerprint: str = Field(pattern=SHA256_PATTERN)

    @field_validator("ingestion_timestamp")
    @classmethod
    def timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ingestion_timestamp must include a timezone")
        return value


class DocumentRegistryManifest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    documents: list[DocumentRecord] = Field(default_factory=list, max_length=10_000)


def canonical_json(value: object) -> bytes:
    """Return canonical UTF-8 JSON bytes for stable content identities."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def fingerprint(value: object) -> str:
    return sha256_bytes(canonical_json(value))


def document_fingerprint(
    *,
    content_sha256: str,
    logical_name: str,
    byte_size: int,
    content_type: str,
) -> str:
    """Identify stable document inputs while excluding ingestion time."""

    return fingerprint(
        {
            "schema_version": "1.0",
            "sha256": content_sha256,
            "logical_name": logical_name,
            "byte_size": byte_size,
            "content_type": content_type,
        }
    )
