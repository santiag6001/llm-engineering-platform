"""Versioned deterministic local CPU embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal

from pydantic import Field

from llm_platform.rag.document import SHA256_PATTERN, StrictModel, fingerprint

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class EmbeddingConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    model: Literal["local-hashing-embedding"] = "local-hashing-embedding"
    model_version: Literal["1.0"] = "1.0"
    dimension: int = Field(default=256, ge=8, le=4096)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class EmbeddingMetadata(StrictModel):
    model: str
    model_version: str
    dimension: int
    fingerprint: str = Field(pattern=SHA256_PATTERN)


class LocalHashingEmbedder:
    """Map normalized lexical features into a fixed CPU vector without downloads."""

    def __init__(self, configuration: EmbeddingConfiguration) -> None:
        self.configuration = configuration

    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata(
            model=self.configuration.model,
            model_version=self.configuration.model_version,
            dimension=self.configuration.dimension,
            fingerprint=self.configuration.fingerprint,
        )

    def embed(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self.configuration.dimension
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % len(vector)
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude:
            vector = [value / magnitude for value in vector]
        return tuple(vector)
