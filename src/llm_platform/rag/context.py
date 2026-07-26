"""Deterministic context assembly with a reproducible identity."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from llm_platform.rag.document import SHA256_PATTERN, StrictModel, fingerprint
from llm_platform.rag.retriever import RetrievalResult


class ContextConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    separator: str = Field(default="\n\n---\n\n", max_length=128)
    token_estimator: Literal["characters_divided_by_four"] = (
        "characters_divided_by_four"
    )


class ContextAssembly(StrictModel):
    text: str
    chunk_ordering: list[str]
    separator_policy: str
    token_estimate: int = Field(ge=0)
    context_fingerprint: str = Field(pattern=SHA256_PATTERN)


def build_context(
    results: list[RetrievalResult],
    configuration: ContextConfiguration | None = None,
) -> ContextAssembly:
    resolved = configuration or ContextConfiguration()
    text = resolved.separator.join(result.text for result in results)
    ordering = [result.chunk_id for result in results]
    token_estimate = math.ceil(len(text) / 4) if text else 0
    return ContextAssembly(
        text=text,
        chunk_ordering=ordering,
        separator_policy=resolved.separator,
        token_estimate=token_estimate,
        context_fingerprint=fingerprint(
            {
                "schema_version": "1.0",
                "chunk_ordering": ordering,
                "chunk_fingerprints": [result.chunk_fingerprint for result in results],
                "separator": resolved.separator,
                "token_estimator": resolved.token_estimator,
                "text": text,
            }
        ),
    )
