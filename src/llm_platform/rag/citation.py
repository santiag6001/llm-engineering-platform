"""Structured retrieval provenance and citation verification."""

from __future__ import annotations

from pydantic import Field, model_validator

from llm_platform.rag.document import (
    CHUNK_ID_PATTERN,
    DOCUMENT_ID_PATTERN,
    StrictModel,
)
from llm_platform.rag.retriever import RetrievalResult


class Citation(StrictModel):
    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)
    score: float = Field(ge=-1, le=1)

    @model_validator(mode="after")
    def range_is_ordered(self) -> Citation:
        if self.character_end < self.character_start:
            raise ValueError("citation character range must be ordered")
        return self


def citations_for(results: list[RetrievalResult]) -> list[Citation]:
    """Create one structured citation for every retrieved chunk."""

    return [
        Citation(
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            character_start=result.character_start,
            character_end=result.character_end,
            score=result.score,
        )
        for result in results
    ]


def citation_correctness(
    citations: list[Citation],
    results: list[RetrievalResult],
) -> float:
    """Return the fraction of citations that exactly match retrieval provenance."""

    if not citations:
        return 1.0 if not results else 0.0
    expected = {
        (
            result.document_id,
            result.chunk_id,
            result.character_start,
            result.character_end,
            result.score,
        )
        for result in results
    }
    correct = sum(
        (
            citation.document_id,
            citation.chunk_id,
            citation.character_start,
            citation.character_end,
            citation.score,
        )
        in expected
        for citation in citations
    )
    return correct / len(citations)
