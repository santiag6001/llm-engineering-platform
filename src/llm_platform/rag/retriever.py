"""Stable top-K and optional maximal-marginal-relevance retrieval."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from llm_platform.rag.document import (
    CHUNK_ID_PATTERN,
    DOCUMENT_ID_PATTERN,
    StrictModel,
    fingerprint,
)
from llm_platform.rag.embedding import LocalHashingEmbedder
from llm_platform.rag.index import IndexEntry, LocalVectorIndex


class RetrieverConfiguration(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float | None = Field(default=None, ge=-1, le=1)
    mmr: bool = False
    mmr_lambda: float = Field(default=0.5, ge=0, le=1)

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.model_dump(mode="json"))


class RetrievalResult(StrictModel):
    chunk_id: str = Field(pattern=CHUNK_ID_PATTERN)
    document_id: str = Field(pattern=DOCUMENT_ID_PATTERN)
    score: float = Field(ge=-1, le=1)
    rank: int = Field(ge=1)
    character_start: int = Field(ge=0)
    character_end: int = Field(ge=0)
    chunk_index: int = Field(ge=0)
    chunk_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str


class Retriever:
    def __init__(
        self,
        index: LocalVectorIndex,
        configuration: RetrieverConfiguration,
    ) -> None:
        self.index = index
        self.configuration = configuration

    def retrieve(self, query: str) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query must not be empty")
        manifest = self.index.load()
        embedder = LocalHashingEmbedder(manifest.metadata.embedding_configuration)
        query_vector = embedder.embed(query)
        scored = [
            (entry, _dot(query_vector, entry.vector)) for entry in manifest.entries
        ]
        threshold = self.configuration.score_threshold
        candidates = [
            item for item in scored if threshold is None or item[1] >= threshold
        ]
        candidates.sort(key=lambda item: (-item[1], item[0].chunk.chunk_id))
        if self.configuration.mmr:
            selected = _mmr(
                candidates,
                limit=self.configuration.top_k,
                weight=self.configuration.mmr_lambda,
            )
        else:
            selected = candidates[: self.configuration.top_k]
        return [
            RetrievalResult(
                chunk_id=entry.chunk.chunk_id,
                document_id=entry.chunk.document_id,
                score=score,
                rank=rank,
                character_start=entry.chunk.character_start,
                character_end=entry.chunk.character_end,
                chunk_index=entry.chunk.chunk_index,
                chunk_fingerprint=entry.chunk.fingerprint,
                text=entry.chunk.text,
            )
            for rank, (entry, score) in enumerate(selected, start=1)
        ]


def _mmr(
    candidates: list[tuple[IndexEntry, float]],
    *,
    limit: int,
    weight: float,
) -> list[tuple[IndexEntry, float]]:
    remaining = list(candidates)
    selected: list[tuple[IndexEntry, float]] = []
    while remaining and len(selected) < limit:
        best = min(
            remaining,
            key=lambda candidate: (
                -_mmr_score(candidate, selected, weight),
                candidate[0].chunk.chunk_id,
            ),
        )
        selected.append(best)
        remaining.remove(best)
    return selected


def _mmr_score(
    candidate: tuple[IndexEntry, float],
    selected: list[tuple[IndexEntry, float]],
    weight: float,
) -> float:
    redundancy = max(
        (_dot(candidate[0].vector, item[0].vector) for item in selected),
        default=0.0,
    )
    return weight * candidate[1] - (1.0 - weight) * redundancy


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))
