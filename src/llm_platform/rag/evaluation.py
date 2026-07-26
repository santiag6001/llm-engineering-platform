"""Deterministic offline retrieval, citation, and context evaluation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, ValidationError

from llm_platform.rag.chunking import ChunkConfiguration
from llm_platform.rag.citation import citation_correctness, citations_for
from llm_platform.rag.context import build_context
from llm_platform.rag.document import (
    CHUNK_ID_PATTERN,
    SHA256_PATTERN,
    StrictModel,
)
from llm_platform.rag.embedding import EmbeddingConfiguration
from llm_platform.rag.index import LocalVectorIndex
from llm_platform.rag.retriever import Retriever, RetrieverConfiguration

MAX_RETRIEVAL_DATASET_BYTES = 16 * 1024 * 1024
SHA256Value = Annotated[str, Field(pattern=SHA256_PATTERN)]


class RetrievalDatasetError(ValueError):
    """A retrieval dataset was malformed or unsupported."""


class RetrievalEvaluationCase(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    query: str = Field(min_length=1, max_length=32_768)
    relevant_chunk_ids: list[str] = Field(min_length=1, max_length=256)


class RetrievalDataset(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    cases: list[RetrievalEvaluationCase] = Field(min_length=1, max_length=10_000)


class RetrievalCaseMetrics(StrictModel):
    case_id: str
    retrieved_chunk_ids: list[str]
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    hit: bool
    citation_correctness: float = Field(ge=0, le=1)
    context_utilization: float = Field(ge=0, le=1)
    context_fingerprint: SHA256Value


class RetrievalMetrics(StrictModel):
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    hit_rate: float = Field(ge=0, le=1)


class CitationMetrics(StrictModel):
    citation_correctness: float = Field(ge=0, le=1)
    context_utilization: float = Field(ge=0, le=1)


class RetrievalEvaluationReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    index_fingerprint: SHA256Value
    retriever_configuration: RetrieverConfiguration
    retrieval_metrics: RetrievalMetrics
    citation_metrics: CitationMetrics
    cases: list[RetrievalCaseMetrics]


class RAGExperimentMetadata(StrictModel):
    """Portable RAG provenance accepted by Phase 6 experiment manifests."""

    schema_version: Literal["1.0"] = "1.0"
    document_fingerprint: SHA256Value
    chunk_configuration: ChunkConfiguration
    chunk_fingerprints: list[SHA256Value]
    embedding_configuration: EmbeddingConfiguration
    index_fingerprint: SHA256Value
    retriever_configuration: RetrieverConfiguration
    retrieval_metrics: RetrievalMetrics
    citation_metrics: CitationMetrics


def load_retrieval_dataset(path: Path) -> RetrievalDataset:
    try:
        if path.stat().st_size > MAX_RETRIEVAL_DATASET_BYTES:
            raise RetrievalDatasetError(
                f"dataset exceeds {MAX_RETRIEVAL_DATASET_BYTES} byte limit"
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
        dataset = RetrievalDataset.model_validate(raw)
    except OSError as exc:
        raise RetrievalDatasetError("could not read retrieval dataset") from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RetrievalDatasetError("retrieval dataset is malformed") from exc
    ids = [case.id for case in dataset.cases]
    if len(ids) != len(set(ids)):
        raise RetrievalDatasetError("retrieval case IDs must be unique")
    for case in dataset.cases:
        if len(case.relevant_chunk_ids) != len(set(case.relevant_chunk_ids)):
            raise RetrievalDatasetError(f"relevant chunk IDs must be unique: {case.id}")
        if any(not _valid_chunk_id(chunk_id) for chunk_id in case.relevant_chunk_ids):
            raise RetrievalDatasetError(f"invalid relevant chunk ID: {case.id}")
    return dataset


def evaluate_retrieval(
    index: LocalVectorIndex,
    dataset: RetrievalDataset,
    configuration: RetrieverConfiguration,
) -> RetrievalEvaluationReport:
    manifest = index.load()
    indexed_chunks = {entry.chunk.chunk_id for entry in manifest.entries}
    retriever = Retriever(index, configuration)
    cases: list[RetrievalCaseMetrics] = []
    for case in dataset.cases:
        unknown = set(case.relevant_chunk_ids).difference(indexed_chunks)
        if unknown:
            raise RetrievalDatasetError(
                f"case references chunks outside the index: {case.id}"
            )
        results = retriever.retrieve(case.query)
        retrieved = [result.chunk_id for result in results]
        relevant = set(case.relevant_chunk_ids)
        relevant_ranks = [
            rank
            for rank, chunk_id in enumerate(retrieved, start=1)
            if chunk_id in relevant
        ]
        relevant_count = len(relevant.intersection(retrieved))
        citations = citations_for(results)
        context = build_context(results)
        cases.append(
            RetrievalCaseMetrics(
                case_id=case.id,
                retrieved_chunk_ids=retrieved,
                precision_at_k=relevant_count / configuration.top_k,
                recall_at_k=relevant_count / len(relevant),
                reciprocal_rank=(1 / relevant_ranks[0] if relevant_ranks else 0),
                hit=bool(relevant_ranks),
                citation_correctness=citation_correctness(citations, results),
                context_utilization=(relevant_count / len(results) if results else 0.0),
                context_fingerprint=context.context_fingerprint,
            )
        )
    count = len(cases)
    metadata = manifest.metadata
    return RetrievalEvaluationReport(
        index_fingerprint=metadata.index_fingerprint,
        retriever_configuration=configuration,
        retrieval_metrics=RetrievalMetrics(
            precision_at_k=sum(case.precision_at_k for case in cases) / count,
            recall_at_k=sum(case.recall_at_k for case in cases) / count,
            mrr=sum(case.reciprocal_rank for case in cases) / count,
            hit_rate=sum(case.hit for case in cases) / count,
        ),
        citation_metrics=CitationMetrics(
            citation_correctness=(
                sum(case.citation_correctness for case in cases) / count
            ),
            context_utilization=(
                sum(case.context_utilization for case in cases) / count
            ),
        ),
        cases=cases,
    )


def experiment_metadata(
    index: LocalVectorIndex,
    report: RetrievalEvaluationReport,
) -> RAGExperimentMetadata:
    metadata = index.load().metadata
    return RAGExperimentMetadata(
        document_fingerprint=metadata.document_fingerprint,
        chunk_configuration=metadata.chunk_configuration,
        chunk_fingerprints=metadata.chunk_fingerprints,
        embedding_configuration=metadata.embedding_configuration,
        index_fingerprint=metadata.index_fingerprint,
        retriever_configuration=report.retriever_configuration,
        retrieval_metrics=report.retrieval_metrics,
        citation_metrics=report.citation_metrics,
    )


def _valid_chunk_id(value: str) -> bool:
    return re.fullmatch(CHUNK_ID_PATTERN, value) is not None
