from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from llm_platform.rag.chunking import ChunkConfiguration, chunk_document
from llm_platform.rag.citation import Citation, citation_correctness, citations_for
from llm_platform.rag.context import ContextConfiguration, build_context
from llm_platform.rag.embedding import (
    EmbeddingConfiguration,
    LocalHashingEmbedder,
)
from llm_platform.rag.evaluation import (
    RetrievalDataset,
    RetrievalEvaluationCase,
    evaluate_retrieval,
    experiment_metadata,
)
from llm_platform.rag.index import LocalVectorIndex
from llm_platform.rag.loader import (
    DocumentRegistry,
    DuplicateDocumentError,
)
from llm_platform.rag.retriever import Retriever, RetrieverConfiguration

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _registry(tmp_path: Path) -> tuple[DocumentRegistry, list[str]]:
    registry = DocumentRegistry(tmp_path / "rag", now=lambda: NOW)
    document_ids = []
    for name, text in (
        ("alpha.txt", "alpha retrieval systems preserve provenance"),
        ("beta.txt", "beta streaming systems preserve backpressure"),
    ):
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        document_ids.append(registry.register(path, logical_name=name).document_id)
    return registry, document_ids


def _index(tmp_path: Path) -> tuple[DocumentRegistry, LocalVectorIndex]:
    registry, _ = _registry(tmp_path)
    index = LocalVectorIndex(tmp_path / "rag" / "index.json", now=lambda: NOW)
    index.build(
        registry,
        chunk_configuration=ChunkConfiguration(
            chunk_size=200,
            overlap=0,
            separator_strategy="character",
        ),
        embedding_configuration=EmbeddingConfiguration(dimension=64),
    )
    return registry, index


def test_document_registration_is_immutable_and_duplicate_free(
    tmp_path: Path,
) -> None:
    path = tmp_path / "document.txt"
    path.write_text("deterministic content", encoding="utf-8")
    registry = DocumentRegistry(tmp_path / "rag", now=lambda: NOW)

    record = registry.register(path, logical_name="logical-doc")

    assert record.document_id == f"doc-{record.sha256}"
    assert record.byte_size == len(path.read_bytes())
    assert record.ingestion_timestamp == NOW
    assert registry.content_text(record.document_id) == "deterministic content"
    assert len(record.fingerprint) == 64
    with pytest.raises(DuplicateDocumentError):
        registry.register(path, logical_name="another-name")
    assert registry.list_documents() == [record]


def test_chunking_is_reproducible_and_configuration_sensitive() -> None:
    document_id = f"doc-{'a' * 64}"
    text = "first paragraph\n\nsecond paragraph\n\nthird paragraph"
    configuration = ChunkConfiguration(
        chunk_size=25,
        overlap=5,
        separator_strategy="paragraph",
    )

    left = chunk_document(document_id, text, configuration)
    right = chunk_document(document_id, text, configuration)
    changed = chunk_document(
        document_id,
        text,
        configuration.model_copy(update={"overlap": 4}),
    )

    assert left == right
    assert [chunk.chunk_index for chunk in left] == list(range(len(left)))
    assert all(
        text[chunk.character_start : chunk.character_end] == chunk.text
        for chunk in left
    )
    assert [chunk.fingerprint for chunk in left] != [
        chunk.fingerprint for chunk in changed
    ]


def test_embedding_metadata_and_vectors_are_stable() -> None:
    configuration = EmbeddingConfiguration(dimension=32)
    embedder = LocalHashingEmbedder(configuration)

    first = embedder.embed("Local CPU embedding")

    assert first == embedder.embed("Local CPU embedding")
    assert len(first) == 32
    assert embedder.metadata.model == "local-hashing-embedding"
    assert embedder.metadata.dimension == 32
    assert embedder.metadata.fingerprint == configuration.fingerprint


def test_index_rebuild_is_identical_and_entries_are_stably_ordered(
    tmp_path: Path,
) -> None:
    registry, _ = _registry(tmp_path)
    index = LocalVectorIndex(tmp_path / "rag" / "index.json", now=lambda: NOW)
    chunks = ChunkConfiguration(chunk_size=20, overlap=3)
    embeddings = EmbeddingConfiguration(dimension=32)

    first = index.build(
        registry,
        chunk_configuration=chunks,
        embedding_configuration=embeddings,
    )
    second = index.build(
        registry,
        chunk_configuration=chunks,
        embedding_configuration=embeddings,
    )

    assert first.metadata.index_fingerprint == second.metadata.index_fingerprint
    assert first.entries == second.entries
    assert second.metadata.embedding_metadata.dimension == 32
    keys = [
        (entry.chunk.document_id, entry.chunk.chunk_index) for entry in second.entries
    ]
    assert keys == sorted(keys)
    assert index.load() == second


def test_retrieval_order_threshold_mmr_context_and_citations(
    tmp_path: Path,
) -> None:
    _, index = _index(tmp_path)
    configuration = RetrieverConfiguration(top_k=2)
    results = Retriever(index, configuration).retrieve("alpha retrieval provenance")

    assert [result.rank for result in results] == [1, 2]
    assert results[0].score >= results[1].score
    assert results[0].text.startswith("alpha")
    assert (
        Retriever(
            index,
            configuration.model_copy(update={"score_threshold": 0.99}),
        ).retrieve("unmatched query")
        == []
    )
    mmr_results = Retriever(
        index,
        configuration.model_copy(update={"mmr": True, "mmr_lambda": 0.7}),
    ).retrieve("systems preserve")
    assert [result.rank for result in mmr_results] == [1, 2]

    context = build_context(
        results,
        ContextConfiguration(separator="\n--context--\n"),
    )
    assert context.chunk_ordering == [result.chunk_id for result in results]
    assert context.token_estimate > 0
    assert context == build_context(
        results,
        ContextConfiguration(separator="\n--context--\n"),
    )
    citations = citations_for(results)
    assert citation_correctness(citations, results) == 1
    invalid = Citation(
        **(
            citations[0].model_dump()
            | {"character_end": citations[0].character_end + 1}
        )
    )
    assert citation_correctness([invalid], results) == 0


def test_retrieval_evaluation_metrics_and_experiment_metadata(
    tmp_path: Path,
) -> None:
    _, index = _index(tmp_path)
    target = Retriever(index, RetrieverConfiguration(top_k=1)).retrieve(
        "alpha retrieval provenance"
    )[0]
    dataset = RetrievalDataset(
        cases=[
            RetrievalEvaluationCase(
                id="hit",
                query="alpha retrieval provenance",
                relevant_chunk_ids=[target.chunk_id],
            ),
            RetrievalEvaluationCase(
                id="miss",
                query="beta streaming backpressure",
                relevant_chunk_ids=[target.chunk_id],
            ),
        ]
    )

    report = evaluate_retrieval(
        index,
        dataset,
        RetrieverConfiguration(top_k=1),
    )
    metadata = experiment_metadata(index, report)

    assert report.retrieval_metrics.precision_at_k == pytest.approx(0.5)
    assert report.retrieval_metrics.recall_at_k == pytest.approx(0.5)
    assert report.retrieval_metrics.mrr == pytest.approx(0.5)
    assert report.retrieval_metrics.hit_rate == pytest.approx(0.5)
    assert report.citation_metrics.citation_correctness == 1
    assert report.citation_metrics.context_utilization == pytest.approx(0.5)
    assert metadata.index_fingerprint == report.index_fingerprint
    assert metadata.chunk_fingerprints == index.load().metadata.chunk_fingerprints


def test_retrieval_dataset_json_shape_is_strict(tmp_path: Path) -> None:
    _, index = _index(tmp_path)
    target = index.load().entries[0].chunk.chunk_id
    payload = RetrievalDataset(
        cases=[
            RetrievalEvaluationCase(
                id="case",
                query="query",
                relevant_chunk_ids=[target],
            )
        ]
    )
    path = tmp_path / "retrieval.json"
    path.write_text(
        json.dumps(payload.model_dump(mode="json")),
        encoding="utf-8",
    )
    assert json.loads(path.read_text())["schema_version"] == "1.0"
