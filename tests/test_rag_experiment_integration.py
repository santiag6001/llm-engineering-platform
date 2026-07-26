from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from llm_platform.experiments.models import EnvironmentMetadata, SourceMetadata
from llm_platform.experiments.registry import ExperimentRegistry
from llm_platform.experiments.runner import ExperimentConfiguration, ExperimentRunner


@pytest.mark.asyncio
async def test_experiment_manifest_binds_rag_provenance(tmp_path: Path) -> None:
    dataset = tmp_path / "evaluation.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "case",
                "category": "rag",
                "messages": [{"role": "user", "content": "question"}],
                "expected": {"exact_match": "answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rag_metadata = tmp_path / "rag-metadata.json"
    rag_metadata.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "document_fingerprint": "d" * 64,
                "chunk_configuration": {
                    "schema_version": "1.0",
                    "chunk_size": 100,
                    "overlap": 10,
                    "separator_strategy": "paragraph",
                },
                "chunk_fingerprints": ["c" * 64],
                "embedding_configuration": {
                    "schema_version": "1.0",
                    "model": "local-hashing-embedding",
                    "model_version": "1.0",
                    "dimension": 64,
                },
                "index_fingerprint": "b" * 64,
                "retriever_configuration": {
                    "schema_version": "1.0",
                    "top_k": 1,
                    "score_threshold": None,
                    "mmr": False,
                    "mmr_lambda": 0.5,
                },
                "retrieval_metrics": {
                    "precision_at_k": 1.0,
                    "recall_at_k": 1.0,
                    "mrr": 1.0,
                    "hit_rate": 1.0,
                },
                "citation_metrics": {
                    "citation_correctness": 1.0,
                    "context_utilization": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    def response(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "backend",
                "choices": [{"message": {"content": "answer"}}],
            },
        )

    environment = EnvironmentMetadata(
        python_version="3.12",
        python_implementation="CPython",
        operating_system="Linux",
        platform_release="test",
        architecture="x86_64",
        project_version="0.1.0",
        container="none",
        dependencies={},
        environment_fingerprint="e" * 64,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        result = await ExperimentRunner(
            ExperimentRegistry(tmp_path / "experiments"),
            client=client,
            now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            run_id_factory=lambda _fingerprint: "rag-experiment",
            source_collector=lambda: SourceMetadata(git_commit="a" * 40),
            environment_collector=lambda: environment,
        ).run(
            ExperimentConfiguration(
                dataset_path=dataset,
                base_url="http://offline.test",
                requested_model="local-model",
                rag_metadata_file=rag_metadata,
            )
        )

    assert result.exit_code == 0
    assert result.manifest.rag is not None
    assert result.manifest.rag.document_fingerprint == "d" * 64
    assert result.manifest.rag.retrieval_metrics.mrr == 1
    assert result.manifest.reproduction.rag_metadata_path == rag_metadata.name
    persisted = json.loads(
        (
            tmp_path / "experiments" / "runs" / "rag-experiment" / "manifest.json"
        ).read_text()
    )
    assert persisted["rag"]["index_fingerprint"] == "b" * 64
