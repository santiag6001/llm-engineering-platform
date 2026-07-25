#!/usr/bin/env python3
"""Deterministic offline experiment registry and artifact smoke test."""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from llm_platform.experiments.comparison import compare_registered_runs
from llm_platform.experiments.models import EnvironmentMetadata, SourceMetadata
from llm_platform.experiments.registry import ExperimentRegistry
from llm_platform.experiments.runner import ExperimentConfiguration, ExperimentRunner


def _response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "offline-backend",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    )


async def run_smoke(root: Path) -> None:
    dataset_path = root / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "id": "offline-case",
                "category": "smoke",
                "messages": [{"role": "user", "content": "prompt"}],
                "expected": {"exact_match": "answer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = ExperimentRegistry(root / "experiments")
    environment = EnvironmentMetadata(
        python_version="3.12.11",
        python_implementation="CPython",
        operating_system="Linux",
        platform_release="offline-smoke",
        architecture="x86_64",
        project_version="0.1.0",
        container="none",
        dependencies={},
        environment_fingerprint="e" * 64,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(_response)) as client:
        result = await ExperimentRunner(
            registry,
            client=client,
            now=lambda: datetime(2026, 1, 1, tzinfo=UTC),
            run_id_factory=lambda _fingerprint: "offline-smoke-run",
            source_collector=lambda: SourceMetadata(
                git_commit="a" * 40,
                git_dirty=False,
                branch="offline",
            ),
            environment_collector=lambda: environment,
        ).run(
            ExperimentConfiguration(
                dataset_path=dataset_path,
                base_url="http://offline.test",
                requested_model="offline-model",
            )
        )
    if result.exit_code != 0:
        raise RuntimeError("offline experiment did not complete successfully")
    registry.set_alias("latest", result.manifest.run_id)
    registry.verify("latest")
    comparison = compare_registered_runs(registry, "latest", result.manifest.run_id)
    if not comparison.identical:
        raise RuntimeError("self-comparison unexpectedly differed")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="llm-experiment-smoke-") as directory:
        asyncio.run(run_smoke(Path(directory)))
    print("offline experiment registry smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
