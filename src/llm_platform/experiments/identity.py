"""Canonical experiment and run identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from llm_platform.experiments.models import (
    DatasetMetadata,
    DeploymentMetadata,
    EvaluationConfiguration,
    GenerationConfiguration,
    PromptIdentity,
    RAGExperimentMetadata,
    RegressionGates,
    SourceMetadata,
)


def canonical_json(value: object) -> bytes:
    """Serialize JSON-compatible values with stable ordering and no whitespace."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def experiment_fingerprint(
    *,
    dataset: DatasetMetadata,
    requested_model: str,
    generation: GenerationConfiguration,
    evaluation: EvaluationConfiguration,
    regression_gates: RegressionGates,
    baseline: str | None,
    prompt: PromptIdentity | None,
    source: SourceMetadata,
    deployment: DeploymentMetadata | None = None,
    rag: RAGExperimentMetadata | None = None,
    schema_version: str = "1.0",
) -> str:
    """Hash canonical experiment inputs, excluding execution and result fields."""

    inputs: Mapping[str, Any] = {
        "schema_version": schema_version,
        "dataset_sha256": dataset.sha256,
        "requested_model": requested_model,
        "generation": generation.model_dump(mode="json"),
        "evaluator": evaluation.evaluator.model_dump(mode="json"),
        "evaluation": {
            "concurrency": evaluation.concurrency,
            "timeout_seconds": evaluation.timeout_seconds,
        },
        "regression": {
            "baseline": baseline,
            "gates": regression_gates.model_dump(mode="json"),
        },
        "prompt": (
            {
                "logical_name": prompt.logical_name,
                "version": prompt.version,
                "content_sha256": prompt.content_sha256,
            }
            if prompt is not None
            else None
        ),
        "source_git_commit": source.git_commit,
        "deployment": (
            deployment.model_dump(mode="json") if deployment is not None else None
        ),
    }
    if rag is not None:
        inputs = {
            **inputs,
            "rag": rag.model_dump(mode="json"),
        }
    return sha256_bytes(canonical_json(inputs))


def create_run_id(
    fingerprint: str,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    nonce: Callable[[], str] = lambda: uuid4().hex[:8],
) -> str:
    """Return a sortable unique execution ID containing a short input digest."""

    timestamp = now().astimezone(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{fingerprint[:12]}-{nonce()}"
