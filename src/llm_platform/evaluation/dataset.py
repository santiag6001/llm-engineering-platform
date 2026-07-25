"""Versioned JSONL evaluation dataset loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from llm_platform.evaluation.models import EvaluationCase, EvaluationDataset

MAX_DATASET_BYTES = 16 * 1024 * 1024
MAX_DATASET_CASES = 10_000


class DatasetValidationError(ValueError):
    """A dataset could not be parsed or did not satisfy its schema."""


def load_dataset(path: Path) -> EvaluationDataset:
    """Load and strictly validate a UTF-8 JSONL dataset."""

    try:
        if path.stat().st_size > MAX_DATASET_BYTES:
            raise DatasetValidationError(
                f"dataset exceeds {MAX_DATASET_BYTES} byte limit"
            )
        content = path.read_bytes()
    except OSError as exc:
        raise DatasetValidationError(f"could not read dataset: {path}") from exc

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("dataset must be valid UTF-8") from exc

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if len(cases) >= MAX_DATASET_CASES:
            raise DatasetValidationError(
                f"dataset exceeds {MAX_DATASET_CASES} case limit"
            )
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(
                f"malformed JSON on line {line_number}"
            ) from exc
        try:
            case = EvaluationCase.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors(include_url=False)[0]
            location = ".".join(str(part) for part in first["loc"])
            raise DatasetValidationError(
                f"invalid case on line {line_number} at {location}: {first['msg']}"
            ) from exc
        if case.id in seen_ids:
            raise DatasetValidationError(f"duplicate case id: {case.id}")
        seen_ids.add(case.id)
        cases.append(case)

    if not cases:
        raise DatasetValidationError("dataset must contain at least one case")

    return EvaluationDataset(
        path=path.as_posix(),
        content_hash=hashlib.sha256(content).hexdigest(),
        cases=cases,
    )
