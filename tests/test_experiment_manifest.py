from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llm_platform.experiments.models import ExperimentManifest, FailureMetadata
from tests.experiment_helpers import manifest


def test_valid_manifest_round_trips_strictly() -> None:
    value, _ = manifest()
    decoded = ExperimentManifest.model_validate_json(value.model_dump_json())
    assert decoded == value
    assert decoded.schema_version == "1.0"


def test_unknown_or_wrong_schema_fields_are_rejected() -> None:
    value, _ = manifest()
    raw = value.model_dump(mode="json")
    raw["unknown"] = True
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)
    raw.pop("unknown")
    raw["schema_version"] = "2.0"
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_missing_required_manifest_field_is_rejected() -> None:
    value, _ = manifest()
    raw = value.model_dump(mode="json")
    del raw["dataset"]
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_failure_messages_and_artifact_paths_are_bounded_and_portable() -> None:
    with pytest.raises(ValidationError):
        FailureMetadata(message="x" * 241)
    value, _ = manifest()
    raw = json.loads(value.model_dump_json())
    raw["artifacts"][0]["path"] = "../escape"
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)


def test_environment_has_no_arbitrary_or_secret_fields() -> None:
    value, _ = manifest()
    raw = value.model_dump(mode="json")
    raw["environment"]["API_TOKEN"] = "secret"
    with pytest.raises(ValidationError):
        ExperimentManifest.model_validate(raw)
