"""Audit comparison for two registered experiment runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from llm_platform.experiments.models import (
    Difference,
    DifferenceCategory,
    ExperimentComparison,
    ExperimentManifest,
)
from llm_platform.experiments.registry import ExperimentRegistry


def compare_manifests(
    left: ExperimentManifest, right: ExperimentManifest
) -> ExperimentComparison:
    """Classify material run differences without creating a pass/fail policy."""

    differences: dict[DifferenceCategory, list[Difference]] = {
        "input_configuration": [],
        "source_code": [],
        "environment": [],
        "quality": [],
        "performance": [],
        "artifacts": [],
    }
    _record(
        differences["input_configuration"],
        "experiment_fingerprint",
        left.experiment_fingerprint,
        right.experiment_fingerprint,
    )
    _record(
        differences["input_configuration"],
        "dataset.sha256",
        left.dataset.sha256,
        right.dataset.sha256,
    )
    _record(
        differences["input_configuration"],
        "model.requested",
        left.model.requested,
        right.model.requested,
    )
    _record(
        differences["input_configuration"],
        "model.backend_observed",
        left.model.backend_observed,
        right.model.backend_observed,
    )
    _record(
        differences["input_configuration"],
        "generation",
        left.generation.model_dump(mode="json"),
        right.generation.model_dump(mode="json"),
    )
    _record(
        differences["source_code"],
        "source.git_commit",
        left.source.git_commit,
        right.source.git_commit,
    )
    _record(
        differences["source_code"],
        "source.git_dirty",
        left.source.git_dirty,
        right.source.git_dirty,
    )
    _record(
        differences["environment"],
        "environment.environment_fingerprint",
        left.environment.environment_fingerprint,
        right.environment.environment_fingerprint,
    )
    _record(
        differences["environment"],
        "deployment",
        (
            left.deployment.model_dump(mode="json")
            if left.deployment is not None
            else None
        ),
        (
            right.deployment.model_dump(mode="json")
            if right.deployment is not None
            else None
        ),
    )
    for field in ("pass_rate", "error_rate", "prompt_tokens", "completion_tokens"):
        _record(
            differences["quality"],
            f"aggregate_results.{field}",
            getattr(left.aggregate_results, field),
            getattr(right.aggregate_results, field),
        )
    _record(
        differences["quality"],
        "regression.decision",
        left.regression.decision,
        right.regression.decision,
    )
    for field in ("p50_duration_seconds", "p95_duration_seconds"):
        _record(
            differences["performance"],
            f"aggregate_results.{field}",
            getattr(left.aggregate_results, field),
            getattr(right.aggregate_results, field),
        )
    _record(
        differences["artifacts"],
        "artifacts",
        _artifact_identity(left),
        _artifact_identity(right),
    )
    return ExperimentComparison(
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        identical=not any(differences.values()),
        differences=differences,
    )


def compare_registered_runs(
    registry: ExperimentRegistry, left: str, right: str
) -> ExperimentComparison:
    return compare_manifests(registry.get(left), registry.get(right))


def render_comparison_json(comparison: ExperimentComparison) -> str:
    return (
        json.dumps(
            comparison.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_comparison_markdown(comparison: ExperimentComparison) -> str:
    lines = [
        "# Experiment Comparison",
        "",
        f"- Left: `{comparison.left_run_id}`",
        f"- Right: `{comparison.right_run_id}`",
        f"- Identical audited fields: `{'yes' if comparison.identical else 'no'}`",
        "",
        "This is an audit view. Only configured regression gates determine pass/fail.",
    ]
    labels: dict[DifferenceCategory, str] = {
        "input_configuration": "Input and configuration changes",
        "source_code": "Source-code changes",
        "environment": "Environment changes",
        "quality": "Quality changes",
        "performance": "Performance changes",
        "artifacts": "Artifact changes",
    }
    for category, label in labels.items():
        lines.extend(["", f"## {label}", ""])
        changes = comparison.differences[category]
        if not changes:
            lines.append("None.")
            continue
        for change in changes:
            lines.append(
                f"- `{change.field}`: `{_display(change.left)}` → "
                f"`{_display(change.right)}`"
            )
    return "\n".join(lines) + "\n"


def write_comparison(
    comparison: ExperimentComparison, output_directory: Path
) -> tuple[Path, Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    if output_directory.is_symlink():
        raise ValueError("comparison output directory must not be a symlink")
    stem = f"{comparison.left_run_id}--{comparison.right_run_id}"
    json_path = output_directory / f"{stem}.json"
    markdown_path = output_directory / f"{stem}.md"
    _atomic_replace(json_path, render_comparison_json(comparison).encode())
    _atomic_replace(markdown_path, render_comparison_markdown(comparison).encode())
    return json_path, markdown_path


def _record(changes: list[Difference], field: str, left: object, right: object) -> None:
    if left != right:
        changes.append(Difference(field=field, left=left, right=right))


def _artifact_identity(manifest: ExperimentManifest) -> dict[str, str]:
    return {artifact.path: artifact.sha256 for artifact in manifest.artifacts}


def _display(value: object) -> str:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return rendered if len(rendered) <= 160 else rendered[:159] + "…"


def _atomic_replace(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".comparison-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
