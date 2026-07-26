"""Command-line interface for the local experiment registry."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from llm_platform.evaluation.dataset import DatasetValidationError
from llm_platform.experiments.comparison import (
    compare_registered_runs,
    render_comparison_json,
    render_comparison_markdown,
    write_comparison,
)
from llm_platform.experiments.models import (
    DeploymentMetadata,
    GenerationConfiguration,
    RegressionGates,
)
from llm_platform.experiments.registry import (
    ExperimentRegistry,
    RegistryError,
    RegistryIntegrityError,
    RunNotFoundError,
)
from llm_platform.experiments.runner import (
    ExperimentConfiguration,
    ExperimentRunner,
)

EXIT_SUCCESS = 0
EXIT_REGRESSION_FAILED = 1
EXIT_INVALID_INPUT = 2
EXIT_OPERATIONAL_FAILURE = 3
EXIT_REGISTRY_INTEGRITY_FAILURE = 4
EXIT_NOT_FOUND = 5


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _build_parser()
    except ValueError:
        print("error: invalid experiment environment default", file=sys.stderr)
        return EXIT_INVALID_INPUT
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except RunNotFoundError as exc:
        print(f"error: {_safe_message(exc)}", file=sys.stderr)
        return EXIT_NOT_FOUND
    except (RegistryIntegrityError, RegistryError, OSError) as exc:
        print(f"error: {_safe_message(exc)}", file=sys.stderr)
        return EXIT_REGISTRY_INTEGRITY_FAILURE
    except (DatasetValidationError, ValidationError, ValueError) as exc:
        print(f"error: {_safe_message(exc)}", file=sys.stderr)
        return EXIT_INVALID_INPUT


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "run":
        return asyncio.run(_run_command(args))
    registry = ExperimentRegistry(args.registry_dir)
    if args.command == "list":
        return _list_command(registry, args)
    if args.command == "show":
        return _show_command(registry, args)
    if args.command == "compare":
        return _compare_command(registry, args)
    if args.command == "alias":
        return _alias_command(registry, args)
    if args.command == "verify":
        return _verify_command(registry, args)
    raise ValueError("unknown experiment command")


async def _run_command(args: argparse.Namespace) -> int:
    registry = ExperimentRegistry(args.registry_dir)
    if args.deployment_runtime is None and (
        args.deployment_image is not None or args.deployment_name is not None
    ):
        raise ValueError("deployment image or name requires --deployment-runtime")
    configuration = ExperimentConfiguration(
        dataset_path=args.dataset,
        dataset_identifier=args.dataset_identifier,
        base_url=args.base_url,
        requested_model=args.model,
        maximum_concurrency=args.max_concurrency,
        timeout_seconds=args.timeout_seconds,
        generation=GenerationConfiguration(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        baseline=args.baseline,
        regression_gates=_regression_gates(args),
        prompt_file=args.prompt_file,
        prompt_name=args.prompt_name,
        prompt_version=args.prompt_version,
        deployment=(
            DeploymentMetadata(
                runtime=args.deployment_runtime,
                image_reference=args.deployment_image,
                configuration_name=args.deployment_name,
            )
            if args.deployment_runtime is not None
            else None
        ),
        rag_metadata_file=args.rag_metadata,
    )
    result = await ExperimentRunner(registry).run(configuration)
    manifest = result.manifest
    print(
        f"experiment {manifest.status}: {manifest.run_id} "
        f"(fingerprint {manifest.experiment_fingerprint[:12]})"
    )
    print(f"run directory: {registry.runs_directory / manifest.run_id}")
    return result.exit_code


def _list_command(registry: ExperimentRegistry, args: argparse.Namespace) -> int:
    manifests = registry.list_runs(
        status=args.status,
        experiment_fingerprint=args.fingerprint,
        requested_model=args.model,
        git_commit=args.git_commit,
    )
    if args.json:
        print(
            json.dumps(
                [manifest.model_dump(mode="json") for manifest in manifests],
                indent=2,
                sort_keys=True,
            )
        )
    elif not manifests:
        print("no experiment runs")
    else:
        for manifest in manifests:
            print(
                f"{manifest.run_id}\t{manifest.status}\t"
                f"{manifest.model.requested}\t"
                f"{manifest.experiment_fingerprint[:12]}"
            )
    return EXIT_SUCCESS


def _show_command(registry: ExperimentRegistry, args: argparse.Namespace) -> int:
    manifest = registry.get(args.run)
    print(
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_SUCCESS


def _compare_command(registry: ExperimentRegistry, args: argparse.Namespace) -> int:
    comparison = compare_registered_runs(registry, args.left, args.right)
    json_path, markdown_path = write_comparison(comparison, args.output_dir)
    print(
        render_comparison_json(comparison)
        if args.json
        else render_comparison_markdown(comparison),
        end="",
    )
    destination = sys.stderr if args.json else sys.stdout
    print(f"JSON: {json_path}", file=destination)
    print(f"Markdown: {markdown_path}", file=destination)
    return EXIT_SUCCESS


def _alias_command(registry: ExperimentRegistry, args: argparse.Namespace) -> int:
    if args.alias_command == "set":
        registry.set_alias(args.alias, args.run_id)
        print(f"alias {args.alias} -> {args.run_id}")
        return EXIT_SUCCESS
    run_id = registry.show_alias(args.alias)
    print(run_id)
    return EXIT_SUCCESS


def _verify_command(registry: ExperimentRegistry, args: argparse.Namespace) -> int:
    manifest = registry.verify(args.run)
    print(f"verified {manifest.run_id}: {len(manifest.artifacts)} artifacts")
    return EXIT_SUCCESS


def _regression_gates(args: argparse.Namespace) -> RegressionGates:
    return RegressionGates(
        minimum_pass_rate=args.min_pass_rate,
        maximum_pass_rate_drop=args.max_pass_rate_drop,
        maximum_error_rate=args.max_error_rate,
        maximum_p95_latency_seconds=args.max_p95_latency_seconds,
        maximum_p95_latency_increase_percent=(args.max_p95_latency_increase_percent),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-experiment",
        description="Run and audit reproducible local LLM experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_list_parser(subparsers)
    show = subparsers.add_parser("show", help="show a run manifest by ID or alias")
    show.add_argument("run", help="run ID or alias")
    _registry_argument(show)

    compare = subparsers.add_parser(
        "compare", help="audit differences between two registered runs"
    )
    compare.add_argument("left", help="left run ID or alias")
    compare.add_argument("right", help="right run ID or alias")
    _registry_argument(compare)
    compare.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/comparisons"),
        help="directory for JSON and Markdown comparisons",
    )
    compare.add_argument("--json", action="store_true", help="print JSON")

    alias = subparsers.add_parser("alias", help="manage mutable local aliases")
    alias_subparsers = alias.add_subparsers(dest="alias_command", required=True)
    alias_set = alias_subparsers.add_parser("set", help="atomically set an alias")
    alias_set.add_argument("alias", help="bounded alias name")
    alias_set.add_argument("run_id", help="existing immutable run ID")
    _registry_argument(alias_set)
    alias_show = alias_subparsers.add_parser("show", help="resolve an alias")
    alias_show.add_argument("alias", help="bounded alias name")
    _registry_argument(alias_show)

    verify = subparsers.add_parser(
        "verify", help="verify a manifest and all artifact checksums"
    )
    verify.add_argument("run", help="run ID or alias")
    _registry_argument(verify)
    return parser


def _add_run_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    run = subparsers.add_parser(
        "run", help="evaluate and atomically register an experiment"
    )
    run.add_argument("--dataset", required=True, type=Path, help="JSONL dataset")
    run.add_argument("--dataset-identifier", help="stable logical dataset name")
    run.add_argument(
        "--base-url",
        default=os.getenv("LLM_PLATFORM_BASE_URL", "http://127.0.0.1:8000"),
        help="platform URL (credentials are forbidden)",
    )
    run.add_argument(
        "--model",
        default=os.getenv("LLM_PLATFORM_MODEL", "local-model"),
        help="requested public model",
    )
    _registry_argument(run)
    run.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("LLM_EVAL_MAX_CONCURRENCY", "1")),
        help="maximum simultaneous evaluation requests",
    )
    run.add_argument(
        "--timeout-seconds",
        "--timeout",
        type=float,
        default=float(os.getenv("LLM_EVAL_TIMEOUT_SECONDS", "120")),
        help="per-request timeout",
    )
    run.add_argument("--temperature", type=float, help="default temperature")
    run.add_argument("--max-tokens", type=int, help="default generation token limit")
    run.add_argument("--baseline", help="registered baseline run ID or alias")
    run.add_argument("--min-pass-rate", type=float)
    run.add_argument("--max-pass-rate-drop", type=float)
    run.add_argument("--max-error-rate", type=float)
    run.add_argument("--max-p95-latency-seconds", type=float)
    run.add_argument("--max-p95-latency-increase-percent", type=float)
    run.add_argument("--prompt-file", type=Path, help="shared system-prompt file")
    run.add_argument("--prompt-name", help="shared prompt logical name")
    run.add_argument("--prompt-version", help="shared prompt version")
    run.add_argument(
        "--deployment-runtime",
        choices=("host", "docker", "compose", "other"),
        help="optional execution deployment type",
    )
    run.add_argument("--deployment-image", help="optional bounded image reference")
    run.add_argument("--deployment-name", help="optional deployment configuration name")
    run.add_argument(
        "--rag-metadata",
        type=Path,
        help="RAG provenance JSON produced by llm-rag evaluate",
    )


def _add_list_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    list_parser = subparsers.add_parser("list", help="list registered runs")
    _registry_argument(list_parser)
    list_parser.add_argument(
        "--status", choices=("completed", "regression_failed", "failed")
    )
    list_parser.add_argument("--fingerprint")
    list_parser.add_argument("--model")
    list_parser.add_argument("--git-commit")
    list_parser.add_argument("--json", action="store_true", help="print JSON")


def _registry_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=Path("experiments"),
        help="local experiment registry root",
    )


def _safe_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_input=False)
        message = errors[0]["msg"] if errors else "validation failed"
    else:
        message = str(exc)
    bounded = " ".join(message.split())
    return bounded[:240] if bounded else "operation failed"


if __name__ == "__main__":
    raise SystemExit(main())
