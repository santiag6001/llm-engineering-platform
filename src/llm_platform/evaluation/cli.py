"""Command-line interface for evaluation and regression gating."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from llm_platform.evaluation.dataset import DatasetValidationError, load_dataset
from llm_platform.evaluation.models import EvaluationReport, RunnerConfiguration
from llm_platform.evaluation.regression import (
    RegressionThresholds,
    compare_reports,
    render_comparison_markdown,
)
from llm_platform.evaluation.reporting import (
    build_report,
    detect_git_metadata,
    write_report_files,
)
from llm_platform.evaluation.runner import EvaluationRunner

EXIT_SUCCESS = 0
EXIT_REGRESSION_FAILED = 1
EXIT_INVALID_INPUT = 2
EXIT_EVALUATION_ERRORS = 3


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _build_parser()
    except ValueError as exc:
        print(
            f"error: invalid evaluation environment default: {_safe_message(exc)}",
            file=sys.stderr,
        )
        return EXIT_INVALID_INPUT
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run_command(args))
        return _compare_command(args)
    except (DatasetValidationError, ValidationError, OSError, ValueError) as exc:
        print(f"error: {_safe_message(exc)}", file=sys.stderr)
        return EXIT_INVALID_INPUT


async def _run_command(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    configuration = RunnerConfiguration(
        base_url=args.base_url,
        model=args.model,
        request_timeout_seconds=args.timeout,
        maximum_concurrency=args.max_concurrency,
    )
    results = await EvaluationRunner(configuration).run(dataset)
    report = build_report(
        dataset,
        configuration,
        results,
        git_metadata=detect_git_metadata(),
    )
    json_path, markdown_path = write_report_files(report, args.output_dir)
    print(
        f"evaluation complete: {report.aggregate.passed_cases}/"
        f"{report.aggregate.total_cases} passed, "
        f"{report.aggregate.error_cases} errors"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return EXIT_EVALUATION_ERRORS if report.aggregate.error_cases else EXIT_SUCCESS


def _compare_command(args: argparse.Namespace) -> int:
    current = _load_report(args.current)
    baseline = _load_report(args.baseline)
    thresholds = RegressionThresholds(
        minimum_pass_rate=args.min_pass_rate,
        maximum_pass_rate_drop=args.max_pass_rate_drop,
        maximum_error_rate=args.max_error_rate,
        maximum_p95_latency_seconds=args.max_p95_latency_seconds,
        maximum_p95_latency_increase_percent=(args.max_p95_latency_increase_percent),
    )
    comparison = compare_reports(current, baseline, thresholds)
    print(render_comparison_markdown(comparison))
    return EXIT_SUCCESS if comparison.passed else EXIT_REGRESSION_FAILED


def _load_report(path: Path) -> EvaluationReport:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed report JSON: {path}") from exc
    return EvaluationReport.model_validate(raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Run deterministic LLM evaluations and regression gates.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="run a buffered evaluation against the platform"
    )
    run.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path")
    run.add_argument(
        "--base-url",
        default=os.getenv("LLM_PLATFORM_BASE_URL", "http://127.0.0.1:8000"),
        help="platform base URL",
    )
    run.add_argument(
        "--model",
        default=os.getenv("LLM_PLATFORM_MODEL", "local-model"),
        help="public model name",
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("LLM_EVAL_TIMEOUT_SECONDS", "120")),
        help="request timeout in seconds",
    )
    run.add_argument(
        "--max-concurrency",
        type=int,
        default=int(os.getenv("LLM_EVAL_MAX_CONCURRENCY", "1")),
        help="maximum simultaneous evaluation requests",
    )
    run.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/reports"),
        help="report output directory",
    )

    compare = subparsers.add_parser(
        "compare", help="compare a current JSON report with a baseline"
    )
    compare.add_argument(
        "--current", type=Path, required=True, help="current JSON report"
    )
    compare.add_argument(
        "--baseline", type=Path, required=True, help="baseline JSON report"
    )
    compare.add_argument(
        "--min-pass-rate", type=float, help="required current pass rate (0..1)"
    )
    compare.add_argument(
        "--max-pass-rate-drop",
        type=float,
        help="maximum baseline-to-current pass-rate decrease (0..1)",
    )
    compare.add_argument(
        "--max-error-rate", type=float, help="maximum current error rate (0..1)"
    )
    compare.add_argument(
        "--max-p95-latency-seconds",
        type=float,
        help="maximum current nearest-rank P95 latency",
    )
    compare.add_argument(
        "--max-p95-latency-increase-percent",
        type=float,
        help="maximum P95 increase from baseline as a percentage",
    )
    return parser


def _safe_message(exc: Exception) -> str:
    message = " ".join(str(exc).split())
    return message[:240] if message else "operation failed"


if __name__ == "__main__":
    raise SystemExit(main())
