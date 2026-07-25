#!/usr/bin/env python3
"""Deterministic gateway container smoke test with no inference backend."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class HttpResult:
    status: int
    content_type: str
    body: str


def fetch(url: str, *, timeout: float = 2.0) -> HttpResult:
    """Return success and HTTP error responses without exposing tracebacks."""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return HttpResult(
                status=response.status,
                content_type=response.headers.get("content-type", ""),
                body=response.read().decode("utf-8"),
            )
    except urllib.error.HTTPError as exc:
        return HttpResult(
            status=exc.code,
            content_type=exc.headers.get("content-type", ""),
            body=exc.read().decode("utf-8"),
        )


def assert_gateway_contract(base_url: str) -> None:
    """Assert the model-free public deployment contract."""

    health = fetch(f"{base_url}/health")
    if health.status != 200 or json.loads(health.body) != {"status": "healthy"}:
        raise RuntimeError(f"unexpected liveness response: HTTP {health.status}")

    metrics = fetch(f"{base_url}/metrics")
    if metrics.status != 200 or "text/plain" not in metrics.content_type:
        raise RuntimeError(f"unexpected metrics response: HTTP {metrics.status}")
    if "llm_platform_http_requests_total" not in metrics.body:
        raise RuntimeError("metrics response omitted platform metrics")

    readiness = fetch(f"{base_url}/ready")
    if readiness.status != 503:
        raise RuntimeError(f"unexpected readiness response: HTTP {readiness.status}")
    if json.loads(readiness.body) != {"status": "unavailable"}:
        raise RuntimeError("readiness response did not use the documented envelope")

    for response in (health, metrics, readiness):
        if "Traceback" in response.body:
            raise RuntimeError("a public response exposed a Python traceback")


def _available_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def _run(
    command: list[str],
    *,
    environment: dict[str, str],
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def run_smoke(*, build: bool) -> None:
    project = f"llm-platform-smoke-{os.getpid()}"
    configured_port = os.getenv("GATEWAY_PORT")
    port = int(configured_port) if configured_port is not None else _available_port()
    environment = os.environ.copy()
    environment.update(
        {
            "COMPOSE_PROJECT_NAME": project,
            "GATEWAY_PORT": str(port),
            "LLAMA_SERVER_BASE_URL": "http://127.0.0.1:9",
            "LLAMA_SERVER_TIMEOUT_SECONDS": "1",
        }
    )
    compose = ["docker", "compose"]
    container_id = ""
    try:
        if build:
            _run(compose + ["build", "gateway"], environment=environment)
        _run(
            compose
            + ["up", "--detach", "--no-deps", "--no-build", "--wait", "gateway"],
            environment=environment,
        )
        container_id = _run(
            compose + ["ps", "--quiet", "gateway"],
            environment=environment,
            capture_output=True,
        ).stdout.strip()
        if not container_id:
            raise RuntimeError("Compose did not return a gateway container ID")

        deadline = time.monotonic() + 30
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                assert_gateway_contract(f"http://127.0.0.1:{port}")
                break
            except (OSError, RuntimeError, ValueError) as exc:
                last_error = exc
                time.sleep(0.25)
        else:
            raise RuntimeError("gateway contract did not become ready") from last_error

        _run(compose + ["stop", "--timeout", "15", "gateway"], environment=environment)
        exit_code = _run(
            ["docker", "inspect", "--format", "{{.State.ExitCode}}", container_id],
            environment=environment,
            capture_output=True,
        ).stdout.strip()
        if exit_code != "0":
            raise RuntimeError(f"gateway exited with status {exit_code}, expected 0")
        print("container smoke test passed")
    finally:
        subprocess.run(
            compose + ["down", "--volumes", "--remove-orphans"],
            cwd=ROOT,
            env=environment,
            check=False,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse an image built by a preceding docker compose build",
    )
    args = parser.parse_args(argv)
    try:
        run_smoke(build=not args.skip_build)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"container smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
