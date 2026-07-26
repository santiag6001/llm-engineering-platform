from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import container_smoke

ROOT = Path(__file__).resolve().parents[1]


def test_gateway_image_contract_is_non_root_and_exec_form() -> None:
    dockerfile = (ROOT / "deploy/docker/Dockerfile").read_text(encoding="utf-8")

    assert "python:3.12.11-slim-bookworm" in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS builder" in dockerfile
    assert "FROM ${PYTHON_IMAGE} AS runtime" in dockerfile
    assert "USER 10001:10001" in dockerfile
    assert 'CMD ["uvicorn",' in dockerfile
    assert "--reload" not in dockerfile
    assert "COPY tests" not in dockerfile


def test_dockerignore_excludes_sensitive_and_large_local_artifacts() -> None:
    ignored = set((ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines())

    assert {".git", ".venv", "tests", "models", "*.gguf", ".env"} <= ignored


def test_compose_preserves_gateway_only_workflow_and_safe_mounts() -> None:
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")
    llama = (ROOT / "compose.llama.yaml").read_text(encoding="utf-8")

    assert "LLAMA_SERVER_BASE_URL" in compose
    assert "http://host.docker.internal:8080" in compose
    assert "condition: service_healthy" in compose
    assert "profiles:" in compose
    assert "no-new-privileges:true" in compose
    assert "docker.sock" not in compose
    assert "network_mode: host" not in compose

    assert "LLAMA_MODEL_PATH:?" in llama
    assert "read_only: true" in llama
    assert "nvidia" not in llama.lower()
    assert "privileged:" not in llama


def test_prometheus_scrapes_process_local_gateway_metrics() -> None:
    configuration = (ROOT / "deploy/prometheus/prometheus.yml").read_text(
        encoding="utf-8"
    )

    assert "gateway:8000" in configuration
    assert "metrics_path: /metrics" in configuration
    assert "remote_write" not in configuration


def test_model_free_smoke_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "/health": container_smoke.HttpResult(
            200,
            "application/json",
            json.dumps({"status": "healthy"}),
        ),
        "/metrics": container_smoke.HttpResult(
            200,
            "text/plain; version=0.0.4; charset=utf-8",
            "# TYPE llm_platform_http_requests_total counter\n",
        ),
        "/ready": container_smoke.HttpResult(
            503,
            "application/json",
            json.dumps({"status": "unavailable"}),
        ),
    }

    def fake_fetch(url: str, *, timeout: float = 2.0) -> container_smoke.HttpResult:
        del timeout
        path = "/" + url.rsplit("/", maxsplit=1)[-1]
        return responses[path]

    monkeypatch.setattr(container_smoke, "fetch", fake_fetch)
    container_smoke.assert_gateway_contract("http://gateway.test")


def test_smoke_contract_rejects_tracebacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = {
        "/health": container_smoke.HttpResult(
            200,
            "application/json",
            json.dumps({"status": "healthy"}),
        ),
        "/metrics": container_smoke.HttpResult(
            200,
            "text/plain",
            "llm_platform_http_requests_total 1\nTraceback",
        ),
        "/ready": container_smoke.HttpResult(
            503,
            "application/json",
            json.dumps({"status": "unavailable"}),
        ),
    }

    def traceback_response(
        url: str, *, timeout: float = 2.0
    ) -> container_smoke.HttpResult:
        del timeout
        path = "/" + url.rsplit("/", maxsplit=1)[-1]
        return responses[path]

    monkeypatch.setattr(container_smoke, "fetch", traceback_response)
    with pytest.raises(RuntimeError, match="traceback"):
        container_smoke.assert_gateway_contract("http://gateway.test")


@pytest.mark.parametrize("exit_code", ["0", "143"])
def test_intentional_smoke_shutdown_accepts_expected_exit_codes(
    exit_code: str,
) -> None:
    container_smoke.assert_intentional_shutdown_exit_code(exit_code)


@pytest.mark.parametrize("exit_code", ["137", "1"])
def test_intentional_smoke_shutdown_rejects_unexpected_exit_codes(
    exit_code: str,
) -> None:
    with pytest.raises(RuntimeError, match=rf"status {exit_code}"):
        container_smoke.assert_intentional_shutdown_exit_code(exit_code)
