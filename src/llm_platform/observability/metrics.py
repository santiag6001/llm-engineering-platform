"""Isolated Prometheus metrics for HTTP and completion lifecycles."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
from prometheus_client.exposition import CONTENT_TYPE_LATEST, generate_latest

from llm_platform.domain.models import (
    CompletionMode,
    CompletionOutcome,
    UpstreamErrorType,
)

_REQUEST_DURATION_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)
_TTFT_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)


class PrometheusMetrics:
    """Own all collectors for one application instance."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry if registry is not None else CollectorRegistry()
        self.http_requests = Counter(
            "llm_platform_http_requests_total",
            "Completed HTTP responses by normalized endpoint.",
            ("endpoint", "method", "status_class"),
            registry=self.registry,
        )
        self.chat_requests = Counter(
            "llm_platform_chat_requests_total",
            "Terminal validated chat completion requests.",
            ("mode", "outcome"),
            registry=self.registry,
        )
        self.generated_tokens = Counter(
            "llm_platform_generated_tokens_total",
            "Backend-reported generated tokens.",
            ("mode",),
            registry=self.registry,
        )
        self.upstream_errors = Counter(
            "llm_platform_upstream_errors_total",
            "Terminal upstream failures by bounded classification.",
            ("mode", "error_type"),
            registry=self.registry,
        )
        self.client_disconnects = Counter(
            "llm_platform_client_disconnects_total",
            "Chat completions cancelled by a disconnected client.",
            ("mode",),
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "llm_platform_request_duration_seconds",
            "Validated chat request start to terminal cleanup.",
            ("mode", "outcome"),
            buckets=_REQUEST_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.time_to_first_token = Histogram(
            "llm_platform_time_to_first_token_seconds",
            "Streaming backend start to first valid content chunk.",
            buckets=_TTFT_BUCKETS,
            registry=self.registry,
        )
        self.upstream_duration = Histogram(
            "llm_platform_upstream_duration_seconds",
            "Backend call start to terminal upstream cleanup.",
            ("mode", "outcome"),
            buckets=_REQUEST_DURATION_BUCKETS,
            registry=self.registry,
        )
        self.active_requests = Gauge(
            "llm_platform_active_requests",
            "Validated chat completion lifecycles currently active.",
            ("mode",),
            registry=self.registry,
        )
        self.active_streams = Gauge(
            "llm_platform_active_streams",
            "Streaming chat completion lifecycles currently active.",
            registry=self.registry,
        )

    def request_started(self, mode: CompletionMode) -> None:
        self.active_requests.labels(mode=mode).inc()
        if mode == "streaming":
            self.active_streams.inc()

    def time_to_first_token_observed(self, duration_seconds: float) -> None:
        self.time_to_first_token.observe(duration_seconds)

    def request_finished(
        self,
        *,
        mode: CompletionMode,
        outcome: CompletionOutcome,
        request_duration_seconds: float,
        upstream_duration_seconds: float,
        generated_tokens: int | None,
        error_type: UpstreamErrorType | None,
    ) -> None:
        try:
            self.chat_requests.labels(mode=mode, outcome=outcome).inc()
            self.request_duration.labels(mode=mode, outcome=outcome).observe(
                request_duration_seconds
            )
            self.upstream_duration.labels(mode=mode, outcome=outcome).observe(
                upstream_duration_seconds
            )
            if generated_tokens is not None:
                self.generated_tokens.labels(mode=mode).inc(generated_tokens)
            if error_type is not None:
                self.upstream_errors.labels(mode=mode, error_type=error_type).inc()
            if outcome == "client_cancelled":
                self.client_disconnects.labels(mode=mode).inc()
        finally:
            self.active_requests.labels(mode=mode).dec()
            if mode == "streaming":
                self.active_streams.dec()

    def observe_http_request(
        self,
        *,
        endpoint: str,
        method: str,
        status_class: str,
    ) -> None:
        self.http_requests.labels(
            endpoint=endpoint,
            method=method,
            status_class=status_class,
        ).inc()

    def render(self) -> bytes:
        """Render this application's registry in Prometheus text format."""

        return generate_latest(self.registry)
