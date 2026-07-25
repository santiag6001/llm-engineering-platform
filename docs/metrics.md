# Metrics

## 1. Scope and architecture

This document is the source of truth for the Phase 3 application metrics
contract. Metrics describe public HTTP traffic and validated chat completion
lifecycles for buffered and streaming requests. Queueing, concurrency limits,
Grafana, deployment, alerts, and backend-native llama.cpp metrics are not part
of this phase.

Each FastAPI application owns a private prometheus-client
`CollectorRegistry`. The composition root supplies a backend-neutral metrics
port to the completion service, and the concrete adapter under
`observability` owns collectors and text exposition. Domain models,
orchestration, and backend adapters do not import Prometheus.

`GET /metrics` renders the registry using the official Prometheus text content
type. Scrapes do not increment the platform HTTP counter, preventing
self-referential traffic.

## 2. Cardinality policy

Every label name and value comes from a fixed allowlist.

Allowed labels are:

- `endpoint`: `chat_completions`, `models`, `health`, or `unmatched`;
- `method`: `GET`, `POST`, or `OTHER`;
- `status_class`: `2xx`, `4xx`, `5xx`, or `other`;
- `mode`: `buffered` or `streaming`;
- `outcome`: one terminal value from section 3;
- `error_type`: one upstream value from section 4.

Unsupported HTTP methods map to `OTHER`, unexpected response classes map to
`other`, and raw paths never become label values.

Forbidden labels and label values include:

- request IDs, trace IDs, client IPs, client identities, and API keys;
- prompts, messages, generated content, or token text;
- exception messages, arbitrary exception class names, and raw backend bodies;
- raw URLs or paths, query strings, authorization headers, and credentials;
- public model values, backend URLs, and model file paths.

Request IDs remain log correlation fields and are intentionally absent from
metrics.

## 3. Terminal outcome taxonomy

Every request that passes public schema validation and enters the completion
service increments `llm_platform_chat_requests_total` exactly once with one of:

| Outcome | Meaning |
|---|---|
| `success` | Buffered result returned or streaming response completed normally |
| `backend_timeout` | Upstream connect, response, or stream read timed out |
| `backend_unavailable` | The upstream server could not be reached |
| `backend_error` | Upstream HTTP, disconnect, or malformed-response failure |
| `internal_error` | Unexpected platform failure during completion |
| `client_cancelled` | Downstream cancellation propagated through upstream cleanup |

Public validation failures never enter the completion service and therefore do
not increment the chat counter. They still increment
`llm_platform_http_requests_total` with the normalized chat endpoint and `4xx`
status class. A later lifecycle phase may distinguish client cancellation from
shutdown cancellation; Phase 3 has no shutdown cancellation state.

Structured streaming terminal logs use the same outcome values. Each log keeps
the request ID for correlation, while the corresponding metric does not.

## 4. Upstream error taxonomy

`llm_platform_upstream_errors_total` uses only these `error_type` values:

| Error type | Meaning |
|---|---|
| `timeout` | Normalized upstream timeout |
| `unavailable` | Connection could not be established |
| `disconnect` | HTTP protocol failure or upstream disconnect |
| `malformed_response` | Invalid buffered response or generic protocol data |
| `malformed_stream` | Invalid, oversized, or truncated SSE stream |
| `http_4xx` | Upstream returned a 4xx response |
| `http_5xx` | Upstream returned a 5xx response |
| `cancelled` | Client cancellation propagated to active upstream work |
| `unknown` | Unexpected failure without a safer bounded classification |

Exception text, upstream error messages, upstream status codes, and arbitrary
exception class names are never label values. Public error envelopes retain
their existing Phase 1/2 behavior independently of this internal taxonomy.

## 5. Metric contract

### Counters

| Metric | Labels | Lifecycle meaning |
|---|---|---|
| `llm_platform_http_requests_total` | `endpoint`, `method`, `status_class` | Increments once when the API has produced an HTTP response status; `/metrics` is excluded |
| `llm_platform_chat_requests_total` | `mode`, `outcome` | Increments exactly once when a validated completion reaches terminal cleanup |
| `llm_platform_generated_tokens_total` | `mode` | Adds trusted backend-reported generated tokens once per terminal request |
| `llm_platform_upstream_errors_total` | `mode`, `error_type` | Increments once for a terminal upstream failure or upstream cancellation |
| `llm_platform_client_disconnects_total` | `mode` | Increments once for a `client_cancelled` completion |

For streaming HTTP responses, the HTTP status is fixed when response headers
are produced. A later SSE error therefore remains an HTTP `2xx`; the terminal
failure is represented by `llm_platform_chat_requests_total` and
`llm_platform_upstream_errors_total`.

### Histograms

| Metric | Labels | Lifecycle meaning |
|---|---|---|
| `llm_platform_request_duration_seconds` | `mode`, `outcome` | Completion service entry through terminal result or stream cleanup |
| `llm_platform_time_to_first_token_seconds` | none | Streaming backend start through receipt of the first valid content-bearing chunk |
| `llm_platform_upstream_duration_seconds` | `mode`, `outcome` | Backend invocation through result/error or terminal upstream stream iteration |

Request and upstream durations begin together in Phase 3 because no admission
queue exists. They have separate stop boundaries for streaming: upstream
duration stops when upstream iteration terminates, while request duration
continues until the downstream stream context is closed. Buffered durations
normally differ only by local cleanup overhead.

The request/upstream finite histogram buckets are:

```text
0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10,
30, 60, 120, 300 seconds
```

The TTFT finite buckets are:

```text
0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120 seconds
```

Every histogram also has the prometheus-client `+Inf` bucket.

### Gauges

| Metric | Labels | Lifecycle meaning |
|---|---|---|
| `llm_platform_active_requests` | `mode` | Validated completion lifecycles that have started but not completed terminal cleanup |
| `llm_platform_active_streams` | none | Streaming completion lifecycles that have started but not completed terminal cleanup |

The service increments gauges immediately before invoking the backend. It
decrements them in terminal cleanup after the upstream context has closed.
Success, timeout, unavailable backend, malformed response/SSE, unexpected
failure, and cancellation all use the same balanced cleanup path. Gauges are
process-local and must return to zero when no completion is active.

## 6. TTFT definition

TTFT is observed only for streaming requests and at most once per request.
Its start is immediately before opening the backend stream. Its stop is receipt
of the first validated chunk whose assistant delta contains a non-empty
`content` string.

Role-only chunks, empty content strings, usage-only chunks, finish chunks, SSE
comments, and `[DONE]` do not stop the TTFT timer. A stream that fails,
disconnects, is cancelled, or completes without a content-bearing chunk
produces no TTFT observation.

This is backend responsiveness, not end-user time from initial HTTP receipt.
There is no queue in Phase 3, but public validation and downstream network time
are still outside the TTFT boundary.

## 7. Generated token semantics

The platform does not estimate tokens. It accepts a token observation only
when the validated buffered response or streaming chunk contains a
non-negative integer `usage.completion_tokens`. Boolean, negative, floating,
string, missing, and otherwise malformed values are ignored.

For a stream, usage is treated as cumulative and the latest valid value is
recorded once when the request terminates. Usage-only final chunks are valid
even when `choices` is empty. Failed requests may report tokens if a trusted
usage chunk arrived before failure; the counter represents reported generated
work rather than successful delivery.

## 8. Terminal and cancellation rules

One completion lifecycle produces exactly one terminal chat outcome:

1. Active gauges increment before the backend operation.
2. TTFT may be observed once during a stream.
3. The latest trusted generated-token usage is retained.
4. A normalized outcome and optional upstream error type are selected.
5. Terminal counters and duration histograms update once.
6. Active gauges decrement in the same terminal cleanup call.

Client cancellation propagates into the awaited stream iterator and backend
HTTP context. The upstream response is closed before the terminal observation
releases gauges. Cancellation increments the chat counter with
`client_cancelled`, the client-disconnect counter, and the upstream-error
counter with `cancelled`. A cancelled or failed stream never emits `[DONE]`.

Metrics adapter exceptions are caught and logged by the completion service so
instrumentation cannot replace the public inference result. The concrete
adapter uses constant label sets and a private registry, making ordinary
collector operations deterministic.

## 9. Current limitations

- Registries and gauges are per process. They do not provide global counts
  across Uvicorn workers or replicas.
- Generic HTTP duration is not exposed; request duration currently describes
  validated chat completion lifecycles.
- Process/Python runtime collectors are intentionally absent from the private
  registry in this phase.
- Queue depth, queue wait, concurrency limits, readiness, build info, input
  tokens, missing-usage counters, and backend-native metrics are not yet
  exposed.
- Prometheus deployment, Grafana dashboards, recording rules, and alerts are
  later-phase work.

## 10. Correctness tests

Tests create a new application and registry per scenario and compare collector
samples without relying on process-global state. The required invariants are:

- buffered and streaming success each record one terminal outcome;
- TTFT is observed once for valid content and never before content;
- generated usage is added only when valid;
- error labels stay within the allowlist and never include exception text;
- active request and stream gauges return to zero after success, timeout,
  malformed SSE, and client cancellation;
- health/readiness traffic never increments the chat counter;
- Phase 1 buffered envelopes and Phase 2 SSE ordering and `[DONE]` behavior are
  unchanged.
