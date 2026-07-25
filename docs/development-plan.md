# Development plan

## 1. Delivery rules

Development proceeds strictly in order. Every phase is a vertical increment
that can be started, tested, observed, and demonstrated on its own.

A phase is complete only when:

- its acceptance behavior works from the public interface;
- automated tests for new behavior pass;
- earlier phase tests still pass;
- configuration and failure behavior are documented;
- the process starts and shuts down cleanly;
- no large model download is required by the default test suite.

Scope discovered for a later phase is recorded rather than partially
implemented early. Refactoring is allowed when it preserves the completed
public contract.

## 2. Phase 0 — Project foundation

**Working system:** an installable Python package with a FastAPI process that
starts, reports health, and shuts down cleanly. It performs no inference.

### Deliverables

- Python 3.12+ packaging, dependency groups, linting, formatting, and type
  checking configuration.
- Initial module boundaries and an application composition root.
- Typed configuration with startup validation.
- `GET /health/live`, `GET /health/ready`, and generated OpenAPI documentation.
- Structured request logging with request IDs, without model prompts/content.
- Unit/API test harness and CI command definitions.

### Tests

- Package installation in a clean environment.
- Liveness success and readiness semantics without a configured backend.
- Invalid configuration fails fast with an actionable, secret-safe message.
- Request ID is accepted/generated and returned.
- Graceful startup and shutdown smoke test.

### Exit criteria

A contributor can run one documented command to start the API and one to run
all checks. Health behavior is covered by automated tests.

## 3. Phase 1 — Non-streaming completion through a backend port

**Working system:** `POST /v1/chat/completions` with `stream=false` returns a
supported OpenAI-shaped response through a fake backend, and can be configured
to use a real llama.cpp server.

### Deliverables

- Versioned public request, response, and error schemas.
- Internal completion command/result models.
- Completion service and `InferenceBackend` port.
- llama.cpp HTTP adapter with a shared asynchronous client.
- Fake backend for deterministic tests.
- `GET /v1/models` for the configured public model.
- Explicit table of supported, ignored (preferably none), and rejected fields.
- Connect and backend response timeouts.

### Tests

- Schema examples and invalid/unsupported parameters.
- API-to-service mapping and backend-to-API result mapping.
- OpenAI error-envelope contract.
- Backend success, non-2xx response, timeout, malformed JSON, and unavailable
  host using a local controllable fake server.
- Optional real llama.cpp smoke test behind an explicit marker.

### Exit criteria

A non-streaming completion works end to end against fake and real backends.
Default tests remain model-free and deterministic.

## 4. Phase 2 — Streaming completion

**Working system:** `stream=true` relays incremental output from the backend to
the client using OpenAI-compatible SSE and terminates correctly.

### Deliverables

- Backend-neutral stream chunk model.
- Incremental llama.cpp SSE parser and mapper.
- OpenAI SSE presenter, including a successful `[DONE]` terminator.
- Bounded streaming handoff/backpressure strategy.
- Client-disconnect and upstream-cancellation propagation.
- Defined error behavior before and after response headers are sent.
- Idle-stream timeout distinct from connection timeout.

### Tests

- Exact content type and SSE frame sequence.
- Chunks arrive before backend completion (no accidental buffering).
- Unicode and events split across transport reads.
- Malformed event, upstream error, idle timeout, and truncated stream.
- Slow consumer stays within the configured memory bound.
- Client disconnect closes upstream work and does not emit `[DONE]`.

### Exit criteria

A command-line client visibly receives tokens incrementally. Buffered behavior
from Phase 1 is unchanged. All stream termination paths have contract tests.

## 5. Phase 3 — Bounded queue and concurrency control

**Working system:** requests above the active limit wait in a bounded FIFO
queue; overload is rejected predictably; cancellation frees capacity.

### Deliverables

- `RequestScheduler` port and in-memory FIFO implementation.
- Configurable maximum active requests and maximum waiting requests.
- Separate queue-wait timeout.
- Atomic enqueue/reject and permit handoff behavior.
- Cancellation of queued and active requests.
- Documented single-process/single-worker constraint.
- Deterministic scheduler test hooks (fake backend/barriers and fake clock where
  appropriate).

### Tests

- Active backend calls never exceed the configured limit.
- FIFO order under controlled concurrency.
- Full queue produces the stable OpenAI-shaped 429 error envelope.
- Queue timeout never invokes the backend.
- Queued client cancellation removes/skips the job.
- Active cancellation releases the permit after upstream cleanup.
- Failure and cancellation races do not leak or double-release permits.

### Exit criteria

A load scenario demonstrates a bounded active count, a bounded queue, and
predictable rejection. After the scenario, active and queued counts return to
zero.

## 6. Phase 4 — Resilience and process lifecycle

**Working system:** the service behaves predictably during backend failure,
startup, and shutdown while preserving established API and scheduling behavior.

### Deliverables

- Typed failure taxonomy and final public status mapping.
- Backend-aware readiness with a bounded probe/cache policy.
- Retry policy decision: no automatic generation retry by default unless a
  narrowly safe case is proven and documented.
- Graceful shutdown: stop admission, handle queued work, drain active requests
  for a configured grace period, then cancel.
- Explicit request, queue, backend, idle-stream, and shutdown timeout semantics.
- Safe limits for public input size and requested generation.

### Tests

- Backend readiness transitions without affecting liveness.
- Backend restart/recovery.
- Shutdown with an empty system, queued requests, active buffered requests, and
  active streams.
- New requests are rejected after draining begins.
- Grace-period expiry cancels work and leaves no permits/tasks.
- Error responses and logs do not expose raw backend bodies or secrets.

### Exit criteria

Automated integration tests can kill or stall the fake backend and initiate
shutdown without hanging the API or leaking tasks.

## 7. Phase 5 — Application observability

**Working system:** the API exposes useful, bounded-cardinality Prometheus
metrics and correlated structured logs for every terminal request outcome.

### Deliverables

- Lifecycle event vocabulary and telemetry port.
- Prometheus counters, gauges, and histograms from
  [the metrics contract](metrics.md).
- `/metrics` endpoint and registry isolation for tests.
- Queue delay, time to first token, backend duration, stream duration, request
  outcome, active/queued work, and token/usage metrics when reliable.
- Build/runtime information with safe labels.
- Log correlation using request ID and a shared outcome taxonomy.

### Tests

- Metric exposition names, types, and required labels.
- Exactly one terminal request increment per request.
- Gauge balance across success, failure, timeout, and cancellation.
- Histogram observations use the correct phase boundaries.
- No request ID, prompt, raw path, user input, or unbounded model/backend value
  appears as a metric label.
- Metrics failures do not fail an inference request.

### Exit criteria

A deterministic scenario produces expected metric deltas, and a failed request
can be correlated in logs using its request ID.

## 8. Phase 6 — Docker Compose, Prometheus, and Grafana

**Working system:** one documented Docker Compose command starts the API,
llama.cpp, Prometheus, and Grafana with provisioned health checks and dashboard.

### Deliverables

- Reproducible, pinned API container build running as a non-root user.
- llama.cpp server configuration suitable for CPU inference.
- Compose services, private networking, volumes, health checks, and shutdown
  grace periods.
- Read-only host model mount with an explicit configuration variable.
- Prometheus scrape configuration.
- Provisioned Grafana data source and version-controlled dashboard.
- Dashboard panels and initial alert expressions described in
  [Metrics](metrics.md).
- A Compose environment example containing no secrets or machine-specific
  absolute paths.

### Tests

- Container image build and configuration validation.
- Compose configuration rendering.
- Smoke test: services become healthy, models endpoint works, one buffered and
  one streamed completion work, Prometheus target is up, and Grafana provisions
  the dashboard.
- Restart behavior preserves required dashboard data/configuration.
- SIGTERM honors the documented drain behavior.

### Exit criteria

A fresh Ubuntu 24.04/WSL2 environment with Docker and a supplied compatible GGUF
model can follow the README to reach a healthy, observable deployment.

## 9. Phase 7 — Reproducible benchmarks

**Working system:** a benchmark runner sends defined workloads, validates
responses, and writes results with enough metadata to repeat and compare runs.

### Deliverables

- Versioned benchmark scenarios for buffered, streaming, saturation, and
  overload behavior.
- Warm-up and measured-run separation.
- Fixed prompts or prompt-generation seed with content/version checksums.
- Result schema in a machine-readable format plus human summary.
- Captured metadata: timestamp, git revision, OS/WSL version, CPU, memory, model
  filename and checksum, quantization, context size, llama.cpp version/flags,
  API settings, concurrency, queue bounds, and scenario version.
- Measurements: request success/error counts, end-to-end latency, queue delay,
  time to first token, inter-token/stream duration where meaningful, output
  tokens, throughput, and CPU/memory observations.
- Documented limitations and comparison rules.

### Tests

- Scenario/config schema validation.
- Deterministic dry run against the fake backend.
- Output metadata completeness.
- Failed or invalid responses are counted, never silently discarded.
- Summary statistics match a small known fixture.

### Exit criteria

Two runs with the same inputs are structurally comparable, and the report makes
hardware/model/config differences obvious. No baseline performance claim is
published without its metadata.

## 10. Phase 8 — Release hardening

**Working system:** the complete educational platform can be cloned, validated,
operated, and studied from its documentation.

### Deliverables

- End-to-end compatibility matrix and known limitations.
- Architecture decision records for the major choices.
- Operator runbook for startup, shutdown, overload, backend failure, and metric
  interpretation.
- Security review of binds, CORS, input limits, container user, dependencies,
  logs, model mounts, and unauthenticated deployment warning.
- Dependency pin/update policy and software bill of materials decision.
- Full CI pipeline for unit, contract, integration, container, and documentation
  checks, with expensive real-model tests opt-in.
- Tagged release checklist.

### Tests

- Documentation commands are exercised in CI where practical.
- Clean-environment installation and Compose smoke test.
- Load soak with no material task, connection, permit, or memory leak.
- Shutdown/restart and backend-recovery scenarios.
- Final API, metrics, and dashboard snapshot/contract checks.

### Exit criteria

All documented acceptance checks pass, known limitations are explicit, and a
new contributor can reproduce both the deployment and a benchmark.

## 11. Deferred roadmap

The following begin only after Phase 8, each as another independently working
vertical increment:

1. API-key authentication and principal context.
2. Per-principal rate limiting ahead of the admission queue.
3. Model registry, multiple backend targets, and per-model scheduling.
4. Kubernetes manifests/Helm packaging for the single-replica topology.
5. Distributed admission, routing, and failure handling.

Each feature must first define its public behavior, new failure modes, metrics,
and migration path. In particular, multiple API replicas must not be advertised
as enforcing a global concurrency limit until coordination is implemented.
