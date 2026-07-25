# AGENTS.md

## Scope

These instructions apply to the entire repository.

`llm-production-platform` is a production-oriented educational FastAPI control
plane for a CPU-hosted llama.cpp server. Optimize for clarity, bounded resource
use, deterministic tests, and explicit operational behavior. Do not position
the project as a replacement for vLLM or SGLang.

## Source of truth

Before changing implementation or deployment files, read:

1. `README.md`
2. `docs/architecture.md`
3. `docs/development-plan.md`
4. `docs/metrics.md` when touching telemetry, dashboards, or benchmarks

If implementation needs to depart from the architecture, update the relevant
documentation and add an architectural decision record in the same change.
Do not silently weaken queue bounds, cancellation behavior, compatibility, or
metric cardinality rules.

## Phase discipline

- Follow `docs/development-plan.md` in order.
- Do not implement later-phase infrastructure speculatively.
- Keep every completed phase runnable and independently testable.
- A phase is not complete until its stated automated tests and exit criteria
  pass.
- Default tests must not download or require a large model.
- Real llama.cpp/model tests must be explicit, optional integration tests.

## Architecture boundaries

- `api` owns FastAPI, public OpenAI-compatible schemas, HTTP errors, and SSE
  presentation.
- `application` owns completion orchestration and the request lifecycle.
- `domain` owns backend-neutral models, policies, and port definitions.
- `backends` owns llama.cpp translation, HTTP transport, and upstream parsing.
- `scheduling` owns bounded queue and concurrency implementations.
- `observability` owns Prometheus and structured logging adapters.
- `config`/the composition root wires concrete dependencies.

Dependencies point toward the application/domain core. Do not import FastAPI,
Prometheus collectors, HTTP client response objects, or llama.cpp wire models
into core orchestration.

Prefer a small interface only where a boundary needs replacement, failure
injection, or contract testing. Avoid generic repositories/managers that do not
represent a real domain or infrastructure boundary.

## Required invariants

- Waiting work and active backend work are independently bounded.
- Queue full is rejected immediately and predictably.
- Client disconnect, timeout, and shutdown cancellation propagate upstream.
- Every admitted request releases its permit exactly once after upstream
  cleanup.
- Streaming is incremental and uses bounded buffering/backpressure.
- A failed or cancelled stream never emits the successful `[DONE]` marker.
- Public parameters are translated, intentionally handled, or rejected; never
  silently ignored.
- One inference request produces exactly one terminal outcome.
- Metric labels are bounded and never contain prompts, generated text, request
  IDs, client identities, raw paths, or arbitrary error text.
- Liveness is process-local; readiness represents ability to accept inference.
- In-process scheduler limits are described as per-process. Do not imply global
  enforcement across workers or replicas.

## Implementation expectations

- Target Python 3.12+ and use type annotations for public and internal
  interfaces.
- Use asynchronous, cancellation-safe I/O for request and backend paths.
- Reuse one lifecycle-managed backend HTTP client; do not create one per
  request.
- Validate configuration at startup and fail with actionable, secret-safe
  errors.
- Keep prompts, generated content, credentials, and raw backend bodies out of
  ordinary logs.
- Use stable request IDs for correlation.
- Normalize expected failures into typed errors and map them only at the API
  boundary.
- Avoid automatic retries of generation requests unless safety and duplicate
  work semantics are explicitly documented and tested.
- Pin or constrain dependencies and deployment images for reproducibility.

## Testing expectations

For each behavior, test at the lowest useful boundary and add an end-to-end
contract test for public behavior.

- Unit tests cover policies and lifecycle state transitions.
- API tests cover public schemas, errors, headers, and SSE frames.
- Backend contract tests use a controllable local fake HTTP server.
- Concurrency tests use barriers/events rather than timing-dependent sleeps.
- Cancellation, timeout, malformed upstream data, and shutdown are normal test
  cases, not edge cases to defer.
- Assert cleanup invariants: no leaked tasks, connections, queue entries, or
  permits; gauges return to baseline.
- Keep tests deterministic. If a real clock is unavoidable, use generous
  boundaries and document why.
- Run the repository's formatting, lint, type, unit, and relevant integration
  checks before declaring work complete.

## OpenAI compatibility

Treat compatibility as an explicit supported subset:

- preserve documented response and error envelopes;
- contract-test SSE ordering and termination;
- document unsupported endpoints and fields;
- do not expose llama.cpp-specific payloads through public schemas;
- do not claim full OpenAI compatibility until a published matrix supports it.

Any public contract change must update documentation and tests in the same
change.

## Documentation and operations

- Keep startup, configuration, health, failure, and shutdown behavior current.
- Record benchmark hardware, model checksum/quantization, llama.cpp version and
  flags, API settings, prompts/scenario version, and git revision.
- Never publish a performance comparison without its environment metadata.
- Keep dashboards and Prometheus configuration version-controlled and
  provisioned; avoid manual-only setup.
- Use relative paths and environment examples that work on Ubuntu 24.04/WSL2.
- Never commit model weights, secrets, local benchmark results containing
  sensitive prompts, `.env` files, or machine-specific absolute paths.

## Change hygiene

- Keep changes scoped to the current phase and preserve unrelated user work.
- Prefer small commits/changes that leave the system working.
- Update documentation with behavior, not aspirations presented as completed
  functionality.
- When a requirement is ambiguous, preserve boundedness, cancellation safety,
  compatibility, and testability, and document the chosen trade-off.

