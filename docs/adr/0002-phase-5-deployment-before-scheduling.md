# ADR 0002: Deliver deployment and CI as Phase 5

- Status: accepted
- Date: 2026-07-25

## Context

After Phase 4 evaluation, the roadmap assigned Phase 5 to bounded runtime
queueing and assigned container deployment to a later phase. The project now
requires reproducible packaging, a model-free container acceptance path, and
automated quality validation before changing request admission or lifecycle
semantics.

Implementing deployment first can package the completed public contracts
without introducing scheduler behavior. The existing liveness endpoint also
supports a gateway-only container workflow while readiness continues to
represent live backend usability.

## Decision

Phase 5 is reproducible containerized deployment and CI/CD. It provides a
non-root gateway image, Compose gateway/Prometheus services, an optional
user-mounted llama-server profile, deterministic model-free smoke validation,
and GitHub Actions quality/image checks.

The previous Phase 5 and later roadmap phases move forward by one number.
Grafana remains a later operational extension and is not added speculatively.
No runtime application contract changes in this decision.

## Consequences

- Phase 1–4 API, metrics, cancellation, and evaluation behavior remains
  unchanged.
- Default CI requires no model, GPU, secrets, hosted LLM, or live backend.
- Liveness drives container health; readiness remains backend-dependent.
- Prometheus deployment is available while metrics remain process-local.
- Runtime queueing, concurrency limiting, resilience lifecycle changes,
  Grafana, benchmarks, Kubernetes, RAG, agents, and fine-tuning remain future
  work.
