# ADR 0001: Insert evaluation and regression as Phase 4

- Status: accepted
- Date: 2026-07-25

## Context

The original development plan assigned Phase 4 to runtime queueing and
concurrency control. The project now requires a reproducible LLM quality and
regression pipeline before later runtime/deployment work. Implementing that
pipeline inside the serving application would couple experiment tooling to
request orchestration and could disturb the completed Phase 1–3 contracts.

## Decision

Phase 4 is the standalone evaluation and regression pipeline. The previous
Phase 4 and all later planned phases move forward by one number.

Evaluation communicates exclusively through the public buffered
`POST /v1/chat/completions` contract. Its package, CLI, datasets, reports, and
tests remain outside the FastAPI composition graph. It uses fixed async workers
to bound evaluation concurrency and does not implement the future serving
runtime admission queue.

## Consequences

- Phase 1–3 runtime behavior and architecture remain unchanged.
- Quality/regression checks are reproducible and testable offline using mock
  HTTP transports.
- Runtime bounded queueing, lifecycle resilience, deployment, benchmarks, RAG,
  agents, and Kubernetes remain future work.
- Documentation and issue references must use the updated phase numbering.
