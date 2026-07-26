# ADR 0004: Deliver production RAG engineering as Phase 7

- Status: accepted
- Date: 2026-07-26

## Context

The current roadmap assigned Phase 7 to bounded runtime scheduling. The project
now requires a deterministic local retrieval platform before changing serving
admission or lifecycle semantics. Folding retrieval into FastAPI would couple
document/index dependencies to the inference runtime and turn an engineering
and evaluation concern into a new public serving contract.

External vector databases, hosted embedding APIs, hosted experiment platforms,
and agent runtimes would also hide the local identities, ordering, persistence,
and evaluation semantics this phase is intended to establish.

## Decision

Phase 7 is a standalone `llm_platform.rag` package and `llm-rag` CLI. It owns a
content-addressed document registry, deterministic chunking, a versioned local
CPU hashing embedder, a persistent JSON vector index, top-K/threshold/MMR
retrieval, context assembly, citations, and retrieval evaluation.

The RAG package is not imported by FastAPI. Phase 6 experiment manifests gain
an optional strict RAG provenance block, supplied from a local retrieval
evaluation artifact. Non-RAG experiment identity remains unchanged.

The former Phase 7 and later roadmap phases move forward by one number. No
runtime scheduler or lifecycle behavior is implemented by this decision.

## Consequences

- Document, chunk, embedding, index, context, and retrieval identities can be
  reproduced and inspected using local files.
- Default validation needs no backend, model download, GPU, Docker daemon,
  hosted API, database, or secret.
- The hashing embedder is a deterministic lexical baseline, not a claim of
  state-of-the-art semantic retrieval.
- The JSON index is intentionally educational and bounded; it is not an
  external vector database or distributed service.
- Queueing, lifecycle resilience, Kubernetes, authentication, external
  experiment platforms, agent runtime, and hosted APIs remain future work.
