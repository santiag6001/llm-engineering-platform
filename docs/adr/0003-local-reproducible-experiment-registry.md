# ADR 0003: Use a local reproducible experiment registry

- Status: accepted
- Date: 2026-07-25

## Context

Phase 4 produces deterministic evaluation reports and regression decisions, but
individual files do not bind code revision, dirty state, dataset and prompt
identity, requested configuration, environment, regression policy, and
artifact integrity into one traceable unit. Phase 6 requires that binding while
remaining offline, deterministic, and educational.

External platforms such as MLflow, Langfuse, Weights & Biases, hosted databases,
and cloud artifact stores would add accounts, network availability, credentials,
retention policy, dependency weight, and vendor-specific behavior before the
project has established its local experiment contract. A database would also
obscure the atomic filesystem and portability semantics this phase intends to
teach.

## Decision

Phase 6 uses a standalone `llm_platform.experiments` package and a local
filesystem registry. Immutable run directories are finalized by same-filesystem
atomic rename, artifacts are addressed by portable relative paths and
SHA-256/size metadata, and strict versioned manifests bind all reproducibility
metadata. Mutable aliases use atomic file replacement and have no history.

The experiment runner reuses the public-HTTP evaluation runner, report models,
Markdown renderer, and regression gates. The FastAPI serving application does
not import the experiment package. Canonical JSON plus SHA-256 defines
experiment and environment fingerprints. Generated content and complete
prompts remain outside manifests.

Phase 6 is inserted before bounded runtime queueing. The former Phase 6 and all
later roadmap phases move forward by one number.

## Consequences

- Experiments remain usable and testable without a model, network, database,
  hosted service, or secret.
- Run directories can be copied and inspected with ordinary filesystem and JSON
  tools.
- Atomic local writes and checksums provide integrity detection but not signing,
  trusted provenance, distributed transactions, or multi-host coordination.
- Alias updates are mutable pointers without an audit history; manifests record
  resolved immutable baseline run IDs.
- Large-scale search, remote sharing, access control, retention, and UI concerns
  remain possible future adapters after the local contract is stable.
- Runtime queueing, RAG, agents, fine-tuning, authentication, Kubernetes, model
  downloading, and external experiment platforms remain out of scope.
