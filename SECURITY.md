# Security Policy

## Supported versions

Security fixes are applied to the latest release line and the default branch.
Older snapshots and local modifications may not receive updates.

## Reporting a vulnerability

Please do not disclose a suspected vulnerability in a public issue, discussion,
pull request, log excerpt, evaluation artifact, or dataset.

Use GitHub's private vulnerability reporting or security-advisory flow for
this repository when available. If no private channel is visible, open a
minimal public issue asking a maintainer to establish private contact. Do not
include exploit details, credentials, private data, model content, internal
URLs, or reproduction artifacts in that issue.

A useful private report includes:

- the affected component and revision;
- the impact and realistic threat model;
- minimal reproduction steps;
- whether credentials, prompts, responses, or model files are involved;
- any suggested mitigation; and
- whether the issue is already public elsewhere.

Maintainers will acknowledge reports on a best-effort basis, validate the
impact, coordinate a fix and disclosure plan when appropriate, and credit the
reporter if requested.

## Security scope

The repository demonstrates production-oriented engineering but is not a
hosted service. Operators remain responsible for network exposure, model
provenance, host security, dependency review, data retention, and secret
management.

Important documented limitations include:

- the gateway has no authentication or rate limiting;
- Compose ports bind to loopback by default and should not be exposed directly
  to an untrusted network;
- model weights are supplied locally and are not verified or downloaded by the
  project;
- the experiment registry and RAG fingerprints detect changes but are not
  cryptographic signatures or trusted attestation;
- generated reports and RAG stores may contain sensitive operational metadata
  or bounded model output; and
- external llama.cpp and container images have their own security lifecycle.

See [docs/deployment.md](docs/deployment.md) for container and network
assumptions and [docs/reproducibility.md](docs/reproducibility.md) for artifact
privacy boundaries.
