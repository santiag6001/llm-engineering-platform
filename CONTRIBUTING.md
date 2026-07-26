# Contributing

Thank you for helping improve LLM Production Platform. Contributions should
keep the project clear, bounded, deterministic, and useful as an educational
production-engineering reference.

## Before opening a change

1. Read [AGENTS.md](AGENTS.md) for repository-wide engineering rules.
2. Read [docs/architecture.md](docs/architecture.md) and the relevant focused
   document under `docs/`.
3. Check [docs/development-plan.md](docs/development-plan.md) before proposing
   new behavior. The v1.0 feature set is complete through milestone 7.
4. Open an issue for a material API, architecture, persistence, or deployment
   proposal before investing in implementation.
5. Use [SECURITY.md](SECURITY.md) for vulnerabilities instead of a public
   issue.

Documentation fixes, test improvements, small bug fixes, and focused
readability improvements are welcome. New infrastructure should not be added
speculatively.

## Development setup

Python 3.12 or newer is required:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

The default development and test workflow requires no model, GPU, Docker
daemon, hosted API, database, or secret.

## Quality checks

Run the same Python checks used by CI:

```bash
ruff format --check .
ruff check .
mypy
pytest
python -m pip check
git diff --check
```

When changing deployment documentation or artifacts, also validate the
relevant Compose and model-free smoke contracts described in
[docs/deployment.md](docs/deployment.md). Real llama.cpp/model testing must
remain explicit and optional.

## Design expectations

- Preserve the established FastAPI, OpenAI-shaped error, and SSE contracts.
- Keep framework and transport objects outside the application/domain core.
- Keep prompts, generated content, credentials, and raw backend bodies out of
  ordinary logs.
- Preserve streaming backpressure and cancellation cleanup.
- Keep metric labels bounded and free of user-controlled values.
- Keep evaluation, experiment, and RAG persistence strict and reproducible.
- Do not add automatic generation retries without documented duplicate-work
  semantics and tests.
- Do not make performance claims without complete environment metadata.

If a change intentionally alters an architectural decision, update the
relevant documentation and include an ADR in the same pull request.

## Tests

Add tests at the lowest useful boundary:

- API tests for public schemas, headers, envelopes, and SSE frames;
- backend contract tests with controllable local transports;
- deterministic unit tests for policies, identity, and persistence;
- end-to-end contract tests for externally visible behavior; and
- cleanup assertions for cancellation and failure paths.

Avoid timing-dependent sleeps. Use events, barriers, mock transports, and
injected clocks where appropriate. The default suite must never download a
model.

## Documentation

Commands in public documentation should be directly runnable from a repository
checkout. Keep the README concise and link to focused contracts rather than
copying their full contents. Update documentation and tests with every public
behavior change.

## Pull requests

Keep pull requests focused and explain:

- the problem and intended outcome;
- the boundaries affected;
- compatibility or migration implications;
- tests and validation performed; and
- any remaining limitations.

Before requesting review, confirm:

- [ ] the full required quality suite passes;
- [ ] no secrets, models, generated reports, or machine-specific paths are
      included;
- [ ] public behavior changes have tests and documentation;
- [ ] new metric labels are bounded;
- [ ] cancellation and cleanup behavior is preserved; and
- [ ] unrelated changes are excluded.

By contributing, you agree that your contribution may be distributed under
the repository's [MIT License](LICENSE).
