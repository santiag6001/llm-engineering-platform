# Reproducible LLM engineering

## 1. Goals and non-goals

Phase 6 adds a standalone, local experiment layer around the Phase 4 evaluation
client. Its purpose is configuration reproducibility, execution-environment
traceability, deterministic report structure, and artifact integrity. A run
records the code, dataset, requested and observed model identities, generation
and evaluator configuration, regression policy and decision, runtime
environment, aggregate results, and immutable artifact locations.

The experiment package is not imported by FastAPI, the completion service, the
backend adapter, or observability. It calls the existing evaluation runner and
regression functions rather than providing another inference path.

This phase does not provide bit-for-bit model-output reproducibility. Sampling,
backend implementation details, thread scheduling, CPU kernels, and model
versions can affect generated text even when the recorded configuration is the
same. It also does not add MLflow, Weights & Biases, Langfuse, a database,
hosted storage, signing, authentication, model downloading, RAG, agents,
fine-tuning, or deployment orchestration.

## 2. Identity and canonicalization

`run_id` identifies one execution. Its default form is a UTC timestamp, the
first 12 hexadecimal characters of the experiment fingerprint, and an
eight-character random nonce. Two executions of the same configuration have
different run IDs.

`experiment_fingerprint` identifies equivalent canonical inputs. Canonical JSON
uses UTF-8, sorted object keys, no insignificant whitespace, JSON-native values,
and rejects non-finite numbers. SHA-256 is calculated over those bytes.

Fingerprint schema `1.0` includes exactly:

- experiment manifest schema version;
- exact-byte dataset SHA-256;
- requested model;
- experiment-level temperature and maximum-token defaults;
- evaluator name, version, and dataset-defined-expectation policy;
- evaluation concurrency and timeout;
- resolved immutable baseline run ID and all configured regression gates;
- shared prompt logical name, version, and exact-byte content SHA-256, when
  configured; and
- source Git commit, when available; and
- optional bounded deployment runtime, image reference, and configuration name.

The fingerprint excludes run ID, creation time, branch, dirty flag, dataset and
prompt file paths, base URL, environment, output metrics, backend-observed
model, generated content, and artifact checksums. Dataset paths and prompt
source paths are traceability metadata only. The exact dataset hash already
captures all per-case messages, expectations, metadata, and case generation
settings. Prompt content is not normalized before hashing.

## 3. Manifest schema

Every registered run has a strict `manifest.json` with schema version `1.0`.
Unknown fields are rejected. The manifest contains:

- run ID, experiment fingerprint, UTC creation time, and status;
- Git commit, dirty state, and branch when Git is usable;
- dataset logical identifier, portable path, exact-byte hash, and case count;
- optional shared prompt identity without prompt content;
- requested model and a single backend-observed model when available;
- experiment generation defaults;
- evaluator identity, concurrency, and timeout;
- user-supplied baseline reference, resolved baseline run ID, gates, and
  `passed`, `failed`, or `not_evaluated` decision;
- bounded environment metadata and its fingerprint;
- optional deployment runtime, image reference, and configuration name;
- artifact kind, portable relative path, byte size, and SHA-256;
- aggregate pass/error rates, nearest-rank P50/P95 duration, and token totals;
- a bounded reproduction specification; and
- a fixed safe failure message for failed runs.

Run statuses are `completed`, `regression_failed`, and `failed`. A deterministic
answer mismatch is represented in evaluation aggregates and does not by itself
make the run operationally failed. A configured regression-gate failure uses
`regression_failed`. Evaluation request/evaluator errors or an unexpected
operational failure use `failed`.

Complete prompts and responses are not copied into the manifest. The underlying
evaluation JSON keeps the existing bounded 240-character response previews and
contains no input messages.

## 4. Prompt and generation identity

Dataset messages and per-case generation fields remain owned by the versioned
JSONL dataset and therefore by its exact-byte hash. Optional experiment-level
`--temperature` and `--max-tokens` values fill only missing per-case fields and
are recorded separately.

An optional shared system prompt is deliberately small:

```bash
--prompt-file experiments/examples/concise-system-prompt.txt \
--prompt-name concise-system \
--prompt-version 1
```

The exact UTF-8 file bytes are hashed, and the text is prepended without
normalization to each case. The manifest stores its logical name, version,
hash, and portable source path, not its content. There is no prompt catalog,
template language, or remote prompt service.

## 5. Environment allowlist

Environment capture includes only:

- Python version and implementation;
- operating-system name and platform release;
- machine architecture;
- project package version;
- `docker`, generic `container`, `none`, or `unknown` detection;
- versions of the explicit `httpx` and `pydantic` dependency allowlist.

The environment fingerprint is canonical SHA-256 over exactly those fields.
No full package inventory or environment-variable dump is collected. Hostname,
username, home directory, IP addresses, credentials, authorization headers,
and arbitrary environment variables are excluded. Optional backend model
identity is taken from validated evaluation responses; Phase 4 does not expose
a backend version/system fingerprint in its report schema.

## 6. Registry and atomicity

The default registry layout is:

```text
experiments/
├── runs/<run-id>/
│   ├── manifest.json
│   ├── evaluation.json          # when evaluation produced a report
│   ├── evaluation.md            # when evaluation produced a report
│   ├── regression.json
│   ├── summary.md
│   └── checksums.json
├── aliases/<alias>.json
├── comparisons/
└── examples/
```

A run is written to a unique `.staging-*` directory under `runs/`. Artifact
files, the checksum index, and the manifest are created without overwriting,
flushed, and the directory is renamed to the final run ID on the same
filesystem. The final name is never reused. Concurrent registration of the
same run ID yields one winner and one duplicate-run error. Staging directories
are ignored by listing and removed after local registration failures.

All persisted paths are portable relative POSIX paths. Absolute paths, parent
components, empty components, and backslashes are rejected. Verification opens
files without following a final symlink and rejects symlinks in artifact path
components. Registry roots and primary directories may not be symlinks.
Individual registered artifacts are limited to 64 MiB.

A completed manifest is written only after its referenced artifacts exist in
staging. Expected evaluation failures receive an immutable failed-run manifest
where dataset identity and a safe failure summary are already available.
Invalid CLI configuration or an unreadable dataset fails before a run can be
identified and is not registered.

## 7. Artifact integrity

`checksums.json` mirrors every manifest artifact's SHA-256 and byte size.
`llm-experiment verify` strictly parses the manifest and checksum index, checks
that every referenced path stays inside the run, rejects unsafe symlinks,
checks file existence and byte size, and recalculates SHA-256. A missing,
modified, escaped, or malformed artifact returns exit code 4.

This is accidental/casual tamper detection, not cryptographic provenance.
There are no signatures, trusted timestamps, transparency logs, or remote
attestation in Phase 6.

## 8. Aliases

Aliases such as `latest`, `candidate`, `baseline`, and `production` are bounded
lowercase names. An alias can point only to an existing immutable run ID.
Updates write a temporary record and atomically replace the alias file.

Aliases are mutable pointers without history. A run using an alias records both
the supplied alias and the resolved immutable baseline run ID. Experiments
never create or update an alias automatically.

## 9. Experiment orchestration

The runner performs:

1. strict configuration and dataset validation;
2. baseline resolution and integrity verification;
3. source, prompt, and environment identity collection;
4. fingerprint and unique run-ID creation;
5. invocation of the existing fixed-worker `EvaluationRunner`;
6. existing evaluation JSON and Markdown rendering;
7. optional invocation of the existing regression comparison and gates;
8. summary, checksum, and manifest construction; and
9. atomic registry registration.

Dataset order, evaluator semantics, response previews, percentile calculation,
Git fields in evaluation reports, and Phase 4 exit meaning are unchanged.
There are no retries and no backend is required by automated tests.

## 10. Reproduction specification

The manifest contains a structured reproduction specification with the
portable dataset path, requested model, `${LLM_PLATFORM_BASE_URL}` placeholder,
generation defaults, evaluation concurrency/timeout, resolved baseline run ID,
gate values, source commit, and project version. `summary.md` renders the
corresponding command without credentials.

This specification reconstructs configuration. The environment section
supports audit and comparison but does not install packages or restore an
operating system. The source commit may not reproduce dirty working-tree
changes; `git_dirty=true` explicitly warns that the commit alone is
insufficient.

## 11. Comparison semantics

`llm-experiment compare` accepts run IDs or aliases and writes JSON and Markdown.
It classifies differences as:

- input/configuration: fingerprint, dataset hash, requested/backend model, and
  generation defaults;
- source-code: Git commit and dirty state;
- environment: environment fingerprint;
- quality: pass/error rates, token totals, and regression decision;
- performance: P50/P95 duration; and
- artifacts: artifact paths and checksums.

Comparison is an audit view. It reports differences but never calls them a
failure. The existing regression gates remain the only experiment pass/fail
policy.

## 12. CLI and exit codes

```bash
llm-experiment run --dataset evaluations/datasets/serving-concepts.jsonl \
  --base-url http://127.0.0.1:8000 --model local-model \
  --registry-dir experiments --max-concurrency 2 --timeout-seconds 120
llm-experiment list --registry-dir experiments
llm-experiment show latest --registry-dir experiments
llm-experiment alias set baseline <run-id> --registry-dir experiments
llm-experiment alias show baseline --registry-dir experiments
llm-experiment compare baseline candidate --registry-dir experiments \
  --output-dir experiments/comparisons
llm-experiment verify baseline --registry-dir experiments
```

Exit codes are:

| Code | Meaning |
|---:|---|
| `0` | Successful operation or completed run whose configured gates passed |
| `1` | Completed experiment with at least one regression-gate failure |
| `2` | Invalid command, configuration, manifest input, prompt, or dataset |
| `3` | Operational evaluation failure with a failed manifest preserved |
| `4` | Registry or artifact integrity failure |
| `5` | Requested run or alias not found |

Expected failures print bounded messages without tracebacks.

## 13. Privacy and limitations

Do not put credentials in a base URL, dataset path, prompt path, alias, or model
name. The CLI rejects URL user info, query strings, and fragments. Manifests do
not contain base URLs, prompt text, response text, host paths, or arbitrary
environment state. Evaluation artifacts retain their existing privacy policy.

Generated runs, aliases, and comparisons are ignored by Git by default.
Curated examples and placeholder files remain tracked. The registry has no
cross-machine locking protocol, history for aliases, schema migration utility,
garbage collection, signing, remote replication, database indexing, UI, or
external experiment-platform integration.
