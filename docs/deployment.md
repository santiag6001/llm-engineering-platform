# Deployment

## 1. Scope

Phase 5 packages and validates the existing Phase 1–4 gateway without changing
its public API, streaming, cancellation, metrics, or evaluation behavior. The
deployment supports a gateway-only workflow, optional Prometheus, and an
optional CPU llama-server using a user-supplied GGUF model. It does not add
queueing, concurrency control, Kubernetes, Grafana, authentication, or a
registry release process.

## 2. Image architecture

`deploy/docker/Dockerfile` uses the pinned Python
`3.12.11-slim-bookworm` base for both stages:

1. The builder creates a virtual environment, installs the exact runtime lock,
   builds the project wheel, installs it without dependency resolution, and
   runs `pip check`.
2. The runtime stage creates UID/GID `10001`, copies only the populated virtual
   environment, switches to that unprivileged identity, and starts Uvicorn with
   an exec-form command.

The final image contains no compiler, test dependency, repository checkout,
test, document, evaluation data, model, cache, report, Git metadata, or local
virtual environment from the host. The builder also removes `pip` after the
installed environment passes its dependency check. `.dockerignore` removes
unneeded sources from the build context. Python bytecode is disabled, logs stay
unbuffered on stdout/stderr, port 8000 is documented with `EXPOSE`, and
`--reload` is never used.

The image tag is a human-readable local Phase 5 tag, not a claim of immutable
publication. The Python tag and all Python runtime packages are explicitly
pinned. Rebuilding still trusts the configured container registry and package
index; artifact signatures, an internal mirror, hashes, SBOM generation, and
registry promotion are future release-hardening decisions.

## 3. Runtime configuration

The application reads only process environment variables. Precedence is:

1. values exported in the invoking shell or supplied by `--env-file`;
2. Compose substitutions and defaults in `compose.yaml`;
3. application defaults when run outside Compose.

Compose passes these supported settings:

| Variable | Compose default | Meaning |
|---|---|---|
| `LLAMA_SERVER_BASE_URL` | `http://host.docker.internal:8080` | Backend origin |
| `LLAMA_SERVER_TIMEOUT_SECONDS` | `120` | Buffered backend HTTP timeout |
| `LLAMA_SERVER_STREAM_IDLE_TIMEOUT_SECONDS` | `30` | Maximum wait between upstream stream reads |
| `LLAMA_SERVER_STREAM_EVENT_MAX_BYTES` | `1048576` | Maximum buffered upstream SSE event |
| `LLM_PLATFORM_MODEL` | `local-model` | Public model identifier |
| `GATEWAY_PORT` | `8000` | Host-only Compose port substitution |

The application does not load `.env` files. Compose may load its conventional
`.env` for interpolation, but `.env*` is excluded from the image and must not
be committed. Settings validation fails startup with a bounded Pydantic error.
URLs containing credentials are unsafe because URLs are operational
configuration and may appear in local tooling; use no embedded credentials.

## 4. Compose services and profiles

The base `compose.yaml` defines:

- `gateway`, always available, built from the repository and bound to
  `127.0.0.1:${GATEWAY_PORT}`;
- `prometheus`, enabled through profile `observability`, with its data in a
  named volume and its port bound to loopback.

The gateway reaches a host llama-server through the explicit
`host.docker.internal:host-gateway` mapping. This preserves a useful
model-free gateway workflow and avoids host networking.

`compose.llama.yaml` adds profile `inference` and rewires the gateway to
`http://llama-server:8080`. Render and start it with:

```bash
LLAMA_MODEL_PATH=./models/your-model.gguf \
docker compose -f compose.yaml -f compose.llama.yaml \
  --profile inference up --build gateway llama-server
```

Add `--profile observability prometheus` to the command to start all three
services. Compose networks are project-scoped defaults; no service uses host
networking, privileged mode, the Docker socket, or an NVIDIA runtime.

## 5. Local model mounting policy

The optional overlay interpolates `LLAMA_MODEL_PATH` with Compose's required
value syntax. If it is absent, configuration rendering stops with:

```text
Set LLAMA_MODEL_PATH to an existing local GGUF file
```

The selected host file is mounted read-only as `/models/model.gguf`. It is
never copied into an image, volume, or repository and is never downloaded.
Use an absolute Linux path or a repository-relative path on Ubuntu. Under WSL2,
a path inside the Linux filesystem generally provides better and more
predictable bind-mount behavior than `/mnt/c/...`; Docker Desktop integration
must be enabled. Docker also reports a clear bind-mount error if the resolved
path does not exist or is not readable.

The default llama.cpp server flags are CPU-compatible: context size 2048 and
four threads. Tune `LLAMA_CONTEXT_SIZE` and `LLAMA_THREADS` for the local
machine. The reviewed default server image can be replaced with
`LLAMA_SERVER_IMAGE`; pin that override rather than using a floating tag.

## 6. Liveness, readiness, and shutdown

The Dockerfile and Compose health check call `GET /health` using Python's
standard library:

- `/health` and `/health/live` report process-local liveness and return 200
  while the FastAPI process is serving;
- `/ready` and `/health/ready` probe llama-server and return 200 only when that
  backend responds successfully, otherwise 503 with `status=unavailable`.

Liveness intentionally does not depend on readiness. A gateway-only container
with no backend is healthy but not ready for inference. Operators and traffic
routing must use readiness when backend usability is required.

Uvicorn runs as PID 1 behind Compose's minimal init process. Compose sends
`SIGTERM`, allows the gateway 15 seconds, and the application lifespan closes
the one shared asynchronous backend client before process exit. The current
runtime has no scheduler drain phase; that belongs to a later phase. The
model-free smoke test checks that ordinary termination returns exit status 0.

## 7. Prometheus

`deploy/prometheus/prometheus.yml` scrapes `gateway:8000/metrics` every 15
seconds with a 10-second timeout. It contains no credentials, service
discovery, alert rules, or remote write. Prometheus starts only under the
`observability` profile and waits for gateway liveness.

The gateway owns one private registry per process. Metrics are process-local
and are not global limits or cross-replica aggregates. Prometheus persistence
uses the `prometheus-data` named volume; the checked-in configuration is
read-only.

## 8. Model-free smoke validation

Run:

```bash
python3 scripts/container_smoke.py
```

The standard-library helper chooses a temporary loopback port and unique
Compose project, configures an intentionally absent backend, builds and starts
only the gateway, and verifies:

- `/health` returns the documented 200 body;
- `/metrics` returns Prometheus text containing platform metrics;
- `/ready` returns the documented 503 body;
- none of those client bodies contains a Python traceback;
- `SIGTERM` stops the gateway with exit status 0.

Cleanup runs even on failure. The script does not implement inference or mock a
large backend framework. In CI, `--skip-build` reuses the image from the
explicit preceding build step.

## 9. CI architecture

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`.
Workflow permissions are read-only, jobs have timeouts, and concurrency
cancels superseded runs.

The quality job uses Python 3.12, installs the exact runtime and development
locks plus the project as an editable no-dependency install, then runs:

1. `ruff format --check .`
2. `ruff check .`
3. `mypy`
4. `pytest`
5. `python -m pip check`
6. `llm-eval --help`, `run --help`, and `compare --help`
7. every `llm-experiment` command help surface
8. the deterministic offline experiment-registry/artifact smoke test
9. every `llm-rag` command help surface
10. `git diff --check`

The parallel container job validates base Compose rendering, builds the
gateway, and runs the model-free smoke test. It does not pull a model, contact
a running llama-server or hosted LLM API, use a GPU, consume secrets, log into
a registry, push an image, or require cloud credentials.

Phases 6 and 7 extend only the deterministic quality job. Neither the
experiment registry nor RAG is a Compose service; generated runs and
`rag-data/` are excluded from the gateway build context, and the image runtime
command remains unchanged.

GitHub Actions are referenced by pinned major release lines. Docker commands
use the hosted runner's installed Compose plugin; no Docker socket is mounted
inside a workload container.

## 10. Evaluation through the deployment

Start the optional local backend and gateway, wait for `/ready` to return 200,
and then run from the host:

```bash
llm-eval run \
  --dataset evaluations/datasets/serving-concepts.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model local-model \
  --output-dir evaluations/reports
```

The output directory is already host-local because evaluation is a separate
CLI, not a Compose service. Review a report before using it as a baseline, then
run `llm-eval compare` with explicit gates. Neither Compose nor CI updates
baselines automatically, and generated reports remain ignored by default.

## 11. Security and supply-chain assumptions

- The gateway runs as a fixed non-root identity with all Linux capabilities
  dropped, `no-new-privileges`, a read-only root filesystem, and bounded
  writable `/tmp`.
- The optional llama-server also uses UID/GID `10001`, a read-only root
  filesystem, bounded writable `/tmp`, dropped capabilities, and
  `no-new-privileges`. Prometheus drops capabilities and cannot gain
  privileges.
- Published ports bind to loopback by default.
- No service is privileged, uses host networking, or mounts the Docker socket.
- Images, Python, and Python packages use scoped version pins; no credentials,
  `.env` file, or model artifact enters the gateway build context.
- Generation requests are not retried and no image is pushed by CI.

The deployment is still unauthenticated and trusts public base-image/package
registries at build time. It does not yet verify content digests or Python
artifact hashes. Operators should not expose it directly to an untrusted
network.

## 12. Current limitations

- Readiness probes every request directly; there is no readiness cache or
  startup dependency gate.
- The gateway is a single Uvicorn process. Metrics are process-local.
- There is no runtime request queue, concurrency bound, lifecycle drain state,
  authentication, rate limiting, Grafana, alerting, Kubernetes, or distributed
  coordination.
- The optional llama.cpp image compatibility depends on the chosen reviewed
  version and local GGUF model.
- CI validates model-free deployment behavior only. A real backend smoke test
  is explicitly optional and machine-local.
- Container publishing, signed provenance, SBOM generation, vulnerability
  scanning, and dependency update automation are deferred to release
  hardening.
