# Evaluation and regression

## 1. Scope and architecture

Phase 4 provides a production-oriented evaluation client for the existing
OpenAI-compatible platform. It is deliberately separate from the serving
runtime:

```text
JSONL dataset -> fixed async workers -> POST /v1/chat/completions
              -> deterministic evaluators -> JSON and Markdown
              -> optional baseline gates
```

The evaluation package imports httpx and Pydantic but no FastAPI route,
completion service, backend adapter, serving settings, or Prometheus adapter.
Conversely, the serving application does not import the evaluation package.
Requests are non-streaming, are never retried, and use a fixed worker count so
task creation and simultaneous requests are bounded.

## 2. Dataset schema

A dataset is UTF-8 JSON Lines. Blank lines are ignored. Each non-empty line is
one case with this strict schema:

| Field | Required | Semantics |
|---|---|---|
| `schema_version` | yes | Literal `"1.0"` |
| `id` | yes | Unique stable ID, 1–128 safe identifier characters |
| `category` | yes | Lowercase bounded category, 1–64 characters |
| `messages` | yes | 1–128 `system`, `user`, or `assistant` text messages |
| `expected` | yes | At least one configured deterministic expectation |
| `generation` | no | `temperature` in 0–2 and/or positive `max_tokens` |
| `metadata` | no | Description up to 500 characters and up to 16 unique tags |

`expected` accepts:

| Field | Type | Meaning |
|---|---|---|
| `exact_match` | string | Entire normalized response must match |
| `contains_all` | non-empty string list | Every normalized value must occur |
| `contains_any` | non-empty string list | At least one normalized value must occur |
| `not_contains` | non-empty string list | No normalized value may occur |
| `minimum_characters` | non-negative integer | Raw response lower length bound |
| `maximum_characters` | non-negative integer | Raw response upper length bound |
| `case_sensitive` | boolean | Defaults to `false` |
| `normalize_whitespace` | boolean | Defaults to `true` |

The minimum character count cannot exceed the maximum. Lists cannot be empty
or contain empty strings. Unknown fields at every dataset schema level are
rejected. Malformed JSON, invalid UTF-8, duplicate IDs, an empty dataset, an
empty message list, unsupported roles, an empty evaluator configuration, and
invalid generation values fail loading before any request is sent.

Dataset files are limited to 16 MiB and 10,000 cases. Individual message
content is limited to 32,768 characters, exact-match values to 4,096
characters, containment lists to 32 values, and each containment value to 256
characters.

The dataset SHA-256 is calculated over the exact source bytes. Reformatting or
changing line endings intentionally changes dataset identity.

## 3. Evaluator semantics

Every successful platform response is checked for non-empty text. The
configured evaluators then run in fixed order: exact match, contains all,
contains any, forbidden strings, and response length.

Each result contains:

- a stable evaluator name;
- pass/fail;
- a 0–1 score when meaningful;
- a concise failure reason;
- a bounded expected summary; and
- a bounded actual summary.

Contains-all score is the fraction of required values found. Contains-any score
is also the fraction found, while pass requires at least one. Binary
evaluators use 0 or 1. Evaluator exceptions become a safe
`evaluator_failure`; they do not stop other cases and no traceback is written
to a report.

## 4. Normalization

When `normalize_whitespace` is true, all Unicode whitespace runs are collapsed
to one ASCII space and leading/trailing whitespace is removed. When
`case_sensitive` is false, Unicode `casefold()` is applied after whitespace
normalization. Exact and containment comparisons use the resulting text.
Response-length bounds use the unnormalized Python character count.

There is no stemming, tokenization, edit distance, fuzzy matching, semantic
embedding, or LLM-as-a-Judge.

## 5. Runner behavior and outcomes

The runner sends:

```http
POST <base-url>/v1/chat/completions
Content-Type: application/json
```

with the configured model, case messages, optional generation settings, and
`stream: false`. A fixed number of workers pull successive dataset indices.
At most `maximum_concurrency` requests are active and at most that many worker
tasks exist. Results are placed back into their original indices, so reports
always follow dataset order.

One case has either `completed` status or `error` status. Errors are:

| Type | Meaning |
|---|---|
| `platform_http_error` | Any platform non-2xx response |
| `timeout` | Configured HTTP timeout |
| `connection_failure` | Other httpx request/connection failure |
| `malformed_platform_response` | Invalid JSON or unsupported completion shape |
| `evaluator_failure` | Unexpected deterministic evaluator failure |

Safe fixed messages are reported. Raw platform bodies, transport exception
text, Python exception types, and tracebacks are excluded. One case error does
not cancel the remaining cases. There are no retries.

## 6. Performance measurements and aggregates

Each case records end-to-end buffered HTTP duration. When the validated
response supplies them, it also records prompt, completion, and total tokens,
the backend response model, and first-choice finish reason.

TTFT is always `null` in the JSON measurement and described as unavailable in
Markdown. Buffered HTTP responses reveal only completion time; claiming TTFT
would be incorrect until a future streaming evaluation mode exists.

Aggregate fields are:

- total, completed, passed, failed, and error cases;
- pass rate (`passed / total`);
- error rate (`errors / total`);
- average, P50, and P95 end-to-end request duration;
- total reported prompt tokens; and
- total reported completion tokens.

Durations include all request attempts, including HTTP/transport errors, when a
duration was captured. Percentiles use the deterministic nearest-rank method:
sort ascending and select rank `ceil(p * N)`, using one-based rank. Empty
samples produce `null`.

## 7. JSON report schema

Report schema version `1.0` contains:

- `run_id` and UTC ISO-8601 `timestamp`;
- `dataset_path` and exact-byte `dataset_content_hash`;
- `platform` base URL, timeout, and maximum concurrency;
- `model_requested`;
- `aggregate` metrics from section 6;
- ordered `cases`, including evaluator results, measurements, bounded response
  preview, and safe error where applicable; and
- optional `git` commit hash and dirty flag.

The JSON schema is strictly parsed when used as a baseline. Report response
previews and evaluator summaries are single-line and limited to 240
characters. Complete prompts are never copied into reports.

Markdown reports summarize configuration, quality, latency, token usage,
failed cases, error cases, and a regression section when a comparison is
attached. They do not contain full prompts or unbounded model output.

## 8. Regression gates

`llm-eval compare` accepts one or more gates:

| CLI option | Gate passes when |
|---|---|
| `--min-pass-rate R` | current pass rate is at least `R` |
| `--max-pass-rate-drop D` | baseline minus current pass rate is at most `D` |
| `--max-error-rate R` | current error rate is at most `R` |
| `--max-p95-latency-seconds S` | current P95 is at most `S` |
| `--max-p95-latency-increase-percent P` | percentage increase from baseline P95 is at most `P` |

Rates and drops are fractions from 0 to 1. Latencies must be positive.
Percentage increase must be non-negative and is
`(current - baseline) / baseline * 100`. A non-positive or missing baseline
P95 fails the relative gate safely. Any missing metric required by an enabled
gate fails that gate. All gates are evaluated even after a failure.

Raw answer differences are informational only. They become regressions only
when deterministic evaluators affect the aggregate pass rate or another
configured gate fails.

## 9. Exit codes

| Code | Meaning |
|---:|---|
| `0` | Run completed without operational case errors, or all comparison gates passed |
| `1` | At least one comparison gate failed |
| `2` | CLI argument, dataset, report, or threshold input was invalid |
| `3` | Run reports were written but at least one case had an operational error |

Ordinary evaluator answer failures do not make `run` non-zero. CI quality
policy belongs in explicit `compare` gates.

## 10. Privacy and bounded-output policy

Reports do not store input messages. Model content appears only as bounded
single-line previews. Expected values and failure summaries use the same
bound. HTTP error bodies, transport exception messages, tracebacks,
credentials, and authorization headers are not reported. Dataset paths and the
configured base URL are operational metadata; base URLs should not contain
credentials.

Git metadata is best-effort. Failure to call Git simply omits it and never
fails an evaluation.

## 11. Current limitations

- Only buffered chat completions are evaluated; streaming behavior and TTFT
  are outside this phase.
- Lexical evaluators cannot establish broad semantic correctness.
- No LLM-as-a-Judge, external hosted LLM API, fuzzy matcher, or embedding model
  is included.
- There is no request retry, warm-up phase, saturation workload, or hardware
  resource capture.
- Token fields are trusted only after structural/non-negative validation; the
  evaluator does not independently tokenize.
- Reports are schema-versioned but no cross-version migration tool exists yet.
- Example cases teach serving concepts and are not a quality promise for the
  small local Qwen model.

## 12. Phase 6 experiment integration

The Phase 6 experiment runner imports and reuses `EvaluationRunner`,
`build_report`, JSON/Markdown rendering, `EvaluationReport`, and the regression
gate functions. Dataset parsing, worker ordering, request payloads, evaluator
semantics, report schema `1.0`, and `llm-eval` exit codes are unchanged.

Experiment manifests wrap rather than replace evaluation reports. The
experiment fingerprint uses the exact dataset content hash, and generated
`evaluation.json` remains valid input to `llm-eval compare`. The FastAPI
application imports neither package.
