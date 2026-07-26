# Production RAG engineering

## 1. Scope and architecture

Phase 7 provides a standalone local retrieval engineering path:

```text
UTF-8 sources -> immutable document registry -> deterministic chunks
              -> local CPU embeddings -> persistent vector index
              -> ranked retrieval -> context + citations
              -> retrieval evaluation -> experiment provenance
```

`llm_platform.rag` is not imported by FastAPI, the completion service, the
llama.cpp adapter, deployment, or observability. It does not add an HTTP
endpoint or change the OpenAI-compatible API. The CLI produces context and
structured provenance that a separately designed generation experiment can
consume.

Everything is local. There is no hosted embedding API, external vector
database, model download, GPU requirement, authentication, request queue,
Kubernetes resource, agent runtime, or hosted experiment service.

## 2. Document lifecycle

`llm-rag ingest` accepts bounded UTF-8 `text/plain`, Markdown, reStructuredText,
or JSON source bytes. The default 16 MiB limit is checked before reading.

Each immutable record contains:

- `document_id`: `doc-` plus the exact-byte SHA-256;
- exact-byte SHA-256 and byte size;
- a bounded logical name and explicit content type;
- a timezone-aware UTC ingestion timestamp; and
- a stable fingerprint over schema, content hash, logical name, size, and
  content type.

The timestamp is traceability metadata and is excluded from the stable
fingerprint. Exact duplicate content is rejected, including when supplied
under another logical name. Source bytes are stored separately from the sorted
strict `documents.json` manifest. Reads recheck the content checksum.

The corpus-level `document_fingerprint` hashes the ordered document
fingerprints. It therefore changes when registered document content or
identity metadata changes.

## 3. Chunking

Chunk configuration schema `1.0` records:

- character `chunk_size`;
- character `overlap`, which must be smaller than the size; and
- `character`, `line`, or `paragraph` separator strategy.

Line and paragraph strategies prefer the last separator in the latter half of
the target window, otherwise they use the exact character boundary. Offsets
always refer to Python Unicode character positions in the decoded document.
Forward progress is enforced even with overlap.

Every chunk records its content SHA-256, parent document ID, start/end
character offsets, zero-based index, text, chunk ID, and fingerprint. The
fingerprint includes the full chunk-configuration fingerprint; changing size,
overlap, or strategy changes chunk identities even when a particular chunk's
text happens to stay the same.

## 4. Local embedding and vector index

The versioned `local-hashing-embedding` model is a deterministic CPU lexical
baseline. It case-folds Unicode word features, maps SHA-256 feature hashes into
a configurable fixed dimension with stable signs, and L2-normalizes the
vector. It uses only the Python standard library and never downloads a model.

Embedding metadata contains model, model version, dimension, and a
configuration fingerprint. This baseline prioritizes reproducibility and
offline tests; it does not claim the semantic quality of a trained embedding
model.

`llm-rag build-index` rebuilds every registered document in document-ID order.
Entries use document ID, chunk index, and chunk ID as stable ordering keys.
`index.json` persists:

- corpus and individual document fingerprints;
- chunk configuration and ordered chunk fingerprints;
- embedding configuration and metadata;
- every chunk and vector;
- build timestamp and entry count; and
- an index fingerprint over all deterministic inputs and vectors.

The build timestamp is excluded from the index fingerprint. Rebuilding
unchanged inputs produces an identical fingerprint and entries. Loading checks
ordering, dimensions, counts, chunk identities, and the index fingerprint.

## 5. Retrieval and context

Retriever configuration schema `1.0` supports top-K from 1 to 100, an optional
inclusive cosine-score threshold, and optional maximal marginal relevance
(MMR) with a bounded lambda. Base candidates sort by descending score and then
chunk ID. MMR selections use the same chunk-ID tie-break, so rankings are
stable.

Every result records chunk ID, document ID, cosine score, one-based rank,
character range, chunk index, chunk fingerprint, and text.

Context assembly preserves retrieval order and joins text with an explicit
separator. It records chunk ordering, separator policy, a deterministic
`ceil(characters / 4)` token estimate, and a fingerprint over ordered chunk
identities, content, separator, and estimator. This is an estimate, not a
model-specific tokenizer claim.

## 6. Citation provenance

One structured citation is emitted per retrieval result:

```json
{
  "document_id": "doc-...",
  "chunk_id": "chunk-...",
  "character_start": 0,
  "character_end": 120,
  "score": 0.75
}
```

Citation correctness verifies exact correspondence to retrieved document,
chunk, range, and score provenance. A failed or modified citation does not
silently pass. The RAG layer assembles retrieval evidence; it does not mutate
the existing chat-completion answer schema.

## 7. Retrieval evaluation

Evaluation datasets are bounded strict JSON:

```json
{
  "schema_version": "1.0",
  "cases": [
    {
      "schema_version": "1.0",
      "id": "provenance",
      "query": "How is provenance recorded?",
      "relevant_chunk_ids": ["chunk-..."]
    }
  ]
}
```

Case IDs and relevant chunk IDs must be unique. Evaluation performs no network
request and no generation. Metrics are averaged across cases:

- **Precision@K:** relevant returned chunks divided by configured K;
- **Recall@K:** relevant returned chunks divided by expected relevant chunks;
- **MRR:** reciprocal rank of the first relevant result, or zero;
- **Hit Rate:** fraction of cases with at least one relevant result;
- **Citation correctness:** fraction of emitted citations matching retrieval
  provenance; and
- **Context utilization:** relevant returned chunks divided by assembled
  context chunks, or zero for empty context.

Reports preserve ordered per-case results and context fingerprints.

## 8. Experiment integration

`llm-rag evaluate --experiment-metadata-output rag-metadata.json` writes a
strict portable provenance object containing:

- corpus/document fingerprint;
- chunk configuration and ordered chunk fingerprints;
- embedding configuration;
- index fingerprint;
- retriever configuration;
- retrieval metrics; and
- citation metrics.

Pass it to the existing experiment runner:

```bash
llm-experiment run \
  --dataset evaluations/datasets/serving-concepts.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model local-model \
  --rag-metadata rag-metadata.json
```

The immutable experiment manifest embeds the strict RAG block, and RAG
provenance participates in the experiment fingerprint. The reproduction
specification records the portable metadata path. Without `--rag-metadata`,
the manifest contains `rag: null` and the Phase 6 experiment fingerprint
algorithm is unchanged.

## 9. CLI workflow

```bash
llm-rag ingest docs/rag.md --store rag-data --content-type text/markdown
llm-rag build-index --store rag-data \
  --chunk-size 800 --overlap 100 --separator-strategy paragraph \
  --dimension 256
llm-rag retrieve "How are citations verified?" --store rag-data --top-k 5
llm-rag inspect --store rag-data
llm-rag show-document <document-id> --store rag-data
llm-rag show-chunk <chunk-id> --store rag-data
llm-rag evaluate --dataset retrieval-dataset.json --store rag-data \
  --top-k 5 --output retrieval-report.json \
  --experiment-metadata-output rag-metadata.json
```

Exit `0` means success, `2` means invalid input/configuration, and `3` means a
registry/index integrity or storage failure. Expected failures are bounded and
traceback-free.

Generated `rag-data/` is ignored by Git and excluded from the gateway build
context. It may contain source content and must be handled according to the
source data's privacy requirements.

## 10. Limitations

- The local hashing model is lexical and collision-prone at small dimensions.
- The JSON index loads fully into memory and is intended for bounded local
  experiments, not large-scale or concurrent serving.
- There is no incremental index mutation: rebuilding is the explicit,
  deterministic update operation.
- Index fingerprints detect accidental changes but are not signatures or
  trusted provenance.
- RAG does not add answer generation, an HTTP RAG endpoint, or an agent loop.
