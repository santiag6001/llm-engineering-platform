"""Command-line interface for deterministic local RAG engineering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from llm_platform.rag.chunking import ChunkConfiguration
from llm_platform.rag.citation import citations_for
from llm_platform.rag.context import build_context
from llm_platform.rag.embedding import EmbeddingConfiguration
from llm_platform.rag.evaluation import (
    RetrievalDatasetError,
    evaluate_retrieval,
    experiment_metadata,
    load_retrieval_dataset,
)
from llm_platform.rag.index import IndexError, LocalVectorIndex
from llm_platform.rag.loader import (
    DocumentRegistry,
    DocumentRegistryError,
    DocumentValidationError,
)
from llm_platform.rag.retriever import Retriever, RetrieverConfiguration

EXIT_SUCCESS = 0
EXIT_INVALID_INPUT = 2
EXIT_STORAGE_ERROR = 3


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (ValidationError, ValueError, DocumentValidationError) as exc:
        print(f"error: {_safe_message(exc)}", file=sys.stderr)
        return EXIT_INVALID_INPUT
    except (DocumentRegistryError, IndexError, OSError) as exc:
        print(f"error: {_safe_message(exc)}", file=sys.stderr)
        return EXIT_STORAGE_ERROR


def _dispatch(args: argparse.Namespace) -> int:
    registry = DocumentRegistry(args.store)
    index = LocalVectorIndex(args.store / "index.json")
    if args.command == "ingest":
        record = registry.register(
            args.source,
            logical_name=args.name,
            content_type=args.content_type,
        )
        _print_json(record.model_dump(mode="json"))
    elif args.command == "build-index":
        manifest = index.build(
            registry,
            chunk_configuration=_chunk_configuration(args),
            embedding_configuration=EmbeddingConfiguration(
                model=args.embedding_model,
                dimension=args.dimension,
            ),
        )
        _print_json(manifest.metadata.model_dump(mode="json"))
    elif args.command == "retrieve":
        configuration = _retriever_configuration(args)
        results = Retriever(index, configuration).retrieve(args.query)
        context = build_context(results)
        _print_json(
            {
                "index_fingerprint": index.load().metadata.index_fingerprint,
                "retriever_configuration": configuration.model_dump(mode="json"),
                "results": [result.model_dump(mode="json") for result in results],
                "context": context.model_dump(mode="json"),
                "citations": [
                    citation.model_dump(mode="json")
                    for citation in citations_for(results)
                ],
            }
        )
    elif args.command == "evaluate":
        dataset = load_retrieval_dataset(args.dataset)
        report = evaluate_retrieval(index, dataset, _retriever_configuration(args))
        rendered = _json_text(report.model_dump(mode="json"))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        if args.experiment_metadata_output is not None:
            metadata = experiment_metadata(index, report)
            args.experiment_metadata_output.parent.mkdir(parents=True, exist_ok=True)
            args.experiment_metadata_output.write_text(
                _json_text(metadata.model_dump(mode="json")),
                encoding="utf-8",
            )
        print(rendered, end="")
    elif args.command == "inspect":
        payload: dict[str, object] = {
            "store": args.store.as_posix(),
            "document_count": len(registry.list_documents()),
            "document_fingerprint": registry.corpus_fingerprint(),
        }
        if index.path.exists():
            payload["index"] = index.load().metadata.model_dump(mode="json")
        else:
            payload["index"] = None
        _print_json(payload)
    elif args.command == "show-document":
        record = registry.get(args.document_id)
        _print_json(
            {
                "document": record.model_dump(mode="json"),
                "content": registry.content_text(record.document_id),
            }
        )
    elif args.command == "show-chunk":
        _print_json(index.get_chunk(args.chunk_id).model_dump(mode="json"))
    else:
        raise ValueError("unknown RAG command")
    return EXIT_SUCCESS


def _chunk_configuration(args: argparse.Namespace) -> ChunkConfiguration:
    return ChunkConfiguration(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        separator_strategy=args.separator_strategy,
    )


def _retriever_configuration(args: argparse.Namespace) -> RetrieverConfiguration:
    return RetrieverConfiguration(
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        mmr=args.mmr,
        mmr_lambda=args.mmr_lambda,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-rag",
        description="Build and evaluate deterministic local RAG artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="register one UTF-8 document")
    ingest.add_argument("source", type=Path, help="local source file")
    ingest.add_argument("--name", help="stable logical document name")
    ingest.add_argument(
        "--content-type",
        default="text/plain",
        choices=("text/plain", "text/markdown", "text/x-rst", "application/json"),
    )
    _store_argument(ingest)

    build = subparsers.add_parser(
        "build-index", help="deterministically rebuild the local vector index"
    )
    _store_argument(build)
    build.add_argument("--chunk-size", type=int, default=800)
    build.add_argument("--overlap", type=int, default=100)
    build.add_argument(
        "--separator-strategy",
        choices=("character", "line", "paragraph"),
        default="paragraph",
    )
    build.add_argument(
        "--embedding-model",
        choices=("local-hashing-embedding",),
        default="local-hashing-embedding",
    )
    build.add_argument("--dimension", type=int, default=256)

    retrieve = subparsers.add_parser(
        "retrieve", help="retrieve ranked chunks and structured provenance"
    )
    retrieve.add_argument("query", help="local retrieval query")
    _store_argument(retrieve)
    _retriever_arguments(retrieve)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate retrieval against a versioned local dataset"
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, help="optional JSON report path")
    evaluate.add_argument(
        "--experiment-metadata-output",
        type=Path,
        help="write RAG provenance accepted by llm-experiment run",
    )
    _store_argument(evaluate)
    _retriever_arguments(evaluate)

    inspect = subparsers.add_parser("inspect", help="inspect store and index metadata")
    _store_argument(inspect)
    show_document = subparsers.add_parser(
        "show-document", help="show registered document metadata and content"
    )
    show_document.add_argument("document_id")
    _store_argument(show_document)
    show_chunk = subparsers.add_parser(
        "show-chunk", help="show one indexed chunk and its offsets"
    )
    show_chunk.add_argument("chunk_id")
    _store_argument(show_chunk)
    return parser


def _store_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store",
        type=Path,
        default=Path("rag-data"),
        help="local RAG artifact root",
    )


def _retriever_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--mmr", action="store_true")
    parser.add_argument("--mmr-lambda", type=float, default=0.5)


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _print_json(value: object) -> None:
    print(_json_text(value), end="")


def _safe_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        errors = exc.errors(include_url=False, include_input=False)
        message = errors[0]["msg"] if errors else "validation failed"
    elif isinstance(exc, RetrievalDatasetError):
        message = str(exc)
    else:
        message = str(exc)
    bounded = " ".join(message.split())
    return bounded[:240] if bounded else "operation failed"


if __name__ == "__main__":
    raise SystemExit(main())
