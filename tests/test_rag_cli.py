from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_platform.rag.cli import main


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["ingest", "--help"],
        ["build-index", "--help"],
        ["retrieve", "--help"],
        ["evaluate", "--help"],
        ["inspect", "--help"],
        ["show-document", "--help"],
        ["show-chunk", "--help"],
    ],
)
def test_every_rag_command_has_help(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 0


def test_cli_ingest_build_retrieve_evaluate_and_inspect(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "rag"
    document = tmp_path / "source.txt"
    document.write_text("retrieval provenance citation", encoding="utf-8")

    assert main(["ingest", str(document), "--store", str(store)]) == 0
    document_result = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "build-index",
                "--store",
                str(store),
                "--chunk-size",
                "100",
                "--overlap",
                "0",
                "--dimension",
                "32",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            ["retrieve", "retrieval provenance", "--store", str(store), "--top-k", "1"]
        )
        == 0
    )
    retrieval = json.loads(capsys.readouterr().out)
    assert retrieval["results"][0]["document_id"] == document_result["document_id"]
    chunk_id = retrieval["results"][0]["chunk_id"]

    dataset = tmp_path / "retrieval-dataset.json"
    dataset.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "cases": [
                    {
                        "schema_version": "1.0",
                        "id": "case",
                        "query": "retrieval provenance",
                        "relevant_chunk_ids": [chunk_id],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata = tmp_path / "rag-metadata.json"
    assert (
        main(
            [
                "evaluate",
                "--dataset",
                str(dataset),
                "--store",
                str(store),
                "--top-k",
                "1",
                "--experiment-metadata-output",
                str(metadata),
            ]
        )
        == 0
    )
    evaluation = json.loads(capsys.readouterr().out)
    assert evaluation["retrieval_metrics"]["hit_rate"] == 1
    assert (
        json.loads(metadata.read_text())["citation_metrics"]["citation_correctness"]
        == 1
    )

    assert (
        main(["show-document", document_result["document_id"], "--store", str(store)])
        == 0
    )
    assert "retrieval provenance citation" in capsys.readouterr().out
    assert main(["show-chunk", chunk_id, "--store", str(store)]) == 0
    assert json.loads(capsys.readouterr().out)["chunk_id"] == chunk_id
    assert main(["inspect", "--store", str(store)]) == 0
    assert json.loads(capsys.readouterr().out)["document_count"] == 1
