"""Output transaction recovery and tree validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from bookextract.canonical import publication_fingerprint_hex, publication_identifier
from bookextract.config import write_json_atomic
from bookextract.errors import ProcessingError
from bookextract.models import (
    EpubSemanticProfile,
    MarkdownSemanticProfile,
    OutputFileEntry,
    OutputManifest,
    OutputTransaction,
    PublicationDocument,
    PublicationMetadata,
    PublicationTextBlock,
    SourceMapEntry,
    TextRole,
    PublicationTextRun,
)
from bookextract.output_paths import validate_output_tree
from bookextract.output_publish import (
    output_destination,
    recover_output_transaction,
)
from bookextract.storage import RunStore


def _write_manifest(
    root: Path,
    *,
    command: str,
    committed_page_count: int,
    primary: str,
    primary_bytes: bytes,
) -> None:
    import hashlib

    digest = hashlib.sha256(primary_bytes).hexdigest()
    pub = PublicationDocument(
        metadata=PublicationMetadata(title="T"),
        blocks=[
            PublicationTextBlock(role=TextRole.PARAGRAPH, content=[PublicationTextRun(text="hi")])
        ],
        markdown_semantic_profile=MarkdownSemanticProfile(),
        epub_semantic_profile=EpubSemanticProfile(include_toc=True),
    )
    manifest = OutputManifest(
        output_manifest_format_version=2,
        command=command,  # type: ignore[arg-type]
        committed_page_count=committed_page_count,
        publication_identifier=publication_identifier(pub),
        publication_fingerprint=publication_fingerprint_hex(pub),
        source_map=[
            SourceMapEntry(
                publication_block_index=0,
                source_pages=[0],
                source_block_ids=["p001-b001"],
            )
        ],
        files=[
            OutputFileEntry(path=primary, sha256=digest, size_bytes=len(primary_bytes)),
        ],
    )
    (root / primary).write_bytes(primary_bytes)
    write_json_atomic(root / "manifest.json", manifest.model_dump(mode="json"))


def test_validate_output_tree_rejects_extra_file(run_dir: Path) -> None:
    store = RunStore(run_dir)
    candidate = store._path(".output-build", "markdown.candidate.test")
    candidate.mkdir(parents=True)
    _write_manifest(
        candidate,
        command="markdown",
        committed_page_count=0,
        primary="book.md",
        primary_bytes=b"# hi\n",
    )
    (candidate / "extra.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(ProcessingError, match="closure mismatch"):
        validate_output_tree(
            candidate,
            expected_command="markdown",
            expected_committed_page_count=0,
        )


def test_recover_previous_moved_publishes_candidate(run_dir: Path) -> None:
    store = RunStore(run_dir)
    store.write_head(1)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    candidate_name = "markdown.candidate.0123456789abcdef"
    candidate = build / candidate_name
    candidate.mkdir()
    _write_manifest(
        candidate,
        command="markdown",
        committed_page_count=1,
        primary="book.md",
        primary_bytes=b"# published\n",
    )
    previous = build / "markdown.previous"
    previous.mkdir()
    _write_manifest(
        previous,
        command="markdown",
        committed_page_count=0,
        primary="book.md",
        primary_bytes=b"# old\n",
    )
    write_json_atomic(
        build / "markdown.transaction.json",
        OutputTransaction(
            output_transaction_format_version=1,
            command="markdown",
            phase="previous-moved",
            candidate=candidate_name,
            previous="markdown.previous",
        ).model_dump(mode="json"),
    )

    recover_output_transaction(store, "markdown")

    destination = output_destination(store, "markdown")
    assert destination.is_dir()
    assert (destination / "book.md").read_text(encoding="utf-8") == "# published\n"
    assert not (build / "markdown.transaction.json").exists()
    assert not candidate.exists()
    assert not previous.exists()


def test_stale_candidate_not_published_when_head_advances(run_dir: Path) -> None:
    store = RunStore(run_dir)
    store.write_head(2)
    destination = output_destination(store, "markdown")
    destination.mkdir(parents=True)
    _write_manifest(
        destination,
        command="markdown",
        committed_page_count=1,
        primary="book.md",
        primary_bytes=b"# kept\n",
    )

    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    stale_name = "markdown.candidate.deadbeefdeadbeef"
    stale = build / stale_name
    stale.mkdir()
    _write_manifest(
        stale,
        command="markdown",
        committed_page_count=1,
        primary="book.md",
        primary_bytes=b"# stale\n",
    )
    write_json_atomic(
        build / "markdown.transaction.json",
        OutputTransaction(
            output_transaction_format_version=1,
            command="markdown",
            phase="candidate-valid",
            candidate=stale_name,
            previous="markdown.previous",
        ).model_dump(mode="json"),
    )

    recover_output_transaction(store, "markdown")

    assert (destination / "book.md").read_text(encoding="utf-8") == "# kept\n"
    assert not stale.exists()
