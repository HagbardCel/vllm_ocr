"""Output transaction recovery and tree validation tests."""

from __future__ import annotations

import hashlib
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
    PublicationTextRun,
    SourceMapEntry,
    TextRole,
)
from bookextract.output_paths import validate_output_tree
from bookextract.output_publish import (
    begin_output_transaction,
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
    sha256_override: str | None = None,
) -> None:
    digest = sha256_override or hashlib.sha256(primary_bytes).hexdigest()
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


def _assert_recovery_artifacts_cleared(store: RunStore, command: str) -> None:
    build = store._path(".output-build")
    marker = build / f"{command}.transaction.json"
    previous = build / f"{command}.previous"
    if marker.exists() or marker.is_symlink():
        raise AssertionError(f"transaction marker still present: {marker}")
    if previous.exists() or previous.is_symlink():
        raise AssertionError(f"previous tree still present: {previous}")
    for entry in build.iterdir():
        if entry.name.startswith(f"{command}.candidate."):
            raise AssertionError(f"candidate tree still present: {entry}")
        if entry.name.startswith(f"{command}.work."):
            raise AssertionError(f"work tree still present: {entry}")


def _invalid_previous_quarantined(store: RunStore, command: str) -> bool:
    recovery_root = store._path("recovery")
    if not recovery_root.is_dir():
        return False
    label = f"{command}-previous-invalid"
    return any((rec_dir / label).exists() for rec_dir in recovery_root.iterdir())


def _write_invalid_previous(build: Path, *, command: str = "markdown") -> Path:
    previous = build / f"{command}.previous"
    previous.mkdir()
    _write_manifest(
        previous,
        command=command,
        committed_page_count=1,
        primary="book.md",
        primary_bytes=b"# invalid\n",
        sha256_override="0" * 64,
    )
    return previous


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


def test_validate_output_tree_rejects_non_asset_manifest_entry(run_dir: Path) -> None:
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
    manifest_path = candidate / "manifest.json"
    manifest = OutputManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest.files.append(
        OutputFileEntry(path="pandoc-build.json", sha256="0" * 64, size_bytes=1)
    )
    write_json_atomic(manifest_path, manifest.model_dump(mode="json"))
    (candidate / "pandoc-build.json").write_bytes(b"{}")
    with pytest.raises(ProcessingError, match="non-asset path"):
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


@pytest.mark.parametrize(
    "primary_bytes,sha256_override",
    [
        (b"# bad\n", "0" * 64),
        (b"# stale\n", None),
    ],
    ids=["invalid-hash", "stale-head"],
)
def test_invalid_candidate_restores_previous(
    run_dir: Path,
    primary_bytes: bytes,
    sha256_override: str | None,
) -> None:
    store = RunStore(run_dir)
    store.write_head(2)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)

    previous = build / "markdown.previous"
    previous.mkdir()
    _write_manifest(
        previous,
        command="markdown",
        committed_page_count=1,
        primary="book.md",
        primary_bytes=b"# previous\n",
    )

    candidate_name = "markdown.candidate.0123456789abcdef"
    candidate = build / candidate_name
    candidate.mkdir()
    committed = 1 if sha256_override is None else 2
    _write_manifest(
        candidate,
        command="markdown",
        committed_page_count=committed,
        primary="book.md",
        primary_bytes=primary_bytes,
        sha256_override=sha256_override,
    )

    write_json_atomic(
        build / "markdown.transaction.json",
        OutputTransaction(
            output_transaction_format_version=1,
            command="markdown",
            phase="candidate-valid",
            candidate=candidate_name,
            previous="markdown.previous",
        ).model_dump(mode="json"),
    )

    recover_output_transaction(store, "markdown")

    destination = output_destination(store, "markdown")
    assert destination.is_dir()
    assert (destination / "book.md").read_text(encoding="utf-8") == "# previous\n"
    assert not candidate.exists()
    assert not (build / "markdown.transaction.json").exists()


def test_malformed_marker_recovers_without_traceback(run_dir: Path) -> None:
    store = RunStore(run_dir)
    store.write_head(1)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    previous = build / "markdown.previous"
    previous.mkdir()
    _write_manifest(
        previous,
        command="markdown",
        committed_page_count=1,
        primary="book.md",
        primary_bytes=b"# previous\n",
    )
    (build / "markdown.transaction.json").write_text("{not json", encoding="utf-8")

    recover_output_transaction(store, "markdown")

    destination = output_destination(store, "markdown")
    assert destination.is_dir()
    assert (destination / "book.md").read_text(encoding="utf-8") == "# previous\n"


def test_swapped_candidate_basename_quarantines_marker(run_dir: Path) -> None:
    store = RunStore(run_dir)
    store.write_head(1)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    previous = build / "markdown.previous"
    previous.mkdir()
    _write_manifest(
        previous,
        command="markdown",
        committed_page_count=1,
        primary="book.md",
        primary_bytes=b"# previous\n",
    )
    write_json_atomic(
        build / "markdown.transaction.json",
        OutputTransaction(
            output_transaction_format_version=1,
            command="markdown",
            phase="candidate-valid",
            candidate="markdown.previous",
            previous="markdown.previous",
        ).model_dump(mode="json"),
    )

    recover_output_transaction(store, "markdown")

    destination = output_destination(store, "markdown")
    assert (destination / "book.md").read_text(encoding="utf-8") == "# previous\n"
    assert not (build / "markdown.transaction.json").exists()


def test_invalid_previous_not_restored_no_marker(run_dir: Path) -> None:
    store = RunStore(run_dir)
    store.write_head(1)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    _write_invalid_previous(build)

    recover_output_transaction(store, "markdown")

    destination = output_destination(store, "markdown")
    if destination.exists() or destination.is_symlink():
        raise AssertionError("destination should be absent")
    _assert_recovery_artifacts_cleared(store, "markdown")
    if not _invalid_previous_quarantined(store, "markdown"):
        raise AssertionError("invalid previous not quarantined")


def test_invalid_previous_not_restored_after_malformed_marker(run_dir: Path) -> None:
    store = RunStore(run_dir)
    store.write_head(1)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    _write_invalid_previous(build)
    (build / "markdown.transaction.json").write_text("{not json", encoding="utf-8")

    recover_output_transaction(store, "markdown")

    destination = output_destination(store, "markdown")
    if destination.exists() or destination.is_symlink():
        raise AssertionError("destination should be absent")
    _assert_recovery_artifacts_cleared(store, "markdown")
    if not _invalid_previous_quarantined(store, "markdown"):
        raise AssertionError("invalid previous not quarantined")


@pytest.mark.parametrize("phase", ["candidate-valid", "previous-moved"])
def test_invalid_previous_not_restored_active_transaction(
    run_dir: Path,
    phase: str,
) -> None:
    store = RunStore(run_dir)
    store.write_head(1)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    _write_invalid_previous(build)
    write_json_atomic(
        build / "markdown.transaction.json",
        OutputTransaction(
            output_transaction_format_version=1,
            command="markdown",
            phase=phase,  # type: ignore[arg-type]
            candidate="markdown.candidate.0123456789abcdef",
            previous="markdown.previous",
        ).model_dump(mode="json"),
    )

    recover_output_transaction(store, "markdown")

    destination = output_destination(store, "markdown")
    if destination.exists() or destination.is_symlink():
        raise AssertionError("destination should be absent")
    _assert_recovery_artifacts_cleared(store, "markdown")
    if not _invalid_previous_quarantined(store, "markdown"):
        raise AssertionError("invalid previous not quarantined")


@pytest.mark.parametrize(
    "marker_setup",
    [
        "directory",
        "valid-symlink",
        "broken-symlink",
    ],
)
def test_non_regular_transaction_marker_quarantined(
    run_dir: Path,
    tmp_path: Path,
    marker_setup: str,
) -> None:
    store = RunStore(run_dir)
    store.write_head(1)
    build = store._path(".output-build")
    build.mkdir(parents=True, exist_ok=True)
    marker = build / "markdown.transaction.json"

    if marker_setup == "directory":
        marker.mkdir()
    elif marker_setup == "valid-symlink":
        real = tmp_path / "real-marker.json"
        real.write_text("{}", encoding="utf-8")
        marker.symlink_to(real)
    else:
        marker.symlink_to(tmp_path / "missing-marker.json")

    recover_output_transaction(store, "markdown")

    if marker.exists() or marker.is_symlink():
        raise AssertionError("obstructing marker still present")
    candidate_path, _ = begin_output_transaction(store, "markdown")
    if not candidate_path.is_dir():
        raise AssertionError("begin_output_transaction failed after marker quarantine")
