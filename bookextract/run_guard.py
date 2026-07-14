"""Run consistency guards for process and render commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import fitz

from bookextract.config import RunRecord
from bookextract.errors import ProcessingError
from bookextract.interpretation.prompts import prompt_sha256
from bookextract.models import BookDocument, PageAssessment
from bookextract.schema import load_wire_schema
from bookextract.storage import RunStore

_PYMUPDF_OPEN_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    fitz.FileDataError,
    fitz.EmptyFileError,
)


def _wire_schema_sha256() -> str:
    schema = load_wire_schema()
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _inspect_source_pdf(path: Path) -> tuple[str, int, int]:
    try:
        sha256, size_bytes = _hash_file(path)
        with fitz.open(path) as document:
            page_count = len(document)
    except _PYMUPDF_OPEN_ERRORS as exc:
        raise ProcessingError(
            code="invalid-source-pdf",
            message=f"cannot open source PDF: {path}",
        ) from exc

    if page_count < 1:
        raise ProcessingError(
            code="invalid-source-pdf",
            message="source PDF contains no pages",
        )
    return sha256, size_bytes, page_count


def load_document_from_commits(store: RunStore) -> BookDocument:
    head = store.read_head()
    document = BookDocument()
    for page_number in range(1, head.committed_page_count + 1):
        raw = store.read_commit_file(page_number, "page-assessment.json")
        document.pages.append(PageAssessment.model_validate_json(raw.decode("utf-8")))
    return document


def _validate_live_source(record: RunRecord, pdf_path: Path) -> None:
    actual_sha, actual_size, actual_pages = _inspect_source_pdf(pdf_path)

    expected_sha = record.source.get("sha256")
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ProcessingError(code="source-hash-mismatch", message="source PDF hash mismatch")

    expected_size = record.source.get("size_bytes")
    if not isinstance(expected_size, int) or actual_size != expected_size:
        raise ProcessingError(
            code="source-hash-mismatch",
            message="source PDF size_bytes mismatch",
        )

    expected_pages = record.source.get("page_count")
    if not isinstance(expected_pages, int) or actual_pages != expected_pages:
        raise ProcessingError(
            code="source-hash-mismatch",
            message="source PDF page_count mismatch",
        )


def assert_process_consistency(
    store: RunStore,
    record: RunRecord,
    *,
    require_inference_location: bool,
) -> None:
    if record.run_format_version != 1:
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"unsupported run_format_version: {record.run_format_version}",
        )
    if record.render_contract.render_contract_format_version != 1:
        raise ProcessingError(
            code="invalid-run-layout",
            message="unsupported render_contract_format_version",
        )
    if record.prompt_sha256 != prompt_sha256():
        raise ProcessingError(code="config-drift", message="prompt contract drift")
    if record.wire_schema_sha256 != _wire_schema_sha256():
        raise ProcessingError(code="schema-drift", message="wire schema drift")

    if record.render_contract.pymupdf_version != fitz.__version__:
        raise ProcessingError(
            code="render-environment-drift",
            message="pymupdf version drift",
        )

    source_loc = store.load_source_location()
    if source_loc.source_location_format_version != 1:
        raise ProcessingError(
            code="invalid-run-layout",
            message="unsupported source_location_format_version",
        )
    if not source_loc.pdf_path.is_file():
        raise ProcessingError(
            code="invalid-source-pdf",
            message=f"source PDF not found: {source_loc.pdf_path}",
        )
    _validate_live_source(record, source_loc.pdf_path)

    if require_inference_location:
        location = store.load_inference_location()
        if location.inference_location_format_version != 1:
            raise ProcessingError(
                code="invalid-run-layout",
                message="unsupported inference_location_format_version",
            )


def assert_render_consistency(
    store: RunStore,
    record: RunRecord,
    command: Literal["markdown", "epub"],
) -> None:
    if record.run_format_version != 1:
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"unsupported run_format_version: {record.run_format_version}",
        )
    store.load_source_location()
    if command == "epub":
        from bookextract.rendering.epub import EpubRenderer

        EpubRenderer()._load_base_defaults()
