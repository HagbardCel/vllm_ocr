"""Run consistency guards for process and render commands."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from bookextract.config import RunRecord
from bookextract.errors import ProcessingError
from bookextract.interpretation.prompts import prompt_sha256
from bookextract.models import BookDocument, PageAssessment
from bookextract.schema import load_wire_schema
from bookextract.storage import RunStore, validate_commit


def _wire_schema_sha256() -> str:
    schema = load_wire_schema()
    return hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_document_from_commits(store: RunStore) -> BookDocument:
    head = store.read_head()
    document = BookDocument()
    for page_number in range(1, head.committed_page_count + 1):
        raw = store.read_commit_file(page_number, "page-assessment.json")
        document.pages.append(PageAssessment.model_validate_json(raw.decode("utf-8")))
    return document


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
    actual_sha = _hash_file(source_loc.pdf_path)
    expected_sha = record.source.get("sha256")
    if not isinstance(expected_sha, str) or actual_sha != expected_sha:
        raise ProcessingError(code="source-hash-mismatch", message="source PDF hash mismatch")

    if require_inference_location:
        location = store.load_inference_location()
        if location.inference_location_format_version != 1:
            raise ProcessingError(
                code="invalid-run-layout",
                message="unsupported inference_location_format_version",
            )


def _validate_pandoc_defaults() -> None:
    from bookextract.rendering.epub import EpubRenderer

    EpubRenderer()._load_base_defaults()


def assert_render_consistency(
    store: RunStore,
    record: RunRecord,
    command: Literal["markdown", "epub"],
) -> None:
    del command
    if record.run_format_version != 1:
        raise ProcessingError(
            code="invalid-run-layout",
            message=f"unsupported run_format_version: {record.run_format_version}",
        )
    store.load_source_location()
    head = store.read_head()
    for page_number in range(1, head.committed_page_count + 1):
        validate_commit(store.commit_dir_for(page_number))
    load_document_from_commits(store)
    _validate_pandoc_defaults()
