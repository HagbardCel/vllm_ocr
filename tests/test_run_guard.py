"""Run guard consistency tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
import pytest

from bookextract.config import (
    EpubRenderConfig,
    ExtractionConfig,
    MarkdownRenderConfig,
    ProcessOptions,
    RenderContract,
    RunRecord,
    SourceLocation,
    write_json_atomic,
)
from bookextract.errors import ProcessingError
from bookextract.interpretation.prompts import prompt_sha256
from bookextract.run_guard import assert_process_consistency, assert_render_consistency
from bookextract.schema import load_wire_schema
from bookextract.storage import RunStore


def _wire_schema_sha256() -> str:
    schema = load_wire_schema()
    return hashlib.sha256(
        __import__("json").dumps(schema, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _make_record(minimal_pdf: Path) -> RunRecord:
    sha256 = hashlib.sha256(minimal_pdf.read_bytes()).hexdigest()
    size_bytes = minimal_pdf.stat().st_size
    with fitz.open(minimal_pdf) as document:
        page_count = len(document)
    return RunRecord(
        run_format_version=1,
        source={"sha256": sha256, "size_bytes": size_bytes, "page_count": page_count},
        extraction=ExtractionConfig(model_alias="test-model", prompt_version="v1"),
        fingerprint_policy={"require_complete_fingerprint": False},
        process_options=ProcessOptions(),
        markdown=MarkdownRenderConfig(),
        epub=EpubRenderConfig(),
        render_contract=RenderContract(
            render_contract_format_version=1,
            pymupdf_version=fitz.__version__,
        ),
        prompt_sha256=prompt_sha256(),
        wire_schema_sha256=_wire_schema_sha256(),
        created_at="2026-01-01T00:00:00Z",
    )


def test_process_guard_detects_pymupdf_drift(run_dir: Path, minimal_pdf: Path) -> None:
    store = RunStore(run_dir)
    record = _make_record(minimal_pdf)
    record.render_contract.pymupdf_version = "0.0.0"
    write_json_atomic(run_dir / "run.json", record.model_dump(mode="json"))
    write_json_atomic(
        run_dir / "source-location.json",
        SourceLocation(source_location_format_version=1, pdf_path=minimal_pdf).model_dump(
            mode="json"
        ),
    )

    with pytest.raises(ProcessingError, match="pymupdf version drift"):
        assert_process_consistency(store, record, require_inference_location=False)


def test_process_guard_validates_source_identity(run_dir: Path, minimal_pdf: Path) -> None:
    store = RunStore(run_dir)
    record = _make_record(minimal_pdf)
    record.source["sha256"] = "0" * 64
    write_json_atomic(run_dir / "run.json", record.model_dump(mode="json"))
    write_json_atomic(
        run_dir / "source-location.json",
        SourceLocation(source_location_format_version=1, pdf_path=minimal_pdf).model_dump(
            mode="json"
        ),
    )

    with pytest.raises(ProcessingError, match="hash mismatch"):
        assert_process_consistency(store, record, require_inference_location=False)


def test_render_guard_epub_validates_pandoc_defaults(run_dir: Path, minimal_pdf: Path) -> None:
    store = RunStore(run_dir)
    record = _make_record(minimal_pdf)
    assert_render_consistency(store, record, "epub")


def test_render_guard_markdown_skips_pandoc(run_dir: Path, minimal_pdf: Path) -> None:
    store = RunStore(run_dir)
    record = _make_record(minimal_pdf)
    assert_render_consistency(store, record, "markdown")
