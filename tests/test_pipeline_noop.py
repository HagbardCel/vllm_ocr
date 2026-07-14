"""Pipeline integration with noop interpreter."""

from __future__ import annotations

from pathlib import Path

from bookextract.artifacts import InterpretationResult
from bookextract.config import ProcessingConfig
from bookextract.models import (
    ExtractedMetadata,
    InterpretationProvenance,
    PageContext,
    PageInput,
    PageInterpretation,
    PageType,
    StructuralOpening,
    StructureKind,
    TextBlock,
    TextRole,
    TextRun,
)
from bookextract.pdf import PdfPageSource
from bookextract.pipeline import process_book
from bookextract.storage import RunStore


class NoopInterpreter:
    def interpret(self, *, page_input: PageInput, context: PageContext) -> InterpretationResult:
        del context
        return InterpretationResult(
            interpretation=PageInterpretation(
                page_type=PageType.BLANK,
                metadata=ExtractedMetadata(),
            ),
            provenance=InterpretationProvenance(
                prompt_version="v1",
                model="test-model",
                backend="noop",
            ),
        )


def test_process_book_noop_single_page(
    run_dir: Path,
    minimal_pdf: Path,
    processing_config: ProcessingConfig,
) -> None:
    store = RunStore(run_dir)
    source = PdfPageSource(minimal_pdf, run_dir / "pages", dpi=72)
    try:
        result = process_book(
            source=source,
            interpreter=NoopInterpreter(),
            store=store,
            config=processing_config,
            max_pages=1,
        )
    finally:
        source.close()

    assert len(result.document.pages) == 1
    assert store.read_head().committed_page_count == 1
    assert (run_dir / "commits" / "page-0001" / "page-assessment.json").is_file()


class ContinuationInterpreter:
    def interpret(self, *, page_input: PageInput, context: PageContext) -> InterpretationResult:
        del context
        return InterpretationResult(
            interpretation=PageInterpretation(
                page_type=PageType.CHAPTER_OPENING,
                opening=StructuralOpening(
                    kind=StructureKind.CHAPTER,
                    title="Chapter 1",
                    level=1,
                ),
                blocks=[
                    TextBlock(
                        role=TextRole.PARAGRAPH,
                        content=[TextRun(text="continued paragraph")],
                        continues_on_next_page=True,
                    )
                ],
            ),
            provenance=InterpretationProvenance(
                prompt_version="v1",
                model="test-model",
                backend="noop",
            ),
        )


def test_partial_run_does_not_validate_complete_book(
    tmp_path: Path,
    multi_page_pdf: Path,
    processing_config: ProcessingConfig,
) -> None:
    from bookextract.config import (
        ProcessOptions,
        RenderContract,
        RunRecord,
        SourceLocation,
        write_json_atomic,
    )
    from bookextract.state import load_or_initialize_state

    run_dir = tmp_path / "partial-run"
    store = RunStore(run_dir)
    store.ensure_layout()
    write_json_atomic(
        run_dir / "run.json",
        RunRecord(
            source={"sha256": "0" * 64, "size": 1, "page_count": 3},
            extraction=processing_config.extraction,
            fingerprint_policy={"require_complete_fingerprint": False},
            process_options=ProcessOptions(),
            render_contract=RenderContract(pymupdf_version="test"),
            prompt_sha256="0" * 64,
            wire_schema_sha256="0" * 64,
            created_at="2026-01-01T00:00:00Z",
        ).model_dump(mode="json"),
    )
    write_json_atomic(
        run_dir / "source-location.json",
        SourceLocation(pdf_path=multi_page_pdf).model_dump(mode="json"),
    )

    source = PdfPageSource(multi_page_pdf, run_dir / "pages", dpi=72)
    try:
        result = process_book(
            source=source,
            interpreter=ContinuationInterpreter(),
            store=store,
            config=processing_config,
            max_pages=1,
        )
        state, _ = load_or_initialize_state(store)
    finally:
        source.close()

    assert len(result.document.pages) == 1
    assert state.current_section is not None
    assert state.current_section.open_paragraph_tail == "continued paragraph"
