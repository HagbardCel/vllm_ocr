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
