"""Structural validation tests."""

from __future__ import annotations

import pytest

from bookextract.errors import StructuralError
from bookextract.models import (
    BookDocument,
    BookState,
    FootnoteBlock,
    FootnoteReference,
    PageInterpretation,
    PageType,
    StructuralOpening,
    StructureKind,
    TextBlock,
    TextRole,
    TextRun,
    TocEntry,
    TocStatus,
)
from bookextract.validation import (
    normalize_title,
    validate_assessment,
    validate_complete_book,
    validate_toc_entries,
)
from tests.conftest import make_assessment


def test_normalize_title_strips_leaders_and_punctuation() -> None:
    assert normalize_title("Chapter One ......") == "chapter one"
    assert normalize_title("Part II:") == "part ii"


def test_validate_toc_entries_rejects_level_jump() -> None:
    entries = [
        TocEntry(title="A", level=1, kind=StructureKind.CHAPTER),
        TocEntry(title="B", level=3, kind=StructureKind.SECTION),
    ]
    with pytest.raises(StructuralError, match="level jump"):
        validate_toc_entries(entries)


def test_validate_chapter_opening_matches_toc(extraction_config) -> None:
    state = BookState(
        toc_status=TocStatus.COMMITTED,
        toc=[TocEntry(title="Chapter One", level=1, kind=StructureKind.CHAPTER)],
        next_expected_opening_index=0,
    )
    assessment = make_assessment(
        2,
        PageInterpretation(
            page_type=PageType.CHAPTER_OPENING,
            opening=StructuralOpening(
                kind=StructureKind.CHAPTER,
                title="Chapter One",
                level=1,
            ),
            blocks=[],
        ),
    )
    validate_assessment(assessment=assessment, state=state, config=extraction_config)


def test_validate_chapter_opening_mismatch(extraction_config) -> None:
    state = BookState(
        toc_status=TocStatus.COMMITTED,
        toc=[TocEntry(title="Expected", level=1, kind=StructureKind.CHAPTER)],
        next_expected_opening_index=0,
    )
    assessment = make_assessment(
        2,
        PageInterpretation(
            page_type=PageType.CHAPTER_OPENING,
            opening=StructuralOpening(
                kind=StructureKind.CHAPTER,
                title="Different",
                level=1,
            ),
            blocks=[],
        ),
    )
    with pytest.raises(StructuralError, match="chapter title mismatch"):
        validate_assessment(assessment=assessment, state=state, config=extraction_config)


def test_validate_same_page_footnotes(extraction_config) -> None:
    extraction_config.require_same_page_footnotes = True
    assessment = make_assessment(
        0,
        PageInterpretation(
            page_type=PageType.BODY,
            blocks=[
                TextBlock(
                    role=TextRole.PARAGRAPH,
                    content=[FootnoteReference(label="1")],
                ),
            ],
        ),
    )
    with pytest.raises(StructuralError, match="unresolved footnote"):
        validate_assessment(
            assessment=assessment,
            state=BookState(),
            config=extraction_config,
        )

    assessment.interpretation.blocks.append(
        FootnoteBlock(label="1", content=[TextRun(text="Note.")])
    )
    validate_assessment(
        assessment=assessment,
        state=BookState(),
        config=extraction_config,
    )


def test_validate_complete_book_toc_not_exhausted() -> None:
    state = BookState(
        toc_status=TocStatus.COMMITTED,
        toc=[TocEntry(title="A", level=1, kind=StructureKind.CHAPTER)],
        next_expected_opening_index=0,
    )
    with pytest.raises(StructuralError, match="TOC not exhausted"):
        validate_complete_book(document=BookDocument(), state=state)
