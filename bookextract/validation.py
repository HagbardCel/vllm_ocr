"""Deterministic structural validation."""

from __future__ import annotations

import re
import unicodedata

from bookextract.config import ExtractionConfig
from bookextract.errors import StructuralError
from bookextract.models import (
    BookDocument,
    BookState,
    ContentBlock,
    FigureBlock,
    FootnoteBlock,
    PageAssessment,
    PageType,
    TextBlock,
    TextRole,
    TocEntry,
    TocStatus,
)

_LEADER_DOTS_RE = re.compile(r"[.\u00b7·…\s]+$")
_TERMINAL_PUNCT_RE = re.compile(r"[.:;,\-–—]+$")


def normalize_unicode(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def remove_leader_dots(value: str) -> str:
    return _LEADER_DOTS_RE.sub("", value).strip()


def strip_terminal_punctuation(value: str) -> str:
    return _TERMINAL_PUNCT_RE.sub("", value).strip()


def normalize_title(value: str) -> str:
    value = normalize_unicode(value)
    value = value.casefold()
    value = remove_leader_dots(value)
    value = normalize_whitespace(value)
    value = strip_terminal_punctuation(value)
    return value


def _inline_to_text(block: TextBlock | FootnoteBlock) -> str:
    parts: list[str] = []
    for segment in block.content:
        if segment.kind == "text":
            parts.append(segment.text)
    return "".join(parts)


def _chapter_titles(blocks: list[ContentBlock]) -> list[TextBlock]:
    return [
        block
        for block in blocks
        if block.kind == "text" and block.role == TextRole.CHAPTER_TITLE
    ]


def _observed_opening_title(assessment: PageAssessment) -> str:
    opening = assessment.interpretation.opening
    if opening is not None:
        return opening.title
    titles = _chapter_titles(assessment.interpretation.blocks)
    if titles:
        return _inline_to_text(titles[0])
    return ""


def validate_toc_entries(entries: list[TocEntry]) -> None:
    if not entries:
        raise StructuralError(
            code="invalid-toc-hierarchy",
            message="TOC page produced no entries",
        )
    prev_level = 0
    for entry in entries:
        if not entry.title.strip():
            raise StructuralError(
                code="invalid-toc-hierarchy",
                message="TOC entry has empty title",
            )
        if prev_level == 0:
            if entry.level < 1:
                raise StructuralError(
                    code="invalid-toc-hierarchy",
                    message="TOC levels must start at 1",
                )
        elif entry.level > prev_level + 1:
            raise StructuralError(
                code="invalid-toc-hierarchy",
                message=f"TOC level jump from {prev_level} to {entry.level}",
            )
        prev_level = entry.level


def validate_toc_page(assessment: PageAssessment, state: BookState) -> None:
    if state.toc_status in (TocStatus.COMMITTED, TocStatus.ABSENT):
        raise StructuralError(
            code="unexpected-toc",
            message=f"unexpected TOC on page {assessment.page_index}",
        )
    validate_toc_entries(assessment.interpretation.toc_entries)


def validate_chapter_opening(assessment: PageAssessment, state: BookState) -> None:
    observed = _observed_opening_title(assessment)
    titles = _chapter_titles(assessment.interpretation.blocks)
    if not observed and not assessment.interpretation.opening:
        raise StructuralError(
            code="missing-chapter-title",
            message=f"chapter opening without title on page {assessment.page_index}",
        )
    if len(titles) > 1:
        raise StructuralError(
            code="multiple-chapter-titles",
            message=f"multiple chapter titles on page {assessment.page_index}",
        )

    if state.toc_status == TocStatus.COMMITTED:
        if state.next_expected_opening_index >= len(state.toc):
            raise StructuralError(
                code="unexpected-chapter",
                message="chapter found after TOC entries exhausted",
            )
        expected = state.toc[state.next_expected_opening_index]
        observed_norm = normalize_title(observed)
        expected_title = normalize_title(expected.title)
        if observed_norm != expected_title:
            raise StructuralError(
                code="toc-chapter-mismatch",
                message=(
                    f"chapter title mismatch on page {assessment.page_index}: "
                    f"expected {expected_title!r}, observed {observed_norm!r}"
                ),
            )


def validate_headings(blocks: list[ContentBlock], page_index: int) -> None:
    last_level: int | None = None
    for block in blocks:
        if block.kind != "text" or block.role != TextRole.HEADING:
            continue
        if block.heading_level is None:
            continue
        if last_level is not None and block.heading_level > last_level + 1:
            raise StructuralError(
                code="heading-level-jump",
                message=(
                    f"heading level jump on page {page_index}: "
                    f"{last_level} -> {block.heading_level}"
                ),
            )
        last_level = block.heading_level


def validate_continuations(assessment: PageAssessment, state: BookState) -> None:
    interpretation = assessment.interpretation
    if interpretation.page_type == PageType.CHAPTER_OPENING:
        if state.current_section and state.current_section.open_paragraph_tail:
            raise StructuralError(
                code="unclosed-paragraph",
                message="chapter opening with open paragraph from prior page",
            )

    body_paragraphs = [
        block
        for block in interpretation.blocks
        if block.kind == "text" and block.role == TextRole.PARAGRAPH
    ]
    continuing = [block for block in body_paragraphs if block.continues_previous]
    open_tail = (
        state.current_section.open_paragraph_tail
        if state.current_section
        else None
    )

    if open_tail and not continuing:
        raise StructuralError(
            code="missing-continuation",
            message=(
                f"open paragraph tail without continues_previous on page "
                f"{assessment.page_index}"
            ),
        )
    if continuing and not open_tail:
        raise StructuralError(
            code="continuation-without-source",
            message=(
                f"continues_previous without open paragraph on page "
                f"{assessment.page_index}"
            ),
        )
    if len(continuing) > 1:
        raise StructuralError(
            code="invalid-continuation-layout",
            message=f"multiple continues_previous paragraphs on page {assessment.page_index}",
        )
    if continuing and body_paragraphs and continuing[0] is not body_paragraphs[0]:
        raise StructuralError(
            code="invalid-continuation-layout",
            message=(
                f"continues_previous is not the first body paragraph on page "
                f"{assessment.page_index}"
            ),
        )
    closers = [block for block in body_paragraphs if block.continues_on_next_page]
    if len(closers) > 1:
        raise StructuralError(
            code="invalid-continuation-layout",
            message=(
                f"multiple continues_on_next_page paragraphs on page "
                f"{assessment.page_index}"
            ),
        )
    if closers and body_paragraphs and closers[-1] is not body_paragraphs[-1]:
        raise StructuralError(
            code="invalid-continuation-layout",
            message=(
                f"continues_on_next_page is not the last body paragraph on page "
                f"{assessment.page_index}"
            ),
        )


def validate_footnotes(assessment: PageAssessment, config: ExtractionConfig) -> None:
    if not config.require_same_page_footnotes:
        return

    refs: set[str] = set()
    note_labels = [
        block.label
        for block in assessment.interpretation.blocks
        if block.kind == "footnote"
    ]
    if len(note_labels) != len(set(note_labels)):
        raise StructuralError(
            code="duplicate-footnote-label",
            message=f"duplicate footnote labels on page {assessment.page_index}",
        )
    notes: set[str] = set(note_labels)

    for block in assessment.interpretation.blocks:
        if block.kind == "text":
            for segment in block.content:
                if segment.kind == "footnote_reference":
                    refs.add(segment.label)

    for label in refs - notes:
        raise StructuralError(
            code="unresolved-footnote",
            message=(
                f"unresolved footnote reference {label!r} on page "
                f"{assessment.page_index}"
            ),
        )
    for label in notes - refs:
        raise StructuralError(
            code="orphan-footnote",
            message=f"orphan footnote body {label!r} on page {assessment.page_index}",
        )


def validate_figures(assessment: PageAssessment, config: ExtractionConfig) -> None:
    if not config.require_figure_crops:
        return
    page_type = assessment.interpretation.page_type
    for block in assessment.interpretation.blocks:
        if block.kind != "figure":
            continue
        figure: FigureBlock = block
        if figure.bbox is not None:
            continue
        if page_type == PageType.FIGURE_PAGE:
            continue
        raise StructuralError(
            code="figure-crop-unavailable",
            message=f"figure without bbox on page {assessment.page_index}",
        )


def validate_assessment(
    *,
    assessment: PageAssessment,
    state: BookState,
    config: ExtractionConfig,
) -> None:
    interpretation = assessment.interpretation
    page_index = assessment.page_index

    if interpretation.page_type == PageType.TOC:
        validate_toc_page(assessment, state)
    elif interpretation.page_type == PageType.CHAPTER_OPENING:
        validate_chapter_opening(assessment, state)

    validate_headings(interpretation.blocks, page_index)
    validate_continuations(assessment, state)
    validate_footnotes(assessment, config)
    validate_figures(assessment, config)


def validate_complete_book(*, document: BookDocument, state: BookState) -> None:
    if state.toc_status == TocStatus.COMMITTED:
        if state.next_expected_opening_index < len(state.toc):
            raise StructuralError(
                code="toc-not-exhausted",
                message=(
                    f"TOC not exhausted: {state.next_expected_opening_index} of "
                    f"{len(state.toc)} entries matched"
                ),
            )

    if state.current_section and state.current_section.open_paragraph_tail:
        raise StructuralError(
            code="unclosed-paragraph",
            message="document ended with an open paragraph",
        )
