"""Book state reduction and commit replay."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from bookextract.models import (
    BookState,
    ContentBlock,
    CurrentSectionState,
    ExtractedMetadata,
    FootnoteBlock,
    HeadingSummary,
    PageAssessment,
    PageInterpretation,
    PageType,
    StructureKind,
    TextBlock,
    TextRole,
    TocStatus,
)

if TYPE_CHECKING:
    from bookextract.storage import RunStore


def _inline_to_text(block: TextBlock | FootnoteBlock) -> str:
    parts: list[str] = []
    for segment in block.content:
        if segment.kind == "text":
            parts.append(segment.text)
        elif segment.kind == "footnote_reference":
            parts.append(segment.label)
    return "".join(parts)


def _merge_metadata(state: BookState, proposal: ExtractedMetadata | None) -> BookState:
    if proposal is None:
        return state
    metadata = state.metadata.model_copy(deep=True)
    if proposal.title and not metadata.title:
        metadata.title = proposal.title
    if proposal.subtitle and not metadata.subtitle:
        metadata.subtitle = proposal.subtitle
    if proposal.language and not metadata.language:
        metadata.language = proposal.language
    if proposal.authors and not metadata.authors:
        metadata.authors = list(proposal.authors)
    return state.model_copy(update={"metadata": metadata})


def assign_block_ids(interpretation: PageInterpretation, page_index: int) -> PageInterpretation:
    prefix = f"p{page_index + 1:03d}"
    new_blocks: list[ContentBlock] = []
    for idx, block in enumerate(interpretation.blocks, start=1):
        block_id = f"{prefix}-b{idx:03d}"
        new_blocks.append(block.model_copy(update={"block_id": block_id}))
    return interpretation.model_copy(update={"blocks": new_blocks})


def finalize_toc_if_required(state: BookState, assessment: PageAssessment) -> BookState:
    if state.toc_status != TocStatus.COLLECTING:
        return state
    if assessment.interpretation.page_type == PageType.TOC:
        return state
    if not state.toc:
        return state.model_copy(update={"toc_status": TocStatus.ABSENT})
    return state.model_copy(update={"toc_status": TocStatus.COMMITTED})


def _section_title(assessment: PageAssessment) -> str:
    opening = assessment.interpretation.opening
    if opening is not None:
        return opening.title
    for block in assessment.interpretation.blocks:
        if block.kind == "text" and block.role == TextRole.CHAPTER_TITLE:
            return _inline_to_text(block)
    return ""


def _section_kind(assessment: PageAssessment) -> StructureKind:
    opening = assessment.interpretation.opening
    if opening is not None:
        return opening.kind
    return StructureKind.CHAPTER


def apply_assessment(state: BookState, assessment: PageAssessment) -> BookState:
    interpretation = assessment.interpretation
    page_index = assessment.page_index
    new_state = state.model_copy(deep=True)
    new_state = _merge_metadata(new_state, interpretation.metadata)
    new_state.previous_page_type = interpretation.page_type
    new_state.processed_page_count = page_index + 1

    if interpretation.page_type == PageType.TOC:
        if new_state.toc_status in (TocStatus.COMMITTED, TocStatus.ABSENT):
            return new_state
        if new_state.toc_status == TocStatus.NOT_SEEN:
            new_state.toc_status = TocStatus.COLLECTING
        new_state.toc.extend(interpretation.toc_entries)
        return new_state

    if new_state.toc_status == TocStatus.NOT_SEEN and interpretation.page_type in (
        PageType.CHAPTER_OPENING,
        PageType.BODY,
    ):
        new_state.toc_status = TocStatus.ABSENT

    if interpretation.page_type == PageType.CHAPTER_OPENING:
        new_state.current_section = CurrentSectionState(
            kind=_section_kind(assessment),
            title=_section_title(assessment),
            toc_index=new_state.next_expected_opening_index,
            started_on_page=page_index,
        )
        if new_state.toc_status == TocStatus.COMMITTED:
            new_state.next_expected_opening_index += 1

    section = new_state.current_section
    if section is not None:
        headings = list(section.recent_headings)
        heading_stack = list(section.heading_stack)
        open_tail: str | None = section.open_paragraph_tail
        for block in interpretation.blocks:
            if block.kind != "text":
                continue
            if block.role == TextRole.HEADING and block.heading_level is not None:
                summary = HeadingSummary(
                    level=block.heading_level,
                    text=_inline_to_text(block),
                    page_index=page_index,
                )
                headings.append(summary)
                heading_stack.append(summary)
            if block.role == TextRole.PARAGRAPH:
                text = _inline_to_text(block)
                if block.continues_on_next_page:
                    open_tail = text[-500:] if len(text) > 500 else text
                elif not block.continues_previous:
                    open_tail = None
        new_state.current_section = section.model_copy(
            update={
                "recent_headings": headings[-12:],
                "heading_stack": heading_stack,
                "open_paragraph_tail": open_tail,
            }
        )

    return new_state


def load_book_state_from_commits(store: RunStore, committed_page_count: int) -> BookState:
    state = BookState()
    for page_number in range(1, committed_page_count + 1):
        assessment_bytes = store.read_commit_file(page_number, "page-assessment.json")
        assessment = PageAssessment.model_validate_json(assessment_bytes.decode("utf-8"))
        state = finalize_toc_if_required(state, assessment)
        state = apply_assessment(state, assessment)
    return state


def load_or_initialize_state(store: RunStore) -> tuple[BookState, int]:
    cache = store.read_state_cache()
    head = store.read_head()
    if cache is not None and cache.committed_page_count == head.committed_page_count:
        return cache.state, head.committed_page_count
    if head.committed_page_count == 0:
        return BookState(), 0
    state = load_book_state_from_commits(store, head.committed_page_count)
    store.write_state_cache(state, head.committed_page_count)
    return state, head.committed_page_count


def build_commit_payload(
    *,
    assessment: PageAssessment,
    context_json: dict[str, object],
    interpretation_json: dict[str, object],
    provenance_json: dict[str, object],
    extra_files: dict[str, bytes] | None = None,
) -> dict[str, bytes]:
    files: dict[str, bytes] = {
        "page-assessment.json": assessment.model_dump_json(indent=2).encode("utf-8")
        + b"\n",
        "page-context.json": (json.dumps(context_json, indent=2) + "\n").encode("utf-8"),
        "interpretation.json": (json.dumps(interpretation_json, indent=2) + "\n").encode(
            "utf-8"
        ),
        "provenance.json": (json.dumps(provenance_json, indent=2) + "\n").encode("utf-8"),
    }
    if extra_files:
        files.update(extra_files)
    return files
