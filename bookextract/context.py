"""Page context projection from authoritative book state."""

from __future__ import annotations

from bookextract.models import BookState, PageContext, TocEntry, TocStatus


def build_page_context(state: BookState) -> PageContext:
    expected: list[TocEntry] = []
    if state.toc_status == TocStatus.COMMITTED:
        start = state.next_expected_opening_index
        expected = state.toc[start : start + 3]

    section = state.current_section
    return PageContext(
        book_title=state.metadata.title,
        authors=state.metadata.authors,
        toc_status=state.toc_status,
        expected_toc_entries=expected,
        current_section_title=section.title if section else None,
        current_section_headings=section.recent_headings[-12:] if section else [],
        open_paragraph_tail=section.open_paragraph_tail if section else None,
        previous_page_type=state.previous_page_type,
    )
