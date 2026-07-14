"""VLM page prompt construction."""

from __future__ import annotations

import json

from bookextract.models import PageContext, TocStatus


class PagePromptBuilder:
    PROMPT_VERSION = "page-v1"

    def build(self, context: PageContext) -> str:
        sections = [
            self._role_section(),
            self._context_section(context),
            self._classification_section(),
            self._block_rules_section(),
            self._footnote_section(),
            self._figure_section(),
            self._continuation_section(),
            self._output_section(),
        ]
        return "\n\n".join(sections)

    def _role_section(self) -> str:
        return (
            "You are a faithful book page transcription assistant.\n"
            "Transcribe visible text exactly. Do not modernize spelling or grammar.\n"
            "Do not silently complete damaged text.\n"
            "Return structured JSON matching the provided schema."
        )

    def _context_section(self, context: PageContext) -> str:
        payload = {
            "book_title": context.book_title,
            "authors": context.authors,
            "toc_status": context.toc_status.value,
            "expected_toc_entries": [
                entry.model_dump(mode="json") for entry in context.expected_toc_entries
            ],
            "current_section_title": context.current_section_title,
            "current_section_headings": [
                h.model_dump(mode="json") for h in context.current_section_headings
            ],
            "open_paragraph_tail": context.open_paragraph_tail,
            "previous_page_type": (
                context.previous_page_type.value if context.previous_page_type else None
            ),
        }
        return (
            "Authoritative book context (use for disambiguation only; "
            "report what is visibly present on the current page):\n"
            + json.dumps(payload, indent=2, ensure_ascii=False)
        )

    def _classification_section(self) -> str:
        return (
            "Classify the page type among: cover, title, copyright, dedication, toc, "
            "preface, chapter_opening, body, figure_page, appendix, bibliography, "
            "index, blank, other.\n"
            "chapter_opening begins a new chapter. toc contains table-of-contents entries."
        )

    def _block_rules_section(self) -> str:
        return (
            "Return blocks in reading order.\n"
            "Use text blocks with roles (chapter_title, heading, paragraph, etc.).\n"
            "Level 1 headings are chapter titles; level 2 is the highest in-chapter heading.\n"
            "Exclude running headers, footers, and page numbers from paragraph content."
        )

    def _footnote_section(self) -> str:
        return (
            "Represent inline footnote markers as footnote_reference segments.\n"
            "Represent note bodies as footnote blocks with matching labels."
        )

    def _figure_section(self) -> str:
        return (
            "Return figure blocks for illustrations with optional normalized bounding boxes "
            "(0-1 coordinates). Omit captions when none are printed."
        )

    def _continuation_section(self) -> str:
        return (
            "Mark continues_previous when the first body paragraph continues the prior page.\n"
            "Mark continues_on_next_page when a paragraph continues onto the next page."
        )

    def _output_section(self) -> str:
        return (
            "Do not generate PDF page indices, asset paths, model metadata, or timestamps.\n"
            "On TOC pages, populate toc_entries. On other pages, leave toc_entries empty."
        )


def prompt_sha256(builder: PagePromptBuilder | None = None) -> str:
    import hashlib

    builder = builder or PagePromptBuilder()
    sample = builder.build(
        PageContext(toc_status=TocStatus.NOT_SEEN),
    )
    return hashlib.sha256(sample.encode("utf-8")).hexdigest()
