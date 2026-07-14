"""Markdown rendering from publication projection."""

from __future__ import annotations

from bookextract.canonical import normalize_publication_document_nfc
from bookextract.config import MarkdownRenderConfig
from bookextract.models import (
    BookDocument,
    PublicationDocument,
    PublicationTextBlock,
    TextRole,
)
from bookextract.rendering.publication import build_publication_document


def _publication_text(block: PublicationTextBlock) -> str:
    parts: list[str] = []
    for segment in block.content:
        if segment.kind == "text":
            parts.append(segment.text)
        elif segment.kind == "footnote_reference":
            parts.append(f"[^{segment.label}]")
    return "".join(parts)


class MarkdownRenderer:
    def __init__(self, config: MarkdownRenderConfig | None = None) -> None:
        self._config = config or MarkdownRenderConfig()

    def render_publication(self, document: PublicationDocument) -> str:
        normalized = normalize_publication_document_nfc(document)
        lines: list[str] = []
        metadata = normalized.metadata
        if metadata.title:
            lines.extend([f"# {metadata.title}", ""])
        if metadata.subtitle:
            lines.extend([f"## {metadata.subtitle}", ""])
        if metadata.authors:
            lines.extend([", ".join(metadata.authors), ""])

        footnote_defs: list[tuple[str, str]] = []
        for block in normalized.blocks:
            if block.kind == "text":
                lines.extend(self._render_text_block(block))
            elif block.kind == "footnote":
                label = block.note_id
                text = _publication_text(
                    PublicationTextBlock(
                        role=TextRole.PARAGRAPH,
                        content=block.content,
                    )
                )
                footnote_defs.append((label, text))
            elif block.kind == "figure":
                caption = block.caption or ""
                asset = block.asset_sha256[:12]
                lines.extend([f"![{caption}](assets/{asset}.png)", ""])
            elif block.kind == "separator":
                lines.extend(["---", ""])

        for label, text in footnote_defs:
            lines.append(f"[^{label}]: {text}")
        return "\n".join(lines).rstrip() + "\n"

    def _render_text_block(self, block: PublicationTextBlock) -> list[str]:
        text = _publication_text(block)
        if block.role == TextRole.CHAPTER_TITLE:
            return [f"# {text}", ""]
        if block.role == TextRole.HEADING and block.heading_level:
            level = min(block.heading_level + 1, 6)
            return [f"{'#' * level} {text}", ""]
        if block.role == TextRole.BLOCKQUOTE:
            return [f"> {text}", ""]
        if block.role in (TextRole.BOOK_TITLE, TextRole.SUBTITLE):
            return [f"# {text}", ""]
        return [text, ""]

    def render_book(
        self,
        document: BookDocument,
        *,
        epub_include_toc: bool = True,
        page_markers: bool | None = None,
    ) -> str:
        from bookextract.config import EpubRenderConfig

        pub_doc, _ = build_publication_document(
            document,
            markdown_config=self._config,
            epub_config=EpubRenderConfig(include_toc=epub_include_toc),
        )
        markdown = self.render_publication(pub_doc)
        if page_markers if page_markers is not None else self._config.include_page_markers:
            if document.pages:
                lines = markdown.splitlines()
                lines.insert(
                    0, f"<!-- source-page: {document.pages[0].page_index + 1} -->"
                )
                return "\n".join(lines) + "\n"
        return markdown
