"""Markdown rendering and footnote ordering tests."""

from __future__ import annotations

from bookextract.config import EpubRenderConfig, MarkdownRenderConfig
from bookextract.models import (
    BookDocument,
    FootnoteBlock,
    FootnoteReference,
    PageInterpretation,
    PageType,
    TextBlock,
    TextRole,
    TextRun,
)
from bookextract.rendering.markdown import MarkdownRenderer
from bookextract.rendering.publication import build_publication_document
from tests.conftest import make_assessment


def test_footnotes_deferred_to_document_end() -> None:
    document = BookDocument(
        pages=[
            make_assessment(
                0,
                PageInterpretation(
                    page_type=PageType.BODY,
                    blocks=[
                        TextBlock(
                            role=TextRole.PARAGRAPH,
                            content=[
                                TextRun(text="See note"),
                                FootnoteReference(label="1"),
                                TextRun(text="."),
                            ],
                        ),
                        FootnoteBlock(label="1", content=[TextRun(text="Footnote body.")]),
                    ],
                ),
            )
        ]
    )
    pub_doc, _ = build_publication_document(
        document,
        markdown_config=MarkdownRenderConfig(),
        epub_config=EpubRenderConfig(),
    )
    footnote_blocks = [b for b in pub_doc.blocks if b.kind == "footnote"]
    text_blocks = [b for b in pub_doc.blocks if b.kind == "text"]
    assert len(text_blocks) == 1
    assert len(footnote_blocks) == 1
    assert pub_doc.blocks.index(footnote_blocks[0]) > pub_doc.blocks.index(text_blocks[0])

    markdown = MarkdownRenderer().render_publication(pub_doc)
    assert "[^1]" in markdown
    assert "[^1]: Footnote body." in markdown
    assert markdown.index("[^1]") < markdown.index("[^1]: Footnote body.")
