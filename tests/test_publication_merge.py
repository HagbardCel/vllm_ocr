"""Publication merge and footnote relabelling tests."""

from __future__ import annotations

import pytest

from bookextract.config import EpubRenderConfig, MarkdownRenderConfig
from bookextract.errors import StructuralError
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
from bookextract.rendering.publication import build_publication_document
from tests.conftest import make_assessment


def test_continuation_merge_across_pages() -> None:
    document = BookDocument(
        pages=[
            make_assessment(
                0,
                PageInterpretation(
                    page_type=PageType.BODY,
                    blocks=[
                        TextBlock(
                            role=TextRole.PARAGRAPH,
                            content=[TextRun(text="Hello ")],
                            continues_on_next_page=True,
                            block_id="p001-b001",
                        )
                    ],
                ),
            ),
            make_assessment(
                1,
                PageInterpretation(
                    page_type=PageType.BODY,
                    blocks=[
                        TextBlock(
                            role=TextRole.PARAGRAPH,
                            content=[TextRun(text="world.")],
                            continues_previous=True,
                            block_id="p002-b001",
                        )
                    ],
                ),
            ),
        ]
    )
    pub_doc, source_map = build_publication_document(
        document,
        markdown_config=MarkdownRenderConfig(),
        epub_config=EpubRenderConfig(),
    )
    text_blocks = [b for b in pub_doc.blocks if b.kind == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0].content[0].text == "Hello world."
    assert source_map[0].source_pages == [0, 1]
    assert source_map[0].source_block_ids == ["p001-b001", "p002-b001"]


def test_heading_between_continuation_pages_raises() -> None:
    document = BookDocument(
        pages=[
            make_assessment(
                0,
                PageInterpretation(
                    page_type=PageType.BODY,
                    blocks=[
                        TextBlock(
                            role=TextRole.PARAGRAPH,
                            content=[TextRun(text="Open")],
                            continues_on_next_page=True,
                        )
                    ],
                ),
            ),
            make_assessment(
                1,
                PageInterpretation(
                    page_type=PageType.BODY,
                    blocks=[
                        TextBlock(
                            role=TextRole.HEADING,
                            heading_level=2,
                            content=[TextRun(text="Break")],
                        ),
                        TextBlock(
                            role=TextRole.PARAGRAPH,
                            content=[TextRun(text="tail")],
                            continues_previous=True,
                        ),
                    ],
                ),
            ),
        ]
    )
    with pytest.raises(StructuralError, match="intervening publication block"):
        build_publication_document(
            document,
            markdown_config=MarkdownRenderConfig(),
            epub_config=EpubRenderConfig(),
        )


def test_footnote_relabel_is_page_scoped() -> None:
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
                                TextRun(text="a"),
                                FootnoteReference(label="1"),
                            ],
                        ),
                        FootnoteBlock(label="1", content=[TextRun(text="note-a")]),
                    ],
                ),
            ),
            make_assessment(
                1,
                PageInterpretation(
                    page_type=PageType.BODY,
                    blocks=[
                        TextBlock(
                            role=TextRole.PARAGRAPH,
                            content=[
                                TextRun(text="b"),
                                FootnoteReference(label="1"),
                            ],
                        ),
                        FootnoteBlock(label="1", content=[TextRun(text="note-b")]),
                    ],
                ),
            ),
        ]
    )
    pub_doc, _ = build_publication_document(
        document,
        markdown_config=MarkdownRenderConfig(),
        epub_config=EpubRenderConfig(),
    )
    footnotes = [b for b in pub_doc.blocks if b.kind == "footnote"]
    assert [b.note_id for b in footnotes] == ["1", "2"]
