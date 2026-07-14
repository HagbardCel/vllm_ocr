"""Wire model and conversion tests."""

from __future__ import annotations

import pytest

from bookextract.conversion import bbox1000_to_normalized, convert_vlm_response
from bookextract.models import PageType, TextRole
from bookextract.wire import (
    BBox1000,
    BlockKind,
    InlineKind,
    VlmBlock,
    VlmInlineSegment,
    VlmPageResponse,
    VlmTocEntry,
)


def test_bbox1000_to_normalized() -> None:
    bbox = bbox1000_to_normalized(BBox1000(left=100, top=200, right=900, bottom=800))
    assert bbox.x0 == pytest.approx(0.1)
    assert bbox.y0 == pytest.approx(0.2)
    assert bbox.x1 == pytest.approx(0.9)
    assert bbox.y1 == pytest.approx(0.8)


def test_convert_vlm_response_text_block() -> None:
    response = VlmPageResponse(
        page_type=PageType.BODY,
        blocks=[
            VlmBlock(
                kind=BlockKind.TEXT,
                role=TextRole.PARAGRAPH,
                segments=[
                    VlmInlineSegment(kind=InlineKind.TEXT, value="Hello "),
                    VlmInlineSegment(kind=InlineKind.FOOTNOTE_REFERENCE, value="1"),
                ],
            )
        ],
    )
    interpretation = convert_vlm_response(response)
    assert interpretation.page_type == PageType.BODY
    assert len(interpretation.blocks) == 1
    block = interpretation.blocks[0]
    assert block.kind == "text"
    assert block.content[0].kind == "text"
    assert block.content[1].kind == "footnote_reference"


def test_convert_vlm_toc_entries() -> None:
    from bookextract.models import StructureKind

    response = VlmPageResponse(
        page_type=PageType.TOC,
        toc_entries=[
            VlmTocEntry(title="Chapter 1", level=1, kind=StructureKind.CHAPTER),
            VlmTocEntry(title="Section 1.1", level=2, kind=StructureKind.SECTION),
        ],
    )
    interpretation = convert_vlm_response(response)
    assert len(interpretation.toc_entries) == 2
    assert interpretation.toc_entries[1].level == 2
