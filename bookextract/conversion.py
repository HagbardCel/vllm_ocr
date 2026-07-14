"""Convert flat VLM wire responses into domain PageInterpretation."""

from __future__ import annotations

from bookextract.models import (
    Alignment,
    BoundingBox,
    ContentBlock,
    ExtractedMetadata,
    FigureBlock,
    FootnoteBlock,
    FootnoteReference,
    InlineContent,
    PageInterpretation,
    RelativeSize,
    SeparatorBlock,
    StructuralOpening,
    TextBlock,
    TextRun,
    TextStyle,
    TocEntry,
)
from bookextract.wire import (
    BBox1000,
    BlockKind,
    InlineKind,
    VlmBlock,
    VlmInlineSegment,
    VlmPageResponse,
    VlmStructuralOpening,
    VlmTocEntry,
)


def bbox1000_to_normalized(bbox: BBox1000) -> BoundingBox:
    return BoundingBox(
        x0=bbox.left / 1000.0,
        y0=bbox.top / 1000.0,
        x1=bbox.right / 1000.0,
        y1=bbox.bottom / 1000.0,
    )


def _wire_style_to_domain(
    *,
    relative_size: RelativeSize | None,
    bold: bool | None,
    italic: bool | None,
    alignment: Alignment | None,
) -> TextStyle | None:
    if (
        relative_size is None
        and bold is None
        and italic is None
        and alignment is None
    ):
        return None
    return TextStyle(
        relative_size=relative_size,
        bold=bold,
        italic=italic,
        alignment=alignment,
    )


def _convert_inline_segment(segment: VlmInlineSegment) -> InlineContent:
    if segment.kind == InlineKind.TEXT:
        return TextRun(text=segment.value)
    return FootnoteReference(label=segment.value)


def _convert_inline_segments(segments: list[VlmInlineSegment]) -> list[InlineContent]:
    return [_convert_inline_segment(segment) for segment in segments]


def _convert_block(block: VlmBlock) -> ContentBlock:
    style = _wire_style_to_domain(
        relative_size=block.relative_size,
        bold=block.bold,
        italic=block.italic,
        alignment=block.alignment,
    )
    bbox = bbox1000_to_normalized(block.bbox) if block.bbox is not None else None

    if block.kind == BlockKind.TEXT:
        if block.role is None:
            raise ValueError("text block requires role")
        return TextBlock(
            role=block.role,
            content=_convert_inline_segments(block.segments),
            heading_level=block.heading_level,
            continues_previous=block.continues_previous,
            continues_on_next_page=block.continues_on_next_page,
            bbox=bbox,
            style=style,
        )

    if block.kind == BlockKind.FOOTNOTE:
        if block.label is None:
            raise ValueError("footnote block requires label")
        return FootnoteBlock(
            label=block.label,
            content=_convert_inline_segments(block.segments),
            bbox=bbox,
            style=style,
        )

    if block.kind == BlockKind.FIGURE:
        return FigureBlock(
            label=block.label,
            caption=block.caption,
            bbox=bbox,
        )

    if block.kind == BlockKind.SEPARATOR:
        return SeparatorBlock()

    raise ValueError(f"unsupported block kind: {block.kind}")


def _convert_opening(opening: VlmStructuralOpening) -> StructuralOpening:
    return StructuralOpening(
        kind=opening.kind,
        label=opening.label,
        title=opening.title,
        subtitle=opening.subtitle,
        level=opening.level,
    )


def _convert_toc_entry(entry: VlmTocEntry) -> TocEntry:
    return TocEntry(
        title=entry.title,
        level=entry.level,
        kind=entry.kind,
        printed_page_label=entry.printed_page_label,
    )


def _convert_metadata(metadata: ExtractedMetadata | None) -> ExtractedMetadata | None:
    if metadata is None:
        return None
    return ExtractedMetadata.model_validate(metadata.model_dump(mode="json"))


def convert_vlm_response(response: VlmPageResponse) -> PageInterpretation:
    return PageInterpretation(
        page_type=response.page_type,
        printed_page_label=response.printed_page_label,
        opening=_convert_opening(response.opening) if response.opening is not None else None,
        metadata=_convert_metadata(response.metadata),
        toc_entries=[_convert_toc_entry(entry) for entry in response.toc_entries],
        blocks=[_convert_block(block) for block in response.blocks],
        warnings=[],
    )
