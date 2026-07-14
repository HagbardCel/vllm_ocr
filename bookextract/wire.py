"""Flat wire models for VLM structured responses."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from bookextract.models import (
    Alignment,
    ExtractedMetadata,
    PageType,
    RelativeSize,
    StructureKind,
    TextRole,
)


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class BlockKind(StrEnum):
    TEXT = "text"
    FOOTNOTE = "footnote"
    FIGURE = "figure"
    SEPARATOR = "separator"


class InlineKind(StrEnum):
    TEXT = "text"
    FOOTNOTE_REFERENCE = "footnote_reference"


class BBox1000(WireModel):
    left: int = Field(ge=0, le=1000)
    top: int = Field(ge=0, le=1000)
    right: int = Field(ge=0, le=1000)
    bottom: int = Field(ge=0, le=1000)


class VlmInlineSegment(WireModel):
    kind: InlineKind
    value: str


class VlmBlock(WireModel):
    kind: BlockKind
    role: TextRole | None = None
    segments: list[VlmInlineSegment] = Field(default_factory=list)
    label: str | None = None
    caption: str | None = None
    heading_level: int | None = Field(default=None, ge=1, le=6)
    bbox: BBox1000 | None = None
    relative_size: RelativeSize | None = None
    bold: bool | None = None
    italic: bool | None = None
    alignment: Alignment | None = None
    continues_previous: bool = False
    continues_on_next_page: bool = False


class VlmStructuralOpening(WireModel):
    kind: StructureKind
    label: str | None = None
    title: str
    subtitle: str | None = None
    level: int = Field(ge=1, le=6)


class VlmTocEntry(WireModel):
    title: str
    level: int = Field(ge=1, le=6)
    kind: StructureKind
    printed_page_label: str | None = None


class VlmPageResponse(WireModel):
    page_type: PageType
    printed_page_label: str | None = None
    opening: VlmStructuralOpening | None = None
    metadata: ExtractedMetadata | None = None
    toc_entries: list[VlmTocEntry] = Field(default_factory=list)
    blocks: list[VlmBlock] = Field(default_factory=list)
