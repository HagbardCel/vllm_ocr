"""Domain models for bookextract v0.1."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

DOMAIN_SCHEMA_VERSION = 1

PAGE_OPENING_KINDS = frozenset(
    {
        "part",
        "chapter",
        "preface",
        "appendix",
        "bibliography",
        "index",
    }
)

INLINE_STRUCTURE_KINDS = frozenset(
    {
        "section",
        "subsection",
    }
)


class DomainModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )


class PageType(StrEnum):
    COVER = "cover"
    TITLE = "title"
    COPYRIGHT = "copyright"
    DEDICATION = "dedication"
    TOC = "toc"
    PREFACE = "preface"
    CHAPTER_OPENING = "chapter_opening"
    BODY = "body"
    FIGURE_PAGE = "figure_page"
    APPENDIX = "appendix"
    BIBLIOGRAPHY = "bibliography"
    INDEX = "index"
    BLANK = "blank"
    OTHER = "other"


class TextRole(StrEnum):
    BOOK_TITLE = "book_title"
    SUBTITLE = "subtitle"
    AUTHOR = "author"
    CHAPTER_TITLE = "chapter_title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    BLOCKQUOTE = "blockquote"
    EPIGRAPH = "epigraph"
    LIST_ITEM = "list_item"
    CAPTION = "caption"
    RUNNING_HEADER = "running_header"
    RUNNING_FOOTER = "running_footer"
    PAGE_NUMBER = "page_number"
    OTHER = "other"


class RelativeSize(StrEnum):
    SMALL = "small"
    BODY = "body"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"


class Alignment(StrEnum):
    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"
    JUSTIFIED = "justified"
    UNKNOWN = "unknown"


class StructureKind(StrEnum):
    PART = "part"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    PREFACE = "preface"
    APPENDIX = "appendix"
    BIBLIOGRAPHY = "bibliography"
    INDEX = "index"
    OTHER = "other"


class TocStatus(StrEnum):
    NOT_SEEN = "not_seen"
    COLLECTING = "collecting"
    COMMITTED = "committed"
    ABSENT = "absent"


class SeparatorKind(StrEnum):
    HORIZONTAL_RULE = "horizontal_rule"


class BoundingBox(DomainModel):
    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)


class RectPoints(DomainModel):
    x0: float
    y0: float
    x1: float
    y1: float


class TextStyle(DomainModel):
    font_family: str | None = None
    font_size_pt: float | None = None
    relative_size: RelativeSize | None = None
    bold: bool | None = None
    italic: bool | None = None
    small_caps: bool | None = None
    alignment: Alignment | None = None


class TextRun(DomainModel):
    kind: Literal["text"] = "text"
    text: str


class FootnoteReference(DomainModel):
    kind: Literal["footnote_reference"] = "footnote_reference"
    label: str


InlineContent = Annotated[
    TextRun | FootnoteReference,
    Field(discriminator="kind"),
]


class TextBlock(DomainModel):
    kind: Literal["text"] = "text"
    role: TextRole
    content: list[InlineContent]
    heading_level: int | None = Field(default=None, ge=1, le=6)
    continues_previous: bool = False
    continues_on_next_page: bool = False
    bbox: BoundingBox | None = None
    style: TextStyle | None = None
    block_id: str | None = None


class FootnoteBlock(DomainModel):
    kind: Literal["footnote"] = "footnote"
    label: str
    content: list[InlineContent]
    bbox: BoundingBox | None = None
    style: TextStyle | None = None
    block_id: str | None = None


class FigureBlock(DomainModel):
    kind: Literal["figure"] = "figure"
    label: str | None = None
    caption: str | None = None
    bbox: BoundingBox | None = None
    asset_path: Path | None = None
    asset_sha256: str | None = None
    block_id: str | None = None


class SeparatorBlock(DomainModel):
    kind: Literal["separator"] = "separator"
    block_id: str | None = None


ContentBlock = Annotated[
    TextBlock | FootnoteBlock | FigureBlock | SeparatorBlock,
    Field(discriminator="kind"),
]


class TocEntry(DomainModel):
    title: str
    level: int = Field(ge=1, le=6)
    kind: StructureKind
    printed_page_label: str | None = None


class ExtractedMetadata(DomainModel):
    title: str | None = None
    subtitle: str | None = None
    authors: list[str] = Field(default_factory=list)
    language: str | None = None


class StructuralOpening(DomainModel):
    kind: StructureKind
    label: str | None = None
    title: str
    subtitle: str | None = None
    level: int = Field(ge=1, le=6)


class PageInterpretation(DomainModel):
    page_type: PageType
    printed_page_label: str | None = None
    opening: StructuralOpening | None = None
    metadata: ExtractedMetadata | None = None
    toc_entries: list[TocEntry] = Field(default_factory=list)
    blocks: list[ContentBlock] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ModelRunInfo(DomainModel):
    backend: str
    model: str
    prompt_version: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: float | None = None


class InterpretationProvenance(DomainModel):
    backend: str
    model: str
    prompt_version: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    elapsed_ms: float | None = None
    finish_reason: str | None = None
    attempts: int = 1
    raw_response_sha256: str | None = None


class PageAssessment(DomainModel):
    page_index: int
    image_path: Path
    interpretation: PageInterpretation
    model_run: ModelRunInfo


class HeadingSummary(DomainModel):
    level: int
    text: str
    page_index: int


class CurrentSectionState(DomainModel):
    kind: StructureKind
    title: str
    toc_index: int | None = None
    started_on_page: int
    heading_stack: list[HeadingSummary] = Field(default_factory=list)
    recent_headings: list[HeadingSummary] = Field(default_factory=list)
    open_paragraph_tail: str | None = None


class BookState(DomainModel):
    metadata: ExtractedMetadata = Field(default_factory=ExtractedMetadata)
    toc_status: TocStatus = TocStatus.NOT_SEEN
    toc: list[TocEntry] = Field(default_factory=list)
    next_expected_opening_index: int = 0
    current_section: CurrentSectionState | None = None
    previous_page_type: PageType | None = None
    processed_page_count: int = 0


class PageContext(DomainModel):
    book_title: str | None = None
    authors: list[str] = Field(default_factory=list)
    toc_status: TocStatus
    expected_toc_entries: list[TocEntry] = Field(default_factory=list)
    current_section_title: str | None = None
    current_section_headings: list[HeadingSummary] = Field(default_factory=list)
    open_paragraph_tail: str | None = None
    previous_page_type: PageType | None = None


class RenderedPage(DomainModel):
    image_path: Path
    width_px: int
    height_px: int
    page_rect: RectPoints
    rotation_degrees: int
    image_sha256: str
    image_size_bytes: int


MAX_PAGE_IMAGE_BYTES = 40 * 1024 * 1024


class PageInput(DomainModel):
    page_index: int
    rendered: RenderedPage

    @property
    def image_path(self) -> Path:
        return self.rendered.image_path


class BookDocument(DomainModel):
    pages: list[PageAssessment] = Field(default_factory=list)


class HeadState(DomainModel):
    head_format_version: int = 1
    committed_page_count: int = 0


class StateCache(DomainModel):
    state_cache_version: int = 1
    committed_page_count: int = 0
    state: BookState


class CommitManifest(DomainModel):
    manifest_format_version: int = 1
    page_index: int
    files: dict[str, str]


class SourceMapEntry(DomainModel):
    publication_block_index: int
    page_index: int
    block_id: str | None = None


class OutputManifest(DomainModel):
    output_manifest_format_version: int = 1
    publication_identifier: str
    publication_fingerprint: str
    source_map: list[SourceMapEntry]


class MultimodalInputTokensContract(DomainModel):
    mode: Literal["chat-input-tokens-multimodal"]
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str


class TextOnlyInputTokensContract(DomainModel):
    mode: Literal["chat-input-tokens-text-only"]
    image_token_policy: Literal["configured-reserve"]
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str


class ApplyTemplateTokenizeContract(DomainModel):
    mode: Literal["apply-template-tokenize"]
    image_token_policy: Literal["configured-reserve"]
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str


class EstimateOnlyContract(DomainModel):
    mode: Literal["estimate-only"]
    estimation_version: int = 1
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str


TokenCountingMode = Literal[
    "chat-input-tokens-multimodal",
    "chat-input-tokens-text-only",
    "apply-template-tokenize",
    "estimate-only",
]

TokenCountingContract = Annotated[
    MultimodalInputTokensContract
    | TextOnlyInputTokensContract
    | ApplyTemplateTokenizeContract
    | EstimateOnlyContract,
    Field(discriminator="mode"),
]


class ThinkingControlContract(DomainModel):
    contract_format_version: int = 1
    enable_thinking: Literal[False] = False
    reasoning_format: Literal["none", "deepseek", "deepseek-legacy"]
    applied_template_probe_supported: bool
    reasoning_content_expected: Literal[False] = False
    model_alias: str
    llama_cpp_build: str
    chat_template_sha256: str


class ContextBudgetResult(DomainModel):
    counted_input_tokens: int
    image_tokens_reserved: int
    output_tokens_reserved: int
    safety_margin_tokens: int
    context_size: int
    counting_mode: TokenCountingMode
    exact_for_projected_input: bool
    multimodal_count_included: bool

    @property
    def required_tokens(self) -> int:
        return (
            self.counted_input_tokens
            + self.image_tokens_reserved
            + self.output_tokens_reserved
            + self.safety_margin_tokens
        )


class FileFingerprint(DomainModel):
    path: str
    size: int
    mtime_ns: int
    sha256: str | None = None


class ServerInferenceIdentity(DomainModel):
    llama_cpp_build: str
    model_alias: str
    context_size: int
    vision_supported: bool
    chat_template_sha256: str
    server_reported_model_path: str


class ServerInvocationCapabilities(DomainModel):
    media_marker: str | None = None
    chat_template_caps: dict[str, object] = Field(default_factory=dict)


class InferenceEnvironment(DomainModel):
    inference_environment_format_version: int = 1
    server: ServerInferenceIdentity
    model_file: FileFingerprint | None = None
    projector_file: FileFingerprint | None = None
    model_binding_verified: bool
    projector_binding: Literal["server-verified", "operator-asserted", "unavailable"]
    fingerprints_complete: bool = False
    token_counting_contract: TokenCountingContract
    thinking_control_contract: ThinkingControlContract


class PublicationMetadata(DomainModel):
    title: str | None = None
    subtitle: str | None = None
    authors: list[str] = Field(default_factory=list)
    language: str | None = None


class PublicationTextRun(DomainModel):
    kind: Literal["text"] = "text"
    text: str


class PublicationFootnoteReference(DomainModel):
    kind: Literal["footnote_reference"] = "footnote_reference"
    label: str


PublicationInlineContent = Annotated[
    PublicationTextRun | PublicationFootnoteReference,
    Field(discriminator="kind"),
]


class PublicationTextBlock(DomainModel):
    kind: Literal["text"] = "text"
    role: TextRole
    content: list[PublicationInlineContent]
    heading_level: int | None = Field(default=None, ge=1, le=6)


class PublicationFootnoteBlock(DomainModel):
    kind: Literal["footnote"] = "footnote"
    note_id: str
    content: list[PublicationInlineContent]


class PublicationFigureBlock(DomainModel):
    kind: Literal["figure"] = "figure"
    caption: str | None = None
    asset_sha256: str


class PublicationSeparatorBlock(DomainModel):
    kind: Literal["separator"] = "separator"
    separator_kind: SeparatorKind = SeparatorKind.HORIZONTAL_RULE


PublicationBlock = Annotated[
    PublicationTextBlock
    | PublicationFootnoteBlock
    | PublicationFigureBlock
    | PublicationSeparatorBlock,
    Field(discriminator="kind"),
]


class MarkdownSemanticProfile(DomainModel):
    profile_format_version: int = 1
    footnote_style: Literal["document-end"] = "document-end"
    heading_style: Literal["atx"] = "atx"


class EpubSemanticProfile(DomainModel):
    profile_format_version: int = 1
    from_format: Literal["markdown+footnotes"] = "markdown+footnotes"
    to_format: Literal["epub3"] = "epub3"
    split_level: Literal[1] = 1
    include_toc: bool


class PublicationDocument(DomainModel):
    identity_format_version: int = 1
    metadata: PublicationMetadata
    blocks: list[PublicationBlock]
    markdown_semantic_profile: MarkdownSemanticProfile
    epub_semantic_profile: EpubSemanticProfile


class RenderedPublicationBlock(DomainModel):
    publication_block_id: str
    content: PublicationBlock
