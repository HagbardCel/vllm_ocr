"""Publication document projection and identity."""

from __future__ import annotations

from bookextract.canonical import (
    normalize_publication_document_nfc,
    normalize_publication_string,
    publication_fingerprint_hex,
    publication_identifier,
    publication_uuid,
)
from bookextract.config import EpubRenderConfig, MarkdownRenderConfig
from bookextract.models import (
    BookDocument,
    EpubSemanticProfile,
    InlineContent,
    MarkdownSemanticProfile,
    PublicationBlock,
    PublicationDocument,
    PublicationFigureBlock,
    PublicationFootnoteBlock,
    PublicationFootnoteReference,
    PublicationInlineContent,
    PublicationMetadata,
    PublicationSeparatorBlock,
    PublicationTextBlock,
    PublicationTextRun,
    SeparatorKind,
    SourceMapEntry,
    TextBlock,
    TextRole,
)

__all__ = [
    "build_publication_document",
    "normalize_publication_document_nfc",
    "normalize_publication_string",
    "publication_fingerprint_hex",
    "publication_identifier",
    "publication_uuid",
]


def _inline_to_publication(content: list[InlineContent]) -> list[PublicationInlineContent]:
    segments: list[PublicationInlineContent] = []
    for segment in content:
        if segment.kind == "text":
            segments.append(PublicationTextRun(text=segment.text))
        elif segment.kind == "footnote_reference":
            segments.append(PublicationFootnoteReference(label=segment.label))
    return segments


def build_publication_document(
    document: BookDocument,
    *,
    markdown_config: MarkdownRenderConfig,
    epub_config: EpubRenderConfig,
) -> tuple[PublicationDocument, list[SourceMapEntry]]:
    del markdown_config  # profile is fixed in v0.1
    metadata = PublicationMetadata()
    for page in document.pages:
        if page.interpretation.metadata:
            meta = page.interpretation.metadata
            if meta.title and not metadata.title:
                metadata.title = meta.title
            if meta.subtitle and not metadata.subtitle:
                metadata.subtitle = meta.subtitle
            if meta.language and not metadata.language:
                metadata.language = meta.language
            if meta.authors and not metadata.authors:
                metadata.authors = list(meta.authors)

    blocks: list[PublicationBlock] = []
    source_map: list[SourceMapEntry] = []
    footnote_order = 0
    deferred_footnotes: list[PublicationFootnoteBlock] = []

    for page in document.pages:
        page_index = page.page_index
        for block in page.interpretation.blocks:
            if block.kind == "text":
                text_block: TextBlock = block
                if text_block.role in (
                    TextRole.RUNNING_HEADER,
                    TextRole.RUNNING_FOOTER,
                    TextRole.PAGE_NUMBER,
                ):
                    continue
                pub = PublicationTextBlock(
                    role=text_block.role,
                    content=_inline_to_publication(text_block.content),
                    heading_level=text_block.heading_level,
                )
                blocks.append(pub)
                source_map.append(
                    SourceMapEntry(
                        publication_block_index=len(blocks) - 1,
                        page_index=page_index,
                        block_id=text_block.block_id,
                    )
                )
            elif block.kind == "footnote":
                footnote_order += 1
                deferred_footnotes.append(
                    PublicationFootnoteBlock(
                        note_id=str(footnote_order),
                        content=_inline_to_publication(block.content),
                    )
                )
            elif block.kind == "figure":
                if not block.asset_sha256:
                    continue
                fig = PublicationFigureBlock(
                    caption=block.caption,
                    asset_sha256=block.asset_sha256,
                )
                blocks.append(fig)
                source_map.append(
                    SourceMapEntry(
                        publication_block_index=len(blocks) - 1,
                        page_index=page_index,
                        block_id=block.block_id,
                    )
                )
            elif block.kind == "separator":
                blocks.append(
                    PublicationSeparatorBlock(separator_kind=SeparatorKind.HORIZONTAL_RULE)
                )
                source_map.append(
                    SourceMapEntry(
                        publication_block_index=len(blocks) - 1,
                        page_index=page_index,
                        block_id=block.block_id,
                    )
                )

    for footnote in deferred_footnotes:
        blocks.append(footnote)
        source_map.append(
            SourceMapEntry(
                publication_block_index=len(blocks) - 1,
                page_index=-1,
                block_id=None,
            )
        )

    pub_doc = PublicationDocument(
        metadata=metadata,
        blocks=blocks,
        markdown_semantic_profile=MarkdownSemanticProfile(),
        epub_semantic_profile=EpubSemanticProfile(include_toc=epub_config.include_toc),
    )
    return pub_doc, source_map
