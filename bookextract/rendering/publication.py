"""Publication document projection and identity."""

from __future__ import annotations

from dataclasses import dataclass

from bookextract.canonical import (
    normalize_publication_document_nfc,
    normalize_publication_string,
    publication_fingerprint_hex,
    publication_identifier,
    publication_uuid,
)
from bookextract.config import EpubRenderConfig, MarkdownRenderConfig
from bookextract.errors import StructuralError
from bookextract.models import (
    BookDocument,
    EpubSemanticProfile,
    FootnoteBlock,
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

CLOSING_PUNCTUATION = frozenset(".,;:!?)]}»\"'")


def _inline_to_publication(content: list[InlineContent]) -> list[PublicationInlineContent]:
    segments: list[PublicationInlineContent] = []
    for segment in content:
        if segment.kind == "text":
            segments.append(PublicationTextRun(text=segment.text))
        elif segment.kind == "footnote_reference":
            segments.append(PublicationFootnoteReference(label=segment.label))
    return segments


def _rewrite_footnote_refs(
    content: list[PublicationInlineContent],
    *,
    page_index: int,
    label_map: dict[tuple[int, str], str],
) -> list[PublicationInlineContent]:
    rewritten: list[PublicationInlineContent] = []
    for segment in content:
        if segment.kind == "footnote_reference":
            key = (page_index, segment.label)
            if key not in label_map:
                raise StructuralError(
                    code="unresolved-footnote",
                    message=f"unresolved footnote reference {segment.label!r} on page {page_index}",
                )
            rewritten.append(PublicationFootnoteReference(label=label_map[key]))
        else:
            rewritten.append(segment)
    return rewritten


def _join_boundary_text(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return left[:-1] + right
    if right[0] in CLOSING_PUNCTUATION:
        return left + right
    if left[-1].isspace() or right[0].isspace():
        return left + right
    return left + " " + right


def _merge_publication_content(
    previous: list[PublicationInlineContent],
    current: list[PublicationInlineContent],
) -> list[PublicationInlineContent]:
    if not previous:
        return list(current)
    if not current:
        return list(previous)

    merged = list(previous)
    last = merged[-1]
    first = current[0]
    if last.kind == "text" and first.kind == "text":
        merged[-1] = PublicationTextRun(text=_join_boundary_text(last.text, first.text))
        merged.extend(current[1:])
        return merged
    merged.extend(current)
    return merged


@dataclass
class _RawBlock:
    block: PublicationBlock
    page_index: int
    block_id: str | None
    continues_previous: bool = False
    continues_on_next_page: bool = False


def _collect_footnote_label_map(document: BookDocument) -> dict[tuple[int, str], str]:
    label_map: dict[tuple[int, str], str] = {}
    next_note_id = 0
    for page in document.pages:
        page_index = page.page_index
        bodies = [
            block
            for block in page.interpretation.blocks
            if block.kind == "footnote"
        ]
        labels = [block.label for block in bodies]
        if len(labels) != len(set(labels)):
            raise StructuralError(
                code="duplicate-footnote-label",
                message=f"duplicate footnote labels on page {page_index}",
            )
        for body in bodies:
            next_note_id += 1
            label_map[(page_index, body.label)] = str(next_note_id)
    return label_map


def _collect_raw_blocks(
    document: BookDocument,
    label_map: dict[tuple[int, str], str],
) -> tuple[list[_RawBlock], list[_RawBlock]]:
    raw_blocks: list[_RawBlock] = []
    deferred_footnotes: list[_RawBlock] = []

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
                if text_block.continues_previous or text_block.continues_on_next_page:
                    if text_block.role != TextRole.PARAGRAPH:
                        raise StructuralError(
                            code="invalid-continuation-layout",
                            message=(
                                f"continuation flags on non-paragraph block "
                                f"on page {page_index}"
                            ),
                        )
                content = _rewrite_footnote_refs(
                    _inline_to_publication(text_block.content),
                    page_index=page_index,
                    label_map=label_map,
                )
                pub = PublicationTextBlock(
                    role=text_block.role,
                    content=content,
                    heading_level=text_block.heading_level,
                )
                raw_blocks.append(
                    _RawBlock(
                        block=pub,
                        page_index=page_index,
                        block_id=text_block.block_id,
                        continues_previous=text_block.continues_previous,
                        continues_on_next_page=text_block.continues_on_next_page,
                    )
                )
            elif block.kind == "footnote":
                footnote: FootnoteBlock = block
                note_id = label_map[(page_index, footnote.label)]
                content = _rewrite_footnote_refs(
                    _inline_to_publication(footnote.content),
                    page_index=page_index,
                    label_map=label_map,
                )
                deferred_footnotes.append(
                    _RawBlock(
                        block=PublicationFootnoteBlock(note_id=note_id, content=content),
                        page_index=page_index,
                        block_id=footnote.block_id,
                    )
                )
            elif block.kind == "figure":
                if not block.asset_sha256:
                    continue
                fig = PublicationFigureBlock(
                    caption=block.caption,
                    asset_sha256=block.asset_sha256,
                )
                raw_blocks.append(
                    _RawBlock(
                        block=fig,
                        page_index=page_index,
                        block_id=block.block_id,
                    )
                )
            elif block.kind == "separator":
                raw_blocks.append(
                    _RawBlock(
                        block=PublicationSeparatorBlock(
                            separator_kind=SeparatorKind.HORIZONTAL_RULE
                        ),
                        page_index=page_index,
                        block_id=block.block_id,
                    )
                )

    return raw_blocks, deferred_footnotes


def _merge_continuations(
    raw_blocks: list[_RawBlock],
) -> tuple[list[_RawBlock], list[list[tuple[int, str | None]]]]:
    output: list[_RawBlock] = []
    provenance: list[list[tuple[int, str | None]]] = []
    open_index: int | None = None
    open_page_index: int | None = None
    intervening = False

    for item in raw_blocks:
        if item.block.kind == "text" and item.continues_previous:
            if open_index is None or open_page_index is None:
                raise StructuralError(
                    code="continuation-without-source",
                    message=(
                        f"continues_previous without open paragraph on page "
                        f"{item.page_index}"
                    ),
                )
            if intervening:
                raise StructuralError(
                    code="invalid-continuation-layout",
                    message=(
                        "intervening publication block between continuation pages "
                        f"on page {item.page_index}"
                    ),
                )
            open_item = output[open_index]
            if open_item.block.kind != "text":
                raise StructuralError(
                    code="invalid-continuation-layout",
                    message="continuation target is not a text block",
                )
            if open_item.block.role != TextRole.PARAGRAPH:
                raise StructuralError(
                    code="invalid-continuation-layout",
                    message="continuation source is not a paragraph",
                )
            if item.block.role != TextRole.PARAGRAPH:
                raise StructuralError(
                    code="invalid-continuation-layout",
                    message="continuation block is not a paragraph",
                )
            if item.page_index != open_page_index + 1:
                raise StructuralError(
                    code="invalid-continuation-layout",
                    message=(
                        f"continues_previous on page {item.page_index} "
                        f"without adjacent source page {open_page_index}"
                    ),
                )
            merged_content = _merge_publication_content(
                open_item.block.content,
                item.block.content,
            )
            output[open_index] = _RawBlock(
                block=open_item.block.model_copy(update={"content": merged_content}),
                page_index=open_item.page_index,
                block_id=open_item.block_id,
                continues_previous=False,
                continues_on_next_page=item.continues_on_next_page,
            )
            provenance[open_index].append((item.page_index, item.block_id))
            if item.continues_on_next_page:
                open_page_index = item.page_index
                intervening = False
            else:
                open_index = None
                open_page_index = None
                intervening = False
            continue

        output.append(item)
        provenance.append([(item.page_index, item.block_id)])
        if item.block.kind == "text" and item.continues_on_next_page:
            open_index = len(output) - 1
            open_page_index = item.page_index
            intervening = False
        elif open_index is not None:
            intervening = True

    return output, provenance


def _dedupe_provenance(
    entries: list[tuple[int, str | None]],
) -> tuple[list[int], list[str]]:
    pages: list[int] = []
    block_ids: list[str] = []
    seen_pages: set[int] = set()
    seen_blocks: set[str] = set()
    for page_index, block_id in entries:
        if page_index not in seen_pages:
            pages.append(page_index)
            seen_pages.add(page_index)
        if block_id is not None and block_id not in seen_blocks:
            block_ids.append(block_id)
            seen_blocks.add(block_id)
    return pages, block_ids


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

    label_map = _collect_footnote_label_map(document)
    raw_blocks, deferred_footnotes = _collect_raw_blocks(document, label_map)
    merged_blocks, provenance = _merge_continuations(raw_blocks)

    blocks: list[PublicationBlock] = [item.block for item in merged_blocks]
    source_map: list[SourceMapEntry] = []
    for index, prov in enumerate(provenance):
        pages, block_ids = _dedupe_provenance(prov)
        source_map.append(
            SourceMapEntry(
                publication_block_index=index,
                source_pages=pages,
                source_block_ids=block_ids,
            )
        )

    for footnote in deferred_footnotes:
        blocks.append(footnote.block)
        pages, block_ids = _dedupe_provenance([(footnote.page_index, footnote.block_id)])
        source_map.append(
            SourceMapEntry(
                publication_block_index=len(blocks) - 1,
                source_pages=pages,
                source_block_ids=block_ids,
            )
        )

    pub_doc = PublicationDocument(
        metadata=metadata,
        blocks=blocks,
        markdown_semantic_profile=MarkdownSemanticProfile(),
        epub_semantic_profile=EpubSemanticProfile(include_toc=epub_config.include_toc),
    )
    return pub_doc, source_map
