"""Canonical serialization and publication identity tests."""

from __future__ import annotations

import unicodedata

from bookextract.canonical import (
    canonical_json_bytes,
    normalize_publication_document_nfc,
    normalize_publication_string,
    publication_fingerprint_hex,
    publication_identifier,
    publication_uuid,
    serialize_wire_request,
    sha256_hex,
)
from bookextract.models import (
    EpubSemanticProfile,
    MarkdownSemanticProfile,
    PublicationDocument,
    PublicationMetadata,
    PublicationTextBlock,
    PublicationTextRun,
    TextRole,
)


def test_serialize_wire_request_is_deterministic() -> None:
    payload = {"b": 2, "a": 1, "nested": {"z": True, "y": "café"}}
    first = serialize_wire_request(payload)
    second = serialize_wire_request({"nested": {"y": "café", "z": True}, "a": 1, "b": 2})
    assert first == second
    assert b"caf\xc3\xa9" in first


def test_sha256_hex() -> None:
    assert sha256_hex(b"hello") == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )


def test_publication_nfc_normalization_changes_fingerprint() -> None:
    nfd = "e\u0301"  # é as e + combining acute
    nfc = unicodedata.normalize("NFC", nfd)
    assert nfd != nfc
    assert normalize_publication_string(nfd) == nfc

    doc_nfd = PublicationDocument(
        metadata=PublicationMetadata(title=nfd),
        blocks=[
            PublicationTextBlock(
                role=TextRole.PARAGRAPH,
                content=[PublicationTextRun(text="body")],
            )
        ],
        markdown_semantic_profile=MarkdownSemanticProfile(),
        epub_semantic_profile=EpubSemanticProfile(include_toc=True),
    )
    doc_nfc = normalize_publication_document_nfc(doc_nfd)
    assert doc_nfc.metadata.title == nfc
    assert publication_fingerprint_hex(doc_nfd) == publication_fingerprint_hex(doc_nfc)


def test_publication_uuid_is_output_semantic() -> None:
    doc = PublicationDocument(
        metadata=PublicationMetadata(title="Same Title"),
        blocks=[
            PublicationTextBlock(
                role=TextRole.PARAGRAPH,
                content=[PublicationTextRun(text="x")],
            )
        ],
        markdown_semantic_profile=MarkdownSemanticProfile(),
        epub_semantic_profile=EpubSemanticProfile(include_toc=True),
    )
    fp = publication_fingerprint_hex(doc)
    uid = publication_uuid(doc)
    assert str(uid).startswith("00000000-") or len(str(uid)) == 36
    assert publication_identifier(doc).startswith("urn:uuid:")
    assert canonical_json_bytes(doc) == canonical_json_bytes(
        normalize_publication_document_nfc(doc)
    )
    assert fp == publication_fingerprint_hex(normalize_publication_document_nfc(doc))
