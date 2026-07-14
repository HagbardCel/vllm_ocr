"""Canonical serialization, normalization, and hashing helpers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from bookextract.models import PublicationDocument

NAMESPACE_URL = uuid.UUID("6ba7b811-9dad-11d1-80b4-00c04fd430c8")


def canonical_json_bytes(value: BaseModel) -> bytes:
    data = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def serialize_wire_request(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_publication_string(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def _normalize_string_fields(value: object) -> object:
    if isinstance(value, str):
        return normalize_publication_string(value)
    if isinstance(value, list):
        return [_normalize_string_fields(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_string_fields(item) for key, item in value.items()}
    return value


def normalize_publication_document_nfc(doc: PublicationDocument) -> PublicationDocument:
    normalized = _normalize_string_fields(doc.model_dump(mode="json"))
    return PublicationDocument.model_validate(normalized)


def publication_fingerprint_hex(doc: PublicationDocument) -> str:
    normalized = normalize_publication_document_nfc(doc)
    return sha256_hex(canonical_json_bytes(normalized))


def publication_uuid(doc: PublicationDocument) -> uuid.UUID:
    fingerprint = publication_fingerprint_hex(doc)
    return uuid.uuid5(NAMESPACE_URL, f"bookextract:{fingerprint}")


def publication_identifier(doc: PublicationDocument) -> str:
    return f"urn:uuid:{publication_uuid(doc)}"
