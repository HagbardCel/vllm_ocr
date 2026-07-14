"""Wire schema loading tests."""

from __future__ import annotations

from bookextract.schema import (
    WIRE_SCHEMA_VERSION,
    build_wire_response_format,
    load_wire_schema,
    normalize_llama_schema,
)


def test_load_wire_schema() -> None:
    schema = load_wire_schema(WIRE_SCHEMA_VERSION)
    assert schema["type"] == "object"
    assert "properties" in schema


def test_normalize_llama_schema_strips_metadata() -> None:
    raw = {
        "type": "object",
        "title": "strip me",
        "properties": {"x": {"type": "string", "description": "also strip"}},
    }
    normalized = normalize_llama_schema(raw)
    assert "title" not in normalized
    assert "description" not in normalized["properties"]["x"]
    assert normalized["additionalProperties"] is False


def test_build_wire_response_format() -> None:
    fmt = build_wire_response_format()
    assert fmt["type"] == "json_schema"
    assert "schema" in fmt
