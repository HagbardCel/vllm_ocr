"""Wire schema loading and llama.cpp response_format construction."""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

WIRE_SCHEMA_VERSION = "vlm-page-response-v1"

_STRIP_SCHEMA_KEYS = frozenset(
    {
        "title",
        "description",
        "default",
        "examples",
        "$schema",
        "$id",
        "$comment",
        "deprecated",
    }
)


def _schema_text(version: str = WIRE_SCHEMA_VERSION) -> str:
    filename = f"{version}.json"
    return resources.files("bookextract.schemas").joinpath(filename).read_text(encoding="utf-8")


def load_wire_schema(version: str = WIRE_SCHEMA_VERSION) -> dict[str, object]:
    loaded = json.loads(_schema_text(version))
    if not isinstance(loaded, dict):
        raise ValueError(f"wire schema must be a JSON object: {version}")
    return loaded


def normalize_llama_schema(schema: dict[str, Any]) -> dict[str, object]:
    """Project-specific cleanup for llama.cpp JSON schema / grammar conversion."""

    def walk(node: Any) -> Any:
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, dict):
            return node

        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _STRIP_SCHEMA_KEYS:
                continue
            if key == "$defs":
                cleaned[key] = {name: walk(defn) for name, defn in value.items()}
                continue
            cleaned[key] = walk(value)

        if cleaned.get("type") == "object" and "additionalProperties" not in cleaned:
            cleaned["additionalProperties"] = False

        return cleaned

    normalized = walk(schema)
    if not isinstance(normalized, dict):
        raise ValueError("normalized schema must be a JSON object")
    return normalized


def build_response_format(schema: dict[str, object]) -> dict[str, object]:
    return {
        "type": "json_schema",
        "schema": schema,
    }


def build_wire_response_format(version: str = WIRE_SCHEMA_VERSION) -> dict[str, object]:
    return build_response_format(load_wire_schema(version))
