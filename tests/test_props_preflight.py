"""Unit tests for /props parsing and strict vision acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bookextract.config import ExtractionConfig, ProcessingConfig, ProcessOptions
from bookextract.errors import ProcessingError
from bookextract.inference.llamacpp import (
    LlamaCppVisionClient,
    LlamaPropsResponse,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_props_parses_build_info_and_top_level_caps() -> None:
    props = LlamaPropsResponse.model_validate_json(_load_fixture("llama_props_vision.json"))
    assert props.build_info == "b5123-abc123def456"
    assert props.chat_template_caps is not None
    assert props.chat_template_caps.get("apply_template") is True
    assert props.media_marker == "<__media__>"


def test_llama_cpp_build_prefers_build_info_string() -> None:
    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="m", prompt_version="v1")
    )
    client = LlamaCppVisionClient(config)
    props = LlamaPropsResponse.model_validate_json(_load_fixture("llama_props_vision.json"))
    assert client._llama_cpp_build_from_props(props) == "b5123-abc123def456"


def test_llama_cpp_build_blank_build_info_falls_back_to_legacy() -> None:
    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="m", prompt_version="v1")
    )
    client = LlamaCppVisionClient(config)
    props = LlamaPropsResponse.model_validate(
        {
            "model_path": "/models/m.gguf",
            "modalities": {"vision": True},
            "build_info": "   ",
            "build": "legacy-commit",
        }
    )
    assert client._llama_cpp_build_from_props(props) == "legacy-commit"


def test_require_vision_rejects_missing_modalities() -> None:
    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="m", prompt_version="v1")
    )
    client = LlamaCppVisionClient(config)
    props = LlamaPropsResponse.model_validate_json(
        _load_fixture("llama_props_no_modalities.json")
    )
    with pytest.raises(ProcessingError, match="missing modalities"):
        client._require_vision_modalities(props)


def test_require_vision_rejects_false() -> None:
    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="m", prompt_version="v1")
    )
    client = LlamaCppVisionClient(config)
    props = LlamaPropsResponse.model_validate_json(
        _load_fixture("llama_props_vision_false.json")
    )
    with pytest.raises(ProcessingError, match="modalities.vision is false"):
        client._require_vision_modalities(props)


def test_preflight_rejects_non_vision_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"ok")
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                content=json.dumps({"data": [{"id": "vision-model"}]}).encode(),
            )
        if request.url.path == "/props":
            return httpx.Response(200, content=_load_fixture("llama_props_vision_false.json"))
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(
        base_url="http://test",
        transport=transport,
    )
    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
        process=ProcessOptions(llama_base_url="http://test"),
    )
    client = LlamaCppVisionClient(config, client=http_client)
    with pytest.raises(ProcessingError, match="modalities.vision is false"):
        client.preflight()


def test_preflight_accepts_vision_server() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b"ok")
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                content=json.dumps({"data": [{"id": "vision-model"}]}).encode(),
            )
        if request.url.path == "/props":
            return httpx.Response(200, content=_load_fixture("llama_props_vision.json"))
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(
        base_url="http://test",
        transport=transport,
    )
    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
        process=ProcessOptions(llama_base_url="http://test"),
    )
    client = LlamaCppVisionClient(config, client=http_client)
    preflight = client.preflight()
    assert preflight.identity.llama_cpp_build == "b5123-abc123def456"
    assert preflight.identity.vision_supported is True
    assert preflight.capabilities.media_marker == "<__media__>"
