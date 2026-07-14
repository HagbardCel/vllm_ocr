"""Apply-template tokenize discovery and frozen-contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from bookextract.config import ExtractionConfig, ProcessingConfig, ProcessOptions
from bookextract.errors import ProcessingError
from bookextract.inference.llamacpp import LlamaCppVisionClient, PreflightResult
from bookextract.models import (
    ApplyTemplateTokenizeContract,
    ServerInferenceIdentity,
    ServerInvocationCapabilities,
    ThinkingControlContract,
)


def _preflight(*, reasoning: bool) -> PreflightResult:
    caps: dict[str, object] = {"apply_template": True}
    if reasoning:
        caps["reasoning"] = True
    identity = ServerInferenceIdentity(
        llama_cpp_build="b1",
        model_alias="vision-model",
        context_size=32768,
        vision_supported=True,
        chat_template_sha256="c" * 64,
        server_reported_model_path="/models/m.gguf",
    )
    return PreflightResult(
        identity=identity,
        capabilities=ServerInvocationCapabilities(
            media_marker="<__media__>",
            chat_template_caps=caps,
        ),
        raw_health=b"ok",
        raw_models=b"{}",
        raw_props=b"{}",
    )


def _thinking_contract() -> ThinkingControlContract:
    return ThinkingControlContract(
        reasoning_format="none",
        applied_template_probe_supported=False,
        model_alias="vision-model",
        llama_cpp_build="b1",
        chat_template_sha256="c" * 64,
    )


def _calibration_image(tmp_path: Path) -> Path:
    image = tmp_path / "img.png"
    image.write_bytes(b"png")
    return image


def test_discover_apply_template_messages_only(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/apply-template":
            assert "image_url" not in request.content.decode()
            assert "data:" not in request.content.decode()
            return httpx.Response(200, content=json.dumps("prompt-text").encode())
        if request.url.path == "/tokenize":
            assert body == {
                "content": "prompt-text",
                "add_special": False,
                "parse_special": True,
            }
            return httpx.Response(200, content=json.dumps({"tokens": [1, 2, 3]}).encode())
        return httpx.Response(404)

    client = LlamaCppVisionClient(
        ProcessingConfig(
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            process=ProcessOptions(llama_base_url="http://test"),
        ),
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    contract = client.discover_token_counting_contract(
        _preflight(reasoning=False),
        prompt="hello",
        image_path=_calibration_image(tmp_path),
        response_format={"type": "json_schema"},
        thinking_contract=_thinking_contract(),
    )
    assert isinstance(contract, ApplyTemplateTokenizeContract)
    assert contract.apply_template_request_mode == "messages-only"
    assert contract.input_projection == "text-only"
    assert calls[0][0] == "/apply-template"
    assert calls[1][0] == "/tokenize"


def test_thinking_capable_rejects_extended_equivalence_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        if request.url.path == "/apply-template":
            return httpx.Response(200, content=json.dumps("same-prompt").encode())
        return httpx.Response(404)

    client = LlamaCppVisionClient(
        ProcessingConfig(
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            process=ProcessOptions(llama_base_url="http://test"),
        ),
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    contract = client.discover_token_counting_contract(
        _preflight(reasoning=True),
        prompt="hello",
        image_path=_calibration_image(tmp_path),
        response_format={"type": "json_schema"},
        thinking_contract=_thinking_contract(),
    )
    assert contract.mode == "estimate-only"


def test_transient_token_discovery_aborts(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(503)
        return httpx.Response(404)

    client = LlamaCppVisionClient(
        ProcessingConfig(
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            process=ProcessOptions(llama_base_url="http://test", retry_backoff_seconds=0),
        ),
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ProcessingError) as exc_info:
        client.discover_token_counting_contract(
            _preflight(reasoning=False),
            prompt="hello",
            image_path=_calibration_image(tmp_path),
            response_format={"type": "json_schema"},
            thinking_contract=_thinking_contract(),
        )
    assert exc_info.value.code == "token-counting-calibration-failed"


def test_frozen_apply_template_uses_persisted_mode(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/apply-template":
            body = json.loads(request.content)
            assert "chat_template_kwargs" in body
            return httpx.Response(200, content=json.dumps("prompt").encode())
        if request.url.path == "/tokenize":
            return httpx.Response(200, content=json.dumps({"tokens": [1]}).encode())
        return httpx.Response(404)

    config = ProcessingConfig(
        extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
        process=ProcessOptions(llama_base_url="http://test"),
    )
    client = LlamaCppVisionClient(
        config,
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    from tests.conftest import make_inference_environment

    env = make_inference_environment().model_copy(
        update={
            "token_counting_contract": ApplyTemplateTokenizeContract(
                mode="apply-template-tokenize",
                apply_template_request_mode="messages-plus-chat-template-kwargs",
                input_projection="text-only",
                image_token_policy="configured-reserve",
                model_alias="vision-model",
                llama_cpp_build="test-build",
                chat_template_sha256="c" * 64,
            )
        }
    )
    client.bind_environment(env)
    payload = client._build_chat_payload(
        prompt="hello",
        image_path=_calibration_image(tmp_path),
        response_format={"type": "json_schema"},
        thinking_contract=env.thinking_control_contract,
        include_image=True,
    )
    count = client._count_with_frozen_contract(payload, env.token_counting_contract)
    assert count == 1
    assert calls == ["/apply-template", "/tokenize"]
