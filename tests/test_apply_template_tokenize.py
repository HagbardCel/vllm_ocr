"""Apply-template tokenize discovery and frozen-contract tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import pytest

from bookextract.config import ExtractionConfig, ProcessingConfig, ProcessOptions
from bookextract.errors import InferenceError, ProcessingError
from bookextract.inference.llamacpp import LlamaCppVisionClient, PreflightResult
from bookextract.models import (
    ApplyTemplateTokenizeContract,
    ServerInferenceIdentity,
    ServerInvocationCapabilities,
    TextOnlyInputTokensContract,
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
        contract_format_version=1,
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


def _client(handler: httpx.MockTransport) -> LlamaCppVisionClient:
    return LlamaCppVisionClient(
        ProcessingConfig(
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            process=ProcessOptions(llama_base_url="http://test", retry_backoff_seconds=0),
        ),
        client=httpx.Client(base_url="http://test", transport=handler),
    )


def test_discover_apply_template_messages_only(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/apply-template":
            assert "model" not in body
            assert all(
                part.get("type") != "image_url"
                for message in body["messages"]
                if isinstance(message.get("content"), list)
                for part in message["content"]
            )
            assert body == {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "hello"}],
                    }
                ]
            }
            return httpx.Response(
                200, content=json.dumps({"prompt": "prompt-text"}).encode()
            )
        if request.url.path == "/tokenize":
            assert body == {
                "content": "prompt-text",
                "add_special": False,
                "parse_special": True,
            }
            return httpx.Response(200, content=json.dumps({"tokens": [1, 2, 3]}).encode())
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
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


_BOOK_TEXT_WITH_LITERALS = (
    "Data: A History discusses the literal API field name image_url."
)


def test_apply_template_allows_book_text_with_data_and_image_url_literals(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    apply_template_body: dict[str, object] | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal apply_template_body
        calls.append(request.url.path)
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        body = json.loads(request.content)
        if request.url.path == "/apply-template":
            apply_template_body = body
            return httpx.Response(
                200, content=json.dumps({"prompt": "prompt-text"}).encode()
            )
        if request.url.path == "/tokenize":
            return httpx.Response(200, content=json.dumps({"tokens": [1, 2, 3]}).encode())
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    contract = client.discover_token_counting_contract(
        _preflight(reasoning=False),
        prompt=_BOOK_TEXT_WITH_LITERALS,
        image_path=_calibration_image(tmp_path),
        response_format={"type": "json_schema"},
        thinking_contract=_thinking_contract(),
    )
    assert isinstance(contract, ApplyTemplateTokenizeContract)
    assert any(path.endswith("input_tokens") for path in calls)
    assert "/apply-template" in calls
    assert "/tokenize" in calls
    assert apply_template_body == {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": _BOOK_TEXT_WITH_LITERALS}],
            }
        ]
    }


def test_apply_template_rejects_image_url_content_part() -> None:
    apply_template_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal apply_template_calls
        if request.url.path == "/apply-template":
            apply_template_calls += 1
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ProcessingError) as exc_info:
        client._request_apply_template(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hello"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AA=="},
                            },
                        ],
                    }
                ]
            },
            discovery=True,
            extended=False,
        )
    assert exc_info.value.code == "token-counting-calibration-failed"
    assert apply_template_calls == 0


def test_discover_apply_template_extended_mode(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        body = json.loads(request.content)
        if request.url.path == "/apply-template":
            calls.append(body)
            enable_thinking = body.get("chat_template_kwargs", {}).get("enable_thinking")
            prompt = "thinking-on" if enable_thinking else "thinking-off"
            return httpx.Response(200, content=json.dumps({"prompt": prompt}).encode())
        if request.url.path == "/tokenize":
            return httpx.Response(200, content=json.dumps({"tokens": [1, 2]}).encode())
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    contract = client.discover_token_counting_contract(
        _preflight(reasoning=True),
        prompt="hello",
        image_path=_calibration_image(tmp_path),
        response_format={"type": "json_schema"},
        thinking_contract=_thinking_contract(),
    )
    assert isinstance(contract, ApplyTemplateTokenizeContract)
    assert contract.apply_template_request_mode == "messages-plus-chat-template-kwargs"
    assert any(
        call.get("chat_template_kwargs") == {"enable_thinking": False} for call in calls
    )


def test_projection_preserves_payload_and_order(tmp_path: Path) -> None:
    client = _client(httpx.MockTransport(lambda r: httpx.Response(404)))
    payload = {
        "messages": [
            {
                "role": "user",
                "name": "reader",
                "content": [
                    {"type": "text", "text": "before"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    {"type": "text", "text": "after"},
                ],
            }
        ]
    }
    original = copy.deepcopy(payload)
    projected = client._project_text_only_messages(payload)
    assert payload == original
    assert projected == [
        {
            "role": "user",
            "name": "reader",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "text", "text": "after"},
            ],
        }
    ]


@pytest.mark.parametrize("status_code", [401, 403, 422])
def test_non_404_input_tokens_aborts_discovery(tmp_path: Path, status_code: int) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(status_code)
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ProcessingError) as exc_info:
        client.discover_token_counting_contract(
            _preflight(reasoning=False),
            prompt="hello",
            image_path=_calibration_image(tmp_path),
            response_format={"type": "json_schema"},
            thinking_contract=_thinking_contract(),
        )
    assert exc_info.value.code == "token-counting-calibration-failed"
    assert "/apply-template" not in calls


def test_malformed_input_tokens_200_aborts_discovery(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(200, content=b"not-json")
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ProcessingError) as exc_info:
        client.discover_token_counting_contract(
            _preflight(reasoning=False),
            prompt="hello",
            image_path=_calibration_image(tmp_path),
            response_format={"type": "json_schema"},
            thinking_contract=_thinking_contract(),
        )
    assert exc_info.value.code == "token-counting-calibration-failed"
    assert "/apply-template" not in calls


def test_apply_template_rejects_alias_and_empty_prompt(tmp_path: Path) -> None:
    for response_body in (
        json.dumps("prompt-text"),
        json.dumps({"text": "prompt-text"}),
        json.dumps({"prompt": ""}),
    ):

        def handler(request: httpx.Request, body: bytes = response_body.encode()) -> httpx.Response:
            if request.url.path.endswith("input_tokens"):
                return httpx.Response(404)
            if request.url.path == "/apply-template":
                return httpx.Response(200, content=body)
            return httpx.Response(404)

        client = _client(httpx.MockTransport(handler))
        with pytest.raises(ProcessingError) as exc_info:
            client.discover_token_counting_contract(
                _preflight(reasoning=False),
                prompt="hello",
                image_path=_calibration_image(tmp_path),
                response_format={"type": "json_schema"},
                thinking_contract=_thinking_contract(),
            )
        assert exc_info.value.code == "token-counting-calibration-failed"


def test_thinking_capable_rejects_extended_equivalence_failure(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(404)
        if request.url.path == "/apply-template":
            return httpx.Response(200, content=json.dumps({"prompt": "same-prompt"}).encode())
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
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

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ProcessingError) as exc_info:
        client.discover_token_counting_contract(
            _preflight(reasoning=False),
            prompt="hello",
            image_path=_calibration_image(tmp_path),
            response_format={"type": "json_schema"},
            thinking_contract=_thinking_contract(),
        )
    assert exc_info.value.code == "token-counting-calibration-failed"


def test_ordinary_500_discovery_aborts_without_apply_template(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path.endswith("input_tokens"):
            return httpx.Response(500)
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ProcessingError) as exc_info:
        client.discover_token_counting_contract(
            _preflight(reasoning=False),
            prompt="hello",
            image_path=_calibration_image(tmp_path),
            response_format={"type": "json_schema"},
            thinking_contract=_thinking_contract(),
        )
    assert exc_info.value.code == "token-counting-calibration-failed"
    assert calls == [
        "/v1/chat/completions/input_tokens",
        "/v1/chat/completions/input_tokens",
    ]
    assert "/apply-template" not in calls


def test_frozen_input_tokens_500_raises_probe_unavailable(tmp_path: Path) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("input_tokens"):
            calls["count"] += 1
            return httpx.Response(500)
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    from tests.conftest import make_inference_environment

    env = make_inference_environment().model_copy(
        update={
            "token_counting_contract": TextOnlyInputTokensContract(
                contract_format_version=1,
                mode="chat-input-tokens-text-only",
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
    with pytest.raises(InferenceError) as exc_info:
        client._count_with_frozen_contract(payload, env.token_counting_contract)
    assert exc_info.value.code == "context-budget-probe-unavailable"
    assert calls["count"] == 2


def test_frozen_apply_template_uses_persisted_mode(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((request.url.path, body))
        if request.url.path == "/apply-template":
            assert body == {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                ],
                "chat_template_kwargs": {"enable_thinking": False},
            }
            return httpx.Response(200, content=json.dumps({"prompt": "prompt"}).encode())
        if request.url.path == "/tokenize":
            return httpx.Response(200, content=json.dumps({"tokens": [1]}).encode())
        return httpx.Response(404)

    client = _client(httpx.MockTransport(handler))
    from tests.conftest import make_inference_environment

    env = make_inference_environment().model_copy(
        update={
            "token_counting_contract": ApplyTemplateTokenizeContract(
                contract_format_version=1,
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
    assert [path for path, _ in calls] == ["/apply-template", "/tokenize"]
