"""Inference retry behavior tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from bookextract.config import ExtractionConfig, ProcessingConfig, ProcessOptions
from bookextract.errors import InferenceError
from bookextract.inference.llamacpp import LlamaCppVisionClient
from bookextract.models import EstimateOnlyContract
from tests.conftest import make_inference_environment


class _SmokeModel(BaseModel):
    ok: bool = True


def _client_with_handler(handler: Any) -> LlamaCppVisionClient:
    config = ProcessingConfig(
        extraction=ExtractionConfig(
            model_alias="vision-model",
            prompt_version="v1",
            max_tokens=32,
        ),
        process=ProcessOptions(
            llama_base_url="http://test",
            max_attempts=3,
            retry_backoff_seconds=0,
        ),
    )
    env = make_inference_environment().model_copy(
        update={
            "token_counting_contract": EstimateOnlyContract(
                mode="estimate-only",
                model_alias="vision-model",
                llama_cpp_build="test-build",
                chat_template_sha256="c" * 64,
            )
        }
    )
    return LlamaCppVisionClient(
        config,
        environment=env,
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )


def test_ordinary_5xx_allows_one_retry(tmp_path) -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(500, content=b"err")
        return httpx.Response(
            200,
            content=json.dumps(
                {
                    "id": "1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "vision-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"content": '{"ok": true}'},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode(),
        )

    client = _client_with_handler(handler)
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    result = client.generate_structured(
        image_path=image,
        page_image_sha256="abc",
        prompt="hello",
        response_model=_SmokeModel,
        schema_ref=b"{}",
    )
    assert result.value.ok is True
    assert attempts["count"] == 2


def test_ordinary_5xx_exhausts_after_one_retry(tmp_path) -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        attempts["count"] += 1
        return httpx.Response(500, content=b"err")

    client = _client_with_handler(handler)
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    with pytest.raises(InferenceError) as exc_info:
        client.generate_structured(
            image_path=image,
            page_image_sha256="abc",
            prompt="hello",
            response_model=_SmokeModel,
            schema_ref=b"{}",
        )
    assert exc_info.value.code == "http-server-error"
    assert attempts["count"] == 2


def test_unexpected_reasoning_records_attempt_with_context(tmp_path) -> None:
    response_body = json.dumps(
        {
            "id": "1",
            "object": "chat.completion",
            "created": 1,
            "model": "vision-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": '{"ok": true}',
                        "reasoning_content": "hidden thought",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        return httpx.Response(200, content=response_body)

    client = _client_with_handler(handler)
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    schema_ref = b'{"type":"object"}'
    with pytest.raises(InferenceError) as exc_info:
        client.generate_structured(
            image_path=image,
            page_image_sha256="deadbeef",
            prompt="visible prompt",
            response_model=_SmokeModel,
            schema_ref=schema_ref,
        )
    err = exc_info.value
    assert err.code == "unexpected-reasoning-content"
    assert err.retryable is False
    assert err.attempts is not None
    assert len(err.attempts) == 1
    attempt = err.attempts[0]
    assert attempt.succeeded is False
    assert attempt.status_code == 200
    assert attempt.error_code == "unexpected-reasoning-content"
    assert attempt.response_body == response_body
    assert err.context.prompt == b"visible prompt"
    assert err.context.page_image_sha256 == "deadbeef"
    assert err.context.schema_ref == schema_ref


def _assert_envelope_failure(
    exc_info: pytest.ExceptionInfo[InferenceError],
    *,
    response_body: bytes,
    content_type: str,
    error_code: str,
    prompt: bytes,
    schema_ref: bytes,
    image_hash: str,
    request_count: int,
) -> None:
    err = exc_info.value
    assert request_count == 1
    assert err.code == error_code
    assert err.retryable is False
    assert err.attempts_exhausted is False
    assert err.attempts is not None
    assert len(err.attempts) == 1
    attempt = err.attempts[0]
    assert attempt.status_code == 200
    assert attempt.response_body == response_body
    assert attempt.content_type == content_type
    assert attempt.error_code == error_code
    assert err.context.prompt == prompt
    assert err.context.page_image_sha256 == image_hash
    assert err.context.schema_ref == schema_ref


def test_invalid_http_response_records_envelope_attempt(tmp_path) -> None:
    response_body = b"not-json-at-all"
    content_type = "application/json; charset=utf-8"
    requests = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        requests["count"] += 1
        return httpx.Response(
            200,
            content=response_body,
            headers={"Content-Type": content_type},
        )

    client = _client_with_handler(handler)
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    schema_ref = b'{"type":"object"}'
    with pytest.raises(InferenceError) as exc_info:
        client.generate_structured(
            image_path=image,
            page_image_sha256="deadbeef",
            prompt="visible prompt",
            response_model=_SmokeModel,
            schema_ref=schema_ref,
        )
    _assert_envelope_failure(
        exc_info,
        response_body=response_body,
        content_type=content_type,
        error_code="invalid-http-response",
        prompt=b"visible prompt",
        schema_ref=schema_ref,
        image_hash="deadbeef",
        request_count=requests["count"],
    )


def test_invalid_completion_envelope_records_envelope_attempt(tmp_path) -> None:
    response_body = json.dumps({"object": "chat.completion"}).encode()
    content_type = "application/json"
    requests = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path != "/v1/chat/completions":
            return httpx.Response(404)
        requests["count"] += 1
        return httpx.Response(
            200,
            content=response_body,
            headers={"Content-Type": content_type},
        )

    client = _client_with_handler(handler)
    image = tmp_path / "page.png"
    image.write_bytes(b"png")
    schema_ref = b'{"type":"object"}'
    with pytest.raises(InferenceError) as exc_info:
        client.generate_structured(
            image_path=image,
            page_image_sha256="image-hash",
            prompt="visible prompt",
            response_model=_SmokeModel,
            schema_ref=schema_ref,
        )
    _assert_envelope_failure(
        exc_info,
        response_body=response_body,
        content_type=content_type,
        error_code="invalid-completion-envelope",
        prompt=b"visible prompt",
        schema_ref=schema_ref,
        image_hash="image-hash",
        request_count=requests["count"],
    )
