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
