"""Thinking calibration and smoke availability tests."""

from __future__ import annotations

import json

import httpx
import pytest

from bookextract.config import ExtractionConfig, ProcessingConfig, ProcessOptions
from bookextract.errors import InferenceError, ProcessingError
from bookextract.inference.llamacpp import (
    LlamaCppVisionClient,
    PreflightResult,
    _ThinkingCandidateRejected,
)
from bookextract.models import (
    ServerInferenceIdentity,
    ServerInvocationCapabilities,
    ThinkingControlContract,
)


def _preflight() -> PreflightResult:
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
            media_marker=None,
            chat_template_caps={},
        ),
        raw_health=b"ok",
        raw_models=b"{}",
        raw_props=b"{}",
    )


def _ok_completion(*, reasoning: str | None = None) -> bytes:
    return json.dumps(
        {
            "id": "1",
            "object": "chat.completion",
            "created": 1,
            "model": "vision-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "content": '{"page_type":"blank","blocks":[]}',
                        "reasoning_content": reasoning,
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()


def test_transient_calibration_aborts_without_next_candidate() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503)

    client = LlamaCppVisionClient(
        ProcessingConfig(
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            process=ProcessOptions(llama_base_url="http://test", retry_backoff_seconds=0),
        ),
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(InferenceError, match="thinking-smoke-unavailable"):
        client.calibrate_thinking_control(_preflight())
    assert calls["count"] == 1


def test_contradictory_response_tries_next_candidate() -> None:
    responses = [
        _ok_completion(reasoning="hidden thought"),
        _ok_completion(),
        _ok_completion(),
        _ok_completion(),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=responses.pop(0))

    client = LlamaCppVisionClient(
        ProcessingConfig(
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            process=ProcessOptions(llama_base_url="http://test", retry_backoff_seconds=0),
        ),
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    contract = client.calibrate_thinking_control(_preflight())
    assert contract.reasoning_format == "deepseek"


def test_smoke_contradiction_becomes_drift() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_ok_completion(reasoning="still thinking"))

    client = LlamaCppVisionClient(
        ProcessingConfig(
            extraction=ExtractionConfig(model_alias="vision-model", prompt_version="v1"),
            process=ProcessOptions(llama_base_url="http://test", retry_backoff_seconds=0),
        ),
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    contract = ThinkingControlContract(
        reasoning_format="none",
        applied_template_probe_supported=False,
        model_alias="vision-model",
        llama_cpp_build="b1",
        chat_template_sha256="c" * 64,
    )
    with pytest.raises(ProcessingError, match="thinking-control-contract-drift"):
        client.run_thinking_smoke(contract)


def test_thinking_candidate_rejected_is_internal() -> None:
    assert issubclass(_ThinkingCandidateRejected, Exception)
