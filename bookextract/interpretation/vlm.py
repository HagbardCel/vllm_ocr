"""VLM-backed page interpreter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from bookextract.artifacts import InferenceAttempt, InterpretationResult, PendingArtifact
from bookextract.canonical import sha256_hex
from bookextract.conversion import convert_vlm_response
from bookextract.interpretation.base import VisionModelClient
from bookextract.interpretation.prompts import PagePromptBuilder
from bookextract.models import (
    InterpretationProvenance,
    PageContext,
    PageInput,
)
from bookextract.schema import load_wire_schema
from bookextract.wire import VlmPageResponse

T = TypeVar("T", bound=BaseModel)


class LlamaCppStructuredClient:
    """Adapter exposing the interpretation-layer VisionModelClient protocol."""

    def __init__(self, client: object) -> None:
        self._client = client

    def generate_structured(
        self,
        *,
        image_path: Path,
        page_image_sha256: str,
        prompt: str,
        response_model: type[T],
    ) -> tuple[T, InterpretationProvenance, tuple[InferenceAttempt, ...], bytes, dict[str, object]]:
        schema_ref = json.dumps(load_wire_schema(), sort_keys=True).encode("utf-8")
        result = self._client.generate_structured(  # type: ignore[attr-defined]
            image_path=image_path,
            page_image_sha256=page_image_sha256,
            prompt=prompt,
            response_model=response_model,
            schema_ref=schema_ref,
        )
        elapsed_ms = sum(
            attempt.elapsed_ms or 0.0
            for attempt in result.attempts
            if attempt.elapsed_ms is not None
        )
        provenance = InterpretationProvenance(
            backend="llama.cpp",
            model=self._client._config.extraction.model_alias,  # type: ignore[attr-defined]
            prompt_version=self._client._config.extraction.prompt_version,  # type: ignore[attr-defined]
            attempts=len(result.attempts),
            elapsed_ms=elapsed_ms or None,
            raw_response_sha256=sha256_hex(result.final_raw_body),
        )
        return (
            result.value,
            provenance,
            result.attempts,
            result.final_raw_body,
            result.request_summary,
        )


class VlmPageInterpreter:
    def __init__(
        self,
        client: VisionModelClient,
        prompt_builder: PagePromptBuilder | None = None,
        *,
        backend: str = "llama.cpp",
        model: str,
        prompt_version: str | None = None,
    ) -> None:
        self._client = client
        self._prompt_builder = prompt_builder or PagePromptBuilder()
        self._backend = backend
        self._model = model
        self._prompt_version = prompt_version or PagePromptBuilder.PROMPT_VERSION

    def interpret(
        self,
        *,
        page_input: PageInput,
        context: PageContext,
    ) -> InterpretationResult:
        prompt = self._prompt_builder.build(context)
        wire_response, provenance, attempts, raw_body, request_summary = (
            self._client.generate_structured(
                image_path=page_input.image_path,
                page_image_sha256=page_input.rendered.image_sha256,
                prompt=prompt,
                response_model=VlmPageResponse,
            )
        )
        interpretation = convert_vlm_response(wire_response)
        failed_attempts = tuple(a for a in attempts if not a.succeeded)
        artifacts: list[PendingArtifact] = [
            PendingArtifact(
                logical_name="prompt",
                filename="prompt.txt",
                media_type="text/plain",
                content=prompt.encode("utf-8"),
            ),
            PendingArtifact(
                logical_name="request-summary",
                filename="request-summary.json",
                media_type="application/json",
                content=json.dumps(request_summary, indent=2, ensure_ascii=False).encode(
                    "utf-8"
                ),
            ),
            PendingArtifact(
                logical_name="response",
                filename="response.json",
                media_type="application/json",
                content=raw_body,
            ),
        ]
        return InterpretationResult(
            interpretation=interpretation,
            provenance=provenance.model_copy(
                update={
                    "backend": self._backend,
                    "model": self._model,
                    "prompt_version": self._prompt_version,
                }
            ),
            artifacts=tuple(artifacts),
            failed_attempts=failed_attempts,
        )
