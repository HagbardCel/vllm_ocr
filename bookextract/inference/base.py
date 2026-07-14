"""Vision model client protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from bookextract.artifacts import InferenceResult

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class VisionModelClient(Protocol):
    def generate_structured(
        self,
        *,
        image_path: Path,
        page_image_sha256: str,
        prompt: str,
        response_model: type[ResponseT],
        schema_ref: bytes,
    ) -> InferenceResult[ResponseT]:
        """Run structured multimodal completion with context-budget enforcement."""
