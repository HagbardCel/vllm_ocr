"""Interpretation protocols."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypeVar

from pydantic import BaseModel

from bookextract.artifacts import InferenceAttempt
from bookextract.models import InterpretationProvenance, PageContext, PageInput

T = TypeVar("T", bound=BaseModel)


class VisionModelClient(Protocol):
    def generate_structured(
        self,
        *,
        image_path: Path,
        page_image_sha256: str,
        prompt: str,
        response_model: type[T],
    ) -> tuple[
        T,
        InterpretationProvenance,
        tuple[InferenceAttempt, ...],
        bytes,
        dict[str, object],
    ]: ...


class PageInterpreter(Protocol):
    def interpret(
        self,
        *,
        page_input: PageInput,
        context: PageContext,
    ) -> object: ...
